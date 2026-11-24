#include "spinor_hr_gpu.h"
#include "cuda_mem_wrapper.h"
#include "source_base/parallel_common.h"
#include "source_base/parallel_device.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_hamilt/module_hcontainer/hcontainer.h"
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <vector>

namespace ModuleGint
{
namespace
{
int checked_size(size_t size)
{
    if (size > INT_MAX) throw std::overflow_error("spinor H(R) MPI index exceeds INT_MAX");
    return static_cast<int>(size);
}

template <typename T>
std::unique_ptr<CudaMemWrapper<T>> upload(const std::vector<T>& values)
{
    std::unique_ptr<CudaMemWrapper<T>> result(new CudaMemWrapper<T>(std::max<size_t>(1, values.size()), 0, false));
    if (!values.empty()) CHECK_CUDA(cudaMemcpy(result->get_device_ptr(), values.data(),
                                             values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return result;
}

__global__ void pack_spinor(int n, const int2* map, const double* v0, const double* vx,
                            const double* vy, const double* vz, bool transverse, double2* output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int offset = map[i].x;
    const int code = map[i].y;
    double2 value = make_double2(0.0, 0.0);
    // Preserve the existing Gint channel/sign convention, including the
    // down/down channel and conjugation of reverse atom pairs.
    switch (code & 3)
    {
    case 0: value.x = v0[offset] + vz[offset]; break;
    case 1: if (transverse) value = make_double2(vx[offset], -vy[offset]); break;
    case 2: if (transverse) value = make_double2(vx[offset], vy[offset]); break;
    case 3: value.x = v0[offset] - vz[offset]; break;
    }
    if (code & 4) value.y = -value.y;
    output[i] = value;
}

__global__ void sum_owned(int n, const int* starts, const int* indices,
                          const double2* values, double2* output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double2 sum = make_double2(0.0, 0.0);
    // Source-rank order, no atomics or non-deterministic reduction.
    for (int j = starts[i]; j < starts[i + 1]; ++j)
    {
        const double2 value = values[indices[j]];
        sum.x += value.x;
        sum.y += value.y;
    }
    output[i] = sum;
}
}

struct SpinorHrGpu::Impl
{
    const Parallel_Orbitals* orbitals;
    std::vector<int> topology;
    std::vector<int> rows;
    std::vector<int> cols;
    int nnr;
    struct Tile
    {
        int output_begin;
        int output_size;
        int nsend;
        int nrecv;
        std::vector<int> sendcounts;
        std::vector<int> recvcounts;
        std::vector<int> senddispls;
        std::vector<int> recvdispls;
        std::unique_ptr<CudaMemWrapper<int2>> map;
        std::unique_ptr<CudaMemWrapper<int>> starts;
        std::unique_ptr<CudaMemWrapper<int>> indices;
    };
    std::vector<Tile> tiles;
    std::unique_ptr<CudaMemWrapper<double2>> send;
    std::unique_ptr<CudaMemWrapper<double2>> recv;
    std::unique_ptr<CudaMemWrapper<double2>> owned;
};

SpinorHrGpu::SpinorHrGpu(const hamilt::HContainer<double>& source,
                       const hamilt::HContainer<std::complex<double>>& destination)
    : impl_(new Impl)
{
    auto& p = *impl_;
    p.orbitals = destination.get_paraV();
    p.topology = destination.get_ijr_info();
    p.rows = p.orbitals->get_indexes_row();
    p.cols = p.orbitals->get_indexes_col();
    p.nnr = checked_size(destination.get_nnr());
    // Block requests carry atom-relative spinor row/column indices, not
    // individual matrix elements. This also supports non-square process grids.
    std::vector<int> local;
    for (int i = 0; i < destination.size_atom_pairs(); ++i)
    {
        const auto& ap = destination.get_atom_pair(i);
        const auto rows = p.orbitals->get_indexes_row(ap.get_atom_i());
        const auto cols = p.orbitals->get_indexes_col(ap.get_atom_j());
        for (int r = 0; r < ap.get_R_size(); ++r)
        {
            const auto R = ap.get_R_index(r);
            const auto* matrix = ap.find_matrix(R);
            local.insert(local.end(), {ap.get_atom_i(), ap.get_atom_j(), R.x, R.y, R.z,
                         checked_size(matrix->get_pointer() - destination.get_wrapper()),
                         checked_size(rows.size()), checked_size(cols.size())});
            local.insert(local.end(), rows.begin(), rows.end());
            local.insert(local.end(), cols.begin(), cols.end());
        }
    }
    int size = 1;
#ifdef __MPI
    size = Parallel_Common::communicator_size(MPI_COMM_WORLD);
#endif
    const int local_size = checked_size(local.size());
    std::vector<int> counts(size, local_size);
#ifdef __MPI
    Parallel_Common::allgather_int(&local_size, 1, counts.data(), MPI_COMM_WORLD);
#endif
    std::vector<int> displs(size + 1, 0);
    for (int i = 0; i < size; ++i) displs[i + 1] = checked_size(size_t(displs[i]) + counts[i]);
    std::vector<int> metadata(displs.back());
#ifdef __MPI
    Parallel_Common::allgatherv_int(local.data(), local_size, metadata.data(), counts.data(),
                                   displs.data(), MPI_COMM_WORLD);
#else
    metadata = local;
#endif
    // Each source can contribute at most once per destination element. Split
    // by destination so all contributions to an element keep their rank order.
    constexpr int buffer_values = 64 * 1024 * 1024 / sizeof(double2);
    const int tile_size = std::max(1, buffer_values / size);
    int max_owned_size = p.nnr;
    Parallel_Reduce::reduce_max(max_owned_size);
    int max_send = 1;
    int max_recv = 1;
    for (int begin = 0; begin < max_owned_size;)
    {
        const int end = begin + std::min(tile_size, max_owned_size - begin);
        p.tiles.emplace_back();
        auto& tile = p.tiles.back();
        tile.output_begin = begin;
        const int local_end = std::min(end, p.nnr);
        tile.output_size = std::max(0, local_end - begin);
        tile.sendcounts.resize(size);
        tile.senddispls.resize(size);
        std::vector<int2> mapping;
        std::vector<int> targets;
        for (int rank = 0; rank < size; ++rank)
        {
            tile.senddispls[rank] = checked_size(mapping.size());
            for (int cursor = displs[rank]; cursor < displs[rank + 1];)
            {
                const int* block = metadata.data() + cursor;
                const int a = block[0];
                const int b = block[1];
                const int block_begin = block[5];
                const int nr = block[6];
                const int nc = block[7];
                cursor += 8 + nr + nc;
                const int first = std::max(begin, block_begin);
                const int last = std::min(end, block_begin + nr * nc);
                if (first >= last)
                {
                    continue;
                }
                const bool reverse = a > b;
                const ModuleBase::Vector3<int> R(block[2], block[3], block[4]);
                const int atom_i = std::min(a, b);
                const int atom_j = std::max(a, b);
                const auto translation = reverse ? -R : R;
                const auto* pair = source.find_pair(atom_i, atom_j);
                const auto* matrix = pair ? pair->find_matrix(translation) : nullptr;
                if (matrix == nullptr)
                {
                    continue;
                }
                const size_t base_offset = matrix->get_pointer() - source.get_wrapper();
                const int base = checked_size(base_offset);
                const int source_cols = matrix->get_col_size();
                for (int index = first; index < last; ++index)
                {
                    const int element = index - block_begin;
                    const int row = element / nc;
                    const int col = element % nc;
                    const int ri = block[8 + row];
                    const int ci = block[8 + nr + col];
                    const int offset = reverse ? (ci / 2) * source_cols + ri / 2
                                               : (ri / 2) * source_cols + ci / 2;
                    const int source_index = base + offset;
                    const int code = 2 * (ri % 2) + ci % 2 + (reverse ? 4 : 0);
                    const int2 entry = make_int2(source_index, code);
                    mapping.push_back(entry);
                    const int local_index = index - begin;
                    targets.push_back(local_index);
                }
            }
            tile.sendcounts[rank] = checked_size(mapping.size()) - tile.senddispls[rank];
        }
        tile.nsend = checked_size(mapping.size());
        tile.recvcounts.resize(size);
        tile.recvdispls.resize(size);
#ifdef __MPI
        Parallel_Common::alltoall_int(tile.sendcounts.data(), tile.recvcounts.data(), MPI_COMM_WORLD);
#else
        tile.recvcounts = tile.sendcounts;
#endif
        tile.nrecv = 0;
        for (int rank = 0; rank < size; ++rank)
        {
            tile.recvdispls[rank] = tile.nrecv;
            const size_t recv_end = static_cast<size_t>(tile.nrecv) + tile.recvcounts[rank];
            tile.nrecv = checked_size(recv_end);
        }
        const int local_recv = std::max(1, tile.nrecv);
        std::vector<int> received(local_recv);
#ifdef __MPI
        const int zero_target = 0;
        const int* send_targets = targets.empty() ? &zero_target : targets.data();
        Parallel_Common::alltoallv_int(send_targets, tile.sendcounts.data(), tile.senddispls.data(),
                                      received.data(), tile.recvcounts.data(), tile.recvdispls.data(), MPI_COMM_WORLD);
#else
        received = targets;
#endif
        std::vector<int> starts(tile.output_size + 1, 0);
        for (int i = 0; i < tile.nrecv; ++i)
        {
            const int index = received[i];
            if (index < 0 || index >= tile.output_size)
            {
                throw std::runtime_error("invalid spinor H(R) destination index");
            }
            ++starts[index + 1];
        }
        for (int i = 0; i < tile.output_size; ++i)
        {
            starts[i + 1] += starts[i];
        }
        std::vector<int> next = starts;
        std::vector<int> indices(tile.nrecv);
        for (int i = 0; i < tile.nrecv; ++i)
        {
            indices[next[received[i]]++] = i;
        }
        tile.map = upload(mapping);
        tile.starts = upload(starts);
        tile.indices = upload(indices);
        max_send = std::max(max_send, tile.nsend);
        max_recv = std::max(max_recv, tile.nrecv);
        begin = end;
    }
    p.send.reset(new CudaMemWrapper<double2>(max_send, 0, true));
    p.recv.reset(new CudaMemWrapper<double2>(max_recv, 0, true));
    p.owned.reset(new CudaMemWrapper<double2>(std::max(1, p.nnr), 0, false));
}

SpinorHrGpu::~SpinorHrGpu() = default;

bool SpinorHrGpu::matches(const hamilt::HContainer<std::complex<double>>& destination) const
{
    return impl_->orbitals == destination.get_paraV() && impl_->nnr == destination.get_nnr()
        && impl_->topology == destination.get_ijr_info()
        && impl_->rows == destination.get_paraV()->get_indexes_row()
        && impl_->cols == destination.get_paraV()->get_indexes_col();
}

const std::complex<double>* SpinorHrGpu::transfer_device(const double* v0,
                                                        const double* vx,
                                                        const double* vy,
                                                        const double* vz,
                                                        bool transverse)
{
    auto& p = *impl_;
    ModuleBase::timer::start("Gint", "spinor_owner_values");
    for (const auto& tile : p.tiles)
    {
        if (tile.nsend)
        {
            const int blocks = (tile.nsend + 255) / 256;
            pack_spinor<<<blocks, 256>>>(tile.nsend, tile.map->get_device_ptr(), v0, vx, vy, vz,
                                         transverse, p.send->get_device_ptr());
            CHECK_CUDA(cudaGetLastError());
            p.send->copy_device_to_host_sync(tile.nsend);
        }
#ifdef __MPI
        // Preserve host MPI transport for builds without CUDA-aware MPI.
        const auto* send = reinterpret_cast<const std::complex<double>*>(p.send->get_host_ptr());
        auto* recv = reinterpret_cast<std::complex<double>*>(p.recv->get_host_ptr());
        Parallel_Common::alltoallv_data(send, tile.sendcounts.data(), tile.senddispls.data(),
                                        recv, tile.recvcounts.data(), tile.recvdispls.data(), MPI_COMM_WORLD);
#else
        std::copy_n(p.send->get_host_ptr(), tile.nrecv, p.recv->get_host_ptr());
#endif
        if (tile.nrecv)
        {
            p.recv->copy_host_to_device_sync(tile.nrecv);
        }
        if (tile.output_size)
        {
            const int blocks = (tile.output_size + 255) / 256;
            double2* output = p.owned->get_device_ptr() + tile.output_begin;
            sum_owned<<<blocks, 256>>>(tile.output_size, tile.starts->get_device_ptr(),
                                        tile.indices->get_device_ptr(), p.recv->get_device_ptr(), output);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    ModuleBase::timer::end("Gint", "spinor_owner_values");
    return reinterpret_cast<const std::complex<double>*>(p.owned->get_device_ptr());
}

void SpinorHrGpu::transfer(const double* v0, const double* vx, const double* vy, const double* vz,
                          bool transverse, hamilt::HContainer<std::complex<double>>& destination)
{
    const std::complex<double>* device_values = transfer_device(v0, vx, vy, vz, transverse);
    const size_t count = impl_->nnr;
    if (count)
    {
        // Current downstream H(R) consumers are host-side. This is the single
        // rank-owned materialization boundary, not four grid-local Pauli copies.
        CHECK_CUDA(cudaMemcpy(destination.get_wrapper(), device_values,
                              count * sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
    }
}
}
