#include "spinor_hr_gpu.h"
#include "cuda_mem_wrapper.h"
#include "source_base/parallel_common.h"
#include "source_base/parallel_device.h"
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
    int nsend;
    int nrecv;
    std::vector<int> sendcounts;
    std::vector<int> recvcounts;
    std::vector<int> senddispls;
    std::vector<int> recvdispls;
    std::unique_ptr<CudaMemWrapper<int2>> map;
    std::unique_ptr<CudaMemWrapper<int>> starts;
    std::unique_ptr<CudaMemWrapper<int>> indices;
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
    p.sendcounts.resize(size);
    p.senddispls.resize(size);
    std::vector<int2> mapping;
    std::vector<int> targets;
    for (int rank = 0; rank < size; ++rank)
    {
        p.senddispls[rank] = checked_size(mapping.size());
        for (int cursor = displs[rank]; cursor < displs[rank + 1];)
        {
            const int* block = metadata.data() + cursor;
            const int a = block[0];
            const int b = block[1];
            const bool reverse = a > b;
            const ModuleBase::Vector3<int> R(block[2], block[3], block[4]);
            const int nr = block[6];
            const int nc = block[7];
            const auto* ap = source.find_pair(std::min(a, b), std::max(a, b));
            const auto* matrix = ap ? ap->find_matrix(reverse ? -R : R) : nullptr;
            if (matrix)
            {
                const int base = checked_size(matrix->get_pointer() - source.get_wrapper());
                for (int row = 0; row < nr; ++row)
                {
                    for (int col = 0; col < nc; ++col)
                    {
                        const int ri = block[8 + row];
                        const int ci = block[8 + nr + col];
                        const int offset = reverse ? (ci / 2) * matrix->get_col_size() + ri / 2
                                                   : (ri / 2) * matrix->get_col_size() + ci / 2;
                        mapping.push_back(make_int2(base + offset, 2 * (ri % 2) + ci % 2 + (reverse ? 4 : 0)));
                        targets.push_back(block[5] + row * nc + col);
                    }
                }
            }
            cursor += 8 + nr + nc;
        }
        p.sendcounts[rank] = checked_size(mapping.size()) - p.senddispls[rank];
    }
    p.nsend = checked_size(mapping.size());
    p.recvcounts.resize(size);
    p.recvdispls.resize(size);
#ifdef __MPI
    Parallel_Common::alltoall_int(p.sendcounts.data(), p.recvcounts.data(), MPI_COMM_WORLD);
#else
    p.recvcounts = p.sendcounts;
#endif
    p.nrecv = 0;
    for (int i = 0; i < size; ++i)
    {
        p.recvdispls[i] = p.nrecv;
        p.nrecv = checked_size(size_t(p.nrecv) + p.recvcounts[i]);
    }
    std::vector<int> received(p.nrecv);
#ifdef __MPI
    Parallel_Common::alltoallv_int(targets.data(), p.sendcounts.data(), p.senddispls.data(),
                                  received.data(), p.recvcounts.data(), p.recvdispls.data(), MPI_COMM_WORLD);
#else
    received = targets;
#endif
    std::vector<int> starts(p.nnr + 1, 0);
    for (int index : received)
    {
        if (index < 0 || index >= p.nnr) throw std::runtime_error("invalid spinor H(R) destination index");
        ++starts[index + 1];
    }
    for (int i = 0; i < p.nnr; ++i) starts[i + 1] += starts[i];
    std::vector<int> next = starts;
    std::vector<int> indices(p.nrecv);
    for (int i = 0; i < p.nrecv; ++i) indices[next[received[i]]++] = i;
    p.map = upload(mapping);
    p.starts = upload(starts);
    p.indices = upload(indices);
    p.send.reset(new CudaMemWrapper<double2>(std::max(1, p.nsend), 0, true));
    p.recv.reset(new CudaMemWrapper<double2>(std::max(1, p.nrecv), 0, true));
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
    if (p.nsend)
    {
        pack_spinor<<<(p.nsend + 255) / 256, 256>>>(p.nsend, p.map->get_device_ptr(), v0, vx, vy, vz,
                                                  transverse, p.send->get_device_ptr());
        CHECK_CUDA(cudaGetLastError());
        p.send->copy_device_to_host_sync(p.nsend);
    }
#ifdef __MPI
    // Deliberately host MPI, even in CUDA-aware builds: retain a stable
    // transport while producer packing and destination reduction stay on GPU.
    Parallel_Common::alltoallv_data(reinterpret_cast<std::complex<double>*>(p.send->get_host_ptr()),
                                    p.sendcounts.data(), p.senddispls.data(),
                                    reinterpret_cast<std::complex<double>*>(p.recv->get_host_ptr()),
                                    p.recvcounts.data(), p.recvdispls.data(), MPI_COMM_WORLD);
#else
    std::copy_n(p.send->get_host_ptr(), p.nrecv, p.recv->get_host_ptr());
#endif
    if (p.nrecv) p.recv->copy_host_to_device_sync(p.nrecv);
    if (p.nnr)
    {
        sum_owned<<<(p.nnr + 255) / 256, 256>>>(p.nnr, p.starts->get_device_ptr(), p.indices->get_device_ptr(),
                                              p.recv->get_device_ptr(), p.owned->get_device_ptr());
        CHECK_CUDA(cudaGetLastError());
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
