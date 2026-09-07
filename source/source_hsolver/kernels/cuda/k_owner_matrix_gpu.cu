#include "k_owner_matrix_gpu.h"
#include "source_base/module_device/device_check.h"
#include "source_base/parallel_2d.h"
#include "source_base/parallel_common.h"
#include "source_base/parallel_device.h"
#include "source_base/timer.h"
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <vector>

namespace hsolver
{
namespace
{
int mpi_size(size_t n)
{
    if (n > INT_MAX) throw std::overflow_error("K-owner matrix exceeds MPI integer indexing");
    return static_cast<int>(n);
}

template <typename T>
struct Buffer
{
    T* device = nullptr;
    T* host = nullptr;
    Buffer(size_t n, bool pinned)
    {
        const size_t bytes = std::max<size_t>(1, n) * sizeof(T);
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device), bytes));
        if (pinned) CHECK_CUDA(cudaMallocHost(reinterpret_cast<void**>(&host), bytes));
    }
    ~Buffer()
    {
        if (host) CHECK_CUDA(cudaFreeHost(host));
        CHECK_CUDA(cudaFree(device));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

__global__ void pack_upper(int n, const int* indices, const double2* h,
                           const double2* s, double2* packed)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        packed[2 * i] = h[indices[i]];
        packed[2 * i + 1] = s[indices[i]];
    }
}

__global__ void unpack_upper(int n, const int* indices, const double2* packed,
                             double2* h, double2* s)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        h[indices[i]] = packed[2 * i];
        s[indices[i]] = packed[2 * i + 1];
    }
}
}

struct KOwnerMatrixGpu::Impl
{
    int rank;
    int local_size;
    int local_upper;
    int global_upper;
    std::vector<int> counts;
    std::vector<int> offsets;
    std::vector<int> sendcounts;
    std::vector<int> sendoffsets;
    std::vector<int> recvcounts;
    std::vector<int> recvoffsets;
    std::unique_ptr<Buffer<int>> pack_indices;
    std::unique_ptr<Buffer<int>> unpack_indices;
    std::unique_ptr<Buffer<double2>> local_h;
    std::unique_ptr<Buffer<double2>> local_s;
    std::unique_ptr<Buffer<double2>> send;
    std::unique_ptr<Buffer<double2>> recv;
    std::unique_ptr<Buffer<double2>> owner_h;
    std::unique_ptr<Buffer<double2>> owner_s;
};

KOwnerMatrixGpu::KOwnerMatrixGpu(const Parallel_2D& layout, int world_rank) : impl_(new Impl)
{
    ModuleBase::timer::start("HSolverLCAO", "owner_gpu_layout");
    auto& p = *impl_;
    p.rank = world_rank;
    p.local_size = mpi_size(layout.get_local_size());
    const int n = layout.get_global_row_size();
    const int dense = mpi_size(static_cast<size_t>(n) * n);
    const int ranks = Parallel_Common::communicator_size(MPI_COMM_WORLD);
    const int dims[2] = {layout.get_row_size(), layout.get_col_size()};
    std::vector<int> all_dims(2 * ranks);
    Parallel_Common::allgather_int(dims, 2, all_dims.data(), MPI_COMM_WORLD);
    std::vector<int> sizes(ranks);
    std::vector<int> displs(ranks);
    int total = 0;
    for (int r = 0; r < ranks; ++r)
    {
        sizes[r] = all_dims[2 * r] + all_dims[2 * r + 1];
        displs[r] = total;
        total += sizes[r];
    }
    std::vector<int> local;
    for (int i = 0; i < dims[0]; ++i) local.push_back(layout.local2global_row(i));
    for (int i = 0; i < dims[1]; ++i) local.push_back(layout.local2global_col(i));
    std::vector<int> indices(total);
    Parallel_Common::allgatherv_int(local.data(), mpi_size(local.size()), indices.data(),
                                  sizes.data(), displs.data(), MPI_COMM_WORLD);
    std::vector<int> pack;
    std::vector<int> unpack;
    p.counts.resize(ranks);
    p.offsets.resize(ranks);
    for (int r = 0; r < ranks; ++r)
    {
        p.offsets[r] = mpi_size(2 * unpack.size());
        const int nr = all_dims[2 * r];
        const int nc = all_dims[2 * r + 1];
        for (int c = 0; c < nc; ++c)
        {
            const int gc = indices[displs[r] + nr + c];
            for (int i = 0; i < nr; ++i)
            {
                const int gr = indices[displs[r] + i];
                if (gr > gc) continue;
                unpack.push_back(gc * n + gr);
                if (r == p.rank) pack.push_back(c * nr + i);
            }
        }
        p.counts[r] = mpi_size(2 * unpack.size()) - p.offsets[r];
    }
    p.local_upper = mpi_size(pack.size());
    p.global_upper = mpi_size(unpack.size());
    if (unpack.size() != static_cast<size_t>(n) * (n + 1) / 2)
        throw std::runtime_error("K-owner H/S layout does not cover a complete matrix");
    p.sendcounts.resize(ranks);
    p.sendoffsets.resize(ranks);
    p.recvcounts.resize(ranks);
    p.recvoffsets.resize(ranks);
    p.pack_indices.reset(new Buffer<int>(pack.size(), false));
    p.unpack_indices.reset(new Buffer<int>(unpack.size(), false));
    if (!pack.empty()) CHECK_CUDA(cudaMemcpy(p.pack_indices->device, pack.data(), pack.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(p.unpack_indices->device, unpack.data(), unpack.size() * sizeof(int), cudaMemcpyHostToDevice));
    p.send.reset(new Buffer<double2>(2 * pack.size(), true));
    p.recv.reset(new Buffer<double2>(2 * unpack.size(), true));
    p.owner_h.reset(new Buffer<double2>(dense, false));
    p.owner_s.reset(new Buffer<double2>(dense, false));
    // Only the upper triangle is consumed by cuSolver (UPLO=U).
    CHECK_CUDA(cudaMemset(p.owner_h->device, 0, dense * sizeof(double2)));
    CHECK_CUDA(cudaMemset(p.owner_s->device, 0, dense * sizeof(double2)));
    ModuleBase::timer::end("HSolverLCAO", "owner_gpu_layout");
}

KOwnerMatrixGpu::~KOwnerMatrixGpu() = default;

void KOwnerMatrixGpu::gather(const std::complex<double>* h, const std::complex<double>* s, int owner)
{
    auto& p = *impl_;
    ModuleBase::timer::start("HSolverLCAO", "owner_local_upload");
    // The device-input path never needs a second local dense H/S copy.
    if (!p.local_h)
    {
        p.local_h.reset(new Buffer<double2>(p.local_size, false));
        p.local_s.reset(new Buffer<double2>(p.local_size, false));
    }
    CHECK_CUDA(cudaMemcpy(p.local_h->device, h, p.local_size * sizeof(double2), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(p.local_s->device, s, p.local_size * sizeof(double2), cudaMemcpyHostToDevice));
    ModuleBase::timer::end("HSolverLCAO", "owner_local_upload");
    gather_device(reinterpret_cast<const std::complex<double>*>(p.local_h->device),
                  reinterpret_cast<const std::complex<double>*>(p.local_s->device), owner);
}

void KOwnerMatrixGpu::gather_device(const std::complex<double>* h, const std::complex<double>* s, int owner)
{
    auto& p = *impl_;
    ModuleBase::timer::start("HSolverLCAO", "owner_gpu_gather");
    if (p.local_upper)
    {
        pack_upper<<<(p.local_upper + 255) / 256, 256>>>(p.local_upper, p.pack_indices->device,
                                                       reinterpret_cast<const double2*>(h),
                                                       reinterpret_cast<const double2*>(s), p.send->device);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(p.send->host, p.send->device, 2 * p.local_upper * sizeof(double2), cudaMemcpyDeviceToHost));
    }
    std::fill(p.sendcounts.begin(), p.sendcounts.end(), 0);
    p.sendcounts[owner] = 2 * p.local_upper;
    std::fill(p.recvcounts.begin(), p.recvcounts.end(), 0);
    if (p.rank == owner)
    {
        p.recvcounts = p.counts;
        p.recvoffsets = p.offsets;
    }
    // Deliberate pinned-host transport: no dependency on CUDA-aware MPI.
    MPI_Comm comm = MPI_COMM_WORLD;
    Parallel_Common::alltoallv_data(reinterpret_cast<std::complex<double>*>(p.send->host),
                                  p.sendcounts.data(), p.sendoffsets.data(),
                                  reinterpret_cast<std::complex<double>*>(p.recv->host),
                                  p.recvcounts.data(), p.recvoffsets.data(), comm);
    if (p.rank == owner)
    {
        CHECK_CUDA(cudaMemcpy(p.recv->device, p.recv->host, 2 * p.global_upper * sizeof(double2), cudaMemcpyHostToDevice));
        unpack_upper<<<(p.global_upper + 255) / 256, 256>>>(p.global_upper, p.unpack_indices->device,
                                                         p.recv->device, p.owner_h->device, p.owner_s->device);
        CHECK_CUDA(cudaGetLastError());
    }
    ModuleBase::timer::end("HSolverLCAO", "owner_gpu_gather");
}

std::complex<double>* KOwnerMatrixGpu::h_device()
{
    return reinterpret_cast<std::complex<double>*>(impl_->owner_h->device);
}
std::complex<double>* KOwnerMatrixGpu::s_device()
{
    return reinterpret_cast<std::complex<double>*>(impl_->owner_s->device);
}
}
