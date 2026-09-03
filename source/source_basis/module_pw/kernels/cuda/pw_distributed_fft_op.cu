#include "source_basis/module_pw/kernels/pw_distributed_fft_op.h"

#include "source_base/module_device/device_check.h"

#include <cuda_runtime.h>
#include <thrust/complex.h>

namespace ModulePW
{
namespace
{

constexpr int threads_per_block = 256;

__global__ void pack_fft_slabs_kernel(
    const int count,
    const int nplane,
    const int* istot2ixy,
    const thrust::complex<double>* slabs,
    thrust::complex<double>* packed)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        const int istot = index / nplane;
        const int iz = index % nplane;
        packed[index] = slabs[istot2ixy[istot] * nplane + iz];
    }
}

__global__ void unpack_fft_sticks_kernel(
    const int count,
    const int nz,
    const int* numz,
    const int* startg,
    const int* z_owner,
    const int* z_local,
    const thrust::complex<double>* packed,
    thrust::complex<double>* sticks)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
    {
        return;
    }
    const int is = index / nz;
    const int iz = index % nz;
    const int ip = z_owner[iz];
    const int nzip = numz[ip];
    sticks[index] = packed[startg[ip] + is * nzip + z_local[iz]];
}

__global__ void pack_fft_sticks_kernel(
    const int count,
    const int nz,
    const int* numz,
    const int* startg,
    const int* z_owner,
    const int* z_local,
    const thrust::complex<double>* sticks,
    thrust::complex<double>* packed)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
    {
        return;
    }
    const int is = index / nz;
    const int iz = index % nz;
    const int ip = z_owner[iz];
    const int nzip = numz[ip];
    packed[startg[ip] + is * nzip + z_local[iz]] = sticks[index];
}

__global__ void unpack_fft_slabs_kernel(
    const int count,
    const int nplane,
    const int* istot2ixy,
    const thrust::complex<double>* packed,
    thrust::complex<double>* slabs)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        const int istot = index / nplane;
        const int iz = index % nplane;
        slabs[istot2ixy[istot] * nplane + iz] = packed[index];
    }
}

} // namespace

void pack_fft_slabs_gpu(const int nstot,
                        const int nplane,
                        const int* istot2ixy,
                        const std::complex<double>* slabs,
                        std::complex<double>* packed)
{
    const int count = nstot * nplane;
    const int blocks = (count + threads_per_block - 1) / threads_per_block;
    pack_fft_slabs_kernel<<<blocks, threads_per_block>>>(
        count,
        nplane,
        istot2ixy,
        reinterpret_cast<const thrust::complex<double>*>(slabs),
        reinterpret_cast<thrust::complex<double>*>(packed));
    CHECK_CUDA(cudaGetLastError());
}

void unpack_fft_sticks_gpu(const int nst,
                           const int nz,
                           const int* numz,
                           const int* startg,
                           const int* z_owner,
                           const int* z_local,
                           const std::complex<double>* packed,
                           std::complex<double>* sticks)
{
    const int count = nst * nz;
    const int blocks = (count + threads_per_block - 1) / threads_per_block;
    unpack_fft_sticks_kernel<<<blocks, threads_per_block>>>(
        count,
        nz,
        numz,
        startg,
        z_owner,
        z_local,
        reinterpret_cast<const thrust::complex<double>*>(packed),
        reinterpret_cast<thrust::complex<double>*>(sticks));
    CHECK_CUDA(cudaGetLastError());
}

void pack_fft_sticks_gpu(const int nst,
                         const int nz,
                         const int* numz,
                         const int* startg,
                         const int* z_owner,
                         const int* z_local,
                         const std::complex<double>* sticks,
                         std::complex<double>* packed)
{
    const int count = nst * nz;
    const int blocks = (count + threads_per_block - 1) / threads_per_block;
    pack_fft_sticks_kernel<<<blocks, threads_per_block>>>(
        count,
        nz,
        numz,
        startg,
        z_owner,
        z_local,
        reinterpret_cast<const thrust::complex<double>*>(sticks),
        reinterpret_cast<thrust::complex<double>*>(packed));
    CHECK_CUDA(cudaGetLastError());
}

void unpack_fft_slabs_gpu(const int nrxx,
                          const int nstot,
                          const int nplane,
                          const int* istot2ixy,
                          const std::complex<double>* packed,
                          std::complex<double>* slabs)
{
    CHECK_CUDA(cudaMemset(slabs,
                          0,
                          static_cast<std::size_t>(nrxx)
                              * sizeof(std::complex<double>)));
    const int count = nstot * nplane;
    const int blocks = (count + threads_per_block - 1) / threads_per_block;
    unpack_fft_slabs_kernel<<<blocks, threads_per_block>>>(
        count,
        nplane,
        istot2ixy,
        reinterpret_cast<const thrust::complex<double>*>(packed),
        reinterpret_cast<thrust::complex<double>*>(slabs));
    CHECK_CUDA(cudaGetLastError());
}

} // namespace ModulePW
