#include "dftu_lcao_gpu.h"

#include <base/macros/macros.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <algorithm>

namespace hamilt
{
namespace dftu_gpu
{
namespace
{

struct Cache
{
    double* projections = nullptr;
    ProjectionTask* tasks = nullptr;
    double* dm = nullptr;
    double* occupations = nullptr;
    thrust::complex<double>* onsite = nullptr;
    thrust::complex<double>* hr = nullptr;
    std::size_t task_count = 0;
    std::size_t dm_size = 0;
    std::size_t hr_size = 0;
    std::size_t onsite_size = 0;
};

__global__ void occupation_kernel(const double* projections,
                                  const ProjectionTask* tasks,
                                  std::size_t task_count,
                                  const double* dm,
                                  double* occupations)
{
    const std::size_t task_index = blockIdx.x;
    if (task_index >= task_count)
    {
        return;
    }
    const ProjectionTask task = tasks[task_index];
    const int output_size = 4 * task.projector_size * task.projector_size;
    for (int output = threadIdx.x; output < output_size; output += blockDim.x)
    {
        if (!task.has_dm)
        {
            continue;
        }
        const int projector_pair = output % (task.projector_size * task.projector_size);
        const int spin = output / (task.projector_size * task.projector_size);
        const int m1 = projector_pair / task.projector_size;
        const int m2 = projector_pair - m1 * task.projector_size;
        const int spin_row = spin / 2;
        const int spin_col = spin - spin_row * 2;
        double sum = 0.0;
        for (int row = 0; row < task.row_orbitals; ++row)
        {
            const double left = projections[task.left_offset + row * task.projector_size + m1];
            for (int col = 0; col < task.col_orbitals; ++col)
            {
                const double right = projections[task.right_offset + col * task.projector_size + m2];
                const std::size_t dm_index = task.dm_offset
                    + (row * 2 + spin_row) * task.matrix_cols
                    + col * 2 + spin_col;
                sum += left * right * dm[dm_index];
            }
        }
        atomicAdd(occupations + task.onsite_offset + output, sum);
    }
}

__global__ void hamiltonian_kernel(const double* projections,
                                   const ProjectionTask* tasks,
                                   std::size_t task_count,
                                   const thrust::complex<double>* onsite,
                                   thrust::complex<double>* hr)
{
    const std::size_t task_index = blockIdx.x;
    if (task_index >= task_count)
    {
        return;
    }
    const ProjectionTask task = tasks[task_index];
    const int output_size = task.row_orbitals * task.col_orbitals * 4;
    for (int output = threadIdx.x; output < output_size; output += blockDim.x)
    {
        if (!task.has_hr)
        {
            continue;
        }
        const int spin = output & 3;
        const int orbital_pair = output >> 2;
        const int row = orbital_pair / task.col_orbitals;
        const int col = orbital_pair - row * task.col_orbitals;
        const int spin_row = spin / 2;
        const int spin_col = spin - spin_row * 2;
        thrust::complex<double> sum(0.0, 0.0);
        for (int m1 = 0; m1 < task.projector_size; ++m1)
        {
            const double left = projections[task.left_offset + row * task.projector_size + m1];
            for (int m2 = 0; m2 < task.projector_size; ++m2)
            {
                const double right = projections[task.right_offset + col * task.projector_size + m2];
                const std::size_t onsite_index = task.onsite_offset
                    + spin * task.projector_size * task.projector_size
                    + m1 * task.projector_size + m2;
                sum += left * right * onsite[onsite_index];
            }
        }
        const std::size_t hr_index = task.hr_offset
            + (row * 2 + spin_row) * task.matrix_cols
            + col * 2 + spin_col;
        atomicAdd(reinterpret_cast<double*>(hr + hr_index), sum.real());
        atomicAdd(reinterpret_cast<double*>(hr + hr_index) + 1, sum.imag());
    }
}

} // namespace

void* create_cache(const double* projections,
                   std::size_t projection_count,
                   const ProjectionTask* tasks,
                   std::size_t task_count,
                   std::size_t dm_size,
                   std::size_t hr_size,
                   std::size_t onsite_size)
{
    Cache* cache = new Cache;
    cache->task_count = task_count;
    cache->dm_size = dm_size;
    cache->hr_size = hr_size;
    cache->onsite_size = onsite_size;
    CHECK_CUDA(cudaMalloc(&cache->projections, std::max<std::size_t>(1, projection_count) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&cache->tasks, std::max<std::size_t>(1, task_count) * sizeof(ProjectionTask)));
    CHECK_CUDA(cudaMalloc(&cache->dm, std::max<std::size_t>(1, dm_size) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&cache->occupations, onsite_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&cache->onsite, onsite_size * sizeof(thrust::complex<double>)));
    CHECK_CUDA(cudaMalloc(&cache->hr, std::max<std::size_t>(1, hr_size) * sizeof(thrust::complex<double>)));
    CHECK_CUDA(cudaMemcpy(cache->projections,
                          projections,
                          projection_count * sizeof(double),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(cache->tasks,
                          tasks,
                          task_count * sizeof(ProjectionTask),
                          cudaMemcpyHostToDevice));
    return cache;
}

void compute_occupations(void* opaque_cache,
                         const double* dm,
                         double* occupations)
{
    Cache* cache = static_cast<Cache*>(opaque_cache);
    CHECK_CUDA(cudaMemcpy(cache->dm,
                          dm,
                          cache->dm_size * sizeof(double),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(cache->occupations, 0, cache->onsite_size * sizeof(double)));
    if (cache->task_count > 0)
    {
        occupation_kernel<<<cache->task_count, 128>>>(cache->projections,
                                                       cache->tasks,
                                                       cache->task_count,
                                                       cache->dm,
                                                       cache->occupations);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaMemcpy(occupations,
                          cache->occupations,
                          cache->onsite_size * sizeof(double),
                          cudaMemcpyDeviceToHost));
}

void add_hubbard_hamiltonian(void* opaque_cache,
                            const std::complex<double>* onsite,
                            std::complex<double>* hr)
{
    Cache* cache = static_cast<Cache*>(opaque_cache);
    CHECK_CUDA(cudaMemcpy(cache->onsite,
                          onsite,
                          cache->onsite_size * sizeof(thrust::complex<double>),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(cache->hr,
                          hr,
                          cache->hr_size * sizeof(thrust::complex<double>),
                          cudaMemcpyHostToDevice));
    if (cache->task_count > 0)
    {
        hamiltonian_kernel<<<cache->task_count, 256>>>(cache->projections,
                                                       cache->tasks,
                                                       cache->task_count,
                                                       cache->onsite,
                                                       cache->hr);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaMemcpy(hr,
                          cache->hr,
                          cache->hr_size * sizeof(thrust::complex<double>),
                          cudaMemcpyDeviceToHost));
}

const std::complex<double>* build_hubbard_hamiltonian(
    void* opaque_cache,
    const std::complex<double>* onsite)
{
    Cache* cache = static_cast<Cache*>(opaque_cache);
    CHECK_CUDA(cudaMemcpy(cache->onsite,
                          onsite,
                          cache->onsite_size * sizeof(thrust::complex<double>),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(cache->hr, 0, cache->hr_size * sizeof(thrust::complex<double>)));
    if (cache->task_count > 0)
    {
        hamiltonian_kernel<<<cache->task_count, 256>>>(cache->projections,
                                                       cache->tasks,
                                                       cache->task_count,
                                                       cache->onsite,
                                                       cache->hr);
        CHECK_CUDA(cudaGetLastError());
    }
    return reinterpret_cast<const std::complex<double>*>(cache->hr);
}

void destroy_cache(void* opaque_cache)
{
    Cache* cache = static_cast<Cache*>(opaque_cache);
    if (cache == nullptr)
    {
        return;
    }
    CHECK_CUDA(cudaFree(cache->projections));
    CHECK_CUDA(cudaFree(cache->tasks));
    CHECK_CUDA(cudaFree(cache->dm));
    CHECK_CUDA(cudaFree(cache->occupations));
    CHECK_CUDA(cudaFree(cache->onsite));
    CHECK_CUDA(cudaFree(cache->hr));
    delete cache;
}

} // namespace dftu_gpu
} // namespace hamilt
