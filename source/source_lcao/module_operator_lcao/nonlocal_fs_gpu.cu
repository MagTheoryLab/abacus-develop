#include "nonlocal_fs_gpu.h"
#include <base/macros/macros.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>

namespace hamilt
{
namespace nonlocal_gpu
{
namespace
{
constexpr int block_size = 128;

__device__ double product_real(double dm, const Coupling& d)
{
    return dm * d.real;
}
__device__ double product_real(thrust::complex<double> dm, const Coupling& d)
{
    // The legacy contraction is Re(DMR * D), without conjugation.
    return dm.real() * d.real - dm.imag() * d.imag;
}

template <typename T>
__global__ void contract(const Task* tasks, const double* projections,
                         const Coupling* couplings, const T* dm,
                         bool cal_force, bool cal_stress, double* force, double* stress)
{
    const Task task = tasks[blockIdx.x];
    double value[9] = {};
    for (int pair = threadIdx.x; pair < task.rows * task.cols; pair += blockDim.x)
    {
        const int row = pair / task.cols;
        const int col = pair % task.cols;
        const double* left = projections + task.left + row * 4 * task.projectors;
        const double* right = projections + task.right + col * 4 * task.projectors;
        for (int no = 0; no < task.coupling_count; ++no)
        {
            const Coupling d = couplings[task.coupling + no];
            const int sr = d.spin / task.npol;
            const int sc = d.spin % task.npol;
            const std::size_t index = task.dm
                + (row * task.npol + sr) * task.cols * task.npol
                + col * task.npol + sc;
            const double weight = product_real(dm[index], d);
            double first[3];
            double second[3];
            for (int axis = 0; axis < 3; ++axis)
            {
                first[axis] = left[d.p1 + (axis + 1) * task.projectors] * right[d.p2] * weight;
                second[axis] = left[d.p1] * right[d.p2 + (axis + 1) * task.projectors] * weight;
                if (cal_force) value[axis] += first[axis];
            }
            if (cal_stress)
            {
                int component = 3;
                for (int a = 0; a < 3; ++a)
                {
                    for (int b = a; b < 3; ++b)
                    {
                        value[component++] += first[a] * task.dis1[b] + second[a] * task.dis2[b];
                    }
                }
            }
        }
    }
    __shared__ double partial[9][block_size];
    for (int c = 0; c < 9; ++c) partial[c][threadIdx.x] = value[c];
    __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            for (int c = 0; c < 9; ++c)
                partial[c][threadIdx.x] += partial[c][threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        if (cal_force)
        {
            for (int a = 0; a < 3; ++a)
            {
                atomicAdd(force + 3 * task.atom + a, partial[a][0]);
                atomicAdd(force + 3 * task.center + a, -partial[a][0]);
            }
        }
        if (cal_stress)
        {
            for (int c = 0; c < 6; ++c) atomicAdd(stress + c, partial[c + 3][0]);
        }
    }
}

template <typename T>
class DeviceArray
{
  public:
    explicit DeviceArray(std::size_t count)
    {
        if (count) CHECK_CUDA(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceArray() { if (data) CHECK_CUDA(cudaFree(data)); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    T* data = nullptr;
};

template <typename T>
void run(const Batch& batch, const T* dm, std::size_t dm_size,
         int nat, bool cal_force, bool cal_stress, double* force, double* stress)
{
    if (batch.tasks.empty()) return;
    DeviceArray<Task> tasks(batch.tasks.size());
    DeviceArray<double> projections(batch.projections.size());
    DeviceArray<Coupling> couplings(batch.couplings.size());
    DeviceArray<T> density(dm_size);
    DeviceArray<double> output(3 * nat + 6);
    CHECK_CUDA(cudaMemcpy(tasks.data, batch.tasks.data(), batch.tasks.size() * sizeof(Task), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(projections.data, batch.projections.data(), batch.projections.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(couplings.data, batch.couplings.data(), batch.couplings.size() * sizeof(Coupling), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(density.data, dm, dm_size * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(output.data, 0, (3 * nat + 6) * sizeof(double)));
    contract<<<batch.tasks.size(), block_size>>>(tasks.data, projections.data, couplings.data,
                                               density.data, cal_force, cal_stress,
                                               output.data, output.data + 3 * nat);
    CHECK_CUDA(cudaGetLastError());
    if (cal_force) CHECK_CUDA(cudaMemcpy(force, output.data, 3 * nat * sizeof(double), cudaMemcpyDeviceToHost));
    if (cal_stress) CHECK_CUDA(cudaMemcpy(stress, output.data + 3 * nat, 6 * sizeof(double), cudaMemcpyDeviceToHost));
}
}
void compute(const Batch& batch, const double* dm, std::size_t dm_size,
             int nat, bool cal_force, bool cal_stress, double* force, double* stress)
{
    run(batch, dm, dm_size, nat, cal_force, cal_stress, force, stress);
}
void compute(const Batch& batch, const std::complex<double>* dm, std::size_t dm_size,
             int nat, bool cal_force, bool cal_stress, double* force, double* stress)
{
    static_assert(sizeof(std::complex<double>) == sizeof(thrust::complex<double>), "complex layout");
    run(batch, reinterpret_cast<const thrust::complex<double>*>(dm), dm_size,
        nat, cal_force, cal_stress, force, stress);
}
}
}
