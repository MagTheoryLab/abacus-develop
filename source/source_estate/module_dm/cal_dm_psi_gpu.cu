#include "cal_dm_psi.h"

#include "source_base/kernels/math_kernel_op.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/timer.h"
#include <base/macros/macros.h>
#include <thrust/complex.h>

#include <vector>

namespace elecstate
{
namespace
{

__global__ void weight_conjugate_wfc_kernel(const int size,
                                             const int nbasis,
                                             const thrust::complex<double>* wfc,
                                             const double* weights,
                                             thrust::complex<double>* out)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        out[index] = thrust::conj(wfc[index]) * weights[index / nbasis];
    }
}

} // namespace

void weight_conjugate_wfc_gpu(const int nbands,
                              const int nbasis,
                              const std::complex<double>* wfc,
                              const double* weights,
                              std::complex<double>* out)
{
    const int size = nbands * nbasis;
    if (size <= 0)
    {
        return;
    }

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    weight_conjugate_wfc_kernel<<<blocks, threads>>>(
        size,
        nbasis,
        reinterpret_cast<const thrust::complex<double>*>(wfc),
        weights,
        reinterpret_cast<thrust::complex<double>*>(out));
    CHECK_CUDA(cudaGetLastError());
}

void cal_dm_psi_gpu_single_rank(const Parallel_Orbitals* para_v,
                                const ModuleBase::matrix& wg,
                                const psi::Psi<std::complex<double>>& wfc,
                                elecstate::DensityMatrix<std::complex<double>, double>& dm)
{
    ModuleBase::TITLE("elecstate", "cal_dm_psi_gpu_single_rank");
    ModuleBase::timer::start("elecstate", "cal_dm_psi_gpu_single_rank");

    const int nbands_local = wfc.get_nbands();
    const int nbasis_local = wfc.get_nbasis();
    const int wfc_size = nbands_local * nbasis_local;
    const int dmk_size = nbasis_local * nbasis_local;

    std::complex<double>* wfc_device = nullptr;
    std::complex<double>* weighted_wfc_device = nullptr;
    std::complex<double>* dmk_device = nullptr;
    double* weights_device = nullptr;
    base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
        wfc_device,
        wfc_size);
    base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
        weighted_wfc_device,
        wfc_size);
    base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
        dmk_device,
        dmk_size);
    base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(weights_device,
                                                                            nbands_local);

    std::vector<double> weights(nbands_local, 0.0);
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    const char normal = 'N';
    const char transpose = 'T';

    for (int ik = 0; ik < wfc.get_nk(); ++ik)
    {
        int ib_global = 0;
        for (int ib_local = 0; ib_local < nbands_local; ++ib_local)
        {
            while (ib_local != para_v->global2local_col(ib_global))
            {
                ++ib_global;
                if (ib_global >= wg.nc)
                {
                    break;
                }
            }
            weights[ib_local] = ib_global < wg.nc ? wg(ik, ib_global) : 0.0;
        }

        wfc.fix_k(ik);
        base_device::memory::synchronize_memory_op<std::complex<double>,
                                                   base_device::DEVICE_GPU,
                                                   base_device::DEVICE_CPU>()(
            wfc_device,
            wfc.get_pointer(),
            wfc_size);
        base_device::memory::synchronize_memory_op<double,
                                                   base_device::DEVICE_GPU,
                                                   base_device::DEVICE_CPU>()(
            weights_device,
            weights.data(),
            nbands_local);
        weight_conjugate_wfc_gpu(nbands_local,
                                 nbasis_local,
                                 wfc_device,
                                 weights_device,
                                 weighted_wfc_device);

        ModuleBase::timer::start("cal_dm_psi_gpu", "zgemm");
        ModuleBase::gemm_op<std::complex<double>, base_device::DEVICE_GPU>()(
            normal,
            transpose,
            nbasis_local,
            nbasis_local,
            nbands_local,
            &one,
            weighted_wfc_device,
            nbasis_local,
            wfc_device,
            nbasis_local,
            &zero,
            dmk_device,
            nbasis_local);
        ModuleBase::timer::end("cal_dm_psi_gpu", "zgemm");

        base_device::memory::synchronize_memory_op<std::complex<double>,
                                                   base_device::DEVICE_CPU,
                                                   base_device::DEVICE_GPU>()(
            dm.get_DMK_pointer(ik),
            dmk_device,
            dmk_size);
    }

    base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
        wfc_device);
    base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
        weighted_wfc_device);
    base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
        dmk_device);
    base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>()(weights_device);

    ModuleBase::timer::end("elecstate", "cal_dm_psi_gpu_single_rank");
}

} // namespace elecstate
