#include "source_base/timer.h"
#include "source_basis/module_pw/kernels/pw_op.h"
#include "pw_basis.h"

#include <vector>

namespace ModulePW
{
#if (defined(__CUDA) || defined(__ROCM))
template <typename FPTYPE>
void PW_Basis::real2recip_gpu(const FPTYPE* in, std::complex<FPTYPE>* out, const bool add, const FPTYPE factor) const
{
    ModuleBase::timer::start(this->classname, "real_to_recip gpu");
    assert(this->poolnproc == 1);
    const size_t size = this->nrxx;
    base_device::memory::cast_memory_op<std::complex<FPTYPE>, FPTYPE,base_device::DEVICE_GPU, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
        in,
        size);

    this->fft_bundle.fft3D_forward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                                   this->nxyz,
                                                                   add,
                                                                   factor,
                                                                   this->ig2ixyz_gpu,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);
    ModuleBase::timer::end(this->classname, "real_to_recip gpu");
}
template <typename FPTYPE>
void PW_Basis::real2recip_gpu(const std::complex<FPTYPE>* in,
                              std::complex<FPTYPE>* out,
                              const bool add,
                              const FPTYPE factor) const
{
    ModuleBase::timer::start(this->classname, "real_to_recip gpu");
    assert(this->poolnproc == 1);
    base_device::memory::synchronize_memory_op<std::complex<FPTYPE>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_GPU>()(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                          in,
                                                                          this->nrxx);
    this->fft_bundle.fft3D_forward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                                   this->nxyz,
                                                                   add,
                                                                   factor,
                                                                   this->ig2ixyz_gpu,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);
    ModuleBase::timer::end(this->classname, "real_to_recip gpu");
}

template <typename FPTYPE>
void PW_Basis::recip2real_gpu(const std::complex<FPTYPE>* in, FPTYPE* out, const bool add, const FPTYPE factor) const
{
    ModuleBase::timer::start(this->classname, "recip_to_real gpu");
    assert(this->poolnproc == 1);
    // ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxr_3d_data<FPTYPE>(), this->nxyz);
    base_device::memory::set_memory_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
        0,
        this->nxyz);
    set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                         this->ig2ixyz_gpu,
                                                         in,
                                                         this->fft_bundle.get_auxr_3d_data<FPTYPE>());
    this->fft_bundle.fft3D_backward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                    this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>()(this->nrxx,
                                                                   add,
                                                                   factor,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);

    ModuleBase::timer::end(this->classname, "recip_to_real gpu");
}
template <typename FPTYPE>
void PW_Basis::recip2real_gpu(const std::complex<FPTYPE>* in,
                              std::complex<FPTYPE>* out,
                              const bool add,
                              const FPTYPE factor) const
{
    ModuleBase::timer::start(this->classname, "recip_to_real gpu");
    assert(this->poolnproc == 1);
    // ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxr_3d_data<double>(), this->nxyz);
    base_device::memory::set_memory_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
        0,
        this->nxyz);

    set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                         this->ig2ixyz_gpu,
                                                         in,
                                                         this->fft_bundle.get_auxr_3d_data<FPTYPE>());
    this->fft_bundle.fft3D_backward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                    this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>()(this->nrxx,
                                                                   add,
                                                                   factor,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);

    ModuleBase::timer::end(this->classname, "recip_to_real gpu");
}

#ifdef __CUDA
void PW_Basis::setup_gpu_fft_companion()
{
    assert(this->poolnproc == 1);
    assert(this->precision == "double");
    if (this->gpu_fft_bundle_ != nullptr)
    {
        return;
    }

    std::vector<int> ig2ixyz(this->npw);
    for (int ig = 0; ig < this->npw; ++ig)
    {
        const int isz = this->ig2isz[ig];
        const int iz = isz % this->nz;
        const int is = isz / this->nz;
        const int ixy = this->is2fftixy[is];
        const int iy = ixy % this->ny;
        const int ix = ixy / this->ny;
        ig2ixyz[ig] = iz + iy * this->nz + ix * this->ny * this->nz;
    }
    base_device::memory::resize_memory_op<int, base_device::DEVICE_GPU>()(
        this->ig2ixyz_gpu,
        this->npw);
    base_device::memory::synchronize_memory_op<int,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_CPU>()(
        this->ig2ixyz_gpu,
        ig2ixyz.data(),
        this->npw);

    this->gpu_fft_bundle_.reset(new ModuleBase::FFT_Bundle("gpu", "double"));
    this->gpu_fft_bundle_->initfft(this->nx,
                                   this->ny,
                                   this->nz,
                                   this->lix,
                                   this->rix,
                                   this->nst,
                                   this->nplane,
                                   1,
                                   this->gamma_only,
                                   this->xprime);
    this->gpu_fft_bundle_->setupFFT();

    base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
        this->gpu_fft_complex_real_,
        this->nrxx);
    base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
        this->gpu_fft_reciprocal_,
        this->npw);
    base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(
        this->gpu_fft_real_,
        this->nrxx);
}

void PW_Basis::real2recip_gpu_companion(
    const double* in,
    std::complex<double>* out,
    const bool add,
    const double factor) const
{
    assert(this->gpu_fft_bundle_ != nullptr);
    base_device::memory::cast_memory_op<std::complex<double>,
                                        double,
                                        base_device::DEVICE_GPU,
                                        base_device::DEVICE_GPU>()(
        this->gpu_fft_bundle_->get_auxr_3d_data<double>(),
        in,
        this->nrxx);
    this->gpu_fft_bundle_->fft3D_forward(
        this->gpu_fft_bundle_->get_auxr_3d_data<double>(),
        this->gpu_fft_bundle_->get_auxr_3d_data<double>());
    set_real_to_recip_output_op<double, base_device::DEVICE_GPU>()(
        this->npw,
        this->nxyz,
        add,
        factor,
        this->ig2ixyz_gpu,
        this->gpu_fft_bundle_->get_auxr_3d_data<double>(),
        out);
}

void PW_Basis::real2recip_gpu_companion(
    const std::complex<double>* in,
    std::complex<double>* out,
    const bool add,
    const double factor) const
{
    assert(this->gpu_fft_bundle_ != nullptr);
    base_device::memory::synchronize_memory_op<std::complex<double>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_GPU>()(
        this->gpu_fft_bundle_->get_auxr_3d_data<double>(),
        in,
        this->nrxx);
    this->gpu_fft_bundle_->fft3D_forward(
        this->gpu_fft_bundle_->get_auxr_3d_data<double>(),
        this->gpu_fft_bundle_->get_auxr_3d_data<double>());
    set_real_to_recip_output_op<double, base_device::DEVICE_GPU>()(
        this->npw,
        this->nxyz,
        add,
        factor,
        this->ig2ixyz_gpu,
        this->gpu_fft_bundle_->get_auxr_3d_data<double>(),
        out);
}

void PW_Basis::recip2real_gpu_companion(
    const std::complex<double>* in,
    double* out,
    const bool add,
    const double factor) const
{
    assert(this->gpu_fft_bundle_ != nullptr);
    base_device::memory::set_memory_op<std::complex<double>,
                                       base_device::DEVICE_GPU>()(
        this->gpu_fft_bundle_->get_auxr_3d_data<double>(),
        0,
        this->nxyz);
    set_3d_fft_box_op<double, base_device::DEVICE_GPU>()(
        this->npw,
        this->ig2ixyz_gpu,
        in,
        this->gpu_fft_bundle_->get_auxr_3d_data<double>());
    this->gpu_fft_bundle_->fft3D_backward(
        this->gpu_fft_bundle_->get_auxr_3d_data<double>(),
        this->gpu_fft_bundle_->get_auxr_3d_data<double>());
    set_recip_to_real_output_op<double, base_device::DEVICE_GPU>()(
        this->nrxx,
        add,
        factor,
        this->gpu_fft_bundle_->get_auxr_3d_data<double>(),
        out);
}

void PW_Basis::recip2real_gpu_companion(
    const std::complex<double>* in,
    std::complex<double>* out,
    const bool add,
    const double factor) const
{
    assert(this->gpu_fft_bundle_ != nullptr);
    base_device::memory::set_memory_op<std::complex<double>,
                                       base_device::DEVICE_GPU>()(
        this->gpu_fft_bundle_->get_auxr_3d_data<double>(),
        0,
        this->nxyz);
    set_3d_fft_box_op<double, base_device::DEVICE_GPU>()(
        this->npw,
        this->ig2ixyz_gpu,
        in,
        this->gpu_fft_bundle_->get_auxr_3d_data<double>());
    this->gpu_fft_bundle_->fft3D_backward(
        this->gpu_fft_bundle_->get_auxr_3d_data<double>(),
        this->gpu_fft_bundle_->get_auxr_3d_data<double>());
    set_recip_to_real_output_op<double, base_device::DEVICE_GPU>()(
        this->nrxx,
        add,
        factor,
        this->gpu_fft_bundle_->get_auxr_3d_data<double>(),
        out);
}

void PW_Basis::real2recip_gpu_host(const double* in,
                                   std::complex<double>* out,
                                   const bool add,
                                   const double factor) const
{
    base_device::memory::synchronize_memory_op<double,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_CPU>()(
        this->gpu_fft_real_,
        in,
        this->nrxx);
    if (add)
    {
        base_device::memory::synchronize_memory_op<std::complex<double>,
                                                   base_device::DEVICE_GPU,
                                                   base_device::DEVICE_CPU>()(
            this->gpu_fft_reciprocal_,
            out,
            this->npw);
    }
    this->real2recip_gpu_companion(
        this->gpu_fft_real_, this->gpu_fft_reciprocal_, add, factor);
    base_device::memory::synchronize_memory_op<std::complex<double>,
                                               base_device::DEVICE_CPU,
                                               base_device::DEVICE_GPU>()(
        out,
        this->gpu_fft_reciprocal_,
        this->npw);
}

void PW_Basis::real2recip_gpu_host(const std::complex<double>* in,
                                   std::complex<double>* out,
                                   const bool add,
                                   const double factor) const
{
    base_device::memory::synchronize_memory_op<std::complex<double>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_CPU>()(
        this->gpu_fft_complex_real_,
        in,
        this->nrxx);
    if (add)
    {
        base_device::memory::synchronize_memory_op<std::complex<double>,
                                                   base_device::DEVICE_GPU,
                                                   base_device::DEVICE_CPU>()(
            this->gpu_fft_reciprocal_,
            out,
            this->npw);
    }
    this->real2recip_gpu_companion(
        this->gpu_fft_complex_real_, this->gpu_fft_reciprocal_, add, factor);
    base_device::memory::synchronize_memory_op<std::complex<double>,
                                               base_device::DEVICE_CPU,
                                               base_device::DEVICE_GPU>()(
        out,
        this->gpu_fft_reciprocal_,
        this->npw);
}

void PW_Basis::recip2real_gpu_host(const std::complex<double>* in,
                                   double* out,
                                   const bool add,
                                   const double factor) const
{
    base_device::memory::synchronize_memory_op<std::complex<double>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_CPU>()(
        this->gpu_fft_reciprocal_,
        in,
        this->npw);
    if (add)
    {
        base_device::memory::synchronize_memory_op<double,
                                                   base_device::DEVICE_GPU,
                                                   base_device::DEVICE_CPU>()(
            this->gpu_fft_real_,
            out,
            this->nrxx);
    }
    this->recip2real_gpu_companion(
        this->gpu_fft_reciprocal_, this->gpu_fft_real_, add, factor);
    base_device::memory::synchronize_memory_op<double,
                                               base_device::DEVICE_CPU,
                                               base_device::DEVICE_GPU>()(
        out,
        this->gpu_fft_real_,
        this->nrxx);
}

void PW_Basis::recip2real_gpu_host(const std::complex<double>* in,
                                   std::complex<double>* out,
                                   const bool add,
                                   const double factor) const
{
    base_device::memory::synchronize_memory_op<std::complex<double>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_CPU>()(
        this->gpu_fft_reciprocal_,
        in,
        this->npw);
    if (add)
    {
        base_device::memory::synchronize_memory_op<std::complex<double>,
                                                   base_device::DEVICE_GPU,
                                                   base_device::DEVICE_CPU>()(
            this->gpu_fft_complex_real_,
            out,
            this->nrxx);
    }
    this->recip2real_gpu_companion(
        this->gpu_fft_reciprocal_, this->gpu_fft_complex_real_, add, factor);
    base_device::memory::synchronize_memory_op<std::complex<double>,
                                               base_device::DEVICE_CPU,
                                               base_device::DEVICE_GPU>()(
        out,
        this->gpu_fft_complex_real_,
        this->nrxx);
}
#endif

template void PW_Basis::real2recip_gpu<double>(const double* in,
                                               std::complex<double>* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::real2recip_gpu<float>(const float* in,
                                              std::complex<float>* out,
                                              const bool add,
                                              const float factor) const;

template void PW_Basis::real2recip_gpu<double>(const std::complex<double>* in,
                                               std::complex<double>* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::real2recip_gpu<float>(const std::complex<float>* in,
                                              std::complex<float>* out,
                                              const bool add,
                                              const float factor) const;

template void PW_Basis::recip2real_gpu<double>(const std::complex<double>* in,
                                               double* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::recip2real_gpu<float>(const std::complex<float>* in,
                                              float* out,
                                              const bool add,
                                              const float factor) const;

template void PW_Basis::recip2real_gpu<double>(const std::complex<double>* in,
                                               std::complex<double>* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::recip2real_gpu<float>(const std::complex<float>* in,
                                              std::complex<float>* out,
                                              const bool add,
                                              const float factor) const;

#endif
} // namespace ModulePW
