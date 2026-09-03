#include "fft_cuda.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/module_device/device_check.h"

namespace ModuleBase
{
template <typename FPTYPE>
void FFT_CUDA<FPTYPE>::initfft(int nx_in, int ny_in, int nz_in)
{
    this->nx = nx_in;
    this->ny = ny_in;
    this->nz = nz_in;
}
template <typename FPTYPE>
void FFT_CUDA<FPTYPE>::initfft(int nx_in,
                               int ny_in,
                               int nz_in,
                               int,
                               int,
                               int ns_in,
                               int nplane_in,
                               int nproc_in,
                               bool,
                               bool)
{
    this->nx = nx_in;
    this->ny = ny_in;
    this->nz = nz_in;
    this->ns = ns_in;
    this->nplane = nplane_in;
    this->nproc = nproc_in;
}
template <>
void FFT_CUDA<float>::setupFFT()
{
    if (this->nproc == 1)
    {
        CHECK_CUFFT(cufftPlan3d(&c_handle, this->nx, this->ny, this->nz, CUFFT_C2C));
        resmem_cd_op()(this->c_auxr_3d, this->nx * this->ny * this->nz);
        return;
    }
    int xy_shape[2] = {this->nx, this->ny};
    int z_shape[1] = {this->nz};
    CHECK_CUFFT(cufftPlanMany(&c_xy_handle,
                              2,
                              xy_shape,
                              xy_shape,
                              this->nplane,
                              1,
                              xy_shape,
                              this->nplane,
                              1,
                              CUFFT_C2C,
                              this->nplane));
    CHECK_CUFFT(cufftPlanMany(&c_z_handle,
                              1,
                              z_shape,
                              z_shape,
                              1,
                              this->nz,
                              z_shape,
                              1,
                              this->nz,
                              CUFFT_C2C,
                              this->ns));
    const int slab_size = this->nx * this->ny * this->nplane;
    const int stick_size = this->ns * this->nz;
    resmem_cd_op()(this->c_auxr_3d, slab_size > stick_size ? slab_size : stick_size);
}
template <>
void FFT_CUDA<double>::setupFFT()
{
    if (this->nproc == 1)
    {
        CHECK_CUFFT(cufftPlan3d(&z_handle, this->nx, this->ny, this->nz, CUFFT_Z2Z));
        resmem_zd_op()(this->z_auxr_3d, this->nx * this->ny * this->nz);
        return;
    }
    int xy_shape[2] = {this->nx, this->ny};
    int z_shape[1] = {this->nz};
    CHECK_CUFFT(cufftPlanMany(&z_xy_handle,
                              2,
                              xy_shape,
                              xy_shape,
                              this->nplane,
                              1,
                              xy_shape,
                              this->nplane,
                              1,
                              CUFFT_Z2Z,
                              this->nplane));
    CHECK_CUFFT(cufftPlanMany(&z_z_handle,
                              1,
                              z_shape,
                              z_shape,
                              1,
                              this->nz,
                              z_shape,
                              1,
                              this->nz,
                              CUFFT_Z2Z,
                              this->ns));
    const int slab_size = this->nx * this->ny * this->nplane;
    const int stick_size = this->ns * this->nz;
    resmem_zd_op()(this->z_auxr_3d, slab_size > stick_size ? slab_size : stick_size);
}
template <>
void FFT_CUDA<float>::cleanFFT()
{
    if (c_handle)
    {
        cufftDestroy(c_handle);
        c_handle = {};
    }
    if (c_xy_handle)
    {
        cufftDestroy(c_xy_handle);
        c_xy_handle = {};
    }
    if (c_z_handle)
    {
        cufftDestroy(c_z_handle);
        c_z_handle = {};
    }
}
template <>
void FFT_CUDA<double>::cleanFFT()
{
    if (z_handle)
    {
        cufftDestroy(z_handle);
        z_handle = {};
    }
    if (z_xy_handle)
    {
        cufftDestroy(z_xy_handle);
        z_xy_handle = {};
    }
    if (z_z_handle)
    {
        cufftDestroy(z_z_handle);
        z_z_handle = {};
    }
}
template <>
void FFT_CUDA<float>::clear()
{
    this->cleanFFT();
    if (c_auxr_3d != nullptr)
    {
        delmem_cd_op()(c_auxr_3d);
        c_auxr_3d = nullptr;
    }
}
template <>
void FFT_CUDA<double>::clear()
{
    this->cleanFFT();
    if (z_auxr_3d != nullptr)
    {
        delmem_zd_op()(z_auxr_3d);
        z_auxr_3d = nullptr;
    }
}

template <>
void FFT_CUDA<float>::fft3D_forward(std::complex<float>* in, std::complex<float>* out) const
{
    CHECK_CUFFT(cufftExecC2C(this->c_handle,
                             reinterpret_cast<cufftComplex*>(in),
                             reinterpret_cast<cufftComplex*>(out),
                             CUFFT_FORWARD));
}
template <>
void FFT_CUDA<double>::fft3D_forward(std::complex<double>* in, std::complex<double>* out) const
{
    CHECK_CUFFT(cufftExecZ2Z(this->z_handle,
                             reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out),
                             CUFFT_FORWARD));
}
template <>
void FFT_CUDA<float>::fft3D_backward(std::complex<float>* in, std::complex<float>* out) const
{
    CHECK_CUFFT(cufftExecC2C(this->c_handle,
                             reinterpret_cast<cufftComplex*>(in),
                             reinterpret_cast<cufftComplex*>(out),
                             CUFFT_INVERSE));
}

template <>
void FFT_CUDA<double>::fft3D_backward(std::complex<double>* in, std::complex<double>* out) const
{
    CHECK_CUFFT(cufftExecZ2Z(this->z_handle,
                             reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out),
                             CUFFT_INVERSE));
}
template <>
void FFT_CUDA<float>::fftxyfor(std::complex<float>* in, std::complex<float>* out) const
{
    CHECK_CUFFT(cufftExecC2C(this->c_xy_handle,
                             reinterpret_cast<cufftComplex*>(in),
                             reinterpret_cast<cufftComplex*>(out),
                             CUFFT_FORWARD));
}
template <>
void FFT_CUDA<double>::fftxyfor(std::complex<double>* in, std::complex<double>* out) const
{
    CHECK_CUFFT(cufftExecZ2Z(this->z_xy_handle,
                             reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out),
                             CUFFT_FORWARD));
}
template <>
void FFT_CUDA<float>::fftxybac(std::complex<float>* in, std::complex<float>* out) const
{
    CHECK_CUFFT(cufftExecC2C(this->c_xy_handle,
                             reinterpret_cast<cufftComplex*>(in),
                             reinterpret_cast<cufftComplex*>(out),
                             CUFFT_INVERSE));
}
template <>
void FFT_CUDA<double>::fftxybac(std::complex<double>* in, std::complex<double>* out) const
{
    CHECK_CUFFT(cufftExecZ2Z(this->z_xy_handle,
                             reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out),
                             CUFFT_INVERSE));
}
template <>
void FFT_CUDA<float>::fftzfor(std::complex<float>* in, std::complex<float>* out) const
{
    CHECK_CUFFT(cufftExecC2C(this->c_z_handle,
                             reinterpret_cast<cufftComplex*>(in),
                             reinterpret_cast<cufftComplex*>(out),
                             CUFFT_FORWARD));
}
template <>
void FFT_CUDA<double>::fftzfor(std::complex<double>* in, std::complex<double>* out) const
{
    CHECK_CUFFT(cufftExecZ2Z(this->z_z_handle,
                             reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out),
                             CUFFT_FORWARD));
}
template <>
void FFT_CUDA<float>::fftzbac(std::complex<float>* in, std::complex<float>* out) const
{
    CHECK_CUFFT(cufftExecC2C(this->c_z_handle,
                             reinterpret_cast<cufftComplex*>(in),
                             reinterpret_cast<cufftComplex*>(out),
                             CUFFT_INVERSE));
}
template <>
void FFT_CUDA<double>::fftzbac(std::complex<double>* in, std::complex<double>* out) const
{
    CHECK_CUFFT(cufftExecZ2Z(this->z_z_handle,
                             reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out),
                             CUFFT_INVERSE));
}
template <>
std::complex<float>* FFT_CUDA<float>::get_auxr_3d_data() const
{
    return this->c_auxr_3d;
}
template <>
std::complex<double>* FFT_CUDA<double>::get_auxr_3d_data() const
{
    return this->z_auxr_3d;
}

template FFT_CUDA<float>::FFT_CUDA();
template FFT_CUDA<float>::~FFT_CUDA();
template FFT_CUDA<double>::FFT_CUDA();
template FFT_CUDA<double>::~FFT_CUDA();
} // namespace ModuleBase
