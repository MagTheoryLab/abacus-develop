#ifndef FFT_CUDA_H
#define FFT_CUDA_H

#include "fft_base.h"
#include "cufft.h"
#include "cuda_runtime.h"
namespace ModuleBase
{
template <typename FPTYPE>
class FFT_CUDA : public FFT_BASE<FPTYPE>
{
    public:
        FFT_CUDA(){};
        ~FFT_CUDA(){};

        void setupFFT() override;

        void clear() override;

        void cleanFFT() override;

        /**
        * @brief Initialize the fft parameters
        * @param nx_in  number of grid points in x direction
        * @param ny_in  number of grid points in y direction
        * @param nz_in  number of grid points in z direction
        *
        */
        void initfft(int nx_in,
                     int ny_in,
                     int nz_in) override;
        void initfft(int nx_in,
                     int ny_in,
                     int nz_in,
                     int lixy_in,
                     int rixy_in,
                     int ns_in,
                     int nplane_in,
                     int nproc_in,
                     bool gamma_only_in,
                     bool xprime_in) override;

        /**
         * @brief Get the real space data
         * @return real space data
         */
        std::complex<FPTYPE>* get_auxr_3d_data() const override;

        /**
         * @brief Forward FFT in 3D
         * @param in  input data, complex FPTYPE
         * @param out  output data, complex FPTYPE
         *
         * This function performs the forward FFT in 3D.
         */
        void fft3D_forward(std::complex<FPTYPE>* in,
                           std::complex<FPTYPE>* out) const override;
        /**
         * @brief Backward FFT in 3D
         * @param in  input data, complex FPTYPE
         * @param out  output data, complex FPTYPE
         *
         * This function performs the backward FFT in 3D.
         */
        void fft3D_backward(std::complex<FPTYPE>* in,
                            std::complex<FPTYPE>* out) const override;
        void fftxyfor(std::complex<FPTYPE>* in,
                      std::complex<FPTYPE>* out) const override;
        void fftxybac(std::complex<FPTYPE>* in,
                      std::complex<FPTYPE>* out) const override;
        void fftzfor(std::complex<FPTYPE>* in,
                     std::complex<FPTYPE>* out) const override;
        void fftzbac(std::complex<FPTYPE>* in,
                     std::complex<FPTYPE>* out) const override;
    private:
        cufftHandle c_handle = {};
        cufftHandle z_handle = {};
        cufftHandle c_xy_handle = {};
        cufftHandle c_z_handle = {};
        cufftHandle z_xy_handle = {};
        cufftHandle z_z_handle = {};

        std::complex<float>* c_auxr_3d = nullptr;  // fft space
        std::complex<double>* z_auxr_3d = nullptr; // fft space
        int nplane = 0;
        int ns = 0;
        int nproc = 1;

};

} // namespace ModuleBase
#endif
