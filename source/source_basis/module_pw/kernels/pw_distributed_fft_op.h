#ifndef PW_DISTRIBUTED_FFT_OP_H
#define PW_DISTRIBUTED_FFT_OP_H

#include <complex>

namespace ModulePW
{

void pack_fft_slabs_gpu(const int nstot,
                        const int nplane,
                        const int* istot2ixy,
                        const std::complex<double>* slabs,
                        std::complex<double>* packed);

void unpack_fft_sticks_gpu(const int nst,
                           const int nz,
                           const int* numz,
                           const int* startg,
                           const int* z_owner,
                           const int* z_local,
                           const std::complex<double>* packed,
                           std::complex<double>* sticks);

void pack_fft_sticks_gpu(const int nst,
                         const int nz,
                         const int* numz,
                         const int* startg,
                         const int* z_owner,
                         const int* z_local,
                         const std::complex<double>* sticks,
                         std::complex<double>* packed);

void unpack_fft_slabs_gpu(const int nrxx,
                          const int nstot,
                          const int nplane,
                          const int* istot2ixy,
                          const std::complex<double>* packed,
                          std::complex<double>* slabs);

} // namespace ModulePW

#endif
