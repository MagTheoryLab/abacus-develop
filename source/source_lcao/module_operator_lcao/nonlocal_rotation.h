#ifndef NONLOCAL_ROTATION_H
#define NONLOCAL_ROTATION_H

namespace hamilt
{
// In the rotating orbital frame, joint rotation covariance gives dH_NL/dtheta_a = -i[S_a, H_NL].
// The spinor DMR already uses elementwise trace order.
template <typename Complex>
#ifdef __CUDACC__
__host__ __device__
#endif
inline void add_soc_rotation(const Complex* h,
                             const Complex& dm_uu,
                             const Complex& dm_ud,
                             const Complex& dm_du,
                             const Complex& dm_dd,
                             double* rotation)
{
    const Complex diagonal = h[0] - h[3];
    const Complex off_diagonal = h[2] - h[1];
    rotation[0] += 0.5 * (off_diagonal * (dm_uu - dm_dd) + diagonal * (dm_du - dm_ud)).imag();
    rotation[1] += 0.5 * ((h[1] + h[2]) * (dm_dd - dm_uu) + diagonal * (dm_ud + dm_du)).real();
    rotation[2] += (h[1] * dm_ud - h[2] * dm_du).imag();
}
}
#endif
