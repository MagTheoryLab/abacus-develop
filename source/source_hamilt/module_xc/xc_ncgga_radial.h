#ifndef XC_NCGGA_RADIAL_H
#define XC_NCGGA_RADIAL_H

#include <array>

namespace ModuleXC
{

struct NcggaRadialPoint
{
    // Value, gradient, and Hessian of the same radial scalar map.  At zero,
    // direction is represented by the zero vector because the scalar map has
    // a unique zero gradient and zero Hessian there.
    double value = 0.0;
    std::array<double, 3> direction = {{0.0, 0.0, 0.0}};
    std::array<double, 3> gradient = {{0.0, 0.0, 0.0}};
    double transverse_hessian = 0.0;
    double radial_hessian = 0.0;

    double jacobian(const int row, const int column) const;
};

// For r = |magnetization| and x = r / eta, the returned scalar is
//   eta * x^3 * (3 x^2 - 8 x + 6), r < eta,
//   r,                                  r >= eta.
// The splice is C2 at both r = 0 and r = eta.  Eta is explicit so this
// mathematical primitive does not choose policy for any XC mode.
NcggaRadialPoint make_ncgga_radial_point(
    const std::array<double, 3>& magnetization,
    const double eta);

} // namespace ModuleXC

#endif
