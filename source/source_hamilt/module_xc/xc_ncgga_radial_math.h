#ifndef XC_NCGGA_RADIAL_MATH_H
#define XC_NCGGA_RADIAL_MATH_H

#include <cmath>

#if defined(__CUDACC__)
#define ABACUS_NCGGA_HOST_DEVICE __host__ __device__
#else
#define ABACUS_NCGGA_HOST_DEVICE
#endif

namespace ModuleXC
{
namespace NcggaRadialMath
{

ABACUS_NCGGA_HOST_DEVICE inline double lca_radial_eta()
{
    return 1.0e-3;
}

struct RadialData
{
    double value = 0.0;
    double direction[3] = {0.0, 0.0, 0.0};
    double gradient[3] = {0.0, 0.0, 0.0};
    double transverse_hessian = 0.0;
    double radial_hessian = 0.0;
};

struct SpinMapData
{
    RadialData radial;
    double absolute_density = 0.0;
    double clipped_magnitude = 0.0;
    double spin_density[2] = {0.0, 0.0};
    double density_sign = 0.0;
    bool saturated = true;
};

ABACUS_NCGGA_HOST_DEVICE inline RadialData make_radial_data(
    const double mx,
    const double my,
    const double mz,
    const double eta)
{
    RadialData point;
    const double magnitude = sqrt(mx * mx + my * my + mz * mz);
    if (magnitude == 0.0)
    {
        return point;
    }

    point.direction[0] = mx / magnitude;
    point.direction[1] = my / magnitude;
    point.direction[2] = mz / magnitude;

    if (magnitude < eta)
    {
        const double x = magnitude / eta;
        const double x2 = x * x;
        const double x3 = x2 * x;
        point.value = eta * x3 * (3.0 * x2 - 8.0 * x + 6.0);
        point.transverse_hessian
            = x * (15.0 * x2 - 32.0 * x + 18.0) / eta;
        point.radial_hessian
            = x * (60.0 * x2 - 96.0 * x + 36.0) / eta;
    }
    else
    {
        point.value = magnitude;
        point.transverse_hessian = 1.0 / magnitude;
        point.radial_hessian = 0.0;
    }

    point.gradient[0] = point.transverse_hessian * mx;
    point.gradient[1] = point.transverse_hessian * my;
    point.gradient[2] = point.transverse_hessian * mz;
    return point;
}

ABACUS_NCGGA_HOST_DEVICE inline double radial_jacobian(
    const RadialData& point,
    const int row,
    const int column)
{
    const double identity = row == column ? 1.0 : 0.0;
    return point.transverse_hessian * identity
           + (point.radial_hessian - point.transverse_hessian)
                 * point.direction[row] * point.direction[column];
}

ABACUS_NCGGA_HOST_DEVICE inline SpinMapData make_spin_map_data(
    const double total_density,
    const RadialData& radial)
{
    SpinMapData point;
    point.radial = radial;
    point.absolute_density = fabs(total_density);
    point.clipped_magnitude
        = radial.value < point.absolute_density
              ? radial.value
              : point.absolute_density;
    point.spin_density[0]
        = 0.5 * (point.absolute_density + point.clipped_magnitude);
    point.spin_density[1]
        = 0.5 * (point.absolute_density - point.clipped_magnitude);
    point.density_sign = total_density > 0.0 ? 1.0
                         : total_density < 0.0 ? -1.0
                                               : 0.0;
    point.saturated = !(radial.value < point.absolute_density);
    return point;
}

ABACUS_NCGGA_HOST_DEVICE inline double spin_map_jacobian(
    const SpinMapData& point,
    const int spin,
    const int channel)
{
    if (channel == 0)
    {
        if (point.saturated)
        {
            return spin == 0 ? point.density_sign : 0.0;
        }
        return 0.5 * point.density_sign;
    }
    if (point.saturated)
    {
        return 0.0;
    }
    const double spin_sign = spin == 0 ? 0.5 : -0.5;
    return spin_sign * point.radial.gradient[channel - 1];
}

} // namespace NcggaRadialMath
} // namespace ModuleXC

#undef ABACUS_NCGGA_HOST_DEVICE

#endif
