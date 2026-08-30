#include "xc_ncgga_radial.h"

#include <cmath>
#include <stdexcept>

namespace ModuleXC
{

double NcggaRadialPoint::jacobian(const int row, const int column) const
{
    const double identity = (row == column) ? 1.0 : 0.0;
    return transverse_hessian * identity
           + (radial_hessian - transverse_hessian)
                 * direction[row] * direction[column];
}

NcggaRadialPoint make_ncgga_radial_point(
    const std::array<double, 3>& magnetization,
    const double eta)
{
    if (!(eta > 0.0))
    {
        throw std::invalid_argument("noncollinear GGA radial eta must be positive");
    }

    NcggaRadialPoint point;
    const double magnitude
        = std::sqrt(magnetization[0] * magnetization[0]
                    + magnetization[1] * magnetization[1]
                    + magnetization[2] * magnetization[2]);
    if (magnitude == 0.0)
    {
        return point;
    }

    for (int component = 0; component < 3; ++component)
    {
        point.direction[component] = magnetization[component] / magnitude;
    }

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

    for (int component = 0; component < 3; ++component)
    {
        point.gradient[component]
            = point.transverse_hessian * magnetization[component];
    }
    return point;
}

} // namespace ModuleXC
