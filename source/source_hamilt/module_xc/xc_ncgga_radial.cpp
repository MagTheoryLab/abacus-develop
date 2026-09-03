#include "xc_ncgga_radial.h"
#include "xc_ncgga_radial_math.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ModuleXC
{

double NcggaRadialPoint::jacobian(const int row, const int column) const
{
    NcggaRadialMath::RadialData data;
    data.transverse_hessian = transverse_hessian;
    data.radial_hessian = radial_hessian;
    for (int component = 0; component < 3; ++component)
    {
        data.direction[component] = direction[component];
    }
    return NcggaRadialMath::radial_jacobian(data, row, column);
}

NcggaRadialPoint make_ncgga_radial_point(
    const std::array<double, 3>& magnetization,
    const double eta)
{
    if (!(eta > 0.0))
    {
        throw std::invalid_argument("noncollinear GGA radial eta must be positive");
    }

    const NcggaRadialMath::RadialData data
        = NcggaRadialMath::make_radial_data(magnetization[0],
                                             magnetization[1],
                                             magnetization[2],
                                             eta);
    NcggaRadialPoint point;
    point.value = data.value;
    point.transverse_hessian = data.transverse_hessian;
    point.radial_hessian = data.radial_hessian;
    for (int component = 0; component < 3; ++component)
    {
        point.direction[component] = data.direction[component];
        point.gradient[component] = data.gradient[component];
    }
    return point;
}

double ncgga_lca_radial_eta()
{
    return NcggaRadialMath::lca_radial_eta();
}

double NcggaSpinMapPoint::jacobian(const int spin, const int channel) const
{
    NcggaRadialMath::SpinMapData data;
    data.absolute_density = absolute_density;
    data.clipped_magnitude = clipped_magnitude;
    data.density_sign = density_sign;
    data.saturated = saturated;
    data.radial.transverse_hessian = radial.transverse_hessian;
    data.radial.radial_hessian = radial.radial_hessian;
    for (int component = 0; component < 3; ++component)
    {
        data.radial.direction[component] = radial.direction[component];
        data.radial.gradient[component] = radial.gradient[component];
    }
    return NcggaRadialMath::spin_map_jacobian(data, spin, channel);
}

NcggaSpinMapPoint make_ncgga_spin_map_point(
    const double total_density,
    const NcggaRadialPoint& radial)
{
    NcggaRadialMath::RadialData radial_data;
    radial_data.value = radial.value;
    radial_data.transverse_hessian = radial.transverse_hessian;
    radial_data.radial_hessian = radial.radial_hessian;
    for (int component = 0; component < 3; ++component)
    {
        radial_data.direction[component] = radial.direction[component];
        radial_data.gradient[component] = radial.gradient[component];
    }
    const NcggaRadialMath::SpinMapData data
        = NcggaRadialMath::make_spin_map_data(total_density, radial_data);
    NcggaSpinMapPoint point;
    point.radial = radial;
    point.absolute_density = data.absolute_density;
    point.clipped_magnitude = data.clipped_magnitude;
    point.spin_density[0] = data.spin_density[0];
    point.spin_density[1] = data.spin_density[1];
    point.density_sign = data.density_sign;
    point.saturated = data.saturated;
    return point;
}

} // namespace ModuleXC
