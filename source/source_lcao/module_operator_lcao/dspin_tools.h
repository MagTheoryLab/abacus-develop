#ifndef DSPIN_TOOLS_H
#define DSPIN_TOOLS_H

#include <array>
#include <complex>

#include "source_base/vector3.h"

namespace hamilt
{
namespace deltaspin
{

/**
 * @brief Expand lambda dot sigma as a row-major 2x2 spinor matrix.
 */
inline std::array<std::complex<double>, 4> lambda_to_spinor_matrix(
    const ModuleBase::Vector3<double>& lambda)
{
    return {{std::complex<double>(lambda.z, 0.0),
             std::complex<double>(lambda.x, -lambda.y),
             std::complex<double>(lambda.x, lambda.y),
             std::complex<double>(-lambda.z, 0.0)}};
}

} // namespace deltaspin
} // namespace hamilt

#endif // DSPIN_TOOLS_H
