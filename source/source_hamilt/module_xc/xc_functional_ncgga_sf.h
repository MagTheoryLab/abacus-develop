#ifndef XC_FUNCTIONAL_NCGGA_SF_H
#define XC_FUNCTIONAL_NCGGA_SF_H

#include "source_base/matrix.h"

#include <tuple>
#include <vector>

class Charge;
namespace ModulePW
{
class PW_Basis;
}
struct UnitCell;

namespace ModuleXC
{
namespace NCGGA_SF_Builtin
{

// gga_grad: 2 = exact discrete reverse of the regularized projected LCA graph,
//           3 = continuous B2 response
std::tuple<double, double, ModuleBase::matrix> v_xc_ncgga_sf_builtin(
    const int& nrxx, const double& omega, const double tpiba, const Charge* const chr,
    const int gga_grad);

// Gradient-metric stress of the exact gga_grad=2 projected-LCA graph.  The
// returned lower triangle is the unnormalised real-grid sum; Stress_Func
// applies the existing pool reduction and 1/nxyz normalisation.
void gradcorr_ncgga_lca_builtin(const Charge* const chr,
                                ModulePW::PW_Basis* rhopw,
                                const double tpiba,
                                std::vector<double>& stress_gga);

// Continuous B2 stress used by the gga_grad=3 stress dispatch.
void gradcorr_ncgga_sf_builtin(const Charge* const chr, ModulePW::PW_Basis* rhopw,
                                const UnitCell* ucell, std::vector<double>& stress_gga);

} // namespace NCGGA_SF_Builtin
} // namespace ModuleXC

#endif
