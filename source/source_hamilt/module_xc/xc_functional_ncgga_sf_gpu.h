#ifndef XC_FUNCTIONAL_NCGGA_SF_GPU_H
#define XC_FUNCTIONAL_NCGGA_SF_GPU_H

#include "source_base/matrix.h"

#include <tuple>
#include <vector>

class Charge;

namespace ModuleXC
{
namespace NCGGA_SF_Builtin
{

// Executes the complete built-in noncollinear gga_grad=2 PBE-family graph on
// one GPU. Returns false without modifying result when the functional, FFT
// layout, or process topology is not supported by this narrow fast path.
bool try_v_xc_ncgga_sf_builtin_gpu(
    int nrxx,
    double omega,
    double tpiba,
    const Charge* chr,
    const std::vector<int>& functional_ids,
    std::tuple<double, double, ModuleBase::matrix>& result);

} // namespace NCGGA_SF_Builtin
} // namespace ModuleXC

#endif
