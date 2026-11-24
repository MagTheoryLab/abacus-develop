#include "deltaspin_lcao.h"
#include "spin_constrain.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_lcao/hamilt_lcao.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_estate/elecstate.h"

/**
 * @file deltaspin_lcao.cpp
 * @brief Wrapper/facade layer between ESolver and DeltaSpin module.
 *
 * @par Purpose
 * Provides a simplified interface to the ESolver layer, hiding the
 * SpinConstrain Singleton details. The ESolver calls these functions
 * rather than accessing SpinConstrain directly.
 *
 * @par Design rationale
 * - Template functions: Support both TK=double (nspin=2) and TK=complex<double> (nspin=4)
 * - Early returns: If sc_mag_switch is false, all functions return immediately
 *   without any overhead
 * - #ifdef __LCAO: The density matrix pointer is only available in LCAO builds
 *
 * @par Workflow
 * 1. ESolver calls init_deltaspin_lcao() at start of calculation
 * 2. ESolver calls cal_mi_lcao_wrapper() to compute magnetic moments after updating the density matrix
 */

namespace ModuleESolver
{

/**
 * @brief Initialize the SpinConstrain singleton with all input parameters.
 *
 * @details Called once at the start of a DeltaSpin calculation. Checks
 * sc_mag_switch first; if disabled, returns immediately without any action.
 *
 * @par Conditional compilation
 * The density matrix pointer (dm) is only available when __LCAO is defined.
 * For non-LCAO builds (PW-only), init_sc() is called without the dm parameter.
 */
template <typename TK>
void init_deltaspin_lcao(const UnitCell& ucell,
                          const Input_para& inp,
                          void* pv,
                          const K_Vectors& kv,
                          void* p_hamilt,
                          void* psi,
                          void* dm,
                          void* pelec)
{
    // Early exit if DeltaSpin is not enabled
    if (!inp.sc_mag_switch)
    {
        return;
    }

    spinconstrain::SpinConstrain<TK>& sc = spinconstrain::SpinConstrain<TK>::getScInstance();
#ifdef __LCAO
    // LCAO build: pass density matrix pointer
    sc.init_sc(inp.sc_thr, inp.nsc, inp.nsc_min, inp.alpha_trial,
               inp.sccut, inp.sc_drop_thr, ucell, inp.sc_direction_only,
               static_cast<Parallel_Orbitals*>(pv),
               inp.nspin, kv, p_hamilt, psi,
               static_cast<module_dm::DensityMatrix<TK, double>*>(dm),
               static_cast<elecstate::ElecState*>(pelec));
#else
    // Non-LCAO build: no density matrix
    sc.init_sc(inp.sc_thr, inp.nsc, inp.nsc_min, inp.alpha_trial,
               inp.sccut, inp.sc_drop_thr, ucell, inp.sc_direction_only,
               static_cast<Parallel_Orbitals*>(pv),
               inp.nspin, kv, p_hamilt, psi,
               static_cast<elecstate::ElecState*>(pelec));
#endif
}

/**
 * @brief Wrapper: calculate magnetic moments for current SCF iteration.
 *
 * @details If DeltaSpin is enabled, calls SpinConstrain::cal_mi_lcao().
 * The moments are stored in Mi_ and can be retrieved via get_target_mag().
 */
template <typename TK>
void cal_mi_lcao_wrapper(const int iter, const Input_para& inp)
{
    if (!inp.sc_mag_switch)
    {
        return;
    }

#ifdef __LCAO
    spinconstrain::SpinConstrain<TK>& sc = spinconstrain::SpinConstrain<TK>::getScInstance();
    sc.cal_mi_lcao(iter);
#endif
}

/// Template instantiations for both spin types
template void init_deltaspin_lcao<double>(const UnitCell& ucell,
                                           const Input_para& inp,
                                           void* pv,
                                           const K_Vectors& kv,
                                           void* p_hamilt,
                                           void* psi,
                                           void* dm,
                                           void* pelec);
template void init_deltaspin_lcao<std::complex<double>>(const UnitCell& ucell,
                                                          const Input_para& inp,
                                                          void* pv,
                                                          const K_Vectors& kv,
                                                          void* p_hamilt,
                                                          void* psi,
                                                          void* dm,
                                                          void* pelec);

template void cal_mi_lcao_wrapper<double>(const int iter, const Input_para& inp);
template void cal_mi_lcao_wrapper<std::complex<double>>(const int iter, const Input_para& inp);

} // namespace ModuleESolver
