#include "source_estate/elecstate_lcao.h"
#include "source_estate/cal_dm.h"
#include "source_base/timer.h"
#include "source_estate/module_dm/cal_dm_psi.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_lcao/module_deltaspin/spin_constrain.h"
#include "source_io/module_parameter/parameter.h"

#include "source_hamilt/module_gint/gint_interface.h"
#include "source_lcao/rho_tau_lcao.h"

#include <vector>

#if defined(__CUDA) && defined(__MPI)
#include "source_base/module_external/scalapack_connector.h"
#include "source_base/parallel_2d.h"
#endif

namespace elecstate
{

#if defined(__CUDA) && defined(__MPI)
template <typename TK>
void ElecStateLCAO<TK>::retain_k_owner_wfc(OwnerWavefunctions&& wfc)
{
    owner_wfc_ = std::move(wfc);
}

template <typename TK>
void ElecStateLCAO<TK>::clear_k_owner_wfc()
{
    owner_wfc_.clear();
}

template <typename TK>
void ElecStateLCAO<TK>::materialize_k_owner_state(psi::Psi<TK>& wfc, DensityMatrix<TK, double>& dm)
{
    if (owner_wfc_.empty())
    {
        return;
    }
    ModuleBase::timer::start("ElecStateLCAO", "materialize_owner");
    const auto* pv = dm.get_paraV_pointer();
    const int nrow = pv->get_global_row_size();
    const int nbands = pv->get_nbands();
    int desc_wfc[9];
    std::copy(pv->desc_wfc, pv->desc_wfc + 9, desc_wfc);
    Parallel_2D owner_layout;
    owner_layout.init(nrow, nbands, pv->get_block_size(), MPI_COMM_SELF);
    for (int ik = 0; ik < static_cast<int>(owner_wfc_.size()); ++ik)
    {
        int desc_owner[9];
        std::copy(owner_layout.desc, owner_layout.desc + 9, desc_owner);
        std::unique_ptr<psi::Psi<TK>> host_wfc;
        if (owner_wfc_[ik])
        {
            host_wfc.reset(new psi::Psi<TK>(*owner_wfc_[ik]));
        }
        else
        {
            desc_owner[1] = -1;
        }
        wfc.fix_k(ik);
        Cpxgemr2d(nrow, nbands, host_wfc ? host_wfc->get_pointer() : nullptr,
                  1, 1, desc_owner, wfc.get_pointer(), 1, 1, desc_wfc, pv->blacs_ctxt);
    }
    // Preserve the ordinary DMR already used by SCF. Only materialize its
    // legacy dense representation for consumers that still require it.
    cal_dm_psi(pv, this->wg, wfc, dm);
    owner_wfc_.clear();
    ModuleBase::timer::end("ElecStateLCAO", "materialize_owner");
}
#endif


template <>
double ElecStateLCAO<double>::get_spin_constrain_energy()
{
    spinconstrain::SpinConstrain<double>& sc = spinconstrain::SpinConstrain<double>::getScInstance();
    return sc.cal_escon();
}

template <>
double ElecStateLCAO<std::complex<double>>::get_spin_constrain_energy()
{
    spinconstrain::SpinConstrain<std::complex<double>>& sc
        = spinconstrain::SpinConstrain<std::complex<double>>::getScInstance();
    return sc.cal_escon();
}

template <>
void ElecStateLCAO<double>::dm2rho(std::vector<double*> pexsi_DM, 
		std::vector<double*> pexsi_EDM,
		DensityMatrix<double, double>* dm)
{
    ModuleBase::timer::start("ElecStateLCAO", "dm2rho");

    int nspin = PARAM.inp.nspin;
    if (PARAM.inp.nspin == 4)
    {
        nspin = 1;
    }

#ifdef __PEXSI
    dm->pexsi_EDM = pexsi_EDM;
#endif

    for (int is = 0; is < nspin; is++)
    {
        dm->set_DMK_pointer(is, pexsi_DM[is]);
    }
    dm->cal_DMR();

    for (int is = 0; is < PARAM.inp.nspin; is++)
    {
        ModuleBase::GlobalFunc::ZEROS(this->charge->rho[is],
                                      this->charge->nrxx); // mohan 2009-11-10
    }

    ModuleBase::GlobalFunc::NOTE("Calculate the charge on real space grid!");
    ModuleGint::cal_gint_rho(dm->get_DMR_vector(), PARAM.inp.nspin, this->charge->rho);
    if (XC_Functional::get_ked_flag())
    {
        for (int is = 0; is < PARAM.inp.nspin; is++)
        {
            ModuleBase::GlobalFunc::ZEROS(this->charge->kin_r[0], this->charge->nrxx);
        }
        ModuleGint::cal_gint_tau(dm->get_DMR_vector(), PARAM.inp.nspin, this->charge->kin_r);
    }

    this->charge->renormalize_rho();

    ModuleBase::timer::end("ElecStateLCAO", "dm2rho");
    return;
}

template <>
void ElecStateLCAO<std::complex<double>>::dm2rho(std::vector<std::complex<double>*> pexsi_DM,
		std::vector<std::complex<double>*> pexsi_EDM,
		DensityMatrix<std::complex<double>, double>* dm)
{
    ModuleBase::WARNING_QUIT("ElecStateLCAO", "pexsi is not completed for multi-k case");
}


template <typename TK>
void ElecStateLCAO<TK>::dmToRho(std::vector<hamilt::HContainer<double>*>& dmr,
                                int nspin,
                                Charge* chr,
                                bool skip_charge)
{
    LCAO_domain::dm2rho(dmr, nspin, chr, skip_charge);
}

template class ElecStateLCAO<double>;               // Gamma_only case
template class ElecStateLCAO<std::complex<double>>; // multi-k case

} // namespace elecstate
