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
bool ElecStateLCAO<TK>::cal_dmr_from_k_owner(const ModuleBase::matrix&, DensityMatrix<TK, double>&) const
{
    return false;
}

template <typename TK>
bool ElecStateLCAO<TK>::cal_dmr_from_k_owner(
    const ModuleBase::matrix&, const DensityMatrix<TK, double>&,
    hamilt::HContainer<std::complex<double>>&) const
{
    return false;
}

template <typename TK>
std::vector<const psi::Psi<TK, base_device::DEVICE_GPU>*> ElecStateLCAO<TK>::k_owner_wfc_view() const
{
    std::vector<const psi::Psi<TK, base_device::DEVICE_GPU>*> view(owner_wfc_.size(), nullptr);
    for (std::size_t ik = 0; ik < owner_wfc_.size(); ++ik)
    {
        view[ik] = owner_wfc_[ik].get();
    }
    return view;
}

template <>
bool ElecStateLCAO<std::complex<double>>::cal_dmr_from_k_owner(
    const ModuleBase::matrix& weights, DensityMatrix<std::complex<double>, double>& dm) const
{
    if (owner_wfc_.empty())
    {
        return false;
    }
    cal_dmr_psi_gpu_k_owner(dm.get_paraV_pointer(), weights, k_owner_wfc_view(), dm);
    return true;
}

template <>
bool ElecStateLCAO<std::complex<double>>::cal_dmr_from_k_owner(
    const ModuleBase::matrix& weights, const DensityMatrix<std::complex<double>, double>& dm,
    hamilt::HContainer<std::complex<double>>& full_dmr) const
{
    if (owner_wfc_.empty())
    {
        return false;
    }
    cal_dmr_psi_gpu_k_owner(dm.get_paraV_pointer(), weights, k_owner_wfc_view(), dm.get_kvec_d(), full_dmr);
    return true;
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
