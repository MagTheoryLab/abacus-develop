#ifndef ELECSTATE_LCAO_H
#define ELECSTATE_LCAO_H

#include "elecstate.h"
#include "source_estate/module_dm/density_matrix.h"

#include <vector>
#include <memory>

namespace elecstate
{
template <typename TK>
class ElecStateLCAO : public ElecState
{
  public:
    ElecStateLCAO()
    {
    } // will be called by ElecStateLCAO_TDDFT
    ElecStateLCAO(Charge* chr_in,
                  const K_Vectors* klist_in,
                  int nks_in,
                  ModulePW::PW_Basis_Big* bigpw_in)
    {
        init_ks(chr_in, klist_in, nks_in, bigpw_in);
        this->classname = "ElecStateLCAO";
    }

    virtual ~ElecStateLCAO() = default;

    static int out_wfc_lcao;
    static bool need_psi_grid;

    double get_spin_constrain_energy() override;

#if defined(__CUDA) && defined(__MPI)
    // The latest solve owns these eigenvectors until the next solve or final
    // force/stress preparation. Non-owner ranks keep null entries, not copies.
    using OwnerWavefunctions = std::vector<std::unique_ptr<psi::Psi<TK, base_device::DEVICE_GPU>>>;
    void retain_k_owner_wfc(OwnerWavefunctions&& wfc);
    void clear_k_owner_wfc();
    void materialize_k_owner_state(psi::Psi<TK>& wfc, DensityMatrix<TK, double>& dm);
#endif

    // use for pexsi

    /**
     * @brief calculate electronic charge density from pointers of density matrix calculated by pexsi
     * @param pexsi_DM: pointers of density matrix (DMK) calculated by pexsi
     * @param pexsi_EDM: pointers of energy-weighed density matrix (EDMK) calculated by pexsi, needed by MD, will be
     * stored in DensityMatrix::pexsi_EDM
     */
	void dm2rho(std::vector<TK*> pexsi_DM,
			std::vector<TK*> pexsi_EDM,
			DensityMatrix<TK, double>* dm);

    /**
     * @brief calculate electronic charge density from the density matrix (DMR)
     *
     * Thin wrapper over LCAO_domain::dm2rho so that HSolverLCAO delegates the
     * charge-density calculation through the ElecState interface, mirroring the
     * plane-wave path (ElecStatePW::psiToRho) and the pexsi branch above. This
     * keeps the source_lcao dependency out of source_hsolver.
     */
    void dmToRho(std::vector<hamilt::HContainer<double>*>& dmr,
                 int nspin,
                 Charge* chr,
                 bool skip_charge = false);

  private:
#if defined(__CUDA) && defined(__MPI)
    OwnerWavefunctions owner_wfc_;
#endif
};

template <typename TK>
int ElecStateLCAO<TK>::out_wfc_lcao = 0;

template <typename TK>
bool ElecStateLCAO<TK>::need_psi_grid = true;

} // namespace elecstate

#endif
