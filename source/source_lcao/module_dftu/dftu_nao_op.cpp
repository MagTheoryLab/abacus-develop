#include "dftu_nao_op.h"

#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_basis/module_nao/two_center_integrator.h"
#include "source_cell/module_neighbor/sltk_grid_driver.h"
#include "source_cell/unitcell.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_lcao/module_operator_lcao/operator_lcao.h"
#include "source_pw/module_pwdft/dftu_base.h"
#include "source_base/parallel_reduce.h"
#include "source_cell/klist.h"
#include "source_cell/module_symmetry/symmetry.h"

#include <memory>
#ifdef __CUDA
#include "dftu_lcao_gpu.h"
#endif

#include "dftu_nao_adj.h"
#include "dftu_nao_fs_r.h"
#include "dftu_nao_ijr.h"
#include "dftu_nao_pots.h"

template <typename TK, typename TR>
hamilt::DFTU_onsite<hamilt::OperatorLCAO<TK, TR>>::DFTU_onsite(HS_Matrix_K<TK>* hsk_in,
                                                 const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
                                                 hamilt::HContainer<TR>* hR_in,
                                                 const UnitCell& ucell_in,
                                                 const Grid_Driver* GridD_in,
                                                 const TwoCenterIntegrator* intor,
                                                 const std::vector<double>& orb_cutoff,
                                                 Plus_U_Base* p_dftu,
                                                 const int nspin_in,
                                                 const double onsite_radius,
                                                 const module_dm::DensityMatrix<TK, double>* dm_in,
                                                 const bool use_gpu)
    : hamilt::OperatorLCAO<TK, TR>(hsk_in, kvec_d_in, hR_in),
      ucell(&ucell_in),
      dftu(p_dftu),
      dm_(dm_in),
      intor_(intor),
      orb_cutoff_(orb_cutoff),
      nspin(nspin_in),
      use_gpu_(use_gpu)
{
    ModuleBase::timer::start("DFTU_onsite", "DFTU_onsite");
    this->cal_type = calculation_type::lcao_dftu;

    assert(this->ucell != nullptr);
    assert(this->dm_ != nullptr);

    // structure snapshot: both members depend only on the atomic structure
    // and are computed once here; the operator is rebuilt every ionic step.
    // Kept in the constructor body (not the initializer list) so failures
    // inside these calls are easy to debug.
    this->adjs_all = DFTU_LCAO::build_adjacent_atoms(this->ucell, this->dftu, GridD_in, this->orb_cutoff_, onsite_radius);
    const Parallel_Orbitals* pv = this->hR->get_atom_pair(0).get_paraV();
    this->nlm_tot = DFTU_LCAO::cal_nlm_all(*this->ucell, *this->dftu, *this->intor_, this->adjs_all, *pv);

    ModuleBase::timer::end("DFTU_onsite", "DFTU_onsite");
}

template <typename TK, typename TR>
bool hamilt::DFTU_onsite<hamilt::OperatorLCAO<TK, TR>>::contribute_hr_gpu(const Parallel_Orbitals*, bool)
{
    return false;
}

#ifdef __CUDA
template <>
bool hamilt::DFTU_onsite<hamilt::OperatorLCAO<std::complex<double>, std::complex<double>>>::contribute_hr_gpu(
    const Parallel_Orbitals* pv,
    bool keep_device)
{
    if (!this->supportsDeviceHR())
    {
        return false;
    }

    const hamilt::HContainer<double>* dmr = this->dm_->get_dmr_ptr(1);
    if (dmr == nullptr)
    {
        return false;
    }

    std::vector<std::size_t> onsite_offsets(this->ucell->nat + 1, 0);
    for (int atom = 0; atom < this->ucell->nat; ++atom)
    {
        int type = 0;
        int type_atom = 0;
        this->ucell->iat2iait(atom, &type_atom, &type);
        onsite_offsets[atom + 1] = onsite_offsets[atom];
        if (this->dftu->has_l_channel(type))
        {
            const int projector_size = 2 * this->dftu->get_l_channel(type) + 1;
            onsite_offsets[atom + 1] += 4 * projector_size * projector_size;
        }
    }
    if (onsite_offsets.back() == 0)
    {
        return false;
    }

    if (this->gpu_cache_ == nullptr)
    {
        // Kernels consume local spinor pairs, not global dense matrices.
        // All ranks must choose the same path before occupation collectives.
        int unsupported_layout = 0;
        for (int atom = 0; atom < this->ucell->nat; ++atom)
        {
            for (const auto& indexes : {pv->get_indexes_row(atom), pv->get_indexes_col(atom)})
            {
                if (indexes.size() % 2 != 0)
                {
                    unsupported_layout = 1;
                    continue;
                }
                for (std::size_t i = 0; i < indexes.size(); i += 2)
                {
                    if (indexes[i] % 2 != 0 || indexes[i + 1] != indexes[i] + 1)
                    {
                        unsupported_layout = 1;
                    }
                }
            }
        }
        Parallel_Reduce::reduce_all(unsupported_layout);
        if (unsupported_layout != 0)
        {
            return false;
        }

        struct AdjacentProjection
        {
            std::size_t row_offset;
            std::size_t col_offset;
            int row_orbitals;
            int col_orbitals;
        };

        std::vector<double> projections;
        std::vector<hamilt::dftu_gpu::ProjectionTask> tasks;
        int correlated_index = 0;
        for (int atom = 0; atom < this->ucell->nat; ++atom)
        {
            int type = 0;
            int type_atom = 0;
            this->ucell->iat2iait(atom, &type_atom, &type);
            if (!this->dftu->has_l_channel(type))
            {
                continue;
            }
            const int projector_size = 2 * this->dftu->get_l_channel(type) + 1;
            AdjacentAtomInfo& adjs = this->adjs_all[correlated_index++];
            std::vector<AdjacentProjection> adjacent(adjs.adj_num + 1);
            for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
            {
                const int adjacent_atom = this->ucell->itia2iat(adjs.ntype[ad], adjs.natom[ad]);
                const auto row_indexes = pv->get_indexes_row(adjacent_atom);
                const auto col_indexes = pv->get_indexes_col(adjacent_atom);
                AdjacentProjection descriptor;
                descriptor.row_offset = projections.size();
                descriptor.row_orbitals = row_indexes.size() / 2;
                for (int index = 0; index < static_cast<int>(row_indexes.size()); index += 2)
                {
                    const auto& values = this->nlm_tot[atom][ad].at(row_indexes[index]);
                    projections.insert(projections.end(), values.begin(), values.end());
                }
                descriptor.col_offset = projections.size();
                descriptor.col_orbitals = col_indexes.size() / 2;
                for (int index = 0; index < static_cast<int>(col_indexes.size()); index += 2)
                {
                    const auto& values = this->nlm_tot[atom][ad].at(col_indexes[index]);
                    projections.insert(projections.end(), values.begin(), values.end());
                }
                adjacent[ad] = descriptor;
            }

            for (int ad1 = 0; ad1 < adjs.adj_num + 1; ++ad1)
            {
                const int atom1 = this->ucell->itia2iat(adjs.ntype[ad1], adjs.natom[ad1]);
                const ModuleBase::Vector3<int>& R1 = adjs.box[ad1];
                for (int ad2 = 0; ad2 < adjs.adj_num + 1; ++ad2)
                {
                    const int atom2 = this->ucell->itia2iat(adjs.ntype[ad2], adjs.natom[ad2]);
                    const ModuleBase::Vector3<int>& R2 = adjs.box[ad2];
                    const ModuleBase::Vector3<int> R = R2 - R1;
                    const hamilt::BaseMatrix<double>* dm_matrix
                        = (dmr == nullptr) ? nullptr : dmr->find_matrix(atom1, atom2, R.x, R.y, R.z);
                    hamilt::BaseMatrix<std::complex<double>>* hr_matrix
                        = this->hR->find_matrix(atom1, atom2, R.x, R.y, R.z);
                    if (dm_matrix == nullptr && hr_matrix == nullptr)
                    {
                        continue;
                    }
                    hamilt::dftu_gpu::ProjectionTask task;
                    task.left_offset = adjacent[ad1].row_offset;
                    task.right_offset = adjacent[ad2].col_offset;
                    task.dm_offset = (dm_matrix == nullptr)
                        ? 0 : dm_matrix->get_pointer() - dmr->get_wrapper();
                    task.hr_offset = (hr_matrix == nullptr)
                        ? 0 : hr_matrix->get_pointer() - this->hR->get_wrapper();
                    task.onsite_offset = onsite_offsets[atom];
                    task.row_orbitals = adjacent[ad1].row_orbitals;
                    task.col_orbitals = adjacent[ad2].col_orbitals;
                    task.matrix_cols = pv->get_ncol_atom(atom2);
                    task.projector_size = projector_size;
                    task.has_dm = dm_matrix != nullptr;
                    task.has_hr = hr_matrix != nullptr;
                    tasks.push_back(task);
                }
            }
        }
        // Empty local work still participates in the occupation reduction.
        const std::size_t dm_size = dmr->get_nnr();
        this->gpu_cache_ = hamilt::dftu_gpu::create_cache(projections.data(),
                                                          projections.size(),
                                                          tasks.data(),
                                                          tasks.size(),
                                                          dm_size,
                                                          this->hR->get_nnr(),
                                                          onsite_offsets.back());
    }

    std::vector<double> occupations(onsite_offsets.back(), 0.0);
    if (!this->dftu->is_occmat_ready())
    {
        hamilt::dftu_gpu::compute_occupations(this->gpu_cache_,
                                               dmr->get_wrapper(),
                                               occupations.data());
        // Match the CPU path's global occupation sum, packed across centers.
        // Neither the distributed DMR nor H(R) is gathered here.
        Parallel_Reduce::reduce_all(occupations.data(), static_cast<int>(occupations.size()));
    }

    std::vector<std::complex<double>> onsite(onsite_offsets.back(), 0.0);
    for (int atom = 0; atom < this->ucell->nat; ++atom)
    {
        int type = 0;
        int type_atom = 0;
        this->ucell->iat2iait(atom, &type_atom, &type);
        if (!this->dftu->has_l_channel(type))
        {
            continue;
        }
        const int target_L = this->dftu->get_l_channel(type);
        const int projector_size = 2 * target_L + 1;
        const std::size_t offset = onsite_offsets[atom];
        std::vector<double> atom_occ(4 * projector_size * projector_size, 0.0);
        if (this->dftu->is_occmat_ready())
        {
            this->dftu->occmat().get_flat(atom, target_L, atom_occ);
        }
        else
        {
            std::copy(occupations.begin() + offset,
                      occupations.begin() + onsite_offsets[atom + 1],
                      atom_occ.begin());
            this->dftu->occmat().set_flat(atom, target_L, this->current_spin, atom_occ);
        }

        std::vector<double> onsite_pauli(atom_occ.size(), 0.0);
        double energy = this->dftu->get_energy();
        DFTU_LCAO::cal_pot_onsite(atom_occ,
                             projector_size,
                             this->dftu->get_u_current(type),
                             onsite_pauli.data(),
                             energy);
        this->dftu->set_energy(energy);
        std::vector<std::complex<double>> onsite_spinor(atom_occ.size(), 0.0);
        DFTU_LCAO::transfer_pot_onsite(onsite_pauli, onsite_spinor);
        std::copy(onsite_spinor.begin(), onsite_spinor.end(), onsite.begin() + offset);
    }

    if (keep_device)
    {
        this->device_hr_ = hamilt::dftu_gpu::build_hubbard_hamiltonian(this->gpu_cache_, onsite.data());
    }
    else
    {
        hamilt::dftu_gpu::add_hubbard_hamiltonian(this->gpu_cache_, onsite.data(), this->hR->get_wrapper());
    }
    return true;
}
#endif

template <typename TK, typename TR>
void hamilt::DFTU_onsite<hamilt::OperatorLCAO<TK, TR>>::finish_hr_contribution()
{
    if (this->nspin == 1)
    {
        this->dftu->set_double_energy();
    }
    if (this->current_spin == this->nspin - 1 || this->nspin == 4)
    {
        this->dftu->set_occmat_stale();
    }
    if (this->nspin == 2)
    {
        this->current_spin = 1 - this->current_spin;
    }
}

template <typename TK, typename TR>
hamilt::DFTU_onsite<hamilt::OperatorLCAO<TK, TR>>::~DFTU_onsite()
{
#ifdef __CUDA
    hamilt::dftu_gpu::destroy_cache(gpu_cache_);
#endif
}

template <typename TK, typename TR>
bool hamilt::DFTU_onsite<hamilt::OperatorLCAO<TK, TR>>::supportsDeviceHR() const
{
    const bool restore_symmetry = kv_ != nullptr && ModuleSymmetry::Symmetry::symm_flag == 1
                                  && !kv_->kstars.empty();
    return use_gpu_ && nspin == 4 && !restore_symmetry;
}

template <typename TK, typename TR>
void hamilt::DFTU_onsite<hamilt::OperatorLCAO<TK, TR>>::contributeHR()
{
    contribute_hr_impl(false);
}

template <typename TK, typename TR>
void hamilt::DFTU_onsite<hamilt::OperatorLCAO<TK, TR>>::contributeHRDevice()
{
    contribute_hr_impl(true);
}

// contributeHR()
/**
 * @brief Contribute DFT+U Hamiltonian to real-space HR matrix
 * 
 * @details This function handles different scenarios based on:
 * 1. Whether occ_mat (occupation matrix) is read from file (is_occmat_ready)
 * 2. Spin configuration (nspin=1, 2, or 4)
 * 3. SCF iteration stage (first vs subsequent iterations)
 * 
 * Case 1: Occ_mat NOT ready (!is_occmat_ready)
 *   - First electronic iteration: calculates occupation matrix from density matrix (DMR)
 *     * Fetches the real-space DMR via dm_->get_dmr_ptr()
 *     * Accumulates contributions from all atom pairs via DFTU_LCAO::cal_occ_ijr()
 *     * Performs MPI reduction to sum occ across processes
 *     * Stores result via set_occ_mat_flat() for use in pot_onsite calculation
 *     * For nspin=1: occ is scaled by 0.5 (since only one spin channel computed)
 *   - Subsequent iterations: occ_mat is computed fresh each iteration from updated DMR
 * 
 * Case 2: Occ_mat IS ready (is_occmat_ready, i.e., read from occ_mat.txt file)
 *   - First electronic iteration: uses pre-read occ_mat directly without DMR calculation
 *     * Skips DMR-based occ calculation entirely
 *     * Reads occ_mat from stored data via get_occ_mat()
 *     * Different indexing for nspin=4 vs nspin=1/2 (see below)
 *   - After first iteration: set_occmat_stale() is called to force recomputation
 * 
 * Spin configurations:
 *   nspin=1 (non-spin-polarized):
 *     - Single spin channel, occ computed once
 *     - Energy correction doubled at end (set_double_energy)
 *     - current_spin always 0
 *   
 *   nspin=2 (collinear spin-polarized):
 *     - Two separate spin channels (spin-up: 0, spin-down: 1)
 *     - current_spin toggles between 0 and 1 across iterations
 *     - set_occmat_stale() called when current_spin == 1 (last spin)
 *     - HR accumulated separately for each spin
 *   
 *   nspin=4 (non-collinear/SOC):
 *     - Single 4x4 Pauli matrix representation per atom
 *     - occ has 4*(2l+1)^2 elements (spin_fold=4)
 *     - get_occ_mat uses spin=0, ipol indices for Pauli blocks
 *     - set_occmat_stale() always called (current_spin check always true)
 *     - No current_spin toggling (all spins handled simultaneously)
 * 
 * @warning THREAD SAFETY: DFTU_LCAO::cal_hr_ijr() updates shared HR matrix entries.
 *          Different iat0 may contribute to same HR(iat1, iat2, R), requiring
 *          critical section protection for multithreaded correctness.
 *          TODO: Consider refactoring to atom_row_list pattern (see nonlocal.cpp)
 *          for better parallel performance instead of critical section.
 */
template <typename TK, typename TR>
void hamilt::DFTU_onsite<hamilt::OperatorLCAO<TK, TR>>::contribute_hr_impl(const bool keep_device)
{
    ModuleBase::TITLE("DFTU_onsite", "contributeHR");
    // Early exit: DMR not available (first SCF iteration before the first
    // diagonalization) AND occ_mat not yet initialized
    this->device_hr_ = nullptr;
    const bool dmr_null = (this->dm_ == nullptr || !this->dm_->is_dmr_ready());
    const bool occ_mat_not_init = !this->dftu->is_occmat_ready();

    if (dmr_null && occ_mat_not_init)
    {
        return;
    }
    if (this->current_spin == 0)
    {
        this->dftu->set_energy(0.0);
    }
    ModuleBase::timer::start("DFTU_onsite", "contributeHR");

    const Parallel_Orbitals* pv = this->hR->get_atom_pair(0).get_paraV();
    if (contribute_hr_gpu(pv, keep_device))
    {
        finish_hr_contribution();
        ModuleBase::timer::end("DFTU_onsite", "contributeHR");
        return;
    }
    if (keep_device)
    {
        ModuleBase::WARNING_QUIT("DFTU_onsite", "device H(R) requires a compatible GPU DFT+U layout");
    }
    // nlm_tot is precomputed in the constructor (structure snapshot)

    // (symmetry) when crystal symmetry reduces the k-mesh, dm_->get_dmr_ptr()
    // was Fourier-transformed from the irreducible k-points only and is not
    // actually symmetric; reconstruct the full-BZ DMR once here (reused by every
    // atom below) via the same D(k) restoration EXX already uses for its own
    // real-space density matrix (ModuleSymmetry::Symmetry_rotation::restore_dm).
    std::unique_ptr<module_dm::DensityMatrix<TK, double>> dmr_sym;
    if (!this->dftu->is_occmat_ready() && this->kv_ != nullptr && ModuleSymmetry::Symmetry::symm_flag == 1
        && !this->kv_->kstars.empty())
    {
        if (!this->symrot_built_)
        {
            const std::array<int, 3> period{ this->kv_->nmp[0], this->kv_->nmp[1], this->kv_->nmp[2] };
            // for return_lattice to calculate Ms
            this->symrot_.find_irred_sector(this->ucell->symm, this->ucell->atoms,
                this->ucell->nat, this->ucell->ntype, this->ucell->iat2it, this->ucell->iat2ia,
                ModuleSymmetry::Symmetry_rotation_k::get_bvk_cells(period), period, this->ucell->lat);
            this->symrot_.cal_Ms(*this->kv_, this->ucell->symm, this->ucell->atoms,
                this->ucell->latvec, this->ucell->lmax,
                this->ucell->nat, this->ucell->iat2it, this->ucell->iat2ia,
                *pv, this->nspin);
            this->symrot_built_ = true;
        }
        const int nspin0 = (this->nspin == 2) ? 2 : 1;
        // (k-point pools, KPAR>1) restore_dm() now returns only the stars of THIS pool's own
        // local irreducible k-points (see its definition for why that's enough); kvec_d_full
        // must be built the same way -- one entry per star member of each local ibz-k,
        // enumerated in the same order restore_dm uses for its spin-0 block (kv.ik2iktot maps
        // the spin-0 and spin-1 blocks to the same sequence of global ibz indices, so a single
        // list built from the spin-0 mapping is valid for the whole nspin0-block DensityMatrix).
        const int nk_local = this->kv_->get_nks() / nspin0;
        const int nks_ibz_global = static_cast<int>(this->kv_->kstars.size());
        std::vector<ModuleBase::Vector3<double>> kvec_d_full;
        for (int ik_local = 0; ik_local < nk_local; ++ik_local)
        {
            const int ik_ibz = this->kv_->ik2iktot[ik_local] % nks_ibz_global;
            for (const std::pair<const int, ModuleBase::Vector3<double>>& isym_kvd : this->kv_->kstars[ik_ibz]) { kvec_d_full.push_back(isym_kvd.second); }
        }
        const std::vector<std::vector<TK>> dmk_full = this->symrot_.restore_dm(*this->kv_, this->dm_->get_dmk_vec(), *pv);
        dmr_sym.reset(new module_dm::DensityMatrix<TK, double>(pv, nspin0, kvec_d_full, static_cast<int>(kvec_d_full.size())));
        dmr_sym->init_dmr(*this->dm_->get_dmr_ptr(1));
        dmr_sym->get_dmk_vec() = dmk_full;
        dmr_sym->cal_dmr(-1);
    }

    // loop over all Hubbard-projector center atoms (iat0)
    int atom_index = 0;
    for (int iat0 = 0; iat0 < this->ucell->nat; iat0++)
    {
        int T0 = 0;
        int I0 = 0;
        ucell->iat2iait(iat0, &I0, &T0);
        if (!this->dftu->has_l_channel(T0))
        {
            continue;
        }
        const int target_L = this->dftu->get_l_channel(T0);
        const int tlp1 = 2 * target_L + 1;
        AdjacentAtomInfo& adjs = this->adjs_all[atom_index++];

        const int spin_fold = (this->nspin == 4) ? 4 : 1;
        std::vector<double> occ(tlp1 * tlp1 * spin_fold, 0.0);

        // compute or load occupation matrix
        if (!this->dftu->is_occmat_ready())
        {
            // DMR is guaranteed ready here: otherwise the early exit above
            // would have returned. DMR index is 1-based, hence +1.
            const hamilt::HContainer<double>* dmr = this->dm_->get_dmr_ptr(this->current_spin + 1);
            if (dmr_sym) { dmr = dmr_sym->get_dmr_ptr(this->current_spin + 1); }
            DFTU_LCAO::compute_occ_from_dmr(*this->ucell,
                                            *this->dftu,
                                            iat0,
                                            target_L,
                                            this->current_spin,
                                            this->nspin,
                                            adjs,
                                            *pv,
                                            this->nlm_tot,
                                            *dmr,
                                            occ);
        }
        else
        {
            DFTU_LCAO::load_occ_from_file(*this->dftu, iat0, target_L, this->nspin, this->current_spin, occ);
        }

        // compute Hubbard potential and energy
        const double u_value = this->dftu->get_u_current(T0);
        std::vector<double> pot_onsite_tmp(occ.size());
        double u_energy = this->dftu->get_energy();
        DFTU_LCAO::cal_pot_onsite(occ, tlp1, u_value, pot_onsite_tmp.data(), u_energy);
        this->dftu->set_energy(u_energy);

        std::vector<TR> pot_onsite(occ.size());
        DFTU_LCAO::transfer_pot_onsite(pot_onsite_tmp, pot_onsite);

        // accumulate HR contributions from neighbor pairs
        DFTU_LCAO::accumulate_hr_for_iat0<TR>(*this->ucell, this->hR, this->nlm_tot, iat0, adjs, *pv, pot_onsite);
    }

    finish_hr_contribution();

    ModuleBase::timer::end("DFTU_onsite", "contributeHR");
}

template class hamilt::DFTU_onsite<hamilt::OperatorLCAO<double, double>>;
template class hamilt::DFTU_onsite<hamilt::OperatorLCAO<std::complex<double>, double>>;
template class hamilt::DFTU_onsite<hamilt::OperatorLCAO<std::complex<double>, std::complex<double>>>;
