#ifndef DM_FROM_PSI_GPU_H
#define DM_FROM_PSI_GPU_H

#include "density_matrix.h"
#include "source_base/matrix.h"
#include "source_psi/psi.h"

namespace module_dm
{
#ifdef __CUDA
/**
 * @brief Build the complex LCAO density matrix with a single MPI rank on CUDA.
 *
 * The LCAO eigensolver interface currently exposes host-owned eigenvectors.
 * This entry preserves the existing host DensityMatrix interface while moving
 * the weighted-conjugate transform and the dominant zgemm to the GPU. It is
 * independent of the selected LCAO eigensolver.
 */
void dm_from_psi_gpu_single_rank(const Parallel_Orbitals* ParaV,
                                const ModuleBase::matrix& wg,
                                const psi::Psi<std::complex<double>>& wfc,
                                module_dm::DensityMatrix<std::complex<double>, double>& DM);

#if defined(__MPI)
/**
 * @brief Build distributed real-space DMR directly from k-owner wavefunctions.
 *
 * Every rank contributes the k points for which owner_wfc[ik] is non-null.
 * Eigenvectors and the transient dense DMK remain on the owner's GPU. Only the
 * rank-concatenated sparse DMR contribution is copied to the host and
 * reduce-scattered to the existing local HContainer layout.
 */
void dmr_from_k_owner_gpu(
    const Parallel_Orbitals* para_v,
    const ModuleBase::matrix& wg,
    const std::vector<const psi::Psi<std::complex<double>, base_device::DEVICE_GPU>*>& owner_wfc,
    module_dm::DensityMatrix<std::complex<double>, double>& dm);

// Full complex spinor DMR for nonlocal force/stress. Unlike the real Pauli
// representation above, this preserves the imaginary matrix components.
void dmr_from_k_owner_gpu(
    const Parallel_Orbitals* para_v,
    const ModuleBase::matrix& weights,
    const std::vector<const psi::Psi<std::complex<double>, base_device::DEVICE_GPU>*>& owner_wfc,
    const std::vector<ModuleBase::Vector3<double>>& kvec_d,
    hamilt::HContainer<std::complex<double>>& full_dmr);
#endif
#endif

} // namespace module_dm
#endif
