#ifndef CAL_DM_PSI_H
#define CAL_DM_PSI_H

#include "density_matrix.h"
#include "source_base/matrix.h"
#include "source_psi/psi.h"

namespace elecstate
{
// for Gamma-Only case where DMK is double
void cal_dm_psi(const Parallel_Orbitals* ParaV,
                const ModuleBase::matrix& wg,
                const psi::Psi<double>& wfc,
                elecstate::DensityMatrix<double, double>& DM);

// for Multi-k case where DMK is std::complex<double>
template <typename TR>
void cal_dm_psi(const Parallel_Orbitals* ParaV,
                const ModuleBase::matrix& wg,
                const psi::Psi<std::complex<double>>& wfc,
                elecstate::DensityMatrix<std::complex<double>, TR>& DM);

#ifdef __CUDA
/**
 * @brief Build the complex LCAO density matrix with a single MPI rank on CUDA.
 *
 * The LCAO eigensolver interface currently exposes host-owned eigenvectors.
 * This entry preserves the existing host DensityMatrix interface while moving
 * the weighted-conjugate transform and the dominant zgemm to the GPU. It is
 * independent of the selected LCAO eigensolver.
 */
void cal_dm_psi_gpu_single_rank(const Parallel_Orbitals* ParaV,
                                const ModuleBase::matrix& wg,
                                const psi::Psi<std::complex<double>>& wfc,
                                elecstate::DensityMatrix<std::complex<double>, double>& DM);
#endif

#ifdef __MPI
// for Gamma-Only case with MPI
void psiMulPsiMpi(const psi::Psi<double>& psi1, const psi::Psi<double>& psi2, double* dm_out, const int* desc_psi, const int* desc_dm);

// for multi-k case with MPI
void psiMulPsiMpi(const psi::Psi<std::complex<double>>& psi1,
                  const psi::Psi<std::complex<double>>& psi2,
                  std::complex<double>* dm_out,
                  const int* desc_psi,
                  const int* desc_dm);

#else
// for Gamma-Only case without MPI
void psiMulPsi(const psi::Psi<double>& psi1, const psi::Psi<double>& psi2, double* dm_out);

// for multi-k case without MPI
void psiMulPsi(const psi::Psi<std::complex<double>>& psi1, const psi::Psi<std::complex<double>>& psi2, std::complex<double>* dm_out);
#endif
}; // namespace elecstate
#endif
