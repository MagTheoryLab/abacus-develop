/// @file dftu_nao_fs_reduce.cpp
/// @brief Post-processing of DFT+U force and stress (MPI reduce + scaling)
///
/// These functions are separated from dftu_nao_fs_r.cpp to keep the link
/// closure minimal for unit testing.

#include "dftu_nao_fs_r.h"

#include "source_base/parallel_reduce.h"
#include "source_cell/unitcell.h"

namespace DFTU_LCAO
{

void reduce_force_impl(ModuleBase::matrix& force, const int nspin)
{
    Parallel_Reduce::reduce_all(force.c, force.nr * force.nc);
    if (nspin != 4)
    {
        for (int i = 0; i < force.nr * force.nc; i++)
        {
            force.c[i] *= 2.0;
        }
    }
}

void reduce_stress_impl(const UnitCell* ucell,
                        const std::vector<double>& stress_tmp,
                        ModuleBase::matrix& stress,
                        const int nspin)
{
    Parallel_Reduce::reduce_all(const_cast<double*>(stress_tmp.data()), 6);
    // Pauli potentials W obey dE_U = (1/2) sum_a Tr(W_a dN_a).
    // Both CPU and GPU stress kernels already include both projector derivatives.
    // Force combines those derivatives instead, cancelling this factor of 1/2.
    const double spin_weight = nspin == 4 ? 0.5 : 1.0;
    const double weight = spin_weight * ucell->lat0 / ucell->omega;
    for (int i = 0; i < 6; i++)
    {
        stress.c[i] = stress_tmp[i] * weight;
    }
    stress.c[8] = stress.c[5]; // stress(2,2)
    stress.c[7] = stress.c[4]; // stress(2,1)
    stress.c[6] = stress.c[2]; // stress(2,0)
    stress.c[5] = stress.c[4]; // stress(1,2)
    stress.c[4] = stress.c[3]; // stress(1,1)
    stress.c[3] = stress.c[1]; // stress(1,0)
}

} // namespace DFTU_LCAO
