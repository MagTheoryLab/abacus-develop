#ifndef DFTU_LCAO_GPU_H
#define DFTU_LCAO_GPU_H

#include <cstddef>
#include <complex>

namespace hamilt
{
namespace dftu_gpu
{

struct ProjectionTask
{
    std::size_t left_offset;
    std::size_t right_offset;
    std::size_t dm_offset;
    std::size_t hr_offset;
    std::size_t onsite_offset;
    int row_orbitals;
    int col_orbitals;
    int matrix_cols;
    int projector_size;
    int has_dm;
    int has_hr;
};

void* create_cache(const double* projections,
                   std::size_t projection_count,
                   const ProjectionTask* tasks,
                   std::size_t task_count,
                   std::size_t dm_size,
                   std::size_t hr_size,
                   std::size_t onsite_size);

void compute_occupations(void* cache,
                         const double* dm,
                         double* occupations);

void add_hubbard_hamiltonian(void* cache,
                            const std::complex<double>* onsite,
                            std::complex<double>* hr);

const std::complex<double>* build_hubbard_hamiltonian(
    void* cache,
    const std::complex<double>* onsite);

void destroy_cache(void* cache);

} // namespace dftu_gpu
} // namespace hamilt

#endif
