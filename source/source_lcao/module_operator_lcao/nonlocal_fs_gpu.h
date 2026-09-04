#ifndef NONLOCAL_FS_GPU_H
#define NONLOCAL_FS_GPU_H
#include <cstddef>
#include <complex>
#include <vector>

namespace hamilt
{
namespace nonlocal_gpu
{
// Shared projector contraction for nonlocal pseudopotentials and DFT+U.
// Offsets refer to rank-local packed arrays, not global orbital indices.
struct Coupling
{
    int p1;
    int p2;
    int spin;
    double real;
    double imag;
};
struct Task
{
    std::size_t left;
    std::size_t right;
    std::size_t dm;
    std::size_t coupling;
    int coupling_count;
    int rows;
    int cols;
    int projectors;
    int npol;
    int center;
    int atom;
    double dis1[3];
    double dis2[3];
};
struct Batch
{
    std::vector<double> projections;
    std::vector<Coupling> couplings;
    std::vector<Task> tasks;
};
void compute(const Batch& batch, const double* dm, std::size_t dm_size,
             int nat, bool cal_force, bool cal_stress, double* force, double* stress);
void compute(const Batch& batch, const std::complex<double>* dm, std::size_t dm_size,
             int nat, bool cal_force, bool cal_stress, double* force, double* stress);
}
}
#endif
