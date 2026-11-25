#ifndef ABACUS_LCAO_RITZ_GPU_H
#define ABACUS_LCAO_RITZ_GPU_H

#include "source_base/vector3.h"

#include <complex>
#include <memory>
#include <vector>

namespace spinconstrain
{
struct RitzOmittedPair
{
    int atom;
    int row;
    int col;
    int rows;
    int cols;
    int projectors;
    int left;
    int right;
    ModuleBase::Vector3<int> translation;
};

struct RitzProjectorGeometry
{
    std::vector<int> projector_counts;
    std::vector<RitzOmittedPair> omitted;
    std::vector<double> factors;
};

// Device scratch and folded projectors belong to one lambda loop. The solver
// retains ownership of the reference eigenvectors for each local k point.
class LcaoRitzGpu
{
  public:
    LcaoRitzGpu(int nlocal, int bands, int ritz_bands, int kpoints, const RitzProjectorGeometry& geometry);
    ~LcaoRitzGpu();

    void prepare(int ik, const std::complex<double>* projectors, const ModuleBase::Vector3<double>& k);
    void solve(int ik,
               const std::complex<double>* vectors,
               const double* reference_energy,
               const double* delta_lambda,
               double* output_energy,
               bool keep_rotation);
    void publish_wavefunctions(int ik, std::complex<double>* waves);
    void moments(int ik, const std::complex<double>* vectors, const double* weights,
                 bool reference, double* output);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
#endif
