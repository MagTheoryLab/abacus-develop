#ifndef HSOLVER_K_OWNER_MATRIX_GPU_H
#define HSOLVER_K_OWNER_MATRIX_GPU_H

#include <complex>
#include <memory>

class Parallel_2D;

namespace hsolver
{
// Gather the authoritative upper triangles of a block-cyclic H/S pair.
// The owner matrices stay on device until the next pair assigned to that owner.
class KOwnerMatrixGpu
{
  public:
    KOwnerMatrixGpu(const Parallel_2D& layout, int world_rank);
    ~KOwnerMatrixGpu();
    void gather(const std::complex<double>* h, const std::complex<double>* s, int owner);
    void gather_device(const std::complex<double>* h, const std::complex<double>* s, int owner);
    std::complex<double>* h_device();
    std::complex<double>* s_device();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
#endif
