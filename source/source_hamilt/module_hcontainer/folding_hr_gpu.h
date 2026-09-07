#ifndef HAMILT_FOLDING_HR_GPU_H
#define HAMILT_FOLDING_HR_GPU_H

#include <complex>
#include <memory>
#include <vector>

namespace ModuleBase { template <typename T> class Vector3; }

namespace hamilt
{
template <typename T> class HContainer;

/// Geometry-scoped complex H(R) folding into a rank-local column-major matrix.
/// Values must be uploaded explicitly after any real-space change. The layout
/// and k points are fixed for the lifetime of this object.
class FoldingHrGpu
{
  public:
    FoldingHrGpu(const HContainer<std::complex<double>>& hr,
                 const std::vector<ModuleBase::Vector3<double>>& kpoints);
    ~FoldingHrGpu();
    void upload(const HContainer<std::complex<double>>& hr);
    /// Borrowed device pointer, valid until the next fold/destruction.
    std::complex<double>* fold(int ik);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
#endif
