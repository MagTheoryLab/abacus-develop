#ifndef HAMILT_FOLDING_HR_GPU_H
#define HAMILT_FOLDING_HR_GPU_H

#include <cstddef>
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
    /// Fold values already stored in the HContainer wrapper order on device.
    /// The input remains owned by the caller and only needs to outlive this call.
    std::complex<double>* fold_device(int ik,
                                      const std::complex<double>* values,
                                      std::size_t count);
    /// Add a device-resident H(R) contribution to the uploaded host-side
    /// contributions while folding, without materializing the addend on host.
    std::complex<double>* fold_with_device_addend(int ik,
                                                  const std::complex<double>* addend,
                                                  std::size_t count);
    /// Fold an uploaded base plus two independently-owned device addends.
    std::complex<double>* fold_with_device_addends(int ik,
                                                   const std::complex<double>* first,
                                                   const std::complex<double>* second,
                                                   std::size_t count);

  private:
    std::complex<double>* fold_impl(int ik,
                                    const std::complex<double>* values,
                                    const std::complex<double>* first,
                                    const std::complex<double>* second,
                                    std::size_t count);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
#endif
