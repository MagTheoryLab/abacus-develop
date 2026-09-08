#pragma once

#include <complex>
#include <memory>

namespace hamilt
{
template <typename T> class HContainer;
}

namespace ModuleGint
{
// Geometry-scoped redistribution: grid-local Pauli values -> orbital-owned
// spinor values. Metadata is immutable after construction; only values move.
class SpinorHrGpu
{
  public:
    SpinorHrGpu(const hamilt::HContainer<double>& source,
                const hamilt::HContainer<std::complex<double>>& destination);
    ~SpinorHrGpu();
    bool matches(const hamilt::HContainer<std::complex<double>>& destination) const;
    /// Build the rank-owned spinor H(R) values and keep them on the device.
    /// The borrowed pointer remains valid until the next transfer or destruction.
    const std::complex<double>* transfer_device(const double* v0,
                                                const double* vx,
                                                const double* vy,
                                                const double* vz,
                                                bool transverse);
    void transfer(const double* v0, const double* vx, const double* vy, const double* vz,
                  bool transverse, hamilt::HContainer<std::complex<double>>& destination);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
