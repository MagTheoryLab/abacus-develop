#pragma once

#include <memory>

namespace hamilt
{
template <typename T> class HContainer;

// Geometry-scoped distributed -> grid-local real density redistribution.
// The serial layout is fixed; repeated calls exchange only updated values.
class DensityGather
{
  public:
    DensityGather(const HContainer<double>& source, const HContainer<double>& serial_layout);
    ~DensityGather();
    bool matches(const HContainer<double>& source) const;
    const HContainer<double>& gather(const HContainer<double>& source);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
