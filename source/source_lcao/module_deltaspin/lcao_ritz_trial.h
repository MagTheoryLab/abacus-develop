#ifndef ABACUS_LCAO_RITZ_TRIAL_H
#define ABACUS_LCAO_RITZ_TRIAL_H

#include "source_base/vector3.h"
#include <complex>
#include <iosfwd>
#include <memory>
#include <vector>

namespace elecstate
{
template <typename TK>
class ElecStateLCAO;
}
namespace hamilt
{
template <typename TK, typename TR>
class OperatorLCAO;
template <typename T>
class DeltaSpin;
}
namespace hsolver
{
struct LcaoRitzReference;
}

namespace spinconstrain
{
class LcaoRitzGpu;

// One fixed SCF reference, borrowed from the k-owner solve. Destroy the trial
// before replacing the reference if the search continues after a full solve.
class LcaoRitzTrial
{
  public:
    using Complex = std::complex<double>;
    using Dspin = hamilt::DeltaSpin<hamilt::OperatorLCAO<Complex, Complex>>;
    using Moments = std::vector<ModuleBase::Vector3<double>>;

    LcaoRitzTrial(const elecstate::ElecStateLCAO<Complex>& estate,
                  const Dspin& dspin,
                  const hsolver::LcaoRitzReference& reference,
                  const Moments& lambda,
                  int nlocal);
    ~LcaoRitzTrial();

    bool evaluate(const Moments& lambda, Moments& moments, int outer_step, std::ostream& log);

  private:
    const elecstate::ElecStateLCAO<Complex>& estate_;
    const Dspin& dspin_;
    const hsolver::LcaoRitzReference& reference_;
    const Moments anchor_lambda_;
    const int nlocal_;
    std::unique_ptr<LcaoRitzGpu> gpu_;
    bool prepared_ = false;
};
}

#endif
