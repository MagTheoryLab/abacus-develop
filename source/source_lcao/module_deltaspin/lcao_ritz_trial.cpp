#include "lcao_ritz_trial.h"

#include "lcao_ritz_gpu.h"
#include "source_base/parallel_reduce.h"
#include "source_estate/elecstate_lcao.h"
#include "source_estate/elecstate_tools.h"
#include "source_hsolver/hsolver_lcao.h"
#include "source_lcao/module_operator_lcao/dspin_lcao.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <ostream>

namespace spinconstrain
{
LcaoRitzTrial::LcaoRitzTrial(elecstate::ElecStateLCAO<Complex>& estate,
                           Dspin& dspin,
                           const hsolver::LcaoRitzReference& reference,
                           const Moments& lambda,
                           int nlocal)
    : estate_(estate), dspin_(dspin), reference_(reference), anchor_lambda_(lambda), nlocal_(nlocal)
{
}

LcaoRitzTrial::~LcaoRitzTrial() = default;

bool LcaoRitzTrial::evaluate(const Moments& lambda, Moments& result, int outer_step, std::ostream& log, bool publish)
{
    const int bands = estate_.ekb.nc;
    const int kpoints = estate_.ekb.nr;
    const int nat = result.size();
    const auto begin = std::chrono::steady_clock::now();
    const auto& waves = estate_.k_owner_wavefunctions();
    if (waves.size() != kpoints || reference_.energy_by_k.size() != kpoints)
    {
        return false;
    }
    const auto moments = [&](const ModuleBase::matrix& wg, bool anchor) {
        std::vector<double> spin_components(3 * nat, 0.0);
        for (int ik = 0; ik < kpoints; ++ik)
        {
            if (waves[ik] != nullptr)
            {
                gpu_->moments(ik, waves[ik]->get_pointer(), &wg(ik, 0), anchor, spin_components.data());
            }
        }
        const int components = spin_components.size();
        Parallel_Reduce::reduce_all(spin_components.data(), components);
        return spin_components;
    };
    if (!prepared_)
    {
        std::vector<int> local_owners(kpoints, 0);
        bool local_owner = false;
        for (int ik = 0; ik < kpoints; ++ik)
        {
            if (waves[ik] != nullptr && waves[ik]->get_nbands() == reference_.bands
                && reference_.energy_by_k[ik].size() == reference_.bands)
            {
                local_owners[ik] = 1;
                local_owner = true;
            }
        }
        auto owners = local_owners;
        Parallel_Reduce::reduce_all(owners.data(), kpoints);
        if (!std::all_of(owners.begin(), owners.end(), [](int count) { return count == 1; }))
        {
            return false;
        }
        const auto geometry = dspin_.ritz_geometry();
        const int projectors = std::accumulate(geometry.projector_counts.begin(), geometry.projector_counts.end(), 0);
        if (projectors <= 0 || size_t(2) * projectors * reference_.bands > std::numeric_limits<int>::max())
        {
            return false;
        }
        // Nonowners participate in folding and occupations without allocating CUDA scratch.
        if (local_owner)
        {
            gpu_.reset(new LcaoRitzGpu(nlocal_, bands, reference_.bands, kpoints, geometry));
        }
        const auto kbegin = estate_.klist->kvec_d.begin();
        const std::vector<ModuleBase::Vector3<double>> kvec(kbegin, kbegin + kpoints);
        const int spatial = nlocal_ / 2;
        const auto& folded = dspin_.ritz_projectors(kvec, local_owners, spatial, projectors);
        for (int ik = 0; ik < kpoints; ++ik)
        {
            if (waves[ik] != nullptr)
            {
                gpu_->prepare(ik, folded[ik].data(), kvec[ik]);
                log << "LCAO_RITZ_OWNER k " << ik << " basis " << nlocal_ << " bands "
                    << bands << " ritz " << reference_.bands << " projectors " << projectors << '\n';
            }
        }
        const auto anchor = moments(estate_.wg, true);
        double error = 0.0;
        for (int ia = 0; ia < nat; ++ia)
        {
            const ModuleBase::Vector3<double> rebuilt(anchor[3 * ia], anchor[3 * ia + 1], anchor[3 * ia + 2]);
            const auto difference = rebuilt - result[ia];
            const double norm = difference.norm();
            if (!std::isfinite(norm))
            {
                return false;
            }
            error = std::max(error, norm);
        }
        const double setup = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        log << "LCAO_RITZ_ANCHOR outer " << outer_step << " error_uB " << error
            << " setup_s " << setup << '\n';
        if (error > 1e-7)
        {
            return false;
        }
        prepared_ = true;
    }
    const auto trial_begin = std::chrono::steady_clock::now();
    std::vector<double> delta(3 * nat);
    for (int ia = 0; ia < nat; ++ia)
    {
        const auto difference = lambda[ia] - anchor_lambda_[ia];
        for (int axis = 0; axis < 3; ++axis)
        {
            delta[3 * ia + axis] = difference[axis];
        }
    }
    ModuleBase::matrix eigen(kpoints, bands);
    for (int ik = 0; ik < kpoints; ++ik)
    {
        if (waves[ik] != nullptr)
        {
            gpu_->solve(ik, waves[ik]->get_pointer(), reference_.energy_by_k[ik].data(), delta.data(),
                        &eigen(ik, 0), publish);
        }
    }
    const int energy_count = kpoints * bands;
    Parallel_Reduce::reduce_all(eigen.c, energy_count);
    ModuleBase::matrix wg(kpoints, bands);
    auto fermi = estate_.eferm;
    auto energy = estate_.f_en;
    auto nelec = estate_.nelec_spin;
    elecstate::calculate_weights(eigen, wg, estate_.klist, fermi, energy, nelec, bands, false);
    const auto trial = moments(wg, false);
    for (int ia = 0; ia < nat; ++ia)
    {
        result[ia] = ModuleBase::Vector3<double>(trial[3 * ia], trial[3 * ia + 1], trial[3 * ia + 2]);
    }
    if (publish)
    {
        for (int ik = 0; ik < kpoints; ++ik)
        {
            if (waves[ik] != nullptr)
            {
                gpu_->publish_wavefunctions(ik, waves[ik]->get_pointer());
            }
        }
        estate_.ekb = std::move(eigen);
        estate_.wg = std::move(wg);
        estate_.eferm = fermi;
        estate_.f_en.demet = energy.demet;
        estate_.nelec_spin = std::move(nelec);
        elecstate::calEBand(estate_.ekb, estate_.wg, estate_.f_en);
        dspin_.update_lambda();
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - trial_begin).count();
    log << "LCAO_RITZ_TRIAL outer " << outer_step << " seconds " << elapsed << '\n';
    return true;
}
}
