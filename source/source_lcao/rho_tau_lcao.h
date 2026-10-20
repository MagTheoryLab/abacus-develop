#ifndef RHO_TAU_LCAO_H
#define RHO_TAU_LCAO_H

#include "source_hamilt/module_hcontainer/hcontainer.h"
#include "source_estate/module_charge/charge.h"

// generate charge density from different basis or methods
namespace LCAO_domain
{
    // Return N / integral(raw rho) for the LDA/GGA density pullback; zero otherwise.
    double dm2rho(std::vector<hamilt::HContainer<double>*> &dmr,
            const int nspin,
            Charge* chr,
            const double nelec,
            const double omega,
            const bool skip_normalize);

    void dm2tau(std::vector<hamilt::HContainer<double>*> &dmr,
            const int nspin,
            Charge* chr);
}

#endif
