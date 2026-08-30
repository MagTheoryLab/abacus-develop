#include "xc_functional_ncgga_sf.h"

#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_base/vector3.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"
#include "source_estate/module_charge/charge.h"
#include "xc_functional.h"
#include "xc_ncgga_radial.h"

#include <array>
#include <cmath>
#include <vector>

namespace
{

struct ContinuousGradientPoint
{
    double gamma_up = 0.0;
    double gamma_down = 0.0;
    ModuleBase::Vector3<double> grad_magnitude;
    ModuleBase::Vector3<double> dgamma_up_dn;
    ModuleBase::Vector3<double> dgamma_down_dn;
    std::array<ModuleBase::Vector3<double>, 3> dgamma_up_dm;
    std::array<ModuleBase::Vector3<double>, 3> dgamma_down_dm;
    std::array<double, 3> dgamma_difference_dm = {{0.0, 0.0, 0.0}};
};

ContinuousGradientPoint make_continuous_gradient_point(
    const ModuleBase::Vector3<double>& grad_n,
    const std::array<ModuleBase::Vector3<double>, 3>& grad_m,
    const std::array<double, 3>& magnetization)
{
    ContinuousGradientPoint point;
    const double gamma_nn = grad_n * grad_n;
    double gamma_mm = 0.0;
    const double magnitude = std::sqrt(magnetization[0] * magnetization[0]
                                       + magnetization[1] * magnetization[1]
                                       + magnetization[2] * magnetization[2]);
    std::array<double, 3> direction = {{0.0, 0.0, 0.0}};
    if (magnitude > 1.0e-20)
    {
        for (int mu = 0; mu < 3; ++mu)
        {
            direction[mu] = magnetization[mu] / magnitude;
        }
    }
    for (int mu = 0; mu < 3; ++mu)
    {
        gamma_mm += grad_m[mu] * grad_m[mu];
        point.grad_magnitude += direction[mu] * grad_m[mu];
    }

    const double longitudinal_cross = grad_n * point.grad_magnitude;
    point.gamma_up = 0.25 * (gamma_nn + gamma_mm) + 0.5 * longitudinal_cross;
    point.gamma_down = 0.25 * (gamma_nn + gamma_mm) - 0.5 * longitudinal_cross;
    point.dgamma_up_dn = 0.5 * grad_n + 0.5 * point.grad_magnitude;
    point.dgamma_down_dn = 0.5 * grad_n - 0.5 * point.grad_magnitude;
    for (int mu = 0; mu < 3; ++mu)
    {
        point.dgamma_up_dm[mu] = 0.5 * grad_m[mu] + 0.5 * direction[mu] * grad_n;
        point.dgamma_down_dm[mu] = 0.5 * grad_m[mu] - 0.5 * direction[mu] * grad_n;
        if (magnitude > 1.0e-20)
        {
            const ModuleBase::Vector3<double> direction_response
                = (grad_m[mu] - direction[mu] * point.grad_magnitude) / magnitude;
            point.dgamma_difference_dm[mu] = grad_n * direction_response;
        }
    }
    return point;
}

} // namespace

namespace ModuleXC
{
namespace NCGGA_SF_Builtin
{

std::tuple<double, double, ModuleBase::matrix> v_xc_ncgga_sf_builtin(
    const int& nrxx, const double& omega, const double tpiba, const Charge* const chr,
    const int gga_grad)
{
    ModuleBase::TITLE("XC_Functional", "v_xc_ncgga_sf_builtin");
    ModuleBase::timer::start("XC_Functional", "v_xc_ncgga_sf_builtin");

    // Caller guarantees nspin==4 with noncollinear magnetism and
    // gga_grad==2 or 3.

    // ======================================================================
    // Builtin noncollinear GGA (gga_grad=2/3)
    //
    // Reference: Scalmani & Frisch, JCTC 8, 1069 (2012)
    //
    // gga_grad=2 uses one regularized local spin map:
    //   t        = rho + rho_core
    //   a        = |t|
    //   c        = min(S_eta(m), a)
    //   rho_s    = 0.5*(a +/- c)
    //   J_sA     = d rho_s / d(t,m_x,m_y,m_z)_A
    //
    // Its projected discrete gradients are defined by the same local map:
    //   g_s = sum_A J_sA G_h(x_A).
    // This is a definition of the LCA energy graph, not a discrete chain rule
    // assertion that g_s equals G_h(rho_s).
    //
    // For GGA, the sigma = |grad(rho_up)|^2 etc. are computed from gdr1,gdr2,
    // then passed to spin-polarized XC functionals (xc_spin, gcx_spin, gcc_spin).
    // The exact reverse contains -D_h(sum_s J_sA h_s) and the local response
    // of J.  It therefore does not invoke a pointwise product rule for G_h.
    //
    // gga_grad=3 instead uses the continuous B2 invariants:
    //   gamma_up/dn = 1/4 (|grad rho|^2 + sum_mu |grad m_mu|^2)
    //                 +/- 1/2 grad rho . grad|m|
    // Its potential includes both the q_n/q_m divergences and the local
    // derivative of m_hat. This is the full variational derivative of B2.
    // ======================================================================

    ModulePW::PW_Basis* rhopw = chr->rhopw;
    const int npw = rhopw->npw;
    const double e2 = ModuleBase::e2;
    constexpr double vanishing = 1e-10;
    constexpr double epsr = 1e-6;
    const double fac = 0.5;
    const bool is_gga = (XC_Functional::get_func_type() == 2 || XC_Functional::get_func_type() == 4);

    // Step 1: construct the mode-specific local spin map.  gga_grad=3 keeps
    // its existing raw-|m| data flow unchanged.
    std::vector<double> rhotmp1(nrxx), rhotmp2(nrxx), amag(nrxx);
    std::vector<double> mag_part(3 * nrxx, 0.0);
    std::vector<NcggaSpinMapPoint> spin_map;
    if (gga_grad == 2)
    {
        spin_map.resize(nrxx);
    }

    for (int ir = 0; ir < nrxx; ++ir)
    {
        const double mx = chr->rho[1][ir], my = chr->rho[2][ir], mz = chr->rho[3][ir];
        if (gga_grad == 2)
        {
            const std::array<double, 3> magnetization = {{mx, my, mz}};
            const NcggaRadialPoint radial
                = make_ncgga_radial_point(magnetization, ncgga_lca_radial_eta());
            spin_map[ir] = make_ncgga_spin_map_point(
                chr->rho[0][ir] + chr->rho_core[ir], radial);
            amag[ir] = radial.value;
            rhotmp1[ir] = spin_map[ir].spin_density[0];
            rhotmp2[ir] = spin_map[ir].spin_density[1];
        }
        else
        {
            amag[ir] = std::sqrt(mx * mx + my * my + mz * mz);
            rhotmp1[ir] = 0.5 * (chr->rho[0][ir] + amag[ir]);
            rhotmp2[ir] = 0.5 * (chr->rho[0][ir] - amag[ir]);
            if (amag[ir] > 1e-12)
            {
                mag_part[ir] = mx / amag[ir];
                mag_part[ir + nrxx] = my / amag[ir];
                mag_part[ir + 2 * nrxx] = mz / amag[ir];
            }
        }
    }
    if (gga_grad == 3)
    {
        for (int ir = 0; ir < nrxx; ++ir)
        {
            rhotmp1[ir] += fac * chr->rho_core[ir];
            rhotmp2[ir] += fac * chr->rho_core[ir];
        }
    }

    std::vector<std::complex<double>> rhogsum1(npw), tmp_recip(npw);
    rhopw->real2recip(chr->rho[0], rhogsum1.data());
    for (int ig = 0; ig < npw; ++ig)
        rhogsum1[ig] += chr->rhog_core[ig];

    std::vector<ModuleBase::Vector3<double>> gdr1(nrxx), gdr2(nrxx), grad_rho(nrxx);
    std::array<std::vector<ModuleBase::Vector3<double>>, 3> grad_m;
    for (int mu = 0; mu < 3; ++mu)
    {
        grad_m[mu].resize(nrxx);
    }
    std::vector<ModuleBase::Vector3<double>> gdr_mag(nrxx);
    XC_Functional::grad_rho(rhogsum1.data(), gdr1.data(), rhopw, tpiba);

    for (int ir = 0; ir < nrxx; ++ir)
    {
        grad_rho[ir] = gdr1[ir];
        if (gga_grad == 2)
        {
            gdr1[ir] = spin_map[ir].jacobian(0, 0) * grad_rho[ir];
            gdr2[ir] = spin_map[ir].jacobian(1, 0) * grad_rho[ir];
        }
        else
        {
            gdr1[ir] = 0.5 * grad_rho[ir];
            gdr2[ir] = 0.5 * grad_rho[ir];
        }
    }
    for (int is = 1; is <= 3; ++is)
    {
        rhopw->real2recip(chr->rho[is], tmp_recip.data());
        XC_Functional::grad_rho(tmp_recip.data(), gdr_mag.data(), rhopw, tpiba);
        grad_m[is - 1] = gdr_mag;
        for (int ir = 0; ir < nrxx; ++ir)
        {
            if (gga_grad == 2)
            {
                gdr1[ir] += spin_map[ir].jacobian(0, is) * gdr_mag[ir];
                gdr2[ir] += spin_map[ir].jacobian(1, is) * gdr_mag[ir];
            }
            else
            {
                const double mp = mag_part[ir + (is - 1) * nrxx];
                const ModuleBase::Vector3<double> g = 0.5 * gdr_mag[ir] * mp;
                gdr1[ir] += g;
                gdr2[ir] -= g;
            }
        }
    }

    // Step 3: LDA contribution.  Method 2 reverses the same spin map used to
    // construct the energy; method 3 keeps its existing raw-|m| conversion.
    double etxc = 0, vtxc = 0;
    ModuleBase::matrix v(4, nrxx);

    for (int ir = 0; ir < nrxx; ++ir)
    {
        const double arho = gga_grad == 2
                                ? spin_map[ir].absolute_density
                                : std::abs(chr->rho[0][ir] + chr->rho_core[ir]);
        if (arho <= vanishing)
            continue;

        double zeta = gga_grad == 2
                          ? spin_map[ir].clipped_magnitude / arho
                          : amag[ir] / arho;
        if (std::abs(zeta) > 1.0)
            zeta = (zeta > 0) ? 1.0 : -1.0;
        double exc = 0, vxc[2] = {0, 0};
        XC_Functional::xc_spin(arho, zeta, exc, vxc[0], vxc[1]);

        if (gga_grad == 2)
        {
            for (int channel = 0; channel < 4; ++channel)
            {
                v(channel, ir)
                    = e2 * (spin_map[ir].jacobian(0, channel) * vxc[0]
                            + spin_map[ir].jacobian(1, channel) * vxc[1]);
            }
        }
        else
        {
            v(0, ir) = e2 * 0.5 * (vxc[0] + vxc[1]);
            if (amag[ir] > vanishing)
            {
                const double vs = e2 * 0.5 * (vxc[0] - vxc[1]);
                const double inv_a = 1.0 / amag[ir];
                for (int mu = 1; mu < 4; ++mu)
                {
                    v(mu, ir) = vs * chr->rho[mu][ir] * inv_a;
                }
            }
        }
        etxc += e2 * exc * arho;
    }

    // Step 4: GGA contribution and variational divergence correction.
    // Method 2 keeps the projected gdr1/gdr2 path; method 3 uses continuous B2.
    if (is_gga)
    {
        double etxcgc = 0;
        std::vector<double> vup_gga(nrxx, 0), vdw_gga(nrxx, 0);
        std::vector<ModuleBase::Vector3<double>> h1(nrxx), h2(nrxx);
        std::vector<ModuleBase::Vector3<double>> q_n(nrxx);
        std::array<std::vector<ModuleBase::Vector3<double>>, 3> q_m;
        std::array<std::vector<double>, 3> local_m;
        for (int mu = 0; mu < 3; ++mu)
        {
            q_m[mu].resize(nrxx);
            local_m[mu].resize(nrxx, 0.0);
        }

        for (int ir = 0; ir < nrxx; ++ir)
        {
            double sx = 0, v1xup = 0, v1xdw = 0, v2xup = 0, v2xdw = 0;
            double sc = 0, v1cup = 0, v1cdw = 0, v2c = 0;
            double grho2a = gdr1[ir] * gdr1[ir];
            double grho2b = gdr2[ir] * gdr2[ir];
            ContinuousGradientPoint point;
            if (gga_grad == 3)
            {
                const std::array<ModuleBase::Vector3<double>, 3> grad_m_point
                    = {{grad_m[0][ir], grad_m[1][ir], grad_m[2][ir]}};
                const std::array<double, 3> magnetization
                    = {{chr->rho[1][ir], chr->rho[2][ir], chr->rho[3][ir]}};
                point = make_continuous_gradient_point(grad_rho[ir], grad_m_point, magnetization);
                grho2a = point.gamma_up;
                grho2b = point.gamma_down;
            }
            const double rh = rhotmp1[ir] + rhotmp2[ir];

            XC_Functional::gcx_spin(rhotmp1[ir], rhotmp2[ir], grho2a, grho2b,
                                    sx, v1xup, v1xdw, v2xup, v2xdw);

            if (rh > epsr)
            {
                const double zeta_input
                    = std::fabs((rhotmp1[ir] - rhotmp2[ir]) / rh);
                double zeta = zeta_input;
                const double grh2 = (gdr1[ir] + gdr2[ir]) * (gdr1[ir] + gdr2[ir]);
                XC_Functional::gcc_spin(rh, zeta, grh2, sc, v1cup, v1cdw, v2c);
                if (gga_grad == 2 && zeta_input > 1.0 - epsr)
                {
                    // gcc_spin evaluates this branch at a fixed clipped zeta.
                    // Reverse that actual branch instead of differentiating
                    // through the discarded input polarization.
                    const double fixed_zeta_density_derivative
                        = 0.5 * ((1.0 + zeta) * v1cup
                                 + (1.0 - zeta) * v1cdw);
                    v1cup = fixed_zeta_density_derivative;
                    v1cdw = fixed_zeta_density_derivative;
                }
            }

            vup_gga[ir] = e2 * (v1xup + v1cup);
            vdw_gga[ir] = e2 * (v1xdw + v1cdw);

            if (gga_grad == 2)
            {
                const double v2cup = v2c, v2cdw = v2c, v2cud = v2c;
                h1[ir] = e2 * ((v2xup + v2cup) * gdr1[ir] + v2cud * gdr2[ir]);
                h2[ir] = e2 * ((v2xdw + v2cdw) * gdr2[ir] + v2cud * gdr1[ir]);
            }
            else
            {
                const double d_up = 0.5 * e2 * v2xup;
                const double d_down = 0.5 * e2 * v2xdw;
                q_n[ir] = d_up * point.dgamma_up_dn
                          + d_down * point.dgamma_down_dn
                          + e2 * v2c * grad_rho[ir];
                for (int mu = 0; mu < 3; ++mu)
                {
                    q_m[mu][ir] = d_up * point.dgamma_up_dm[mu]
                                  + d_down * point.dgamma_down_dm[mu];
                    local_m[mu][ir]
                        = 0.5 * (d_up - d_down) * point.dgamma_difference_dm[mu];
                }
            }

            etxcgc += e2 * (sx + sc);
        }

        for (int ir = 0; ir < nrxx; ++ir)
        {
            if (gga_grad == 2)
            {
                for (int channel = 0; channel < 4; ++channel)
                {
                    v(channel, ir)
                        += spin_map[ir].jacobian(0, channel) * vup_gga[ir]
                           + spin_map[ir].jacobian(1, channel) * vdw_gga[ir];
                }
            }
            else
            {
                v(0, ir) += 0.5 * (vup_gga[ir] + vdw_gga[ir]);
                const double vdiff = 0.5 * (vup_gga[ir] - vdw_gga[ir]);
                for (int mu = 1; mu < 4; ++mu)
                {
                    v(mu, ir) += vdiff * mag_part[ir + (mu - 1) * nrxx];
                    v(mu, ir) += local_m[mu - 1][ir];
                }
            }
        }

        std::vector<double> dh(nrxx);
        std::vector<ModuleBase::Vector3<double>> tmp_h(nrxx);

        if (gga_grad == 2)
        {
            // Exact reverse of g_s=sum_A J_sA G_h(x_A):
            //   v_B = -D_h(sum_s J_sB h_s)
            //         + sum_s,A dJ_sA/dx_B h_s.G_h(x_A).
            for (int channel = 0; channel < 4; ++channel)
            {
                for (int ir = 0; ir < nrxx; ++ir)
                {
                    tmp_h[ir]
                        = spin_map[ir].jacobian(0, channel) * h1[ir]
                          + spin_map[ir].jacobian(1, channel) * h2[ir];
                }
                XC_Functional::grad_dot(tmp_h.data(), dh.data(), rhopw, tpiba);
                for (int ir = 0; ir < nrxx; ++ir)
                {
                    v(channel, ir) -= dh[ir];
                    if (channel == 0 || spin_map[ir].saturated)
                    {
                        continue;
                    }
                    const ModuleBase::Vector3<double> spin_flux
                        = 0.5 * (h1[ir] - h2[ir]);
                    double local_response = 0.0;
                    for (int nu = 0; nu < 3; ++nu)
                    {
                        local_response
                            += spin_map[ir].radial.jacobian(nu, channel - 1)
                               * (spin_flux * grad_m[nu][ir]);
                    }
                    v(channel, ir) += local_response;
                }
            }
        }
        else // gga_grad == 3
        {
            // Total-density channel.
            for (int ir = 0; ir < nrxx; ++ir)
            {
                tmp_h[ir] = q_n[ir];
            }
            XC_Functional::grad_dot(tmp_h.data(), dh.data(), rhopw, tpiba);
            for (int ir = 0; ir < nrxx; ++ir)
            {
                v(0, ir) -= dh[ir];
            }

            // Continuous B2 magnetic response: v_mu -= div(q_m_mu), plus
            // the local direction-response term added above.
            for (int mu = 1; mu < 4; ++mu)
            {
                for (int ir = 0; ir < nrxx; ++ir)
                    tmp_h[ir] = q_m[mu - 1][ir];
                XC_Functional::grad_dot(tmp_h.data(), dh.data(), rhopw, tpiba);
                for (int ir = 0; ir < nrxx; ++ir)
                    v(mu, ir) -= dh[ir];
            }
        }

        etxc += etxcgc;
    }

    // vtxc uses the same completed four-component potential returned to the
    // caller.  This unifies the bookkeeping for both modes.
    vtxc = 0.0;
    for (int ir = 0; ir < nrxx; ++ir)
    {
        for (int is = 0; is < 4; ++is)
        {
            vtxc += v(is, ir) * chr->rho[is][ir];
        }
    }

#ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc);
    Parallel_Reduce::reduce_pool(vtxc);
#endif
    etxc *= omega / rhopw->nxyz;
    vtxc *= omega / rhopw->nxyz;

    ModuleBase::timer::end("XC_Functional", "v_xc_ncgga_sf_builtin");
    return std::make_tuple(etxc, vtxc, std::move(v));
}

void gradcorr_ncgga_lca_builtin(const Charge* const chr,
                                ModulePW::PW_Basis* rhopw,
                                const double tpiba,
                                std::vector<double>& stress_gga)
{
    stress_gga.assign(9, 0.0);

    const int nrxx = rhopw->nrxx;
    const int npw = rhopw->npw;
    const double e2 = ModuleBase::e2;
    constexpr double epsr = 1.0e-6;

    // Rebuild the same complete local map used by the gga_grad=2 energy:
    //   rho_s = N_s(n + rho_core, m),
    //   g_s   = sum_A dN_s/dx_A G_h x_A.
    // Metric differentiation keeps the real-grid values x_A fixed, so the
    // map Jacobian is unchanged and every G_h x_A transforms covariantly.
    std::vector<NcggaSpinMapPoint> spin_map(nrxx);
    std::array<std::vector<ModuleBase::Vector3<double>>, 4> field_gradient;
    for (int channel = 0; channel < 4; ++channel)
    {
        field_gradient[channel].resize(nrxx);
    }
    std::array<std::vector<ModuleBase::Vector3<double>>, 2> spin_gradient;
    for (int spin = 0; spin < 2; ++spin)
    {
        spin_gradient[spin].resize(nrxx);
    }

    std::vector<std::complex<double>> reciprocal(npw);
    rhopw->real2recip(chr->rho[0], reciprocal.data());
    for (int ig = 0; ig < npw; ++ig)
    {
        reciprocal[ig] += chr->rhog_core[ig];
    }
    XC_Functional::grad_rho(
        reciprocal.data(), field_gradient[0].data(), rhopw, tpiba);

    for (int channel = 1; channel < 4; ++channel)
    {
        rhopw->real2recip(chr->rho[channel], reciprocal.data());
        XC_Functional::grad_rho(
            reciprocal.data(), field_gradient[channel].data(), rhopw, tpiba);
    }

    for (int ir = 0; ir < nrxx; ++ir)
    {
        const std::array<double, 3> magnetization
            = {{chr->rho[1][ir], chr->rho[2][ir], chr->rho[3][ir]}};
        spin_map[ir] = make_ncgga_spin_map_point(
            chr->rho[0][ir] + chr->rho_core[ir],
            make_ncgga_radial_point(magnetization, ncgga_lca_radial_eta()));
        for (int spin = 0; spin < 2; ++spin)
        {
            for (int channel = 0; channel < 4; ++channel)
            {
                spin_gradient[spin][ir]
                    += spin_map[ir].jacobian(spin, channel)
                       * field_gradient[channel][ir];
            }
        }
    }

    for (int ir = 0; ir < nrxx; ++ir)
    {
        const double rho_up = spin_map[ir].spin_density[0];
        const double rho_down = spin_map[ir].spin_density[1];
        const ModuleBase::Vector3<double>& grad_up = spin_gradient[0][ir];
        const ModuleBase::Vector3<double>& grad_down = spin_gradient[1][ir];

        double sx = 0.0;
        double v1xup = 0.0;
        double v1xdw = 0.0;
        double v2xup = 0.0;
        double v2xdw = 0.0;
        XC_Functional::gcx_spin(rho_up,
                                rho_down,
                                grad_up * grad_up,
                                grad_down * grad_down,
                                sx,
                                v1xup,
                                v1xdw,
                                v2xup,
                                v2xdw);

        double sc = 0.0;
        double v1cup = 0.0;
        double v1cdw = 0.0;
        double v2c = 0.0;
        const double rho = rho_up + rho_down;
        if (rho > epsr)
        {
            double zeta
                = std::fabs((rho_up - rho_down) / rho);
            const ModuleBase::Vector3<double> grad_rho = grad_up + grad_down;
            XC_Functional::gcc_spin(rho,
                                    zeta,
                                    grad_rho * grad_rho,
                                    sc,
                                    v1cup,
                                    v1cdw,
                                    v2c);
        }

        const ModuleBase::Vector3<double> h_up
            = e2 * ((v2xup + v2c) * grad_up + v2c * grad_down);
        const ModuleBase::Vector3<double> h_down
            = e2 * ((v2xdw + v2c) * grad_down + v2c * grad_up);
        const double grad_up_component[3]
            = {grad_up.x, grad_up.y, grad_up.z};
        const double grad_down_component[3]
            = {grad_down.x, grad_down.y, grad_down.z};
        const double h_up_component[3] = {h_up.x, h_up.y, h_up.z};
        const double h_down_component[3]
            = {h_down.x, h_down.y, h_down.z};
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column <= row; ++column)
            {
                stress_gga[row * 3 + column]
                    += h_up_component[row] * grad_up_component[column]
                       + h_down_component[row] * grad_down_component[column];
            }
        }
    }
}

void gradcorr_ncgga_sf_builtin(const Charge* const chr, ModulePW::PW_Basis* rhopw,
                                const UnitCell* ucell, std::vector<double>& stress_gga)
{
    stress_gga.assign(9, 0.0);

    const int nrxx = rhopw->nrxx;
    const int npw = rhopw->npw;
    const double e2 = ModuleBase::e2;
    constexpr double epsr = 1.0e-6;
    const double fac = 0.5;

    std::vector<double> rhotmp1(nrxx), rhotmp2(nrxx), amag(nrxx);
    for (int ir = 0; ir < nrxx; ++ir)
    {
        const double mx = chr->rho[1][ir], my = chr->rho[2][ir], mz = chr->rho[3][ir];
        amag[ir] = std::sqrt(mx * mx + my * my + mz * mz);
        rhotmp1[ir] = 0.5 * (chr->rho[0][ir] + amag[ir]);
        rhotmp2[ir] = 0.5 * (chr->rho[0][ir] - amag[ir]);
    }
    for (int ir = 0; ir < nrxx; ++ir)
    {
        rhotmp1[ir] += fac * chr->rho_core[ir];
        rhotmp2[ir] += fac * chr->rho_core[ir];
    }

    std::vector<std::complex<double>> rhogsum1(npw), tmp_recip(npw);
    rhopw->real2recip(chr->rho[0], rhogsum1.data());
    for (int ig = 0; ig < npw; ++ig)
        rhogsum1[ig] += chr->rhog_core[ig];

    std::vector<ModuleBase::Vector3<double>> grad_n(nrxx);
    std::array<std::vector<ModuleBase::Vector3<double>>, 3> grad_m;
    for (int mu = 0; mu < 3; ++mu)
    {
        grad_m[mu].resize(nrxx);
    }
    XC_Functional::grad_rho(rhogsum1.data(), grad_n.data(), rhopw, ucell->tpiba);
    for (int is = 1; is <= 3; ++is)
    {
        rhopw->real2recip(chr->rho[is], tmp_recip.data());
        XC_Functional::grad_rho(tmp_recip.data(), grad_m[is - 1].data(), rhopw, ucell->tpiba);
    }

    for (int ir = 0; ir < nrxx; ++ir)
    {
        double sx = 0, v1xup = 0, v1xdw = 0, v2xup = 0, v2xdw = 0;
        double sc = 0, v1cup = 0, v1cdw = 0, v2c = 0;
        const std::array<ModuleBase::Vector3<double>, 3> grad_m_point
            = {{grad_m[0][ir], grad_m[1][ir], grad_m[2][ir]}};
        const std::array<double, 3> magnetization
            = {{chr->rho[1][ir], chr->rho[2][ir], chr->rho[3][ir]}};
        const ContinuousGradientPoint point
            = make_continuous_gradient_point(grad_n[ir], grad_m_point, magnetization);
        const double rh = rhotmp1[ir] + rhotmp2[ir];

        XC_Functional::gcx_spin(rhotmp1[ir], rhotmp2[ir],
                                point.gamma_up,
                                point.gamma_down,
                                sx, v1xup, v1xdw, v2xup, v2xdw);

        if (rh > epsr)
        {
            double zeta = (rhotmp1[ir] - rhotmp2[ir]) / rh;
            zeta = std::fabs(zeta);
            const double grh2 = grad_n[ir] * grad_n[ir];
            XC_Functional::gcc_spin(rh, zeta, grh2, sc, v1cup, v1cdw, v2c);
        }

        const double d_up = 0.5 * e2 * v2xup;
        const double d_down = 0.5 * e2 * v2xdw;
        const ModuleBase::Vector3<double> q_n
            = d_up * point.dgamma_up_dn
              + d_down * point.dgamma_down_dn
              + e2 * v2c * grad_n[ir];
        std::array<ModuleBase::Vector3<double>, 3> q_m;
        for (int mu = 0; mu < 3; ++mu)
        {
            q_m[mu] = d_up * point.dgamma_up_dm[mu]
                      + d_down * point.dgamma_down_dm[mu];
        }

        const double grad_n_components[3] = {grad_n[ir].x, grad_n[ir].y, grad_n[ir].z};
        const double q_n_components[3] = {q_n.x, q_n.y, q_n.z};
        const double grad_m_components[3][3] = {
            {grad_m[0][ir].x, grad_m[0][ir].y, grad_m[0][ir].z},
            {grad_m[1][ir].x, grad_m[1][ir].y, grad_m[1][ir].z},
            {grad_m[2][ir].x, grad_m[2][ir].y, grad_m[2][ir].z}};
        const double q_m_components[3][3] = {
            {q_m[0].x, q_m[0].y, q_m[0].z},
            {q_m[1].x, q_m[1].y, q_m[1].z},
            {q_m[2].x, q_m[2].y, q_m[2].z}};
        for (int l = 0; l < 3; ++l)
        {
            for (int m = 0; m <= l; ++m)
            {
                const int ind = l * 3 + m;
                stress_gga[ind] += q_n_components[l] * grad_n_components[m];
                for (int mu = 0; mu < 3; ++mu)
                {
                    stress_gga[ind] += q_m_components[mu][l] * grad_m_components[mu][m];
                }
            }
        }
    }
}

} // namespace NCGGA_SF_Builtin
} // namespace ModuleXC
