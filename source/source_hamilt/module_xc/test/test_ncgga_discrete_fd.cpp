#include "../xc_functional.h"
#include "../xc_functional_ncgga_sf.h"
#include "../libxc_abacus.h"

#include "source_base/constants.h"
#include "source_base/matrix3.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_estate/module_charge/charge.h"

#ifdef __MPI
#include "source_base/parallel_comm.h"
#include <mpi.h>
#endif

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

// This focused target does not link the full elecstate object library. The
// fixture supplies vector-backed charge storage, so only the trivial lifetime
// boundary is needed here.
Charge::Charge() {}
Charge::~Charge() {}

namespace
{

double pool_sum(const double local)
{
#ifdef __MPI
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, POOL_WORLD);
    return global;
#else
    return local;
#endif
}

double pool_min(const double local)
{
#ifdef __MPI
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MIN, POOL_WORLD);
    return global;
#else
    return local;
#endif
}

double pool_max(const double local)
{
#ifdef __MPI
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, POOL_WORLD);
    return global;
#else
    return local;
#endif
}

bool is_pool_root()
{
#ifdef __MPI
    int rank = 0;
    MPI_Comm_rank(POOL_WORLD, &rank);
    return rank == 0;
#else
    return true;
#endif
}

class RealPwNcgga : public testing::Test
{
  protected:
    ModulePW::PW_Basis pw;
    std::array<std::vector<double>, 4> perturbation;

    void SetUp() override
    {
#ifdef __MPI
        int rank = 0;
        int size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        pw.initmpi(size, rank, MPI_COMM_WORLD);
#endif
        // Keep tpiba away from one so an omitted or duplicated reciprocal-
        // length factor cannot accidentally pass the adjoint test.
        const double lat0 = 7.0;
        const ModuleBase::Matrix3 lattice(1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0);
        // An odd z dimension gives unequal real-space slabs in the MPI2 test.
        pw.initgrids(lat0, lattice, 24, 10, 9);
        pw.initparameters(false, 80.0, 2, false);
        pw.setuptransform();
        pw.collect_local_pw();

        ASSERT_EQ(pw.nx, 24);
        ASSERT_EQ(pw.ny, 10);
        ASSERT_EQ(pw.nz, 9);
        ASSERT_EQ(pw.nxyz, 2160);
        ASSERT_GT(pw.npwtot, 100);
        ASSERT_FALSE(pw.gamma_only);
        ASSERT_NEAR(pw.tpiba, ModuleBase::TWO_PI / lat0, 1.0e-14);
        ASSERT_GT(std::abs(pw.tpiba - 1.0), 5.0e-2);

        for (int channel = 0; channel < 4; ++channel)
        {
            perturbation[channel].resize(pw.nrxx);
        }
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            // PW_Basis stores local real data as
            // ir = iz_local + (iy + ix * ny) * nplane.
            const int ix = ir / (pw.ny * pw.nplane);
            const int iy = (ir / pw.nplane) % pw.ny;
            const int iz = ir % pw.nplane + pw.startz_current;
            const double x = ModuleBase::TWO_PI * static_cast<double>(ix) / pw.nx;
            const double y = ModuleBase::TWO_PI * static_cast<double>(iy) / pw.ny;
            const double z = ModuleBase::TWO_PI * static_cast<double>(iz) / pw.nz;
            perturbation[0][ir] = 0.31 * std::cos(2.0 * x + 0.41)
                                      - 0.19 * std::sin(7.0 * x - 0.12)
                                      + 0.11 * std::cos(y + z - 0.16);
            perturbation[1][ir] = 0.27 * std::sin(x + 0.37)
                                      + 0.21 * std::cos(8.0 * x + 0.19)
                                      + 0.13 * std::sin(y - 0.24);
            perturbation[2][ir] = 0.29 * std::cos(3.0 * x - 0.22)
                                      - 0.17 * std::sin(6.0 * x + 0.31)
                                      + 0.12 * std::cos(z + 0.28);
            perturbation[3][ir] = 0.25 * std::sin(5.0 * x + 0.18)
                                      + 0.23 * std::cos(7.0 * x - 0.29)
                                      + 0.10 * std::sin(y - z + 0.32);
        }
    }
};

TEST_F(RealPwNcgga, PerturbationsSurviveThePwCutoff)
{
    for (int channel = 0; channel < 4; ++channel)
    {
        std::vector<std::complex<double>> reciprocal(pw.npw);
        pw.real2recip(perturbation[channel].data(), reciprocal.data());
        double local_norm2 = 0.0;
        for (int ig = 0; ig < pw.npw; ++ig)
        {
            local_norm2 += std::norm(reciprocal[ig]);
        }
        EXPECT_GT(pool_sum(local_norm2), 1.0e-4);
    }
}

TEST_F(RealPwNcgga, GradAndDivAreNegativeAdjoints)
{
    std::vector<double> scalar(pw.nrxx);
    std::vector<ModuleBase::Vector3<double>> vector_field(pw.nrxx);
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        const int ix = ir / (pw.ny * pw.nplane);
        const int iy = (ir / pw.nplane) % pw.ny;
        const int iz = ir % pw.nplane + pw.startz_current;
        const double x = ModuleBase::TWO_PI * static_cast<double>(ix) / pw.nx;
        const double y = ModuleBase::TWO_PI * static_cast<double>(iy) / pw.ny;
        const double z = ModuleBase::TWO_PI * static_cast<double>(iz) / pw.nz;
        scalar[ir] = 0.4 * std::sin(3.0 * x + 0.2)
                     - 0.3 * std::cos(8.0 * x - 0.1)
                     + 0.2 * std::sin(y - z + 0.4);
        vector_field[ir].x = 0.7 * std::cos(2.0 * x + 0.3)
                             + 0.2 * std::sin(7.0 * x);
        vector_field[ir].y = 0.3 * std::sin(y - 0.4)
                             + 0.1 * std::cos(x + z);
        vector_field[ir].z = -0.25 * std::cos(z + 0.1)
                             + 0.08 * std::sin(x - y);
    }

    std::vector<std::complex<double>> reciprocal(pw.npw);
    std::vector<ModuleBase::Vector3<double>> gradient(pw.nrxx);
    std::vector<double> divergence(pw.nrxx);
    pw.real2recip(scalar.data(), reciprocal.data());
    XC_Functional::grad_rho(reciprocal.data(), gradient.data(), &pw, pw.tpiba);
    XC_Functional::grad_dot(vector_field.data(), divergence.data(), &pw, pw.tpiba);

    double local_identity = 0.0;
    double local_norm = 0.0;
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        const double left = vector_field[ir] * gradient[ir];
        const double right = divergence[ir] * scalar[ir];
        local_identity += left + right;
        local_norm += std::abs(left) + std::abs(right);
    }
    const double identity = pw.omega / pw.nxyz * pool_sum(local_identity);
    const double norm = pw.omega / pw.nxyz * pool_sum(local_norm);
    EXPECT_GT(norm, 1.0e-4);
    EXPECT_LE(std::abs(identity), 5.0e-11 * std::max(1.0, norm));
    if (is_pool_root())
    {
        std::cout << std::setprecision(17)
                  << "NCGGA_ADJOINT identity=" << identity
                  << " norm=" << norm
                  << " scaled_error=" << std::abs(identity) / norm << '\n';
    }
}

#ifdef __LIBXC
TEST_F(RealPwNcgga, LibxcOrdinaryGgaSigmaFloorDifferentiatesTheWeightedEnergy)
{
    const int functional_id = XC_GGA_X_HJS_B88_V2;
    xc_func_type threshold_probe;
    ASSERT_EQ(xc_func_init(&threshold_probe, functional_id, XC_UNPOLARIZED), 0);
    const double density_floor = threshold_probe.dens_threshold;
    const double sigma_floor
        = threshold_probe.sigma_threshold * threshold_probe.sigma_threshold;
    xc_func_end(&threshold_probe);
    ASSERT_GT(density_floor, 0.0);
    ASSERT_GT(sigma_floor, 0.0);

    std::vector<double> density(pw.nrxx);
    std::vector<double> direction(pw.nrxx);
    std::array<double*, 1> density_ptr = {{density.data()}};
    std::vector<double> core_density(pw.nrxx, 0.0);
    std::vector<std::complex<double>> core_density_reciprocal(pw.npw, 0.0);

    const double mean_density = 2.0e-6;
    const double base_gradient_amplitude = 0.45 * std::sqrt(sigma_floor);
    const double density_amplitude = base_gradient_amplitude / pw.tpiba;
    const double steps[] = {4.0e-10, 2.0e-10, 1.0e-10};
    ASSERT_GT(mean_density - density_amplitude - steps[0],
              1.10 * density_floor);
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        const int ix = ir / (pw.ny * pw.nplane);
        const double x = ModuleBase::TWO_PI * static_cast<double>(ix) / pw.nx;
        direction[ir] = std::cos(x + 0.23);
        density[ir] = mean_density + density_amplitude * direction[ir];
    }

    Charge charge;
    charge.rhopw = &pw;
    charge.nrxx = pw.nrxx;
    charge.nxyz = pw.nxyz;
    charge.ngmc = pw.npw;
    charge.nspin = 1;
    charge.rho = density_ptr.data();
    charge.rho_core = core_density.data();
    charge.rhog_core = core_density_reciprocal.data();

    const std::vector<int> functionals = {functional_id};
    const auto evaluate = [&]()
    {
        return XC_Functional_Libxc::v_xc_libxc(functionals,
                                               pw.nrxx,
                                               pw.omega,
                                               pw.tpiba,
                                               &charge,
                                               1,
                                               false,
                                               false,
                                               0,
                                               nullptr,
                                               0.0,
                                               0.0);
    };

    const auto maximum_sigma = [&]()
    {
        std::vector<std::complex<double>> reciprocal(pw.npw);
        std::vector<ModuleBase::Vector3<double>> gradient(pw.nrxx);
        pw.real2recip(density.data(), reciprocal.data());
        XC_Functional::grad_rho(reciprocal.data(), gradient.data(), &pw, pw.tpiba);
        double local_maximum = 0.0;
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            local_maximum = std::max(local_maximum, gradient[ir] * gradient[ir]);
        }
        return pool_max(local_maximum);
    };

    const auto reference = evaluate();
    double local_analytic = 0.0;
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        local_analytic += std::get<2>(reference)(0, ir) * direction[ir];
    }
    const double analytic
        = pw.omega / pw.nxyz * pool_sum(local_analytic);

    const double original_amplitude = density_amplitude;
    std::array<double, 3> errors = {{0.0, 0.0, 0.0}};
    for (std::size_t ieps = 0; ieps < 3; ++ieps)
    {
        const double step = steps[ieps];
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            density[ir] = mean_density + (original_amplitude + step) * direction[ir];
        }
        const double sigma_plus = maximum_sigma();
        const double energy_plus = std::get<0>(evaluate());
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            density[ir] = mean_density + (original_amplitude - step) * direction[ir];
        }
        const double sigma_minus = maximum_sigma();
        const double energy_minus = std::get<0>(evaluate());
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            density[ir] = mean_density + original_amplitude * direction[ir];
        }

        const double finite_difference = (energy_plus - energy_minus) / (2.0 * step);
        errors[ieps] = std::abs(analytic - finite_difference);
        EXPECT_LT(std::max(sigma_plus, sigma_minus), 0.60 * sigma_floor);
        EXPECT_GT(std::min(sigma_plus, sigma_minus), 0.10 * sigma_floor);
        EXPECT_NEAR(analytic, finite_difference, 2.0e-7)
            << "step=" << step;
        if (is_pool_root())
        {
            std::cout << std::setprecision(17)
                      << "LIBXC_SIGMA_FLOOR_FD eps=" << step
                      << " sigma_floor=" << sigma_floor
                      << " max_sigma=" << std::max(sigma_plus, sigma_minus)
                      << " energy=" << std::get<0>(reference)
                      << " energy_plus=" << energy_plus
                      << " energy_minus=" << energy_minus
                      << " analytic=" << analytic
                      << " finite_difference=" << finite_difference
                      << " absolute_error=" << errors[ieps] << '\n';
        }
    }
    EXPECT_LE(errors[1], 0.40 * errors[0] + 1.0e-9);
}
#endif

TEST_F(RealPwNcgga, BuiltinContinuousB2HasNondegenerateTransverseEnergy)
{
    std::array<std::vector<double>, 4> density;
    std::array<double*, 4> density_ptr;
    for (int channel = 0; channel < 4; ++channel)
    {
        density[channel].resize(pw.nrxx);
        density_ptr[channel] = density[channel].data();
    }
    std::vector<double> core_density(pw.nrxx, 0.0);
    std::vector<std::complex<double>> core_density_reciprocal(pw.npw, 0.0);

    double local_min_magnitude = 1.0e100;
    double local_min_spin_gap = 1.0e100;
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        const int ix = ir / (pw.ny * pw.nplane);
        const int iy = (ir / pw.nplane) % pw.ny;
        const int iz = ir % pw.nplane + pw.startz_current;
        const double x = ModuleBase::TWO_PI * static_cast<double>(ix) / pw.nx;
        const double y = ModuleBase::TWO_PI * static_cast<double>(iy) / pw.ny;
        const double z = ModuleBase::TWO_PI * static_cast<double>(iz) / pw.nz;
        const double magnitude = 0.42 + 0.04 * std::cos(2.0 * x - y + 0.17);
        const double polar_angle = 0.90;
        const double azimuth = x + 2.0 * y + z + 0.23;
        const double sin_polar = std::sin(polar_angle);

        density[0][ir] = 1.60 + 0.12 * std::cos(x - y + 0.31)
                               + 0.08 * std::sin(2.0 * z - 0.27);
        density[1][ir] = magnitude * sin_polar * std::cos(azimuth);
        density[2][ir] = magnitude * sin_polar * std::sin(azimuth);
        density[3][ir] = magnitude * std::cos(polar_angle);
        local_min_magnitude = std::min(local_min_magnitude, magnitude);
        local_min_spin_gap
            = std::min(local_min_spin_gap, density[0][ir] - magnitude);
    }

    Charge charge;
    charge.rhopw = &pw;
    charge.nrxx = pw.nrxx;
    charge.nxyz = pw.nxyz;
    charge.ngmc = pw.npw;
    charge.nspin = 4;
    charge.rho = density_ptr.data();
    charge.rho_core = core_density.data();
    charge.rhog_core = core_density_reciprocal.data();

    std::array<std::vector<ModuleBase::Vector3<double>>, 3> grad_m;
    std::vector<std::complex<double>> reciprocal(pw.npw);
    for (int mu = 0; mu < 3; ++mu)
    {
        grad_m[mu].resize(pw.nrxx);
        pw.real2recip(density[mu + 1].data(), reciprocal.data());
        XC_Functional::grad_rho(reciprocal.data(), grad_m[mu].data(), &pw, pw.tpiba);
    }
    double local_transverse_power = 0.0;
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        const double magnitude = std::sqrt(density[1][ir] * density[1][ir]
                                           + density[2][ir] * density[2][ir]
                                           + density[3][ir] * density[3][ir]);
        ModuleBase::Vector3<double> projected_gradient;
        double component_power = 0.0;
        for (int mu = 0; mu < 3; ++mu)
        {
            component_power += grad_m[mu][ir] * grad_m[mu][ir];
            projected_gradient
                += density[mu + 1][ir] / magnitude * grad_m[mu][ir];
        }
        local_transverse_power
            += component_power - projected_gradient * projected_gradient;
    }
    const double transverse_power
        = pw.omega / pw.nxyz * pool_sum(local_transverse_power);
    const double min_magnitude = pool_min(local_min_magnitude);
    const double min_spin_gap = pool_min(local_min_spin_gap);

    XC_Functional::set_xc_type("PBE");
    const auto projected = ModuleXC::NCGGA_SF_Builtin::v_xc_ncgga_sf_builtin(
        pw.nrxx, pw.omega, pw.tpiba, &charge, 2);
    const auto continuous_b2 = ModuleXC::NCGGA_SF_Builtin::v_xc_ncgga_sf_builtin(
        pw.nrxx, pw.omega, pw.tpiba, &charge, 3);
    const double energy_projected = std::get<0>(projected);
    const double energy_b2 = std::get<0>(continuous_b2);
    const double energy_difference = energy_b2 - energy_projected;
    const double energy_scale
        = std::max(1.0, std::max(std::abs(energy_projected), std::abs(energy_b2)));

    EXPECT_GT(min_magnitude, 0.30);
    EXPECT_GT(min_spin_gap, 0.90);
    EXPECT_GT(transverse_power, 1.0e-4);
    EXPECT_GT(std::abs(energy_difference), 1.0e-8 * energy_scale);
    if (is_pool_root())
    {
        std::cout << std::setprecision(17)
                  << "NCGGA_B2_NONDEGENERACY energy_gga2=" << energy_projected
                  << " energy_gga3=" << energy_b2
                  << " delta=" << energy_difference
                  << " transverse_power=" << transverse_power
                  << " min_magnitude=" << min_magnitude
                  << " min_spin_gap=" << min_spin_gap << '\n';
    }
}

} // namespace

#ifdef __MPI
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    POOL_WORLD = MPI_COMM_WORLD;
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
#endif
