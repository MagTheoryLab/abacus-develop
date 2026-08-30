#include "../xc_functional.h"

#include "source_base/constants.h"
#include "source_base/matrix3.h"
#include "source_basis/module_pw/pw_basis.h"

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
