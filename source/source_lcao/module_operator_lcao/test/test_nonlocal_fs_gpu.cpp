#include "../nonlocal_fs_gpu.h"
#include <gtest/gtest.h>
#include <complex>
#include <vector>

namespace
{
template <typename T>
T density_value(int i)
{
    return T(0.01 * (i % 17 - 8));
}
template <>
std::complex<double> density_value(int i)
{
    return {0.01 * (i % 17 - 8), 0.02 * (i % 11 - 5)};
}
template <typename T>
void check(int npol, bool dense)
{
    using namespace hamilt::nonlocal_gpu;
    Batch batch;
    Task task = {};
    task.rows = 3;
    task.cols = 5;
    task.projectors = dense ? 5 : 4;
    task.npol = npol;
    task.center = 0;
    task.atom = 1;
    task.right = task.rows * 4 * task.projectors;
    for (int a = 0; a < 3; ++a)
    {
        task.dis1[a] = 0.2 * (a + 1);
        task.dis2[a] = -0.3 * (a + 2);
    }
    batch.projections.resize((task.rows + task.cols) * 4 * task.projectors);
    for (std::size_t i = 0; i < batch.projections.size(); ++i)
        batch.projections[i] = 0.03 * (static_cast<int>(i % 23) - 11);
    for (int spin = 0; spin < npol * npol; ++spin)
    {
        for (int p = 0; p < (dense ? task.projectors * task.projectors : task.projectors); ++p)
        {
            Coupling d = {};
            d.p1 = dense ? p / task.projectors : p;
            d.p2 = dense ? p % task.projectors : (p + spin + 1) % task.projectors;
            d.spin = spin;
            d.real = 0.2 * (p + 1);
            d.imag = npol == 2 && !dense ? 0.1 * (spin - p) : 0.0;
            batch.couplings.push_back(d);
        }
    }
    task.coupling_count = batch.couplings.size();
    batch.tasks.push_back(task);
    task.atom = 2;
    batch.tasks.push_back(task);
    std::vector<T> dm(task.rows * task.cols * npol * npol);
    for (std::size_t i = 0; i < dm.size(); ++i) dm[i] = density_value<T>(i);
    for (int mode = 1; mode <= 3; ++mode)
    {
        const bool force_on = mode & 1;
        const bool stress_on = mode & 2;
        std::vector<double> expected_force(9, 0.0);
        std::vector<double> expected_stress(6, 0.0);
        // Independent scalar oracle: first sum projector derivatives, then
        // contract the spin DMR, as in the original CPU implementation.
        for (const auto& t : batch.tasks)
        {
            for (int row = 0; row < t.rows; ++row)
            {
                const double* l = batch.projections.data() + t.left + row * 4 * t.projectors;
                for (int col = 0; col < t.cols; ++col)
                {
                    const double* r = batch.projections.data() + t.right + col * 4 * t.projectors;
                    for (int spin = 0; spin < npol * npol; ++spin)
                    {
                        std::complex<double> f[3] = {};
                        std::complex<double> s[6] = {};
                        for (const auto& d : batch.couplings)
                        {
                            if (d.spin != spin) continue;
                            const std::complex<double> coefficient(d.real, d.imag);
                            for (int a = 0; a < 3; ++a)
                                f[a] += l[d.p1 + (a + 1) * t.projectors] * r[d.p2] * coefficient;
                            int c = 0;
                            for (int a = 0; a < 3; ++a)
                            {
                                for (int b = a; b < 3; ++b)
                                {
                                    s[c++] += (l[d.p1 + (a + 1) * t.projectors] * t.dis1[b] * r[d.p2]
                                        + l[d.p1] * r[d.p2 + (a + 1) * t.projectors] * t.dis2[b]) * coefficient;
                                }
                            }
                        }
                        const T density = dm[(row * npol + spin / npol) * t.cols * npol + col * npol + spin % npol];
                        if (force_on)
                        {
                            for (int a = 0; a < 3; ++a)
                            {
                                expected_force[t.atom * 3 + a] += std::real(density * f[a]);
                                expected_force[t.center * 3 + a] -= std::real(density * f[a]);
                            }
                        }
                        if (stress_on)
                            for (int c = 0; c < 6; ++c) expected_stress[c] += std::real(density * s[c]);
                    }
                }
            }
        }
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            std::vector<double> force(9, 0.0);
            std::vector<double> stress(6, 0.0);
            compute(batch, dm.data(), dm.size(), 3, force_on, stress_on, force.data(), stress.data());
            for (int i = 0; i < 9; ++i) EXPECT_NEAR(force[i], expected_force[i], 1e-12);
            for (int i = 0; i < 6; ++i) EXPECT_NEAR(stress[i], expected_stress[i], 1e-12);
        }
    }
}
}
TEST(NonlocalForceStressGPU, RealScalarOracle) { check<double>(1, false); }
TEST(NonlocalForceStressGPU, SpinorScalarOracle) { check<std::complex<double>>(2, false); }
// DFT+U stores four real Pauli components in the 2x2 DMR slots, not
// a complex spinor matrix. No Pauli-to-spinor conversion belongs here.
TEST(NonlocalForceStressGPU, HubbardPauliScalarOracle) { check<double>(2, true); }
TEST(NonlocalForceStressGPU, HubbardCollinearScalarOracle) { check<double>(1, true); }
TEST(NonlocalForceStressGPU, EmptyRank)
{
    hamilt::nonlocal_gpu::Batch empty;
    double force[3] = {};
    double stress[6] = {};
    hamilt::nonlocal_gpu::compute(empty, static_cast<const double*>(nullptr), 0, 1, true, true, force, stress);
    for (double f : force) EXPECT_EQ(f, 0.0);
    for (double s : stress) EXPECT_EQ(s, 0.0);
}
