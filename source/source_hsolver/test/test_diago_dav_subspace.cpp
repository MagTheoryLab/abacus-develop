#include "source_hsolver/diago_dav_subspace.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/parallel_reduce.h"
#include "source_base/parallel_comm.h"
#ifdef __CUDA
#include "source_base/parallel_global.h"
#endif
#include <memory>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace
{
template <typename Device, typename HAction, typename SAction>
class TestHSOperator : public hsolver::HSOperator<std::complex<double>, Device>
{
  public:
    TestHSOperator(const HAction& h, const SAction& s) : h_(h), s_(s) {}
    void update_k(const int) override {}
    void hpsi(const std::complex<double>* x, std::complex<double>* hx,
              const int ld, const int count) const override
    {
        h_(x, hx, ld, count);
    }
    void spsi(const std::complex<double>* x, std::complex<double>* sx,
              const int ld, const int count) const override
    {
        s_(x, sx, ld, count);
    }
  private:
    const HAction& h_;
    const SAction& s_;
};

template <typename Device>
void check_nearly_dependent_corrections(const bool generalized,
                                       const bool distributed,
                                       const bool identity_overlap,
                                       const bool exercise_restarts)
{
    using Complex = std::complex<double>;
    int ranks = 1;
    int rank = 0;
#ifdef __MPI
    std::unique_ptr<MPICommGroup> group;
    if (distributed)
    {
        group.reset(new MPICommGroup(POOL_WORLD));
        ranks = group->gsize;
        rank = group->grank;
    }
#endif
    const int dim = 16;
    const int bands = 3;
    const int stride = dim + 3;
    std::vector<double> metric(dim, 1.0);
    if (generalized)
    {
        for (int i = 0; i < dim; ++i)
        {
            metric[i] = 1.0 + 0.125 * (i * ranks + rank);
        }
    }
    std::vector<Complex> psi(bands * stride, Complex(0.0));
    // All three approximate eigenvectors have the same small contaminating
    // direction. Independently normalized corrections are almost dependent.
    for (int band = 0; band < bands; ++band)
    {
        // Interleave global components across ranks so the Gram matrix needs
        // contributions from every rank, not just the eigensolver root.
        if (band % ranks == rank)
        {
            const int local = band / ranks;
            psi[band * stride + local] = 1.0 / std::sqrt(metric[local]);
        }
        if (7 % ranks == rank)
        {
            psi[band * stride + 7 / ranks] = Complex(1e-7 * (band + 1), -2e-7);
        }
        if (exercise_restarts)
        {
            for (int i = 0; i < dim; ++i)
            {
                const double phase = (band + 1) * (i * ranks + rank + 1);
                psi[band * stride + i] += Complex(0.1 * std::sin(phase), 0.1 * std::cos(phase));
            }
        }
        std::fill(psi.begin() + band * stride + dim,
                  psi.begin() + (band + 1) * stride, Complex(123.0, -45.0));
    }
    int hamiltonian_applications = 0;
    const auto h = [&metric, &hamiltonian_applications, ranks, rank](const Complex* in, Complex* out, const int ld, const int count)
    {
        ++hamiltonian_applications;
        std::vector<Complex> input(ld * count);
        std::vector<Complex> output(ld * count);
        base_device::memory::synchronize_memory_op<Complex, base_device::DEVICE_CPU, Device>()(
            input.data(), in, ld * count);
        for (int band = 0; band < count; ++band)
        {
            for (int i = 0; i < ld; ++i)
            {
                output[band * ld + i] = (i * ranks + rank + 1) * metric[i] * input[band * ld + i];
            }
        }
        base_device::memory::synchronize_memory_op<Complex, Device, base_device::DEVICE_CPU>()(
            out, output.data(), ld * count);
    };
    int overlap_applications = 0;
    const auto s = [&metric, &overlap_applications](const Complex* in, Complex* out, const int ld, const int count)
    {
        ++overlap_applications;
        std::vector<Complex> input(ld * count);
        std::vector<Complex> output(ld * count);
        base_device::memory::synchronize_memory_op<Complex, base_device::DEVICE_CPU, Device>()(
            input.data(), in, ld * count);
        for (int band = 0; band < count; ++band)
        {
            for (int i = 0; i < ld; ++i)
            {
                output[band * ld + i] = metric[i] * input[band * ld + i];
            }
        }
        base_device::memory::synchronize_memory_op<Complex, Device, base_device::DEVICE_CPU>()(
            out, output.data(), ld * count);
    };
    std::vector<double> precondition(dim, 1.0);
    std::vector<double> eigenvalues(bands, 0.0);
    const double threshold = exercise_restarts ? 1e-14 : 1e-12;
    const int max_iterations = exercise_restarts ? 200 : 50;
    const std::vector<double> thresholds(bands, threshold);
#ifdef __MPI
    const hsolver::diag_comm_info comm = {distributed ? POOL_WORLD : MPI_COMM_SELF, rank, ranks};
#else
    const hsolver::diag_comm_info comm = {0, 1};
#endif
    hsolver::Diago_DavSubspace<Complex, Device> solver(precondition, bands, dim, 2,
                                              threshold, max_iterations, comm, 0, 0, identity_overlap);
    container::Tensor device_psi(container::DataTypeToEnum<Complex>::value,
                                 container::DeviceTypeToEnum<Device>::value, {bands * stride});
    base_device::memory::synchronize_memory_op<Complex, Device, base_device::DEVICE_CPU>()(
        device_psi.data<Complex>(), psi.data(), bands * stride);
    TestHSOperator<Device, decltype(h), decltype(s)> op(h, s);
    solver.diag(op, device_psi.data<Complex>(), stride, eigenvalues.data(), thresholds, false);
    if (identity_overlap)
    {
        EXPECT_EQ(overlap_applications, 0);
    }
    else
    {
        EXPECT_GT(overlap_applications, 0);
    }
    if (exercise_restarts)
    {
        // Three residual columns fill the six-column workspace immediately.
        // Further H applications require restarting from the Ritz vectors.
        EXPECT_GT(hamiltonian_applications, 3);
    }
    base_device::memory::synchronize_memory_op<Complex, base_device::DEVICE_CPU, Device>()(
        psi.data(), device_psi.data<Complex>(), bands * stride);
    for (int band = 0; band < bands; ++band)
    {
        EXPECT_NEAR(eigenvalues[band], band + 1.0, 1e-10);
        double residual = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            residual += std::norm(((i * ranks + rank + 1) - eigenvalues[band])
                                  * metric[i] * psi[band * stride + i]);
        }
        if (distributed)
        {
            Parallel_Reduce::reduce_pool(residual);
        }
        const double residual_tolerance = exercise_restarts ? 1e-7 : 1e-9;
        EXPECT_LT(std::sqrt(residual), residual_tolerance);
        for (int other = 0; other < bands; ++other)
        {
            Complex overlap = 0.0;
            for (int i = 0; i < dim; ++i)
            {
                overlap += std::conj(psi[band * stride + i]) * metric[i]
                           * psi[other * stride + i];
            }
            if (distributed)
            {
                Parallel_Reduce::reduce_pool(overlap);
            }
            EXPECT_NEAR(std::abs(overlap - Complex(band == other ? 1.0 : 0.0)), 0.0, 1e-10);
        }
    }
}
}

TEST(DiagoDavSubspace, NearlyDependentCorrections)
{
    check_nearly_dependent_corrections<base_device::DEVICE_CPU>(false, false, false, false);
}

TEST(DiagoDavSubspace, IdentityOverlapNearlyDependentCorrections)
{
    check_nearly_dependent_corrections<base_device::DEVICE_CPU>(false, false, true, false);
}

TEST(DiagoDavSubspace, IdentityOverlapRestarts)
{
    check_nearly_dependent_corrections<base_device::DEVICE_CPU>(false, false, false, true);
    check_nearly_dependent_corrections<base_device::DEVICE_CPU>(false, false, true, true);
}

TEST(DiagoDavSubspace, GeneralizedOverlapAndPaddedWavefunctions)
{
    check_nearly_dependent_corrections<base_device::DEVICE_CPU>(true, false, false, false);
}

#ifdef __CUDA
TEST(DiagoDavSubspace, GpuNearlyDependentCorrections)
{
    ModuleBase::createGpuBlasHandle();
    container::kernels::createGpuSolverHandle();
    check_nearly_dependent_corrections<base_device::DEVICE_GPU>(false, false, false, false);
    check_nearly_dependent_corrections<base_device::DEVICE_GPU>(false, false, true, false);
    check_nearly_dependent_corrections<base_device::DEVICE_GPU>(true, false, false, false);
    check_nearly_dependent_corrections<base_device::DEVICE_GPU>(false, false, true, true);
    container::kernels::destroyGpuSolverHandle();
    ModuleBase::destoryBLAShandle();
}
#ifdef __MPI
TEST(DiagoDavSubspace, DistributedCorrections)
{
    MPICommGroup group(POOL_WORLD);
    if (group.gsize < 2)
    {
        GTEST_SKIP() << "Requires at least two ranks";
    }
    ModuleBase::createGpuBlasHandle();
    container::kernels::createGpuSolverHandle();
    check_nearly_dependent_corrections<base_device::DEVICE_CPU>(true, true, false, false);
    check_nearly_dependent_corrections<base_device::DEVICE_CPU>(false, true, true, false);
    check_nearly_dependent_corrections<base_device::DEVICE_GPU>(true, true, false, false);
    check_nearly_dependent_corrections<base_device::DEVICE_GPU>(false, true, true, false);
    container::kernels::destroyGpuSolverHandle();
    ModuleBase::destoryBLAShandle();
}
#endif
#endif

#ifdef __CUDA
int main(int argc, char** argv)
{
    int ranks = 1;
    int threads = 1;
    int rank = 0;
    Parallel_Global::read_pal_param(argc, argv, ranks, threads, rank);
    POOL_WORLD = MPI_COMM_WORLD;
    KP_WORLD = MPI_COMM_NULL;
    BP_WORLD = MPI_COMM_NULL;
    INT_BGROUP = MPI_COMM_NULL;
    GRID_WORLD = MPI_COMM_NULL;
    DIAG_WORLD = MPI_COMM_NULL;
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    Parallel_Global::finalize_mpi();
    return result;
}
#endif
