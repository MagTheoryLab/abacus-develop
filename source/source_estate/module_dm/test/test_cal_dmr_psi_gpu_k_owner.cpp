#include "gtest/gtest.h"
#include "source_estate/module_dm/cal_dm_psi.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/parallel_global.h"
#include "source_base/parallel_reduce.h"
#include <cuda_runtime.h>
#include <cmath>
#include <memory>

namespace
{
using Complex = std::complex<double>;
int test_rank = 0;
int test_ranks = 1;

Complex coefficient(int ik, int band, int orbital)
{
    return Complex(std::sin(0.17 * (1 + ik + 3 * band + orbital)),
                   std::cos(0.23 * (2 + 2 * ik + band + 2 * orbital))) / 4.0;
}

// Deliberately independent scalar contraction, with no GPU packing helpers.
Complex reference(int row, int col, const ModuleBase::Vector3<int>& R,
                  const std::vector<ModuleBase::Vector3<double>>& kvec,
                  const ModuleBase::matrix& weights)
{
    Complex result = 0.0;
    for (int ik = 0; ik < weights.nr; ++ik)
    {
        const double arg = 2.0 * std::acos(-1.0) * (kvec[ik] * ModuleBase::Vector3<double>(R.x, R.y, R.z));
        for (int ib = 0; ib < weights.nc; ++ib)
        {
            result += std::polar(1.0, arg) * weights(ik, ib)
                      * std::conj(coefficient(ik, ib, row)) * coefficient(ik, ib, col);
        }
    }
    return result;
}

void check_contraction(int nlocal)
{
    const int nk = 3; // A four-rank run includes a rank with no owned k point.
    const int nbands = 3;
    Parallel_Orbitals pv;
    pv.init(nlocal, nlocal, 2, MPI_COMM_WORLD);
    const int atom_begin[] = {0};
    pv.set_atomic_trace(atom_begin, 1, nlocal);
    std::vector<ModuleBase::Vector3<double>> kvec = {{0.13, 0.21, -0.07}, {-0.17, 0.02, 0.31}, {0.0, 0.0, 0.0}};
    hamilt::HContainer<double> layout(&pv);
    if (pv.get_nrow_atom(0) > 0 && pv.get_ncol_atom(0) > 0)
    {
        for (int r = -1; r <= 1; ++r)
        {
            hamilt::AtomPair<double> pair(0, 0, r, 1, -1, &pv);
            layout.insert_pair(pair);
        }
    }
    layout.allocate(nullptr, true);
    elecstate::DensityMatrix<Complex, double> dm(&pv, 1, kvec, nk);
    dm.init_DMR(layout);
    hamilt::HContainer<Complex> full(&pv);
    std::vector<int> ijrs = layout.get_ijr_info();
    full.insert_ijrs(&ijrs);
    full.allocate(nullptr, true);
    std::vector<std::unique_ptr<psi::Psi<Complex, base_device::DEVICE_GPU>>> owned(nk);
    std::vector<const psi::Psi<Complex, base_device::DEVICE_GPU>*> view(nk, nullptr);
    for (int ik = 0; ik < nk; ++ik)
    {
        if (ik % test_ranks != test_rank)
        {
            continue;
        }
        psi::Psi<Complex> host;
        host.resize(1, nbands, nlocal);
        for (int ib = 0; ib < nbands; ++ib)
        {
            for (int io = 0; io < nlocal; ++io)
            {
                host.get_pointer()[ib * nlocal + io] = coefficient(ik, ib, io);
            }
        }
        owned[ik].reset(new psi::Psi<Complex, base_device::DEVICE_GPU>(host));
        view[ik] = owned[ik].get();
    }
    // Ordinary occupations, energy weights (negative and zero included), then
    // ordinary occupations again: each call must replace, never accumulate.
    for (int pass = 0; pass < 3; ++pass)
    {
        ModuleBase::matrix weights(nk, nbands);
        for (int ik = 0; ik < nk; ++ik)
        {
            for (int ib = 0; ib < nbands; ++ib)
            {
                weights(ik, ib) = (0.15 + 0.03 * ik + 0.07 * ib) * (pass == 1 ? ib - 1.0 : 1.0);
            }
        }
        elecstate::cal_dmr_psi_gpu_k_owner(&pv, weights, view, dm);
        elecstate::cal_dmr_psi_gpu_k_owner(&pv, weights, view, kvec, full);
        for (int ia = 0; ia < static_cast<int>(full.size_atom_pairs()); ++ia)
        {
            const auto& pair = full.get_atom_pair(ia);
            const auto& real_pair = dm.get_DMR_pointer(1)->get_atom_pair(ia);
            for (int ir = 0; ir < pair.get_R_size(); ++ir)
            {
                const auto R = pair.get_R_index(ir);
                const auto* values = pair.get_HR_values(ir).get_pointer();
                const auto* pauli = real_pair.get_HR_values(ir).get_pointer();
                for (int i = 0; i < pair.get_row_size(); ++i)
                {
                    for (int j = 0; j < pair.get_col_size(); ++j)
                    {
                        const int row = pv.local2global_row(i);
                        const int col = pv.local2global_col(j);
                        const int index = i * pair.get_col_size() + j;
                        EXPECT_NEAR(std::abs(values[index] - reference(row, col, R, kvec, weights)), 0.0, 2e-13);
                        const int up = row - row % 2;
                        const int left = col - col % 2;
                        const Complex uu = reference(up, left, R, kvec, weights);
                        const Complex ud = reference(up, left + 1, R, kvec, weights);
                        const Complex du = reference(up + 1, left, R, kvec, weights);
                        const Complex dd = reference(up + 1, left + 1, R, kvec, weights);
                        const double expected[] = {(uu + dd).real(), (ud + du).real(), (ud - du).imag(), (uu - dd).real()};
                        EXPECT_NEAR(pauli[index], expected[(i % 2) * 2 + j % 2], 2e-13);
                    }
                }
            }
        }
    }
}
} // namespace

TEST(KOwnerDMR, ComplexAndEnergyWeightedAgainstScalarOracle)
{
    check_contraction(12);
}

TEST(KOwnerDMR, EmptyLocalBlocksParticipate)
{
    check_contraction(2);
}

int main(int argc, char** argv)
{
    int threads = 1;
    Parallel_Global::read_pal_param(argc, argv, test_ranks, threads, test_rank);
    // This fixture does not create ABACUS pools. Mark them absent before the
    // shared finalizer (OpenMPI's MPI_COMM_NULL is not a zero pointer).
    POOL_WORLD = MPI_COMM_NULL;
    KP_WORLD = MPI_COMM_NULL;
    BP_WORLD = MPI_COMM_NULL;
    INT_BGROUP = MPI_COMM_NULL;
    GRID_WORLD = MPI_COMM_NULL;
    DIAG_WORLD = MPI_COMM_NULL;
    testing::InitGoogleTest(&argc, argv);
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    {
        Parallel_Global::finalize_mpi();
        return 1;
    }
    cudaSetDevice(test_rank % devices);
    ModuleBase::createGpuBlasHandle();
    int result = RUN_ALL_TESTS();
    Parallel_Reduce::reduce_all(result);
    ModuleBase::destoryBLAShandle();
    Parallel_Global::finalize_mpi();
    return result == 0 ? 0 : 1;
}
