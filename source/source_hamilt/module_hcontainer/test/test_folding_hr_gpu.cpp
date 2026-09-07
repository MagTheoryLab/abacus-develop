#include "gtest/gtest.h"
#include "source_hamilt/module_hcontainer/folding_hr_gpu.h"
#include "source_hamilt/module_hcontainer/hcontainer_funcs.h"
#include "source_base/module_device/device_check.h"
#include "source_base/parallel_global.h"
#include <algorithm>
#include <stdexcept>

namespace
{
int test_rank = 0;
int test_size = 1;
}

TEST(FoldingHrGpu, DistributedComplexOracleAndValueUpdate)
{
    int count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    ASSERT_GT(count, 0);
    CHECK_CUDA(cudaSetDevice(test_rank % count));
    const int trace[] = {0, 31, 50};
    Parallel_Orbitals pv;
    pv.init(67, 67, 2, MPI_COMM_WORLD);
    pv.set_atomic_trace(trace, 3, 67);
    hamilt::HContainer<std::complex<double>> hr(&pv);
    for (int a = 0; a < 3; ++a)
    {
        for (int b = 0; b < 3; ++b)
        {
            if (a == 1 && b == 2) continue;
            if (pv.is_invalid_atom_pair(a, b)) continue;
            hamilt::AtomPair<std::complex<double>> pair(a, b, &pv);
            pair.get_HR_values(0, 0, 0);
            pair.get_HR_values(-1, 2, 1);
            pair.get_HR_values(2, -1, -3);
            hr.insert_pair(pair);
        }
    }
    hr.allocate(nullptr, true);
    const std::vector<ModuleBase::Vector3<double>> keys = {{0, 0, 0}, {0.13, -0.27, 0.31}, {-0.2, 0.11, -0.7}};
    hamilt::FoldingHrGpu plan(hr, keys);
    std::vector<std::complex<double>> oracle(pv.get_local_size());
    std::vector<std::complex<double>> result(pv.get_local_size());
    for (int pass = 0; pass < 3; ++pass)
    {
        for (int ip = 0; ip < hr.size_atom_pairs(); ++ip)
        {
            auto& pair = hr.get_atom_pair(ip);
            for (int ir = 0; ir < pair.get_R_size(); ++ir)
                for (int i = 0; i < pair.get_row_size() * pair.get_col_size(); ++i)
                    pair.get_pointer(ir)[i] = {0.17 * (pass + 1) + 0.13 * ir - 0.02 * ip + 0.003 * i,
                                               -0.21 * pass + 0.07 * ir + 0.003 * ip - 0.011 * i};
        }
        plan.upload(hr);
        for (int ik = 0; ik < int(keys.size()); ++ik)
        {
            std::fill(oracle.begin(), oracle.end(), std::complex<double>(0, 0));
            hamilt::folding_HR(hr, oracle.data(), keys[ik], pv.get_row_size(), 1);
            CHECK_CUDA(cudaMemcpy(result.data(), plan.fold(ik), result.size() * sizeof(result[0]), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < result.size(); ++i)
                EXPECT_NEAR(std::abs(result[i] - oracle[i]), 0.0, 1e-12);
        }
    }
    EXPECT_THROW(plan.fold(-1), std::out_of_range);
    EXPECT_THROW(plan.fold(keys.size()), std::out_of_range);
}

TEST(FoldingHrGpu, EmptyLocalContainer)
{
    Parallel_Orbitals pv;
    pv.init(19, 19, 2, MPI_COMM_WORLD);
    hamilt::HContainer<std::complex<double>> hr(&pv);
    const std::vector<ModuleBase::Vector3<double>> keys = {{0.1, 0.2, 0.3}};
    hamilt::FoldingHrGpu plan(hr, keys);
    plan.upload(hr);
    std::vector<std::complex<double>> result(pv.get_local_size());
    CHECK_CUDA(cudaMemcpy(result.data(), plan.fold(0), result.size() * sizeof(result[0]), cudaMemcpyDeviceToHost));
    for (const auto& v : result) EXPECT_EQ(v, std::complex<double>(0, 0));
}

int main(int argc, char** argv)
{
    int threads = 1;
    Parallel_Global::read_pal_param(argc, argv, test_size, threads, test_rank);
    POOL_WORLD = MPI_COMM_NULL;
    KP_WORLD = MPI_COMM_NULL;
    INT_BGROUP = MPI_COMM_NULL;
    BP_WORLD = MPI_COMM_NULL;
    GRID_WORLD = MPI_COMM_NULL;
    DIAG_WORLD = MPI_COMM_NULL;
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    Parallel_Global::finalize_mpi();
    return result;
}
