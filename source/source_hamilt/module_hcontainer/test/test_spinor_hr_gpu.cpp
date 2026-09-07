#include "gtest/gtest.h"
#include "source_hamilt/module_gint/kernel/spinor_hr_gpu.h"
#include "source_hamilt/module_gint/kernel/cuda_mem_wrapper.h"
#include "source_hamilt/module_hcontainer/hcontainer.h"
#include "source_base/parallel_global.h"

namespace
{
int test_rank = 0;
int test_size = 1;
}

TEST(SpinorHrGpu, DistributedPauliOracleAndReuse)
{
    const int rank = test_rank;
    const int size = test_size;
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    ASSERT_GT(device_count, 0);
    CHECK_CUDA(cudaSetDevice(rank % device_count));
    const int trace[] = {0, 6};
    Parallel_Orbitals pv;
    pv.init(12, 12, 2, MPI_COMM_WORLD);
    pv.set_atomic_trace(trace, 2, 12);
    hamilt::HContainer<double> source(2);
    hamilt::HContainer<std::complex<double>> destination(&pv);
    for (int a = 0; a < 2; ++a)
    {
        for (int b = 0; b < 2; ++b)
        {
            hamilt::AtomPair<double> pair(a, b);
            pair.set_size(3, 3);
            pair.get_HR_values(0, 0, 0);
            pair.get_HR_values(1, 0, 0);
            pair.get_HR_values(-1, 0, 0);
            // Different grid ranks see different pairs; some owned entries
            // have no contributing grid rank in the one-rank fixture.
            if ((rank + a + b) % 2 == 0) source.insert_pair(pair);
            if (!pv.is_invalid_atom_pair(a, b))
            {
                hamilt::AtomPair<std::complex<double>> target(a, b, &pv);
                target.get_HR_values(0, 0, 0);
                target.get_HR_values(1, 0, 0);
                target.get_HR_values(-1, 0, 0);
                destination.insert_pair(target);
            }
        }
    }
    source.allocate(nullptr, true);
    destination.allocate(nullptr, true);
    CudaMemWrapper<double> channels[4];
    for (int s = 0; s < 4; ++s)
    {
        channels[s] = CudaMemWrapper<double>(source.get_nnr(), 0, true);
    }
    ModuleGint::SpinorHrGpu plan(source, destination);
    ASSERT_TRUE(plan.matches(destination));
    for (int pass = 0; pass < 3; ++pass)
    {
        for (int s = 0; s < 4; ++s)
        {
            for (int ap_index = 0; ap_index < source.size_atom_pairs(); ++ap_index)
            {
                const auto& ap = source.get_atom_pair(ap_index);
                for (int ir = 0; ir < ap.get_R_size(); ++ir)
                {
                    const auto R = ap.get_R_index(ir);
                    const auto* matrix = ap.find_matrix(R);
                    const auto offset = matrix->get_pointer() - source.get_wrapper();
                    for (int row = 0; row < 3; ++row)
                        for (int col = 0; col < 3; ++col)
                            channels[s].get_host_ptr()[offset + row * 3 + col]
                                = (rank + 1) * (1 + pass + 4 * s + 0.1 * row + 0.01 * col + 0.001 * R.x);
                }
            }
            channels[s].copy_host_to_device_sync();
        }
        const bool transverse = pass != 1;
        plan.transfer(channels[0].get_device_ptr(), channels[1].get_device_ptr(),
                      channels[2].get_device_ptr(), channels[3].get_device_ptr(), transverse, destination);
        for (int i = 0; i < destination.size_atom_pairs(); ++i)
        {
            const auto& ap = destination.get_atom_pair(i);
            const bool reverse = ap.get_atom_i() > ap.get_atom_j();
            const auto rows = pv.get_indexes_row(ap.get_atom_i());
            const auto cols = pv.get_indexes_col(ap.get_atom_j());
            for (int ir = 0; ir < ap.get_R_size(); ++ir)
            {
                const auto R = ap.get_R_index(ir);
                const auto* matrix = ap.find_matrix(R);
                for (int row = 0; row < int(rows.size()); ++row)
                {
                    for (int col = 0; col < int(cols.size()); ++col)
                    {
                        const int r = rows[row];
                        const int c = cols[col];
                        const double base = 1 + pass + 0.1 * (reverse ? c / 2 : r / 2)
                            + 0.01 * (reverse ? r / 2 : c / 2) + 0.001 * (reverse ? -R.x : R.x);
                        std::complex<double> expected;
                        if (r % 2 == c % 2) expected = r % 2 ? -12.0 : 2 * base + 12.0;
                        else if (transverse) expected = {base + 4, (r % 2 ? 1 : -1) * (base + 8)};
                        if (reverse) expected = std::conj(expected);
                        int rank_weight = 0;
                        for (int source_rank = 0; source_rank < size; ++source_rank)
                            if ((source_rank + ap.get_atom_i() + ap.get_atom_j()) % 2 == 0)
                                rank_weight += source_rank + 1;
                        expected *= rank_weight;
                        EXPECT_NEAR(std::abs(matrix->get_value(row, col) - expected), 0.0, 1e-12);
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    int threads = 1;
    Parallel_Global::read_pal_param(argc, argv, test_size, threads, test_rank);
    // This standalone fixture does not create the application's pools.
    // Set the unused handles before the shared finalizer inspects them.
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
