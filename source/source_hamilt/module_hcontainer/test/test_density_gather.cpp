#include "gtest/gtest.h"
#include "../density_gather.h"
#include "../hcontainer.h"
#include "source_base/parallel_global.h"
#include <algorithm>

namespace
{
int test_rank = 0;
int test_size = 1;
}

TEST(DensityGather, UnevenSpinorBlocksReuseAndIndependentOracle)
{
    const int trace[] = {0, 6, 16};
    Parallel_Orbitals pv;
    pv.init(24, 24, 2, MPI_COMM_WORLD);
    pv.set_atomic_trace(trace, 3, 24);
    const int widths[] = {6, 10, 8};
    hamilt::HContainer<double> source(&pv);
    hamilt::HContainer<double> serial(3);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        {
            if (!pv.is_invalid_atom_pair(a, b))
            {
                hamilt::AtomPair<double> pair(a, b, &pv);
                pair.get_HR_values(-1, 2, 0);
                pair.get_HR_values(0, 0, 0);
                source.insert_pair(pair);
            }
            if ((a + b + test_rank) % 2 == 0)
            {
                hamilt::AtomPair<double> pair(a, b);
                pair.set_size(widths[b], widths[a]);
                pair.get_HR_values(-1, 2, 0);
                pair.get_HR_values(0, 0, 0);
                pair.get_HR_values(2, 0, -1);
                serial.insert_pair(pair);
            }
        }
    source.allocate(nullptr, true);
    serial.allocate(nullptr, true);
    // Force multiple batches with unequal and empty peer messages.
    hamilt::DensityGather plan(source, serial, 1);
    ASSERT_TRUE(plan.matches(source));
    hamilt::HContainer<double> replacement(source);
    const auto topology = source.get_ijr_info();
    const size_t source_size = source.get_nnr();
    for (int pass = 0; pass < 3; ++pass)
    {
        SCOPED_TRACE(pass);
        auto& input = pass == 1 ? replacement : source;
        ASSERT_EQ(input.get_paraV(), &pv);
        ASSERT_EQ(input.get_nnr(), source_size);
        ASSERT_EQ(input.get_ijr_info(), topology);
        ASSERT_TRUE(plan.matches(input));
        for (int i = 0; i < input.size_atom_pairs(); ++i)
        {
            auto& ap = input.get_atom_pair(i);
            const auto rows = pv.get_indexes_row(ap.get_atom_i());
            const auto cols = pv.get_indexes_col(ap.get_atom_j());
            for (int ir = 0; ir < ap.get_R_size(); ++ir)
                for (size_t row = 0; row < rows.size(); ++row)
                    for (size_t col = 0; col < cols.size(); ++col)
                        ap.get_pointer(ir)[row * cols.size() + col]
                            = 10000 * pass + 1000 * ap.get_atom_i() + 100 * ap.get_atom_j()
                              + 10 * rows[row] + cols[col] + 0.25 * ap.get_R_index(ir).x;
        }
        const auto& actual = plan.gather(input);
        EXPECT_TRUE(plan.matches(input));
        for (int i = 0; i < actual.size_atom_pairs(); ++i)
        {
            const auto& ap = actual.get_atom_pair(i);
            for (int ir = 0; ir < ap.get_R_size(); ++ir)
                for (int row = 0; row < ap.get_row_size(); ++row)
                    for (int col = 0; col < ap.get_col_size(); ++col)
                    {
                        const int rx = ap.get_R_index(ir).x;
                        const double expected = rx == 2 ? 0.0
                            : 10000 * pass + 1000 * ap.get_atom_i() + 100 * ap.get_atom_j()
                                + 10 * row + col + 0.25 * rx;
                        const int index = row * ap.get_col_size() + col;
                        EXPECT_DOUBLE_EQ(ap.get_pointer(ir)[index], expected);
                    }
        }
    }
    hamilt::HContainer<double> empty(&pv);
    empty.allocate(nullptr, true);
    EXPECT_FALSE(plan.matches(empty));
    hamilt::HContainer<double> empty_serial(3);
    empty_serial.allocate(nullptr, true);
    hamilt::DensityGather empty_plan(empty, empty_serial, 1);
    EXPECT_TRUE(empty_plan.matches(empty));
    EXPECT_EQ(empty_plan.gather(empty).get_nnr(), 0);
}

TEST(DensityGather, ReplicatedLayoutsKeepSourceRankOrder)
{
    const int trace[] = {0};
    Parallel_Orbitals pv;
    pv.set_serial(2, 2);
    pv.set_atomic_trace(trace, 1, 2);
    hamilt::HContainer<double> source(&pv);
    hamilt::AtomPair<double> local_pair(0, 0, &pv);
    local_pair.get_HR_values(0, 0, 0);
    source.insert_pair(local_pair);
    source.allocate(nullptr, true);
    const double value = test_rank + 1.0;
    std::fill_n(source.get_wrapper(), source.get_nnr(), value);

    hamilt::HContainer<double> serial(1);
    hamilt::AtomPair<double> serial_pair(0, 0);
    serial_pair.set_size(2, 2);
    serial_pair.get_HR_values(0, 0, 0);
    serial.insert_pair(serial_pair);
    serial.allocate(nullptr, true);
    hamilt::DensityGather plan(source, serial, 1);
    const auto& actual = plan.gather(source);
    ASSERT_EQ(actual.get_nnr(), 4u);
    for (size_t i = 0; i < actual.get_nnr(); ++i)
    {
        EXPECT_DOUBLE_EQ(actual.get_wrapper()[i], test_size);
    }
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
