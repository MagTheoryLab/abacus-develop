#include "gtest/gtest.h"
#include "../density_gather.h"
#include "../hcontainer.h"
#include "../hcontainer_funcs.h"
#include "source_base/parallel_global.h"

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
                serial.insert_pair(pair);
            }
        }
    source.allocate(nullptr, true);
    serial.allocate(nullptr, true);
    hamilt::DensityGather plan(source, serial);
    ASSERT_TRUE(plan.matches(source));
    for (int pass = 0; pass < 3; ++pass)
    {
        for (int i = 0; i < source.size_atom_pairs(); ++i)
        {
            auto& ap = source.get_atom_pair(i);
            const auto rows = pv.get_indexes_row(ap.get_atom_i());
            const auto cols = pv.get_indexes_col(ap.get_atom_j());
            for (int ir = 0; ir < ap.get_R_size(); ++ir)
                for (size_t row = 0; row < rows.size(); ++row)
                    for (size_t col = 0; col < cols.size(); ++col)
                        ap.get_pointer(ir)[row * cols.size() + col]
                            = 10000 * pass + 1000 * ap.get_atom_i() + 100 * ap.get_atom_j()
                              + 10 * rows[row] + cols[col] + 0.25 * ap.get_R_index(ir).x;
        }
        const auto& actual = plan.gather(source);
        serial.set_zero();
        hamilt::transferParallels2Serials(source, &serial);
        for (int i = 0; i < actual.size_atom_pairs(); ++i)
        {
            const auto& ap = actual.get_atom_pair(i);
            for (int ir = 0; ir < ap.get_R_size(); ++ir)
                for (int row = 0; row < ap.get_row_size(); ++row)
                    for (int col = 0; col < ap.get_col_size(); ++col)
                    {
                        const double expected = 10000 * pass + 1000 * ap.get_atom_i() + 100 * ap.get_atom_j()
                                                + 10 * row + col + 0.25 * ap.get_R_index(ir).x;
                        const int index = row * ap.get_col_size() + col;
                        EXPECT_DOUBLE_EQ(ap.get_pointer(ir)[index], expected);
                        EXPECT_DOUBLE_EQ(ap.get_pointer(ir)[index],
                            serial.find_matrix(ap.get_atom_i(), ap.get_atom_j(), ap.get_R_index(ir))->get_pointer()[index]);
                    }
        }
    }
    hamilt::HContainer<double> empty(&pv);
    empty.allocate(nullptr, true);
    EXPECT_FALSE(plan.matches(empty));
    hamilt::HContainer<double> empty_serial(3);
    empty_serial.allocate(nullptr, true);
    hamilt::DensityGather empty_plan(empty, empty_serial);
    EXPECT_TRUE(empty_plan.matches(empty));
    EXPECT_EQ(empty_plan.gather(empty).get_nnr(), 0);
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
