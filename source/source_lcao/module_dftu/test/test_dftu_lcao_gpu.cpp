#include "../dftu_lcao_gpu.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <complex>
#include <vector>

// Independent scalar contractions with noncontiguous local orbital ownership.
// Summing four local occupation blocks must reproduce the full matrix oracle.
TEST(DFTULcaoGPU, DistributedBlocksMatchScalar)
{
    const int rows = 5;
    const int cols = 3;
    const int m = 3;
    std::vector<double> left(rows * m);
    std::vector<double> right(cols * m);
    std::vector<double> dm(4 * rows * cols);
    std::vector<std::complex<double>> potential(4 * m * m);
    for (int i = 0; i < rows * m; ++i) { left[i] = 0.03 * (i - 4); }
    for (int i = 0; i < cols * m; ++i) { right[i] = 0.07 * (i + 1); }
    for (int i = 0; i < static_cast<int>(dm.size()); ++i) { dm[i] = 0.01 * (i % 17 - 8); }
    for (int i = 0; i < static_cast<int>(potential.size()); ++i)
    {
        potential[i] = std::complex<double>(0.02 * (i - 7), 0.01 * (i % 5 - 2));
    }
    std::vector<double> expected(4 * m * m, 0.0);
    std::vector<double> actual(expected.size(), 0.0);
    for (int sr = 0; sr < 2; ++sr)
    {
        for (int sc = 0; sc < 2; ++sc)
        {
            for (int a = 0; a < m; ++a)
            {
                for (int b = 0; b < m; ++b)
                {
                    for (int r = 0; r < rows; ++r)
                    {
                        for (int c = 0; c < cols; ++c)
                        {
                            expected[(2 * sr + sc) * m * m + a * m + b]
                                += left[r * m + a] * right[c * m + b]
                                 * dm[(2 * r + sr) * (2 * cols) + 2 * c + sc];
                        }
                    }
                }
            }
        }
    }
    for (int rank_row = 0; rank_row < 2; ++rank_row)
    {
        for (int rank_col = 0; rank_col < 2; ++rank_col)
        {
            std::vector<int> rr;
            std::vector<int> cc;
            std::vector<double> projections;
            for (int r = rank_row; r < rows; r += 2)
            {
                rr.push_back(r);
                projections.insert(projections.end(), left.begin() + r * m, left.begin() + (r + 1) * m);
            }
            const std::size_t right_offset = projections.size();
            for (int c = rank_col; c < cols; c += 2)
            {
                cc.push_back(c);
                projections.insert(projections.end(), right.begin() + c * m, right.begin() + (c + 1) * m);
            }
            const int stride = 2 * cc.size();
            // Nonzero offsets also test packed HContainer block placement.
            std::vector<double> local_dm(7 + 4 * rr.size() * cc.size(), 0.0);
            std::vector<std::complex<double>> hr(11 + 4 * rr.size() * cc.size(), {0.25, -0.1});
            auto expected_hr = hr;
            for (int r = 0; r < static_cast<int>(rr.size()); ++r)
            {
                for (int c = 0; c < static_cast<int>(cc.size()); ++c)
                {
                    for (int sr = 0; sr < 2; ++sr)
                    {
                        for (int sc = 0; sc < 2; ++sc)
                        {
                            const int local = (2 * r + sr) * stride + 2 * c + sc;
                            local_dm[7 + local] = dm[(2 * rr[r] + sr) * 2 * cols + 2 * cc[c] + sc];
                            for (int a = 0; a < m; ++a)
                            {
                                for (int b = 0; b < m; ++b)
                                {
                                    expected_hr[11 + local] += left[rr[r] * m + a] * right[cc[c] * m + b]
                                        * potential[(2 * sr + sc) * m * m + a * m + b];
                                }
                            }
                        }
                    }
                }
            }
            hamilt::dftu_gpu::ProjectionTask task{};
            task.right_offset = right_offset;
            task.dm_offset = 7;
            task.hr_offset = 11;
            task.row_orbitals = rr.size();
            task.col_orbitals = cc.size();
            task.matrix_cols = stride;
            task.projector_size = m;
            task.has_dm = 1;
            task.has_hr = 1;
            void* cache = hamilt::dftu_gpu::create_cache(projections.data(), projections.size(), &task, 1,
                                                       local_dm.size(), hr.size(), expected.size());
            std::vector<double> partial(expected.size());
            hamilt::dftu_gpu::compute_occupations(cache, local_dm.data(), partial.data());
            hamilt::dftu_gpu::add_hubbard_hamiltonian(cache, potential.data(), hr.data());
            const auto* device_addend = hamilt::dftu_gpu::build_hubbard_hamiltonian(cache, potential.data());
            std::vector<std::complex<double>> addend(hr.size());
            ASSERT_EQ(cudaMemcpy(addend.data(),
                                 device_addend,
                                 addend.size() * sizeof(std::complex<double>),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            for (std::size_t i = 0; i < partial.size(); ++i) { actual[i] += partial[i]; }
            for (std::size_t i = 0; i < hr.size(); ++i)
            {
                EXPECT_NEAR(hr[i].real(), expected_hr[i].real(), 1e-10);
                EXPECT_NEAR(hr[i].imag(), expected_hr[i].imag(), 1e-10);
                EXPECT_NEAR(addend[i].real(), expected_hr[i].real() - 0.25, 1e-10);
                EXPECT_NEAR(addend[i].imag(), expected_hr[i].imag() + 0.1, 1e-10);
            }
            hamilt::dftu_gpu::destroy_cache(cache);
        }
    }
    for (std::size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], expected[i], 1e-10); }
}

TEST(DFTULcaoGPU, EmptyLocalWorkProducesZeroOccupation)
{
    double placeholder = 0.0;
    std::complex<double> hr(1.0, 2.0);
    std::vector<double> occupations(4, 1.0);
    std::vector<std::complex<double>> onsite(4, 0.0);
    void* cache = hamilt::dftu_gpu::create_cache(&placeholder, 0, nullptr, 0, 0, 1, 4);
    hamilt::dftu_gpu::compute_occupations(cache, &placeholder, occupations.data());
    hamilt::dftu_gpu::add_hubbard_hamiltonian(cache, onsite.data(), &hr);
    for (double value : occupations) { EXPECT_DOUBLE_EQ(value, 0.0); }
    EXPECT_EQ(hr, std::complex<double>(1.0, 2.0));
    hamilt::dftu_gpu::destroy_cache(cache);
}
