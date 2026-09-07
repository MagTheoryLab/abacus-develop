#include "gtest/gtest.h"
#include "source_hsolver/kernels/cuda/k_owner_matrix_gpu.h"
#include "source_hsolver/kernels/cuda/diag_cusolver.cuh"
#include "source_base/module_device/device_check.h"
#include "source_base/parallel_2d.h"
#include "source_base/parallel_global.h"
#include <vector>

namespace
{
int test_rank = 0;
int test_size = 1;
using Complex = std::complex<double>;
Complex h_value(int i, int j, int pass)
{
    if (i == j) return 20.0 + i + pass;
    return Complex(0.01 * (i + j + pass), 0.005 * (i - j));
}
Complex s_value(int i, int j)
{
    return i == j ? 1.0 + 0.01 * i : 0.0;
}
}

TEST(KOwnerMatrixGpu, UpperTriangleRotatingOwnerAndDeviceSolve)
{
    int devices = 0;
    CHECK_CUDA(cudaGetDeviceCount(&devices));
    ASSERT_GT(devices, 0);
    CHECK_CUDA(cudaSetDevice(test_rank % devices));
    const int n = 13;
    Parallel_2D layout;
    layout.init(n, n, 2, MPI_COMM_WORLD);
    hsolver::KOwnerMatrixGpu gather(layout, test_rank);
    std::vector<Complex> h(layout.get_local_size());
    std::vector<Complex> s(h.size());
    for (int pass = 0; pass < 3; ++pass)
    {
        for (int c = 0; c < layout.get_col_size(); ++c)
            for (int r = 0; r < layout.get_row_size(); ++r)
            {
                const int i = layout.local2global_row(r);
                const int j = layout.local2global_col(c);
                const int offset = c * layout.get_row_size() + r;
                h[offset] = i <= j ? h_value(i, j, pass) : Complex(-9999.0, 123.0);
                s[offset] = i <= j ? s_value(i, j) : Complex(-9999.0, 123.0);
            }
        for (int owner = 0; owner < test_size; ++owner)
        {
            gather.gather(h.data(), s.data(), owner);
            if (test_rank != owner) continue;
            std::vector<Complex> full_h(n * n);
            std::vector<Complex> full_s(n * n);
            CHECK_CUDA(cudaMemcpy(full_h.data(), gather.h_device(), n * n * sizeof(Complex), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(full_s.data(), gather.s_device(), n * n * sizeof(Complex), cudaMemcpyDeviceToHost));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                {
                    EXPECT_EQ(full_h[j * n + i], i <= j ? h_value(i, j, pass) : Complex(0.0));
                    EXPECT_EQ(full_s[j * n + i], i <= j ? s_value(i, j) : Complex(0.0));
                }
            Complex* vectors_device = nullptr;
            CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&vectors_device), n * n * sizeof(Complex)));
            std::vector<double> eigen(n);
            Diag_Cusolver_gvd solver;
            solver.Dngvd_device_input(n, gather.h_device(), gather.s_device(), eigen.data(), vectors_device, n);
            std::vector<Complex> vectors(n * n);
            CHECK_CUDA(cudaMemcpy(vectors.data(), vectors_device, n * n * sizeof(Complex), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaFree(vectors_device));
            // Independent scalar generalized-eigenvector residual (no packing helper).
            for (int band = 0; band < n; ++band)
                for (int i = 0; i < n; ++i)
                {
                    Complex residual = 0.0;
                    for (int j = 0; j < n; ++j)
                        residual += (h_value(i, j, pass) - eigen[band] * s_value(i, j)) * vectors[band * n + j];
                    EXPECT_LT(std::abs(residual), 1e-10);
                }
        }
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
