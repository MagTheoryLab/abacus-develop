#include "source_pw/module_pwdft/kernels/stress_op.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

TEST(StressLocGPU, RadialIntegralMatchesScalarSimpson)
{
    // Include a nonzero first radius, a partial CUDA block and both G=0 layouts.
    for (int offset : {0, 1})
    {
        const int mesh = 101;
        const int shells = 259;
        const double omega = 37.0;
        const double four_pi = 4.0 * std::acos(-1.0);
        std::vector<double> r(mesh);
        std::vector<double> rho(mesh);
        std::vector<double> rab(mesh, 0.02);
        std::vector<double> g(shells + offset + 1);
        std::vector<double> result(shells);
        for (int i = 0; i < mesh; ++i)
        {
            r[i] = 0.01 + 0.02 * i;
            rho[i] = std::exp(-r[i]) * (1.0 + r[i]);
        }
        for (int i = 0; i < shells; ++i)
        {
            g[i + offset] = 0.15 + 0.04 * i;
        }
        g.back() = 8.0;
        double* buffer = nullptr;
        const std::size_t count = 3 * mesh + g.size() + shells;
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&buffer), count * sizeof(double)));
        double* dr = buffer;
        double* drho = dr + mesh;
        double* drab = drho + mesh;
        double* dg = drab + mesh;
        double* output = dg + g.size();
        ASSERT_EQ(cudaSuccess, cudaMemcpy(dr, r.data(), mesh * sizeof(double), cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(drho, rho.data(), mesh * sizeof(double), cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(drab, rab.data(), mesh * sizeof(double), cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(dg, g.data(), g.size() * sizeof(double), cudaMemcpyHostToDevice));
        hamilt::cal_stress_drhoc_aux_op<double, base_device::DEVICE_GPU>()(
            dr, drho, dg + offset, drab, output, mesh, offset, shells, omega, 3);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(result.data(), output, shells * sizeof(double), cudaMemcpyDeviceToHost));
        ASSERT_EQ(cudaSuccess, cudaFree(buffer));
        for (int i = 0; i < shells; ++i)
        {
            const double q = g[i + offset];
            double integral = 0.0;
            for (int j = 0; j < mesh; ++j)
            {
                const double weight = j == 0 || j == mesh - 1 ? 1.0 : (j % 2 ? 4.0 : 2.0);
                integral += weight * rab[j] * rho[j]
                            * (r[j] * std::cos(q * r[j]) / q - std::sin(q * r[j]) / (q * q));
            }
            const double q2 = q * q;
            const double reference = integral / 3.0 * four_pi / omega / (2.0 * q)
                + four_pi / omega * g.back() * std::exp(-q2 / 4.0) * (q2 / 4.0 + 1.0) / (q2 * q2);
            EXPECT_NEAR(result[i], reference, 1e-11 + 1e-12 * std::abs(reference));
        }
    }
}
