#include "gtest/gtest.h"
#include "source_base/module_device/device_check.h"
#include "source_lcao/module_deltaspin/lcao_ritz_gpu.h"

#include <complex>
#include <memory>
#include <vector>

TEST(LcaoRitzGpu, MixedBlockWidthsPreserveRealSpaceCutoff)
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    using Complex = std::complex<double>;
    const int nlocal = 6;
    const int bands = 4;
    const int ritz_bands = 5;
    const int kpoints = 3;
    const int owned_k = 2;
    spinconstrain::RitzProjectorGeometry geometry;
    geometry.projector_counts = {1, 2};
    geometry.factors = {1.0, 0.5, 0.0};
    const ModuleBase::Vector3<int> origin(0, 0, 0);
    geometry.omitted = {
        {0, 0, 1, 1, 2, 1, 0, 1, origin},
        {0, 1, 0, 2, 1, 1, 1, 0, origin},
    };
    const ModuleBase::Vector3<double> k(0.125, 0.0, 0.0);
    spinconstrain::LcaoRitzGpu ritz(nlocal, bands, ritz_bands, kpoints, geometry);
    const Complex projector[] = {1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0};
    ritz.prepare(owned_k, projector, k);

    std::vector<Complex> columns(nlocal * ritz_bands);
    for (int i = 0; i < ritz_bands; ++i)
    {
        columns[i * (nlocal + 1)] = 1.0;
    }
    Complex* columns_device = nullptr;
    const size_t bytes = columns.size() * sizeof(Complex);
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&columns_device), bytes));
    std::unique_ptr<Complex, decltype(&cudaFree)> storage(columns_device, cudaFree);
    CHECK_CUDA(cudaMemcpy(columns_device, columns.data(), bytes, cudaMemcpyHostToDevice));

    const double reference[] = {0.0, 0.4, 1.0, 1.4, 2.0};
    const double lambda[] = {0.0, 0.0, 0.1, 0.0, 0.0, 0.0};
    const double weights[] = {1.0, 0.0, 0.5, 0.0};
    const double expected_energy[] = {0.1, 0.3, 1.025, 1.375};
    double eigen[bands] = {};
    ritz.solve(owned_k, columns_device, reference, lambda, eigen, true);
    for (int i = 0; i < bands; ++i)
    {
        EXPECT_NEAR(eigen[i], expected_energy[i], 1e-12);
    }
    const double expected_moments[] = {0.0, 0.0, 1.125, 0.0, 0.0, 0.5};
    for (bool is_reference : {true, false})
    {
        double moments[6] = {};
        ritz.moments(owned_k, columns_device, weights, is_reference, moments);
        for (int i = 0; i < 6; ++i)
        {
            EXPECT_NEAR(moments[i], expected_moments[i], 1e-12);
        }
    }
    ritz.publish_wavefunctions(owned_k, columns_device);
    double published_moments[6] = {};
    ritz.moments(owned_k, columns_device, weights, true, published_moments);
    for (int i = 0; i < 6; ++i)
    {
        EXPECT_NEAR(published_moments[i], expected_moments[i], 1e-12);
    }
}

TEST(LcaoRitzGpu, PublishesRotationWithoutOmittedPairs)
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    using Complex = std::complex<double>;
    spinconstrain::RitzProjectorGeometry geometry;
    geometry.projector_counts = {1};
    const ModuleBase::Vector3<double> gamma(0.0, 0.0, 0.0);
    const Complex projector = 1.0;
    spinconstrain::LcaoRitzGpu ritz(2, 1, 2, 1, geometry);
    ritz.prepare(0, &projector, gamma);

    Complex columns[] = {1.0, 0.0, 0.0, 1.0};
    Complex* device_columns = nullptr;
    const size_t bytes = sizeof(columns);
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_columns), bytes));
    std::unique_ptr<Complex, decltype(&cudaFree)> storage(device_columns, cudaFree);
    CHECK_CUDA(cudaMemcpy(device_columns, columns, bytes, cudaMemcpyHostToDevice));

    const double reference[] = {0.0, 0.0};
    const double lambda[] = {0.25, 0.0, 0.0};
    double energy = 0.0;
    ritz.solve(0, device_columns, reference, lambda, &energy, true);
    ritz.publish_wavefunctions(0, device_columns);
    CHECK_CUDA(cudaMemcpy(columns, device_columns, bytes, cudaMemcpyDeviceToHost));
    const Complex coherence = std::conj(columns[0]) * columns[1];
    EXPECT_NEAR(energy, -0.25, 1e-12);
    EXPECT_NEAR(std::norm(columns[0]), 0.5, 1e-12);
    EXPECT_NEAR(std::norm(columns[1]), 0.5, 1e-12);
    EXPECT_NEAR(coherence.real(), -0.5, 1e-12);
    EXPECT_NEAR(coherence.imag(), 0.0, 1e-12);
    EXPECT_EQ(columns[2], Complex(0.0));
    EXPECT_EQ(columns[3], Complex(1.0));
}
