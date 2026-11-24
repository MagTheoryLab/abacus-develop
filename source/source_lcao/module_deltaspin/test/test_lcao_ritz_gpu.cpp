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
    ritz.solve(owned_k, columns_device, reference, lambda, eigen);
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
}
