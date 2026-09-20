#include "source_hsolver/kernels/hegvd_op.h"
#include "source_base/module_device/memory_op.h"
#include <gtest/gtest.h>
#include <complex>
#include <stdexcept>
#include <string>

#if defined(__UT_USE_CUDA)
TEST(HegvdOp, ReportsNonPositiveOverlapInRelease)
{
    using Complex = std::complex<double>;
    using GPU = base_device::DEVICE_GPU;
    using CPU = base_device::DEVICE_CPU;
    Complex* a = nullptr;
    Complex* b = nullptr;
    Complex* v = nullptr;
    double* eigenvalues = nullptr;
    base_device::memory::resize_memory_op<Complex, GPU>()(a, 4);
    base_device::memory::resize_memory_op<Complex, GPU>()(b, 4);
    base_device::memory::resize_memory_op<Complex, GPU>()(v, 4);
    base_device::memory::resize_memory_op<double, GPU>()(eigenvalues, 2);
    const Complex host_a[] = {1.0, 0.0, 0.0, 2.0};
    const Complex host_b[] = {1.0, 0.0, 0.0, -1.0};
    base_device::memory::synchronize_memory_op<Complex, GPU, CPU>()(a, host_a, 4);
    base_device::memory::synchronize_memory_op<Complex, GPU, CPU>()(b, host_b, 4);
    hsolver::createGpuSolverHandle();
    try
    {
        hsolver::hegvd_op<Complex, GPU>()(nullptr, 2, 2, a, b, eigenvalues, v);
        ADD_FAILURE() << "An indefinite overlap must not return eigenvectors";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_NE(std::string(error.what()).find("info=4"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("n=2"), std::string::npos);
    }
    hsolver::destroyGpuSolverHandle();
    base_device::memory::delete_memory_op<Complex, GPU>()(a);
    base_device::memory::delete_memory_op<Complex, GPU>()(b);
    base_device::memory::delete_memory_op<Complex, GPU>()(v);
    base_device::memory::delete_memory_op<double, GPU>()(eigenvalues);
}
#endif
