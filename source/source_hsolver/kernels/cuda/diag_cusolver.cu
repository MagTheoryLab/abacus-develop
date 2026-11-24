#include <assert.h>
#include "diag_cusolver.cuh"
#include "source_base/module_device/device_check.h"

Diag_Cusolver_gvd::Diag_Cusolver_gvd(){
// step 1: create cusolver/cublas handle
    cusolverH = NULL;
    CHECK_CUSOLVER( cusolverDnCreate(&cusolverH) );

    itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
    jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    // LCAO supplies the authoritative upper triangle of the Hermitian H/S matrices.
    uplo = CUBLAS_FILL_MODE_UPPER;

    d_A = NULL;
    d_B = NULL;
    d_work = NULL;

    d_A2 = NULL;
    d_B2 = NULL;
    d_work2 = NULL;

    d_W = NULL;
    devInfo = NULL;

    info_gpu = 0;
    is_init = 0;
}

void Diag_Cusolver_gvd::finalize(){
// free resources and destroy
    if (d_A      ) {CHECK_CUDA( cudaFree(d_A) );  d_A  = NULL;}
    if (d_B      ) {CHECK_CUDA( cudaFree(d_B) );  d_B  = NULL;}
    if (d_A2     ) {CHECK_CUDA( cudaFree(d_A2) ); d_A2 = NULL;}
    if (d_B2     ) {CHECK_CUDA( cudaFree(d_B2) ); d_B2 = NULL;}
    if (d_work   ) {CHECK_CUDA( cudaFree(d_work) ); d_work = NULL;}
    if (d_work2  ) {CHECK_CUDA( cudaFree(d_work2) ); d_work2 = NULL;}
    if (d_W      ) {CHECK_CUDA( cudaFree(d_W) );  d_W  = NULL;}
    if (devInfo  ) {CHECK_CUDA( cudaFree(devInfo) );   devInfo = NULL;}
    m = 0;
    lda = 0;
    lwork_double = 0;
    lwork_complex = 0;
}

Diag_Cusolver_gvd::~Diag_Cusolver_gvd(){
    finalize();
    if (cusolverH) {CHECK_CUSOLVER( cusolverDnDestroy(cusolverH) );    cusolverH = NULL;}
    //CHECK_CUDA( cudaDeviceReset() );
}


void Diag_Cusolver_gvd::prepare_dimension(int N)
{
    if (N == m && d_W != nullptr)
    {
        return;
    }
    this->finalize();
    m = lda = N;
    CHECK_CUDA( cudaMalloc ((void**)&d_W, sizeof(double) * m) );
    CHECK_CUDA( cudaMalloc ((void**)&devInfo, sizeof(int)) );
}

void Diag_Cusolver_gvd::solve_double(int N, int M, double* A, double* B, double* W)
{
    assert(N == M);
    this->prepare_dimension(M);
    const size_t bytes = sizeof(double) * lda * m;
    if (d_A == nullptr)
    {
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes));
    }
    CHECK_CUDA(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));
    this->solve_double_inplace(d_A, d_B, W);
}

void Diag_Cusolver_gvd::solve_double_inplace(double* A, double* B, double* W)
{
    if (d_work == nullptr)
    {
        CHECK_CUSOLVER(cusolverDnDsygvd_bufferSize(
            cusolverH, itype, jobz, uplo, m, A, lda, B, lda, d_W, &lwork_double));
        CHECK_CUDA(cudaMalloc((void**)&d_work, sizeof(double) * lwork_double));
    }
    CHECK_CUSOLVER(cusolverDnDsygvd(
        cusolverH, itype, jobz, uplo, m, A, lda, B, lda, d_W, d_work, lwork_double, devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(W, d_W, sizeof(double) * m, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    assert(0 == info_gpu);
}

void Diag_Cusolver_gvd::solve_complex(
    int N, int M, std::complex<double>* A, std::complex<double>* B, double* W)
{
    assert(N == M);
    this->prepare_dimension(M);
    const size_t bytes = sizeof(cuDoubleComplex) * lda * m;
    if (d_A2 == nullptr)
    {
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_A2), bytes));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_B2), bytes));
    }
    CHECK_CUDA(cudaMemcpy(d_A2, A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B2, B, bytes, cudaMemcpyHostToDevice));
    this->solve_complex_inplace(d_A2, d_B2, W);
}

void Diag_Cusolver_gvd::solve_complex_inplace(cuDoubleComplex* A, cuDoubleComplex* B, double* W)
{
    if (d_work2 == nullptr)
    {
        CHECK_CUSOLVER(cusolverDnZhegvd_bufferSize(
            cusolverH, itype, jobz, uplo, m, A, lda, B, lda, d_W, &lwork_complex));
        CHECK_CUDA(cudaMalloc((void**)&d_work2, sizeof(cuDoubleComplex) * lwork_complex));
    }
    CHECK_CUSOLVER(cusolverDnZhegvd(
        cusolverH, itype, jobz, uplo, m, A, lda, B, lda, d_W, d_work2, lwork_complex, devInfo));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(W, d_W, sizeof(double) * m, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    assert(0 == info_gpu);
}

void Diag_Cusolver_gvd::Dngvd_double(int N, int M, double* A, double* B, double* W, double* V)
{
    this->solve_double(N, M, A, B, W);
    CHECK_CUDA(cudaMemcpy(V, d_A, sizeof(double) * lda * m, cudaMemcpyDeviceToHost));
}

void Diag_Cusolver_gvd::Dngvd_double_device(
    int N, int M, double* A, double* B, double* W, double* V, int nvec)
{
    assert(nvec <= N);
    this->solve_double(N, M, A, B, W);
    CHECK_CUDA(cudaMemcpy(V, d_A, sizeof(double) * lda * nvec, cudaMemcpyDeviceToDevice));
}

void Diag_Cusolver_gvd::Dngvd_complex(
    int N, int M, std::complex<double>* A, std::complex<double>* B, double* W, std::complex<double>* V)
{
    this->solve_complex(N, M, A, B, W);
    CHECK_CUDA(cudaMemcpy(V, d_A2, sizeof(std::complex<double>) * lda * m, cudaMemcpyDeviceToHost));
}

void Diag_Cusolver_gvd::Dngvd_complex_device(
    int N,
    int M,
    std::complex<double>* A,
    std::complex<double>* B,
    double* W,
    std::complex<double>* V,
    int nvec)
{
    assert(nvec <= N);
    this->solve_complex(N, M, A, B, W);
    CHECK_CUDA(cudaMemcpy(V, d_A2, sizeof(std::complex<double>) * lda * nvec, cudaMemcpyDeviceToDevice));
}

void Diag_Cusolver_gvd::Dngvd_device_inplace(int N, double* A, double* B, double* W, double* V, int nvec)
{
    assert(nvec <= N);
    this->prepare_dimension(N);
    this->solve_double_inplace(A, B, W);
    CHECK_CUDA(cudaMemcpy(V, A, sizeof(double) * lda * nvec, cudaMemcpyDeviceToDevice));
}

void Diag_Cusolver_gvd::Dngvd_device_inplace(int N, std::complex<double>* A, std::complex<double>* B,
                                           double* W, std::complex<double>* V, int nvec)
{
    assert(nvec <= N);
    this->prepare_dimension(N);
    auto* matrix_a = reinterpret_cast<cuDoubleComplex*>(A);
    auto* matrix_b = reinterpret_cast<cuDoubleComplex*>(B);
    this->solve_complex_inplace(matrix_a, matrix_b, W);
    CHECK_CUDA(cudaMemcpy(V, A, sizeof(std::complex<double>) * lda * nvec, cudaMemcpyDeviceToDevice));
}
