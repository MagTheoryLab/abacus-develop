#ifndef DIAG_CUSOLVER_CUH
#define DIAG_CUSOLVER_CUH
#include <cuda.h>
#include <complex>

#include <cuda_runtime.h>
#include <cusolverDn.h>


///
/// The diag_cusolver contains subroutines for generalized eigenvector problems of dense matrice in lcao on a single GPU.
/// Cusolver apis, i.e. cusolverDnDsygvd and cusolverDnZhegvd, are wrapped with respect to different data types for easy calling.
/// For details of low-level methods relevent, one can refer to [cuSOLVER documentation](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-sygvd).
///
// Xu Shu add 2022-03-23
//

class Diag_Cusolver_gvd{

//-------------------
// private variables
//-------------------

    cusolverDnHandle_t cusolverH = nullptr;

    cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; //problem type: A*x = (lambda)*B*x
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    int m = 0;
    int lda = 0;

    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_work = nullptr;

    cuDoubleComplex *d_A2 = nullptr;
    cuDoubleComplex *d_B2 = nullptr;
    cuDoubleComplex *d_work2 = nullptr;

    double *d_W = nullptr;
    int *devInfo = nullptr;

    int lwork_double = 0;
    int lwork_complex = 0;
    int info_gpu = 0;

    void prepare_dimension(int N);
    void solve_double(int N, int M, double* A, double* B, double* W);
    void solve_complex(int N, int M, std::complex<double>* A, std::complex<double>* B, double* W);
    void solve_double_inplace(double* A, double* B, double* W);
    void solve_complex_inplace(cuDoubleComplex* A, cuDoubleComplex* B, double* W);

    void finalize();  // for recycling the usage of the static class Diag_Cusolver_gvd
public:

    // H/S and V are device buffers; W is a host output. H/S are overwritten.
    void Dngvd_device_inplace(int N, double* A, double* B, double* W, double* V, int nvec);
    void Dngvd_device_inplace(int N, std::complex<double>* A, std::complex<double>* B,
                             double* W, std::complex<double>* V, int nvec);

    int is_init = 0;    // For expensive gpu initialization only once when using cusolver for lcao

    Diag_Cusolver_gvd();
    ~Diag_Cusolver_gvd();

//   subroutines that are related to calculating generalized eigenvalues and eigenvectors for dense matrix pairs:
//  - Dngvd_double : dense double type matrix
//  - Dngvd_complex : dense complex type matrix
//      Input Parameters
//          N: the number of rows of the matrix
//          M: the number of cols of the matrix
//          A: the hermitian matrix A in A x=lambda B (column major)
//          B: the SPD matrix B in A x=lambda B (column major)
//      Output Parameter
//          W: generalized eigenvalues
//          V: generalized eigenvectors (column major)

    void Dngvd_double(int N, int M, double *A, double *B, double *W, double *V);
    void Dngvd_complex(int N, int M, std::complex<double> *A, std::complex<double> *B, double *W, std::complex<double> *V);
    void Dngvd_double_device(int N, int M, double* A, double* B, double* W, double* V, int nvec);
    void Dngvd_complex_device(int N,
                              int M,
                              std::complex<double>* A,
                              std::complex<double>* B,
                              double* W,
                              std::complex<double>* V,
                              int nvec);

    void Dngvd(int N, int M, double *A, double *B, double *W, double *V)
    {
        return Dngvd_double(N, M, A, B, W, V);
    };

    void Dngvd(int N, int M, std::complex<double> *A, std::complex<double> *B, double *W, std::complex<double> *V)
    {
        return Dngvd_complex(N, M, A, B, W, V);
    };

    void Dngvd_device(int N, int M, double* A, double* B, double* W, double* V, int nvec)
    {
        return Dngvd_double_device(N, M, A, B, W, V, nvec);
    };

    void Dngvd_device(int N,
                      int M,
                      std::complex<double>* A,
                      std::complex<double>* B,
                      double* W,
                      std::complex<double>* V,
                      int nvec)
    {
        return Dngvd_complex_device(N, M, A, B, W, V, nvec);
    };

};

#endif
