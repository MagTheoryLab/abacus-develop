#include "lcao_ritz_gpu.h"

#include "source_base/constants.h"
#include "source_base/tool_quit.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <tuple>

namespace spinconstrain
{
namespace
{
using Z = cuDoubleComplex;
constexpr int threads = 256;

void check(int status, const char* operation)
{
    if (status != 0)
    {
        ModuleBase::WARNING_QUIT("LcaoRitzGpu", std::string(operation) + " failed: " + std::to_string(status));
    }
}

struct DeviceFree
{
    void operator()(void* data) const { cudaFree(data); }
};

template <typename T>
using Device = std::unique_ptr<T, DeviceFree>;

template <typename T>
Device<T> allocate(size_t size)
{
    T* data = nullptr;
    if (size != 0)
    {
        check(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)), "allocate");
    }
    return Device<T>(data);
}

template <typename T>
Device<T> upload(const T* data, size_t size)
{
    auto out = allocate<T>(size);
    if (size != 0)
    {
        check(cudaMemcpy(out.get(), data, size * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    return out;
}

int grid_size(int count)
{
    return count / threads + (count % threads != 0);
}

struct Omit
{
    int atom;
    int row;
    int col;
    int rows;
    int cols;
    int projectors;
    int left;
    int right;
    Z phase;
};

struct Tile
{
    int begin;
    int end;
    int offset;
};

__global__ void split_spinors(const Z* c, Z* split, int n, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        const int orbital = i % n;
        const int band = i / n;
        split[band * n + (orbital % 2) * (n / 2) + orbital / 2] = c[i];
    }
}

__global__ void apply_lambda(const Z* d, const double* lambda, const int* atoms,
                             Z* ld, int projectors, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int row = i % projectors;
    const int band = i / projectors;
    const int up = band * (2 * projectors) + row;
    const int down = up + projectors;
    const int atom = atoms[row];
    const double x = lambda[3 * atom];
    const double y = lambda[3 * atom + 1];
    const double z = lambda[3 * atom + 2];
    ld[up] = cuCadd(cuCmul(make_cuDoubleComplex(z, 0), d[up]),
                   cuCmul(make_cuDoubleComplex(x, -y), d[down]));
    ld[down] = cuCsub(cuCmul(make_cuDoubleComplex(x, y), d[up]),
                     cuCmul(make_cuDoubleComplex(z, 0), d[down]));
}

__device__ double projector_product(const double* factors, const Omit& block, int i, int j)
{
    double result = 0;
    const double* left = factors + block.left + i * block.projectors;
    const double* right = factors + block.right + j * block.projectors;
    for (int p = 0; p < block.projectors; ++p)
    {
        result += left[p] * right[p];
    }
    return result;
}

__global__ void subtract_omitted(const Omit* omitted, const Tile* tiles,
                                 const double* factors, const double* lambda, Z* dense, int n)
{
    const Tile tile = tiles[blockIdx.x];
    const Omit first = omitted[tile.begin];
    const int index = tile.offset + threadIdx.x;
    if (index >= first.rows * first.cols) return;
    const int i = index % first.rows;
    const int j = index / first.rows;
    Z x = {};
    Z y = {};
    Z z = {};
    for (int p = tile.begin; p < tile.end; ++p)
    {
        const Omit block = omitted[p];
        const double value = projector_product(factors, block, i, j);
        const Z coefficient = make_cuDoubleComplex(value * block.phase.x, value * block.phase.y);
        x = cuCadd(x, cuCmul(coefficient, make_cuDoubleComplex(lambda[3 * block.atom], 0)));
        y = cuCadd(y, cuCmul(coefficient, make_cuDoubleComplex(lambda[3 * block.atom + 1], 0)));
        z = cuCadd(z, cuCmul(coefficient, make_cuDoubleComplex(lambda[3 * block.atom + 2], 0)));
    }
    const int pos = 2 * (first.row + i) + 2 * (first.col + j) * n;
    dense[pos] = make_cuDoubleComplex(-z.x, -z.y);
    dense[pos + n] = make_cuDoubleComplex(-x.x - y.y, -x.y + y.x);
    dense[pos + 1] = make_cuDoubleComplex(-x.x + y.y, -x.y - y.x);
    dense[pos + n + 1] = z;
}

__global__ void add_diagonal(Z* h, const double* e, int bands)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bands) h[i * (bands + 1)].x += e[i];
}

__global__ void spin_moments(const Z* d, const double* w, const int* atom_start,
                             int projectors, int bands, double* result)
{
    __shared__ double shared_x[threads];
    __shared__ double shared_y[threads];
    __shared__ double shared_z[threads];
    const int atom = blockIdx.y;
    const int band = blockIdx.x * blockDim.x + threadIdx.x;
    double x = 0;
    double y = 0;
    double z = 0;
    if (band < bands)
    {
        const int start = band * 2 * projectors;
        for (int i = atom_start[atom]; i < atom_start[atom + 1]; ++i)
        {
            const Z up = d[start + i];
            const Z down = d[start + i + projectors];
            const Z cross = cuCmul(cuConj(up), down);
            x += 2 * cross.x;
            y += 2 * cross.y;
            z += up.x * up.x + up.y * up.y - down.x * down.x - down.y * down.y;
        }
        x *= w[band];
        y *= w[band];
        z *= w[band];
    }
    const int i = threadIdx.x;
    shared_x[i] = x;
    shared_y[i] = y;
    shared_z[i] = z;
    __syncthreads();
    for (int off = threads / 2; off > 0; off /= 2)
    {
        if (i < off)
        {
            shared_x[i] += shared_x[i + off];
            shared_y[i] += shared_y[i + off];
            shared_z[i] += shared_z[i + off];
        }
        __syncthreads();
    }
    if (i == 0)
    {
        atomicAdd(result + 3 * atom, shared_x[0]);
        atomicAdd(result + 3 * atom + 1, shared_y[0]);
        atomicAdd(result + 3 * atom + 2, shared_z[0]);
    }
}

__global__ void conjugate_weighted(const Z* c, const double* w, Z* weighted, int n, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        const Z z = c[i];
        const double f = w[i / n];
        weighted[i] = make_cuDoubleComplex(z.x * f, -z.y * f);
    }
}

__global__ void subtract_omitted_moments(const Omit* omitted, const double* factors, const Z* dm, int n, double* result)
{
    __shared__ double sx[threads];
    __shared__ double sy[threads];
    __shared__ double sz[threads];
    const Omit block = omitted[blockIdx.x];
    double x = 0;
    double y = 0;
    double z = 0;
    const int tid = threadIdx.x;
    for (int t = tid; t < block.rows * block.cols; t += threads)
    {
        const int i = t % block.rows;
        const int j = t / block.rows;
        const int pos = 2 * (block.row + i) + 2 * (block.col + j) * n;
        const double value = projector_product(factors, block, i, j);
        const Z uu = cuCmul(block.phase, dm[pos]);
        const Z ud = cuCmul(block.phase, dm[pos + n]);
        const Z du = cuCmul(block.phase, dm[pos + 1]);
        const Z dd = cuCmul(block.phase, dm[pos + n + 1]);
        x -= value * (ud.x + du.x);
        y -= value * (ud.y - du.y);
        z -= value * (uu.x - dd.x);
    }
    sx[tid] = x;
    sy[tid] = y;
    sz[tid] = z;
    __syncthreads();
    for (int stride = threads / 2; stride != 0; stride /= 2)
    {
        if (tid < stride)
        {
            sx[tid] += sx[tid + stride];
            sy[tid] += sy[tid + stride];
            sz[tid] += sz[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        atomicAdd(result + 3 * block.atom, sx[0]);
        atomicAdd(result + 3 * block.atom + 1, sy[0]);
        atomicAdd(result + 3 * block.atom + 2, sz[0]);
    }
}

struct FoldedProjectors
{
    Device<Z> projector;
    Device<Z> mixed;
    Device<Z> rotation;
    Device<Omit> omitted;
    Device<Tile> tiles;
    int omitted_count = 0;
    int tile_count = 0;
};
} // namespace

struct LcaoRitzGpu::Impl
{
    Impl(int nlocal, int nbands, int ritz_bands, int kpoints, const RitzProjectorGeometry& geometry)
        : n(nlocal), bands(nbands), ritz(ritz_bands), k(kpoints)
    {
        std::vector<int> owner;
        std::vector<int> starts(1, 0);
        for (int atom = 0; atom < geometry.projector_counts.size(); ++atom)
        {
            owner.insert(owner.end(), geometry.projector_counts[atom], atom);
            starts.push_back(owner.size());
        }
        nat = geometry.projector_counts.size();
        projectors = owner.size();
        factors = upload(geometry.factors.data(), geometry.factors.size());
        owners = upload(owner.data(), owner.size());
        atom_start = upload(starts.data(), starts.size());
        omitted = geometry.omitted;
        std::stable_sort(omitted.begin(), omitted.end(), [](const RitzOmittedPair& a, const RitzOmittedPair& b)
        {
            return std::tie(a.row, a.col, a.atom, a.translation.x, a.translation.y, a.translation.z)
                   < std::tie(b.row, b.col, b.atom, b.translation.x, b.translation.y, b.translation.z);
        });

        split = allocate<Z>(size_t(n) * ritz);
        projected = allocate<Z>(size_t(2) * projectors * ritz);
        perturbed = allocate<Z>(size_t(2) * projectors * ritz);
        reduced = allocate<Z>(size_t(ritz) * ritz);
        if (!omitted.empty())
        {
            dense = allocate<Z>(size_t(n) * n);
            auxiliary = allocate<Z>(size_t(n) * ritz);
            vector_out = allocate<Z>(size_t(n) * bands);
        }
        reference_energy = allocate<double>(ritz);
        trial_energy = allocate<double>(ritz);
        lambda = allocate<double>(size_t(3) * nat);
        weights = allocate<double>(bands);
        moment = allocate<double>(size_t(3) * nat);
        info = allocate<int>(1);
        check(cublasCreate(&blas), "cublasCreate");
        check(cusolverDnCreate(&eig), "cusolverDnCreate");
        check(cusolverDnZheevd_bufferSize(eig, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                        ritz, reduced.get(), ritz, trial_energy.get(), &work_size), "heevd workspace");
        work = allocate<Z>(work_size);
    }

    ~Impl()
    {
        cusolverDnDestroy(eig);
        cublasDestroy(blas);
    }

    void gemm(cublasOperation_t a, cublasOperation_t b, int rows, int cols, int inner,
              const Z* left, int ldl, const Z* right, int ldr, Z* out, const Z& beta)
    {
        check(cublasZgemm(blas, a, b, rows, cols, inner, &one, left, ldl, right, ldr, &beta, out, rows), "zgemm");
    }

    void project(const FoldedProjectors& slot, const Z* vectors, int columns)
    {
        const int count = n * columns;
        split_spinors<<<grid_size(count), threads>>>(vectors, split.get(), n, count);
        check(cudaGetLastError(), "split spinors");
        gemm(CUBLAS_OP_N, CUBLAS_OP_N, projectors, 2 * columns, n / 2,
             slot.projector.get(), projectors, split.get(), n / 2, projected.get(), zero);
    }

    const int n;
    const int bands;
    const int ritz;
    int nat = 0;
    int projectors = 0;
    int work_size = 0;
    std::vector<FoldedProjectors> k;
    std::vector<RitzOmittedPair> omitted;
    Device<double> factors;
    Device<int> owners;
    Device<int> atom_start;
    Device<int> info;
    Device<Z> split;
    Device<Z> projected;
    Device<Z> perturbed;
    Device<Z> reduced;
    Device<Z> dense;
    Device<Z> auxiliary;
    Device<Z> vector_out;
    Device<Z> work;
    Device<double> reference_energy;
    Device<double> trial_energy;
    Device<double> lambda;
    Device<double> weights;
    Device<double> moment;
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t eig = nullptr;
    const Z one = make_cuDoubleComplex(1.0, 0.0);
    const Z zero = make_cuDoubleComplex(0.0, 0.0);
};

LcaoRitzGpu::LcaoRitzGpu(int nlocal, int bands, int ritz_bands, int kpoints, const RitzProjectorGeometry& geometry)
{
    impl_.reset(new Impl(nlocal, bands, ritz_bands, kpoints, geometry));
}

LcaoRitzGpu::~LcaoRitzGpu() = default;

void LcaoRitzGpu::prepare(int ik, const std::complex<double>* projectors, const ModuleBase::Vector3<double>& kpoint)
{
    auto& p = *impl_;
    auto& slot = p.k[ik];
    const size_t extent = size_t(p.projectors) * (p.n / 2);
    slot.projector = upload(reinterpret_cast<const Z*>(projectors), extent);
    slot.mixed = allocate<Z>(size_t(2) * p.projectors * p.bands);
    if (!p.omitted.empty())
    {
        slot.rotation = allocate<Z>(size_t(p.ritz) * p.bands);
    }
    std::vector<Omit> folded;
    std::vector<Tile> tiles;
    int begin = 0;
    while (begin < p.omitted.size())
    {
        const auto& representative = p.omitted[begin];
        int end = begin;
        while (end < p.omitted.size() && p.omitted[end].row == representative.row
               && p.omitted[end].col == representative.col)
        {
            const auto& b = p.omitted[end];
            const double kr = kpoint.x * b.translation.x + kpoint.y * b.translation.y + kpoint.z * b.translation.z;
            const double angle = ModuleBase::TWO_PI * kr;
            folded.push_back({b.atom, b.row, b.col, b.rows, b.cols, b.projectors, b.left, b.right,
                              make_cuDoubleComplex(std::cos(angle), std::sin(angle))});
            ++end;
        }
        const int count = representative.rows * representative.cols;
        for (int offset = 0; offset < count; offset += threads)
        {
            tiles.push_back({begin, end, offset});
        }
        begin = end;
    }
    slot.omitted = upload(folded.data(), folded.size());
    slot.tiles = upload(tiles.data(), tiles.size());
    slot.omitted_count = folded.size();
    slot.tile_count = tiles.size();
}

void LcaoRitzGpu::solve(int ik, const std::complex<double>* waves, const double* ref,
                      const double* delta, double* eigen, bool keep_rotation)
{
    auto& p = *impl_;
    auto& slot = p.k[ik];
    const auto* vectors = reinterpret_cast<const Z*>(waves);
    check(cudaMemcpy(p.lambda.get(), delta, size_t(3) * p.nat * sizeof(double), cudaMemcpyHostToDevice), "lambda");
    check(cudaMemcpy(p.reference_energy.get(), ref, p.ritz * sizeof(double), cudaMemcpyHostToDevice), "reference energy");
    p.project(slot, vectors, p.ritz);
    const int count = p.projectors * p.ritz;
    const int rows = 2 * p.projectors;
    apply_lambda<<<grid_size(count), threads>>>(p.projected.get(), p.lambda.get(), p.owners.get(),
                                                p.perturbed.get(), p.projectors, count);
    check(cudaGetLastError(), "projected lambda");
    p.gemm(CUBLAS_OP_C, CUBLAS_OP_N, p.ritz, p.ritz, rows,
           p.projected.get(), rows, p.perturbed.get(), rows, p.reduced.get(), p.zero);
    if (slot.omitted_count != 0)
    {
        check(cudaMemset(p.dense.get(), 0, size_t(p.n) * p.n * sizeof(Z)), "zero H cutoff correction");
        subtract_omitted<<<slot.tile_count, threads>>>(slot.omitted.get(), slot.tiles.get(), p.factors.get(),
                                                       p.lambda.get(), p.dense.get(), p.n);
        check(cudaGetLastError(), "H cutoff correction");
        p.gemm(CUBLAS_OP_N, CUBLAS_OP_N, p.n, p.ritz, p.n,
               p.dense.get(), p.n, vectors, p.n, p.auxiliary.get(), p.zero);
        p.gemm(CUBLAS_OP_C, CUBLAS_OP_N, p.ritz, p.ritz, p.n,
               vectors, p.n, p.auxiliary.get(), p.n, p.reduced.get(), p.one);
    }
    add_diagonal<<<grid_size(p.ritz), threads>>>(p.reduced.get(), p.reference_energy.get(), p.ritz);
    check(cudaGetLastError(), "add reference spectrum");
    check(cusolverDnZheevd(p.eig, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                           p.ritz, p.reduced.get(), p.ritz, p.trial_energy.get(),
                           p.work.get(), p.work_size, p.info.get()), "heevd");
    int info = 0;
    check(cudaMemcpy(&info, p.info.get(), sizeof(int), cudaMemcpyDeviceToHost), "heevd info");
    check(info, "heevd result");
    check(cudaMemcpy(eigen, p.trial_energy.get(), p.bands * sizeof(double), cudaMemcpyDeviceToHost), "trial energy");
    p.gemm(CUBLAS_OP_N, CUBLAS_OP_N, rows, p.bands, p.ritz,
           p.projected.get(), rows, p.reduced.get(), p.ritz, slot.mixed.get(), p.zero);
    if (keep_rotation && !slot.rotation)
    {
        slot.rotation = allocate<Z>(size_t(p.ritz) * p.bands);
    }
    if (slot.rotation)
    {
        check(cudaMemcpy(slot.rotation.get(), p.reduced.get(), size_t(p.ritz) * p.bands * sizeof(Z),
                         cudaMemcpyDeviceToDevice), "save cutoff rotation");
    }
}

void LcaoRitzGpu::publish_wavefunctions(int ik, std::complex<double>* waves)
{
    auto& p = *impl_;
    const auto& slot = p.k[ik];
    auto* vectors = reinterpret_cast<Z*>(waves);
    // The spin-split scratch is free after the final projected solve.
    p.gemm(CUBLAS_OP_N, CUBLAS_OP_N, p.n, p.bands, p.ritz,
           vectors, p.n, slot.rotation.get(), p.ritz, p.split.get(), p.zero);
    const size_t bytes = size_t(p.n) * p.bands * sizeof(Z);
    check(cudaMemcpy(vectors, p.split.get(), bytes, cudaMemcpyDeviceToDevice), "publish Ritz wavefunctions");
}

void LcaoRitzGpu::moments(int ik, const std::complex<double>* waves, const double* wg,
                        bool reference, double* output)
{
    auto& p = *impl_;
    const auto& slot = p.k[ik];
    const auto* vectors = reinterpret_cast<const Z*>(waves);
    check(cudaMemcpy(p.weights.get(), wg, p.bands * sizeof(double), cudaMemcpyHostToDevice), "weights");
    check(cudaMemset(p.moment.get(), 0, size_t(3) * p.nat * sizeof(double)), "zero moments");
    if (reference)
    {
        p.project(slot, vectors, p.bands);
    }
    const auto* proj = reference ? p.projected.get() : slot.mixed.get();
    const dim3 grid(grid_size(p.bands), p.nat);
    spin_moments<<<grid, threads>>>(proj, p.weights.get(), p.atom_start.get(), p.projectors, p.bands, p.moment.get());
    check(cudaGetLastError(), "projector moments");
    if (slot.omitted_count != 0)
    {
        if (!reference)
        {
            p.gemm(CUBLAS_OP_N, CUBLAS_OP_N, p.n, p.bands, p.ritz,
                   vectors, p.n, slot.rotation.get(), p.ritz, p.vector_out.get(), p.zero);
            vectors = p.vector_out.get();
        }
        const int count = p.n * p.bands;
        conjugate_weighted<<<grid_size(count), threads>>>(vectors, p.weights.get(), p.auxiliary.get(), p.n, count);
        check(cudaGetLastError(), "weight wavefunctions");
        p.gemm(CUBLAS_OP_N, CUBLAS_OP_T, p.n, p.n, p.bands,
               p.auxiliary.get(), p.n, vectors, p.n, p.dense.get(), p.zero);
        subtract_omitted_moments<<<slot.omitted_count, threads>>>(slot.omitted.get(), p.factors.get(),
                                                                p.dense.get(), p.n, p.moment.get());
        check(cudaGetLastError(), "moment cutoff correction");
    }
    std::vector<double> values(3 * p.nat);
    check(cudaMemcpy(values.data(), p.moment.get(), values.size() * sizeof(double), cudaMemcpyDeviceToHost), "moments");
    for (int i = 0; i < values.size(); ++i) output[i] += values[i];
}
}
