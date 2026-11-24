#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

namespace
{
using Z = cuDoubleComplex;
using Clock = std::chrono::steady_clock;
using FullSolver = decltype(&cusolverDnZhegvd);
constexpr int n = 3456;
constexpr int m = 1728;
constexpr int nb = 1152;
constexpr int nat = 64;
constexpr int blocks = 4;
constexpr int threads = 256;
constexpr int block_orbitals = 27;
constexpr int block_projectors = 16;
constexpr int factor_size = block_orbitals * block_projectors;
constexpr int pairs_per_thread = (block_orbitals * block_orbitals + threads - 1) / threads;

template <class Status>
void check(Status status, const char* label)
{
    if (status != 0)
    {
        std::fprintf(stderr, "lazy Ritz probe: %s failed (%d)\n", label, int(status));
        std::abort();
    }
}

template <class T>
T* alloc(size_t count)
{
    T* data = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)), "alloc");
    return data;
}

template <class T>
void upload(T* device, const T* host, size_t count)
{
    check(cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice), "upload");
}

struct Raw
{
    int atom;
    int i;
    int j;
    int rx;
    int ry;
    int rz;
    int a;
    int b;
};

struct Omit
{
    int atom;
    int i;
    int j;
    int a;
    int b;
    Z phase;
};

struct Group
{
    int begin;
    int end;
};

__global__ void split_spinors(const Z* c, Z* split, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int orbital = i % n;
    const int source_band = i / n;
    const int target = source_band * n + (orbital % 2) * (n / 2) + orbital / 2;
    split[target] = c[i];
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

__device__ void load_factors(const double* factors, const Omit& block, double* lp, double* rp)
{
    for (int i = threadIdx.x; i < factor_size; i += threads)
    {
        lp[i] = factors[block.a * factor_size + i];
        rp[i] = factors[block.b * factor_size + i];
    }
}

__device__ double projector_product(const double* lp, const double* rp, int i, int j)
{
    double result = 0;
    for (int v = 0; v < block_projectors; ++v)
    {
        result += lp[i * block_projectors + v] * rp[j * block_projectors + v];
    }
    return result;
}

__global__ void subtract_omitted(const Omit* omitted, const Group* groups,
                                 const double* factors, const double* lambda, Z* dense)
{
    __shared__ double left[factor_size];
    __shared__ double right[factor_size];
    const Group range = groups[blockIdx.x];
    Z xs[pairs_per_thread] = {};
    Z ys[pairs_per_thread] = {};
    Z zs[pairs_per_thread] = {};
    for (int p = range.begin; p < range.end; ++p)
    {
        const Omit block = omitted[p];
        load_factors(factors, block, left, right);
        __syncthreads();
        const Z x = cuCmul(block.phase, make_cuDoubleComplex(lambda[3 * block.atom], 0));
        const Z y = cuCmul(block.phase, make_cuDoubleComplex(lambda[3 * block.atom + 1], 0));
        const Z z = cuCmul(block.phase, make_cuDoubleComplex(lambda[3 * block.atom + 2], 0));
        for (int t = 0; t < pairs_per_thread; ++t)
        {
            const int pair = threadIdx.x + t * threads;
            if (pair < block_orbitals * block_orbitals)
            {
                const double v = projector_product(left, right, pair % block_orbitals, pair / block_orbitals);
                const Z value = make_cuDoubleComplex(v,0);
                xs[t] = cuCadd(xs[t], cuCmul(value, x));
                ys[t] = cuCadd(ys[t], cuCmul(value, y));
                zs[t] = cuCadd(zs[t], cuCmul(value, z));
            }
        }
        __syncthreads();
    }
    const Omit block = omitted[range.begin];
    for (int t = 0; t < pairs_per_thread; ++t)
    {
        const int pair = threadIdx.x + t * threads;
        if (pair < block_orbitals * block_orbitals)
        {
            const int i = block.i + pair % block_orbitals;
            const int j = block.j + pair / block_orbitals;
            const int pos = 2 * i + 2 * j * n;
            const Z x = xs[t];
            const Z y = ys[t];
            const Z z = zs[t];
            dense[pos] = make_cuDoubleComplex(-z.x, -z.y);
            dense[pos+n] = make_cuDoubleComplex(-x.x - y.y, -x.y + y.x);
            dense[pos+1] = make_cuDoubleComplex(-x.x + y.y, -x.y - y.x);
            dense[pos+n+1] = z;
        }
    }
}

__global__ void add_diagonal(Z* h, const double* e)
{
    const int band = blockIdx.x * blockDim.x + threadIdx.x;
    if (band < m) h[band * (m + 1)].x += e[band];
}

__global__ void spin_moments(const Z* d, const double* w, const int* atom_start,
                             int projectors, double* result)
{
    __shared__ double shared_x[threads];
    __shared__ double shared_y[threads];
    __shared__ double shared_z[threads];
    const int atom = blockIdx.y;
    const int band = blockIdx.x * blockDim.x + threadIdx.x;
    double x = 0;
    double y = 0;
    double z = 0;
    if (band < nb)
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

__global__ void conjugate_weighted(const Z* c, const double* w, Z* weighted)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * nb) return;
    const Z z = c[i];
    const double f = w[i / n];
    weighted[i] = make_cuDoubleComplex(z.x * f, -z.y * f);
}

__global__ void subtract_omitted_moments(const Omit* omitted, const double* factors, const Z* dm, double* result)
{
    __shared__ double left[factor_size];
    __shared__ double right[factor_size];
    __shared__ double sx[threads];
    __shared__ double sy[threads];
    __shared__ double sz[threads];
    const Omit block = omitted[blockIdx.x];
    load_factors(factors, block, left, right);
    __syncthreads();
    double x = 0;
    double y = 0;
    double z = 0;
    const int tid = threadIdx.x;
    for (int t = tid; t < block_orbitals * block_orbitals; t += threads)
    {
        const int i = t % block_orbitals;
        const int j = t / block_orbitals;
        const int pos = 2 * (block.i + i) + 2 * (block.j + j) * n;
        const double value = projector_product(left, right, i, j);
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

struct KState
{
    Z* vectors = nullptr;
    double* energies = nullptr;
    Z* projector = nullptr;
    Omit* omitted = nullptr;
    int omitted_count = 0;
    Group* groups = nullptr;
    int group_count = 0;
    double* energies_trial = nullptr;
    Z* mixed_projectors = nullptr;
    Z* rotation = nullptr;
};

struct Probe
{
    int rank = std::atoi(std::getenv("OMPI_COMM_WORLD_RANK"));
    int calls = 0;
    int projector_count = 0;
    KState k[blocks];
    std::vector<Raw> raw;
    double* factors = nullptr;
    FILE* log = nullptr;
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t eig = nullptr;
    Z* projected = nullptr;
    Z* scattered = nullptr;
    Z* ld = nullptr;
    Z* small = nullptr;
    Z* auxiliary = nullptr;
    Z* dense = nullptr;
    Z* vector_out = nullptr;
    Z* work = nullptr;
    int work_size = 0;
    int* atoms = nullptr;
    int* atom_start = nullptr;
    int* info = nullptr;
    double* lambda = nullptr;
    double* weights = nullptr;
    double* moment = nullptr;
    const Z one = make_cuDoubleComplex(1.0, 0.0);
    const Z zero = make_cuDoubleComplex(0.0, 0.0);

    Probe()
    {
        const auto name = std::string("lazy-ritz-") + std::to_string(rank) + ".csv";
        log = std::fopen(name.c_str(), "w");
        if (!log) std::abort();
        std::fprintf(log, "k,omitted,project_s,mask_s,diagonalize_s,rotate_s\n");
        check(cublasCreate(&blas), "cublasCreate");
        check(cusolverDnCreate(&eig), "cusolverDnCreate");
    }

    KState& state(int ik)
    {
        if (rank > 1 || ik % 2 != rank || ik / 2 >= blocks) std::abort();
        return k[ik / 2];
    }

    void init(const int* counts, const int* packed, int len)
    {
        if (projector_count) return;
        std::vector<int> host_atoms;
        int offset[nat + 1] = {};
        for (int i = 0; i < nat; ++i)
        {
            host_atoms.insert(host_atoms.end(), counts[i], i);
            offset[i + 1] = host_atoms.size();
        }
        projector_count = host_atoms.size();
        atoms = alloc<int>(projector_count);
        upload(atoms, host_atoms.data(), host_atoms.size());
        atom_start = alloc<int>(nat + 1);
        upload(atom_start, offset, nat + 1);
        const size_t dsize = size_t(2 * projector_count) * m;
        projected = alloc<Z>(dsize);
        ld = alloc<Z>(dsize);
        scattered = alloc<Z>(size_t(n) * m);
        small = alloc<Z>(size_t(m) * m);
        auxiliary = alloc<Z>(size_t(n) * m);
        vector_out = alloc<Z>(size_t(n) * nb);
        lambda = alloc<double>(3 * nat);
        moment = alloc<double>(3 * nat);
        weights = alloc<double>(nb);
        info = alloc<int>(1);
        if (len != 0) dense = alloc<Z>(size_t(n) * n);
        const int count = packed[1];
        const int coef = packed[0];
        if (2 + count * 8 + coef * 2 != len) std::abort();
        raw.reserve(count);
        for (int i = 2; i < 2 + count * 8; i += 8)
        {
            raw.push_back({packed[i], packed[i+1], packed[i+2], packed[i+5], packed[i+6], packed[i+7],
                           packed[i+3], packed[i+4]});
        }
        std::sort(raw.begin(), raw.end(), [](const Raw& a, const Raw& b)
            {
                return std::tie(a.i, a.j, a.atom, a.rx, a.ry, a.rz)
                     < std::tie(b.i, b.j, b.atom, b.rx, b.ry, b.rz);
            });
        std::vector<double> values(coef);
        std::memcpy(values.data(), packed + 2 + count * 8, size_t(coef) * sizeof(double));
        factors = alloc<double>(coef);
        upload(factors, values.data(), coef);
    }

    void prepare(int ik, const Z* b, const int* counts, const double* kpoint, const int* packed, int len)
    {
        init(counts, packed, len);
        auto& slot = state(ik);
        if (!slot.vectors || slot.projector) std::abort();
        const size_t bsize = size_t(projector_count) * (n / 2);
        slot.projector = alloc<Z>(bsize);
        upload(slot.projector, b, bsize);
        slot.energies_trial = alloc<double>(m);
        slot.mixed_projectors = alloc<Z>(size_t(2 * projector_count) * nb);
        std::vector<Omit> folded;
        std::vector<Group> groups;
        for (const auto& r : raw)
        {
            const double angle = 6.2831853071795864769 * (kpoint[0] * r.rx + kpoint[1] * r.ry + kpoint[2] * r.rz);
            const Z phase = make_cuDoubleComplex(std::cos(angle), std::sin(angle));
            const int i = folded.size();
            if (folded.empty() || folded.back().i != r.i || folded.back().j != r.j)
            {
                if (!groups.empty()) groups.back().end = i;
                groups.push_back({i, i});
            }
            folded.push_back({r.atom, r.i, r.j, r.a, r.b, phase});
        }
        slot.omitted_count = folded.size();
        if (slot.omitted_count != 0)
        {
            groups.back().end = folded.size();
            slot.group_count = groups.size();
            slot.groups = alloc<Group>(groups.size());
            upload(slot.groups, groups.data(), groups.size());
            slot.omitted = alloc<Omit>(folded.size());
            upload(slot.omitted, folded.data(), folded.size());
            slot.rotation = alloc<Z>(size_t(m) * nb);
        }
        std::fprintf(stderr, "LAZY_RITZ_GEOMETRY rank=%d k=%d projectors=%d folded_omissions=%d\n",
                     rank, ik, projector_count, slot.omitted_count);
    }

    void gemm(cublasOperation_t a, cublasOperation_t b, int rows, int cols, int inner,
              const Z* l, int ldl, const Z* r, int ldr, Z* out, const Z* beta)
    {
        check(cublasZgemm(blas, a, b, rows, cols, inner, &one, l, ldl, r, ldr, beta, out, rows), "zgemm");
    }

    void project(const KState& slot, int columns)
    {
        const int count = n * columns;
        split_spinors<<<(count + threads - 1) / threads, threads>>>(slot.vectors, scattered, count);
        check(cudaGetLastError(), "spin split");
        gemm(CUBLAS_OP_N, CUBLAS_OP_N, projector_count, 2 * columns, n / 2,
             slot.projector, projector_count, scattered, n / 2, projected, &zero);
    }

    void solve(int ik, const double* step, double* output, double* times)
    {
        auto& slot = state(ik);
        auto begin = Clock::now();
        upload(lambda, step, 3 * nat);
        project(slot, m);
        const int count = projector_count * m;
        apply_lambda<<<(count + threads - 1) / threads, threads>>>(projected, lambda, atoms, ld, projector_count, count);
        check(cudaGetLastError(), "projected lambda");
        const int rows = 2 * projector_count;
        gemm(CUBLAS_OP_C, CUBLAS_OP_N, m, m, rows, projected, rows, ld, rows, small, &zero);
        check(cudaDeviceSynchronize(), "projected lambda synchronization");
        const double project_s = std::chrono::duration<double>(Clock::now() - begin).count();
        times[0] += project_s;
        begin = Clock::now();
        if (slot.omitted_count != 0)
        {
            check(cudaMemset(dense, 0, size_t(n) * n * sizeof(Z)), "zero omitted Hamiltonian");
            subtract_omitted<<<slot.group_count, threads>>>(slot.omitted, slot.groups, factors, lambda, dense);
            check(cudaGetLastError(), "omitted Hamiltonian");
            gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, dense, n, slot.vectors, n, auxiliary, &zero);
            gemm(CUBLAS_OP_C, CUBLAS_OP_N, m, m, n, slot.vectors, n, auxiliary, n, small, &one);
            check(cudaDeviceSynchronize(), "omitted Hamiltonian synchronization");
        }
        const double mask_s = std::chrono::duration<double>(Clock::now() - begin).count();
        times[1] += mask_s;
        begin = Clock::now();
        add_diagonal<<<(m + threads - 1) / threads, threads>>>(small, slot.energies);
        check(cudaGetLastError(), "reference energy");
        if (!work)
        {
            check(cusolverDnZheevd_bufferSize(eig, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                            m, small, m, slot.energies_trial, &work_size), "heevd workspace");
            work = alloc<Z>(work_size);
        }
        check(cusolverDnZheevd(eig, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                               m, small, m, slot.energies_trial, work, work_size, info), "heevd");
        int stat = 0;
        check(cudaMemcpy(&stat, info, sizeof(int), cudaMemcpyDeviceToHost), "heevd info");
        if (stat != 0) std::abort();
        check(cudaMemcpy(output, slot.energies_trial, nb * sizeof(double), cudaMemcpyDeviceToHost), "trial eigenvalues");
        const double diagonalize_s = std::chrono::duration<double>(Clock::now() - begin).count();
        times[2] += diagonalize_s;
        begin = Clock::now();
        gemm(CUBLAS_OP_N, CUBLAS_OP_N, rows, nb, m, projected, rows, small, m, slot.mixed_projectors, &zero);
        if (slot.omitted_count)
        {
            check(cudaMemcpy(slot.rotation, small, size_t(m) * nb * sizeof(Z), cudaMemcpyDeviceToDevice),
                  "store Ritz rotation for omitted moments");
        }
        check(cudaDeviceSynchronize(), "rotation synchronization");
        const double rotate_s = std::chrono::duration<double>(Clock::now() - begin).count();
        times[3] += rotate_s;
        std::fprintf(log, "%d,%d,%.9g,%.9g,%.9g,%.9g\n",
                     ik, slot.omitted_count, project_s, mask_s, diagonalize_s, rotate_s);
        std::fflush(log);
    }

    void moments(int ik, const double* w, bool reference, double* result)
    {
        const auto& slot = state(ik);
        upload(weights, w, nb);
        check(cudaMemset(moment, 0, 3 * nat * sizeof(double)), "zero trial moments");
        if (reference) project(slot, nb);
        const auto* proj = reference ? projected : slot.mixed_projectors;
        const dim3 grid((nb + threads - 1) / threads, nat);
        spin_moments<<<grid, threads>>>(proj, weights, atom_start, projector_count, moment);
        check(cudaGetLastError(), "projector moments");
        if (slot.omitted_count != 0)
        {
            const Z* vectors = slot.vectors;
            if (!reference)
            {
                gemm(CUBLAS_OP_N, CUBLAS_OP_N, n, nb, m, slot.vectors, n, slot.rotation, m, vector_out, &zero);
                vectors = vector_out;
            }
            conjugate_weighted<<<(n * nb + threads - 1) / threads, threads>>>(vectors, weights, auxiliary);
            check(cudaGetLastError(), "conjugate weighted wavefunctions");
            gemm(CUBLAS_OP_N, CUBLAS_OP_T, n, n, nb, auxiliary, n, vectors, n, dense, &zero);
            subtract_omitted_moments<<<slot.omitted_count, threads>>>(slot.omitted, factors, dense, moment);
            check(cudaGetLastError(), "omitted moments");
        }
        double contributions[3 * nat];
        check(cudaMemcpy(contributions, moment, sizeof(contributions), cudaMemcpyDeviceToHost), "trial moments");
        for (int i = 0; i < 3 * nat; ++i) result[i] += contributions[i];
    }

};

Probe& probe()
{
    static Probe* instance = new Probe;
    return *instance;
}
} // namespace

extern "C" void abacus_lazy_prepare(int ik, const Z* b, const int* counts, const double* xyz, const int* packed, int len)
{
    probe().prepare(ik, b, counts, xyz, packed, len);
}

extern "C" void abacus_lazy_solve(int ik, const double* delta, double* output, double* times)
{
    probe().solve(ik, delta, output, times);
}

extern "C" void abacus_lazy_moments(int ik, const double* weights, int reference, double* moments)
{
    probe().moments(ik, weights, reference, moments);
}

extern "C" cusolverStatus_t cusolverDnZhegvd(
    cusolverDnHandle_t handle, cusolverEigType_t kind, cusolverEigMode_t job,
    cublasFillMode_t uplo, int size, Z* h, int ldh, Z* s, int lds,
    double* output, Z* work, int count, int* info)
{
    static const auto real = reinterpret_cast<FullSolver>(dlsym(RTLD_NEXT, "cusolverDnZhegvd"));
    if (!real) std::abort();
    const auto status = real(handle, kind, job, uplo, size, h, ldh, s, lds, output, work, count, info);
    if (status == CUSOLVER_STATUS_SUCCESS && size == n && ldh == n && lds == n
        && kind == CUSOLVER_EIG_TYPE_1 && job == CUSOLVER_EIG_MODE_VECTOR && uplo == CUBLAS_FILL_MODE_UPPER
        && std::getenv("ABACUS_LAZY_PROJECTOR_PROBE"))
    {
        auto& p = probe();
        const int ik = (p.calls++ % blocks) * 2 + p.rank;
        auto& slot = p.state(ik);
        if (!slot.vectors)
        {
            slot.vectors = alloc<Z>(size_t(n) * m);
            slot.energies = alloc<double>(m);
        }
        check(cudaMemcpy(slot.vectors, h, size_t(n) * m * sizeof(Z), cudaMemcpyDeviceToDevice), "reference vectors");
        check(cudaMemcpy(slot.energies, output, m * sizeof(double), cudaMemcpyDeviceToDevice), "reference energies");
    }
    return status;
}
