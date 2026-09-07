#include "folding_hr_gpu.h"
#include "hcontainer.h"
#include "source_base/libm/libm.h"
#include "source_base/module_device/device_check.h"
#include "source_base/timer.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <stdexcept>

namespace hamilt
{
namespace
{
int index_size(size_t n)
{
    if (n > INT_MAX) throw std::overflow_error("GPU H(R) folding index exceeds INT_MAX");
    return static_cast<int>(n);
}

template <typename T>
struct Buffer
{
    T* device = nullptr;
    T* host = nullptr;
    Buffer(size_t n, bool pinned)
    {
        const size_t bytes = std::max<size_t>(1, n) * sizeof(T);
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device), bytes));
        if (pinned) CHECK_CUDA(cudaMallocHost(reinterpret_cast<void**>(&host), bytes));
    }
    ~Buffer()
    {
        if (host) CHECK_CUDA(cudaFreeHost(host));
        CHECK_CUDA(cudaFree(device));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Pair
{
    int rows;
    int cols;
    int row;
    int col;
    int nr;
    int values;
    int phases;
};

__global__ void fold_pairs(const Pair* pairs, const int2* tiles, const double2* values,
                           const double2* phases, double2* out, int ld)
{
    const int2 tile = tiles[blockIdx.x];
    const Pair p = pairs[tile.x];
    const int i = tile.y + threadIdx.x;
    if (i >= p.rows * p.cols) return;
    double re = 0.0;
    double im = 0.0;
    for (int ir = 0; ir < p.nr; ++ir)
    {
        const double2 v = values[p.values + ir * p.rows * p.cols + i];
        const double2 z = phases[p.phases + ir];
        // Preserve the CPU complex multiply/add order, without contraction.
        re = __dadd_rn(re, __dsub_rn(__dmul_rn(z.x, v.x), __dmul_rn(z.y, v.y)));
        im = __dadd_rn(im, __dadd_rn(__dmul_rn(z.x, v.y), __dmul_rn(z.y, v.x)));
    }
    out[(p.col + i % p.cols) * ld + p.row + i / p.cols] = make_double2(re, im);
}
}

struct FoldingHrGpu::Impl
{
    int rows;
    int cols;
    int nk;
    int ntiles;
    size_t nvalues;
    std::vector<Pair> pairs;
    std::vector<ModuleBase::Vector3<int>> lattice;
    std::unique_ptr<Buffer<Pair>> device_pairs;
    std::unique_ptr<Buffer<int2>> tiles;
    std::unique_ptr<Buffer<double2>> values;
    std::unique_ptr<Buffer<double2>> phases;
    std::unique_ptr<Buffer<double2>> output;
};

FoldingHrGpu::FoldingHrGpu(const HContainer<std::complex<double>>& hr,
                         const std::vector<ModuleBase::Vector3<double>>& kpoints)
    : impl_(new Impl)
{
    ModuleBase::timer::start("FoldingHrGpu", "layout");
    auto& p = *impl_;
    p.rows = hr.get_paraV()->get_row_size();
    p.cols = hr.get_paraV()->get_col_size();
    index_size(size_t(p.rows) * p.cols);
    p.nk = index_size(kpoints.size());
    p.nvalues = 0;
    std::vector<int2> tiles;
    for (int ip = 0; ip < hr.size_atom_pairs(); ++ip)
    {
        const auto& ap = hr.get_atom_pair(ip);
        const Pair pair = {ap.get_row_size(), ap.get_col_size(), ap.get_begin_row(), ap.get_begin_col(),
                           index_size(ap.get_R_size()), index_size(p.nvalues), index_size(p.lattice.size())};
        if (pair.row < 0 || pair.col < 0 || pair.rows < 0 || pair.cols < 0
            || pair.row + pair.rows > p.rows || pair.col + pair.cols > p.cols)
        {
            throw std::invalid_argument("GPU folding invalid local atom-pair layout");
        }
        const int count = index_size(size_t(pair.rows) * pair.cols);
        p.pairs.push_back(pair);
        for (int ir = 0; ir < pair.nr; ++ir) p.lattice.push_back(ap.get_R_index(ir));
        p.nvalues += size_t(count) * pair.nr;
        index_size(p.nvalues);
        for (size_t i = 0; i < size_t(count); i += 256) tiles.push_back(make_int2(ip, index_size(i)));
    }
    p.ntiles = index_size(tiles.size());
    p.device_pairs.reset(new Buffer<Pair>(p.pairs.size(), false));
    p.tiles.reset(new Buffer<int2>(tiles.size(), false));
    p.values.reset(new Buffer<double2>(p.nvalues, true));
    p.phases.reset(new Buffer<double2>(p.lattice.size() * p.nk, false));
    p.output.reset(new Buffer<double2>(size_t(p.rows) * p.cols, false));
    if (!p.pairs.empty())
        CHECK_CUDA(cudaMemcpy(p.device_pairs->device, p.pairs.data(), p.pairs.size() * sizeof(Pair), cudaMemcpyHostToDevice));
    if (!tiles.empty())
        CHECK_CUDA(cudaMemcpy(p.tiles->device, tiles.data(), tiles.size() * sizeof(int2), cudaMemcpyHostToDevice));
    std::vector<double2> phases(p.lattice.size() * p.nk);
    for (int ik = 0; ik < p.nk; ++ik)
    {
        for (size_t ir = 0; ir < p.lattice.size(); ++ir)
        {
            const auto r = p.lattice[ir];
            const ModuleBase::Vector3<double> dr(r.x, r.y, r.z);
            const double angle = (kpoints[ik] * dr) * ModuleBase::TWO_PI;
            double s = 0.0;
            double c = 0.0;
            ModuleBase::libm::sincos(angle, &s, &c);
            phases[ik * p.lattice.size() + ir] = make_double2(c, s);
        }
    }
    if (!phases.empty())
        CHECK_CUDA(cudaMemcpy(p.phases->device, phases.data(), phases.size() * sizeof(double2), cudaMemcpyHostToDevice));
    ModuleBase::timer::end("FoldingHrGpu", "layout");
}

FoldingHrGpu::~FoldingHrGpu() = default;

void FoldingHrGpu::upload(const HContainer<std::complex<double>>& hr)
{
    ModuleBase::timer::start("FoldingHrGpu", "upload");
    auto& p = *impl_;
    if (hr.size_atom_pairs() != p.pairs.size()
        || hr.get_paraV()->get_row_size() != p.rows || hr.get_paraV()->get_col_size() != p.cols)
        throw std::invalid_argument("GPU folding layout changed");
    for (int ip = 0; ip < hr.size_atom_pairs(); ++ip)
    {
        const auto& ap = hr.get_atom_pair(ip);
        const auto& pair = p.pairs[ip];
        if (ap.get_row_size() != pair.rows || ap.get_col_size() != pair.cols
            || ap.get_begin_row() != pair.row || ap.get_begin_col() != pair.col || ap.get_R_size() != pair.nr)
            throw std::invalid_argument("GPU folding atom-pair layout changed");
        const size_t count = size_t(pair.rows) * pair.cols;
        for (int ir = 0; ir < pair.nr; ++ir)
        {
            const auto r = ap.get_R_index(ir);
            const auto expected = p.lattice[pair.phases + ir];
            if (r.x != expected.x || r.y != expected.y || r.z != expected.z)
                throw std::invalid_argument("GPU folding lattice order changed");
            if (count) std::memcpy(p.values->host + pair.values + ir * count, ap.get_pointer(ir),
                                   count * sizeof(double2));
        }
    }
    if (p.nvalues)
        CHECK_CUDA(cudaMemcpy(p.values->device, p.values->host, p.nvalues * sizeof(double2), cudaMemcpyHostToDevice));
    ModuleBase::timer::end("FoldingHrGpu", "upload");
}

std::complex<double>* FoldingHrGpu::fold(int ik)
{
    auto& p = *impl_;
    if (ik < 0 || ik >= p.nk) throw std::out_of_range("GPU folding k index");
    ModuleBase::timer::start("FoldingHrGpu", "fold");
    CHECK_CUDA(cudaMemset(p.output->device, 0, size_t(p.rows) * p.cols * sizeof(double2)));
    if (p.ntiles)
    {
        fold_pairs<<<p.ntiles, 256>>>(p.device_pairs->device, p.tiles->device, p.values->device,
                                    p.phases->device + ik * p.lattice.size(), p.output->device, p.rows);
        CHECK_CUDA(cudaGetLastError());
    }
    ModuleBase::timer::end("FoldingHrGpu", "fold");
    return reinterpret_cast<std::complex<double>*>(p.output->device);
}
}
