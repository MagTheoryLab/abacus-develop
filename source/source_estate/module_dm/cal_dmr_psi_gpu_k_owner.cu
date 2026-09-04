#include "cal_dm_psi.h"

#include "source_base/constants.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/parallel_common.h"
#include "source_base/timer.h"
#include <base/macros/macros.h>
#include <thrust/complex.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace elecstate
{
namespace
{

constexpr int dmr_block_descriptor_size = 5;

__global__ void weight_conjugate_wfc_kernel(const int size,
                                             const int nbasis,
                                             const thrust::complex<double>* wfc,
                                             const double* weights,
                                             thrust::complex<double>* out)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        out[index] = thrust::conj(wfc[index]) * weights[index / nbasis];
    }
}

__device__ int find_dmr_block(const int* blocks, const int block_count, const int index)
{
    int left = 0;
    int right = block_count;
    while (left + 1 < right)
    {
        const int middle = left + (right - left) / 2;
        if (blocks[middle * dmr_block_descriptor_size] <= index)
        {
            left = middle;
        }
        else
        {
            right = middle;
        }
    }
    return left;
}

template <bool full_complex>
__global__ void dense_dmk_to_sparse_dmr_kernel(const int output_size,
                                                const int nlocal,
                                                const thrust::complex<double>* dmk,
                                                const int* blocks,
                                                const int block_count,
                                                const int* orbital_indices,
                                                const thrust::complex<double>* phases,
                                                double* dmr)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_size)
    {
        return;
    }

    const int iblock = find_dmr_block(blocks, block_count, index);
    const int* block = blocks + iblock * dmr_block_descriptor_size;
    const int local_index = index - block[0];
    const int irow = local_index / block[2];
    const int icol = local_index - irow * block[2];
    const thrust::complex<double> phase = phases[iblock];
    const int* rows = orbital_indices + block[3];
    const int* cols = orbital_indices + block[4];

    if (full_complex)
    {
        const thrust::complex<double> value = phase * dmk[rows[irow] + cols[icol] * nlocal];
        dmr[2 * index] += value.real();
        dmr[2 * index + 1] += value.imag();
        return;
    }

    const int row0 = rows[irow - irow % 2];
    const int col0 = cols[icol - icol % 2];
    const thrust::complex<double> up_up = phase * dmk[row0 + col0 * nlocal];
    const thrust::complex<double> up_down = phase * dmk[row0 + (col0 + 1) * nlocal];
    const thrust::complex<double> down_up = phase * dmk[(row0 + 1) + col0 * nlocal];
    const thrust::complex<double> down_down = phase * dmk[(row0 + 1) + (col0 + 1) * nlocal];
    const int component = (irow % 2) * 2 + icol % 2;

    if (component == 0)
    {
        dmr[index] += up_up.real() + down_down.real();
    }
    else if (component == 1)
    {
        dmr[index] += up_down.real() + down_up.real();
    }
    else if (component == 2)
    {
        dmr[index] += up_down.imag() - down_up.imag();
    }
    else
    {
        dmr[index] += up_up.real() - down_down.real();
    }
}

template <typename TR>
void cal_dmr_psi_gpu_k_owner_impl(
    const Parallel_Orbitals* para_v,
    const ModuleBase::matrix& wg,
    const std::vector<const psi::Psi<std::complex<double>, base_device::DEVICE_GPU>*>& owner_wfc,
    const std::vector<ModuleBase::Vector3<double>>& kvec_d,
    hamilt::HContainer<TR>* local_dmr)
{
    ModuleBase::TITLE("elecstate", "cal_dmr_psi_gpu_k_owner");
    ModuleBase::timer::start("elecstate", "cal_dmr_psi_gpu_k_owner");

    constexpr int scalars_per_element = sizeof(TR) / sizeof(double);
    static_assert(scalars_per_element == 1 || scalars_per_element == 2, "unsupported DMR scalar layout");
    if (owner_wfc.size() != static_cast<std::size_t>(wg.nr) || owner_wfc.size() != kvec_d.size())
    {
        throw std::runtime_error("k-owner wavefunctions, weights and k vectors have inconsistent sizes");
    }
    const std::size_t local_nnr_size = local_dmr->get_nnr();
    if (local_nnr_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("local DMR is too large for MPI_Reduce_scatter");
    }
    const int local_nnr = static_cast<int>(local_nnr_size);

    const int world_size = Parallel_Common::communicator_size(MPI_COMM_WORLD);
    std::vector<int> recvcounts(world_size, 0);
    Parallel_Common::allgather_int(&local_nnr, 1, recvcounts.data(), MPI_COMM_WORLD);
    std::vector<int> dmr_displs(world_size, 0);
    int global_nnr = 0;
    for (int rank = 0; rank < world_size; ++rank)
    {
        if (recvcounts[rank] > std::numeric_limits<int>::max() - global_nnr)
        {
            throw std::runtime_error("global DMR is too large for 32-bit GPU indexing");
        }
        dmr_displs[rank] = global_nnr;
        global_nnr += recvcounts[rank];
    }

    // Pack each local IJR block once. Orbital indexes are global, while the
    // first field is the offset in this rank's contiguous HContainer wrapper.
    std::vector<int> local_metadata;
    const TR* local_wrapper = local_dmr->get_wrapper();
    for (int iap = 0; iap < static_cast<int>(local_dmr->size_atom_pairs()); ++iap)
    {
        const hamilt::AtomPair<TR>& atom_pair = local_dmr->get_atom_pair(iap);
        const int iat1 = atom_pair.get_atom_i();
        const int iat2 = atom_pair.get_atom_j();
        const int row_size = atom_pair.get_row_size();
        const int col_size = atom_pair.get_col_size();
        std::vector<int> rows(row_size);
        std::vector<int> cols(col_size);
        for (int irow = 0; irow < row_size; ++irow)
        {
            rows[irow] = para_v->local2global_row(para_v->atom_begin_row[iat1] + irow);
        }
        for (int icol = 0; icol < col_size; ++icol)
        {
            cols[icol] = para_v->local2global_col(para_v->atom_begin_col[iat2] + icol);
        }
        if (row_size % 2 != 0 || col_size % 2 != 0)
        {
            throw std::runtime_error("non-collinear DMR block splits a spinor pair");
        }
        for (int irow = 0; irow < row_size; irow += 2)
        {
            if (rows[irow] % 2 != 0 || rows[irow + 1] != rows[irow] + 1)
            {
                throw std::runtime_error("non-collinear DMR row ownership splits a spinor pair");
            }
        }
        for (int icol = 0; icol < col_size; icol += 2)
        {
            if (cols[icol] % 2 != 0 || cols[icol + 1] != cols[icol] + 1)
            {
                throw std::runtime_error("non-collinear DMR column ownership splits a spinor pair");
            }
        }
        for (int iR = 0; iR < atom_pair.get_R_size(); ++iR)
        {
            const ModuleBase::Vector3<int> R = atom_pair.get_R_index(iR);
            const TR* block_pointer = atom_pair.get_HR_values(iR).get_pointer();
            const std::ptrdiff_t offset = block_pointer - local_wrapper;
            if (offset < 0 || offset > std::numeric_limits<int>::max())
            {
                throw std::runtime_error("local DMR offset exceeds 32-bit metadata");
            }
            local_metadata.push_back(static_cast<int>(offset));
            local_metadata.push_back(row_size);
            local_metadata.push_back(col_size);
            local_metadata.push_back(R.x);
            local_metadata.push_back(R.y);
            local_metadata.push_back(R.z);
            local_metadata.insert(local_metadata.end(), rows.begin(), rows.end());
            local_metadata.insert(local_metadata.end(), cols.begin(), cols.end());
        }
    }

    if (local_metadata.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("local DMR metadata exceeds 32-bit indexing");
    }
    const int local_metadata_size = static_cast<int>(local_metadata.size());
    std::vector<int> metadata_counts(world_size, 0);
    Parallel_Common::allgather_int(&local_metadata_size, 1, metadata_counts.data(), MPI_COMM_WORLD);
    std::vector<int> metadata_displs(world_size, 0);
    int global_metadata_size = 0;
    for (int rank = 0; rank < world_size; ++rank)
    {
        if (metadata_counts[rank] > std::numeric_limits<int>::max() - global_metadata_size)
        {
            throw std::runtime_error("global DMR metadata exceeds 32-bit indexing");
        }
        metadata_displs[rank] = global_metadata_size;
        global_metadata_size += metadata_counts[rank];
    }
    std::vector<int> global_metadata(global_metadata_size);
    Parallel_Common::allgatherv_int(local_metadata.data(),
                                    local_metadata_size,
                                    global_metadata.data(),
                                    metadata_counts.data(),
                                    metadata_displs.data(),
                                    MPI_COMM_WORLD);

    std::vector<int> blocks;
    std::vector<int> orbital_indices;
    std::vector<ModuleBase::Vector3<int>> block_R;
    int expected_offset = 0;
    for (int rank = 0; rank < world_size; ++rank)
    {
        int position = metadata_displs[rank];
        const int end = position + metadata_counts[rank];
        while (position < end)
        {
            const int offset = dmr_displs[rank] + global_metadata[position++];
            const int row_size = global_metadata[position++];
            const int col_size = global_metadata[position++];
            if (offset != expected_offset || row_size <= 0 || col_size <= 0
                || row_size > std::numeric_limits<int>::max() / col_size)
            {
                throw std::runtime_error("distributed DMR blocks do not form a contiguous sparse layout");
            }
            const int block_size = row_size * col_size;
            if (block_size > global_nnr - expected_offset)
            {
                throw std::runtime_error("distributed DMR block sizes exceed the sparse storage size");
            }
            expected_offset += block_size;
            ModuleBase::Vector3<int> R;
            R.x = global_metadata[position++];
            R.y = global_metadata[position++];
            R.z = global_metadata[position++];
            const int row_index_offset = static_cast<int>(orbital_indices.size());
            orbital_indices.insert(orbital_indices.end(),
                                   global_metadata.begin() + position,
                                   global_metadata.begin() + position + row_size);
            position += row_size;
            const int col_index_offset = static_cast<int>(orbital_indices.size());
            orbital_indices.insert(orbital_indices.end(),
                                   global_metadata.begin() + position,
                                   global_metadata.begin() + position + col_size);
            position += col_size;
            blocks.push_back(offset);
            blocks.push_back(row_size);
            blocks.push_back(col_size);
            blocks.push_back(row_index_offset);
            blocks.push_back(col_index_offset);
            block_R.push_back(R);
        }
    }
    if (block_R.empty() || global_nnr == 0)
    {
        local_dmr->set_zero();
        ModuleBase::timer::end("elecstate", "cal_dmr_psi_gpu_k_owner");
        return;
    }
    if (expected_offset != global_nnr)
    {
        throw std::runtime_error("distributed DMR block sizes do not match the sparse storage size");
    }
    if (blocks.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || orbital_indices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("global DMR GPU metadata exceeds 32-bit indexing");
    }
    const int nlocal = para_v->get_global_row_size();
    const int nbands = wg.nc;
    if (nlocal <= 0 || nbands <= 0
        || nlocal > std::numeric_limits<int>::max() / nlocal
        || nlocal > std::numeric_limits<int>::max() / nbands)
    {
        throw std::runtime_error("k-owner dense GPU workspace exceeds 32-bit indexing");
    }
    const int wfc_size = nlocal * nbands;
    const int dmk_size = nlocal * nlocal;
    if (global_nnr > std::numeric_limits<int>::max() / scalars_per_element)
    {
        throw std::runtime_error("complex sparse DMR exceeds 32-bit MPI counts");
    }
    const int scalar_count = global_nnr * scalars_per_element;
    std::complex<double>* weighted_wfc_device = nullptr;
    std::complex<double>* dmk_device = nullptr;
    std::complex<double>* phases_device = nullptr;
    int* blocks_device = nullptr;
    int* orbital_indices_device = nullptr;
    double* weights_device = nullptr;
    double* dmr_device = nullptr;
    base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(weighted_wfc_device, wfc_size);
    base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(dmk_device, dmk_size);
    base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(phases_device, block_R.size());
    base_device::memory::resize_memory_op<int, base_device::DEVICE_GPU>()(blocks_device, blocks.size());
    base_device::memory::resize_memory_op<int, base_device::DEVICE_GPU>()(orbital_indices_device, orbital_indices.size());
    base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(weights_device, nbands);
    base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(dmr_device, scalar_count);
    base_device::memory::synchronize_memory_op<int, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
        blocks_device, blocks.data(), blocks.size());
    base_device::memory::synchronize_memory_op<int, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
        orbital_indices_device, orbital_indices.data(), orbital_indices.size());
    CHECK_CUDA(cudaMemset(dmr_device, 0, static_cast<std::size_t>(scalar_count) * sizeof(double)));

    std::vector<double> weights(nbands, 0.0);
    std::vector<std::complex<double>> phases(block_R.size());
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    const char normal = 'N';
    const char transpose = 'T';
    const int threads = 256;
    const int grid = (global_nnr + threads - 1) / threads;

    for (int ik = 0; ik < static_cast<int>(owner_wfc.size()); ++ik)
    {
        if (owner_wfc[ik] == nullptr)
        {
            continue;
        }
        for (int ib = 0; ib < nbands; ++ib)
        {
            weights[ib] = wg(ik, ib);
        }
        for (int iblock = 0; iblock < static_cast<int>(block_R.size()); ++iblock)
        {
            const double k_dot_R = kvec_d[ik].x * block_R[iblock].x
                                   + kvec_d[ik].y * block_R[iblock].y
                                   + kvec_d[ik].z * block_R[iblock].z;
            const double arg = k_dot_R * ModuleBase::TWO_PI;
            phases[iblock] = std::complex<double>(std::cos(arg), std::sin(arg));
        }
        base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
            weights_device, weights.data(), nbands);
        base_device::memory::synchronize_memory_op<std::complex<double>, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
            phases_device, phases.data(), phases.size());

        const std::complex<double>* wfc_device = owner_wfc[ik]->get_pointer();
        const int wfc_threads = 256;
        const int wfc_blocks = (wfc_size + wfc_threads - 1) / wfc_threads;
        weight_conjugate_wfc_kernel<<<wfc_blocks, wfc_threads>>>(
            wfc_size,
            nlocal,
            reinterpret_cast<const thrust::complex<double>*>(wfc_device),
            weights_device,
            reinterpret_cast<thrust::complex<double>*>(weighted_wfc_device));
        CHECK_CUDA(cudaGetLastError());
        ModuleBase::gemm_op<std::complex<double>, base_device::DEVICE_GPU>()(
            normal, transpose, nlocal, nlocal, nbands, &one,
            weighted_wfc_device, nlocal, wfc_device, nlocal, &zero, dmk_device, nlocal);
        dense_dmk_to_sparse_dmr_kernel<scalars_per_element == 2><<<grid, threads>>>(
            global_nnr,
            nlocal,
            reinterpret_cast<const thrust::complex<double>*>(dmk_device),
            blocks_device,
            static_cast<int>(block_R.size()),
            orbital_indices_device,
            reinterpret_cast<const thrust::complex<double>*>(phases_device),
            dmr_device);
        CHECK_CUDA(cudaGetLastError());
    }

    std::vector<double> sparse_contribution(scalar_count, 0.0);
    base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
        sparse_contribution.data(), dmr_device, scalar_count);
    local_dmr->set_zero();
    for (auto& count : recvcounts)
    {
        count *= scalars_per_element;
    }
    Parallel_Common::reduce_scatter_double(sparse_contribution.data(),
                                           reinterpret_cast<double*>(local_dmr->get_wrapper()),
                                           recvcounts.data(),
                                           MPI_COMM_WORLD);

    base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(weighted_wfc_device);
    base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(dmk_device);
    base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(phases_device);
    base_device::memory::delete_memory_op<int, base_device::DEVICE_GPU>()(blocks_device);
    base_device::memory::delete_memory_op<int, base_device::DEVICE_GPU>()(orbital_indices_device);
    base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>()(weights_device);
    base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>()(dmr_device);

    ModuleBase::timer::end("elecstate", "cal_dmr_psi_gpu_k_owner");
}

} // namespace

void cal_dmr_psi_gpu_k_owner(
    const Parallel_Orbitals* para_v,
    const ModuleBase::matrix& wg,
    const std::vector<const psi::Psi<std::complex<double>, base_device::DEVICE_GPU>*>& owner_wfc,
    elecstate::DensityMatrix<std::complex<double>, double>& dm)
{
    cal_dmr_psi_gpu_k_owner_impl(para_v, wg, owner_wfc, dm.get_kvec_d(), dm.get_DMR_pointer(1));
}

void cal_dmr_psi_gpu_k_owner(
    const Parallel_Orbitals* para_v,
    const ModuleBase::matrix& weights,
    const std::vector<const psi::Psi<std::complex<double>, base_device::DEVICE_GPU>*>& owner_wfc,
    const std::vector<ModuleBase::Vector3<double>>& kvec_d,
    hamilt::HContainer<std::complex<double>>& full_dmr)
{
    cal_dmr_psi_gpu_k_owner_impl(para_v, weights, owner_wfc, kvec_d, &full_dmr);
}

} // namespace elecstate
