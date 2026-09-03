#include "xc_functional_ncgga_sf_gpu.h"

#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_base/vector3.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_estate/module_charge/charge.h"
#include "xc_builtin_pbe_math.h"
#include "xc_ids.h"
#include "xc_ncgga_radial_math.h"

#include <base/macros/macros.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>

#include <complex>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

namespace ModuleXC
{
namespace NCGGA_SF_Builtin
{
namespace
{

constexpr int threads_per_block = 256;
constexpr double vanishing = 1.0e-10;
constexpr double epsr = 1.0e-6;
constexpr double e2 = 2.0;

struct PbeFamily
{
    int exchange_flag = -1;
    int correlation_flag = -1;
};

bool get_pbe_family(const std::vector<int>& functional_ids,
                    PbeFamily& family)
{
    if (functional_ids.size() != 2)
    {
        return false;
    }

    if (functional_ids[0] == XC_GGA_X_PBE)
    {
        family.exchange_flag = 0;
    }
    else if (functional_ids[0] == XC_GGA_X_PBE_R)
    {
        family.exchange_flag = 1;
    }
    else if (functional_ids[0] == XC_GGA_X_PBE_SOL)
    {
        family.exchange_flag = 2;
    }
    else
    {
        return false;
    }

    if (functional_ids[1] == XC_GGA_C_PBE)
    {
        family.correlation_flag = 1;
    }
    else if (functional_ids[1] == XC_GGA_C_PBE_SOL)
    {
        family.correlation_flag = 2;
    }
    else
    {
        return false;
    }
    return true;
}

template <typename T>
void free_device(T*& pointer)
{
    if (pointer != nullptr)
    {
        cudaFree(pointer);
        pointer = nullptr;
    }
}

template <typename T>
void allocate_device(T*& pointer, const std::size_t count)
{
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&pointer),
                          count * sizeof(T)));
}

struct Workspace
{
    const ModulePW::PW_Basis* basis = nullptr;
    int nrxx = 0;
    int npw = 0;
    double* rho = nullptr;
    double* field_gradient = nullptr;
    double* spin_flux = nullptr;
    double* h = nullptr;
    double* potential = nullptr;
    double* real_scratch = nullptr;
    double* gcar = nullptr;
    double* reductions = nullptr;
    thrust::complex<double>* rhog_core = nullptr;
    thrust::complex<double>* reciprocal = nullptr;
    thrust::complex<double>* reciprocal_scratch = nullptr;

    ~Workspace()
    {
        release();
    }

    void release()
    {
        free_device(rho);
        free_device(field_gradient);
        free_device(spin_flux);
        free_device(h);
        free_device(potential);
        free_device(real_scratch);
        free_device(gcar);
        free_device(reductions);
        free_device(rhog_core);
        free_device(reciprocal);
        free_device(reciprocal_scratch);
        basis = nullptr;
        nrxx = 0;
        npw = 0;
    }

    void ensure(const ModulePW::PW_Basis* new_basis)
    {
        if (basis == new_basis
            && nrxx == new_basis->nrxx
            && npw == new_basis->npw)
        {
            return;
        }

        release();
        basis = new_basis;
        nrxx = new_basis->nrxx;
        npw = new_basis->npw;
        const std::size_t real_size = static_cast<std::size_t>(nrxx);
        const std::size_t reciprocal_size = static_cast<std::size_t>(npw);
        allocate_device(rho, 5 * real_size);
        allocate_device(field_gradient, 12 * real_size);
        allocate_device(spin_flux, 3 * real_size);
        allocate_device(h, 6 * real_size);
        allocate_device(potential, 4 * real_size);
        allocate_device(real_scratch, real_size);
        allocate_device(gcar, 3 * reciprocal_size);
        allocate_device(reductions, 2);
        allocate_device(rhog_core, reciprocal_size);
        allocate_device(reciprocal, reciprocal_size);
        allocate_device(reciprocal_scratch, reciprocal_size);
    }
};

Workspace& get_workspace()
{
    static Workspace workspace;
    return workspace;
}

__global__ void add_reciprocal_kernel(
    const int count,
    const thrust::complex<double>* addend,
    thrust::complex<double>* values)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        values[index] += addend[index];
    }
}

__global__ void reciprocal_gradient_kernel(
    const int count,
    const int component,
    const bool add,
    const double scale,
    const double* gcar,
    const thrust::complex<double>* input,
    thrust::complex<double>* output)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
    {
        return;
    }
    const double g = scale * gcar[3 * index + component];
    const thrust::complex<double> value(-g * input[index].imag(),
                                        g * input[index].real());
    if (add)
    {
        output[index] += value;
    }
    else
    {
        output[index] = value;
    }
}

__device__ inline double dot3(const double* left, const double* right)
{
    return left[0] * right[0]
           + left[1] * right[1]
           + left[2] * right[2];
}

__device__ inline void evaluate_lda_pbe(
    const double rho,
    const double zeta,
    double& exc,
    double& vup,
    double& vdown)
{
    double exchange_energy = 0.0;
    double exchange_vup = 0.0;
    double exchange_vdown = 0.0;
    BuiltinPbeMath::slater_spin(
        rho, zeta, exchange_energy, exchange_vup, exchange_vdown);
    const double rs = 0.62035049089940 / pow(rho, 1.0 / 3.0);
    double correlation_energy = 0.0;
    double correlation_vup = 0.0;
    double correlation_vdown = 0.0;
    BuiltinPbeMath::pw_spin(
        rs,
        zeta,
        correlation_energy,
        correlation_vup,
        correlation_vdown);
    exc = exchange_energy + correlation_energy;
    vup = exchange_vup + correlation_vup;
    vdown = exchange_vdown + correlation_vdown;
}

__device__ inline void evaluate_gga_pbe(
    const int exchange_flag,
    const int correlation_flag,
    const double rho_up,
    const double rho_down,
    const double grad_up_squared,
    const double grad_down_squared,
    const double total_grad_squared,
    double& energy,
    double& vup,
    double& vdown,
    double& v2x_up,
    double& v2x_down,
    double& v2c)
{
    double exchange_up = 0.0;
    double exchange_down = 0.0;
    double v1x_up = 0.0;
    double v1x_down = 0.0;
    v2x_up = 0.0;
    v2x_down = 0.0;
    const double total_rho = rho_up + rho_down;
    if (total_rho > vanishing)
    {
        if (rho_up > vanishing
            && sqrt(fabs(grad_up_squared)) > vanishing)
        {
            BuiltinPbeMath::pbex(2.0 * rho_up,
                                 4.0 * grad_up_squared,
                                 exchange_flag,
                                 exchange_up,
                                 v1x_up,
                                 v2x_up);
        }
        if (rho_down > vanishing
            && sqrt(fabs(grad_down_squared)) > vanishing)
        {
            BuiltinPbeMath::pbex(2.0 * rho_down,
                                 4.0 * grad_down_squared,
                                 exchange_flag,
                                 exchange_down,
                                 v1x_down,
                                 v2x_down);
        }
    }
    const double exchange_energy = 0.5 * (exchange_up + exchange_down);
    v2x_up *= 2.0;
    v2x_down *= 2.0;

    double correlation_energy = 0.0;
    double v1c_up = 0.0;
    double v1c_down = 0.0;
    v2c = 0.0;
    if (total_rho > epsr)
    {
        const double zeta_input
            = fabs((rho_up - rho_down) / total_rho);
        double zeta = zeta_input;
        if (!(fabs(zeta) - 1.0 > vanishing
              || sqrt(fabs(total_grad_squared)) <= vanishing))
        {
            const double clipped
                = fabs(zeta) < 1.0 - epsr ? fabs(zeta) : 1.0 - epsr;
            zeta = zeta > 0.0 ? clipped : -clipped;
            BuiltinPbeMath::pbec_spin(total_rho,
                                      zeta,
                                      total_grad_squared,
                                      correlation_flag,
                                      correlation_energy,
                                      v1c_up,
                                      v1c_down,
                                      v2c);
            if (zeta_input > 1.0 - epsr)
            {
                const double density_derivative
                    = 0.5 * ((1.0 + zeta) * v1c_up
                             + (1.0 - zeta) * v1c_down);
                v1c_up = density_derivative;
                v1c_down = density_derivative;
            }
        }
    }
    energy = exchange_energy + correlation_energy;
    vup = v1x_up + v1c_up;
    vdown = v1x_down + v1c_down;
}

__global__ void pointwise_xc_kernel(
    const int nrxx,
    const int exchange_flag,
    const int correlation_flag,
    const double* rho,
    const double* field_gradient,
    double* h,
    double* potential,
    double* etxc)
{
    const int ir = blockIdx.x * blockDim.x + threadIdx.x;
    double local_energy = 0.0;
    if (ir < nrxx)
    {
        const double total_density = rho[ir] + rho[4 * nrxx + ir];
        const NcggaRadialMath::RadialData radial
            = NcggaRadialMath::make_radial_data(rho[nrxx + ir],
                                                rho[2 * nrxx + ir],
                                                rho[3 * nrxx + ir],
                                                NcggaRadialMath::lca_radial_eta());
        const NcggaRadialMath::SpinMapData spin_map
            = NcggaRadialMath::make_spin_map_data(total_density, radial);
        double gradient_up[3] = {0.0, 0.0, 0.0};
        double gradient_down[3] = {0.0, 0.0, 0.0};
        for (int channel = 0; channel < 4; ++channel)
        {
            const double jacobian_up
                = NcggaRadialMath::spin_map_jacobian(
                    spin_map, 0, channel);
            const double jacobian_down
                = NcggaRadialMath::spin_map_jacobian(
                    spin_map, 1, channel);
            for (int component = 0; component < 3; ++component)
            {
                const double gradient
                    = field_gradient[(3 * channel + component) * nrxx + ir];
                gradient_up[component] += jacobian_up * gradient;
                gradient_down[component] += jacobian_down * gradient;
            }
        }

        double local_potential[4] = {0.0, 0.0, 0.0, 0.0};
        if (spin_map.absolute_density > vanishing)
        {
            double zeta
                = spin_map.clipped_magnitude / spin_map.absolute_density;
            if (fabs(zeta) > 1.0)
            {
                zeta = zeta > 0.0 ? 1.0 : -1.0;
            }
            double exc = 0.0;
            double vxc_up = 0.0;
            double vxc_down = 0.0;
            evaluate_lda_pbe(spin_map.absolute_density,
                             zeta,
                             exc,
                             vxc_up,
                             vxc_down);
            for (int channel = 0; channel < 4; ++channel)
            {
                local_potential[channel]
                    = e2
                      * (NcggaRadialMath::spin_map_jacobian(
                             spin_map, 0, channel)
                             * vxc_up
                         + NcggaRadialMath::spin_map_jacobian(
                               spin_map, 1, channel)
                               * vxc_down);
            }
            local_energy += e2 * exc * spin_map.absolute_density;
        }

        const double grad_up_squared = dot3(gradient_up, gradient_up);
        const double grad_down_squared
            = dot3(gradient_down, gradient_down);
        double total_gradient[3];
        for (int component = 0; component < 3; ++component)
        {
            total_gradient[component]
                = gradient_up[component] + gradient_down[component];
        }
        const double total_grad_squared
            = dot3(total_gradient, total_gradient);
        double gga_energy = 0.0;
        double gga_vup = 0.0;
        double gga_vdown = 0.0;
        double v2x_up = 0.0;
        double v2x_down = 0.0;
        double v2c = 0.0;
        evaluate_gga_pbe(exchange_flag,
                         correlation_flag,
                         spin_map.spin_density[0],
                         spin_map.spin_density[1],
                         grad_up_squared,
                         grad_down_squared,
                         total_grad_squared,
                         gga_energy,
                         gga_vup,
                         gga_vdown,
                         v2x_up,
                         v2x_down,
                         v2c);
        gga_vup *= e2;
        gga_vdown *= e2;
        for (int channel = 0; channel < 4; ++channel)
        {
            local_potential[channel]
                += NcggaRadialMath::spin_map_jacobian(
                       spin_map, 0, channel)
                       * gga_vup
                   + NcggaRadialMath::spin_map_jacobian(
                         spin_map, 1, channel)
                         * gga_vdown;
        }

        double h_up[3];
        double h_down[3];
        for (int component = 0; component < 3; ++component)
        {
            h_up[component]
                = e2
                  * ((v2x_up + v2c) * gradient_up[component]
                     + v2c * gradient_down[component]);
            h_down[component]
                = e2
                  * ((v2x_down + v2c) * gradient_down[component]
                     + v2c * gradient_up[component]);
            h[component * nrxx + ir] = h_up[component];
            h[(3 + component) * nrxx + ir] = h_down[component];
        }

        if (!spin_map.saturated)
        {
            double spin_flux[3];
            for (int component = 0; component < 3; ++component)
            {
                spin_flux[component]
                    = 0.5 * (h_up[component] - h_down[component]);
            }
            for (int channel = 1; channel < 4; ++channel)
            {
                double local_response = 0.0;
                for (int nu = 0; nu < 3; ++nu)
                {
                    const double h_dot_gradient
                        = spin_flux[0]
                              * field_gradient[(3 * (nu + 1)) * nrxx + ir]
                          + spin_flux[1]
                                * field_gradient[(3 * (nu + 1) + 1)
                                                     * nrxx
                                                 + ir]
                          + spin_flux[2]
                                * field_gradient[(3 * (nu + 1) + 2)
                                                     * nrxx
                                                 + ir];
                    local_response
                        += NcggaRadialMath::radial_jacobian(
                               radial, nu, channel - 1)
                           * h_dot_gradient;
                }
                local_potential[channel] += local_response;
            }
        }

        for (int channel = 0; channel < 4; ++channel)
        {
            potential[channel * nrxx + ir] = local_potential[channel];
        }
        local_energy += e2 * gga_energy;
    }

    __shared__ double block_sum[threads_per_block];
    block_sum[threadIdx.x] = local_energy;
    __syncthreads();
    for (int stride = threads_per_block / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        atomicAdd(etxc, block_sum[0]);
    }
}

__global__ void build_spin_flux_kernel(
    const int nrxx,
    const int channel,
    const double* rho,
    const double* h,
    double* spin_flux)
{
    const int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir >= nrxx)
    {
        return;
    }
    const double total_density = rho[ir] + rho[4 * nrxx + ir];
    const NcggaRadialMath::RadialData radial
        = NcggaRadialMath::make_radial_data(rho[nrxx + ir],
                                            rho[2 * nrxx + ir],
                                            rho[3 * nrxx + ir],
                                            NcggaRadialMath::lca_radial_eta());
    const NcggaRadialMath::SpinMapData spin_map
        = NcggaRadialMath::make_spin_map_data(total_density, radial);
    const double jacobian_up
        = NcggaRadialMath::spin_map_jacobian(spin_map, 0, channel);
    const double jacobian_down
        = NcggaRadialMath::spin_map_jacobian(spin_map, 1, channel);
    for (int component = 0; component < 3; ++component)
    {
        spin_flux[component * nrxx + ir]
            = jacobian_up * h[component * nrxx + ir]
              + jacobian_down * h[(3 + component) * nrxx + ir];
    }
}

__global__ void subtract_divergence_kernel(
    const int nrxx,
    const int channel,
    const double* divergence,
    double* potential)
{
    const int ir = blockIdx.x * blockDim.x + threadIdx.x;
    if (ir < nrxx)
    {
        potential[channel * nrxx + ir] -= divergence[ir];
    }
}

__global__ void vtxc_kernel(
    const int nrxx,
    const double* rho,
    const double* potential,
    double* vtxc)
{
    const int ir = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0;
    if (ir < nrxx)
    {
        for (int channel = 0; channel < 4; ++channel)
        {
            value += potential[channel * nrxx + ir]
                     * rho[channel * nrxx + ir];
        }
    }

    __shared__ double block_sum[threads_per_block];
    block_sum[threadIdx.x] = value;
    __syncthreads();
    for (int stride = threads_per_block / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        atomicAdd(vtxc, block_sum[0]);
    }
}

void calculate_field_gradient(Workspace& workspace,
                              const int channel,
                              const double tpiba)
{
    const int reciprocal_blocks
        = (workspace.npw + threads_per_block - 1) / threads_per_block;
    const double* real_input = workspace.rho + channel * workspace.nrxx;
    workspace.basis->real2recip_gpu_companion(
        real_input,
        reinterpret_cast<std::complex<double>*>(workspace.reciprocal),
        false,
        1.0);
    if (channel == 0)
    {
        add_reciprocal_kernel<<<reciprocal_blocks, threads_per_block>>>(
            workspace.npw, workspace.rhog_core, workspace.reciprocal);
    }
    for (int component = 0; component < 3; ++component)
    {
        reciprocal_gradient_kernel<<<reciprocal_blocks,
                                     threads_per_block>>>(
            workspace.npw,
            component,
            false,
            tpiba,
            workspace.gcar,
            workspace.reciprocal,
            workspace.reciprocal_scratch);
        workspace.basis->recip2real_gpu_companion(
            reinterpret_cast<std::complex<double>*>(
                workspace.reciprocal_scratch),
            workspace.field_gradient
                + (3 * channel + component) * workspace.nrxx,
            false,
            1.0);
    }
}

void apply_divergence(Workspace& workspace,
                      const int channel,
                      const double tpiba)
{
    const int real_blocks
        = (workspace.nrxx + threads_per_block - 1) / threads_per_block;
    const int reciprocal_blocks
        = (workspace.npw + threads_per_block - 1) / threads_per_block;
    build_spin_flux_kernel<<<real_blocks, threads_per_block>>>(
        workspace.nrxx,
        channel,
        workspace.rho,
        workspace.h,
        workspace.spin_flux);
    for (int component = 0; component < 3; ++component)
    {
        workspace.basis->real2recip_gpu_companion(
            workspace.spin_flux + component * workspace.nrxx,
            reinterpret_cast<std::complex<double>*>(workspace.reciprocal),
            false,
            1.0);
        reciprocal_gradient_kernel<<<reciprocal_blocks,
                                     threads_per_block>>>(
            workspace.npw,
            component,
            component != 0,
            tpiba,
            workspace.gcar,
            workspace.reciprocal,
            workspace.reciprocal_scratch);
    }
    workspace.basis->recip2real_gpu_companion(
        reinterpret_cast<std::complex<double>*>(
            workspace.reciprocal_scratch),
        workspace.real_scratch,
        false,
        1.0);
    subtract_divergence_kernel<<<real_blocks, threads_per_block>>>(
        workspace.nrxx,
        channel,
        workspace.real_scratch,
        workspace.potential);
}

} // namespace

bool try_v_xc_ncgga_sf_builtin_gpu(
    const int nrxx,
    const double omega,
    const double tpiba,
    const Charge* chr,
    const std::vector<int>& functional_ids,
    std::tuple<double, double, ModuleBase::matrix>& result)
{
    PbeFamily family;
    if (chr == nullptr
        || chr->rhopw == nullptr
        || nrxx != chr->rhopw->nrxx
        || !chr->rhopw->has_gpu_fft_companion()
        || !get_pbe_family(functional_ids, family))
    {
        return false;
    }

    static_assert(sizeof(std::complex<double>)
                      == sizeof(thrust::complex<double>),
                  "host and device complex layouts must have equal size");
    static_assert(sizeof(ModuleBase::Vector3<double>) == 3 * sizeof(double),
                  "PW G vectors must be tightly packed");
    static_assert(std::is_standard_layout<ModuleBase::Vector3<double>>::value,
                  "PW G vectors must use standard layout");

    ModuleBase::timer::start("XC_Functional", "v_xc_ncgga_sf_gpu");
    Workspace& workspace = get_workspace();
    workspace.ensure(chr->rhopw);
    const std::size_t real_bytes
        = static_cast<std::size_t>(nrxx) * sizeof(double);
    const std::size_t reciprocal_bytes
        = static_cast<std::size_t>(workspace.npw)
          * sizeof(thrust::complex<double>);
    for (int channel = 0; channel < 4; ++channel)
    {
        CHECK_CUDA(cudaMemcpy(workspace.rho + channel * nrxx,
                              chr->rho[channel],
                              real_bytes,
                              cudaMemcpyHostToDevice));
    }
    if (chr->rho_core != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(workspace.rho + 4 * nrxx,
                              chr->rho_core,
                              real_bytes,
                              cudaMemcpyHostToDevice));
    }
    else
    {
        CHECK_CUDA(cudaMemset(workspace.rho + 4 * nrxx, 0, real_bytes));
    }
    if (chr->rhog_core != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(workspace.rhog_core,
                              chr->rhog_core,
                              reciprocal_bytes,
                              cudaMemcpyHostToDevice));
    }
    else
    {
        CHECK_CUDA(cudaMemset(workspace.rhog_core, 0, reciprocal_bytes));
    }
    CHECK_CUDA(cudaMemcpy(
        workspace.gcar,
        reinterpret_cast<const double*>(chr->rhopw->gcar),
        3 * static_cast<std::size_t>(workspace.npw) * sizeof(double),
        cudaMemcpyHostToDevice));

    for (int channel = 0; channel < 4; ++channel)
    {
        calculate_field_gradient(workspace, channel, tpiba);
    }

    CHECK_CUDA(cudaMemset(workspace.reductions, 0, 2 * sizeof(double)));
    const int real_blocks
        = (workspace.nrxx + threads_per_block - 1) / threads_per_block;
    pointwise_xc_kernel<<<real_blocks, threads_per_block>>>(
        workspace.nrxx,
        family.exchange_flag,
        family.correlation_flag,
        workspace.rho,
        workspace.field_gradient,
        workspace.h,
        workspace.potential,
        workspace.reductions);
    CHECK_CUDA(cudaGetLastError());

    for (int channel = 0; channel < 4; ++channel)
    {
        apply_divergence(workspace, channel, tpiba);
    }

    vtxc_kernel<<<real_blocks, threads_per_block>>>(
        workspace.nrxx,
        workspace.rho,
        workspace.potential,
        workspace.reductions + 1);
    CHECK_CUDA(cudaGetLastError());

    double host_reductions[2] = {0.0, 0.0};
    CHECK_CUDA(cudaMemcpy(host_reductions,
                          workspace.reductions,
                          2 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    ModuleBase::matrix potential(4, nrxx, false);
    CHECK_CUDA(cudaMemcpy(potential.c,
                          workspace.potential,
                          4 * real_bytes,
                          cudaMemcpyDeviceToHost));

    double etxc = host_reductions[0];
    double vtxc = host_reductions[1];
#ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc);
    Parallel_Reduce::reduce_pool(vtxc);
#endif
    const double grid_scale = omega / chr->rhopw->nxyz;
    etxc *= grid_scale;
    vtxc *= grid_scale;
    result = std::make_tuple(etxc, vtxc, std::move(potential));
    ModuleBase::timer::end("XC_Functional", "v_xc_ncgga_sf_gpu");
    return true;
}

} // namespace NCGGA_SF_Builtin
} // namespace ModuleXC
