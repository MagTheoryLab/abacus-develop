#include <algorithm>
#include "diago_dav_subspace.h"

#include "diago_iter_assist.h"

#include "source_base/module_device/device.h"
#include "source_base/parallel_cell.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/kernels/dsp/dsp_connector.h"
// #include "source_base/module_container/ATen/kernels/lapack.h"

#include <ATen/kernels/lapack.h>

#include "source_hsolver/kernels/hegvd_op.h"
#include "source_hsolver/diag_hs_para.h"
#include "source_hsolver/kernels/bpcg_kernel_op.h" // normalize_op, precondition_op, apply_eigenvalues_op

#include <vector>
#include <cmath>
#include <limits>

#ifdef __MPI
#include <mpi.h>
#include "source_base/parallel_comm.h"
#include "source_base/parallel_device.h"
#endif

using namespace hsolver;

template <typename T, typename Device>
Diago_DavSubspace<T, Device>::Diago_DavSubspace(const std::vector<Real>& precondition_in,
                                                const int& nband_in,
                                                const int& nbasis_in,
                                                const int& david_ndim_in,
                                                const double& diag_thr_in,
                                                const int& diag_nmax_in,
                                                const diag_comm_info& diag_comm_in,
                                                const int diag_subspace_in,
                                                const int diago_subspace_bs_in,
                                                const bool overlap_is_identity_in)
    : precondition(precondition_in), n_band(nband_in), dim(nbasis_in), nbase_x(nband_in * david_ndim_in),
      overlap_is_identity(overlap_is_identity_in), diag_thr(diag_thr_in), iter_nmax(diag_nmax_in), diag_comm(diag_comm_in),
        diag_subspace(diag_subspace_in), diago_subspace_bs(diago_subspace_bs_in)
{
    this->device = base_device::get_device_type(this->ctx);

    this->one = &one_;
    this->zero = &zero_;
    this->neg_one = &neg_one_;

    assert(david_ndim_in > 1);
    // Block ownership can be empty or uneven; use the actual communicator sum.
    ModuleBase::CommunicationDomain domain;
#ifdef __MPI
    domain.initialize(this->diag_comm.comm);
#endif
    const int global_dimension = domain.sum(this->dim);
    assert(this->nbase_x < global_dimension);
    assert(diag_subspace >= 0 && diag_subspace < 3);

    // TODO: Added memory usage statistics

    const int local_storage = std::max(this->nbase_x * this->dim, 1);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    resmem_complex_op()(this->psi_in_iter, local_storage, "DAV::psi_in_iter");
    setmem_complex_op()(this->psi_in_iter, 0, this->nbase_x * this->dim);

    // the product of H and psi in the reduced psi set
    resmem_complex_op()(this->hpsi, local_storage, "DAV::hpsi");
    setmem_complex_op()(this->hpsi, 0, this->nbase_x * this->dim);

    // For NC, keep a single basis: S*psi is exactly psi, including after restart.
    if (this->overlap_is_identity)
    {
        this->spsi = this->psi_in_iter;
    }
    else
    {
        resmem_complex_op()(this->spsi, local_storage, "DAV::spsi");
        setmem_complex_op()(this->spsi, 0, this->nbase_x * this->dim);
    }

    // Hamiltonian on the reduced psi set
    resmem_complex_op()(this->hcc, this->nbase_x * this->nbase_x, "DAV::hcc");
    setmem_complex_op()(this->hcc, 0, this->nbase_x * this->nbase_x);

    // Overlap on the reduced psi set
    resmem_complex_op()(this->scc, this->nbase_x * this->nbase_x, "DAV::scc");
    setmem_complex_op()(this->scc, 0, this->nbase_x * this->nbase_x);

    // Eigenvectors
    resmem_complex_op()(this->vcc, this->nbase_x * this->nbase_x, "DAV::vcc");
    setmem_complex_op()(this->vcc, 0, this->nbase_x * this->nbase_x);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#if defined(__CUDA) || defined(__ROCM)
    if (this->device == base_device::GpuDevice)
    {
        resmem_real_op()(this->d_precondition, nbasis_in);
        syncmem_var_h2d_op()(this->d_precondition, this->precondition.data(), nbasis_in);
        resmem_complex_op()(this->d_scc, this->nbase_x * this->nbase_x);
        resmem_real_op()(this->d_eigenvalue, this->nbase_x);
    }
#endif
}

template <typename T, typename Device>
Diago_DavSubspace<T, Device>::~Diago_DavSubspace()
{
    delmem_complex_op()(this->psi_in_iter);

    delmem_complex_op()(this->hpsi);
    if (!this->overlap_is_identity)
    {
        delmem_complex_op()(this->spsi);
    }
    delmem_complex_op()(this->hcc);
    delmem_complex_op()(this->scc);
    delmem_complex_op()(this->vcc);

#if defined(__CUDA) || defined(__ROCM)
    if (this->device == base_device::GpuDevice)
    {
        delmem_real_op()(this->d_precondition);
        delmem_complex_op()(this->d_scc);
        delmem_real_op()(this->d_eigenvalue);
    }
#endif
}

template <typename T, typename Device>
int Diago_DavSubspace<T, Device>::diag_once(const HSOperator<T, Device>& op,
                                            T* psi_in,
                                            const int psi_in_dmax,
                                            Real* eigenvalue_in_hsolver,
                                            const std::vector<double>& ethr_band)
{
    ModuleBase::timer::start("Diago_DavSubspace", "diag_once");
    const int matrix_ld = std::max(this->dim, 1);
    const int psi_ld = std::max(psi_in_dmax, 1);

    // the eigenvalues in dav iter
    std::vector<Real> eigenvalue_iter(this->nbase_x, 0.0);

    // convflag[m] = true if the m th band is convergent
    std::vector<bool> convflag(this->n_band, false);

    // unconv[m] store the number of the m th unconvergent band
    std::vector<int> unconv(this->n_band);

    // the dimension of the reduced psi set
    int nbase = 0;

    // the number of the unconvergent bands
    this->notconv = this->n_band;

    ModuleBase::timer::start("Diago_DavSubspace", "first");

    syncmem_complex_2d_op()(this->psi_in_iter, this->dim, psi_in, psi_in_dmax, this->dim, this->n_band);
    for (int m = 0; m < this->n_band; m++)
    {
        unconv[m] = m;
    }

    if (!this->overlap_is_identity)
    {
        op.spsi(this->psi_in_iter, this->spsi, this->dim, this->notconv);
    }
    const int initial_rank = this->orthogonalize(0, this->n_band);
    if (initial_rank != this->n_band)
    {
        ModuleBase::WARNING_QUIT("Diago_DavSubspace", "Linearly dependent initial wavefunctions");
    }
    op.hpsi(this->psi_in_iter, this->hpsi, this->dim, this->notconv);

    this->cal_elem(this->dim, nbase, this->notconv, this->psi_in_iter, this->spsi, this->hpsi, this->hcc, this->scc);

    this->diag_zhegvx(nbase, this->notconv, this->hcc, this->scc, this->nbase_x, &eigenvalue_iter, this->vcc);

    for (size_t m = 0; m < this->n_band; m++)
    {
        eigenvalue_in_hsolver[m] = eigenvalue_iter[m];
    }

    ModuleBase::timer::end("Diago_DavSubspace", "first");

    int dav_iter = 0;

    do
    {
        dav_iter++;

        const int added = this->cal_grad(op,
                       this->dim,
                       nbase,
                       this->notconv,
                       this->psi_in_iter,
                       this->hpsi,
                       this->spsi,
                       this->vcc,
                       unconv.data(),
                       &eigenvalue_iter);

        if (added == 0)
        {
            // Preserve the current Ritz vectors and the unconverged count. A
            // rank-deficient correction block is not evidence of convergence.
            ModuleBase::gemm_op<T, Device>()('N', 'N', this->dim, this->n_band, nbase,
                this->one, this->psi_in_iter, matrix_ld, this->vcc, this->nbase_x,
                this->zero, psi_in, psi_ld);
            ModuleBase::timer::end("Diago_DavSubspace", "diag_once");
            return dav_iter;
        }

        this->cal_elem(this->dim,
                       nbase,
                       added,
                       this->psi_in_iter,
                       this->spsi,
                       this->hpsi,
                       this->hcc,
                       this->scc);

        this->diag_zhegvx(nbase, this->n_band, this->hcc, this->scc, this->nbase_x, &eigenvalue_iter, this->vcc);

        // check convergence and update eigenvalues
        ModuleBase::timer::start("Diago_DavSubspace", "check_update");

        this->notconv = 0;
        for (int m = 0; m < this->n_band; m++)
        {
            convflag[m] = (std::abs(eigenvalue_iter[m] - eigenvalue_in_hsolver[m]) < ethr_band[m]);

            if (!convflag[m])
            {
                unconv[this->notconv] = m;
                this->notconv++;
            }

            eigenvalue_in_hsolver[m] = eigenvalue_iter[m];
        }

        ModuleBase::timer::end("Diago_DavSubspace", "check_update");

        if ((this->notconv == 0) || (nbase + this->notconv + 1 > this->nbase_x) || (dav_iter == this->iter_nmax))
        {
            ModuleBase::timer::start("Diago_DavSubspace", "last");

            // updata eigenvectors of Hamiltonian
            setmem_complex_op()(psi_in, 0, n_band * psi_in_dmax);

#ifdef __DSP
            ModuleBase::gemm_op_mt<T, Device>() // In order to not coding another whole template, using this method to
                                                // minimize the code change.
#else
            ModuleBase::gemm_op<T, Device>()
#endif
                ('N',
                 'N',
                 this->dim,
                 this->n_band,
                 nbase,
                 this->one,
                 this->psi_in_iter,
                 matrix_ld,
                 this->vcc,
                 this->nbase_x,
                 this->zero,
                 psi_in,
                 psi_ld);

            if (!this->notconv || (dav_iter == this->iter_nmax))
            {
                // overall convergence or last iteration: exit the iteration

                ModuleBase::timer::end("Diago_DavSubspace", "last");
                break;
            }
            else
            {
                // if the dimension of the reduced basis set is becoming too large,
                // then replace the first N (=nband) basis vectors with the current
                // estimate of the eigenvectors and set the basis dimension to N;

                // update this->psi_in_iter according to psi_in
                syncmem_complex_2d_op()(this->psi_in_iter, this->dim, psi_in, psi_in_dmax, this->dim, this->n_band);

                this->refresh(this->dim,
                              this->n_band,
                              nbase,
                              eigenvalue_in_hsolver,
                              this->psi_in_iter,
                              this->hpsi,
                              this->spsi,
                              this->hcc,
                              this->scc,
                              this->vcc);

                ModuleBase::timer::end("Diago_DavSubspace", "last");
            }
        }

    } while (true);

    ModuleBase::timer::end("Diago_DavSubspace", "diag_once");

    return dav_iter;
}

template <typename T, typename Device>
int Diago_DavSubspace<T, Device>::cal_grad(const HSOperator<T, Device>& op,
                                            const int& dim,
                                            const int& nbase,
                                            const int& notconv,
                                            T* psi_iter,
                                            T* hpsi,
                                            T* spsi,
                                            T* vcc_in,
                                            const int* unconv,
                                            std::vector<Real>* eigenvalue_iter)
{
    ModuleBase::timer::start("Diago_DavSubspace", "cal_grad");
    // Residual construction scales and compacts coefficients. Keep the Ritz
    // vectors intact when all correction directions are dependent.
    T* vcc = nullptr;
    resmem_complex_op()(vcc, this->nbase_x * this->n_band);
    syncmem_complex_op()(vcc, vcc_in, this->nbase_x * this->n_band);

    const int matrix_ld = std::max(this->dim, 1);

    for (size_t i = 0; i < notconv; i++)
    {
        if (unconv[i] != i)
        {
            syncmem_complex_op()(vcc + i * this->nbase_x, vcc + unconv[i] * this->nbase_x, nbase);
            (*eigenvalue_iter)[i] = (*eigenvalue_iter)[unconv[i]];
        }
    }

    if (notconv > 1){

#ifdef __DSP
    ModuleBase::gemm_op_mt<T, Device>()
#else
    ModuleBase::gemm_op<T, Device>()
#endif
                        ('N',
                         'N',
                         this->dim,
                         notconv,
                         nbase,
                         this->one,
                         hpsi,
                         matrix_ld,
                         vcc,
                         this->nbase_x,
                         this->zero,
                         psi_iter + (nbase) * this->dim,
                         matrix_ld);
    } else
    {

#ifdef __DSP
    ModuleBase::gemv_op_mt<T, Device>()
#else
    ModuleBase::gemv_op<T, Device>()
#endif
                        ('N',
                         this->dim,     // m: row of A
                         nbase,         // n: col of A
                         this->one,     // alpha
                         hpsi,          // A dim * nbase
                         matrix_ld,     // LDA: if(N) max(1,m)
                         vcc,           // X nbase
                         1,             // incx
                         this->zero,    // beta
                         psi_iter + (nbase) * this->dim,  // Y dim
                         1              // incy
                        );
    }


    // Eigenvalues operation section
    Real* e_temp_hd = eigenvalue_iter->data();
    if (this->device == base_device::GpuDevice)
    {
        syncmem_var_h2d_op()(this->d_eigenvalue, eigenvalue_iter->data(), this->nbase_x);
        e_temp_hd = this->d_eigenvalue;
    }

    // vcc = - vcc * eigenvalue
    ModuleBase::matrix_mul_vector_op<T, Device>()(nbase, notconv, vcc, this->nbase_x, e_temp_hd, -1.0, vcc, this->nbase_x);

    if (notconv > 1){

#ifdef __DSP
    ModuleBase::gemm_op_mt<T, Device>()
#else
    ModuleBase::gemm_op<T, Device>()
#endif
        ('N',
         'N',
         this->dim,
         notconv,
         nbase,
         this->one,
         spsi,
         matrix_ld,
         vcc,
         this->nbase_x,
         this->one,
         psi_iter + nbase * this->dim,
         matrix_ld);
    } else
    {
#ifdef __DSP
    ModuleBase::gemv_op_mt<T, Device>()
#else
    ModuleBase::gemv_op<T, Device>()
#endif
        ('N',
         this->dim,     // m: row of A
         nbase,         // n: col of A
         this->one,     // alpha
         spsi,          // A dim * nbase
         matrix_ld,     // LDA: if(N) max(1,m)
         vcc,           // X nbase
         1,             // incx
         this->one,    // beta
         psi_iter + nbase * this->dim,  // Y dim
         1              // incy
        );
    }

    // Precondition section
#if defined(__CUDA) || defined(__ROCM)
    if (this->device == base_device::GpuDevice)
    {
        precondition_op<T, Device>()(this->dim,
                                    psi_iter,
                                    nbase,
                                    notconv,
                                    d_precondition,
                                    this->d_eigenvalue);
    }
    else
#endif
    {
        precondition_op<T, Device>()(this->dim,
                                    psi_iter,
                                    nbase,
                                    notconv,
                                    this->precondition.data(),
                                    (*eigenvalue_iter).data());
    }

    if (!this->overlap_is_identity)
    {
        op.spsi(psi_iter + nbase * dim, spsi + nbase * dim, dim, notconv);
    }
    const int added = this->orthogonalize(nbase, notconv);
    if (added > 0)
    {
        op.hpsi(psi_iter + nbase * dim, hpsi + nbase * dim, dim, added);
    }
    delmem_complex_op()(vcc);

    ModuleBase::timer::end("Diago_DavSubspace", "cal_grad");
    return added;
}

template <typename T, typename Device>
int Diago_DavSubspace<T, Device>::orthogonalize(const int nbase, const int count)
{
    const int matrix_ld = std::max(this->dim, 1);
    // Project out the retained basis and whiten the correction block twice.
    // Independent normalization does not remove nearly dependent directions;
    // their ill-conditioned overlap invalidates the identity used by refresh.
    // The unused H*psi columns provide scratch space without a PW-sized allocation.
    ct::Tensor overlap(ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value,
                       {std::max(nbase, count) * count});
    ct::Tensor spectrum(ct::DataTypeToEnum<Real>::value, ct::DeviceTypeToEnum<ct_Device>::value,
                        {count});
    T* q = this->psi_in_iter + nbase * this->dim;
    T* sq = this->spsi + nbase * this->dim;
    T* scratch = this->hpsi + nbase * this->dim;
    T* coefficients = overlap.data<T>();
    std::vector<Real> eigenvalues(count);
    std::vector<T> transform(count * count);
    int rank = count;
    for (int pass = 0; pass < 2 && rank > 0; ++pass)
    {
        Real original_norm = ModuleBase::dot_real_op<T, Device>()(this->dim * rank, q, sq, false);
        if (this->diag_comm.nproc > 1)
        {
            Parallel_Reduce::reduce_pool(original_norm);
        }
        if (nbase > 0)
        {
            ModuleBase::gemm_op<T, Device>()('C', 'N', nbase, rank, this->dim,
                this->one, this->psi_in_iter, matrix_ld, sq, matrix_ld,
                this->zero, coefficients, nbase);
            if (this->diag_comm.nproc > 1)
            {
                Parallel_Reduce::reduce_pool(coefficients, nbase * rank);
            }
            ModuleBase::gemm_op<T, Device>()('N', 'N', this->dim, rank, nbase,
                this->neg_one, this->psi_in_iter, matrix_ld, coefficients, nbase,
                this->one, q, matrix_ld);
            if (!this->overlap_is_identity)
            {
                ModuleBase::gemm_op<T, Device>()('N', 'N', this->dim, rank, nbase,
                    this->neg_one, this->spsi, matrix_ld, coefficients, nbase,
                    this->one, sq, matrix_ld);
            }
        }
        ModuleBase::gemm_op<T, Device>()('C', 'N', rank, rank, this->dim,
            this->one, q, matrix_ld, sq, matrix_ld, this->zero, coefficients, rank);
        if (this->diag_comm.nproc > 1)
        {
            Parallel_Reduce::reduce_pool(coefficients, rank * rank);
        }
        std::fill(eigenvalues.begin(), eigenvalues.end(), Real(0));
        std::fill(transform.begin(), transform.end(), T(0));
        if (this->diag_comm.rank == 0)
        {
            ct::kernels::lapack_heevd<T, ct_Device>()(rank, coefficients, rank, spectrum.data<Real>());
            syncmem_var_d2h_op()(eigenvalues.data(), spectrum.data<Real>(), rank);
            syncmem_d2h_op()(transform.data(), coefficients, rank * rank);
        }
        if (this->diag_comm.nproc > 1)
        {
            // Only the root contributes, so every rank uses the same rotation
            // even for a degenerate overlap spectrum.
            Parallel_Reduce::reduce_pool(eigenvalues.data(), rank);
            Parallel_Reduce::reduce_pool(transform.data(), rank * rank);
        }
        const Real cutoff = 64 * std::numeric_limits<Real>::epsilon()
                            * std::max(std::abs(original_norm), std::abs(eigenvalues[rank - 1]));
        if (!std::isfinite(eigenvalues.front()) || !std::isfinite(eigenvalues[rank - 1])
            || eigenvalues.front() < -cutoff)
        {
            ModuleBase::WARNING_QUIT("Diago_DavSubspace", "Invalid wavefunction overlap");
        }
        int retained = 0;
        for (int j = 0; j < rank; ++j)
        {
            if (eigenvalues[j] <= cutoff)
            {
                continue;
            }
            const Real scale = 1 / std::sqrt(eigenvalues[j]);
            for (int i = 0; i < rank; ++i)
            {
                transform[retained * rank + i] = transform[j * rank + i] * scale;
            }
            ++retained;
        }
        if (retained > 0)
        {
            syncmem_h2d_op()(coefficients, transform.data(), rank * retained);
            ModuleBase::gemm_op<T, Device>()('N', 'N', this->dim, retained, rank,
                this->one, q, matrix_ld, coefficients, rank,
                this->zero, scratch, matrix_ld);
            syncmem_complex_op()(q, scratch, this->dim * retained);
            if (!this->overlap_is_identity)
            {
                ModuleBase::gemm_op<T, Device>()('N', 'N', this->dim, retained, rank,
                    this->one, sq, matrix_ld, coefficients, rank,
                    this->zero, scratch, matrix_ld);
                syncmem_complex_op()(sq, scratch, this->dim * retained);
            }
        }
        rank = retained;
    }
    return rank;
}

template <typename T, typename Device>
void Diago_DavSubspace<T, Device>::cal_elem(const int& dim,
                                            int& nbase,
                                            const int& notconv,
                                            const T* psi_iter,
                                            const T* spsi,
                                            const T* hpsi,
                                            T* hcc,
                                            T* scc)
{
    ModuleBase::timer::start("Diago_DavSubspace", "cal_elem");
    const int matrix_ld = std::max(this->dim, 1);

    if (this->dim == 0)
    {
        const int contribution_size = this->nbase_x * notconv;
        T* const h_contribution = hcc + nbase * this->nbase_x;
        T* const s_contribution = scc + nbase * this->nbase_x;
        // GEMV does not overwrite its output when the local row count is zero.
        setmem_complex_op()(h_contribution, 0, contribution_size);
        setmem_complex_op()(s_contribution, 0, contribution_size);
    }

    if (notconv > 1){
#ifdef __DSP
    ModuleBase::gemm_op_mt<T, Device>()
#else
    ModuleBase::gemm_op<T, Device>()
#endif
        ('C',
         'N',
         nbase + notconv,
         notconv,
         this->dim,
         this->one,
         psi_iter,
         matrix_ld,
         &hpsi[nbase * this->dim],
         matrix_ld,
         this->zero,
         &hcc[nbase * this->nbase_x],
         this->nbase_x);

#ifdef __DSP
    ModuleBase::gemm_op_mt<T, Device>()
#else
    ModuleBase::gemm_op<T, Device>()
#endif
        ('C',
         'N',
         nbase + notconv,
         notconv,
         this->dim,
         this->one,
         psi_iter,
         matrix_ld,
         spsi + nbase * this->dim,
         matrix_ld,
         this->zero,
         &scc[nbase * this->nbase_x],
         this->nbase_x);

        } else {

#ifdef __DSP
    ModuleBase::gemv_op_mt<T, Device>()
#else
    ModuleBase::gemv_op<T, Device>()
#endif
        ('C',
         this->dim,                 // m: row of A
         nbase + notconv,          // n: col of A
         this->one,                // alpha
         psi_iter,                 // A dim * nbase
         matrix_ld,                // LDA: if(N) max(1,m)
         &hpsi[nbase * this->dim], // X nbase
         1,                        // incx
         this->zero,               // beta
         &hcc[nbase * this->nbase_x], // Y dim
         1                         // incy
        );
#ifdef __DSP
    ModuleBase::gemv_op_mt<T, Device>()
#else
    ModuleBase::gemv_op<T, Device>()
#endif
        ('C',
         this->dim,                 // m: row of A
         nbase + notconv,          // n: col of A
         this->one,                // alpha
         psi_iter,                 // A dim * nbase
         matrix_ld,                // LDA: if(N) max(1,m)
         spsi + nbase * this->dim, // X nbase
         1,                        // incx
         this->zero,               // beta
         &scc[nbase * this->nbase_x], // Y dim
         1                         // incy
        );

        }


#ifdef __MPI
    if (this->diag_comm.nproc > 1)
    {
#ifdef __DSP
        // Only on dsp hardware need an extra space to reduce data
        mtfunc::dsp_dav_subspace_reduce(hcc, scc, nbase, this->nbase_x, notconv, this->diag_comm.comm);
#else
        assert(this->diag_comm.comm == POOL_WORLD);
        Parallel_Reduce::reduce_pool(hcc + nbase * this->nbase_x, notconv * this->nbase_x);
        Parallel_Reduce::reduce_pool(scc + nbase * this->nbase_x, notconv * this->nbase_x);
#endif
    }
#endif

    const size_t last_nbase = nbase; // init: last_nbase = 0
    nbase = nbase + notconv;

    ModuleBase::timer::end("Diago_DavSubspace", "cal_elem");
    return;
}

template <typename T, typename Device>
void Diago_DavSubspace<T, Device>::diag_zhegvx(const int& nbase,
                                               const int& nband,
                                               T* hcc,
                                               T* scc,
                                               const int& nbase_x,
                                               std::vector<Real>* eigenvalue_iter,
                                               T* vcc)
{
    ModuleBase::timer::start("Diago_DavSubspace", "diag_zhegvx");
    assert(nbase_x >= std::max(1, nbase));

    if (this->device == base_device::GpuDevice)
    {
#if defined(__CUDA) || defined(__ROCM)
        if (this->diag_comm.rank == 0)
        {
            syncmem_complex_op()(this->d_scc, scc, nbase * this->nbase_x);
            ct::kernels::lapack_hegvd<T, ct_Device>()(nbase, this->nbase_x, this->hcc, this->d_scc, this->d_eigenvalue, this->vcc);
            syncmem_var_d2h_op()((*eigenvalue_iter).data(), this->d_eigenvalue, this->nbase_x);
        }
#endif
    }
    else
    {
        if (this->diag_subspace == 0)
        {
            if (this->diag_comm.rank == 0)
            {
                std::vector<std::vector<T>> h_diag(nbase, std::vector<T>(nbase, *this->zero));
                std::vector<std::vector<T>> s_diag(nbase, std::vector<T>(nbase, *this->zero));

                for (size_t i = 0; i < nbase; i++)
                {
                    for (size_t j = 0; j < nbase; j++)
                    {
                        h_diag[i][j] = hcc[i * this->nbase_x + j];
                        s_diag[i][j] = scc[i * this->nbase_x + j];
                    }
                }
                hegvx_op<T, Device>()(this->ctx,
                                      nbase,
                                      this->nbase_x,
                                      this->hcc,
                                      this->scc,
                                      nband,
                                      (*eigenvalue_iter).data(),
                                      this->vcc);
                // reset:
                for (size_t i = 0; i < nbase; i++)
                {
                    for (size_t j = 0; j < nbase; j++)
                    {
                        hcc[i * this->nbase_x + j] = h_diag[i][j];
                        scc[i * this->nbase_x + j] = s_diag[i][j];
                    }

                    for (size_t j = nbase; j < this->nbase_x; j++)
                    {
                        hcc[i * this->nbase_x + j] = *this->zero;
                        hcc[j * this->nbase_x + i] = *this->zero;
                        scc[i * this->nbase_x + j] = *this->zero;
                        scc[j * this->nbase_x + i] = *this->zero;
                    }
                }
            }
        }
        else
        {
#ifdef __MPI
            std::vector<T> h_diag;
            std::vector<T> s_diag;
            std::vector<T> vcc_tmp;
            if (this->diag_comm.rank == 0)
            {
                h_diag.resize(nbase * nbase, *this->zero);
                s_diag.resize(nbase * nbase, *this->zero);
                vcc_tmp.resize(nbase * nbase, *this->zero);
                for (size_t i = 0; i < nbase; i++)
                {
                    for (size_t j = 0; j < nbase; j++)
                    {
                        h_diag[i * nbase + j] = hcc[i * this->nbase_x + j];
                        s_diag[i * nbase + j] = scc[i * this->nbase_x + j];
                    }
                }
            }
            diago_hs_para(h_diag.data(),
                          s_diag.data(),
                          nbase,
                          nband,
                          (*eigenvalue_iter).data(),
                          vcc_tmp.data(),
                          this->diag_comm.comm,
                          this->diag_subspace,
                          this->diago_subspace_bs);
            if (this->diag_comm.rank == 0)
            {
                for (size_t i = 0; i < nband; i++)
                {
                    for (size_t j = 0; j < nbase; j++)
                    {
                        vcc[i * this->nbase_x + j] = vcc_tmp[i * nbase + j];
                    }
                }
            }
#else
            std::cout << "Error: parallel diagonalization is not supported in serial mode." << std::endl;
            exit(1);
#endif
        }
    }

#ifdef __MPI
    if (this->diag_comm.nproc > 1)
    {
        // vcc: nbase * nband
        for (int i = 0; i < nband; i++)
        {
            // fix a bug here when vcc is real: bcast according to the true type
            Parallel_Common::bcast_data(&vcc[i * this->nbase_x], nbase, this->diag_comm.comm, 0);
        }
        Parallel_Common::bcast_data((*eigenvalue_iter).data(), nband, this->diag_comm.comm, 0);
    }
#endif

    ModuleBase::timer::end("Diago_DavSubspace", "diag_zhegvx");
    return;
}

template <typename T, typename Device>
void Diago_DavSubspace<T, Device>::refresh(const int& dim,
                                           const int& nband,
                                           int& nbase,
                                           const Real* eigenvalue_in_hsolver,
                                           //    const psi::Psi<T, Device>& psi,
                                           T* psi_iter,
                                           T* hpsi,
                                           T* spsi,
                                           T* hcc,
                                           T* scc,
                                           T* vcc)
{
    ModuleBase::timer::start("Diago_DavSubspace", "refresh");
    const int matrix_ld = std::max(this->dim, 1);

#ifdef __DSP
    ModuleBase::gemm_op_mt<T, Device>()
#else
    ModuleBase::gemm_op<T, Device>()
#endif
                        ('N',
                         'N',
                         this->dim,
                         nband,
                         nbase,
                         this->one,
                         this->hpsi,
                         matrix_ld,
                         this->vcc,
                         this->nbase_x,
                         this->zero,
                         psi_iter + nband * this->dim,
                         matrix_ld);

    // update hpsi
    syncmem_complex_op()(hpsi, psi_iter + nband * this->dim, this->dim * nband);

    // psi_iter already contains the restarted Ritz vectors. For S = I,
    // spsi aliases those vectors, so rotating it again would use the new basis.
    if (!this->overlap_is_identity)
    {
#ifdef __DSP
        ModuleBase::gemm_op_mt<T, Device>()
#else
        ModuleBase::gemm_op<T, Device>()
#endif
        ('N',
         'N',
         this->dim,
         nband,
         nbase,
         this->one,
         this->spsi,
         matrix_ld,
         this->vcc,
         this->nbase_x,
         this->zero,
         psi_iter + nband * this->dim,
         matrix_ld);

        // update spsi
        syncmem_complex_op()(spsi, psi_iter + nband * this->dim, this->dim * nband);
    }

    nbase = nband;

    // set hcc/scc/vcc to 0
    setmem_complex_2d_op()(hcc, this->nbase_x, 0, nbase, nbase);
    setmem_complex_2d_op()(scc, this->nbase_x, 0, nbase, nbase);
    setmem_complex_2d_op()(vcc, this->nbase_x, 0, nbase, nbase);

    if (this->device == base_device::GpuDevice)
    {
#if defined(__CUDA) || defined(__ROCM)
        syncmem_var_h2d_op()(this->d_eigenvalue, eigenvalue_in_hsolver, nbase);
#endif
        refresh_hcc_scc_vcc_op<T, Device>()(nbase, hcc, scc, vcc, this->nbase_x, this->d_eigenvalue, this->one_);
    }
    else
    {
        for (int i = 0; i < nbase; i++)
        {
            hcc[i * this->nbase_x + i] = eigenvalue_in_hsolver[i];
            scc[i * this->nbase_x + i] = this->one[0];
            vcc[i * this->nbase_x + i] = this->one[0];
        }
    }
    ModuleBase::timer::end("Diago_DavSubspace", "refresh");

    return;
}

template <typename T, typename Device>
int Diago_DavSubspace<T, Device>::diag(const HSOperator<T, Device>& op,
                                       T* psi_in,
                                       const int psi_in_dmax,
                                       Real* eigenvalue_in_hsolver,
                                       const std::vector<double>& ethr_band,
                                       const bool& scf_type)
{
    /// record the times of trying iterative diagonalization
    this->notconv = 0;

    int sum_iter = 0;
    int ntry = 0;

    do
    {

        try
        {
            sum_iter += this->diag_once(op, psi_in, psi_in_dmax, eigenvalue_in_hsolver, ethr_band);
        }
        catch (const std::exception& error)
        {
            ModuleBase::WARNING_QUIT("Diago_DavSubspace::diag", error.what());
        }

        ++ntry;

    } while (this->test_exit_cond(ntry, this->notconv, scf_type));

    if (notconv > std::max(5, this->n_band / 4))
    {
        std::cout << "\n notconv = " << this->notconv;
        std::cout << "\n Diago_DavSubspace::diag', too many bands are not converged! \n";
    }

    return sum_iter;
}

template <typename T, typename Device>
bool Diago_DavSubspace<T, Device>::test_exit_cond(const int& ntry, const int& notconv, const bool& scf)
{
    // scf = true; // scf
    // scf = false; // nscf

    // If ntry <=5, try to do it better, if ntry > 5, exit.
    const bool f1 = (ntry <= 5);

    // In non-self consistent calculation, do until totally converged.
    const bool f2 = ((!scf && (notconv > 0)));

    // if self consistent calculation, if not converged > 5
    const bool f3 = ((scf && (notconv > 5)));

    return (f1 && (f2 || f3));
}

namespace hsolver
{

template class Diago_DavSubspace<std::complex<float>, base_device::DEVICE_CPU>;
template class Diago_DavSubspace<std::complex<double>, base_device::DEVICE_CPU>;

#if ((defined __CUDA) || (defined __ROCM))
template class Diago_DavSubspace<std::complex<float>, base_device::DEVICE_GPU>;
template class Diago_DavSubspace<std::complex<double>, base_device::DEVICE_GPU>;
#endif

#ifdef __LCAO
template class Diago_DavSubspace<double, base_device::DEVICE_CPU>;

#if ((defined __CUDA) || (defined __ROCM))
template class Diago_DavSubspace<double, base_device::DEVICE_GPU>;
#endif

#endif
} // namespace hsolver
