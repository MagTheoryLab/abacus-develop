#include "source_lcao/hamilt_lcao.h"
#include "source_lcao/hamilt_lcao_factory.h"
#include "source_base/memory_recorder.h"
#include "source_base/timer.h"
#include "source_pw/module_pwdft/dftu_base.h"
#include "source_lcao/setup_exx.h"
#include "source_lcao/setup_deepks.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_estate/module_pot/potential_new.h"
#include "source_hamilt/module_hcontainer/hcontainer_funcs.h"
#include <vector>
#ifdef __CUDA
#include "source_hamilt/module_hcontainer/folding_hr_gpu.h"
#endif

#ifdef __MLALGO
#include "source_lcao/module_deepks/lcao_deepks.h"
#endif
#ifdef __EXX
#include "source_lcao/module_ri/exx_lri_interface.h"
#include "module_operator_lcao/op_exx_lcao.h"
#endif

#include "module_operator_lcao/operator_lcao.h"
#include "module_operator_lcao/overlap.h"


namespace hamilt
{

template <typename TK, typename TR>
HamiltLCAO<TK, TR>::~HamiltLCAO()
{
    delete this->ops;
}

template <typename TK, typename TR>
bool HamiltLCAO<TK, TR>::updateHk_device(int ik, TK*& h, TK*& s)
{
    return false;
}

#ifdef __CUDA
template <>
bool HamiltLCAO<std::complex<double>, std::complex<double>>::updateHk_device(
    int ik, std::complex<double>*& h, std::complex<double>*& s)
{
    // These operators also contribute directly in k space. Keep their existing
    // folding/gauge workflow intact, before mutating any real-space state.
    for (auto* op = this->ops; op != nullptr; op = op->next_op)
    {
        if (op->get_cal_type() == calculation_type::lcao_deepks
            || op->get_cal_type() == calculation_type::lcao_tddft_periodic)
        {
            return false;
        }
        const auto* lcao_op
            = dynamic_cast<const OperatorLCAO<std::complex<double>, std::complex<double>>*>(op);
        if (lcao_op != nullptr
            && (op->get_cal_type() == calculation_type::lcao_gint
                || op->get_cal_type() == calculation_type::lcao_dftu)
            && !lcao_op->supportsDeviceHR())
        {
            return false;
        }
        if (op->get_cal_type() == calculation_type::lcao_sc_lambda
            || op->get_cal_type() == calculation_type::lcao_exx)
        {
            return false;
        }
    }
    this->updateHk_impl(ik, false);
    if (!this->h_fold_)
    {
        this->h_fold_.reset(new FoldingHrGpu(*this->hR, this->kv->kvec_d));
        this->h_fold_->upload(*this->hR);
    }
    if (!this->s_fold_)
    {
        this->s_fold_.reset(new FoldingHrGpu(*this->sR, this->kv->kvec_d));
        this->s_fold_->upload(*this->sR);
    }
    const std::complex<double>* addends[2] = {nullptr, nullptr};
    int addend_count = 0;
    for (auto* op = this->ops; op != nullptr; op = op->next_op)
    {
        const auto* lcao_op
            = dynamic_cast<const OperatorLCAO<std::complex<double>, std::complex<double>>*>(op);
        if (lcao_op != nullptr && lcao_op->deviceHRContribution() != nullptr)
        {
            if (addend_count == 2)
            {
                ModuleBase::WARNING_QUIT("HamiltLCAO::updateHk_device",
                                         "more than two device H(R) addends are not supported");
            }
            addends[addend_count++] = lcao_op->deviceHRContribution();
        }
    }
    if (addend_count == 0) h = this->h_fold_->fold(ik);
    else if (addend_count == 1)
    {
        h = this->h_fold_->fold_with_device_addend(ik, addends[0], this->hR->get_nnr());
    }
    else
    {
        h = this->h_fold_->fold_with_device_addends(ik, addends[0], addends[1], this->hR->get_nnr());
    }
    s = this->s_fold_->fold(ik);
    return true;
}
#endif

template <typename TK, typename TR>
HamiltLCAO<TK, TR>::HamiltLCAO(const UnitCell& ucell,
                               const Grid_Driver& grid_d,
                               const Parallel_Orbitals* paraV,
                               const K_Vectors& kv_in,
                               const TwoCenterIntegrator& intor_overlap_orb,
                               const std::vector<double>& orb_cutoff)
{
    this->classname = "HamiltLCAO";

    this->kv = &kv_in;

    // initialize the overlap matrix;
    // hR is also allocated so the Overlap node never receives a null hR_in
    // (hsk stays null: this vacuum path has no k-space matrix and never calls init())
    this->hR.reset(new HContainer<TR>(paraV));
    this->sR.reset(new HContainer<TR>(paraV));

    this->getOperator() = new Overlap<OperatorLCAO<TK, TR>>(this->hsk.get(),
                                                               this->kv->kvec_d, this->hR.get(), this->sR.get(),
                                                               &ucell, orb_cutoff, &grid_d,
                                                               &intor_overlap_orb);
}

template <typename TK, typename TR>
HamiltLCAO<TK, TR>::HamiltLCAO(const UnitCell& ucell,
                               const Grid_Driver& grid_d,
                               const Parallel_Orbitals* paraV,
                               elecstate::Potential* pot_in,
                               const K_Vectors& kv_in,
                               const TwoCenterBundle& two_center_bundle,
                               const LCAO_Orbitals& orb,
                               module_dm::DensityMatrix<TK, double>* DM_in,
                               Plus_U_Base* p_dftu, // mohan add 2025-11-05
                               Setup_DeePKS<TK> &deepks,
                               const int istep,
                               Exx_NAO<TK> &exx_nao,
                               const Exx_Info& exx_info,
                               const Input_para& inp,
                               const bool load_exx_flag)
{
    this->classname = "HamiltLCAO";

    this->kv = &kv_in;

    // snapshot INPUT flags used later by getHR_vector/updateHk/refresh,
    // so those methods do not read global PARAM
    this->nspin = inp.nspin;
    this->vl_in_h = inp.vl_in_h;

    // Real space Hamiltonian is inited with template TR
    this->hR.reset(new HContainer<TR>(paraV));
    this->sR.reset(new HContainer<TR>(paraV));
    this->hsk.reset(new HS_Matrix_K<TK>(paraV));

    // Effective potential term (\sum_r <psi(r)|Veff(r)|psi(r)>) is registered without template
    std::vector<std::string> pot_register_in;
    if (inp.vl_in_h)
    {
        if (inp.vion_in_h) { pot_register_in.push_back("local"); }
        if (inp.vh_in_h)   { pot_register_in.push_back("hartree"); }
        pot_register_in.push_back("xc");
        if (inp.imp_sol)     { pot_register_in.push_back("surchem"); }
        if (inp.efield_flag) { pot_register_in.push_back("efield"); }
        if (inp.gate_flag)   { pot_register_in.push_back("gatefield"); }
        if (inp.esolver_type == "tddft") { pot_register_in.push_back("tddft"); }
        if (inp.ml_exx) { pot_register_in.push_back("ml_exx"); } // sunliang
    }

    // Gamma_only case to initialize HamiltLCAO
    LcaoOpsBundle<TK, TR> bundle;
    if (std::is_same<TK, double>::value)
    {
        bundle = build_gamma_ops<TK, TR>(ucell, grid_d, paraV, pot_in, two_center_bundle,
                                         orb, DM_in, p_dftu, deepks, inp, pot_register_in,
                                         this->kv, this->hsk.get(), this->hR.get(), this->sR.get());
    }
    // multi-k-points case to initialize HamiltLCAO, ops will be used
    else if (std::is_same<TK, std::complex<double>>::value)
    {
        bundle = build_multik_ops<TK, TR>(ucell, grid_d, paraV, pot_in, two_center_bundle,
                                          orb, DM_in, p_dftu, deepks, inp, pot_register_in,
                                          this->kv, this->hsk.get(), this->hR.get(), this->sR.get());
    }
    this->getOperator() = bundle.ops;
#ifdef __MLALGO
    this->V_delta_R = bundle.v_delta_R;
#endif

#ifdef __EXX
    if (exx_info.info_global.cal_exx)
    {
        // Peize Lin add 2016-12-03
        // set xc type before the first cal of xc in pelec->init_scf
        // and calculate Cs, Vs
        // Keep exact exchange in H(R) for every workflow. For RT-TDDFT the
        // factory selects complex H(R) when EXX is active, so the operator
        // chain folds the complete Hamiltonian with one common TD phase.
        Operator<TK>* exx = new OperatorEXX<OperatorLCAO<TK, TR>>(this->hsk.get(),
                                                                  this->hR.get(), ucell, *this->kv,
                                                                  exx_nao.exd.get(), exx_nao.exc.get(),
                                                                  exx_info, Add_Hexx_Type::R, istep,
                                                                  load_exx_flag);
        this->getOperator()->add(exx);
    }
#endif

    // if NSPIN==2, HR should be separated into two parts, save HR into this->hRS2
    int memory_fold = 1;
    if (this->nspin == 2)
    {
        this->hRS2.resize(this->hR->get_nnr() * 2);
        this->hR->allocate(this->hRS2.data(), 0);
        memory_fold = 2;
    }

    ModuleBase::Memory::record("HamiltLCAO::hR", this->hR->get_memory_size() * memory_fold);
    ModuleBase::Memory::record("HamiltLCAO::sR", this->sR->get_memory_size());
}

template <typename TK, typename TR>
std::vector<HContainer<TR>*> HamiltLCAO<TK, TR>::getHR_vector()
{
    if (this->nspin == 2)
    {
        const int nnr = this->hRS2.size() / 2;
        this->hr_spin_up_.reset(new HContainer<TR>(*this->hR, this->hRS2.data()));
        this->hr_spin_dn_.reset(new HContainer<TR>(*this->hR, this->hRS2.data() + nnr));
        return {this->hr_spin_up_.get(), this->hr_spin_dn_.get()};
    }
    else
    {
        return {this->hR.get()};
    }
}

template <typename TK, typename TR>
OperatorLCAO<TK, TR>* HamiltLCAO<TK, TR>::getOperatorLCAO()
{
    if (this->ops_lcao_ == nullptr)
    {
        this->ops_lcao_ = dynamic_cast<OperatorLCAO<TK, TR>*>(this->ops);
    }
    return this->ops_lcao_;
}

// case for multi-k-points
template <typename TK, typename TR>
void HamiltLCAO<TK, TR>::matrix(ModuleBase::MatrixBlock<TK>& hk_in, ModuleBase::MatrixBlock<TK>& sk_in)
{
    OperatorLCAO<TK, TR>* const op = this->getOperatorLCAO();
    assert(op != nullptr);
    op->matrixHk(hk_in, sk_in);
}

template <typename TK, typename TR>
void HamiltLCAO<TK, TR>::updateHk(const int ik)
{
    this->updateHk_impl(ik, true);
}

template <typename TK, typename TR>
void HamiltLCAO<TK, TR>::updateHk_impl(const int ik, const bool fold_k)
{
    ModuleBase::TITLE("HamiltLCAO", "updateHk");
    ModuleBase::timer::start("HamiltLCAO", "updateHk");

    // update global spin index
    if (this->nspin == 2)
    {
        // if Veff is added and current_spin is changed, refresh HR
        if (this->vl_in_h && this->kv->isk[ik] != this->current_spin)
        {
            // change data pointer of HR
            this->hR->allocate(this->hRS2.data() + this->hRS2.size() / 2 * this->kv->isk[ik], 0);
            if (this->refresh_times > 0)
            {
                this->refresh_times--;
                this->getOperatorLCAO()->set_hr_done(false);
            }
        }
        this->current_spin = this->kv->isk[ik];
        this->getOperatorLCAO()->set_current_spin(this->kv->isk[ik]);
    }
    this->getOperatorLCAO()->init(ik, fold_k);
    ModuleBase::timer::end("HamiltLCAO", "updateHk");
}

template <typename TK, typename TR>
void HamiltLCAO<TK, TR>::refresh(bool yes)
{
    ModuleBase::TITLE("HamiltLCAO", "refresh");
    if(yes)
    {
        this->getOperatorLCAO()->set_hr_done(false);
        if (this->nspin == 2)
        {
            this->refresh_times = 1;
            this->current_spin = 0;
            if (this->hR->get_nnr() != this->hRS2.size() / 2)
            {
                // operator has changed, resize hRS2
                this->hRS2.resize(this->hR->get_nnr() * 2);
            }
            this->hR->allocate(this->hRS2.data(), 0);
        }
    }
    else {
        this->getOperatorLCAO()->set_hr_done(true);
        this->refresh_times = 0;
        if (this->nspin == 2)
        {
            // HR has been loaded from file into both halves of hRS2.
            // Reset to spin-up; updateHk will switch pointers as needed.
            this->current_spin = 0;
            this->hR->allocate(this->hRS2.data(), 0);
        }
    }
}

// get Operator base class pointer
template <typename TK, typename TR>
Operator<TK>*& HamiltLCAO<TK, TR>::getOperator()
{
    return this->ops;
}

template <typename TK, typename TR>
void HamiltLCAO<TK, TR>::updateSk(
    const int ik,
    const int hk_type)
{
    ModuleBase::TITLE("HamiltLCAO", "updateSk");
    ModuleBase::timer::start("HamiltLCAO", "updateSk");

    ModuleBase::GlobalFunc::ZEROS(this->getSk(), this->hsk->get_size());

    if (hk_type == 1) // collumn-major matrix for SK
    {
        const int nrow = this->hsk->get_pv()->get_row_size();
        hamilt::folding_HR(*this->sR, this->getSk(), this->kv->kvec_d[ik], nrow, 1);
    }
    else if (hk_type == 0) // row-major matrix for SK
    {
        const int ncol = this->hsk->get_pv()->get_col_size();
        hamilt::folding_HR(*this->sR, this->getSk(), this->kv->kvec_d[ik], ncol, 0);
    }
    else
    {
        ModuleBase::WARNING_QUIT("updateSk", "the value of hk_type is incorrect.");
    }

    ModuleBase::timer::end("HamiltLCAO", "updateSk");
}

// case for nspin<4, gamma-only k-point
template class HamiltLCAO<double, double>;
// case for nspin<4, multi-k-points
template class HamiltLCAO<std::complex<double>, double>;
// case for nspin == 4, non-collinear spin case
template class HamiltLCAO<std::complex<double>, std::complex<double>>;
} // namespace hamilt
