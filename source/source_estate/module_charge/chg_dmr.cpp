#include "chg_dmr.h"

#include <functional>
#include <vector>

#include "source_base/module_mixing/mixing.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_base/tool_title.h"

namespace module_charge
{
namespace
{

/**
 * @brief Two-beta plain step for the magnetic density matrix.
 *
 * The first nnr elements are the charge channel (beta), the next nnr
 * elements are the magnetization channel (beta_mag).
 *
 * @param out      mixed output, length 2 * nnr
 * @param in       mixed input, length 2 * nnr
 * @param sres     residual, length 2 * nnr
 * @param nnr      number of DMR elements per spin channel
 * @param beta     mixing beta for the charge channel
 * @param beta_mag mixing beta for the magnetization channel
 */
void twobeta_step(double* out,
                  const double* in,
                  const double* sres,
                  const int nnr,
                  const double beta,
                  const double beta_mag)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
    for (int i = 0; i < nnr; ++i)
    {
        out[i] = in[i] + beta * sres[i];
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
    for (int i = nnr; i < 2 * nnr; ++i)
    {
        out[i] = in[i] + beta_mag * sres[i];
    }
}

/**
 * @brief Validate the arguments of mix_dmr. Aborts via WARNING_QUIT on the
 *        first invalid input.
 */
void check_dmr_inputs(const std::vector<double*>& dmr_out,
                      const std::vector<const double*>& dmr_in,
                      const int nnr,
                      const Base_Mixing::Mixing* mixing,
                      const MixingConfig& cfg)
{
    if (mixing == nullptr)
    {
        ModuleBase::WARNING_QUIT("module_charge::mix_dmr", "mixing pointer is null");
    }
    // nnr is local to each MPI rank and may legitimately be zero when no atom
    // pairs survive the cutoff on that rank; only negative values are invalid.
    if (nnr < 0)
    {
        ModuleBase::WARNING_QUIT("module_charge::mix_dmr", "nnr must be >= 0");
    }
    if (cfg.nspin != 1 && cfg.nspin != 2 && cfg.nspin != 4)
    {
        ModuleBase::WARNING_QUIT("module_charge::mix_dmr", "unsupported nspin, require 1, 2 or 4");
    }
    const int nspin_need = (cfg.nspin == 2) ? 2 : 1;
    if (static_cast<int>(dmr_out.size()) < nspin_need
        || static_cast<int>(dmr_in.size()) < nspin_need)
    {
        ModuleBase::WARNING_QUIT("module_charge::mix_dmr", "not enough DMR buffers for nspin");
    }
    // Non-null buffers are required only when there are elements to mix; an
    // empty partition (nnr == 0) may legitimately carry null pointers.
    for (int is = 0; is < nspin_need; ++is)
    {
        if (nnr > 0 && (dmr_out[is] == nullptr || dmr_in[is] == nullptr))
        {
            ModuleBase::WARNING_QUIT("module_charge::mix_dmr", "DMR buffer pointer is null");
        }
    }
}

} // namespace

void init_mixing_dmr(Base_Mixing::Mixing* mixing,
                     Base_Mixing::Mixing_Data& mdata,
                     const int nnr,
                     const MixingConfig& cfg)
{
    ModuleBase::TITLE("module_charge", "init_mixing_dmr");
    ModuleBase::timer::start("module_charge", "init_mixing_dmr");
    if (mixing == nullptr)
    {
        ModuleBase::WARNING_QUIT("module_charge::init_mixing_dmr", "mixing pointer is null");
    }
    // nnr is local to each MPI rank and may legitimately be zero when no atom
    // pairs survive the cutoff on that rank; only negative values are invalid.
    if (nnr < 0)
    {
        ModuleBase::WARNING_QUIT("module_charge::init_mixing_dmr", "nnr must be >= 0");
    }

    const int dmr_nspin = (cfg.nspin == 2) ? 2 : 1;
    // DMR mixing currently supports only the real-space convergence threshold.
    if (cfg.scf_thr_type == 1)
    {
        ModuleBase::WARNING_QUIT("module_charge::init_mixing_dmr",
                                 "This Mixing of Density Matrix is not supported for PW basis yet");
    }
    else if (cfg.scf_thr_type == 2)
    {
        mixing->init_mixing_data(mdata, nnr * dmr_nspin, sizeof(double));
    }

    // Clear the history counters while keeping the allocated storage.
    mdata.reset();
    ModuleBase::timer::end("module_charge", "init_mixing_dmr");
}

void mix_dmr(const std::vector<double*>& dmr_out,
             const std::vector<const double*>& dmr_in,
             const int nnr,
             Base_Mixing::Mixing* mixing,
             Base_Mixing::Mixing_Data& mdata,
             const MixingConfig& cfg,
             const std::vector<DmrSpinBlock>& spin_blocks)
{
    ModuleBase::TITLE("module_charge", "mix_dmr");
    ModuleBase::timer::start("module_charge", "mix_dmr");
    check_dmr_inputs(dmr_out, dmr_in, nnr, mixing, cfg);

    if (cfg.nspin == 1)
    {
        mixing->push_data(mdata, dmr_in[0], dmr_out[0], nullptr, false);
        mixing->mix_data(mdata, dmr_out[0]);
    }
    else if (cfg.nspin == 4)
    {
        // Each atom-pair/R block is a row-major matrix with interleaved Pauli
        // channels, as consumed by Gint. Its row stride can vary between blocks.
        int covered = 0;
        for (const auto& block : spin_blocks)
        {
            if (block.offset != covered || block.rows <= 0 || block.cols <= 0
                || block.rows % 2 != 0 || block.cols % 2 != 0
                || block.rows > (nnr - covered) / block.cols)
            {
                ModuleBase::WARNING_QUIT("module_charge::mix_dmr", "invalid nspin=4 DMR block layout");
            }
            covered += block.rows * block.cols;
        }
        if (covered != nnr)
        {
            ModuleBase::WARNING_QUIT("module_charge::mix_dmr", "incomplete nspin=4 DMR block layout");
        }
        const double beta = cfg.mixing_beta;
        const double beta_mag = cfg.mixing_beta_mag;
        std::function<void(double*, const double*, const double*)> spin_step
            = [&spin_blocks, beta, beta_mag](double* out, const double* in, const double* residual) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                  for (int ib = 0; ib < static_cast<int>(spin_blocks.size()); ++ib)
                  {
                      const auto& block = spin_blocks[ib];
                      for (int row = 0; row < block.rows; ++row)
                      {
                          for (int col = 0; col < block.cols; ++col)
                          {
                              const int index = block.offset + row * block.cols + col;
                              const bool charge = row % 2 == 0 && col % 2 == 0;
                              const double channel_beta = charge ? beta : beta_mag;
                              out[index] = in[index] + channel_beta * residual[index];
                          }
                      }
                  }
              };
        // Reuse the rho mixing coefficients/history convention; only change
        // the plain step's magnetic beta. DMR has no Kerker preconditioner.
        mixing->push_data(mdata, dmr_in[0], dmr_out[0], nullptr, spin_step, false);
        mixing->mix_data(mdata, dmr_out[0]);
    }
    else // cfg.nspin == 2
    {
        // Magnetic density matrix: up/down channels are transformed into
        // charge/magnetization channels before mixing and back afterwards.
        std::vector<double> dmr_mag(nnr * cfg.nspin, 0.0);
        std::vector<double> dmr_mag_save(nnr * cfg.nspin, 0.0);

        // Transfer the current DMR into the charge/magnetization layout.
        for (int ir = 0; ir < nnr; ++ir)
        {
            dmr_mag[ir] = dmr_out[0][ir] + dmr_out[1][ir];
            dmr_mag[ir + nnr] = dmr_out[0][ir] - dmr_out[1][ir];
        }
        // Transfer the saved DMR into the charge/magnetization layout.
        for (int ir = 0; ir < nnr; ++ir)
        {
            dmr_mag_save[ir] = dmr_in[0][ir] + dmr_in[1][ir];
            dmr_mag_save[ir + nnr] = dmr_in[0][ir] - dmr_in[1][ir];
        }

        const double beta = cfg.mixing_beta;
        const double beta_mag = cfg.mixing_beta_mag;
        std::function<void(double*, const double*, const double*)> twobeta
            = [nnr, beta, beta_mag](double* out, const double* in, const double* sres) {
                  twobeta_step(out, in, sres, nnr, beta, beta_mag);
              };
        // No Kerker screening in DMR mixing.
        mixing->push_data(mdata, dmr_mag_save.data(), dmr_mag.data(), nullptr, twobeta, false);
        mixing->mix_data(mdata, dmr_mag.data());

        // Transform the mixed charge/magnetization channels back to up/down.
        for (int ir = 0; ir < nnr; ++ir)
        {
            dmr_out[0][ir] = 0.5 * (dmr_mag[ir] + dmr_mag[ir + nnr]);
            dmr_out[1][ir] = 0.5 * (dmr_mag[ir] - dmr_mag[ir + nnr]);
        }
    }
    ModuleBase::timer::end("module_charge", "mix_dmr");
}

} // namespace module_charge
