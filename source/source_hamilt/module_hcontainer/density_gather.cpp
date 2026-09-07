#include "density_gather.h"
#include "hcontainer.h"
#include "transfer.h"
#include "source_base/parallel_common.h"
#include "source_base/timer.h"
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace hamilt
{
namespace
{
int checked_size(size_t size)
{
    if (size > INT_MAX) throw std::overflow_error("density gather MPI buffer exceeds INT_MAX");
    return static_cast<int>(size);
}

int offsets(const std::vector<int>& counts, std::vector<int>& displs)
{
    displs.resize(counts.size());
    size_t total = 0;
    for (size_t i = 0; i < counts.size(); ++i)
    {
        displs[i] = checked_size(total);
        total += counts[i];
    }
    return checked_size(total);
}
}

struct DensityGather::Impl
{
    HContainer<double> source;
    HContainer<double> serial;
    const Parallel_Orbitals* orbitals;
    std::vector<int> topology;
    std::vector<int> rows;
    std::vector<int> cols;
    int size = 1;
#ifdef __MPI
    std::unique_ptr<HTransPara<double>> parallel_transfer;
    std::unique_ptr<HTransSerial<double>> serial_transfer;
#endif
    std::vector<int> sendcounts;
    std::vector<int> recvcounts;
    std::vector<int> senddispls;
    std::vector<int> recvdispls;
    std::vector<double> send;
    std::vector<double> recv;

    Impl(const HContainer<double>& input, const HContainer<double>& layout)
        : source(input), serial(layout), orbitals(input.get_paraV()), topology(input.get_ijr_info())
    {
        if (orbitals)
        {
            rows = orbitals->get_indexes_row();
            cols = orbitals->get_indexes_col();
        }
    }
};

DensityGather::DensityGather(const HContainer<double>& source, const HContainer<double>& serial_layout)
    : impl_(new Impl(source, serial_layout))
{
    ModuleBase::timer::start("Gint", "dm_gather_layout");
    auto& p = *impl_;
#ifdef __MPI
    p.size = Parallel_Common::communicator_size(MPI_COMM_WORLD);
    p.parallel_transfer.reset(new HTransPara<double>(p.size, &p.source));
    p.serial_transfer.reset(new HTransSerial<double>(p.size, &p.serial));
    std::vector<int> sendcounts(p.size);
    std::vector<int> recvcounts(p.size);
    std::vector<int> senddispls;
    std::vector<int> recvdispls;
    std::vector<int> send;
    std::vector<int> recv;
    // Reuse the canonical HContainer transfer's atom/orbital index protocol.
    for (int phase = 0; phase < 2; ++phase)
    {
        send.clear();
        for (int rank = 0; rank < p.size; ++rank)
        {
            std::vector<int> indices;
            if (phase == 0) p.serial_transfer->cal_ap_indexes(rank, &indices);
            else p.parallel_transfer->cal_orb_indexes(rank, &indices);
            sendcounts[rank] = checked_size(indices.size());
            send.insert(send.end(), indices.begin(), indices.end());
        }
        offsets(sendcounts, senddispls);
        Parallel_Common::alltoall_int(sendcounts.data(), recvcounts.data(), MPI_COMM_WORLD);
        recv.resize(std::max(1, offsets(recvcounts, recvdispls)));
        Parallel_Common::alltoallv_int(send.data(), sendcounts.data(), senddispls.data(),
                                      recv.data(), recvcounts.data(), recvdispls.data(), MPI_COMM_WORLD);
        for (int rank = 0; rank < p.size; ++rank)
        {
            const int* indices = recv.data() + recvdispls[rank];
            if (phase == 0) p.parallel_transfer->receive_ap_indexes(rank, indices, recvcounts[rank]);
            else p.serial_transfer->receive_orb_indexes(rank, indices, recvcounts[rank]);
        }
    }
    p.sendcounts.resize(p.size);
    p.recvcounts.resize(p.size);
    p.parallel_transfer->get_value_size(p.sendcounts.data());
    p.serial_transfer->get_value_size(p.recvcounts.data());
    p.send.resize(std::max(1, offsets(p.sendcounts, p.senddispls)));
    p.recv.resize(std::max(1, offsets(p.recvcounts, p.recvdispls)));
#endif
    ModuleBase::timer::end("Gint", "dm_gather_layout");
}

DensityGather::~DensityGather() = default;

bool DensityGather::matches(const HContainer<double>& source) const
{
    const auto& p = *impl_;
    if (p.orbitals != source.get_paraV() || p.source.get_nnr() != source.get_nnr()
        || p.topology != source.get_ijr_info()) return false;
    return !p.orbitals || (p.rows == p.orbitals->get_indexes_row() && p.cols == p.orbitals->get_indexes_col());
}

const HContainer<double>& DensityGather::gather(const HContainer<double>& source)
{
    auto& p = *impl_;
    ModuleBase::timer::start("Gint", "dm_gather_values");
    if (!matches(source)) throw std::invalid_argument("density gather source layout changed");
    if (source.get_nnr()) std::copy_n(source.get_wrapper(), source.get_nnr(), p.source.get_wrapper());
    p.serial.set_zero();
#ifdef __MPI
    for (int rank = 0; rank < p.size; ++rank)
        if (p.sendcounts[rank]) p.parallel_transfer->pack_data(rank, p.send.data() + p.senddispls[rank]);
    Parallel_Common::alltoallv_double(p.send.data(), p.sendcounts.data(), p.senddispls.data(),
                                   p.recv.data(), p.recvcounts.data(), p.recvdispls.data(), MPI_COMM_WORLD);
    for (int rank = 0; rank < p.size; ++rank)
        if (p.recvcounts[rank]) p.serial_transfer->receive_data(rank, p.recv.data() + p.recvdispls[rank]);
#else
    p.serial.add(p.source);
#endif
    ModuleBase::timer::end("Gint", "dm_gather_values");
    return p.serial;
}
}
