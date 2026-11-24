#include "density_gather.h"
#include "hcontainer.h"
#include "transfer.h"
#include "source_base/parallel_common.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <utility>

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
    size_t source_size;
    HContainer<double> serial;
    const Parallel_Orbitals* orbitals;
    std::vector<int> topology;
    std::vector<int> rows;
    std::vector<int> cols;
    int size = 1;
#ifdef __MPI
    std::unique_ptr<HTransPara<double>> parallel_transfer;
    std::unique_ptr<HTransSerial<double>> serial_transfer;
    int world_rank = 0;
    int peers_per_batch = 1;
    std::vector<int> send_totals;
    std::vector<int> recv_totals;
    std::vector<int> sendcounts;
    std::vector<int> recvcounts;
    std::vector<int> senddispls;
    std::vector<int> recvdispls;

    void select_batch(int begin, int end)
    {
        std::fill(sendcounts.begin(), sendcounts.end(), 0);
        std::fill(recvcounts.begin(), recvcounts.end(), 0);
        // A rank sending forward by a given offset receives the same offset backward.
        for (int offset = begin; offset < end; ++offset)
        {
            const int dest = (world_rank + offset) % size;
            const int src = (world_rank + size - offset) % size;
            sendcounts[dest] = send_totals[dest];
            recvcounts[src] = recv_totals[src];
        }
    }
#endif
    std::vector<double> send;
    std::vector<double> recv;

    Impl(const HContainer<double>& input, const HContainer<double>& layout)
        : serial(layout)
    {
        source_size = input.get_nnr();
        orbitals = input.get_paraV();
        topology = input.get_ijr_info();
        if (orbitals)
        {
            rows = orbitals->get_indexes_row();
            cols = orbitals->get_indexes_col();
        }
    }
};

DensityGather::DensityGather(const HContainer<double>& source,
                             const HContainer<double>& serial_layout,
                             int buffer_values)
    : impl_(new Impl(source, serial_layout))
{
    ModuleBase::timer::start("Gint", "dm_gather_layout");
    auto& p = *impl_;
#ifdef __MPI
    p.size = Parallel_Common::communicator_size(MPI_COMM_WORLD);
    p.parallel_transfer.reset(new HTransPara<double>(p.size, p.orbitals));
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
    p.parallel_transfer->get_value_size(sendcounts.data());
    p.serial_transfer->get_value_size(recvcounts.data());
    int max_peer = *std::max_element(sendcounts.begin(), sendcounts.end());
    Parallel_Reduce::reduce_max(max_peer);
    const int peer_values = std::max(1, max_peer);
    p.world_rank = Parallel_Common::communicator_rank(MPI_COMM_WORLD);
    const int layout_size = p.orbitals->get_dim0() * p.orbitals->get_dim1();
    int replicated_layout = layout_size != p.size;
    Parallel_Reduce::reduce_max(replicated_layout);
    // A replicated layout can overwrite the same slots from different sources.
    // Keep its original source-rank order in one exchange.
    p.peers_per_batch = replicated_layout ? p.size : std::max(1, buffer_values / peer_values);
    p.send_totals = std::move(sendcounts);
    p.recv_totals = std::move(recvcounts);
    p.sendcounts.resize(p.size);
    p.recvcounts.resize(p.size);
    int max_send = 1;
    int max_recv = 1;
    for (int begin = 0; begin < p.size;)
    {
        const int batch_size = std::min(p.size - begin, p.peers_per_batch);
        const int end = begin + batch_size;
        p.select_batch(begin, end);
        const int send_size = offsets(p.sendcounts, p.senddispls);
        const int recv_size = offsets(p.recvcounts, p.recvdispls);
        max_send = std::max(max_send, send_size);
        max_recv = std::max(max_recv, recv_size);
        begin = end;
    }
    p.send.resize(max_send);
    p.recv.resize(max_recv);
#endif
    ModuleBase::timer::end("Gint", "dm_gather_layout");
}

DensityGather::~DensityGather() = default;

bool DensityGather::matches(const HContainer<double>& source) const
{
    const auto& p = *impl_;
    if (p.orbitals != source.get_paraV() || p.source_size != source.get_nnr()
        || p.topology != source.get_ijr_info()) return false;
    return !p.orbitals || (p.rows == p.orbitals->get_indexes_row() && p.cols == p.orbitals->get_indexes_col());
}

const HContainer<double>& DensityGather::gather(const HContainer<double>& source)
{
    auto& p = *impl_;
    ModuleBase::timer::start("Gint", "dm_gather_values");
    if (!matches(source)) throw std::invalid_argument("density gather source layout changed");
    p.serial.set_zero();
#ifdef __MPI
    for (int begin = 0; begin < p.size;)
    {
        const int batch_size = std::min(p.size - begin, p.peers_per_batch);
        const int end = begin + batch_size;
        p.select_batch(begin, end);
        offsets(p.sendcounts, p.senddispls);
        offsets(p.recvcounts, p.recvdispls);
        for (int rank = 0; rank < p.size; ++rank)
        {
            if (p.sendcounts[rank])
            {
                double* values = p.send.data() + p.senddispls[rank];
                p.parallel_transfer->pack_data(rank, source, values);
            }
        }
        Parallel_Common::alltoallv_double(p.send.data(), p.sendcounts.data(), p.senddispls.data(),
                                         p.recv.data(), p.recvcounts.data(), p.recvdispls.data(),
                                         MPI_COMM_WORLD);
        for (int rank = 0; rank < p.size; ++rank)
        {
            if (p.recvcounts[rank])
            {
                const double* values = p.recv.data() + p.recvdispls[rank];
                p.serial_transfer->receive_data(rank, values);
            }
        }
        begin = end;
    }
#else
    p.serial.add(source);
#endif
    ModuleBase::timer::end("Gint", "dm_gather_values");
    return p.serial;
}
}
