#ifndef PARALLEL_COMMON_H
#define PARALLEL_COMMON_H

#ifdef __MPI
#include "mpi.h"
#endif

#include <complex>
#include <string>

namespace Parallel_Common
{

//(1) bcast array
void bcast_complex_double(std::complex<double>* object, const int n);
void bcast_string(std::string* object, const int n);
void bcast_double(double* object, const int n);
void bcast_int(int* object, const int n);
void bcast_char(char* object, const int n);

//(2) bcast single
void bcast_complex_double(std::complex<double>& object);
void bcast_string(std::string& object);
void bcast_double(double& object);
void bcast_int(int& object);
void bcast_bool(bool& object);

#ifdef __MPI
int communicator_size(MPI_Comm comm);

// Variable-size integer metadata exchange used by distributed sparse layouts.
void allgather_int(const int* sendbuf, int sendcount, int* recvbuf, MPI_Comm comm);
void allgatherv_int(const int* sendbuf,
                    int sendcount,
                    int* recvbuf,
                    const int* recvcounts,
                    const int* displs,
                    MPI_Comm comm);
// Sum rank-concatenated sparse contributions and leave each rank only its
// locally owned segment.
void reduce_scatter_double(const double* sendbuf,
                           double* recvbuf,
                           const int* recvcounts,
                           MPI_Comm comm);
#endif

} // namespace Parallel_Common

#endif
