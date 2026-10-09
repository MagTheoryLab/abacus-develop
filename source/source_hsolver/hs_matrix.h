#ifndef HSOLVER_HS_MATRIX_H
#define HSOLVER_HS_MATRIX_H

#include "source_base/matrix_block.h"

namespace hsolver
{

/**
 * @brief What a direct (dense) eigensolver needs from the generalized
 *        eigenproblem H x = e S x: the matrices H(k) and S(k) themselves.
 *
 * The code that owns the Hamiltonian implements this interface
 * (hamilt::HamiltHSMatrix in source_hamilt/hamilt_hs_adapter.h); HSolverLCAO
 * and Parallel_K2D only ever ask it for the two matrix views.
 */
template <typename T>
class HSMatrix
{
  public:
    virtual ~HSMatrix() = default;

    /// H(k) and S(k) of k point ik as non-owning views. The memory they point
    /// to belongs to the implementer and stays valid until the next call.
    virtual void hs_at_k(const int ik, ModuleBase::MatrixBlock<T>& hk, ModuleBase::MatrixBlock<T>& sk) = 0;

    /// Optional device views with the same ownership and lifetime as hs_at_k().
    /// Return false when this matrix source only supports host storage.
    virtual bool hs_at_k_device(const int ik, T*& hk, T*& sk)
    {
        return false;
    }
};

} // namespace hsolver

#endif // HSOLVER_HS_MATRIX_H
