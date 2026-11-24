# Fe64 LCAO DeltaSpin subspace probe

This standalone CUDA experiment benchmarks two ways to perform the inner
DeltaSpin search for an eight-k-point Fe64 SOC calculation. It uses a 3456
spinor basis, 1152 bands, a 1728-dimensional Ritz space and 16 projectors per
site. Each k-point owner keeps its reference eigenvectors on its GPU. The
projected Hamiltonian and moment contractions preserve the real-space
`pre_hr` orbital-pair cutoff.

- `trial` takes its reference from the preceding exact solve and uses
  projection for the line-search trial only.
- `pw-style` also evaluates intermediate corrected lambda values in the
  reference subspace. It refreshes the full accepted state and computes the
  actual residual from the density matrix that supplies the outer charge
  update. A failed early convergence check refreshes the reference before
  continuing within the remaining lambda budget.

The probe interposes the CUDA eigensolver and assumes one geometry per
process and two k-point owners with four k-points each. The CUDA file is
built explicitly for the experiment. ABACUS's normal build and INPUT
interface do not include this probe.

## Run the comparison

In an experimental checkout based on `f6a2ebc92`, apply the hook patch and
build ABACUS with the CUDA and MPI options used for the k-owner path.

```sh
probe_dir="$PWD/tools/03_code_analysis/lcao_pw_style_probe"
git apply --check "$probe_dir/projected_loop.patch"
git apply "$probe_dir/projected_loop.patch"
nvcc -std=c++11 -O3 -arch=sm_70 --shared --cudart shared -Xcompiler=-fPIC \
  -o /path/to/lazy_probe.so "$probe_dir/lazy_probe.cu" -lcusolver -lcublas -ldl
```

Use two separate copies of the same Fe64 SOC input with identical geometry,
orbitals, pseudopotentials, charge checkpoint and initial lambda. Use
`nsc=4`, `scf_thr=1e-9`, `sc_thr=1e-7` and `scf_ene_thr=1e-9`. With the same
GPUs and MPI placement, launch ABACUS once in each case directory. Select
`trial` or `pw-style` via `ABACUS_LAZY_PROJECTOR_PROBE`.

```sh
OMP_NUM_THREADS=1 mpirun -np 8 env LD_PRELOAD=/path/to/lazy_probe.so \
  ABACUS_LAZY_PROJECTOR_PROBE=pw-style /path/to/abacus_basic_gpu
```

In `pw-style`, `PW_STYLE_REFRESH` records the predicted RMS and the actual
RMS from the accepted DMR. Check the final charge residual, actual magnetic
RMS and absolute `DeltaE_womix` against the three input thresholds before
comparing the `Driver:atomic_world` timers.

## Observed Fe64 result

The measured checkout included the GPU H(R) folding and density-gather
changes in both modes. The paired run used two V100-SXM2 GPUs, eight MPI
ranks, one OMP thread and CUDA 12.9.1. The 20.7% wall-time reduction is from
one restart pair.

| Measurement | Trial projection only | PW-style |
| --- | ---: | ---: |
| First charge and moment thresholds, seconds | 231.26 | 209.84 |
| First 32 SCF iterations, seconds | 257.13 | 218.60 |
| Wall time to all three thresholds, seconds | 277.60 | 220.18 |
| SCF iterations to all three thresholds | 37 | 32 |
| Full density-matrix builds | 79 | 48 |
| Final charge residual | 1.8696e-10 | 5.7343e-10 |
| Final magnetic RMS, μB | 7.07899e-8 | 5.64057e-8 |
| Final absolute energy change, eV | 6.68213e-10 | 7.91956e-10 |

The measured candidate reached the joint thresholds without reanchoring or
falling back. Its last full solve produced both the reported magnetic
residual and the DMR used for charge. Results cover the specified Fe64
geometry and k-point layout.
