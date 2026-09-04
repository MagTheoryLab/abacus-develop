abacus_disable_feature_definitions(__MLALGO)
abacus_disable_feature_definitions(__CUDA)
abacus_disable_feature_definitions(__ROCM)

if(USE_CUDA AND ENABLE_MPI)
AddTest(
  TARGET MODULE_ESTATE_k_owner_dmr_gpu
  LIBS parameter psi base device
  KEEP_FEATURE_DEFINITIONS __CUDA __MPI
  SOURCES test_cal_dmr_psi_gpu_k_owner.cpp ../cal_dmr_psi_gpu_k_owner.cu
    ../density_matrix.cpp ../density_matrix_io.cpp tmp_mocks.cpp
    ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/base_matrix.cpp
    ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/hcontainer.cpp
    ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/atom_pair.cpp
    ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp
)
set_tests_properties(MODULE_ESTATE_k_owner_dmr_gpu PROPERTIES TIMEOUT 60)
endif()

install(DIRECTORY support DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if(TARGET MODULE_ESTATE_dm_io_test_serial)
  abacus_disable_feature_definitions(__MPI)
endif()

AddTest(
  TARGET MODULE_ESTATE_dm_io_test_serial
  LIBS parameter base device cell_info symmetry
  SOURCES test_dm_io.cpp ../density_matrix.cpp ../density_matrix_io.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/base_matrix.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/hcontainer.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/atom_pair.cpp
  ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp
  ${ABACUS_SOURCE_DIR}/source_cell/klist.cpp ${ABACUS_SOURCE_DIR}/source_cell/klist_io.cpp
  ${ABACUS_SOURCE_DIR}/source_cell/parallel_kpoints.cpp
  ${ABACUS_SOURCE_DIR}/source_cell/reciprocal_grid.cpp

)

AddTest(
  TARGET MODULE_ESTATE_dm_constructor_test
  LIBS parameter  base device symmetry
  SOURCES test_dm_constructor.cpp ../density_matrix.cpp ../density_matrix_io.cpp tmp_mocks.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/base_matrix.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/hcontainer.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/atom_pair.cpp
  ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp
  ${ABACUS_SOURCE_DIR}/source_cell/klist.cpp ${ABACUS_SOURCE_DIR}/source_cell/klist_io.cpp
  ${ABACUS_SOURCE_DIR}/source_cell/parallel_kpoints.cpp
  ${ABACUS_SOURCE_DIR}/source_cell/reciprocal_grid.cpp
)

AddTest(
  TARGET MODULE_ESTATE_dm_init_test
  LIBS parameter  base device symmetry
  SOURCES test_dm_r_init.cpp ../density_matrix.cpp ../density_matrix_io.cpp tmp_mocks.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/base_matrix.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/hcontainer.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/atom_pair.cpp
  ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp
  ${ABACUS_SOURCE_DIR}/source_cell/klist.cpp ${ABACUS_SOURCE_DIR}/source_cell/klist_io.cpp
  ${ABACUS_SOURCE_DIR}/source_cell/parallel_kpoints.cpp
  ${ABACUS_SOURCE_DIR}/source_cell/reciprocal_grid.cpp
)

AddTest(
  TARGET MODULE_ESTATE_dm_cal_DMR_test
  LIBS parameter  base device symmetry
  SOURCES test_cal_dm_r.cpp ../density_matrix.cpp ../density_matrix_io.cpp tmp_mocks.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/base_matrix.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/hcontainer.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/atom_pair.cpp
  ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp
  ${ABACUS_SOURCE_DIR}/source_cell/klist.cpp ${ABACUS_SOURCE_DIR}/source_cell/klist_io.cpp
  ${ABACUS_SOURCE_DIR}/source_cell/parallel_kpoints.cpp
  ${ABACUS_SOURCE_DIR}/source_cell/reciprocal_grid.cpp
)

AddTest(
  TARGET MODULE_ESTATE_dm_soc_magnetization_roundtrip_test
  LIBS parameter  base device
  SOURCES test_soc_magnetization_roundtrip.cpp ../density_matrix.cpp ../density_matrix_io.cpp tmp_mocks.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/base_matrix.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/hcontainer.cpp
  ${ABACUS_SOURCE_DIR}/source_hamilt/module_hcontainer/atom_pair.cpp
  ${ABACUS_SOURCE_DIR}/source_basis/module_ao/parallel_orbitals.cpp
)
