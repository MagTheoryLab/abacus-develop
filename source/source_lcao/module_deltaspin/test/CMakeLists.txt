abacus_disable_feature_definitions(__CUDA)

if(ENABLE_LCAO)

if(USE_CUDA AND ENABLE_MPI)
  AddTest(
    TARGET MODULE_LCAO_deltaspin_lcao_ritz_gpu
    LIBS parameter base device CUDA::cudart CUDA::cublas CUDA::cusolver
    KEEP_FEATURE_DEFINITIONS __CUDA __MPI
    SOURCES test_lcao_ritz_gpu.cpp ../lcao_ritz_gpu.cu
  )
  set_tests_properties(MODULE_LCAO_deltaspin_lcao_ritz_gpu PROPERTIES TIMEOUT 60)
endif()

AddTest(
  TARGET MODULE_LCAO_deltaspin_basic_func_test
  LIBS base device parameter
  SOURCES basic_test.cpp
    ../basic_funcs.cpp
)

AddTest(
  TARGET MODULE_LCAO_deltaspin_spin_constrain_test
  LIBS base device parameter symmetry
  SOURCES spin_constrain_test.cpp
    ../spin_constrain.cpp
    ../deltaspin_state.cpp
    ../../../source_basis/module_ao/parallel_orbitals.cpp
    ../../../source_cell/klist.cpp ../../../source_cell/klist_io.cpp
    ../../../source_cell/parallel_kpoints.cpp
    ../../../source_cell/reciprocal_grid.cpp
)

AddTest(
  TARGET MODULE_LCAO_deltaspin_template_helpers
  LIBS base device parameter symmetry
  SOURCES template_helpers_test.cpp
    ../spin_constrain.cpp
    ../deltaspin_state.cpp
    ../template_helpers.cpp
    ../lambda_loop_helper.cpp
    ../basic_funcs.cpp
    ../../../source_cell/klist.cpp ../../../source_cell/klist_io.cpp
    ../../../source_cell/parallel_kpoints.cpp
    ../../../source_cell/reciprocal_grid.cpp
)

AddTest(
  TARGET deltaspin_pw_test
  LIBS base device parameter
  SOURCES deltaspin_pw_test.cpp
)

AddTest(
  TARGET deltaspin_core_test
  LIBS base device
  SOURCES deltaspin_core_test.cpp
    ../mi_tools.cpp
    ../../../source_estate/occ_comput.cpp
)

AddTest(
  TARGET test_lambda_loop_helper
  LIBS base device parameter symmetry
  SOURCES test_lambda_loop_helper.cpp
    ../spin_constrain.cpp
    ../deltaspin_state.cpp
    ../lambda_loop_helper.cpp
    ../basic_funcs.cpp
    ../../../source_cell/klist.cpp ../../../source_cell/klist_io.cpp
    ../../../source_cell/parallel_kpoints.cpp
    ../../../source_cell/reciprocal_grid.cpp
)

AddTest(
  TARGET MODULE_LCAO_deltaspin_state_test
  LIBS base device
  SOURCES test_deltaspin_state.cpp
    ../deltaspin_state.cpp
)

endif()
