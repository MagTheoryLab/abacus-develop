
if(ENABLE_LCAO)
AddTest(
  TARGET MODULE_LCAO_density_gather_test
  LIBS parameter psi base device
  KEEP_FEATURE_DEFINITIONS __MPI
  SOURCES test_density_gather.cpp ../density_gather.cpp ../transfer.cpp ../func_transfer.cpp
    ../base_matrix.cpp ../hcontainer.cpp ../atom_pair.cpp
    ../../../source_basis/module_ao/parallel_orbitals.cpp tmp_mocks.cpp
)
if(USE_CUDA AND ENABLE_MPI)
AddTest(
  TARGET MODULE_LCAO_folding_hr_gpu_test
  LIBS parameter psi base device CUDA::cudart
  KEEP_FEATURE_DEFINITIONS __CUDA __MPI
  SOURCES test_folding_hr_gpu.cpp ../folding_hr_gpu.cu ../func_folding.cpp
    ../base_matrix.cpp ../hcontainer.cpp ../atom_pair.cpp
    ../../../source_basis/module_ao/parallel_orbitals.cpp tmp_mocks.cpp
)
AddTest(
  TARGET MODULE_LCAO_spinor_hr_gpu_test
  LIBS parameter psi base device CUDA::cudart
  KEEP_FEATURE_DEFINITIONS __CUDA __MPI
  SOURCES test_spinor_hr_gpu.cpp ../../module_gint/kernel/spinor_hr_gpu.cu
    ../folding_hr_gpu.cu ../func_folding.cpp
    ../base_matrix.cpp ../hcontainer.cpp ../atom_pair.cpp
    ../../../source_basis/module_ao/parallel_orbitals.cpp tmp_mocks.cpp
)
endif()
AddTest(
  TARGET MODULE_LCAO_hcontainer_test
  LIBS parameter psi base device
  SOURCES test_hcontainer.cpp ../base_matrix.cpp ../hcontainer.cpp ../atom_pair.cpp  
  ../../../source_basis/module_ao/parallel_orbitals.cpp tmp_mocks.cpp
)

AddTest(
  TARGET MODULE_LCAO_hcontainer_complex_test
  LIBS parameter psi base device
  SOURCES test_hcontainer_complex.cpp ../base_matrix.cpp ../hcontainer.cpp ../atom_pair.cpp  
  ../../../source_basis/module_ao/parallel_orbitals.cpp tmp_mocks.cpp
)

AddTest(
  TARGET MODULE_LCAO_hcontainer_cost_test
  LIBS parameter psi base device
  SOURCES test_hcontainer_time.cpp ../base_matrix.cpp ../hcontainer.cpp ../atom_pair.cpp  
  ../../../source_basis/module_ao/parallel_orbitals.cpp tmp_mocks.cpp
)

AddTest(
  TARGET MODULE_LCAO_hcontainer_folding_test
  LIBS parameter psi base device
  SOURCES test_func_folding.cpp ../base_matrix.cpp ../hcontainer.cpp ../atom_pair.cpp  
  ../func_folding.cpp ../../../source_basis/module_ao/parallel_orbitals.cpp tmp_mocks.cpp
)

AddTest(
  TARGET MODULE_LCAO_hcontainer_transfer_test
  LIBS parameter psi base device
  SOURCES test_transfer.cpp ../func_transfer.cpp ../base_matrix.cpp ../hcontainer.cpp ../atom_pair.cpp  
  ../transfer.cpp ../../../source_basis/module_ao/parallel_orbitals.cpp tmp_mocks.cpp
)

install(FILES parallel_hcontainer_tests.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
find_program(BASH bash)
add_test(NAME MODULE_LCAO_hcontainer_para_test
      COMMAND ${BASH} parallel_hcontainer_tests.sh
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

AddTest(
  TARGET MODULE_LCAO_hcontainer_output_test
  LIBS parameter base device
  SOURCES test_hcontainer_output.cpp
    tmp_mocks.cpp
    ../output_hcontainer.cpp
    ../base_matrix.cpp
    ../hcontainer.cpp
    ../atom_pair.cpp
    ../../../source_basis/module_ao/parallel_orbitals.cpp
)

install(DIRECTORY support DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

AddTest(
  TARGET MODULE_LCAO_hcontainer_add_value_test
  LIBS parameter ${math_libs} psi base device
  SOURCES test_add_value.cpp ../base_matrix.cpp ../hcontainer.cpp ../atom_pair.cpp
  ../../../source_basis/module_ao/parallel_orbitals.cpp tmp_mocks.cpp
)

endif() 
