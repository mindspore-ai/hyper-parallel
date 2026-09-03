cmake_minimum_required(VERSION 3.18)
project(aclshmem_torch)

# torch link path: ABI must match PyTorch / torch_npu (=1, official default since 2.7).
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate position-independent code.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Read the CANN environment.
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

# Read the aclshmem environment.
if(NOT DEFINED ENV{SHMEM_HOME_PATH})
    message(FATAL_ERROR "Cannot find SHMEM_HOME_PATH. Run the explicit native dependency/build workflow first.")
endif()
set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})

if(NOT ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB OR
  NOT EXISTS "${ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB}")
    message(FATAL_ERROR
        "ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB must name the kernel library installed by "
        "build_symmetric_memory.sh. Got: '${ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB}'")
endif()

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

add_compile_options(
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    # avoid ascendc interference
    -DL2_CACHE_HINT
)

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${ASCEND_HOME_PATH}/compiler/tikcpp
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof
)

if(NOT DEFINED Python3_EXECUTABLE)
    message(WARNING "Python3_EXECUTABLE is not defined, using default python3")
else()
    message("Using python: ${Python3_EXECUTABLE}")
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

function(find_python_package_root package_name output_variable)
    set(package_lookup_code
        "from importlib.util import find_spec"
        "from pathlib import Path"
        "spec = find_spec('${package_name}')"
        "print(Path(spec.origin).resolve().parent if spec and spec.origin else '')"
    )
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "${package_lookup_code}"
        RESULT_VARIABLE package_lookup_status
        OUTPUT_VARIABLE package_root
        ERROR_VARIABLE package_lookup_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT package_lookup_status EQUAL 0 OR NOT IS_DIRECTORY "${package_root}")
        message(FATAL_ERROR
            "Cannot locate the ${package_name} package used by ${Python3_EXECUTABLE}: "
            "${package_lookup_error}")
    endif()
    set(${output_variable} "${package_root}" PARENT_SCOPE)
endfunction()

# This adapter is a C++ operator library loaded by torch.ops.load_library().
# Direct package discovery keeps the build independent of Caffe2's optional
# CUDA toolkit check and uses only the Torch and torch_npu C++ build inputs.
find_python_package_root("torch" TORCH_PACKAGE_ROOT)
find_python_package_root("torch_npu" TORCH_NPU_PACKAGE_ROOT)

set(TORCH_INCLUDE_ROOT "${TORCH_PACKAGE_ROOT}/include")
set(TORCH_NPU_INCLUDE_ROOT "${TORCH_NPU_PACKAGE_ROOT}/include")
set(TORCH_LIBRARY_ROOT "${TORCH_PACKAGE_ROOT}/lib")
set(TORCH_NPU_LIBRARY_ROOT "${TORCH_NPU_PACKAGE_ROOT}/lib")
set(TORCH_LINK_LIBRARIES
    "${TORCH_LIBRARY_ROOT}/libc10.so"
    "${TORCH_LIBRARY_ROOT}/libtorch.so"
    "${TORCH_LIBRARY_ROOT}/libtorch_cpu.so"
    "${TORCH_NPU_LIBRARY_ROOT}/libtorch_npu.so"
)

foreach(required_path IN ITEMS
    "${TORCH_INCLUDE_ROOT}"
    "${TORCH_INCLUDE_ROOT}/torch/csrc/api/include"
    "${TORCH_NPU_INCLUDE_ROOT}"
    ${TORCH_LINK_LIBRARIES})
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR
            "The active Torch environment is missing a required C++ build input: ${required_path}")
    endif()
endforeach()

include_directories(
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_ROOT}
    ${TORCH_INCLUDE_ROOT}/torch/csrc/api/include
    ${TORCH_NPU_INCLUDE_ROOT}
    ${SHMEM_HOME_PATH}/shmem/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ops/include
)

link_directories(
    ${SHMEM_HOME_PATH}/shmem/lib
)

if(BUILD_TORCH_LIB)
    set(TORCH_BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/torch_bindings.cpp)
    add_library(aclshmem_torch
        SHARED ${TORCH_BINDINGS_SRC})
    target_link_libraries(
        aclshmem_torch PRIVATE
        ${TORCH_LINK_LIBRARIES}
        hyper_parallel_shmem_utils
        hyper_parallel_shmem
        "${ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB}")
    set_target_properties(aclshmem_torch PROPERTIES
        INSTALL_RPATH $ORIGIN:$ORIGIN/../..:$ORIGIN/../../shmem)
    install(TARGETS aclshmem_torch DESTINATION . COMPONENT aclshmem_torch)
endif()
