# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18)
project(hyper_parallel_shmem_torch LANGUAGES CXX)

# Build the new SHMEM stack host side: the runtime static library
# (runtime + CANN host adapter) and the pybind11 Torch binding module
# (hyper_parallel_shmem_torch.so). The ops kernel library built by the
# ccsrc/ops project is linked into the binding module.

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
    message(FATAL_ERROR
        "Cannot find SHMEM_HOME_PATH. Run the explicit native "
        "dependency/build workflow first.")
endif()
set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})

if(NOT SHMEM_OPS_LIB OR
  NOT EXISTS "${SHMEM_OPS_LIB}")
    message(FATAL_ERROR
        "SHMEM_OPS_LIB must name the ops kernel library "
        "installed by the ccsrc/ops build. Got: "
        "'${SHMEM_OPS_LIB}'")
endif()

add_compile_options(
    -O2
    -Wno-macro-redefined -Wno-ignored-attributes
)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

function(find_python_package_root package_name output_variable)
    string(CONCAT package_lookup_code
        "import importlib.util, sys; "
        "spec = importlib.util.find_spec(sys.argv[1]); "
        "locations = getattr(spec, 'submodule_search_locations', None) or []; "
        "print(next(iter(locations), ''))")
    execute_process(
        COMMAND ${Python3_EXECUTABLE}
                -c "${package_lookup_code}" "${package_name}"
        RESULT_VARIABLE package_lookup_status
        OUTPUT_VARIABLE package_root
        ERROR_VARIABLE package_lookup_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT package_lookup_status EQUAL 0 OR
      NOT IS_DIRECTORY "${package_root}")
        message(FATAL_ERROR
            "Cannot locate ${package_name} for ${Python3_EXECUTABLE}: "
            "${package_lookup_error}")
    endif()
    set(${output_variable} "${package_root}" PARENT_SCOPE)
endfunction()

find_python_package_root("torch" TORCH_PACKAGE_ROOT)
find_python_package_root("torch_npu" TORCH_NPU_PACKAGE_ROOT)
set(torch_abi_code
    "import torch; print(int(torch.compiled_with_cxx11_abi()))")
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env TORCH_DEVICE_BACKEND_AUTOLOAD=0
            ${Python3_EXECUTABLE} -c "${torch_abi_code}"
    RESULT_VARIABLE torch_abi_status
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    ERROR_VARIABLE torch_abi_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT torch_abi_status EQUAL 0 OR
  NOT TORCH_CXX11_ABI MATCHES "^[01]$")
    message(FATAL_ERROR
        "Cannot determine the active Torch C++ ABI: ${torch_abi_error}")
endif()

set(TORCH_INCLUDE_ROOT "${TORCH_PACKAGE_ROOT}/include")
set(TORCH_NPU_INCLUDE_ROOT "${TORCH_NPU_PACKAGE_ROOT}/include")
set(TORCH_LINK_LIBRARIES
    "${TORCH_PACKAGE_ROOT}/lib/libc10.so"
    "${TORCH_PACKAGE_ROOT}/lib/libtorch.so"
    "${TORCH_PACKAGE_ROOT}/lib/libtorch_cpu.so"
    "${TORCH_NPU_PACKAGE_ROOT}/lib/libtorch_npu.so")

foreach(required_path IN ITEMS
    ${Python3_INCLUDE_DIRS}
    "${TORCH_INCLUDE_ROOT}"
    "${TORCH_INCLUDE_ROOT}/torch/csrc/api/include"
    "${TORCH_NPU_INCLUDE_ROOT}/torch_npu/csrc/core/npu/NPUStream.h"
    ${TORCH_LINK_LIBRARIES})
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR
            "The active Torch environment is missing a build input: "
            "${required_path}")
    endif()
endforeach()

# Runtime and binding are compiled into one module, so they share the active
# Torch C++ ABI.
add_compile_definitions(
    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})

# Runtime static library: allocation ledger, config, log, runtime state
# machine and the CANN host adapter. Internal to the binding module.
add_library(hyper_parallel_shmem_runtime STATIC
    runtime/allocation.cpp
    runtime/config.cpp
    runtime/log.cpp
    runtime/runtime.cpp
    cann/host.cpp
)
target_include_directories(hyper_parallel_shmem_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_HOME_PATH}/include
    ${SHMEM_HOME_PATH}/shmem/include
)

# pybind11 Torch binding module, imported by name from Python.
add_library(hyper_parallel_shmem_torch MODULE
    bindings/torch/torch_bindings.cpp
)
set_target_properties(hyper_parallel_shmem_torch PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)
target_include_directories(hyper_parallel_shmem_torch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_ROOT}
    ${TORCH_INCLUDE_ROOT}/torch/csrc/api/include
    ${TORCH_NPU_INCLUDE_ROOT}
    ${ASCEND_HOME_PATH}/include
    ${SHMEM_HOME_PATH}/shmem/include
)
target_link_libraries(hyper_parallel_shmem_torch PRIVATE
    hyper_parallel_shmem_runtime
    ${TORCH_LINK_LIBRARIES}
    "${SHMEM_HOME_PATH}/shmem/lib/libhyper_parallel_shmem_utils.so"
    "${SHMEM_HOME_PATH}/shmem/lib/libhyper_parallel_shmem.so"
    "${SHMEM_OPS_LIB}"
)
target_link_options(hyper_parallel_shmem_torch PRIVATE "-Wl,--enable-new-dtags")
# Installed under lib/framework/torch: kernel library at ../.., SHMEM SDK
# libraries at ../../shmem.
set_target_properties(hyper_parallel_shmem_torch PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../..:$ORIGIN/../../shmem"
)

install(TARGETS hyper_parallel_shmem_torch LIBRARY DESTINATION .)
