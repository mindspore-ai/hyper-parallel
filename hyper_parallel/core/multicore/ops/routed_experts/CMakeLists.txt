# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0.
cmake_minimum_required(VERSION 3.18)
set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)
project(hyper_parallel_routed_experts_kernels LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/hardening.cmake)

foreach(required IN ITEMS ASCEND_HOME_PATH SHMEM_HOME_PATH SHMEM_SOURCE_DIR)
    if(NOT DEFINED ENV{${required}})
        message(FATAL_ERROR "Missing ${required}; activate the explicit CANN/SHMEM build environment")
    endif()
endforeach()
set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
set(SDK_ROOT $ENV{SHMEM_HOME_PATH})

add_library(hyper_parallel_routed_experts_kernels SHARED
    transport.cpp gemm.cpp gemm_host.cpp gradient.cpp gradient_host.cpp combine.cpp combine_backward.cpp
    forward.cpp forward_vector.cpp forward_host.cpp)
set_source_files_properties(gemm.cpp gradient.cpp forward.cpp PROPERTIES COMPILE_OPTIONS
    "--cce-aicore-arch=dav-c220-cube;-DASCENDC_CUBE_ONLY")
target_compile_options(hyper_parallel_routed_experts_kernels PRIVATE
    -O2 -xcce --cce-aicore-arch=dav-c220-vec
    -Wno-macro-redefined -Wno-ignored-attributes
    -DL2_CACHE_HINT -DTILING_KEY_VAR
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false")
target_include_directories(hyper_parallel_routed_experts_kernels PRIVATE
    ${CANN_ROOT}/compiler/tikcpp
    ${CANN_ROOT}/compiler/tikcpp/tikcfw
    ${CANN_ROOT}/compiler/tikcpp/tikcfw/impl
    ${CANN_ROOT}/compiler/tikcpp/tikcfw/interface
    ${CANN_ROOT}/include
    ${CANN_ROOT}/include/experiment/runtime
    ${SDK_ROOT}/shmem/include
    $ENV{SHMEM_SOURCE_DIR}/src/device
    ${CMAKE_CURRENT_LIST_DIR}/../../shmem/ccsrc)
target_link_directories(hyper_parallel_routed_experts_kernels PRIVATE
    ${CANN_ROOT}/lib64 ${SDK_ROOT}/shmem/lib)
target_link_libraries(hyper_parallel_routed_experts_kernels PRIVATE
    hyper_parallel_shmem runtime ascendcl tiling_api platform c_sec dl nnopbase)
target_link_options(hyper_parallel_routed_experts_kernels PRIVATE --cce-fatobj-link)
hp_enable_link_hardening(hyper_parallel_routed_experts_kernels)
install(TARGETS hyper_parallel_routed_experts_kernels LIBRARY DESTINATION .)
install(CODE [[
    file(RPATH_REMOVE FILE "${CMAKE_INSTALL_PREFIX}/libhyper_parallel_routed_experts_kernels.so")
]])
