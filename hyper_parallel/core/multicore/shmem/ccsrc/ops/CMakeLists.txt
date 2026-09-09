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
project(hyper_parallel_shmem_ops)

# Build the SHMEM op device kernels and their host launchers into one fat
# kernel library (libhyper_parallel_shmem_ops.so). The exposed interface is
# POD-only (see ccsrc/ops/include/shmem_kernel.h); the host launcher is
# compiled with ABI=0 so both ABI variants of framework wrappers can link it.

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
    message(FATAL_ERROR "Cannot find SHMEM_HOME_PATH, please run set_env.sh.")
else()
    set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})
endif()
if(NOT DEFINED ENV{SHMEM_SOURCE_DIR})
    message(FATAL_ERROR "Cannot find SHMEM_SOURCE_DIR, please run the explicit native dependency workflow.")
else()
    set(SHMEM_SOURCE_DIR $ENV{SHMEM_SOURCE_DIR})
endif()

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

add_compile_options(
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    # avoid ascendc interference
    -DL2_CACHE_HINT
    -DTILING_KEY_VAR
)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
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
    ${ASCEND_HOME_PATH}/include/aclnn
)

link_directories(
    ${ASCEND_HOME_PATH}/lib64
    ${SHMEM_HOME_PATH}/shmem/lib
)

link_libraries(runtime stdc++ ascendcl m tiling_api platform c_sec dl nnopbase)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/obj)

# Device kernel objects, compiled by the cce frontend. data_plane headers are
# resolved from the ccsrc include root.
add_library(all_gather_kernel OBJECT all_gather/kernel/all_gather_kernel.cpp)
set_target_properties(all_gather_kernel PROPERTIES
    OBJECT_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/obj"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/obj"
)
target_compile_options(all_gather_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-vec)
target_include_directories(all_gather_kernel PRIVATE
    ${SHMEM_HOME_PATH}/shmem/include
    ${SHMEM_SOURCE_DIR}/src/device
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Host launcher objects, compiled in plain C++ mode with ABI=0. The interface
# in ops/include/shmem_kernel.h is POD-only, so framework wrappers built with
# either ABI can link the resulting library safely.
add_library(all_gather_host OBJECT all_gather/host/all_gather.cpp)
set_target_properties(all_gather_host PROPERTIES
    OBJECT_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/obj"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/obj"
)
target_compile_options(all_gather_host PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS} -D_GLIBCXX_USE_CXX11_ABI=0)
target_include_directories(all_gather_host PRIVATE
    ${SHMEM_HOME_PATH}/shmem/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Fat kernel library: device objects + host launcher, linked against the
# private SHMEM SDK libraries staged by the sdk phase.
add_library(hyper_parallel_shmem_ops SHARED
    $<TARGET_OBJECTS:all_gather_kernel>
    $<TARGET_OBJECTS:all_gather_host>
)
target_link_options(hyper_parallel_shmem_ops PRIVATE --cce-fatobj-link)
target_link_libraries(hyper_parallel_shmem_ops PRIVATE hyper_parallel_shmem)
set_target_properties(hyper_parallel_shmem_ops PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "$ORIGIN:$ORIGIN/shmem"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/shmem"
)

install(TARGETS hyper_parallel_shmem_ops LIBRARY DESTINATION lib)
# Stage the private SHMEM SDK libraries next to the kernel library so the
# install tree is self-contained even when the legacy ops phase is removed.
file(GLOB SHMEM_SO_FILES "${SHMEM_HOME_PATH}/shmem/lib/*")
install(FILES ${SHMEM_SO_FILES} DESTINATION lib/shmem)
