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
project(hyper_parallel_shmem_native_ut LANGUAGES CXX)

# Host-compilable unit tests for the SHMEM native stack. The CANN host and
# device surfaces are replaced by test fakes, so these targets build and run
# without an Ascend toolkit or NPU. Each target is one fake-boundary
# substitution unit and dispatches cases via `--case <name>`; the reinit
# target keeps its own main with positional case dispatch.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CCSRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../hyper_parallel/core/multicore/shmem/ccsrc)
if(NOT EXISTS "${CCSRC_DIR}/runtime/runtime.h")
    message(FATAL_ERROR "Cannot locate the SHMEM ccsrc tree at ${CCSRC_DIR}")
endif()

# All four binaries land in one directory for the pytest launcher.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Core semantics target: chunking, allocation registry, config and the
# runtime state machine, with cann/host replaced by fake_host_surface.
add_executable(hp_shmem_core_test
    test_main.cpp
    test_transfer_chunking.cpp
    test_allocation_registry.cpp
    test_config.cpp
    test_runtime.cpp
    fakes/fake_host_surface.cpp
    ${CCSRC_DIR}/runtime/allocation.cpp
    ${CCSRC_DIR}/runtime/config.cpp
    ${CCSRC_DIR}/runtime/log.cpp
    ${CCSRC_DIR}/runtime/runtime.cpp
)
target_compile_definitions(hp_shmem_core_test PRIVATE HP_SHMEM_CORE_TEST)
target_include_directories(hp_shmem_core_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CCSRC_DIR}
)

# Host adapter target: the real cann/host translation unit against a fake
# aclshmem/acl CANN ABI.
add_executable(hp_shmem_host_adapter_test
    test_main.cpp
    test_host_adapter.cpp
    fakes/fake_acl_cann.cpp
    ${CCSRC_DIR}/cann/host.cpp
)
target_compile_definitions(hp_shmem_host_adapter_test PRIVATE HP_SHMEM_HOST_ADAPTER_TEST)
target_include_directories(hp_shmem_host_adapter_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CCSRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/fakes/host_include
)

# Device surface target: header-only device surface and data plane against a
# fake device-side aclshmem ABI.
add_executable(hp_shmem_device_surface_test
    test_main.cpp
    test_device_surface.cpp
    fakes/fake_shmem_device.cpp
)
target_compile_definitions(hp_shmem_device_surface_test PRIVATE HP_SHMEM_DEVICE_SURFACE_TEST)
target_include_directories(hp_shmem_device_surface_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CCSRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/fakes/device_include
)

# Multi-lifecycle reinit target: self-contained fakes and log stubs, so it
# links only the runtime sources it exercises.
add_executable(hp_shmem_reinit_test
    test_runtime_reinit.cpp
    ${CCSRC_DIR}/runtime/allocation.cpp
    ${CCSRC_DIR}/runtime/config.cpp
    ${CCSRC_DIR}/runtime/runtime.cpp
)
target_include_directories(hp_shmem_reinit_test PRIVATE ${CCSRC_DIR})
