# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0.
cmake_minimum_required(VERSION 3.18)
project(hyper_parallel_routed_experts_torch LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/hardening.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/torch_compat.cmake)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
hp_detect_torch_build_contract("${Python3_EXECUTABLE}" TORCH)
if(NOT EXISTS "${ROUTED_KERNEL_LIBRARY}")
    message(FATAL_ERROR "ROUTED_KERNEL_LIBRARY must point to the explicitly built independent kernel library")
endif()
add_library(hyper_parallel_routed_experts_torch SHARED
    ../csrc/routed_experts.cpp ../csrc/routed_gradient.cpp ../csrc/routed_combine.cpp
    ../csrc/routed_combine_backward.cpp ../csrc/routed_activation.cpp ../csrc/routed_forward.cpp
    ../csrc/routed_metadata.cpp)
hp_apply_torch_cxx_standard(hyper_parallel_routed_experts_torch "${TORCH_CXX_STANDARD}")
target_compile_definitions(hyper_parallel_routed_experts_torch PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI} CURRENT_VERSION=VERSION_NEWEST)
target_include_directories(hyper_parallel_routed_experts_torch PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${TORCH_PACKAGE_ROOT}/include
    ${TORCH_PACKAGE_ROOT}/include/torch/csrc/api/include
    ${TORCH_NPU_PACKAGE_ROOT}/include
    ${TORCH_NPU_PACKAGE_ROOT}/include/third_party/acl/inc
    ${TORCH_NPU_PACKAGE_ROOT}/include/third_party/op-plugin
    ${TORCH_NPU_PACKAGE_ROOT}/include/third_party/op-plugin/op_plugin/include
    ${CMAKE_CURRENT_LIST_DIR}/../../ops/routed_experts)
target_link_libraries(hyper_parallel_routed_experts_torch PRIVATE
    ${TORCH_PACKAGE_ROOT}/lib/libc10.so
    ${TORCH_PACKAGE_ROOT}/lib/libtorch.so
    ${TORCH_PACKAGE_ROOT}/lib/libtorch_cpu.so
    ${TORCH_NPU_PACKAGE_ROOT}/lib/libtorch_npu.so
    ${ROUTED_KERNEL_LIBRARY})
hp_enable_elf_hardening(hyper_parallel_routed_experts_torch)
install(TARGETS hyper_parallel_routed_experts_torch LIBRARY DESTINATION .)
install(CODE [[
    file(RPATH_REMOVE FILE "${CMAKE_INSTALL_PREFIX}/libhyper_parallel_routed_experts_torch.so")
]])
