# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set(FWD_HOST ${HP_MULTICORE_SOURCE_ROOT}/hyper_mega_moe/op_host)
set(BWD_HOST ${HP_MULTICORE_SOURCE_ROOT}/hyper_mega_moe_grad/op_host)
set(MHC_HOST ${HP_MULTICORE_SOURCE_ROOT}/hyper_mega_mhc/op_host)
set(MHC_GRAD_HOST ${HP_MULTICORE_SOURCE_ROOT}/hyper_mega_mhc_grad/op_host)
set(CANN_PACKAGE_ARCH_ROOT ${ASCEND_CANN_PACKAGE_PATH})
string(APPEND CANN_PACKAGE_ARCH_ROOT "/${CMAKE_SYSTEM_PROCESSOR}-linux")
set(MHC_GRAD_TILING
    ${MHC_GRAD_HOST}/mhc_pre_sinkhorn_backward_tiling/arch22
)
set(CANN_OP_COMMON_INCLUDE
    ${CANN_PACKAGE_ARCH_ROOT}/include/op_common
)
set(CANN_OP_HOST_INCLUDE ${CANN_OP_COMMON_INCLUDE}/op_host)
set(CANN_ACLNN_INCLUDE
    ${CANN_PACKAGE_ARCH_ROOT}/include/aclnn
)
set(CANN_ACLNN_OP_INCLUDE
    ${CANN_PACKAGE_ARCH_ROOT}/include/aclnnop
)
set(CANN_ARCH_LIBRARY
    ${CANN_PACKAGE_ARCH_ROOT}/lib64
)

set(OP_SCHEMA_SOURCES
    ${FWD_HOST}/hyper_mega_moe_def.cpp
    ${FWD_HOST}/hyper_mega_moe_infershape.cpp
    ${BWD_HOST}/hyper_mega_moe_grad_def.cpp
    ${BWD_HOST}/hyper_mega_moe_grad_infershape.cpp
    ${MHC_HOST}/hyper_mega_mhc_def.cpp
    ${MHC_HOST}/hyper_mega_mhc_infershape.cpp
    ${MHC_GRAD_HOST}/hyper_mega_mhc_grad_def.cpp
    ${MHC_GRAD_HOST}/hyper_mega_mhc_grad_infershape.cpp
)
set(OP_TILING_SOURCES
    ${FWD_HOST}/hyper_mega_moe_tiling.cpp
    ${BWD_HOST}/hyper_mega_moe_grad_tiling.cpp
    ${MHC_HOST}/hyper_mega_mhc_tiling.cpp
    ${MHC_HOST}/mhc_pre_sinkhorn_tiling/mhc_pre_sinkhorn_tiling.cpp
    ${MHC_HOST}/mhc_pre_sinkhorn_tiling/mhc_pre_sinkhorn_tiling_arch35.cpp
    ${MHC_GRAD_HOST}/hyper_mega_mhc_grad_tiling.cpp
    ${MHC_GRAD_TILING}/mhc_pre_sinkhorn_backward_arch22_tiling.cpp
)

npu_op_code_gen(
    SRC ${OP_SCHEMA_SOURCES}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    JOIN_OP_DEF False
    COMPILE_OPTIONS
        -I${CANN_OP_COMMON_INCLUDE}
        -I${CANN_OP_HOST_INCLUDE}
)

set(ACLNN_SOURCES
    ${FWD_HOST}/op_api/aclnn_hyper_mega_moe.cpp
    ${FWD_HOST}/op_api/hyper_mega_moe.cpp
    ${BWD_HOST}/op_api/aclnn_hyper_mega_moe_grad.cpp
    ${BWD_HOST}/op_api/hyper_mega_moe_grad.cpp
    ${MHC_HOST}/op_api/aclnn_hyper_mega_mhc.cpp
    ${MHC_HOST}/op_api/hyper_mega_mhc.cpp
    ${MHC_GRAD_HOST}/op_api/aclnn_hyper_mega_mhc_grad.cpp
    ${MHC_GRAD_HOST}/op_api/hyper_mega_mhc_grad.cpp
)
npu_op_library(cust_opapi ACLNN ${ACLNN_SOURCES})
target_compile_options(cust_opapi PRIVATE -fvisibility=hidden)
target_include_directories(cust_opapi PRIVATE
    ${FWD_HOST}/op_api
    ${BWD_HOST}/op_api
    ${MHC_HOST}/op_api
    ${MHC_GRAD_HOST}/op_api
    ${CANN_OP_COMMON_INCLUDE}
    ${CANN_OP_HOST_INCLUDE}
    ${CANN_ACLNN_INCLUDE}
    ${CANN_ACLNN_OP_INCLUDE}
)
target_link_directories(cust_opapi PRIVATE ${CANN_ARCH_LIBRARY})
target_link_libraries(cust_opapi PRIVATE opapi_math)
hp_enable_elf_hardening(cust_opapi)

file(GLOB GROUP_PROTO_SOURCES ${ASCEND_AUTOGEN_PATH}/group_op_proto/*.cc)
file(GLOB OP_PROTO_SOURCES ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(
    ${GROUP_PROTO_SOURCES} ${OP_PROTO_SOURCES} PROPERTIES GENERATED TRUE
)
npu_op_library(cust_op_proto GRAPH
    ${OP_SCHEMA_SOURCES}
    ${GROUP_PROTO_SOURCES}
    ${OP_PROTO_SOURCES}
)
target_compile_options(cust_op_proto PRIVATE -fvisibility=hidden)
target_include_directories(cust_op_proto PRIVATE
    ${FWD_HOST}
    ${BWD_HOST}
    ${MHC_HOST}
    ${MHC_GRAD_HOST}
    ${HP_MULTICORE_SOURCE_ROOT}/ops_transformer_common
    ${CANN_OP_COMMON_INCLUDE}
    ${CANN_OP_HOST_INCLUDE}
)
hp_enable_elf_hardening(cust_op_proto)

file(GLOB FALLBACK_SOURCES ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)
set_source_files_properties(${FALLBACK_SOURCES} PROPERTIES GENERATED TRUE)
npu_op_library(cust_optiling TILING
    ${OP_SCHEMA_SOURCES}
    ${OP_TILING_SOURCES}
    ${FALLBACK_SOURCES}
)
target_compile_options(cust_optiling PRIVATE -fvisibility=hidden)
target_include_directories(cust_optiling PRIVATE
    ${FWD_HOST}
    ${BWD_HOST}
    ${MHC_HOST}
    ${MHC_HOST}/mhc_pre_sinkhorn_tiling
    ${MHC_HOST}/op_host
    ${MHC_GRAD_HOST}
    ${MHC_GRAD_HOST}/mhc_pre_sinkhorn_backward_tiling
    ${HP_MULTICORE_SOURCE_ROOT}/hyper_mega_mhc_grad/op_kernel
    ${HP_MULTICORE_SOURCE_ROOT}/ops_transformer_common
    ${CANN_OP_COMMON_INCLUDE}
    ${CANN_OP_HOST_INCLUDE}
)
target_link_directories(cust_optiling PRIVATE ${CANN_ARCH_LIBRARY})
target_link_libraries(cust_optiling PRIVATE ops_base)
hp_enable_elf_hardening(cust_optiling)

npu_op_package_add(${package_name}
    LIBRARY cust_optiling cust_opapi cust_op_proto
)
npu_op_package_add(${package_name}
    TYPE ACLNN
    FILES
        ${FWD_HOST}/op_api/aclnn_hyper_mega_moe.h
        ${BWD_HOST}/op_api/aclnn_hyper_mega_moe_grad.h
        ${MHC_HOST}/op_api/aclnn_hyper_mega_mhc.h
        ${MHC_GRAD_HOST}/op_api/aclnn_hyper_mega_mhc_grad.h
)
