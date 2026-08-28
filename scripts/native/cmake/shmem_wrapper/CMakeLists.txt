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

cmake_minimum_required(VERSION 3.18)
project(hyper_parallel_shmem_wrapper LANGUAGES NONE)

if(NOT DEFINED HP_SHMEM_SOURCE_DIR OR NOT EXISTS "${HP_SHMEM_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "HP_SHMEM_SOURCE_DIR must identify the verified SHMEM source tree")
endif()

# Keep the upstream source tree immutable. This wrapper owns packaging-only
# policy and deliberately applies it after upstream has declared every target.
add_subdirectory("${HP_SHMEM_SOURCE_DIR}" shmem-upstream)

set(CMAKE_SKIP_RPATH FALSE)
set_target_properties(shmem PROPERTIES OUTPUT_NAME hyper_parallel_shmem)
set_target_properties(shmem_utils PROPERTIES OUTPUT_NAME hyper_parallel_shmem_utils)
foreach(target_name
        shmem
        shmem_utils
        aclshmem_bootstrap_uid
        aclshmem_bootstrap_config_store
        aclshmem_bootstrap_mpi)
    if(TARGET ${target_name})
        set_target_properties(${target_name} PROPERTIES
            SKIP_BUILD_RPATH FALSE
            SKIP_INSTALL_RPATH FALSE
            BUILD_WITH_INSTALL_RPATH TRUE
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN")
        # SHMEM v1.6.0 sets CMAKE_SKIP_RPATH in its child directory. LINK_FLAGS
        # is used here so the packaging policy remains outside upstream and the
        # Makefile receives one literal $ORIGIN rather than $$ORIGIN.
        set_property(TARGET ${target_name} APPEND_STRING PROPERTY LINK_FLAGS
            " -Wl,--enable-new-dtags,-rpath,'$ORIGIN'")
    endif()
endforeach()
