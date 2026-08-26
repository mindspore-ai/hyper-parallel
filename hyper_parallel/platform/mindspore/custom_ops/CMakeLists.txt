# =============================================================================
# Copyright 2026 Huawei Technologies Co., Ltd
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
# =============================================================================
#
# Build hyper_parallel_custom_ops_ms.so — pre-compiled MindSpore custom ops
# extension for Lightning Indexer and mHC operators.

cmake_minimum_required(VERSION 3.18)
project(hyper_parallel_custom_ops_ms)

# mindspore link path: ABI must match mindspore (=0).
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

set(MS_EXTENSION_NAME "hyper_parallel_custom_ops_ms")
set(BUILD_EXTENSION_DIR "${CMAKE_BINARY_DIR}/lib")

# =============================================================================
# Collect source files
# =============================================================================

# Gather all .cc files in the current source directory (flat structure)
file(GLOB SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

message(STATUS "MS_EXTENSION_NAME: ${MS_EXTENSION_NAME}")
message(STATUS "SRC_FILES: ${SRC_FILES}")
message(STATUS "BUILD_EXTENSION_DIR: ${BUILD_EXTENSION_DIR}")

# =============================================================================
# Custom Op Builder — generate Python script and execute via cmake target
# =============================================================================

set(ENABLE_DEBUG False)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(ENABLE_DEBUG True)
endif()

# Convert CMake list to Python list string for the generated script
set(SRC_FILES_PYTHON "")
foreach(SRC_FILE ${SRC_FILES})
    if(SRC_FILES_PYTHON STREQUAL "")
        set(SRC_FILES_PYTHON "'${SRC_FILE}'")
    else()
        set(SRC_FILES_PYTHON "${SRC_FILES_PYTHON}, '${SRC_FILE}'")
    endif()
endforeach()

set(PYTHON_SCRIPT_PATH "${CMAKE_BINARY_DIR}/build_custom_with_ms.py")
file(WRITE ${PYTHON_SCRIPT_PATH} "
import mindspore as ms

src_files = [${SRC_FILES_PYTHON}]

class _HyperParallelCustomOpBuilder(ms.ops.CustomOpBuilder):
    \"\"\"Remove MindSpore 2.10's bare --rpath flag from the host link command.\"\"\"

    def get_ldflags(self):
        flags = super().get_ldflags()
        return [flag for flag in flags if flag != '-Wl,--disable-new-dtags,--rpath']


_HyperParallelCustomOpBuilder(
    name='${MS_EXTENSION_NAME}',
    sources=src_files,
    op_def=[],
    op_doc=[],
    backend='Ascend',
    cflags='-D_GLIBCXX_USE_CXX11_ABI=0',
    build_dir='${BUILD_EXTENSION_DIR}',
    debug_mode=${ENABLE_DEBUG}
).build()

print('Build complete: ' + '${BUILD_EXTENSION_DIR}/${MS_EXTENSION_NAME}.so')
")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_target(
    BuildCustomOp ALL
    COMMAND cd ${CMAKE_BINARY_DIR} && ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS
            ${Python3_EXECUTABLE} ${PYTHON_SCRIPT_PATH}
    COMMENT "Building ${MS_EXTENSION_NAME} custom operator with MindSpore"
)
