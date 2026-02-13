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
# ============================================================================

cmake_minimum_required(VERSION 3.16)
project(hyper_parallel_mindspore)

if(WIN32)
    message(FATAL_ERROR "This project currently supports Linux only. Windows support TBD.")
endif()

# Priority 1: User-specified path via CMake cache variable (highest priority)
set(MINDSPORE_ROOT "" CACHE PATH "Path to MindSpore installation (overrides all auto-discovery)")

# Priority 2: Environment variable MINDSPORE_ROOT
if(NOT MINDSPORE_ROOT)
    set(MINDSPORE_ROOT "$ENV{MINDSPORE_ROOT}")
    if(MINDSPORE_ROOT)
        message(STATUS "Using MINDSPORE_ROOT from environment: ${MINDSPORE_ROOT}")
    endif()
endif()

# Priority 3: Auto-discover via Python interpreter (supports PYTHONPATH/venv/conda/system)
if(NOT MINDSPORE_ROOT)
    message(STATUS "MINDSPORE_ROOT not set - attempting auto-discovery via Python...")

    # Find Python interpreter (required for path discovery)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(NOT Python3_Interpreter_FOUND)
        # Fallback: try generic 'python3' command
        find_program(PYTHON_EXECUTABLE NAMES python3 python)
        if(PYTHON_EXECUTABLE)
            set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}")
            set(Python3_Interpreter_FOUND TRUE)
        endif()
    endif()

    if(Python3_Interpreter_FOUND)
        # Execute Python to get installation prefix and site-packages location
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "
import sys
import os
import site

# Get PYTHONPATH entries (if set)
pythonpath = os.environ.get('PYTHONPATH', '')
if pythonpath:
    for entry in pythonpath.split(os.pathsep):
        if entry.strip():
            print(f'PYTHONPATH:{entry.strip()}')

# Get base prefix (handles venv/conda correctly)
prefix = sys.prefix
print(f'PREFIX:{prefix}')

# Get site-packages paths (may have multiple)
for sp in site.getsitepackages():
    print(f'SITE:{sp}')

# Also check user site-packages (for --user installs)
if site.ENABLE_USER_SITE:
    print(f'USER_SITE:{site.getusersitepackages()}')
"
            OUTPUT_VARIABLE PYTHON_DISCOVERY_OUTPUT
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # Normalize line endings and split into list
        string(REPLACE "\r" "" PYTHON_DISCOVERY_OUTPUT "${PYTHON_DISCOVERY_OUTPUT}")
        string(REPLACE "\n" ";" PYTHON_DISCOVERY_LIST "${PYTHON_DISCOVERY_OUTPUT}")
        list(FILTER PYTHON_DISCOVERY_LIST EXCLUDE REGEX "^[ \t]*$")

        # Parse Python output to extract candidate paths
        set(PYTHON_PREFIX "")
        set(PYTHON_SITE_PACKAGES "")
        set(PYTHON_USER_SITE "")
        set(PYTHONPATH_ENTRIES "")

        foreach(line IN LISTS PYTHON_DISCOVERY_LIST)
            if(line MATCHES "^PREFIX:(.+)$")
                set(PYTHON_PREFIX "${CMAKE_MATCH_1}")
            elseif(line MATCHES "^SITE:(.+)$")
                list(APPEND PYTHON_SITE_PACKAGES "${CMAKE_MATCH_1}")
            elseif(line MATCHES "^USER_SITE:(.+)$")
                set(PYTHON_USER_SITE "${CMAKE_MATCH_1}")
            elseif(line MATCHES "^PYTHONPATH:(.+)$")
                list(APPEND PYTHONPATH_ENTRIES "${CMAKE_MATCH_1}")
            endif()
        endforeach()


        # Build candidate paths from Python discovery
        set(CANDIDATE_PATHS "")

        # Candidate 1: PYTHONPATH entries (user explicitly set - highest priority after env var)
        if(PYTHONPATH_ENTRIES)
            foreach(entry IN LISTS PYTHONPATH_ENTRIES)
                get_filename_component(ABS_ENTRY "${entry}" ABSOLUTE)
                message(STATUS "Adding candidate from PYTHONPATH: ${ABS_ENTRY}/mindspore")
                list(APPEND CANDIDATE_PATHS "${ABS_ENTRY}/mindspore")
            endforeach()
        endif()

        # Candidate 2: site-packages/mindspore (standard pip/conda install)
        foreach(sp IN LISTS PYTHON_SITE_PACKAGES)
            message(STATUS "Adding candidate from PYTHON_SITE_PACKAGES: ${PYTHON_SITE_PACKAGES}/mindspore")
            list(APPEND CANDIDATE_PATHS "${sp}/mindspore")
        endforeach()

        # Candidate 3: user site-packages (for --user installs)
        if(PYTHON_USER_SITE)
            message(STATUS "Adding candidate from PYTHON_USER_SITE: ${PYTHON_USER_SITE}/mindspore")
            list(APPEND CANDIDATE_PATHS "${PYTHON_USER_SITE}/mindspore")
        endif()

        # Candidate 4: conda/system prefix fallback (less common for dev packages)
        if(PYTHON_PREFIX)
            message(STATUS "Adding candidate from PYTHON_PREFIX: ${PYTHON_PREFIX}/mindspore")
            list(APPEND CANDIDATE_PATHS
                "${PYTHON_PREFIX}/lib/mindspore"   # Conda-style
            )
        endif()

        # Deduplicate candidate paths ===
        if(CANDIDATE_PATHS)
            list(REMOVE_DUPLICATES CANDIDATE_PATHS)
        endif()

        # Search candidates with detailed diagnostics ===
        message(STATUS "Searching MindSpore development files in candidate paths:")
        set(HEADER_CHECK_PATHS
            "include/mindspore/core/include/base/base.h"  # Older versions
        )

        set(FOUND_PATH "")
        foreach(path IN LISTS CANDIDATE_PATHS)
            get_filename_component(ABS_PATH "${path}" ABSOLUTE)
            set(FOUND_HEADER "")
            foreach(rel_header IN LISTS HEADER_CHECK_PATHS)
                set(header_path "${ABS_PATH}/${rel_header}")
                if(EXISTS "${header_path}")
                    set(FOUND_HEADER "${rel_header}")
                    break()
                endif()
            endforeach()

            if(FOUND_HEADER)
                set(FOUND_PATH "${ABS_PATH}")
                message(STATUS "  ✓ ${ABS_PATH}")
                message(STATUS "      Verified via: ${FOUND_HEADER}")
                break()  # Take first match (highest priority)
            else()
                message(STATUS "  ✗ ${ABS_PATH} (headers not found)")
            endif()
        endforeach()

        if(FOUND_PATH)
            set(MINDSPORE_ROOT "${FOUND_PATH}")
        else()
            message(STATUS "✗ Python-based discovery failed - no valid MindSpore development installation found")
            message(STATUS "  Hint: The PyPI package 'mindspore' does NOT include development headers.")
            message(STATUS "        Download the development package from https://www.mindspore.cn/install")
        endif()
    else()
        message(STATUS "✗ Python interpreter not found - skipping Python-based discovery")
    endif()
endif()

# Priority 4: Final validation and configuration
if(MINDSPORE_ROOT)
    if(NOT EXISTS "${MINDSPORE_ROOT}")
        message(FATAL_ERROR "MINDSPORE_ROOT path does not exist: ${MINDSPORE_ROOT}")
    endif()

    # Standard MindSpore installation layout
    set(MINDSPORE_INCLUDE_DIR "${MINDSPORE_ROOT}/include")
    set(MINDSPORE_LIBRARY_DIR "${MINDSPORE_ROOT}/lib")

    # Handle lib64 systems
    if(EXISTS "${MINDSPORE_ROOT}/lib64")
        set(MINDSPORE_LIBRARY_DIR "${MINDSPORE_ROOT}/lib64")
    endif()

    # Verify critical header files exist
    if(NOT EXISTS "${MINDSPORE_INCLUDE_DIR}/mindspore/core/include/base/base.h")
        message(FATAL_ERROR
            "MindSpore headers not found in ${MINDSPORE_INCLUDE_DIR}\n"
            "Expected: ${MINDSPORE_INCLUDE_DIR}/mindspore/core/include/base/base.h\n"
            "Please verify MINDSPORE_ROOT points to a MindSpore *development* installation (not just Python package)."
        )
    endif()

    # Locate library file (support multiple naming conventions)
    find_library(MINDSPORE_LIBRARY
        NAMES mindspore_backend_common mindspore_core mindspore_common
        PATHS "${MINDSPORE_LIBRARY_DIR}" "${MINDSPORE_ROOT}/lib64"
        NO_DEFAULT_PATH
    )

    if(NOT MINDSPORE_LIBRARY)
        message(FATAL_ERROR
            "MindSpore library not found in ${MINDSPORE_LIBRARY_DIR}. "
            "Searched for: mindspore_common mindspore_core mindspore_frontend\n"
            "Please ensure they are exists in lib/ directory."
        )
    endif()

    set(MINDSPORE_FOUND TRUE)
    message(STATUS "Configured MindSpore development environment:")
    message(STATUS "  Root:      ${MINDSPORE_ROOT}")
    message(STATUS "  Include:   ${MINDSPORE_INCLUDE_DIR}")
    message(STATUS "  Library:   ${MINDSPORE_LIBRARY}")
else()
    message(WARNING
        "MindSpore development files not found. Custom pass compilation requires:\n"
        "  1. MindSpore >= 2.8.1 development files (headers + libraries), AND\n"
        "  2. Either:\n"
        "     a) Set MINDSPORE_ROOT environment variable, OR\n"
        "     b) Install MindSpore to standard locations (via pip or conda, etc.)\n"
        "\n"
    )
    set(MINDSPORE_FOUND FALSE)
endif()

# ============================================================================
# Build configuration (only proceed if MindSpore found)
# ============================================================================
# Link against MindSpore if found
if(NOT MINDSPORE_FOUND)
    message(FATAL_ERROR
        "MindSpore not found. Please:\n"
        "  1. Set MINDSPORE_ROOT environment variable, OR\n"
        "  2. Install MindSpore development files, OR\n"
        "  3. Skip custom pass compilation (safe for non-MindSpore users)"
    )
endif()

# Configurable hash table selection for MindSpore compatibility
# Tries to use robin_hood hashing when available for performance benefits
set(ROBIN_HOOD_H_PATH "${MINDSPORE_ROOT}/include/third_party/robin_hood_hashing/include/robin_hood.h")

# Track which hash table implementation we're using
set(HASH_TABLE_IMPL "std::unordered_map")  # Default implementation

if(EXISTS "${ROBIN_HOOD_H_PATH}")
    # Use robin_hood hashing for improved performance
    message(STATUS "Found robin_hood.h: ${ROBIN_HOOD_H_PATH}")

    add_compile_definitions(ENABLE_FAST_HASH_TABLE=1)
    include_directories("${MINDSPORE_ROOT}/include/third_party/robin_hood_hashing")

    set(HASH_TABLE_IMPL "robin_hood")
else()
    # Fall back to standard library implementation
    message(STATUS "robin_hood.h not found, defaulting to ${HASH_TABLE_IMPL}")
endif()

message(STATUS "Selected hash table implementation: ${HASH_TABLE_IMPL}")

# Enable C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE "Release")
set(BUILD_SHARED_LIBS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Recursively collect all pass source files
file(GLOB_RECURSE PASS_SOURCES
    "${CMAKE_SOURCE_DIR}/*.cc"
    "${CMAKE_SOURCE_DIR}/passes/**/*.cc"
    "${CMAKE_SOURCE_DIR}/passes/**/*.cpp"
    "${CMAKE_SOURCE_DIR}/passes/**/*.h"
)

if(NOT PASS_SOURCES)
    message(FATAL_ERROR "No pass source files found in ${CMAKE_SOURCE_DIR}/passes/")
endif()

# Create shared library target
add_library(hyper_parallel_mindspore SHARED ${PASS_SOURCES})

# Configure output naming
set_target_properties(hyper_parallel_mindspore PROPERTIES
    OUTPUT_NAME "hyper_parallel_mindspore"
    PREFIX "lib"
    SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON
)

# Configure include directories
target_include_directories(hyper_parallel_mindspore PRIVATE
    ${MINDSPORE_INCLUDE_DIR}
    ${MINDSPORE_INCLUDE_DIR}/mindspore
    ${MINDSPORE_INCLUDE_DIR}/mindspore/ccsrc
    ${MINDSPORE_INCLUDE_DIR}/mindspore/core/include
    ${MINDSPORE_INCLUDE_DIR}/mindspore/ops
    ${MINDSPORE_INCLUDE_DIR}/mindspore/ops/include
    ${MINDSPORE_INCLUDE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/passes
    ${CMAKE_SOURCE_DIR}
)

# Link against MindSpore
target_link_libraries(hyper_parallel_mindspore PRIVATE ${MINDSPORE_LIBRARY})

# Export symbols (critical: MindSpore loader requires visible symbols)
set_target_properties(hyper_parallel_mindspore PROPERTIES
    CXX_VISIBILITY_PRESET default  # Must be 'default' to export symbols
    VISIBILITY_INLINES_HIDDEN OFF
)

# Default settings
option(ENABLE_GLIBCXX "enable_glibcxx" OFF)

# System-related overrides
if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(ENABLE_GLIBCXX ON)
endif()

# Environment variable overrides
if(DEFINED ENV{ENABLE_GLIBCXX})
    set(ENABLE_GLIBCXX $ENV{ENABLE_GLIBCXX})
endif()

# ABI flag settings
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    if(NOT ENABLE_GLIBCXX)
        add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()

# Set compilation options
target_compile_options(hyper_parallel_mindspore PRIVATE
    -fPIC
    -std=c++17
    -Wall
    -Wextra
)

# Installation rules (for make install, not used by setup.py)
install(TARGETS hyper_parallel_mindspore
    LIBRARY DESTINATION lib
)
