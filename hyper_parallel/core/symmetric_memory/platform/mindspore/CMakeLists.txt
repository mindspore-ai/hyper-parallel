cmake_minimum_required(VERSION 3.18)
project(aclshmem_ms)

# =============================================================================
# Configuration and Validation
# =============================================================================

set(MS_EXTENSION_NAME "aclshmem_ms")
set(BUILD_EXTENSION_DIR "${CMAKE_BINARY_DIR}/kernel_meta")
if(MS_EXTENSION_NAME STREQUAL "")
    message(FATAL_ERROR "MS_EXTENSION_NAME must be set. Use -DMS_EXTENSION_NAME=<name>")
endif()
if(BUILD_EXTENSION_DIR STREQUAL "")
    message(FATAL_ERROR "BUILD_EXTENSION_DIR must be set. Use -DBUILD_EXTENSION_DIR=<path>")
endif()

# =============================================================================
# Collect Source Files from Ops Directories
# =============================================================================

add_subdirectory(c_api)
add_subdirectory(framework)

set(OPS_SRC_FILES ${C_API_SRC_FILES} ${FRAMEWORK_SRC_FILES})

if(NOT DEFINED ENV{SHMEM_HOME_PATH})
    if(NOT DEFINED ENV{HP_THIRD_PARTY_DIR})
        message(FATAL_ERROR "Cannot find SHMEM from source or prebuild, please check log.")
    else()
        set(SHMEM_HOME_PATH $ENV{HP_THIRD_PARTY_DIR})
    endif()
else()
    set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})
endif()

# Include find_ms_path.cmake to set up MS_PATH (mindspore path)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_ms_path.cmake)

find_library(ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB
    NAMES aclshmem_symmetric_memory_kernel
    PATHS
    ${CMAKE_CURRENT_BINARY_DIR}/../ops
    ${CMAKE_CURRENT_BINARY_DIR}/../../lib/hyper_parallel/lib
    NO_DEFAULT_PATH
)


# Set library and source variables
set(SRC_FILES  ${OPS_SRC_FILES})  #如果有framework再加上其路径
set(LIB_DIRS ${SHMEM_HOME_PATH}/shmem/lib
    ${CMAKE_CURRENT_BINARY_DIR}/../../lib/hyper_parallel/lib/shmem)
set(LIBS hpshmem)
set(INCLUDE_DIRS ${SHMEM_HOME_PATH}/shmem/include ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ops/include ${MS_PATH}/include ${MS_PATH}/core/include)

message(STATUS "LIB_DIRS: ${LIB_DIRS}")
message(STATUS "LIBS: ${LIBS}")
message(STATUS "SRC_FILES: ${SRC_FILES}")
message(STATUS "INCLUDE_DIRS: ${INCLUDE_DIRS}")

# =============================================================================
# Build Configuration
# =============================================================================

# Convert include directories to CFLAGS format
set(CFLAGS_INCLUDES "")
foreach(INC_DIR_ITEM ${INCLUDE_DIRS})
    if(CFLAGS_INCLUDES STREQUAL "")
        set(CFLAGS_INCLUDES "-I${INC_DIR_ITEM}")
    else()
        set(CFLAGS_INCLUDES "${CFLAGS_INCLUDES} -I${INC_DIR_ITEM}")
    endif()
endforeach()

message(STATUS "CFLAGS_INCLUDES: ${CFLAGS_INCLUDES}")


# Convert lib directories and libs to LDLAGS format
set(LDLAGS_LIB_DIRS "")
foreach(LIB_DIR_ITEM ${LIB_DIRS})
    if(LDLAGS_LIB_DIRS STREQUAL "")
        set(LDLAGS_LIB_DIRS "-L${LIB_DIR_ITEM}")
    else()
        set(LDLAGS_LIB_DIRS "${LDLAGS_LIB_DIRS} -L${LIB_DIR_ITEM}")
    endif()
endforeach()

message(STATUS "LDLAGS_LIB_DIRS: ${LDLAGS_LIB_DIRS}")

set(LDLAGS_LIBS "")
foreach(LIB_ITEM ${LIBS})
    if(LDLAGS_LIBS STREQUAL "")
        set(LDLAGS_LIBS "-l${LIB_ITEM}")
    else()
        set(LDLAGS_LIBS "${LDLAGS_LIBS} -l${LIB_ITEM}")
    endif()
endforeach()

set(LDLAGS_LIBS "${LDLAGS_LIBS} ${ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB}")

message(STATUS "LDLAGS_LIBS: ${LDLAGS_LIBS}")


# =============================================================================
# Get YAML files
# =============================================================================
function(get_yaml_files YAML_FILES OUTPUT_VAR)
    set(YAML_STRING "[")
    set(FIRST_ITEM TRUE)
    foreach(YAML_FILE ${YAML_FILES})
        if(NOT FIRST_ITEM)
            set(YAML_STRING "${YAML_STRING}, ")
        endif()
        set(YAML_STRING "${YAML_STRING}'${YAML_FILE}'")
        set(FIRST_ITEM FALSE)
    endforeach()
    set(YAML_STRING "${YAML_STRING}]")
    set(${OUTPUT_VAR} "${YAML_STRING}" PARENT_SCOPE)
endfunction()

file(GLOB_RECURSE OPS_YAML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*/*/*_op.yaml")
message(STATUS "OPS_YAML_FILES: ${OPS_YAML_FILES}")
get_yaml_files("${OPS_YAML_FILES}" DEF_YAML_STRING)
set(DOC_YAML_STRING "[]")

# =============================================================================
# Custom Op Builder
# =============================================================================

# Generate Python script for building custom ops with MindSpore's CustomOpBuilder
set(ENABLE_DEBUG False)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(ENABLE_DEBUG True)
endif()
set(PYTHON_SCRIPT_PATH "${CMAKE_BINARY_DIR}/build_custom_with_ms.py")
file(WRITE ${PYTHON_SCRIPT_PATH} "
import mindspore as ms
src_files = '${SRC_FILES}'.split(';')
rpath_flag = \"-Wl,-rpath,'$$ORIGIN'/../../../../lib:'$$ORIGIN'/../../../../lib/shmem \"
ms.ops.CustomOpBuilder(
    name='${MS_EXTENSION_NAME}',
    sources=src_files,
    op_def=${DEF_YAML_STRING},
    op_doc=${DOC_YAML_STRING},
    backend='Ascend',
    cflags='${CFLAGS_INCLUDES}',
    ldflags=f'{rpath_flag} ${LDLAGS_LIB_DIRS} ${LDLAGS_LIBS}',
    build_dir='${BUILD_EXTENSION_DIR}',
    debug_mode=${ENABLE_DEBUG}
).build()
")

# Find Python and create custom target
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_target(
    BuildCustomOp ALL
    COMMAND cd ${CMAKE_BINARY_DIR} && ${Python3_EXECUTABLE} ${PYTHON_SCRIPT_PATH}
    DEPENDS ${ASCENDC_TARGET_NAME}
    COMMENT "Building custom operator with MindSpore"
)

install(FILES ${BUILD_EXTENSION_DIR}/${MS_EXTENSION_NAME}.so DESTINATION ${MS_EXTENSION_NAME})
install(FILES ${BUILD_EXTENSION_DIR}/${MS_EXTENSION_NAME}_auto_generate/gen_ops_def.py
              ${BUILD_EXTENSION_DIR}/${MS_EXTENSION_NAME}_auto_generate/gen_ops_prim.py
              DESTINATION ${MS_EXTENSION_NAME})