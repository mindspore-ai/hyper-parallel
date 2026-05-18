cmake_minimum_required(VERSION 3.18)
project(multicore_ms)

set(MS_EXTENSION_NAME "hyper_parallel_mega_moe_ms")
set(BUILD_EXTENSION_DIR "${CMAKE_BINARY_DIR}/lib")

# =============================================================================
# Prebuild Auto-Extract
# If no vendor env var is set, extract prebuild/mega_moe.tar.gz
# (vendors/ inside contains pre-built vendor libs).
# =============================================================================

set(_PREBUILD_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../prebuild")
set(_PREBUILD_DIR  "${_PREBUILD_ROOT}/mega_moe")
set(_TARBALL       "${_PREBUILD_DIR}.tar.gz")

if(NOT DEFINED ENV{HP_MULTICORE_DIR}
  AND NOT DEFINED ENV{CANN_VENDOR_FWD_LIBDIR}
  AND NOT DEFINED ENV{CANN_VENDOR_BWD_LIBDIR}
  AND NOT DEFINED ENV{CANN_VENDOR_LIBDIR})

    if(NOT IS_DIRECTORY "${_PREBUILD_DIR}" AND EXISTS "${_TARBALL}")
        message(STATUS "Extracting prebuild: ${_TARBALL}")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${_TARBALL}"
            WORKING_DIRECTORY "${_PREBUILD_ROOT}"
            RESULT_VARIABLE _extract_ret)
        if(NOT _extract_ret EQUAL 0)
            message(WARNING "Failed to extract ${_TARBALL} (exit ${_extract_ret})")
        endif()
    endif()

    if(IS_DIRECTORY "${_PREBUILD_DIR}/vendors")
        set(ENV{HP_MULTICORE_DIR} "${_PREBUILD_DIR}/vendors")
        message(STATUS "Auto-detected HP_MULTICORE_DIR: ${_PREBUILD_DIR}/vendors")
    endif()
endif()

# =============================================================================
# CANN Vendor Library Paths
# Priority: CANN_VENDOR_FWD/BWD_LIBDIR > HP_MULTICORE_DIR > CANN_VENDOR_LIBDIR
# =============================================================================

if(DEFINED ENV{HP_MULTICORE_DIR})
    if(DEFINED ENV{CANN_VENDOR_FWD_LIBDIR})
        set(CANN_VENDOR_FWD_LIBDIR "$ENV{CANN_VENDOR_FWD_LIBDIR}")
    else()
        set(CANN_VENDOR_FWD_LIBDIR "$ENV{HP_MULTICORE_DIR}/mega_moe_nn/op_api/lib")
    endif()
    if(DEFINED ENV{CANN_VENDOR_BWD_LIBDIR})
        set(CANN_VENDOR_BWD_LIBDIR "$ENV{CANN_VENDOR_BWD_LIBDIR}")
    else()
        set(CANN_VENDOR_BWD_LIBDIR "$ENV{HP_MULTICORE_DIR}/mega_moe_grad_nn/op_api/lib")
    endif()
elseif(DEFINED ENV{CANN_VENDOR_LIBDIR})
    set(CANN_VENDOR_FWD_LIBDIR "$ENV{CANN_VENDOR_LIBDIR}")
    set(CANN_VENDOR_BWD_LIBDIR "$ENV{CANN_VENDOR_LIBDIR}")
else()
    message(FATAL_ERROR
        "Cannot locate cust_opapi libraries. Set one of:\n"
        "  HP_MULTICORE_DIR  (vendors root; fwd/bwd paths derived automatically)\n"
        "  CANN_VENDOR_FWD_LIBDIR + CANN_VENDOR_BWD_LIBDIR  (explicit overrides)\n"
        "  CANN_VENDOR_LIBDIR  (legacy single-lib fallback)\n"
        "  Or place prebuild/mega_moe.tar.gz and re-run cmake.")
endif()

string(REGEX REPLACE "/op_api/lib/?$" "" CANN_VENDOR_HOME "${CANN_VENDOR_FWD_LIBDIR}")

message(STATUS "CANN_VENDOR_FWD_LIBDIR : ${CANN_VENDOR_FWD_LIBDIR}")
message(STATUS "CANN_VENDOR_BWD_LIBDIR : ${CANN_VENDOR_BWD_LIBDIR}")
message(STATUS "CANN_VENDOR_HOME       : ${CANN_VENDOR_HOME}")

# =============================================================================
# Source Files
# =============================================================================

add_subdirectory(c_api)
add_subdirectory(framework)

set(OPS_SRC_FILES ${C_API_SRC_FILES} ${FRAMEWORK_SRC_FILES})

# =============================================================================
# Find MindSpore
# =============================================================================

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_ms_path.cmake)

# =============================================================================
# Build flags
# =============================================================================

set(SRC_FILES  ${OPS_SRC_FILES})
set(LIB_DIRS
    ${CANN_VENDOR_FWD_LIBDIR}
    ${CANN_VENDOR_BWD_LIBDIR}
    ${CMAKE_CURRENT_BINARY_DIR}/../../lib/hyper_parallel/lib)
set(LIBS       cust_opapi)
set(INCLUDE_DIRS
    ${CANN_VENDOR_HOME}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MS_PATH}/include
    ${MS_PATH}/core/include)

message(STATUS "LIB_DIRS     : ${LIB_DIRS}")
message(STATUS "LIBS         : ${LIBS}")
message(STATUS "SRC_FILES    : ${SRC_FILES}")
message(STATUS "INCLUDE_DIRS : ${INCLUDE_DIRS}")

set(CFLAGS_INCLUDES "")
foreach(INC_DIR_ITEM ${INCLUDE_DIRS})
    if(CFLAGS_INCLUDES STREQUAL "")
        set(CFLAGS_INCLUDES "-I${INC_DIR_ITEM}")
    else()
        set(CFLAGS_INCLUDES "${CFLAGS_INCLUDES} -I${INC_DIR_ITEM}")
    endif()
endforeach()

set(LDLAGS_LIB_DIRS "")
foreach(LIB_DIR_ITEM ${LIB_DIRS})
    if(LDLAGS_LIB_DIRS STREQUAL "")
        set(LDLAGS_LIB_DIRS "-L${LIB_DIR_ITEM}")
    else()
        set(LDLAGS_LIB_DIRS "${LDLAGS_LIB_DIRS} -L${LIB_DIR_ITEM}")
    endif()
endforeach()

set(LDLAGS_LIBS "")
foreach(LIB_ITEM ${LIBS})
    if(LDLAGS_LIBS STREQUAL "")
        set(LDLAGS_LIBS "-l${LIB_ITEM}")
    else()
        set(LDLAGS_LIBS "${LDLAGS_LIBS} -l${LIB_ITEM}")
    endif()
endforeach()

message(STATUS "CFLAGS_INCLUDES : ${CFLAGS_INCLUDES}")
message(STATUS "LDLAGS_LIB_DIRS : ${LDLAGS_LIB_DIRS}")
message(STATUS "LDLAGS_LIBS     : ${LDLAGS_LIBS}")

# =============================================================================
# YAML Op Definition Files
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

set(ENABLE_DEBUG False)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(ENABLE_DEBUG True)
endif()

set(PYTHON_SCRIPT_PATH "${CMAKE_BINARY_DIR}/build_custom_with_ms.py")
file(WRITE ${PYTHON_SCRIPT_PATH} "
import mindspore as ms
src_files = '${SRC_FILES}'.split(';')
rpath_flag = (
    \"-Wl,-rpath,'$$ORIGIN'/../../../../lib\"
    \" -Wl,-rpath,${CANN_VENDOR_FWD_LIBDIR}\"
    \" -Wl,-rpath,${CANN_VENDOR_BWD_LIBDIR} \"
)
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

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_target(
    BuildCustomOp ALL
    COMMAND cd ${CMAKE_BINARY_DIR} && ${Python3_EXECUTABLE} ${PYTHON_SCRIPT_PATH}
    COMMENT "Building hyper_parallel_mega_moe_ms custom operator with MindSpore"
)

# =============================================================================
# Environment Setup Banner
# =============================================================================

string(REGEX REPLACE "/op_api/lib/?$" "" _FWD_ROOT "${CANN_VENDOR_FWD_LIBDIR}")
string(REGEX REPLACE "/op_api/lib/?$" "" _BWD_ROOT "${CANN_VENDOR_BWD_LIBDIR}")

message(STATUS "")
message(STATUS "================================================================")
message(STATUS "  hyper_parallel multicore: environment setup")
message(STATUS "================================================================")
message(STATUS "  Run the following in your terminal before importing:")
message(STATUS "")
message(STATUS "  export ASCEND_CUSTOM_OPP_PATH=${_FWD_ROOT}:\${ASCEND_CUSTOM_OPP_PATH}")
message(STATUS "  export LD_LIBRARY_PATH=${CANN_VENDOR_FWD_LIBDIR}:\${LD_LIBRARY_PATH}")
if(NOT "${_BWD_ROOT}" STREQUAL "${_FWD_ROOT}")
    message(STATUS "  export ASCEND_CUSTOM_OPP_PATH=${_BWD_ROOT}:\${ASCEND_CUSTOM_OPP_PATH}")
    message(STATUS "  export LD_LIBRARY_PATH=${CANN_VENDOR_BWD_LIBDIR}:\${LD_LIBRARY_PATH}")
endif()
message(STATUS "")
message(STATUS "  (Run 'source ~/.bashrc' or open a new terminal to apply)")
message(STATUS "================================================================")
message(STATUS "")

# =============================================================================
# Install
# =============================================================================

install(FILES ${BUILD_EXTENSION_DIR}/${MS_EXTENSION_NAME}.so
        DESTINATION ${MS_EXTENSION_NAME})
install(FILES
        ${BUILD_EXTENSION_DIR}/${MS_EXTENSION_NAME}_auto_generate/gen_ops_def.py
        ${BUILD_EXTENSION_DIR}/${MS_EXTENSION_NAME}_auto_generate/gen_ops_prim.py
        DESTINATION ${MS_EXTENSION_NAME})
