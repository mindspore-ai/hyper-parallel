cmake_minimum_required(VERSION 3.18)
project(multicore_ms)

# mindspore link path: ABI must match mindspore (=0).
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

set(MS_EXTENSION_NAME "hyper_parallel_mega_moe_ms")
set(BUILD_EXTENSION_DIR "${CMAKE_BINARY_DIR}/lib")

# The operator build produces one package containing both forward and backward.
# Only the verified, explicit vendor root is accepted; this CMake project never
# extracts a repository tarball or mutates user shell configuration.
if(NOT HP_MULTICORE_VENDOR_ROOT AND DEFINED ENV{HP_MULTICORE_VENDOR_ROOT})
    set(HP_MULTICORE_VENDOR_ROOT "$ENV{HP_MULTICORE_VENDOR_ROOT}")
endif()
if(NOT HP_MULTICORE_VENDOR_ROOT)
    message(FATAL_ERROR "HP_MULTICORE_VENDOR_ROOT must identify hyper_parallel_multicore_nn")
endif()
set(CANN_VENDOR_LIBDIR "${HP_MULTICORE_VENDOR_ROOT}/op_api/lib")
if(NOT EXISTS "${CANN_VENDOR_LIBDIR}/libcust_opapi.so")
    message(FATAL_ERROR "Unified libcust_opapi.so is missing from ${CANN_VENDOR_LIBDIR}")
endif()

message(STATUS "HP_MULTICORE_VENDOR_ROOT : ${HP_MULTICORE_VENDOR_ROOT}")
message(STATUS "CANN_VENDOR_LIBDIR       : ${CANN_VENDOR_LIBDIR}")

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
set(LIB_DIRS ${CANN_VENDOR_LIBDIR})
# LAUNCH_ACLNN_FUNC resolves aclnnHyperMegaMoe* through MindSpore's
# GetOpApiFunc at runtime. Directly linking the generic libcust_opapi.so SONAME
# would allow another installed custom vendor to capture this dependency.
set(LIBS)
set(INCLUDE_DIRS
    ${HP_MULTICORE_VENDOR_ROOT}/op_api/include
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

# Propagate ABI=0 to the CustomOpBuilder-driven g++ compilation. The
# add_compile_definitions() at file top only affects targets compiled by
# CMake itself; this line is what actually reaches the .so.
set(CFLAGS_INCLUDES "${CFLAGS_INCLUDES} -D_GLIBCXX_USE_CXX11_ABI=0")

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


class HyperParallelCustomOpBuilder(ms.ops.CustomOpBuilder):
    def get_ldflags(self):
        flags = super().get_ldflags()
        broken_rpath_flag = '-Wl,--disable-new-dtags,--rpath'
        return [flag for flag in flags if flag != broken_rpath_flag]


src_files = '${SRC_FILES}'.split(';')
rpath_flag = \"-Wl,--enable-new-dtags,-rpath,'$$ORIGIN'/../../vendors/hyper_parallel_multicore_nn/op_api/lib \"
HyperParallelCustomOpBuilder(
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
    COMMAND cd ${CMAKE_BINARY_DIR} && ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS
            ${Python3_EXECUTABLE} ${PYTHON_SCRIPT_PATH}
    COMMENT "Building hyper_parallel_mega_moe_ms custom operator with MindSpore"
)

# =============================================================================
# Install
# =============================================================================

install(FILES ${BUILD_EXTENSION_DIR}/${MS_EXTENSION_NAME}.so
        DESTINATION ${MS_EXTENSION_NAME})
install(FILES
        ${BUILD_EXTENSION_DIR}/${MS_EXTENSION_NAME}_auto_generate/gen_ops_def.py
        ${BUILD_EXTENSION_DIR}/${MS_EXTENSION_NAME}_auto_generate/gen_ops_prim.py
        DESTINATION ${MS_EXTENSION_NAME})
