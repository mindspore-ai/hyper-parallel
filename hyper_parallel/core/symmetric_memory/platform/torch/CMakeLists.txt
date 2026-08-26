cmake_minimum_required(VERSION 3.18)
project(aclshmem_torch)

# torch link path: ABI must match PyTorch / torch_npu (=1, official default since 2.7).
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

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
    message(FATAL_ERROR "Cannot find SHMEM_HOME_PATH. Run the explicit native dependency/build workflow first.")
endif()
set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})

if(NOT ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB OR
  NOT EXISTS "${ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB}")
    message(FATAL_ERROR
        "ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB must name the kernel library installed by "
        "build_symmetric_memory.sh. Got: '${ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB}'")
endif()

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

add_compile_options(
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    # avoid ascendc interference
    -DL2_CACHE_HINT
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
)

if(NOT DEFINED Python3_EXECUTABLE)
    message(WARNING "Python3_EXECUTABLE is not defined, using default python3")
else()
    message("Using python: ${Python3_EXECUTABLE}")
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "${Python3_SITELIB}")
set(Torch_DIR ${Python3_SITELIB}/torch/share/cmake/Torch)
find_package(Torch REQUIRED)

include_directories(
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_SITELIB}/torch_npu/include
    ${SHMEM_HOME_PATH}/shmem/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ops/include
)

link_directories(
    ${Python3_SITELIB}/torch/lib
    ${Python3_SITELIB}/torch_npu/lib
    ${SHMEM_HOME_PATH}/shmem/lib
)

link_libraries("${TORCH_LIBRARIES}" torch_npu hyper_parallel_shmem)

if(BUILD_TORCH_LIB)
    set(TORCH_BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/torch_bindings.cpp)
    add_library(aclshmem_torch
        SHARED ${TORCH_BINDINGS_SRC})
    target_link_libraries(
        aclshmem_torch PRIVATE
        hyper_parallel_shmem_utils
        hyper_parallel_shmem
        "${ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB}")
    set_target_properties(aclshmem_torch PROPERTIES
        INSTALL_RPATH $ORIGIN:$ORIGIN/../..:$ORIGIN/../../shmem)
    install(TARGETS aclshmem_torch DESTINATION . COMPONENT aclshmem_torch)
endif()
