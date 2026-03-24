cmake_minimum_required(VERSION 3.18)
project(aclshmem_torch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 生成位置无关代码
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 获取CANN相关环境变量
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

# 获取aclshmem相关环境变量
if(NOT DEFINED ENV{SHMEM_HOME_PATH})
    if(NOT DEFINED ENV{HP_THIRD_PARTY_DIR})
        message(FATAL_ERROR "Cannot find SHMEM from source or prebuild, please check log.")
    else()
        set(SHMEM_HOME_PATH $ENV{HP_THIRD_PARTY_DIR})
    endif()
else()
    set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})
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
    ${CMAKE_BINARY_DIR}
    ${Python3_SITELIB}/torch/lib
    ${Python3_SITELIB}/torch_npu/lib
    ${SHMEM_HOME_PATH}/shmem/lib
    ${CMAKE_CURRENT_BINARY_DIR}/../ops
    ${CMAKE_CURRENT_BINARY_DIR}/../../lib/hyper_parallel/lib
    ${CMAKE_CURRENT_BINARY_DIR}/../../lib/hyper_parallel/lib/shmem
)

link_libraries(aclshmem_symmetric_memory_kernel "${TORCH_LIBRARIES}" torch_npu hpshmem)

if(BUILD_TORCH_LIB)
    set(TORCH_BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/torch_bindings.cpp)
    add_library(aclshmem_torch
        SHARED ${TORCH_BINDINGS_SRC})
    target_link_libraries(aclshmem_torch PRIVATE config_store hybm_core hpshmem aclshmem_symmetric_memory_kernel)
    set_target_properties(aclshmem_torch PROPERTIES
        INSTALL_RPATH $ORIGIN:$ORIGIN/../../../lib:$ORIGIN/../../../lib/shmem)
    install(TARGETS aclshmem_torch DESTINATION . COMPONENT aclshmem_torch)
endif()