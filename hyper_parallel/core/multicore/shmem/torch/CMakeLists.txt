cmake_minimum_required(VERSION 3.18)
project(aclshmem_torch)

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
    message(FATAL_ERROR
        "Cannot find SHMEM_HOME_PATH. Run the explicit native "
        "dependency/build workflow first.")
endif()
set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})

if(NOT ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB OR
  NOT EXISTS "${ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB}")
    message(FATAL_ERROR
        "ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB must name the kernel library "
        "installed by shmem/build.sh. Got: "
        "'${ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB}'")
endif()

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
    ${ASCEND_HOME_PATH}/include
)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

function(find_python_package_root package_name output_variable)
    string(CONCAT package_lookup_code
        "import importlib.util, sys; "
        "spec = importlib.util.find_spec(sys.argv[1]); "
        "locations = getattr(spec, 'submodule_search_locations', None) or []; "
        "print(next(iter(locations), ''))")
    execute_process(
        COMMAND ${Python3_EXECUTABLE}
                -c "${package_lookup_code}" "${package_name}"
        RESULT_VARIABLE package_lookup_status
        OUTPUT_VARIABLE package_root
        ERROR_VARIABLE package_lookup_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT package_lookup_status EQUAL 0 OR
      NOT IS_DIRECTORY "${package_root}")
        message(FATAL_ERROR
            "Cannot locate ${package_name} for ${Python3_EXECUTABLE}: "
            "${package_lookup_error}")
    endif()
    set(${output_variable} "${package_root}" PARENT_SCOPE)
endfunction()

find_python_package_root("torch" TORCH_PACKAGE_ROOT)
find_python_package_root("torch_npu" TORCH_NPU_PACKAGE_ROOT)
set(torch_abi_code
    "import torch; print(int(torch.compiled_with_cxx11_abi()))")
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env TORCH_DEVICE_BACKEND_AUTOLOAD=0
            ${Python3_EXECUTABLE} -c "${torch_abi_code}"
    RESULT_VARIABLE torch_abi_status
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    ERROR_VARIABLE torch_abi_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT torch_abi_status EQUAL 0 OR
  NOT TORCH_CXX11_ABI MATCHES "^[01]$")
    message(FATAL_ERROR
        "Cannot determine the active Torch C++ ABI: ${torch_abi_error}")
endif()

set(TORCH_INCLUDE_ROOT "${TORCH_PACKAGE_ROOT}/include")
set(TORCH_NPU_INCLUDE_ROOT "${TORCH_NPU_PACKAGE_ROOT}/include")
set(TORCH_LINK_LIBRARIES
    "${TORCH_PACKAGE_ROOT}/lib/libc10.so"
    "${TORCH_PACKAGE_ROOT}/lib/libtorch.so"
    "${TORCH_PACKAGE_ROOT}/lib/libtorch_cpu.so"
    "${TORCH_NPU_PACKAGE_ROOT}/lib/libtorch_npu.so")

foreach(required_path IN ITEMS
    ${Python3_INCLUDE_DIRS}
    "${TORCH_INCLUDE_ROOT}"
    "${TORCH_INCLUDE_ROOT}/torch/csrc/api/include"
    "${TORCH_NPU_INCLUDE_ROOT}/torch_npu/csrc/core/npu/NPUStream.h"
    ${TORCH_LINK_LIBRARIES})
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR
            "The active Torch environment is missing a build input: "
            "${required_path}")
    endif()
endforeach()

add_compile_definitions(
    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})

include_directories(
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_ROOT}
    ${TORCH_INCLUDE_ROOT}/torch/csrc/api/include
    ${TORCH_NPU_INCLUDE_ROOT}
    ${SHMEM_HOME_PATH}/shmem/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../ops/include
)

if(BUILD_TORCH_LIB)
    set(TORCH_BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/torch_bindings.cpp)
    add_library(aclshmem_torch
        SHARED ${TORCH_BINDINGS_SRC})
    target_link_libraries(
        aclshmem_torch PRIVATE
        ${TORCH_LINK_LIBRARIES}
        "${SHMEM_HOME_PATH}/shmem/lib/libhyper_parallel_shmem_utils.so"
        "${SHMEM_HOME_PATH}/shmem/lib/libhyper_parallel_shmem.so"
        "${ACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB}")
    target_link_options(aclshmem_torch PRIVATE "-Wl,--enable-new-dtags")
    set_target_properties(aclshmem_torch PROPERTIES
        INSTALL_RPATH $ORIGIN:$ORIGIN/../..:$ORIGIN/../../shmem)
    install(TARGETS aclshmem_torch DESTINATION . COMPONENT aclshmem_torch)
endif()
