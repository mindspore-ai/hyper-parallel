cmake_minimum_required(VERSION 3.18)
project(hyper_parallel_multicore_torch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT HP_MULTICORE_VENDOR_ROOT OR
    NOT EXISTS "${HP_MULTICORE_VENDOR_ROOT}/op_api/include")
    message(FATAL_ERROR
        "HP_MULTICORE_VENDOR_ROOT must identify a complete "
        "hyper_parallel_multicore_nn vendor")
endif()

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
set(OP_PLUGIN_ROOT "${TORCH_NPU_INCLUDE_ROOT}/third_party/op-plugin")
set(TORCH_LINK_LIBRARIES
    "${TORCH_PACKAGE_ROOT}/lib/libc10.so"
    "${TORCH_PACKAGE_ROOT}/lib/libtorch.so"
    "${TORCH_PACKAGE_ROOT}/lib/libtorch_cpu.so"
    "${TORCH_NPU_PACKAGE_ROOT}/lib/libtorch_npu.so")

foreach(required_path IN ITEMS
    ${Python3_INCLUDE_DIRS}
    "${TORCH_INCLUDE_ROOT}"
    "${TORCH_INCLUDE_ROOT}/torch/csrc/api/include"
    "${TORCH_NPU_INCLUDE_ROOT}"
    "${OP_PLUGIN_ROOT}/op_plugin/include/npu_cpp_extension.h"
    ${TORCH_LINK_LIBRARIES})
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR
            "The active Torch environment is missing a build input: "
            "${required_path}")
    endif()
endforeach()

add_library(hyper_parallel_mega_moe_torch SHARED
    csrc/registration.cpp
    csrc/mega_moe.cpp
    csrc/mega_moe_grad.cpp)
target_compile_definitions(hyper_parallel_mega_moe_torch PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
target_include_directories(hyper_parallel_mega_moe_torch PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_ROOT}
    ${TORCH_INCLUDE_ROOT}/torch/csrc/api/include
    ${TORCH_NPU_INCLUDE_ROOT}
    ${TORCH_NPU_INCLUDE_ROOT}/third_party/acl/inc
    ${OP_PLUGIN_ROOT}
    ${OP_PLUGIN_ROOT}/op_plugin/include
    ${HP_MULTICORE_VENDOR_ROOT}/op_api/include)
target_link_libraries(hyper_parallel_mega_moe_torch PRIVATE
    ${TORCH_LINK_LIBRARIES})
target_link_options(hyper_parallel_mega_moe_torch PRIVATE
    "-Wl,--enable-new-dtags")
set(MULTICORE_VENDOR_RUNPATH
    "$ORIGIN/../../vendors/hyper_parallel_multicore_nn/op_api/lib")
set_target_properties(hyper_parallel_mega_moe_torch PROPERTIES
    INSTALL_RPATH "${MULTICORE_VENDOR_RUNPATH}")

install(TARGETS hyper_parallel_mega_moe_torch DESTINATION .)
