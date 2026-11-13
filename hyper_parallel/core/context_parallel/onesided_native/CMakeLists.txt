cmake_minimum_required(VERSION 3.19)
project(hp_state_pull LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_program(BISHENG bisheng REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(CANN "$ENV{ASCEND_HOME_PATH}")
set(SHMEM_SDK "${SHMEM_SDK}" CACHE PATH "Existing compatible Ascend SHMEM SDK")
if(NOT CANN OR NOT SHMEM_SDK)
    message(FATAL_ERROR "Source CANN set_env.sh and provide -DSHMEM_SDK=<SDK root>")
endif()
find_library(SHMEM_LIBRARY NAMES hyper_parallel_shmem shmem
    PATHS "${SHMEM_SDK}/lib" NO_DEFAULT_PATH REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import torch,torch_npu,pathlib; print(pathlib.Path(torch.__file__).parent); print(pathlib.Path(torch_npu.__file__).parent); print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_PATHS OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" TORCH_PATHS "${TORCH_PATHS}")
list(GET TORCH_PATHS 0 TORCH_ROOT)
list(GET TORCH_PATHS 1 NPU_ROOT)
list(GET TORCH_PATHS 2 TORCH_ABI)
set(INCS
    -I${CANN}/include
    -I${CANN}/compiler/tikcpp
    -I${CANN}/compiler/tikcpp/tikcfw
    -I${CANN}/compiler/tikcpp/tikcfw/impl
    -I${CANN}/compiler/tikcpp/tikcfw/interface
    -I${SHMEM_SDK}/include
    -I${SHMEM_SDK}/src/device)
# This payload is explicitly A2; accepting another target would imply untested support.
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/device.o
    COMMAND ${BISHENG} -xcce -c ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
        -o ${CMAKE_CURRENT_BINARY_DIR}/device.o -O2 -fPIC -std=c++17
        --cce-aicore-arch=dav-c220-vec -DL2_CACHE_HINT -DTILING_KEY_VAR ${INCS}
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
    DEPENDS device.cpp)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libhp_state_pull_device.so
    COMMAND ${BISHENG} --cce-fatobj-link -shared ${CMAKE_CURRENT_BINARY_DIR}/device.o
        -o ${CMAKE_CURRENT_BINARY_DIR}/libhp_state_pull_device.so
        ${SHMEM_LIBRARY} -L${CANN}/lib64
        -lascendcl -lruntime -lstdc++ -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/device.o)
add_custom_target(device_lib DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libhp_state_pull_device.so)
add_library(_hp_state_pull MODULE binding.cpp)
add_dependencies(_hp_state_pull device_lib)
target_compile_definitions(_hp_state_pull PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_ABI})
target_compile_options(_hp_state_pull PRIVATE -O1)
target_include_directories(_hp_state_pull PRIVATE
    ${CANN}/include ${Python3_INCLUDE_DIRS} ${TORCH_ROOT}/include
    ${TORCH_ROOT}/include/torch/csrc/api/include ${NPU_ROOT}/include ${SHMEM_SDK}/include)
target_link_libraries(_hp_state_pull PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/libhp_state_pull_device.so
    ${TORCH_ROOT}/lib/libtorch_python.so ${TORCH_ROOT}/lib/libtorch.so
    ${TORCH_ROOT}/lib/libtorch_cpu.so ${TORCH_ROOT}/lib/libc10.so
    ${NPU_ROOT}/lib/libtorch_npu.so ${SHMEM_LIBRARY} ${CANN}/lib64/libascendcl.so)
set_target_properties(_hp_state_pull PROPERTIES PREFIX ""
    BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${SHMEM_SDK}/lib;${CANN}/lib64;${TORCH_ROOT}/lib;${NPU_ROOT}/lib")
