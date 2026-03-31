cmake_minimum_required(VERSION 3.18)
project(aclshmem_kernel)

# 设置C++标准
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
    message(FATAL_ERROR "Cannot find SHMEM_HOME_PATH, please run set_env.sh.")
else()
    set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})
endif()

option(USE_UNIT_TEST "USE_UNIT_TEST" OFF)
message(STATUS "USE_UNIT_TEST:${USE_UNIT_TEST}")

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

add_compile_options(
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    # avoid ascendc interference
    -DL2_CACHE_HINT
    -DTILING_KEY_VAR
)

option(ENABLE_HYBM_BACKEND "Enable HYBM backend" OFF)
if(ENABLE_HYBM_BACKEND)
    add_compile_definitions(BACKEND_HYBM)
endif()

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
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
    ${ASCEND_HOME_PATH}/include/aclnn
)

link_directories(
    ${ASCEND_HOME_PATH}/lib64
    ${SHMEM_HOME_PATH}/shmem/lib
)

link_libraries(runtime stdc++ ascendcl m tiling_api platform c_sec dl nnopbase)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/obj)

function(aclshmem_add_share_lib NAME)
    add_library(${NAME}_kernel OBJECT ${NAME}/kernel/${NAME}_kernel.cpp)
    set_target_properties(${NAME}_kernel PROPERTIES
        OBJECT_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/obj"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/obj"  # 兼容性保留
    )

    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-vec)
    target_include_directories(${NAME}_kernel PRIVATE
        ${SHMEM_HOME_PATH}/shmem/include
        ${SHMEM_HOME_PATH}/../src/device
    )

    add_library(${NAME}_shared SHARED ${NAME}/host/${NAME}.cpp)

    set_target_properties(${NAME}_shared PROPERTIES
        OBJECT_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/obj"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/obj"
    )

    target_compile_options(${NAME}_shared PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS} -D_GLIBCXX_USE_CXX11_ABI=0)
    target_include_directories(${NAME}_shared PRIVATE
        ${SHMEM_HOME_PATH}/shmem/include
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(${NAME}_shared hpshmem ${NAME}_kernel)

    set(GLOBAL_LIBS ${GLOBAL_LIBS} ${NAME}_kernel ${NAME}_shared PARENT_SCOPE)
endfunction()

foreach(EXAMPLE_KERNELS
    put_mem
    put_mem_signal
    get_mem
    signal_op
    signal_wait_until
)
    aclshmem_add_share_lib(${EXAMPLE_KERNELS})
endforeach()

add_library(aclshmem_symmetric_memory_kernel SHARED ./shmem_kernel.cpp)
target_link_options(aclshmem_symmetric_memory_kernel PRIVATE --cce-fatobj-link)

foreach(kernel_lib IN LISTS GLOBAL_LIBS)
    target_sources(aclshmem_symmetric_memory_kernel PRIVATE $<TARGET_OBJECTS:${kernel_lib}>)
endforeach()



if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set_target_properties(aclshmem_symmetric_memory_kernel PROPERTIES
            INSTALL_RPATH $ORIGIN:$ORIGIN/shmem
            )
    install(FILES ${CMAKE_BINARY_DIR}/libaclshmem_symmetric_memory_kernel.so DESTINATION lib)
    file(GLOB SHMEM_SO_FILES "${SHMEM_HOME_PATH}/shmem/lib/*")
    install(FILES ${SHMEM_SO_FILES} DESTINATION lib/shmem)
endif()
