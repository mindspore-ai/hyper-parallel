# cmake_minimum_required(VERSION 3.14)
# # 设置工程名
# project(ACLNN_EXAMPLE_GMM)

# # 设置C++编译标准
# add_compile_options(-std=c++11)

# # 设置编译输出目录为当前目录下的bin文件夹
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./")

# # 设置调试和发布模式的编译选项
# set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
# set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# set(TARGET_SUBDIR "multicore_moe_ffn_grad_nn")
# # set(TARGET_SUBDIR "custom_nn")
# set(SELF_OPERATOR_DIR "/home/z00797459/workspace/cann-ops-adv/ops-nn/install")


# # 添加可执行文件（请替换为实际算子可执行文件），指定算子调用的*.cpp文件
# add_executable(execute_test_op_gmm_grad
# main_gmm_swiglu_gmm_grad_data.cpp)

# # ASCEND_PATH（CANN软件包目录，请根据实际路径修改）
# if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
#     set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
# else()
#     set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest")
# endif()

# # 设置头文件路径
# set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")

# include_directories(
#     ${SELF_OPERATOR_DIR}/vendors/${TARGET_SUBDIR}/op_api/include    # 仅自定义算子需要
#     ${INCLUDE_BASE_DIR}
#     # ${INCLUDE_BASE_DIR}/aclnn                                   # 仅内置算子需要
# )

# # 链接所需的动态库
# target_link_libraries(execute_test_op_gmm_grad PRIVATE             # 替换实际算子可执行文件
#     ${ASCEND_PATH}/lib64/libascendcl.so
#     ${ASCEND_PATH}/lib64/libnnopbase.so
#     ${SELF_OPERATOR_DIR}/vendors/${TARGET_SUBDIR}/op_api/lib/libcust_opapi.so   # 仅自定义算子需要
#     # ${ASCEND_PATH}/lib64/libopapi.so    # 仅内置算子需要
# )

# # 安装目标文件到bin目录
# install(TARGETS execute_test_op_gmm_grad DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})



