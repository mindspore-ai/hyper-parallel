/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file main.cpp
 */
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>

#include "tiling_context.hpp"

#include "acl/acl.h"
// #include "aclnn_swi_glu.h"
// #include "aclnn_add_custom.h"
#include "aclnn_multicore_moe_ffn.h"

#include "runtime_head.hpp"

#include "tiling_data_pre_process.hpp"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void) close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}
int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int CreateAclTensorToTilingData(const uint8_t* hostData, size_t size, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  uint64_t shape_size = 1;
  std::cout<<"t1_size shape_value: start" <<std::endl;
  int64_t size_data = static_cast<int64_t>(size);
  int64_t* size_ptr = &size_data;
  std::cout<<"t1_size size_ptr:" << *size_ptr <<std::endl;

  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  int64_t strides_value = 1;
  int64_t *strides_ptr = &strides_value;

  *tensor = aclCreateTensor(size_ptr, shape_size, dataType, strides_ptr, 0, aclFormat::ACL_FORMAT_ND,
                            size_ptr, shape_size, *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 7;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> xShape = {8192, 4096};
  std::vector<int64_t> inputYShape = {8192, 2048};
  std::vector<int64_t> outShape = {8192, 2048};
  std::vector<int64_t> outputZShape = {8192, 2048};
  int64_t first_shape = 2048;
  int64_t second_shape = 4096;
  int64_t split_value = 64;
  int task_num = 128;

  void* xDeviceAddr = nullptr;
  void *inputYDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void *outputZDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor *inputY = nullptr;
  aclTensor* out = nullptr;
  aclTensor *outputZ = nullptr;
  size_t inputYShapeSize = inputYShape[0] * inputYShape[1];
  size_t outputZShapeSize = outputZShape[0] * outputZShape[1];
  std::vector<aclFloat16> xHostData(xShape[0]*xShape[1], 1);
  std::vector<aclFloat16> inputYHostData(inputYShape[0] * inputYShape[1], 1);
  std::vector<aclFloat16> outHostData(outShape[0]*outShape[1], 1);
  std::vector<aclFloat16> outputZHostData(outputZShape[0] * outputZShape[1]);
  size_t dtypeSize = sizeof(aclFloat16);

  aclTensor *t1 = nullptr;
  void* t1DeviceAddr = nullptr;
  TilingData tilingData;
  get_tiling_data_add(tilingData, split_value);
  uint8_t* t1_ptr = reinterpret_cast<uint8_t*>(&tilingData);
  size_t t1_size = sizeof(TilingData);
  std::cout<<"multicore_moe_ffn t1_size:" << t1_size<<std::endl;
  ret = CreateAclTensorToTilingData(t1_ptr, t1_size, &t1DeviceAddr, aclDataType::ACL_UINT8, &t1);

  aclTensor *t2 = nullptr;
  void* t2DeviceAddr = nullptr;
  SwiGluTilingData swiGluTilingData;
  get_tiling_data_swiglu(swiGluTilingData, split_value);
  uint8_t* t2_ptr = reinterpret_cast<uint8_t*>(&swiGluTilingData);
  size_t t2_size = sizeof(SwiGluTilingData);
  std::cout<<"multicore_moe_ffn t2_size:" << t2_size<<std::endl;
  ret = CreateAclTensorToTilingData(t2_ptr, t2_size, &t2DeviceAddr, aclDataType::ACL_UINT8, &t2);

//   std::vector<TaskDesc> all_tasks;
  RuntimeConfig runtimeConfig;
  init_runtime_config(runtimeConfig, NUM_WORKERS);
  for (int i = 0; i < task_num; i++) {
    TensorDesc inputs[MAX_INPUTS_PER_TASK];
    TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
    uint32_t input_len = 1;
    uint32_t output_len = 1;
    for (uint32_t j = 0; j < input_len; j++) {
        TensorDesc tensorDesc;
        tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
        tensorDesc.input_position = 0; //x
        tensorDesc.base_ptr_offset = i * second_shape * split_value * 1;
        inputs[j] = tensorDesc;
    }
    for (uint32_t j = 0; j < output_len; j++) {
        TensorDesc tensorDesc;
        tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
        tensorDesc.input_position = 2; // out
        tensorDesc.base_ptr_offset = i * first_shape * split_value * 1;
        outputs[j] = tensorDesc;
    }
    TaskDesc taskDesc = register_swi_glu_task(inputs, input_len, outputs, output_len);
    taskDesc.trigger_event = i + 1;
    taskDesc.dependent_event = -1 + 1;

    int chu_all = task_num / NUM_WORKERS;
    int remain_all = task_num % NUM_WORKERS;

    int chu = i / NUM_WORKERS;
    int remain = i % NUM_WORKERS;
    if (chu < chu_all){
      runtimeConfig.all_tasks[(chu*2) * NUM_WORKERS +remain] = taskDesc;
    } else {
      runtimeConfig.all_tasks[(chu*2) * NUM_WORKERS +remain] = taskDesc;
    }
  }
  for (int i = 0; i < task_num; i++) {
    TensorDesc inputs[MAX_INPUTS_PER_TASK];
    TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
    uint32_t input_len = 2;
    uint32_t output_len = 1;
    for (uint32_t j = 0; j < input_len; j++) {
        TensorDesc tensorDesc;
        tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
        if (j == 0) {
          tensorDesc.input_position = 2; //x
        } else {
          tensorDesc.input_position = 1; //x
        }
        tensorDesc.base_ptr_offset = i * first_shape * split_value * 1;
        inputs[j] = tensorDesc;
    }
    for (uint32_t j = 0; j < output_len; j++) {
        TensorDesc tensorDesc;
        tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
        tensorDesc.input_position = 5; // out
        tensorDesc.base_ptr_offset = i * first_shape * split_value * 1;
        outputs[j] = tensorDesc;
    }
    TaskDesc taskDesc = register_add_custom_task(inputs, input_len, outputs, output_len);
    taskDesc.trigger_event = task_num + 1; //-1;
    taskDesc.dependent_event = i + 1;

    int chu_all = task_num / NUM_WORKERS;
    int remain_all = task_num % NUM_WORKERS;

    int chu = i / NUM_WORKERS;
    int remain = i % NUM_WORKERS;

    if (chu < chu_all){
      runtimeConfig.all_tasks[(chu*2+1) * NUM_WORKERS +remain] = taskDesc;
    } else {
      runtimeConfig.all_tasks[(chu*2) * NUM_WORKERS +remain_all + remain] = taskDesc;
    }
  }

  TaskDesc taskDesc;
  taskDesc.task_type = TaskType::TASK_TERMINATE;
  runtimeConfig.all_tasks[task_num*2] = taskDesc;

  // for (int i = 0; i < MAX_TASK_NUM; i++) {
  //   LOG_PRINT("zgp task num idx %ld. \n", i);
  //   LOG_PRINT("zgp task num all_tasks %ld is: %d\n", i, runtimeConfig.all_tasks[i]);
  //   LOG_PRINT("zgp task num all_tasks %ld trigger_event is: %d\n", i, runtimeConfig.all_tasks[i].trigger_event);
  //   LOG_PRINT("zgp task num all_tasks %ld dependent_event is: %d\n", i, runtimeConfig.all_tasks[i].dependent_event);
  // }

//   std::vector<EventDesc> all_events;
  EventDesc event_desc_0;
  event_desc_0.num_triggers = task_num;
  event_desc_0.first_task_id = 0;
  event_desc_0.last_task_id = 0 + task_num;
  event_desc_0.event_type = EventType::EVENT_LAUNCH_TASKS;
  // all_events.emplace_back(event_desc_0);
  runtimeConfig.all_events[0] = event_desc_0;
  runtimeConfig.all_event_num_triggers[0] = 0;

  for (int i = 1; i < task_num + 1; i++) {
    EventDesc event_desc;
    event_desc.num_triggers = 1;
    event_desc.first_task_id = i - 1 + task_num;
    event_desc.last_task_id = i - 1 + task_num + 1;
    event_desc.event_type = EventType::EVENT_LAUNCH_TASKS;
    // all_events.emplace_back(event_desc);
    runtimeConfig.all_events[i] = event_desc;
    runtimeConfig.all_event_num_triggers[i] = 1;
  }
  EventDesc event_desc_1;
  event_desc_1.event_type = EventType::EVENT_END_OF_TASK_GRAPH;
  event_desc_1.num_triggers = 1;
  event_desc_1.first_task_id = task_num * 2;
  event_desc_1.last_task_id = task_num * 2 + 1;
  runtimeConfig.all_events[task_num+1] = event_desc_1;
  runtimeConfig.all_event_num_triggers[task_num + 1] = task_num;

  runtimeConfig.sched_queue[0] = 0;
  runtimeConfig.task_num = task_num * 2;

  // for (int i = 0; i < MAX_EVENT_NUM; i++) {
  //   LOG_PRINT("zgp event all_events %ld is: %d\n", i, runtimeConfig.all_events[i].event_type);
  // }

  uint8_t* runtimeConfig_ptr = reinterpret_cast<uint8_t*>(&runtimeConfig);
  size_t runtimeConfig_size = sizeof(RuntimeConfig);
  std::cout<<"multicore_moe_ffn runtimeConfig_size:" << runtimeConfig_size<<std::endl;
  std::cout<<"main code:multicore_moe_ffn runtimeConfig.sched_queue_last_pos:" << runtimeConfig.sched_queue_last_pos<<std::endl;
  std::cout<<"main code:multicore_moe_ffn runtimeConfig. 62008 data value:" << (*(uint32_t *)(runtimeConfig_ptr + 4 * sizeof(uint32_t)))<<std::endl;
  aclTensor *runtimeConfig_tensor = nullptr;
  void* runtimeConfigAddr = nullptr;
  ret = CreateAclTensorToTilingData(runtimeConfig_ptr, runtimeConfig_size, &runtimeConfigAddr, aclDataType::ACL_UINT8, &runtimeConfig_tensor);

  int dimOptional = -1;
  void ** inputX = (void **)(&xHostData);
  void ** input2 = (void **)(&inputYHostData);
  void ** out_1 = (void **)(&outHostData);

  INFO_LOG("all start 0");
  size_t fileSize;
  size_t xShapeSize = GetShapeSize(xShape);
  ReadFile("/home/z00797459/workspace/cann-ops-adv/cann-ops/src/activation/mega_kernel/examples/AclNNInvocationNaive_multicore_swiglu_add_task/input/input_x.bin", fileSize, *inputX, xShapeSize * dtypeSize);
  ReadFile("/home/z00797459/workspace/cann-ops-adv/cann-ops/src/activation/mega_kernel/examples/AclNNInvocationNaive_multicore_swiglu_add_task/input/input_y.bin", fileSize, *input2, inputYShapeSize * dtypeSize);

  // ReadFile("/home/z00797459/workspace/cann-ops-adv/cann-ops/src/activation/swi_glu/examples/AclNNInvocationNaive/output/output_out.bin", fileSize, *out_1, outputZShapeSize * dtypeSize);




  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  INFO_LOG("swiglu start 0");
  ret = CreateAclTensor(inputYHostData, inputYShape, &inputYDeviceAddr, aclDataType::ACL_FLOAT16, &inputY);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  INFO_LOG("swiglu start 1");
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  INFO_LOG("swiglu start 2");

  std::cout<<"multicore_moe_ffn xDeviceAddr:" << xDeviceAddr<<std::endl;
  std::cout<<"multicore_moe_ffn inputYDeviceAddr:" << inputYDeviceAddr<<std::endl;
  std::cout<<"multicore_moe_ffn outDeviceAddr:" << outDeviceAddr<<std::endl;

  std::cout<<"multicore_moe_ffn xDeviceAddr 1:" << &xDeviceAddr<<std::endl;
  std::cout<<"multicore_moe_ffn inputYDeviceAddr 1:" << &inputYDeviceAddr<<std::endl;
  std::cout<<"multicore_moe_ffn outDeviceAddr 1:" << &outDeviceAddr<<std::endl;

  ret = CreateAclTensor(outputZHostData, outputZShape, &outputZDeviceAddr, aclDataType::ACL_FLOAT16, &outputZ);
  CHECK_RET(ret == ACL_SUCCESS, return FAILED);
  INFO_LOG("add start 0");
  uint64_t workspaceSize_1 = 0;
  aclOpExecutor* executor_1;
  // ret = aclnnAddCustomGetWorkspaceSize(out, inputY, outputZ, &workspaceSize_1, &executor_1);
  ret = aclnnMulticoreMoeFfnGetWorkspaceSize(x, inputY, out, t1, t2, runtimeConfig_tensor, outputZ, &workspaceSize_1, &executor_1);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulticoreMoeFfnGetWorkspaceSize failed. ERROR: %d\n", ret); return FAILED);
  void *workspaceAddr_1 = nullptr;
  INFO_LOG("add start 1");
  if (workspaceSize_1 > 0) {
    ret = aclrtMalloc(&workspaceAddr_1, workspaceSize_1, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return FAILED;);
  }
  INFO_LOG("add start 2");
  ret = aclnnMulticoreMoeFfn(workspaceAddr_1, workspaceSize_1, executor_1, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulticoreMoeFfn failed. ERROR: %d\n", ret); return FAILED);
  INFO_LOG("add start 3");
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);
  INFO_LOG("add start 4");

  // // +++++++++++++++++++++++++++++++++
  // ret = aclnnMulticoreMoeFfnGetWorkspaceSize(x, inputY, out, t1, t2, runtimeConfig_tensor, outputZ, &workspaceSize_1, &executor_1);
  // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulticoreMoeFfnGetWorkspaceSize failed. ERROR: %d\n", ret); return FAILED);
  // workspaceAddr_1 = nullptr;
  // INFO_LOG("add start 1");
  // if (workspaceSize_1 > 0) {
  //   ret = aclrtMalloc(&workspaceAddr_1, workspaceSize_1, ACL_MEM_MALLOC_HUGE_FIRST);
  //   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return FAILED;);
  // }
  // INFO_LOG("add start 2");
  // ret = aclnnMulticoreMoeFfn(workspaceAddr_1, workspaceSize_1, executor_1, stream);
  // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulticoreMoeFfn failed. ERROR: %d\n", ret); return FAILED);
  // INFO_LOG("add start 3");
  // ret = aclrtSynchronizeStream(stream);
  // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);
  // INFO_LOG("add start 4");
  // // ++++++++++++++++++++++++++++++++++++++

  auto size = GetShapeSize(outputZShape);
  std::vector<aclFloat16> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputZDeviceAddr,
                      size * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
  void ** output1=(void **)(&resultData);
//   for (int64_t i = 0; i < size; i++) {
//     LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
//   }

  size_t dataType = sizeof(uint16_t);
  WriteFile("../output/output_swiglu_addcustom_onetest.bin", *output1, outputZShapeSize * dataType);
  INFO_LOG("Write output success");


  // std::vector<int32_t> resultData1(100, 0);
  // uint8_t *outputZDeviceAddrN = reinterpret_cast<uint8_t *>(outputZDeviceAddr);
  // ret = aclrtMemcpy(resultData1.data(), resultData1.size() * sizeof(resultData1[0]), outputZDeviceAddrN + 2*sizeof(uint32_t),
  //                     100 * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
  // for (int i = 0; i < 100; i++) {
  //   LOG_PRINT("zgp result[%ld] is: %d\n", i, resultData1[i]);
  // }

  aclDestroyTensor(x);
  aclDestroyTensor(inputY);
  aclDestroyTensor(out);
  aclDestroyTensor(outputZ);
  aclrtFree(xDeviceAddr);
  aclrtFree(inputYDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(outputZDeviceAddr);
  if (workspaceSize_1 > 0) {
    aclrtFree(workspaceAddr_1);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}