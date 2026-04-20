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
#include <sstream>

#include "tiling_context.hpp"
#include "task_register.hpp"

#include "acl/acl.h"
// #include "aclnn_swi_glu.h"
// #include "aclnn_add_custom.h"
#include "aclnn_multicore_moe_ffn_grad.h"

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

std::vector<uint32_t> splitStringToUint32(const std::string& input) {
    std::vector<uint32_t> result;
    std::stringstream ss(input);
    std::string token;

    while (std::getline(ss, token, ',')) {
        try {
            result.push_back(static_cast<uint32_t>(std::stoul(token)));
        } catch (const std::exception& e) {
            std::cerr << "转换错误: " << token << " - " << e.what() << std::endl;
        }
    }

    return result;
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

template <typename T>
int CreateAclTensor_New(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
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

template <typename T>
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<T> hostData(size, 2);
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


int CreateAclTensorList(const std::vector<aclFloat16>& hostData, const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr,
                        aclDataType dataType, aclTensorList** tensor) {
    int size = shapes.size();
    aclTensor* tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor_New<aclFloat16>(hostData, shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}


template <typename T>
int CreateAclTensor_NewTrans(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
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
    std::vector<int64_t> strides_new(shape.size(), 1);
    std::vector<int64_t> shape_new(shape.size(), 1);
    if (shape.size() == 3) {
      strides_new[1] = strides[2];
      strides_new[2] = strides[1];
      shape_new[1] = shape[2];
      shape_new[2] = shape[1];
    } else {
      strides_new[0] = strides[1];
      strides_new[1] = strides[0];
      shape_new[0] = shape[1];
      shape_new[1] = shape[0];
    }

    *tensor = aclCreateTensor(shape_new.data(), shape_new.size(), dataType, strides_new.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int CreateAclTensorListTrans(const std::vector<aclFloat16>& hostData, const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr,
                        aclDataType dataType, aclTensorList** tensor) {
    int size = shapes.size();
    aclTensor* tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor_NewTrans<aclFloat16>(hostData, shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
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

int CreateAclTensorToWorkspace(size_t size, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  uint64_t shape_size = 1;
  std::cout<<"t1_size shape_value: start" <<std::endl;
  int64_t size_data = static_cast<int64_t>(size);
  int64_t* size_ptr = &size_data;
  std::cout<<"t1_size size_ptr:" << *size_ptr <<std::endl;

  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  int64_t strides_value = 1;
  int64_t *strides_ptr = &strides_value;

  *tensor = aclCreateTensor(size_ptr, shape_size, dataType, strides_ptr, 0, aclFormat::ACL_FORMAT_ND,
                            size_ptr, shape_size, *deviceAddr);
  return 0;
}

template <typename T>
int write_data_to_file_all(std::vector<int64_t>& shape, void* deviceAddr, std::string filename) {
  auto gmmxG2_size = GetShapeSize(shape);
  std::vector<T> gmmxG1_resultData(gmmxG2_size, 0);
  auto ret = aclrtMemcpy(gmmxG1_resultData.data(), gmmxG1_resultData.size() * sizeof(gmmxG1_resultData[0]), deviceAddr,
                      gmmxG2_size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
  void ** gmmxG1_output1=(void **)(&gmmxG1_resultData);
  WriteFile(filename, *gmmxG1_output1, gmmxG2_size * sizeof(T));
  INFO_LOG("Write output success");
}

template <typename T>
int write_data_to_file_all(int64_t gmmxG2_size, void* deviceAddr, std::string filename) {
  std::vector<T> gmmxG1_resultData(gmmxG2_size, 0);
  auto ret = aclrtMemcpy(gmmxG1_resultData.data(), gmmxG1_resultData.size() * sizeof(gmmxG1_resultData[0]), deviceAddr,
                      gmmxG2_size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return FAILED);
  void ** gmmxG1_output1=(void **)(&gmmxG1_resultData);
  WriteFile(filename, *gmmxG1_output1, gmmxG2_size * sizeof(T));
  INFO_LOG("Write output success");
}

int main() {
  int32_t deviceId = 7;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  size_t dtypeSize = sizeof(aclFloat16);
  // GM_ADDR target, GM_ADDR target_offset, GM_ADDR src, GM_ADDR src_offset, GM_ADDR size

  TaskSplitValue taskSplitValue;
  // ++++++++++++++++++++++++++++++==========================================================================A1
  std::vector<int64_t> targetShape = {taskSplitValue.per_rank_seq, 7168};
  std::vector<int64_t> targetOffsetShape = {taskSplitValue.all_expert_num};
  std::vector<int64_t> srcShape = {taskSplitValue.per_rank_seq, 7168};
  std::vector<int64_t> srcOffsetShape = {taskSplitValue.all_expert_num};
  std::vector<int64_t> sizeShape = {taskSplitValue.all_expert_num};
  void* targetDeviceAddr;
  void* targetOffsetDeviceAddr;
  void* srcDeviceAddr;
  void* srcOffsetDeviceAddr;
  void* sizeDeviceAddr;
  aclTensor* target = nullptr;
  aclTensor* targetOffset = nullptr;
  aclTensor* src = nullptr;
  aclTensor* srcOffset = nullptr;
  aclTensor* sizeA1 = nullptr;
  std::vector<aclFloat16> targetHostData(targetShape[0]*targetShape[1], 1);
  std::vector<int64_t> targetOffsetHostData(targetOffsetShape[0], 1);
  std::vector<aclFloat16> srcHostData(srcShape[0]*srcShape[1]);
  std::vector<int64_t> srcOffsetHostData(srcOffsetShape[0]);
  std::vector<int32_t> sizeHostData(sizeShape[0]);
  ret = CreateAclTensor_New<aclFloat16>(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_BF16, &target);
  ret = CreateAclTensor_New<int64_t>(targetOffsetHostData, targetOffsetShape, &targetOffsetDeviceAddr, aclDataType::ACL_INT64, &targetOffset);
  ret = CreateAclTensor_New<aclFloat16>(srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_BF16, &src);
  ret = CreateAclTensor_New<int64_t>(srcOffsetHostData, srcOffsetShape, &srcOffsetDeviceAddr, aclDataType::ACL_INT64, &srcOffset);
  ret = CreateAclTensor_New<int32_t>(sizeHostData, sizeShape, &sizeDeviceAddr, aclDataType::ACL_INT32, &sizeA1);
  taskSplitValue.alltoall_split_value = 128;
  taskSplitValue.alltoall_task_num = taskSplitValue.all_expert_num * ((int)(taskSplitValue.per_expert_seq_to_other/taskSplitValue.alltoall_split_value));
  std::cout<<"zgp debug, main_gmm_swiglu_gmm here 0!"<<std::endl;

  // ++++++++++++++++++++++++++++++==========================================================================GMM1
  // "4096,7168;2048,7168"
  std::vector<std::vector<int64_t>> xShape = {{taskSplitValue.per_rank_seq, 7168}};
  std::vector<std::vector<int64_t>> weightShape= {{taskSplitValue.single_rank_expert_num, 2048, 7168}};
  std::vector<std::vector<int64_t>> yShape = {{taskSplitValue.per_rank_seq, 2048}};
  std::vector<int64_t> groupListShape = {{taskSplitValue.single_rank_expert_num}};
  std::vector<int64_t> groupListData(taskSplitValue.single_rank_expert_num, 1);
  void* xDeviceAddr[1];
  void* weightDeviceAddr[1];
  void* biasDeviceAddr[1];
  void* yDeviceAddr[1];
  void* groupListDeviceAddr;
  aclTensorList* x = nullptr;
  aclTensorList* weight = nullptr;
  aclTensor* groupList = nullptr;
  aclTensorList* y = nullptr;
  int64_t splitItem = 3;
  int64_t groupType = 0;
  std::vector<aclFloat16> xHostData(xShape[0][0]*xShape[0][1], 1);
  std::vector<aclFloat16> weightHostData(weightShape[0][0] * weightShape[0][1] * weightShape[0][2], 1);
  std::vector<aclFloat16> yHostData(yShape[0][0]*yShape[0][1]);
  // size_t fileSize;
  // void ** inputX = (void **)(&xHostData);
  // void ** input2 = (void **)(&weightHostData);
  // ReadFile("../input/input_gmm_x_512.bin", fileSize, *inputX, GetShapeSize(xShape[0]) * dtypeSize);
  // ReadFile("../input/input_gmm_weight_512.bin", fileSize, *input2, GetShapeSize(weightShape[0]) * dtypeSize);
  ret = CreateAclTensorList(xHostData, xShape, xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::cout<<"zgp debug, main_gmm_swiglu_gmm here 0 T1!"<<std::endl;
  ret = CreateAclTensorListTrans(weightHostData, weightShape, weightDeviceAddr, aclDataType::ACL_BF16, &weight);
  // ret = CreateAclTensorList(weightHostData, weightShape, weightDeviceAddr, aclDataType::ACL_BF16, &weight);
  std::cout<<"zgp debug, main_gmm_swiglu_gmm here 0 T2!"<<std::endl;
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorList(yHostData, yShape, yDeviceAddr, aclDataType::ACL_BF16, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  taskSplitValue.gmm_split_value = 4096;
  taskSplitValue.gmm_task_num = NUM_WORKERS_CUBE * taskSplitValue.single_rank_expert_num;
  std::string gmm_tiling_data = get_tiling_data_first_gmm(taskSplitValue.gmm_split_value);
  std::vector<uint32_t> gmm_tiling_data_vector = splitStringToUint32(gmm_tiling_data);
  std::vector<uint32_t> gmm_tiling_data_vector_max(gmm_tiling_data_vector.size() * NUM_WORKERS_CUBE);
  for (int i = 0; i < gmm_tiling_data_vector_max.size(); i++) {
    int index = i % gmm_tiling_data_vector.size();
    gmm_tiling_data_vector_max[i] = gmm_tiling_data_vector[index];
  }
  uint8_t* gmm_tiling_data_ptr = reinterpret_cast<uint8_t*>(gmm_tiling_data_vector_max.data());
  size_t gmm_tiling_data_size = gmm_tiling_data_vector_max.size() * sizeof(uint32_t);
  aclTensor* gmmTiling = nullptr;
  void* gmmTilingDeviceAddr = nullptr;
  ret = CreateAclTensorToTilingData(gmm_tiling_data_ptr, gmm_tiling_data_size, &gmmTilingDeviceAddr, aclDataType::ACL_UINT8, &gmmTiling);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::cout<<"zgp debug, main_gmm_swiglu_gmm here 1!"<<std::endl;

  // ++++++++++++++++++++++++++++++==========================================================================swiglu_grad
  std::vector<int64_t> swigluXShape = {taskSplitValue.per_rank_seq, 2048};
  std::vector<int64_t> swigluYShape = {taskSplitValue.per_rank_seq, 4096};
  std::vector<int64_t> swigluOutShape = {taskSplitValue.per_rank_seq, 4096};
  void* swigluXDeviceAddr = nullptr;
  void* swigluYDeviceAddr = nullptr;
  void* swigluOutDeviceAddr = nullptr;
  aclTensor* swigluX = nullptr;
  aclTensor* swigluY = nullptr;
  aclTensor* swigluOut = nullptr;
  std::vector<aclFloat16> swigluXHostData(swigluXShape[0]*swigluXShape[1], 1);
  std::vector<aclFloat16> swigluYHostData(swigluYShape[0]*swigluYShape[1], 1);
  std::vector<aclFloat16> swigluOutHostData(swigluOutShape[0]*swigluOutShape[1], 1);
  // size_t swigluXShapeSize = GetShapeSize(swigluXShape);
  // size_t swigluOutputZShapeSize = GetShapeSize(swigluOutShape);
  // void ** swigluX_input = (void **)(&swigluXHostData);
  // void ** swigluOut_input = (void **)(&swigluOutHostData);
  // ReadFile("../input/swiglu_in.bin", fileSize, *swigluX_input, swigluXShapeSize * dtypeSize);
  ret = CreateAclTensor(swigluXHostData, swigluXShape, &swigluXDeviceAddr, aclDataType::ACL_BF16, &swigluX);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(swigluYHostData, swigluYShape, &swigluYDeviceAddr, aclDataType::ACL_BF16, &swigluY);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(swigluOutHostData, swigluOutShape, &swigluOutDeviceAddr, aclDataType::ACL_BF16, &swigluOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  taskSplitValue.swiglu_split_value = 128;
  taskSplitValue.swiglu_task_num = swigluXShape[0]/taskSplitValue.swiglu_split_value;
  SwiGluTilingData swiGluTilingData;
  get_tiling_data_swiglu_grad(swiGluTilingData, taskSplitValue.swiglu_split_value);
  uint8_t* swiglu_tiling_data_ptr_ = reinterpret_cast<uint8_t*>(&swiGluTilingData);
  size_t swiglu_tiling_data_size_ = sizeof(swiGluTilingData);
  std::vector<uint8_t> swiglu_tiling_vector_data(swiglu_tiling_data_size_ * (NUM_WORKERS_VECTOR + 1));
  for (int i = 0; i < swiglu_tiling_vector_data.size(); i++) {
    int index = i % swiglu_tiling_data_size_;
    swiglu_tiling_vector_data[i] = *(swiglu_tiling_data_ptr_ + index);
  }
  swiglu_tiling_data_ptr_ = reinterpret_cast<uint8_t*>(swiglu_tiling_vector_data.data());
  swiglu_tiling_data_size_ = swiglu_tiling_vector_data.size() * sizeof(uint8_t);
  std::cout<<"multicore_moe_ffn swiglu tiling data size:" << swiglu_tiling_data_size_<<std::endl;
  aclTensor *swigluTiling = nullptr;
  void* swigluTilingDeviceAddr = nullptr;
  ret = CreateAclTensorToTilingData(swiglu_tiling_data_ptr_, swiglu_tiling_data_size_, &swigluTilingDeviceAddr, aclDataType::ACL_UINT8, &swigluTiling);
  std::cout<<"zgp debug, main_gmm_swiglu_gmm here 2!"<<std::endl;

  // ++++++++++++++++++++++++++++++==========================================================================GMM2
  // "4096,7168;2048,7168"
  std::vector<std::vector<int64_t>> xG2Shape = {{taskSplitValue.per_rank_seq, 4096}};
  std::vector<std::vector<int64_t>> weightG2Shape= {{taskSplitValue.single_rank_expert_num, 7168, 4096}};
  std::vector<std::vector<int64_t>> yG2Shape = {{taskSplitValue.per_rank_seq, 7168}};
  std::vector<int64_t> groupListG2Shape = {{taskSplitValue.single_rank_expert_num}};
  std::vector<int64_t> groupListG2Data(taskSplitValue.single_rank_expert_num, 1);
  void* xG2DeviceAddr[1];
  void* weightG2DeviceAddr[1];
  void* yG2DeviceAddr[1];
  void* groupListG2DeviceAddr;
  aclTensorList* xG2 = nullptr;
  aclTensorList* weightG2 = nullptr;
  aclTensor* groupListG2 = nullptr;
  aclTensorList* yG2 = nullptr;
  std::vector<aclFloat16> xG2HostData(xG2Shape[0][0]*xG2Shape[0][1], 1);
  std::vector<aclFloat16> weightG2HostData(weightG2Shape[0][0] * weightG2Shape[0][1] * weightG2Shape[0][2], 1);
  std::vector<aclFloat16> yG2HostData(yG2Shape[0][0]*yG2Shape[0][1]);
  // void ** inputXG2 = (void **)(&xG2HostData);
  // void ** input2G2 = (void **)(&weightG2HostData);
  // // ReadFile("../output/swiglu_out.bin", fileSize, *inputXG2, GetShapeSize(xG2Shape[0]) * dtypeSize);
  // ReadFile("../input/input_gmm_weight2_512.bin", fileSize, *input2G2, GetShapeSize(weightG2Shape[0]) * dtypeSize);
  ret = CreateAclTensorList(xG2HostData, xG2Shape, xG2DeviceAddr, aclDataType::ACL_BF16, &xG2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorListTrans(weightG2HostData, weightG2Shape, weightG2DeviceAddr, aclDataType::ACL_BF16, &weightG2);
  // ret = CreateAclTensorList(weightG2HostData, weightG2Shape, weightG2DeviceAddr, aclDataType::ACL_BF16, &weightG2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorList(yG2HostData, yG2Shape, yG2DeviceAddr, aclDataType::ACL_BF16, &yG2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor_New<int64_t>(groupListG2Data, groupListG2Shape, &groupListG2DeviceAddr, aclDataType::ACL_INT64, &groupListG2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  taskSplitValue.gmm_split_value_g2 = 4096;
  taskSplitValue.gmm_task_num_g2 = NUM_WORKERS_CUBE * taskSplitValue.single_rank_expert_num;
  std::string gmm_g2_tiling_data = get_tiling_data_second_gmm(taskSplitValue.gmm_split_value_g2);
  std::vector<uint32_t> gmm_g2_tiling_data_vector = splitStringToUint32(gmm_g2_tiling_data);
  std::vector<uint32_t> gmm_g2_tiling_data_vector_max(gmm_g2_tiling_data_vector.size() * NUM_WORKERS_CUBE);
  for (int i = 0; i < gmm_g2_tiling_data_vector_max.size(); i++) {
    int index = i % gmm_g2_tiling_data_vector.size();
    gmm_g2_tiling_data_vector_max[i] = gmm_g2_tiling_data_vector[index];
  }
  uint8_t* gmm_g2_tiling_data_ptr = reinterpret_cast<uint8_t*>(gmm_g2_tiling_data_vector_max.data());
  size_t gmm_g2_tiling_data_size = gmm_g2_tiling_data_vector_max.size() * sizeof(uint32_t);
  aclTensor* gmmTilingG2 = nullptr;
  void* gmmG2TilingDeviceAddr = nullptr;
  ret = CreateAclTensorToTilingData(gmm_g2_tiling_data_ptr, gmm_g2_tiling_data_size, &gmmG2TilingDeviceAddr, aclDataType::ACL_UINT8, &gmmTilingG2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::cout<<"zgp debug, main_gmm_swiglu_gmm here 3!"<<std::endl;

  // ++++++++++++++++++++++++++++++==========================================================================GMM3
  // "4096,7168;2048,7168"
  std::vector<std::vector<int64_t>> xG3Shape = {{taskSplitValue.per_rank_seq, 7168}};
  std::vector<std::vector<int64_t>> weightG3Shape= {{taskSplitValue.per_rank_seq, 4096}};
  std::vector<std::vector<int64_t>> yG3Shape = {{taskSplitValue.single_rank_expert_num, 7168, 4096}};
  std::vector<int64_t> groupListG3Shape = {{taskSplitValue.single_rank_expert_num}};
  std::vector<int64_t> groupListG3Data(taskSplitValue.single_rank_expert_num, 1);
  void* xG3DeviceAddr[1];
  void* weightG3DeviceAddr[1];
  void* yG3DeviceAddr[1];
  void* groupListG3DeviceAddr;
  aclTensorList* xG3 = nullptr;
  aclTensorList* weightG3 = nullptr;
  aclTensor* groupListG3 = nullptr;
  aclTensorList* yG3 = nullptr;
  std::vector<aclFloat16> xG3HostData(xG3Shape[0][0]*xG3Shape[0][1], 1);
  std::vector<aclFloat16> weightG3HostData(weightG3Shape[0][0] * weightG3Shape[0][1], 1);
  std::vector<aclFloat16> yG3HostData(yG3Shape[0][0]*yG3Shape[0][1] * yG3Shape[0][2]);
  // void ** inputXG3 = (void **)(&xG3HostData);
  // void ** input2G3 = (void **)(&weightG3HostData);
  // // ReadFile("../output/swiglu_out.bin", fileSize, *inputXG3, GetShapeSize(xG3Shape[0]) * dtypeSize);
  // ReadFile("../input/input_gmm_weight2_512.bin", fileSize, *input2G3, GetShapeSize(weightG3Shape[0]) * dtypeSize);
  ret = CreateAclTensorListTrans(xG3HostData, xG3Shape, xG3DeviceAddr, aclDataType::ACL_BF16, &xG3);
  // ret = CreateAclTensorList(xG3HostData, xG3Shape, xG3DeviceAddr, aclDataType::ACL_BF16, &xG3);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // ret = CreateAclTensorListTrans(weightG3HostData, weightG3Shape, weightG3DeviceAddr, aclDataType::ACL_BF16, &weightG3);
  ret = CreateAclTensorList(weightG3HostData, weightG3Shape, weightG3DeviceAddr, aclDataType::ACL_BF16, &weightG3);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorList(yG3HostData, yG3Shape, yG3DeviceAddr, aclDataType::ACL_BF16, &yG3);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor_New<int64_t>(groupListG3Data, groupListG3Shape, &groupListG3DeviceAddr, aclDataType::ACL_INT64, &groupListG3);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  taskSplitValue.gmm_split_value_g3 = 4096;
  taskSplitValue.gmm_task_num_g3 = NUM_WORKERS_CUBE * taskSplitValue.single_rank_expert_num;
  std::string gmm_g3_tiling_data = get_tiling_data_third_gmm(taskSplitValue.gmm_split_value_g3);
  std::vector<uint32_t> gmm_g3_tiling_data_vector = splitStringToUint32(gmm_g3_tiling_data);
  std::vector<uint32_t> gmm_g3_tiling_data_vector_max(gmm_g3_tiling_data_vector.size() * NUM_WORKERS_CUBE);
  for (int i = 0; i < gmm_g3_tiling_data_vector_max.size(); i++) {
    int index = i % gmm_g3_tiling_data_vector.size();
    gmm_g3_tiling_data_vector_max[i] = gmm_g3_tiling_data_vector[index];
  }
  uint8_t* gmm_g3_tiling_data_ptr = reinterpret_cast<uint8_t*>(gmm_g3_tiling_data_vector_max.data());
  size_t gmm_g3_tiling_data_size = gmm_g3_tiling_data_vector_max.size() * sizeof(uint32_t);
  aclTensor* gmmTilingG3 = nullptr;
  void* gmmG3TilingDeviceAddr = nullptr;
  ret = CreateAclTensorToTilingData(gmm_g3_tiling_data_ptr, gmm_g3_tiling_data_size, &gmmG3TilingDeviceAddr, aclDataType::ACL_UINT8, &gmmTilingG3);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::cout<<"zgp debug, main_gmm_swiglu_gmm here 4!"<<std::endl;

  // ++++++++++++++++++++++++++++++==========================================================================GMM4
  // "4096,7168;2048,7168"
  std::vector<std::vector<int64_t>> xG4Shape = {{taskSplitValue.per_rank_seq, 2048}};
  std::vector<std::vector<int64_t>> weightG4Shape= {{taskSplitValue.per_rank_seq, 7168}};
  std::vector<std::vector<int64_t>> yG4Shape = {{taskSplitValue.single_rank_expert_num, 2048, 7168}};
  std::vector<int64_t> groupListG4Shape = {{taskSplitValue.single_rank_expert_num}};
  std::vector<int64_t> groupListG4Data(taskSplitValue.single_rank_expert_num, 1);
  void* xG4DeviceAddr[1];
  void* weightG4DeviceAddr[1];
  void* yG4DeviceAddr[1];
  void* groupListG4DeviceAddr;
  aclTensorList* xG4 = nullptr;
  aclTensorList* weightG4 = nullptr;
  aclTensor* groupListG4 = nullptr;
  aclTensorList* yG4 = nullptr;
  std::vector<aclFloat16> xG4HostData(xG4Shape[0][0]*xG4Shape[0][1], 1);
  std::vector<aclFloat16> weightG4HostData(weightG4Shape[0][0] * weightG4Shape[0][1], 1);
  std::vector<aclFloat16> yG4HostData(yG4Shape[0][0]*yG4Shape[0][1] * yG4Shape[0][2]);
  // void ** inputXG4 = (void **)(&xG4HostData);
  // void ** input2G4 = (void **)(&weightG4HostData);
  // // ReadFile("../output/swiglu_out.bin", fileSize, *inputXG4, GetShapeSize(xG4Shape[0]) * dtypeSize);
  // ReadFile("../input/input_gmm_weight2_512.bin", fileSize, *input2G4, GetShapeSize(weightG4Shape[0]) * dtypeSize);
  ret = CreateAclTensorListTrans(xG4HostData, xG4Shape, xG4DeviceAddr, aclDataType::ACL_BF16, &xG4);
  // ret = CreateAclTensorList(xG4HostData, xG4Shape, xG4DeviceAddr, aclDataType::ACL_BF16, &xG4);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // ret = CreateAclTensorListTrans(weightG4HostData, weightG4Shape, weightG4DeviceAddr, aclDataType::ACL_BF16, &weightG4);
  ret = CreateAclTensorList(weightG4HostData, weightG4Shape, weightG4DeviceAddr, aclDataType::ACL_BF16, &weightG4);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorList(yG4HostData, yG4Shape, yG4DeviceAddr, aclDataType::ACL_BF16, &yG4);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor_New<int64_t>(groupListG4Data, groupListG4Shape, &groupListG4DeviceAddr, aclDataType::ACL_INT64, &groupListG4);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  taskSplitValue.gmm_split_value_g4 = 4096;
  taskSplitValue.gmm_task_num_g4 = NUM_WORKERS_CUBE * taskSplitValue.single_rank_expert_num;
  std::string gmm_g4_tiling_data = get_tiling_data_fourth_gmm(taskSplitValue.gmm_split_value_g4);
  std::vector<uint32_t> gmm_g4_tiling_data_vector = splitStringToUint32(gmm_g4_tiling_data);
  std::vector<uint32_t> gmm_g4_tiling_data_vector_max(gmm_g4_tiling_data_vector.size() * NUM_WORKERS_CUBE);
  for (int i = 0; i < gmm_g4_tiling_data_vector_max.size(); i++) {
    int index = i % gmm_g4_tiling_data_vector.size();
    gmm_g4_tiling_data_vector_max[i] = gmm_g4_tiling_data_vector[index];
  }
  uint8_t* gmm_g4_tiling_data_ptr = reinterpret_cast<uint8_t*>(gmm_g4_tiling_data_vector_max.data());
  size_t gmm_g4_tiling_data_size = gmm_g4_tiling_data_vector_max.size() * sizeof(uint32_t);
  aclTensor* gmmTilingG4 = nullptr;
  void* gmmG4TilingDeviceAddr = nullptr;
  ret = CreateAclTensorToTilingData(gmm_g4_tiling_data_ptr, gmm_g4_tiling_data_size, &gmmG4TilingDeviceAddr, aclDataType::ACL_UINT8, &gmmTilingG4);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::cout<<"zgp debug, main_gmm_swiglu_gmm here 5!"<<std::endl;

  // ++++++++++++++++++++++++++++++==========================================================================A2
  std::vector<int64_t> targetA2Shape = {taskSplitValue.per_rank_seq, 7168};
  std::vector<int64_t> targetOffsetA2Shape = {taskSplitValue.all_expert_num};
  std::vector<int64_t> srcA2Shape = {taskSplitValue.per_rank_seq, 7168};
  std::vector<int64_t> srcOffsetA2Shape = {taskSplitValue.all_expert_num};
  std::vector<int64_t> sizeA2Shape = {taskSplitValue.all_expert_num};
  void* targetA2DeviceAddr;
  void* targetOffsetA2DeviceAddr;
  void* srcA2DeviceAddr;
  void* srcOffsetA2DeviceAddr;
  void* sizeA2DeviceAddr;
  aclTensor* targetA2 = nullptr;
  aclTensor* targetOffsetA2 = nullptr;
  aclTensor* srcA2 = nullptr;
  aclTensor* srcOffsetA2 = nullptr;
  aclTensor* sizeA2 = nullptr;
  std::vector<aclFloat16> targetA2HostData(targetA2Shape[0]*targetA2Shape[1], 1);
  std::vector<int64_t> targetOffsetA2HostData(targetOffsetA2Shape[0], 1);
  std::vector<aclFloat16> srcA2HostData(srcA2Shape[0]*srcA2Shape[1]);
  std::vector<int64_t> srcOffsetA2HostData(srcOffsetA2Shape[0]);
  std::vector<int32_t> sizeA2HostData(sizeA2Shape[0]);
  ret = CreateAclTensor_New<aclFloat16>(targetA2HostData, targetA2Shape, &targetA2DeviceAddr, aclDataType::ACL_BF16, &targetA2);
  ret = CreateAclTensor_New<int64_t>(targetOffsetA2HostData, targetOffsetA2Shape, &targetOffsetA2DeviceAddr, aclDataType::ACL_INT64, &targetOffsetA2);
  ret = CreateAclTensor_New<aclFloat16>(srcA2HostData, srcA2Shape, &srcA2DeviceAddr, aclDataType::ACL_BF16, &srcA2);
  ret = CreateAclTensor_New<int64_t>(srcOffsetA2HostData, srcOffsetA2Shape, &srcOffsetA2DeviceAddr, aclDataType::ACL_INT64, &srcOffsetA2);
  ret = CreateAclTensor_New<int32_t>(sizeA2HostData, sizeA2Shape, &sizeA2DeviceAddr, aclDataType::ACL_INT32, &sizeA2);
  taskSplitValue.alltoall_split_value_a2 = 128;
  taskSplitValue.alltoall_task_num_a2 = taskSplitValue.all_expert_num * ((int)(taskSplitValue.per_expert_seq_to_other/taskSplitValue.alltoall_split_value));
  std::cout<<"zgp debug, main_gmm_swiglu_gmm here 6!"<<std::endl;


  // ======================================================================================================= //
  std::string res_dir = "multicore_moe_ffn_grad_tp4_ep4_910b";
  for (int64_t i =0;i<taskSplitValue.ep;i++) {
    taskSplitValue.rank_id = i;
    RuntimeConfig* runtimeConfig = new RuntimeConfig();
    init_task_split_value(taskSplitValue);
    init_runtime_config(runtimeConfig, NUM_WORKERS_VECTOR);
    std::vector<std::vector<int64_t>> alltoall_input_shape = {targetOffsetShape, srcShape, srcOffsetShape, sizeShape};
    std::vector<std::vector<int64_t>> alltoall_output_shape = {targetShape};
    std::vector<int64_t> alltoall_param_data = {1,2,3,4,0};
    task_split_alltoallv(runtimeConfig, alltoall_param_data, dtypeSize, alltoall_input_shape, alltoall_output_shape, "", taskSplitValue);
    std::vector<std::vector<std::vector<int64_t>>> gmm_input_shape = {xShape, weightShape, {groupListShape}};
    std::vector<std::vector<std::vector<int64_t>>> gmm_output_shape = {yShape};
    std::vector<int64_t> gmm_param_data = {0,7,19,8};
    task_split_gmm(runtimeConfig, gmm_param_data, dtypeSize, gmm_input_shape, gmm_output_shape, "", taskSplitValue);
    std::vector<std::vector<std::vector<int64_t>>> gmm_g4_input_shape = {xG4Shape, weightG4Shape, {groupListG4Shape}};
    std::vector<std::vector<std::vector<int64_t>>> gmm_g4_output_shape = {yG4Shape};
    std::vector<int64_t> gmm_g4_param_data = {5,0,19,6};
    task_split_gmm_g4(runtimeConfig, gmm_g4_param_data, dtypeSize, gmm_g4_input_shape, gmm_g4_output_shape, "", taskSplitValue);
    std::vector<std::vector<int64_t>> swiglu_input_shape = {swigluXShape, swigluYShape};
    std::vector<std::vector<int64_t>> swiglu_output_shape = {swigluOutShape};
    std::vector<int64_t> swiglu_param_data = {8,9,10}; // {6,8};
    task_split_swiglu_grad(runtimeConfig, swiglu_param_data, dtypeSize, swiglu_input_shape, swiglu_output_shape, "", taskSplitValue);
    std::vector<std::vector<std::vector<int64_t>>> gmm_g2_input_shape = {xG2Shape, weightG2Shape, {groupListG2Shape}};
    std::vector<std::vector<std::vector<int64_t>>> gmm_g2_output_shape = {yG2Shape};
    std::vector<int64_t> gmm_g2_param_data = {10,11,19,12};
    task_split_gmm_g2(runtimeConfig, gmm_g2_param_data, dtypeSize, gmm_g2_input_shape, gmm_g2_output_shape, "", taskSplitValue);
    std::vector<std::vector<int64_t>> alltoall_a2_input_shape = {targetOffsetA2Shape, srcA2Shape, srcOffsetA2Shape, sizeA2Shape};
    std::vector<std::vector<int64_t>> alltoall_a2_output_shape = {targetA2Shape};
    std::vector<int64_t> alltoall_a2_param_data = {14,12,15,16,13};
    task_split_alltoallv_a2(runtimeConfig, alltoall_a2_param_data, dtypeSize, alltoall_a2_input_shape, alltoall_a2_output_shape, "", taskSplitValue);
    std::vector<std::vector<std::vector<int64_t>>> gmm_g3_input_shape = {xG3Shape, weightG3Shape, {groupListG3Shape}};
    std::vector<std::vector<std::vector<int64_t>>> gmm_g3_output_shape = {yG3Shape};
    std::vector<int64_t> gmm_g3_param_data = {17,10,19,18};
    task_split_gmm_g3(runtimeConfig, gmm_g3_param_data, dtypeSize, gmm_g3_input_shape, gmm_g3_output_shape, "", taskSplitValue);
    int64_t task_num_all = taskSplitValue.alltoall_task_num + taskSplitValue.gmm_task_num + taskSplitValue.swiglu_task_num + taskSplitValue.gmm_task_num_g2 + taskSplitValue.alltoall_task_num_a2 + taskSplitValue.gmm_task_num_g3 + taskSplitValue.gmm_task_num_g4 + 1;
    add_terminate_task(runtimeConfig, taskSplitValue);
    revise_task_queue(runtimeConfig, taskSplitValue);
    revise_gmm_task_queue(runtimeConfig, taskSplitValue);
    add_dynamic_data(runtimeConfig, taskSplitValue);
    runtimeConfig->task_num = task_num_all;
    runtimeConfig->atomic_add_values[0] = 1;
    uint8_t* runtimeConfig_ptr = reinterpret_cast<uint8_t*>(runtimeConfig);
    size_t runtimeConfig_size = sizeof(RuntimeConfig);
    aclTensor *runtimeConfig_tensor = nullptr;
    void* runtimeConfigAddr = nullptr;
    ret = CreateAclTensorToTilingData(runtimeConfig_ptr, runtimeConfig_size, &runtimeConfigAddr, aclDataType::ACL_UINT8, &runtimeConfig_tensor);
    std::string s_s = "../" + res_dir + "/runtime_config_input_rank_" + std::to_string(i) + ".bin";
    write_data_to_file_all<uint8_t>(static_cast<int64_t>(runtimeConfig_size), runtimeConfigAddr, s_s);
    aclDestroyTensor(runtimeConfig_tensor);
    aclrtFree(runtimeConfigAddr);
    delete runtimeConfig;
    runtimeConfig = nullptr;

    std::vector<int32_t> all_event_counters(MAX_EVENT_NUM, 0);
    uint8_t* event_counters_ptr = reinterpret_cast<uint8_t*>(all_event_counters.data());
    size_t all_event_counters_size = all_event_counters.size() * sizeof(int32_t);
    aclTensor *all_event_counters_tensor = nullptr;
    void* all_event_counters_addr = nullptr;
    ret = CreateAclTensorToTilingData(event_counters_ptr, all_event_counters_size, &all_event_counters_addr, aclDataType::ACL_UINT8, &all_event_counters_tensor);
    std::string all_event_counters_s = "../" + res_dir + "/all_event_counters.bin";
    write_data_to_file_all<uint8_t>(static_cast<int64_t>(all_event_counters_size), all_event_counters_addr, all_event_counters_s);
    aclDestroyTensor(all_event_counters_tensor);
    aclrtFree(all_event_counters_addr);
  }

  void *gmmWorkspaceDeviceAddr = nullptr;
  uint64_t gmmWorkspaceSize = 95420928;
  aclTensor* gmmWorkspace = nullptr;
  if (gmmWorkspaceSize > 0) {
    ret = CreateAclTensorToWorkspace(gmmWorkspaceSize, &gmmWorkspaceDeviceAddr, aclDataType::ACL_UINT8, &gmmWorkspace);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return FAILED;);
  }
  std::cout<<"zgp debug, main_gmm_swiglu_gmm here 8!"<<std::endl;
  void *swigluWorkspaceDeviceAddr = nullptr;
  uint64_t swigluWorkspaceSize = 95420416;
  aclTensor* swigluWorkspace = nullptr;
  if (swigluWorkspaceSize > 0) {
    ret = CreateAclTensorToWorkspace(swigluWorkspaceSize, &swigluWorkspaceDeviceAddr, aclDataType::ACL_UINT8, &swigluWorkspace);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return FAILED;);
  }
  uint64_t workspaceSize_1 = 0;
  aclOpExecutor* executor_1;
  std::cout<<"zgp debug, main_gmm_swiglu_gmm here 9!"<<std::endl;
  std::vector<int64_t> outputShape = {1};
  void* outputDeviceAddr;
  aclTensor* output = nullptr;
  std::vector<aclFloat16> outputHostData(outputShape[0], 1);
  ret = CreateAclTensor_New<aclFloat16>(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_BF16, &output);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  aclTensor* tensors_target[1];
  tensors_target[0] = target;
  aclTensorList* tensors_target_list = aclCreateTensorList(tensors_target, 1);
  aclTensor* tensors_src[1];
  tensors_src[0] = src;
  aclTensorList* tensors_src_list = aclCreateTensorList(tensors_src, 1);
  aclTensor* tensors_swigluY[1];
  tensors_swigluY[0] = swigluY;
  aclTensorList* tensors_swigluY_list = aclCreateTensorList(tensors_swigluY, 1);
  aclTensor* tensors_swigluOut[1];
  tensors_swigluOut[0] = swigluOut;
  aclTensorList* tensors_swigluOut_list = aclCreateTensorList(tensors_swigluOut, 1);
  aclTensor* tensors_targetA2[1];
  tensors_targetA2[0] = targetA2;
  aclTensorList* tensors_targetA2_list = aclCreateTensorList(tensors_targetA2, 1);
  // ret = aclnnMulticoreMoeFfnGradGetWorkspaceSize(tensors_target_list, targetOffset, tensors_src_list, srcOffset, sizeA1,
  //                                         xG4, yG4, weight, y,
  //                                         tensors_swigluY_list, tensors_swigluOut_list,
  //                                         weightG2,yG2,
  //                                         tensors_targetA2_list,targetOffsetA2,srcOffsetA2,sizeA2,
  //                                         xG3,yG3, groupList,
  //                                         gmmTiling, gmmTilingG2, gmmTilingG3, gmmTilingG4,
  //                                         swigluTiling, gmmWorkspace, swigluWorkspace,
  //                                         runtimeConfig_tensor,
  //                                         0, 2,
  //                                         false, true, false, true, true, false, true, false,output,
  //                                         &workspaceSize_1, &executor_1);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulticoreMoeFfnGradGetWorkspaceSize failed. ERROR: %d\n", ret); return FAILED);
  void *workspaceAddr_1 = nullptr;
  INFO_LOG("swiglu start 1");
  std::cout<<"zgp debug, workspaceSize_1:" << workspaceSize_1 << std::endl;
  if (workspaceSize_1 > 0) {
    ret = aclrtMalloc(&workspaceAddr_1, workspaceSize_1, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return FAILED;);
  }

  INFO_LOG("swiglu start 2");
  std::cout<<"zgp debug, workspaceSize_1 1:" << workspaceSize_1 << std::endl;
  // ret = aclnnMulticoreMoeFfnGrad(workspaceAddr_1, workspaceSize_1, executor_1, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulticoreMoeFfn failed. ERROR: %d\n", ret); return FAILED);
  INFO_LOG("swiglu start 3");

  // ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);
  INFO_LOG("swiglu start 4");


  // write_data_to_file_all<aclFloat16>(swigluOutShape, swigluOutDeviceAddr, "../output/swiglu_result.bin");
  // write_data_to_file_all<aclFloat16>(yG2Shape[0], yG2DeviceAddr[0], "../output/gmm_result.bin");
  // write_data_to_file_all<aclFloat16>(yShape[0], yDeviceAddr[0], "../output/gmmG1_result.bin");
  // write_data_to_file_all<aclFloat16>(xG2Shape[0], xG2DeviceAddr[0], "../output/gmmxG1_result.bin");

  write_data_to_file_all<uint8_t>(static_cast<int64_t>(gmm_tiling_data_size), gmmTilingDeviceAddr, "../" + res_dir + "/gmm_tiling_data.bin");
  write_data_to_file_all<uint8_t>(static_cast<int64_t>(gmm_g2_tiling_data_size), gmmG2TilingDeviceAddr, "../" + res_dir + "/gmmTilingG2.bin");
  write_data_to_file_all<uint8_t>(static_cast<int64_t>(gmm_g3_tiling_data_size), gmmG3TilingDeviceAddr, "../" + res_dir + "/gmmTilingG3.bin");
  write_data_to_file_all<uint8_t>(static_cast<int64_t>(gmm_g4_tiling_data_size), gmmG4TilingDeviceAddr, "../" + res_dir + "/gmmTilingG4.bin");
  write_data_to_file_all<uint8_t>(static_cast<int64_t>(swiglu_tiling_data_size_), swigluTilingDeviceAddr, "../" + res_dir + "/swiglu_tiling.bin");
  write_data_to_file_all<uint8_t>(static_cast<int64_t>(gmmWorkspaceSize), gmmWorkspaceDeviceAddr, "../" + res_dir + "/gmm_workspace.bin");
  // write_data_to_file_all<uint8_t>(static_cast<int64_t>(runtimeConfig_size), runtimeConfigAddr, "../" + res_dir + "/runtime_config_input_rank0.bin");
  // write_data_to_file_all<uint8_t>(static_cast<int64_t>(runtimeConfig_size), runtimeConfigAddr, "../" + res_dir + "/runtime_config_input_rank1.bin");
  std::cout<<"zgp debug, main_gmm_swiglu_gmm here 10!"<<std::endl;
  aclDestroyTensorList(x);
  aclDestroyTensorList(weight);
  aclDestroyTensor(groupList);
  aclDestroyTensorList(y);
  aclDestroyTensor(swigluX);
  aclDestroyTensor(swigluOut);
  for (int i = 0; i < 1; i++) {
    aclrtFree(xDeviceAddr[i]);
    aclrtFree(weightDeviceAddr[i]);
    // aclrtFree(biasDeviceAddr[i]);
    aclrtFree(yDeviceAddr[i]);
  }
  aclrtFree(swigluXDeviceAddr);
  aclrtFree(swigluOutDeviceAddr);
  if (workspaceSize_1 > 0) {
    aclrtFree(workspaceAddr_1);
  }
  if (gmmWorkspaceSize > 0) {
    aclrtFree(gmmWorkspaceDeviceAddr);
    aclDestroyTensor(gmmWorkspace);
  }
  if (swigluWorkspaceSize > 0) {
    aclrtFree(swigluWorkspaceDeviceAddr);
    aclDestroyTensor(swigluWorkspace);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
