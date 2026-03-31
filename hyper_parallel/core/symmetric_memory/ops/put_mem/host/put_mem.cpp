/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>

#include "acl/acl.h"

#include "shmem.h"
#include "include/shmem_kernel.h"

namespace ShmemKernel {

extern void put_mem(uint32_t block_dim, void *stream, uint64_t elementSize, uint8_t *target, uint8_t *target_offset,
                    uint8_t *src, uint8_t *src_offset, uint8_t *size, int64_t target_pe, bool non_blocking,
                    uint8_t *Syncmem);

int aclshmem_put_mem(uint32_t block_dim, aclrtStream stream, uint64_t elementSize, void *target, void *target_offset,
                     void *src, void *src_offset, void *size, int64_t target_pe, bool non_blocking) {
  void *sync_mem_device = nullptr;
  aclError ret = aclrtMalloc(&sync_mem_device, 8 * block_dim * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    std::cerr << "aclrtMalloc failed: " << ret << std::endl;
    aclFinalize();
    return -1;
  }
  ret = aclrtMemset(sync_mem_device, 8 * block_dim * sizeof(int32_t), 0, 8 * block_dim * sizeof(int32_t));
  if (ret != ACL_SUCCESS) {
    std::cerr << "aclrtMemset failed: " << ret << std::endl;
    aclFinalize();
    return -1;
  }
  int status = 0;
  // put_mem
  put_mem(block_dim, stream, elementSize, (uint8_t *)target, (uint8_t *)target_offset, (uint8_t *)src,
          (uint8_t *)src_offset, (uint8_t *)size, target_pe, non_blocking, (uint8_t *)sync_mem_device);
  aclrtFree(sync_mem_device);
  return status;
}

}  // namespace ShmemKernel
