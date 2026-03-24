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

extern void signal_wait_until(uint32_t block_dim, void *stream, uint64_t elementSize, uint8_t *depend_target,
                              uint8_t *signal, uint8_t *signal_offset, uint8_t *compare_value, int64_t compare_op);

int aclshmem_signal_wait_until(aclrtStream stream, uint64_t elementSize, void *depend_target, void *signal,
                               void *signal_offset, void *compare_value, int64_t compare_op) {
  int status = 0;
  uint32_t block_dim = 1;
  // signal_wait_until
  signal_wait_until(block_dim, stream, elementSize, (uint8_t *)depend_target, (uint8_t *)signal,
                    (uint8_t *)signal_offset, (uint8_t *)compare_value, compare_op);
  return status;
}

}  // namespace ShmemKernel
