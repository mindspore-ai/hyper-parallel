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

extern void signal_op(uint32_t block_dim, void *stream, uint8_t *signal, uint8_t *signal_offset, uint8_t *signal_value,
                      int64_t signal_op, int64_t target_pe);

int aclshmem_signal_op(aclrtStream stream, void *signal, void *signal_offset, void *signal_value, int64_t signal_oper,
                       int64_t target_pe) {
  int status = 0;
  uint32_t block_dim = 1;
  // signal_op
  signal_op(block_dim, stream, (uint8_t *)signal, (uint8_t *)signal_offset, (uint8_t *)signal_value, signal_oper,
            target_pe);
  return status;
}

}  // namespace ShmemKernel
