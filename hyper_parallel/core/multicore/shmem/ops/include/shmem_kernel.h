/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_LIB_ACLSHMEM_KERNEL_H
#define SHARED_LIB_ACLSHMEM_KERNEL_H

#include <cstdint>

#include "acl/acl.h"

namespace ShmemKernel {

int aclshmem_put_mem(uint32_t block_dim, aclrtStream stream, uint64_t elementSize, void *target, void *target_offset,
                     void *src, void *src_offset, void *size, int64_t target_pe, bool non_blocking);

int aclshmem_get_mem(uint32_t block_dim, aclrtStream stream, uint64_t elementSize, void *target, void *target_offset,
                     void *src, void *src_offset, void *size, int64_t target_pe, bool non_blocking);

int aclshmem_put_mem_signal(uint32_t block_dim, aclrtStream stream, uint64_t elementSize, void *target,
                            void *target_offset, void *src, void *src_offset, void *size, void *signal,
                            void *signal_offset, void *signal_value, int64_t signal_op, int64_t target_pe,
                            bool non_blocking);

int aclshmem_signal_op(aclrtStream stream, void *signal, void *signal_offset, void *signal_value, int64_t signal_op,
                       int64_t target_pe);

int aclshmem_signal_wait_until(aclrtStream stream, uint64_t elementSize, void *depend_target, void *signal,
                               void *signal_offset, void *compare_value, int64_t compare_op);
}  // namespace ShmemKernel

#endif  // SHARED_LIB_ACLSHMEM_KERNEL_H
