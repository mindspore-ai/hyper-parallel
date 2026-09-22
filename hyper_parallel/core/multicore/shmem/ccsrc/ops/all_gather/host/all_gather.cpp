/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "acl/acl.h"

#include "ops/include/shmem_kernel.h"

namespace ShmemKernel {
namespace {

constexpr uint64_t kMediumMessageBytes = 64 * 1024;
constexpr uint64_t kLargeMessageBytes = 1024 * 1024;
constexpr uint32_t kSmallBlockDim = 1;
constexpr uint32_t kMediumBlockDim = 4;
constexpr uint32_t kLargeBlockDim = 24;

constexpr uint32_t SelectBlockDim(uint64_t local_bytes) {
  if (local_bytes < kMediumMessageBytes) {
    return kSmallBlockDim;
  }
  return local_bytes < kLargeMessageBytes ? kMediumBlockDim : kLargeBlockDim;
}

static_assert(SelectBlockDim(kMediumMessageBytes - 1) == kSmallBlockDim);
static_assert(SelectBlockDim(kMediumMessageBytes) == kMediumBlockDim);
static_assert(SelectBlockDim(kLargeMessageBytes - 1) == kMediumBlockDim);
static_assert(SelectBlockDim(kLargeMessageBytes) == kLargeBlockDim);

}  // namespace

extern void launch_all_gather(uint32_t block_dim, void *stream, uint8_t *output, uint8_t *input, uint64_t local_bytes,
                              int32_t root_rank, int32_t root_size);

aclError aclshmem_all_gather(aclrtStream stream, void *output, void *input, uint64_t local_bytes, int32_t root_rank,
                             int32_t root_size) {
  if (local_bytes == 0) {
    return ACL_SUCCESS;
  }
  // The CANN thread error variable is sticky: any unconsumed error from an earlier ACL call on this thread (e.g. the
  // torch_npu queue drain or the entry Barrier) would survive until read. Drain it before launch so the post-launch
  // read is attributable to this launch alone. ACL_RT_THREAD_LEVEL is the only level the Runtime accepts.
  static_cast<void>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
  launch_all_gather(SelectBlockDim(local_bytes), stream, static_cast<uint8_t *>(output), static_cast<uint8_t *>(input),
                    local_bytes, root_rank, root_size);
  // launch_all_gather returns void: surface host-side launch failures (invalid argument, bad stream, ...) through the
  // thread error state instead of silently enqueuing nothing.
  return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

}  // namespace ShmemKernel
