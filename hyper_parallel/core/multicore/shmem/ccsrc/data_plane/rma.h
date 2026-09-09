/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#include "cann/device.h"
#include "data_plane/sync.h"

namespace hyper_parallel::multicore::shmem::data_plane {

/**
 * @brief Translate one local symmetric GM address to the corresponding address on a target Root PE.
 *
 * @param
 * local_symmetric_address Symmetric GM address in the calling PE's address space.
 * @param target_pe Destination in
 * CANN Root WORLD coordinates.
 * @return Corresponding symmetric GM address on target_pe.
 */
ACLSHMEM_DEVICE __gm__ void *remote_ptr(__gm__ void *local_symmetric_address, int32_t target_pe) {
  return cann::device::remote_ptr(local_symmetric_address, target_pe);
}

/**
 * @brief Synchronize-copy local GM bytes to a symmetric address on one target Root PE.
 *
 * @param remote_dst Symmetric GM destination represented by its local address.
 * @param local_src Local GM source owned by the caller.
 * @param bytes Transfer length in Byte; zero is a no-op.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @pre The consumer validates addresses, length, target PE and MTE reachability.
 * @note One nonzero call maps to one CANN call. Return means the data is remote-visible.
 */
__aicore__ inline void put(__gm__ void *remote_dst, __gm__ const void *local_src, uint32_t bytes, int32_t target_pe) {
  if (bytes != 0) {
    cann::device::put(remote_dst, local_src, bytes, target_pe);
  }
}

/**
 * @brief Synchronize-copy bytes from a symmetric address on one Root PE to local GM.
 *
 * @param local_dst Local GM destination owned by the caller.
 * @param remote_src Symmetric GM source represented by its local address.
 * @param bytes Transfer length in Byte; zero is a no-op.
 * @param target_pe Source in CANN Root WORLD coordinates.
 * @pre The consumer validates addresses, length, target PE and MTE reachability.
 * @note One nonzero call maps to one CANN call. Return makes local_dst available to later work.
 */
__aicore__ inline void get(__gm__ void *local_dst, __gm__ const void *remote_src, uint32_t bytes, int32_t target_pe) {
  if (bytes != 0) {
    cann::device::get(local_dst, remote_src, bytes, target_pe);
  }
}

/**
 * @brief Synchronize-copy one data chunk and then update a Signal on the same target Root PE.
 *
 * @param remote_dst Symmetric GM destination represented by its local address.
 * @param local_src Local GM source owned by the caller.
 * @param bytes Transfer length in Byte; zero skips the data transfer but still updates the Signal.
 * @param remote_signal Symmetric GM address of the remote Signal word.
 * @param signal_value Value used by the Signal operation.
 * @param signal_op Set or Add operation performed after the Put completes.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @pre For a multi-chunk message, the consumer uses put() for preceding chunks and calls put_signal() only for the
 * final chunk, so the logical message updates its Signal exactly once.
 * @note Return means the data is remote-visible before the Signal update becomes visible.
 */
__aicore__ inline void put_signal(__gm__ void *remote_dst, __gm__ const void *local_src, uint32_t bytes,
                                  __gm__ int32_t *remote_signal, int32_t signal_value, SignalOp signal_op,
                                  int32_t target_pe) {
  put(remote_dst, local_src, bytes, target_pe);
  signal(remote_signal, signal_value, signal_op, target_pe);
}

}  // namespace hyper_parallel::multicore::shmem::data_plane
