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

#include "cann/signal.h"
#include "shmem.h"

namespace hyper_parallel::multicore::shmem::cann::device {

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
  return aclshmem_ptr(local_symmetric_address, target_pe);
}

/** @brief Order earlier SHMEM operations before later SHMEM operations issued by the calling AIV. */
ACLSHMEM_DEVICE void fence() { aclshmem_fence(); }

ACLSHMEM_DEVICE constexpr int32_t ToCannCompareOp(CompareOp comparison) {
  switch (comparison) {
    case CompareOp::Equal:
      return ACLSHMEM_CMP_EQ;
    case CompareOp::NotEqual:
      return ACLSHMEM_CMP_NE;
    case CompareOp::Greater:
      return ACLSHMEM_CMP_GT;
    case CompareOp::GreaterEqual:
      return ACLSHMEM_CMP_GE;
    case CompareOp::Less:
      return ACLSHMEM_CMP_LT;
    case CompareOp::LessEqual:
      return ACLSHMEM_CMP_LE;
  }
  return ACLSHMEM_CMP_EQ;
}

/**
 * @brief Synchronize-copy local GM bytes to a symmetric address on one target Root PE.
 *
 * @param remote_dst Symmetric GM destination represented by its local address.
 * @param local_src Local GM source owned by the caller.
 * @param bytes Transfer length in Byte.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @note The caller validates all arguments. Return means the data is remote-visible.
 */
ACLSHMEM_DEVICE void put(__gm__ void *remote_dst, __gm__ const void *local_src, uint32_t bytes, int32_t target_pe) {
  aclshmem_putmem(remote_dst, const_cast<__gm__ void *>(local_src), bytes, target_pe);
}

/**
 * @brief Synchronize-copy bytes from a symmetric address on one Root PE to local GM.
 *
 * @param local_dst Local GM destination owned by the caller.
 * @param remote_src Symmetric GM source represented by its local address.
 * @param bytes Transfer length in Byte.
 * @param target_pe Source in CANN Root WORLD coordinates.
 * @note The caller validates all arguments. Return makes local_dst available to later work in the calling Kernel.
 */
ACLSHMEM_DEVICE void get(__gm__ void *local_dst, __gm__ const void *remote_src, uint32_t bytes, int32_t target_pe) {
  aclshmem_getmem(local_dst, const_cast<__gm__ void *>(remote_src), bytes, target_pe);
}

/**
 * @brief Apply one Set or Add operation to a symmetric Signal on a target Root PE.
 *
 * @param remote_signal
 * Symmetric GM address of the remote Signal word.
 * @param value Value used by the operation.
 * @param operation Set
 * or Add.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @note Return means the Signal update is
 * remote-visible.
 */
ACLSHMEM_DEVICE void signal(__gm__ int32_t *remote_signal, int32_t value, SignalOp operation, int32_t target_pe) {
  aclshmemx_signal_op(remote_signal, value, operation == SignalOp::Add ? ACLSHMEM_SIGNAL_ADD : ACLSHMEM_SIGNAL_SET,
                      target_pe);
}

/** @brief Set one symmetric Signal on a target Root PE. */
ACLSHMEM_DEVICE void signal_set(__gm__ int32_t *remote_signal, int32_t value, int32_t target_pe) {
  signal(remote_signal, value, SignalOp::Set, target_pe);
}

/**
 * @brief Atomically add to one symmetric Signal on a target Root PE.
 *
 * @param remote_signal Symmetric GM address of the remote Signal word.
 * @param value Value added to the Signal word.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @note Return means the Signal update is remote-visible.
 */
ACLSHMEM_DEVICE void signal_add(__gm__ int32_t *remote_signal, int32_t value, int32_t target_pe) {
  signal(remote_signal, value, SignalOp::Add, target_pe);
}

/**
 * @brief Block until one local Signal satisfies a comparison.
 *
 * @param local_signal Local GM address of the Signal word; no PE translation is performed.
 * @param compare Comparison applied between the observed Signal and value.
 * @param value Comparison value.
 * @return Signal value observed when the comparison is satisfied.
 */
ACLSHMEM_DEVICE int32_t signal_wait(__gm__ int32_t *local_signal, CompareOp comparison, int32_t value) {
  return aclshmem_signal_wait_until(local_signal, ToCannCompareOp(comparison), value);
}

/**
 * @brief Complete a collective device barrier over CANN Root WORLD.
 *
 * @note Every Root PE must participate. The caller satisfies CANN MIX Kernel and FFTS requirements and does not mix
 * this protocol with AscendC SyncAll.
 */
ACLSHMEM_DEVICE void barrier() { aclshmem_barrier(ACLSHMEM_TEAM_WORLD); }

/**
 * @brief Issue an explicit MTE NBI Put to one target Root PE.
 *
 * @param remote_dst Symmetric GM destination represented by its local address.
 * @param local_src Local GM source owned by the caller.
 * @param ub_scratch Caller-owned temporary UB address.
 * @param ub_scratch_bytes UB capacity in Byte.
 * @param bytes Transfer length in Byte.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @param event_id Caller-owned MTE synchronization Event ID.
 * @note Return means issued, not completed. The caller owns UB/Event synchronization and eventual quiet.
 */
ACLSHMEM_DEVICE void mte_put_nbi(__gm__ uint8_t *remote_dst, __gm__ const uint8_t *local_src,
                                 __ubuf__ uint8_t *ub_scratch, uint32_t ub_scratch_bytes, uint32_t bytes,
                                 int32_t target_pe, uint32_t event_id) {
  aclshmemx_mte_put_nbi<uint8_t>(remote_dst, const_cast<__gm__ uint8_t *>(local_src), ub_scratch, ub_scratch_bytes,
                                 bytes, target_pe, event_id);
}

/** @brief Complete explicit MTE NBI operations previously issued by the calling AIV. */
ACLSHMEM_DEVICE void mte_quiet() { aclshmemx_mte_quiet(); }

}  // namespace hyper_parallel::multicore::shmem::cann::device
