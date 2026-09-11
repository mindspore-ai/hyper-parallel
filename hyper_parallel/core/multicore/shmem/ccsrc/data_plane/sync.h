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

namespace hyper_parallel::multicore::shmem::data_plane {

using cann::CompareOp;
using cann::SignalOp;

/** @brief Byte spacing between adjacent symmetric Signal slots. */
inline constexpr uint32_t kSignalSlotBytes = 64U;

/** @brief Order earlier SHMEM operations before later SHMEM operations issued by the calling AIV. */
ACLSHMEM_DEVICE void fence() { cann::device::fence(); }

/**
 * @brief Apply one Set or Add operation to a symmetric Signal on a target Root PE.
 *
 * @param remote_signal
 * Symmetric GM address of the remote Signal word.
 * @param value Value used by the operation.
 * @param operation Set
 * or Add.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 */
ACLSHMEM_DEVICE void signal(__gm__ int32_t *remote_signal, int32_t value, SignalOp operation, int32_t target_pe) {
  cann::device::signal(remote_signal, value, operation, target_pe);
}

/**
 * @brief Set one symmetric Signal on a target Root PE.
 *
 * @param remote_signal Symmetric GM address of the remote Signal word.
 * @param value Value written to the Signal word.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @note Return means the Signal update is remote-visible.
 */
__aicore__ inline void signal_set(__gm__ int32_t *remote_signal, int32_t value, int32_t target_pe) {
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
__aicore__ inline void signal_add(__gm__ int32_t *remote_signal, int32_t value, int32_t target_pe) {
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
__aicore__ inline int32_t signal_wait(__gm__ int32_t *local_signal, CompareOp comparison, int32_t value) {
  return cann::device::signal_wait(local_signal, comparison, value);
}

/**
 * @brief Complete a collective device barrier over CANN Root WORLD.
 *
 * @note Every Root PE must participate. The consumer satisfies CANN MIX Kernel and FFTS requirements and does not mix
 * this protocol with AscendC SyncAll.
 */
__aicore__ inline void barrier() { cann::device::barrier(); }

}  // namespace hyper_parallel::multicore::shmem::data_plane
