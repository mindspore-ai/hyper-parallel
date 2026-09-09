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

namespace hyper_parallel::multicore::shmem::data_plane::mte_rma {

/**
 * @brief Issue an explicit MTE NBI Put using caller-owned UB and Event resources.
 *
 * @param remote_dst Symmetric GM destination represented by its local address.
 * @param local_src Local GM source owned by the caller.
 * @param ub_scratch Caller-owned temporary UB address.
 * @param ub_scratch_bytes UB capacity in Byte.
 * @param bytes Transfer length in Byte.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @param event_id Caller-owned MTE synchronization Event ID.
 * @pre The consumer validates arguments, MTE reachability and resource reuse dependencies.
 * @note Return means issued, not completed. The consumer must eventually call quiet().
 */
__aicore__ inline void put_nbi(__gm__ uint8_t *remote_dst, __gm__ const uint8_t *local_src,
                               __ubuf__ uint8_t *ub_scratch, uint32_t ub_scratch_bytes, uint32_t bytes,
                               int32_t target_pe, uint32_t event_id) {
  cann::device::mte_put_nbi(remote_dst, local_src, ub_scratch, ub_scratch_bytes, bytes, target_pe, event_id);
}

/** @brief Complete explicit MTE NBI operations previously issued by the calling AIV. */
__aicore__ inline void quiet() { cann::device::mte_quiet(); }

}  // namespace hyper_parallel::multicore::shmem::data_plane::mte_rma
