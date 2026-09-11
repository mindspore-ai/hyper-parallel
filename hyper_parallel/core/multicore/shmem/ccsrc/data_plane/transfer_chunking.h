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

/* Dual host/device qualifier for these pure functions: plain inline for host
 * toolchains (native tests included), forced inline for both compilation
 * passes under the cce toolchain so Kernels can call them. */
#if defined(__CCE__)
#define HP_SHMEM_CHUNKING_HOST_DEVICE __forceinline__[host, aicore]
#else
#define HP_SHMEM_CHUNKING_HOST_DEVICE inline
#endif

namespace hyper_parallel::multicore::shmem::data_plane {

/** @brief Consumer-selected limits for splitting one byte message into transfer Chunks. */
struct ChunkPolicy {
  uint32_t max_chunk_bytes;
  uint32_t minimum_tail_bytes;
};

/** @brief Immutable result of applying one ChunkPolicy to a byte message. */
struct ChunkPlan {
  uint64_t total_bytes;
  ChunkPolicy policy;
  uint64_t chunk_count;
};

/** @brief One exact transfer Chunk relative to the consumer-owned message base. */
struct TransferChunk {
  uint64_t offset_bytes;
  uint32_t size_bytes;
};

/** @brief One worker's contiguous partition of a consumer-owned work-item index space. */
struct WorkPartition {
  uint64_t first_item;
  uint64_t item_count;
};

/**
 * @brief Plan capacity-bounded Chunks for one byte message.
 * @param total_bytes Exact valid message length in Byte; zero produces no Chunks.
 * @param policy Policy with nonzero max_chunk_bytes. A zero minimum_tail_bytes disables tail rebalancing.
 * @pre The Host consumer guarantees minimum_tail_bytes <= max_chunk_bytes / 2 when tail rebalancing is enabled.
 * @return A plan whose Chunk count is the minimum required by max_chunk_bytes.
 */
HP_SHMEM_CHUNKING_HOST_DEVICE ChunkPlan make_chunk_plan(uint64_t total_bytes, ChunkPolicy policy) {
  const uint64_t chunk_count =
    total_bytes / policy.max_chunk_bytes + (total_bytes % policy.max_chunk_bytes == 0 ? 0 : 1);
  return {total_bytes, policy, chunk_count};
}

/**
 * @brief Return one exact Chunk from a plan.
 * @param plan Plan produced by make_chunk_plan().
 * @param chunk_index Chunk index in [0, plan.chunk_count).
 * @pre The caller supplies a valid chunk_index and a Host-validated policy.
 * @return Chunk offset relative to the message base and its exact valid byte length.
 *
 * For a multi-Chunk message whose natural tail is smaller than minimum_tail_bytes, the final boundary moves backward
 * by minimum_tail_bytes. The penultimate Chunk shrinks and the final Chunk grows without expanding the valid range.
 */
HP_SHMEM_CHUNKING_HOST_DEVICE TransferChunk get_chunk(const ChunkPlan &plan, uint64_t chunk_index) {
  const uint64_t tail_bytes = plan.total_bytes % plan.policy.max_chunk_bytes;
  const bool rebalance_tail = plan.chunk_count >= 2 && tail_bytes != 0 && tail_bytes < plan.policy.minimum_tail_bytes;
  if (rebalance_tail && chunk_index == plan.chunk_count - 2) {
    return {chunk_index * plan.policy.max_chunk_bytes, plan.policy.max_chunk_bytes - plan.policy.minimum_tail_bytes};
  }
  if (rebalance_tail && chunk_index == plan.chunk_count - 1) {
    return {chunk_index * plan.policy.max_chunk_bytes - plan.policy.minimum_tail_bytes,
            static_cast<uint32_t>(tail_bytes + plan.policy.minimum_tail_bytes)};
  }

  const uint64_t offset_bytes = chunk_index * plan.policy.max_chunk_bytes;
  const uint64_t remaining_bytes = plan.total_bytes - offset_bytes;
  const uint32_t size_bytes = remaining_bytes < plan.policy.max_chunk_bytes ? static_cast<uint32_t>(remaining_bytes)
                                                                            : plan.policy.max_chunk_bytes;
  return {offset_bytes, size_bytes};
}

/**
 * @brief Evenly assign a contiguous work-item range to one worker.
 * @param total_items Number of work items in the consumer-owned index space.
 * @param worker_count Nonzero worker count.
 * @param worker_index Worker index in [0, worker_count).
 * @pre The Host consumer validates worker_count and worker_index before Kernel launch.
 * @return A contiguous partition; surplus workers receive an empty partition after the final item.
 */
HP_SHMEM_CHUNKING_HOST_DEVICE WorkPartition partition_work(uint64_t total_items, uint32_t worker_count,
                                                           uint32_t worker_index) {
  const uint64_t base_items = total_items / worker_count;
  const uint64_t extra_items = total_items % worker_count;
  const uint64_t preceding_extra_items = worker_index < extra_items ? worker_index : extra_items;
  return {worker_index * base_items + preceding_extra_items, base_items + (worker_index < extra_items ? 1 : 0)};
}

/**
 * @brief Divide capacity equally and round each partition down to an alignment.
 * @param total_capacity_bytes Total consumer-owned capacity in Byte.
 * @param partition_count Nonzero number of equal-capacity partitions.
 * @param alignment_bytes Nonzero capacity alignment in Byte.
 * @return Per-partition capacity rounded down without exceeding the total capacity.
 */
HP_SHMEM_CHUNKING_HOST_DEVICE constexpr uint32_t divide_aligned_capacity(uint32_t total_capacity_bytes,
                                                                         uint32_t partition_count,
                                                                         uint32_t alignment_bytes) {
  const uint32_t unaligned_capacity = total_capacity_bytes / partition_count;
  return unaligned_capacity - unaligned_capacity % alignment_bytes;
}

}  // namespace hyper_parallel::multicore::shmem::data_plane

#undef HP_SHMEM_CHUNKING_HOST_DEVICE
