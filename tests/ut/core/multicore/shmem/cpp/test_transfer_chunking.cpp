/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_plane/transfer_chunking.h"

#include <array>
#include <cstdint>
#include <string>

#include "test_support.h"

namespace hyper_parallel::multicore::shmem::tests {
namespace {

using data_plane::ChunkPlan;
using data_plane::ChunkPolicy;
using data_plane::TransferChunk;

constexpr ChunkPolicy kPolicy{8192U, 32U};

void CheckCompletePlan(const ChunkPlan &plan) {
  uint64_t next_offset = 0;
  uint64_t covered_bytes = 0;
  for (uint64_t chunk_index = 0; chunk_index < plan.chunk_count; ++chunk_index) {
    const TransferChunk chunk = data_plane::get_chunk(plan, chunk_index);
    const std::string context = "Chunk (chunk_index=" + std::to_string(chunk_index) + ")";
    CheckEq(chunk.offset_bytes, next_offset, context + " offset");
    Check(chunk.size_bytes != 0, context + " must not be empty");
    Check(chunk.size_bytes <= plan.policy.max_chunk_bytes, context + " must not exceed max_chunk_bytes");
    next_offset += chunk.size_bytes;
    covered_bytes += chunk.size_bytes;
  }
  CheckEq(covered_bytes, plan.total_bytes, "Total bytes covered by Chunks");
}

}  // namespace

void test_make_chunk_plan_chunk_count() {
  struct Case {
    uint64_t total_bytes;
    uint64_t expected_chunks;
  };
  constexpr std::array<Case, 4> kCases{{{0U, 0U}, {1U, 1U}, {8192U, 1U}, {8193U, 2U}}};
  for (const Case &test_case : kCases) {
    const ChunkPlan plan = data_plane::make_chunk_plan(test_case.total_bytes, kPolicy);
    const std::string context = "Chunk count (total_bytes=" + std::to_string(test_case.total_bytes) + ")";
    CheckEq(plan.chunk_count, test_case.expected_chunks, context);
  }
}

void test_divide_aligned_capacity() {
  CheckEq(data_plane::divide_aligned_capacity(16384U, 2U, 32U), 8192U, "Evenly divided aligned capacity");
  CheckEq(data_plane::divide_aligned_capacity(1000U, 3U, 64U), 320U,
          "Unevenly divided capacity rounded down to alignment");
}

void test_get_chunk_tail_rebalance() {
  struct Case {
    uint64_t total_bytes;
    uint32_t expected_penultimate_bytes;
    uint32_t expected_tail_bytes;
  };
  constexpr std::array<Case, 3> kCases{{{8193U, 8160U, 33U}, {8223U, 8160U, 63U}, {8224U, 8192U, 32U}}};
  for (const Case &test_case : kCases) {
    const ChunkPlan plan = data_plane::make_chunk_plan(test_case.total_bytes, kPolicy);
    const TransferChunk penultimate = data_plane::get_chunk(plan, plan.chunk_count - 2U);
    const TransferChunk tail = data_plane::get_chunk(plan, plan.chunk_count - 1U);
    const std::string context = "Tail rebalance (total_bytes=" + std::to_string(test_case.total_bytes) + ")";
    CheckEq(penultimate.size_bytes, test_case.expected_penultimate_bytes, context + " penultimate Chunk size");
    CheckEq(tail.size_bytes, test_case.expected_tail_bytes, context + " tail Chunk size");
    CheckEq(tail.offset_bytes, penultimate.offset_bytes + penultimate.size_bytes, context + " tail Chunk offset");
    CheckCompletePlan(plan);
  }
}

void test_partition_work_remainder_distribution() {
  constexpr uint64_t kTotalItems = 10U;
  constexpr uint32_t kWorkerCount = 3U;
  constexpr std::array<uint64_t, kWorkerCount> kExpectedFirst{{0U, 4U, 7U}};
  constexpr std::array<uint64_t, kWorkerCount> kExpectedCount{{4U, 3U, 3U}};

  uint64_t next_item = 0;
  for (uint32_t worker_index = 0; worker_index < kWorkerCount; ++worker_index) {
    const auto partition = data_plane::partition_work(kTotalItems, kWorkerCount, worker_index);
    const std::string context = "Work partition (worker_index=" + std::to_string(worker_index) + ")";
    CheckEq(partition.first_item, kExpectedFirst[worker_index], context + " first item");
    CheckEq(partition.item_count, kExpectedCount[worker_index], context + " item count");
    CheckEq(partition.first_item, next_item, context + " contiguous first item");
    next_item += partition.item_count;
  }
  CheckEq(next_item, kTotalItems, "Total work items covered by partitions");
}

}  // namespace hyper_parallel::multicore::shmem::tests
