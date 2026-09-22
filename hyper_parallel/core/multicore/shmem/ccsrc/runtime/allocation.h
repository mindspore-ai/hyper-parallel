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
#include <unordered_map>
#include <vector>

#include "runtime/types.h"

namespace hyper_parallel::multicore::shmem::runtime {

/** @brief Input for one Root WORLD collective symmetric Allocation. */
struct AllocationSpec {
  uint64_t bytes;
  uint64_t alignment_bytes;
};

/** @brief Runtime-owned facts for one complete active symmetric Allocation. */
struct AllocationRecord {
  uint64_t allocation_id;
  uintptr_t allocation_base;
  uint64_t allocation_bytes;
  uint64_t alignment_bytes;
};

/** @brief Non-owning contiguous Tensor or Tensor-view byte range supplied by a framework binding. */
struct AllocationView {
  uint64_t allocation_id;
  uintptr_t view_data;
  uint64_t view_bytes;
  int32_t device_index;
};

/**
 * @brief Own process-local metadata for active symmetric Allocations on one NPU.
 *
 * This registry never calls CANN and has no internal synchronization. Runtime serializes each validation, CANN call,
 * and metadata update with its Host mutex.
 */
class AllocationRegistry final {
 public:
  /** @brief Create an empty registry bound to device_index and the next process-wide Allocation identity. */
  AllocationRegistry(int32_t device_index, uint64_t next_allocation_id);
  AllocationRegistry(const AllocationRegistry &) = delete;
  AllocationRegistry &operator=(const AllocationRegistry &) = delete;
  AllocationRegistry(AllocationRegistry &&) = delete;
  AllocationRegistry &operator=(AllocationRegistry &&) = delete;

  /**
   * @brief Validate Runtime-owned Allocation inputs without changing state.
   *
   * @return OK for a positive byte count, valid optional alignment, and available identity.
   */
  Status Validate(const AllocationSpec &spec) const;

  /**
   * @brief Register an address successfully returned by CANN and assign a non-reusable identity.
   *
   * @param spec Request previously accepted by Validate().
   * @param allocation_base Nonzero address accepted as authoritative CANN Allocation output.
   * @return A copy of the new active record.
   * @throws std::logic_error If an internal identity invariant is broken.
   */
  AllocationRecord Register(const AllocationSpec &spec, uintptr_t allocation_base);

  /**
   * @brief Match a view against the complete active Allocation required by free.
   *
   * @return The active record, DOUBLE_FREE for a released identity, or INVALID_ARGUMENT for an unknown identity or
   * mismatched complete range.
   */
  Result<AllocationRecord> MatchForFree(const AllocationView &buffer) const;

  /**
   * @brief Resolve a contiguous full Allocation or subview to its complete active Allocation record.
   *
   * Range validation avoids overflowing address addition. The returned record is an owned copy.
   */
  Result<AllocationRecord> Resolve(const AllocationView &buffer) const;

  /** @brief Remove an active record after its CANN free has completed. */
  void Remove(uint64_t allocation_id);

  /** @brief Return the number of active Allocations. */
  uint64_t allocated_count() const noexcept;

  /** @brief Return the sum of requested bytes for active Allocations. */
  uint64_t allocated_bytes() const noexcept;

  /** @brief Return the largest active requested-byte sum observed by this registry. */
  uint64_t max_allocated_bytes() const noexcept;

  /** @brief Return the next process-wide identity to preserve across clean Runtime lifecycles. */
  uint64_t next_allocation_id() const noexcept;

  /** @brief Return the active records ordered by Allocation identity for diagnostics. */
  std::vector<AllocationRecord> ActiveRecords() const;

 private:
  Result<AllocationRecord> Find(uint64_t allocation_id) const;

  int32_t device_index_;
  uint64_t next_allocation_id_;
  uint64_t allocated_bytes_{0};
  uint64_t max_allocated_bytes_{0};
  std::unordered_map<uint64_t, AllocationRecord> records_;
};

}  // namespace hyper_parallel::multicore::shmem::runtime
