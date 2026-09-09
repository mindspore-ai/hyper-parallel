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
#include <optional>
#include <vector>

#include "device/device_model.h"
#include "runtime/allocation.h"
#include "runtime/config.h"
#include "runtime/types.h"

namespace hyper_parallel::multicore::shmem::runtime {

/** @brief Runtime operation associated with the latest Runtime-owned failure. */
enum class DfxOperation : uint8_t {
  Initialize,
  Allocate,
  Free,
  Barrier,
  Shutdown,
};

/** @brief Boundary at which the latest Runtime operation failed. */
enum class DfxPhase : uint8_t {
  Validation,
  CannCall,
};

/** @brief Latest failure copied by Runtime, including its Root WORLD location and original status. */
struct DfxFailure {
  DfxOperation operation;
  DfxPhase phase;
  int32_t root_rank;
  Status status;
};

/**
 * @brief Owned snapshot of current process-local Runtime facts.
 * @details Exposes active Allocation identities, base addresses, and byte counts for diagnostics; contains no CANN
 * resources, internal references, or operation history.
 * @post A clean shutdown returns Runtime to Uninitialized and clears every optional lifecycle-specific field.
 */
struct DfxSnapshot {
  State state;
  std::optional<RootWorldInfo> root;
  std::optional<DeviceIdentity> device_identity;
  std::optional<Config> config;
  std::optional<uint64_t> allocated_count;
  std::optional<uint64_t> allocated_bytes;
  std::optional<uint64_t> max_allocated_bytes;
  std::optional<std::vector<AllocationRecord>> active_allocations;
  std::optional<DfxFailure> latest_failure;
};

}  // namespace hyper_parallel::multicore::shmem::runtime
