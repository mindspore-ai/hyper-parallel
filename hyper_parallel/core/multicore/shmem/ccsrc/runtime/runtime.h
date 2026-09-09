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
#include <mutex>
#include <optional>
#include <string_view>

#include "device/device_model.h"
#include "runtime/allocation.h"
#include "runtime/config.h"
#include "runtime/dfx.h"
#include "runtime/types.h"

namespace hyper_parallel::multicore::shmem::runtime {

/**
 * @brief Own the single process-local CANN SHMEM lifecycle and Allocation metadata.
 *
 * Runtime does not own framework bootstrap state. Its destructor performs no collective cleanup; the framework
 * lifecycle layer must call Shutdown() explicitly after establishing quiescence.
 */
class Runtime final {
 public:
  /** @brief Return the process-wide Runtime instance without initializing CANN SHMEM. */
  static Runtime &Instance();

  /**
   * @brief Initialize one process-local CANN SHMEM Root WORLD lifecycle.
   *
   * @param root Expected CANN Root WORLD rank and size.
   * @param config Validated Runtime configuration copied into Runtime state.
   * @param effective_bootstrap_endpoint Initialization-only TCP endpoint consumed synchronously by CANN.
   * @return OK from Uninitialized. An initialization failure rolls Runtime back to Uninitialized so the caller can
   * retry; a clean Shutdown returns Runtime to Uninitialized for another lifecycle.
   */
  Status Initialize(const RootWorldInfo &root, const Config &config, std::string_view effective_bootstrap_endpoint);

  /** @brief Locally return the CANN Root WORLD rank while Ready, without CANN access or DFX updates. */
  Result<int32_t> RootRank() const;

  /** @brief Locally return the CANN Root WORLD size while Ready, without CANN access or DFX updates. */
  Result<int32_t> RootSize() const;

  /** @brief Locally return the frozen device model while Ready, without CANN access or DFX updates. */
  Result<DeviceModel> QueryDeviceModel() const;

  /**
   * @brief Validate local preconditions for closing the active Runtime lifecycle.
   *
   * The current ACL device must match the initialized device and every symmetric Allocation must already be freed.
   * This local guard observes but never changes the caller's device context or calls CANN finalize.
   */
  Status ValidateShutdown();

  /**
   * @brief Collectively allocate one symmetric-memory range from the Root WORLD Heap.
   *
   * bytes and alignment_bytes are measured in Byte. Zero alignment selects ordinary allocation; a positive alignment
   * must be a power of two. Before entering the CANN allocator, Runtime verifies that the caller-selected ACL device
   * matches the device frozen at initialization. All Root PEs must issue collective Allocations in the same order with
   * the same arguments.
   */
  Result<AllocationRecord> Allocate(const AllocationSpec &spec);

  /**
   * @brief Collectively free one complete active symmetric Allocation.
   *
   * The caller must establish Stream quiescence before entry and keep the Runtime-bound ACL device current. A Tensor
   * subview cannot be freed independently.
   */
  Status Free(const AllocationView &buffer);

  /**
   * @brief Resolve a contiguous Buffer view to its complete active Allocation record.
   *
   * This local read-only query performs no CANN call and does not update DFX on failure.
   */
  Result<AllocationRecord> ResolveAllocation(const AllocationView &buffer) const;

  /**
   * @brief Enqueue a Root WORLD barrier on a non-owning NPU Stream.
   *
   * Success means Host enqueue returned; Host-visible completion requires the framework to synchronize the Stream.
   */
  Status Barrier(const StreamView &stream);

  /** @brief Copy current Runtime facts and the latest Runtime-owned failure without accessing CANN. */
  DfxSnapshot DebugState() const;

  /**
   * @brief End the process Runtime lifecycle after framework quiescence.
   *
   * Ready with no active Allocations calls CANN finalize once and returns to Uninitialized on success. Active
   * Allocations reject shutdown before finalize. A finalize failure becomes terminal (Finalized) because the CANN
   * state is then indeterminate. Calling Shutdown while Uninitialized or Finalized is a local no-op.
   */
  Status Shutdown();

  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;
  Runtime(Runtime &&) = delete;
  Runtime &operator=(Runtime &&) = delete;

 private:
  Runtime() = default;
  ~Runtime() = default;

  Status ValidateInitialInput(const RootWorldInfo &root, const Config &config,
                              std::string_view effective_bootstrap_endpoint) const;
  // Precondition: the caller holds mutex_ and State::Ready guarantees root_, device_identity_, and allocations_.
  Status ValidateCurrentDeviceLocked() const;
  Status RequireReady() const;
  int32_t RootRankForDfx() const noexcept;
  void RecordFailure(DfxOperation operation, DfxPhase phase, int32_t root_rank, const Status &error, const char *file,
                     int line);
  Status FailInitialization(DfxPhase phase, int32_t root_rank, const Status &error, const char *file, int line);

  mutable std::mutex mutex_;
  State state_{State::Uninitialized};
  // State::Ready implies that all four authoritative values below are engaged.
  std::optional<DeviceIdentity> device_identity_;
  std::optional<RootWorldInfo> root_;
  std::optional<AllocationRegistry> allocations_;
  std::optional<Config> config_;
  std::optional<DfxFailure> latest_failure_;
  uint64_t next_allocation_id_{1};
};

}  // namespace hyper_parallel::multicore::shmem::runtime
