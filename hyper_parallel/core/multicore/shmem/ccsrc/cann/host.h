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
#include <string_view>

#include "cann/signal.h"
#include "device/device_model.h"
#include "runtime/config.h"
#include "runtime/types.h"

namespace hyper_parallel::multicore::shmem::cann {

/**
 * @brief Complete framework-independent input for one CANN SHMEM initialization call.
 *
 * The Host Surface consumes all fields synchronously and retains no string_view or Runtime state.
 */
struct InitOptions {
  int32_t root_rank;
  int32_t root_size;
  uint64_t heap_size_bytes;
  uint32_t timeout_seconds;
  runtime::DataEngine data_engine;
  std::string_view effective_bootstrap_endpoint;
};

namespace host {

/**
 * @brief Initialize one process-local CANN SHMEM Root WORLD and fixed symmetric Heap.
 *
 * root_rank uses CANN Root WORLD coordinates. DataEngine::Mte maps to the public MTE-only data-operation mask.
 */
runtime::Status initialize(const InitOptions &options);

/** @brief Finalize the initialized process-local CANN SHMEM Runtime. */
runtime::Status finalize();

/** @brief Allocate a positive byte count from the CANN symmetric Heap. */
runtime::Result<uintptr_t> allocate(uint64_t bytes);

/** @brief Allocate bytes with an explicit power-of-two base alignment, both measured in Byte. */
runtime::Result<uintptr_t> aligned_allocate(uint64_t alignment_bytes, uint64_t bytes);

/** @brief Free one complete symmetric Allocation identified by its nonzero local base address. */
void free_memory(uintptr_t allocation_base);

/**
 * @brief Enqueue a Root WORLD barrier on an opaque NPU Stream.
 *
 * Return means Host enqueue completed; it does not mean the device barrier has completed.
 */
void barrier_on_stream(const runtime::StreamView &stream);

/** @brief Enqueue a byte Put from local GM to a symmetric address on one target Root PE. */
void put_on_stream(uintptr_t remote_dst, uintptr_t local_src, uint64_t bytes, int32_t target_pe,
                   const runtime::StreamView &stream);

/** @brief Enqueue a byte Get from a symmetric address on one source Root PE to local GM. */
void get_on_stream(uintptr_t local_dst, uintptr_t remote_src, uint64_t bytes, int32_t source_pe,
                   const runtime::StreamView &stream);

/**
 * @brief Enqueue one Set or Add update to a symmetric Signal on a target Root PE.
 *
 * Host return only confirms
 * enqueue. Device errors surface when the caller synchronizes the Stream.
 */
void signal_on_stream(uintptr_t remote_signal, int32_t value, SignalOp operation, int32_t target_pe,
                      const runtime::StreamView &stream);

/**
 * @brief Enqueue a wait until one local symmetric Signal satisfies comparison.
 */
void wait_signal_on_stream(uintptr_t local_signal, CompareOp comparison, int32_t value,
                           const runtime::StreamView &stream);

/**
 * @brief Query the current ACL device index.
 *
 * This function observes the caller-selected device without changing device context or retaining state.
 *
 * @return The current device index, or CANN_ERROR with the original ACL error code when the query fails.
 */
runtime::Result<int32_t> query_current_device_index();

/**
 * @brief Query the current ACL device and its raw SoC name once.
 *
 * @return Current device index, owned SoC name, and normalized model; or an error if either query fails.
 */
runtime::Result<DeviceIdentity> query_device_identity();

}  // namespace host
}  // namespace hyper_parallel::multicore::shmem::cann
