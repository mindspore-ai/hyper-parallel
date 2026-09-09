/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fake_host_surface.h"

#include <optional>
#include <utility>

namespace hyper_parallel::multicore::shmem::tests::fake_host {
namespace {

State state;

}  // namespace

State &Get() { return state; }

void Reset() { state = State{}; }

runtime::Status Error(runtime::ErrorCode error_code, std::string message, int32_t cann_error_code) {
  const std::optional<int32_t> raw_code =
    error_code == runtime::ErrorCode::CannError || error_code == runtime::ErrorCode::Timeout
      ? std::optional<int32_t>(cann_error_code)
      : std::nullopt;
  return runtime::Status{error_code, raw_code, std::move(message)};
}

}  // namespace hyper_parallel::multicore::shmem::tests::fake_host

namespace hyper_parallel::multicore::shmem::cann::host {

runtime::Status initialize(const InitOptions &) {
  ++tests::fake_host::Get().initialize_calls;
  return tests::fake_host::Get().initialize_status;
}

runtime::Status finalize() {
  ++tests::fake_host::Get().finalize_calls;
  return tests::fake_host::Get().finalize_status;
}

runtime::Result<uintptr_t> allocate(uint64_t bytes) {
  auto &state = tests::fake_host::Get();
  ++state.allocate_calls;
  state.last_allocate_bytes = bytes;
  if (state.allocation_status.error_code != runtime::ErrorCode::Ok) {
    return runtime::Result<uintptr_t>::Failure(state.allocation_status);
  }
  return runtime::Result<uintptr_t>::Success(state.next_allocation_base);
}

runtime::Result<uintptr_t> aligned_allocate(uint64_t alignment_bytes, uint64_t bytes) {
  auto &state = tests::fake_host::Get();
  ++state.aligned_allocate_calls;
  state.last_alignment_bytes = alignment_bytes;
  state.last_allocate_bytes = bytes;
  if (state.aligned_allocation_status.error_code != runtime::ErrorCode::Ok) {
    return runtime::Result<uintptr_t>::Failure(state.aligned_allocation_status);
  }
  return runtime::Result<uintptr_t>::Success(state.next_allocation_base);
}

void free_memory(uintptr_t allocation_base) {
  auto &state = tests::fake_host::Get();
  ++state.free_calls;
  state.last_freed_address = allocation_base;
}

void barrier_on_stream(const runtime::StreamView &stream) {
  auto &state = tests::fake_host::Get();
  ++state.barrier_calls;
  state.last_barrier_stream = stream;
}

runtime::Result<int32_t> query_current_device_index() {
  auto &state = tests::fake_host::Get();
  ++state.current_device_calls;
  if (state.current_device_status.error_code != runtime::ErrorCode::Ok) {
    return runtime::Result<int32_t>::Failure(state.current_device_status);
  }
  return runtime::Result<int32_t>::Success(state.current_device_index);
}

runtime::Result<DeviceIdentity> query_device_identity() {
  const auto &state = tests::fake_host::Get();
  if (state.device_identity_status.error_code != runtime::ErrorCode::Ok) {
    return runtime::Result<DeviceIdentity>::Failure(state.device_identity_status);
  }
  return runtime::Result<DeviceIdentity>::Success(state.device_identity);
}

}  // namespace hyper_parallel::multicore::shmem::cann::host
