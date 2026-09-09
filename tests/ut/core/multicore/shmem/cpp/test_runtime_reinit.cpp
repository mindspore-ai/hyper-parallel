/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cann/host.h"
#include "runtime/log.h"
#include "runtime/runtime.h"

namespace hp_shmem = hyper_parallel::multicore::shmem;
namespace cann = hp_shmem::cann;
namespace runtime = hp_shmem::runtime;

namespace {

constexpr uint64_t kFirstHeapBytes = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t kSecondHeapBytes = 512ULL * 1024ULL * 1024ULL;
constexpr uintptr_t kReusedAllocationBase = 0x100000U;

struct FakeHostState {
  std::vector<uint64_t> initialized_heap_bytes;
  runtime::Status initialize_status;
  runtime::Status finalize_status;
  uint64_t finalize_calls{0};
  uint64_t free_calls{0};
};

FakeHostState g_host;

runtime::Status Error(runtime::ErrorCode error_code, std::string message) {
  return runtime::Status{error_code, std::nullopt, std::move(message)};
}

runtime::Config Config(uint64_t heap_size_bytes) {
  return runtime::Config{heap_size_bytes, runtime::kDefaultTimeoutSeconds, runtime::DataEngine::Mte,
                         std::string(runtime::kDefaultBootstrapEndpoint)};
}

void Check(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void Initialize(uint64_t heap_size_bytes) {
  const runtime::Status status = runtime::Runtime::Instance().Initialize(
    runtime::RootWorldInfo{0, 1}, Config(heap_size_bytes), runtime::kDefaultBootstrapEndpoint);
  Check(status.error_code == runtime::ErrorCode::Ok, "Runtime initialization failed");
}

runtime::AllocationView View(const runtime::AllocationRecord &record) {
  return runtime::AllocationView{record.allocation_id, record.allocation_base, record.allocation_bytes, 0};
}

void TestCleanReinitAndHeapForwarding() {
  Initialize(kFirstHeapBytes);
  const runtime::DfxSnapshot ready_snapshot = runtime::Runtime::Instance().DebugState();
  Check(ready_snapshot.config.has_value() && ready_snapshot.config->heap_size_bytes == kFirstHeapBytes,
        "Ready Runtime did not expose the effective config");
  const auto first = runtime::Runtime::Instance().Allocate(runtime::AllocationSpec{64U, 0U});
  Check(first.ok(), "First-cycle Allocation failed");
  Check(runtime::Runtime::Instance().DebugState().max_allocated_bytes == 64U,
        "First-cycle max allocated bytes was not recorded");
  const runtime::AllocationView old_view = View(first.value());
  Check(runtime::Runtime::Instance().Free(old_view).error_code == runtime::ErrorCode::Ok, "First-cycle free failed");
  Check(runtime::Runtime::Instance().Shutdown().error_code == runtime::ErrorCode::Ok, "First-cycle shutdown failed");
  const runtime::DfxSnapshot clean_snapshot = runtime::Runtime::Instance().DebugState();
  Check(clean_snapshot.state == runtime::State::Uninitialized,
        "Clean shutdown did not return Runtime to Uninitialized");
  Check(!clean_snapshot.root.has_value(), "Clean shutdown retained Root identity");
  Check(!clean_snapshot.device_identity.has_value(), "Clean shutdown retained device identity");
  Check(!clean_snapshot.config.has_value(), "Clean shutdown retained the effective config");
  Check(!clean_snapshot.allocated_count.has_value(), "Clean shutdown retained active Allocation count");
  Check(!clean_snapshot.allocated_bytes.has_value(), "Clean shutdown retained active Allocation bytes");
  Check(!clean_snapshot.max_allocated_bytes.has_value(),
        "Clean shutdown retained the max allocated bytes");
  Check(!clean_snapshot.latest_failure.has_value(), "Clean shutdown retained latest failure");

  Initialize(kSecondHeapBytes);
  const runtime::DfxSnapshot second_initial_snapshot = runtime::Runtime::Instance().DebugState();
  Check(
    second_initial_snapshot.config.has_value() && second_initial_snapshot.config->heap_size_bytes == kSecondHeapBytes,
    "Second-cycle config did not reflect the new heap size");
  Check(second_initial_snapshot.allocated_count == 0U, "Second-cycle allocated count did not reset");
  Check(second_initial_snapshot.allocated_bytes == 0U, "Second-cycle allocated bytes did not reset");
  Check(second_initial_snapshot.max_allocated_bytes == 0U,
        "Second-cycle max allocated bytes did not reset");
  const auto second = runtime::Runtime::Instance().Allocate(runtime::AllocationSpec{64U, 0U});
  Check(second.ok(), "Second-cycle Allocation failed");
  Check(second.value().allocation_id > first.value().allocation_id,
        "Allocation identity was reused across Runtime lifecycles");
  Check(second.value().allocation_base == first.value().allocation_base,
        "Fake Host did not exercise CANN address reuse");
  Check(runtime::Runtime::Instance().Free(old_view).error_code == runtime::ErrorCode::DoubleFree,
        "Old Tensor identity matched a new-cycle Allocation");
  Check(runtime::Runtime::Instance().Free(View(second.value())).error_code == runtime::ErrorCode::Ok,
        "Second-cycle free failed");
  Check(runtime::Runtime::Instance().Shutdown().error_code == runtime::ErrorCode::Ok, "Second-cycle shutdown failed");

  Initialize(kFirstHeapBytes);
  const auto third = runtime::Runtime::Instance().Allocate(runtime::AllocationSpec{32U, 0U});
  Check(third.ok(), "Third-cycle Allocation failed");
  Check(third.value().allocation_id > second.value().allocation_id,
        "Allocation identity was reused in the third Runtime lifecycle");
  Check(runtime::Runtime::Instance().DebugState().max_allocated_bytes == 32U,
        "Third-cycle max allocated bytes did not start from zero");
  Check(runtime::Runtime::Instance().Free(View(third.value())).error_code == runtime::ErrorCode::Ok,
        "Third-cycle free failed");
  Check(runtime::Runtime::Instance().Shutdown().error_code == runtime::ErrorCode::Ok, "Third-cycle shutdown failed");
  Check(g_host.initialized_heap_bytes == std::vector<uint64_t>({kFirstHeapBytes, kSecondHeapBytes, kFirstHeapBytes}),
        "Heap sizes were not forwarded independently for each lifecycle");
  Check(g_host.finalize_calls == 3, "Each clean lifecycle must finalize exactly once");
}

void TestActiveAllocationRejectsShutdown() {
  Initialize(kFirstHeapBytes);
  const auto allocation = runtime::Runtime::Instance().Allocate(runtime::AllocationSpec{64U, 0U});
  Check(allocation.ok(), "Allocation failed");
  const runtime::Status validation = runtime::Runtime::Instance().ValidateShutdown();
  Check(validation.error_code == runtime::ErrorCode::InvalidState, "Shutdown validation accepted an active Allocation");
  const runtime::Status shutdown = runtime::Runtime::Instance().Shutdown();
  Check(shutdown.error_code == runtime::ErrorCode::InvalidState, "Shutdown accepted an active Allocation");
  Check(g_host.finalize_calls == 0, "Rejected shutdown entered CANN finalize");
  Check(runtime::Runtime::Instance().DebugState().state == runtime::State::Ready,
        "Rejected shutdown changed Runtime state");

  Check(runtime::Runtime::Instance().Free(View(allocation.value())).error_code == runtime::ErrorCode::Ok,
        "Allocation cleanup failed");
  Check(runtime::Runtime::Instance().Shutdown().error_code == runtime::ErrorCode::Ok,
        "Shutdown retry failed after Allocation cleanup");
}

void TestInitializationFailureAllowsRetry() {
  g_host.initialize_status = Error(runtime::ErrorCode::CannError, "injected initialize failure");
  const runtime::Status first = runtime::Runtime::Instance().Initialize(
    runtime::RootWorldInfo{0, 1}, Config(kFirstHeapBytes), runtime::kDefaultBootstrapEndpoint);
  Check(first.error_code == runtime::ErrorCode::CannError, "Initialization failure was not preserved");
  Check(runtime::Runtime::Instance().DebugState().state == runtime::State::Uninitialized,
        "Initialization failure did not roll back to Uninitialized");
  Check(g_host.finalize_calls == 0, "Initialization failure entered CANN finalize");

  g_host.initialize_status = runtime::Status{};
  const runtime::Status retry = runtime::Runtime::Instance().Initialize(
    runtime::RootWorldInfo{0, 1}, Config(kSecondHeapBytes), runtime::kDefaultBootstrapEndpoint);
  Check(retry.error_code == runtime::ErrorCode::Ok, "Initialization retry was rejected");
  Check(runtime::Runtime::Instance().DebugState().state == runtime::State::Ready,
        "Runtime was not Ready after the initialization retry");
  Check(g_host.initialized_heap_bytes == std::vector<uint64_t>({kFirstHeapBytes, kSecondHeapBytes}),
        "Initialization retry did not forward its own heap size");
  Check(runtime::Runtime::Instance().Shutdown().error_code == runtime::ErrorCode::Ok,
        "Shutdown after the initialization retry failed");
}

void TestFinalizeFailureIsTerminal() {
  Initialize(kFirstHeapBytes);
  g_host.finalize_status = Error(runtime::ErrorCode::CannError, "injected finalize failure");
  const runtime::Status shutdown = runtime::Runtime::Instance().Shutdown();
  Check(shutdown.error_code == runtime::ErrorCode::CannError, "Finalize failure was not preserved");
  Check(runtime::Runtime::Instance().DebugState().state == runtime::State::Finalized,
        "Finalize failure did not enter terminal state");
  const runtime::Status retry = runtime::Runtime::Instance().Initialize(
    runtime::RootWorldInfo{0, 1}, Config(kSecondHeapBytes), runtime::kDefaultBootstrapEndpoint);
  Check(retry.error_code == runtime::ErrorCode::InvalidState, "Finalize failure allowed reinitialization");
}

}  // namespace

namespace hyper_parallel::multicore::shmem::cann::host {

runtime::Status initialize(const InitOptions &options) {
  g_host.initialized_heap_bytes.push_back(options.heap_size_bytes);
  return g_host.initialize_status;
}

runtime::Status finalize() {
  ++g_host.finalize_calls;
  return g_host.finalize_status;
}

runtime::Result<uintptr_t> allocate(uint64_t) { return runtime::Result<uintptr_t>::Success(kReusedAllocationBase); }

runtime::Result<uintptr_t> aligned_allocate(uint64_t, uint64_t) {
  return runtime::Result<uintptr_t>::Success(kReusedAllocationBase);
}

void free_memory(uintptr_t) { ++g_host.free_calls; }

void barrier_on_stream(const runtime::StreamView &) {}

runtime::Result<int32_t> query_current_device_index() { return runtime::Result<int32_t>::Success(0); }

runtime::Result<DeviceIdentity> query_device_identity() {
  return runtime::Result<DeviceIdentity>::Success(DeviceIdentity{0, DeviceModel::kAscend910B, "Ascend910B1"});
}

}  // namespace hyper_parallel::multicore::shmem::cann::host

namespace hyper_parallel::multicore::shmem::runtime::log {

Level ActiveLevel() noexcept { return Level::Error; }

void Line(Level, int32_t, const char *, int, const char *, ...) noexcept {}

void Failure(const char *, int, DfxOperation, DfxPhase, int32_t, const Status &) noexcept {}

}  // namespace hyper_parallel::multicore::shmem::runtime::log

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Expected one case name\n";
    return 2;
  }
  const std::string_view test_case = argv[1];
  try {
    if (test_case == "clean_reinit") {
      TestCleanReinitAndHeapForwarding();
    } else if (test_case == "active_allocation") {
      TestActiveAllocationRejectsShutdown();
    } else if (test_case == "initialize_failure") {
      TestInitializationFailureAllowsRetry();
    } else if (test_case == "finalize_failure") {
      TestFinalizeFailureIsTerminal();
    } else {
      std::cerr << "Unknown case: " << test_case << '\n';
      return 2;
    }
  } catch (const std::exception &error) {
    std::cerr << test_case << " failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
