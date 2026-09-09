/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/runtime.h"

#include <cstdint>
#include <string>

#include "fakes/fake_host_surface.h"
#include "test_support.h"

namespace hyper_parallel::multicore::shmem::tests {
namespace {

using runtime::AllocationRecord;
using runtime::AllocationSpec;
using runtime::AllocationView;
using runtime::Config;
using runtime::DataEngine;
using runtime::DfxOperation;
using runtime::DfxPhase;
using runtime::ErrorCode;
using runtime::RootWorldInfo;
using runtime::Runtime;
using runtime::State;
using runtime::Status;

constexpr int32_t kDeviceIndex = 0;

Config DefaultConfig() {
  return Config{runtime::kDefaultHeapSizeBytes, runtime::kDefaultTimeoutSeconds, DataEngine::Mte,
                std::string(runtime::kDefaultBootstrapEndpoint)};
}

void InitializeReadyRuntime() {
  fake_host::Reset();
  const Status status =
    Runtime::Instance().Initialize(RootWorldInfo{0, 1}, DefaultConfig(), runtime::kDefaultBootstrapEndpoint);
  CheckEq(status.error_code, ErrorCode::Ok, "Runtime initialization error code");
}

AllocationRecord AllocateOne(uint64_t bytes = 128U) {
  const auto allocation = Runtime::Instance().Allocate(AllocationSpec{bytes, 0U});
  Check(allocation.ok(), "Runtime Allocation must succeed for this test");
  return allocation.value();
}

AllocationView CompleteView(const AllocationRecord &record) {
  return {record.allocation_id, record.allocation_base, record.allocation_bytes, kDeviceIndex};
}

void CheckLatestFreeFailure(DfxPhase expected_phase, ErrorCode expected_error) {
  const auto snapshot = Runtime::Instance().DebugState();
  Check(snapshot.latest_failure.has_value(), "Free failure must be visible in the latest DFX failure");
  CheckEq(snapshot.latest_failure->operation, DfxOperation::Free, "Latest Free failure operation");
  CheckEq(snapshot.latest_failure->phase, expected_phase, "Latest Free failure phase");
  CheckEq(snapshot.latest_failure->status.error_code, expected_error, "Latest Free failure error code");
}

}  // namespace

void test_runtime_free_device_guard_phase_classification() {
  InitializeReadyRuntime();
  const AllocationRecord record = AllocateOne();
  const AllocationView view = CompleteView(record);

  fake_host::Get().current_device_status = fake_host::Error(ErrorCode::CannError, "aclrtGetDevice failed", 501U);
  Status status = Runtime::Instance().Free(view);
  CheckEq(status.error_code, ErrorCode::CannError, "Free ACL device-query failure error code");
  CheckLatestFreeFailure(DfxPhase::CannCall, ErrorCode::CannError);

  fake_host::Get().current_device_status = Status{};
  fake_host::Get().current_device_index = kDeviceIndex + 1;
  status = Runtime::Instance().Free(view);
  CheckEq(status.error_code, ErrorCode::InvalidArgument, "Free device-drift error code");
  CheckLatestFreeFailure(DfxPhase::Validation, ErrorCode::InvalidArgument);

  fake_host::Get().current_device_index = kDeviceIndex;
  CheckEq(Runtime::Instance().Free(view).error_code, ErrorCode::Ok,
          "Free retry error code after restoring the Runtime-bound device");
  CheckEq(Runtime::Instance().Shutdown().error_code, ErrorCode::Ok, "Runtime shutdown error code");
}

void test_runtime_free_failure_preserves_record() {
  InitializeReadyRuntime();
  const AllocationRecord record = AllocateOne(256U);
  const AllocationView view = CompleteView(record);

  fake_host::Get().current_device_index = kDeviceIndex + 1;
  const Status failure = Runtime::Instance().Free(view);
  CheckEq(failure.error_code, ErrorCode::InvalidArgument, "Rejected Free error code after device drift");
  CheckEq(fake_host::Get().free_calls, 0U, "CANN free calls after rejected Free");
  const auto failed_snapshot = Runtime::Instance().DebugState();
  CheckEq(failed_snapshot.state, State::Ready, "Runtime state after rejected Free");
  Check(failed_snapshot.allocated_count.has_value(),
        "Rejected Free snapshot must contain the active Allocation count");
  CheckEq(*failed_snapshot.allocated_count, 1U, "Active Allocation count after rejected Free");
  Check(failed_snapshot.allocated_bytes.has_value(),
        "Rejected Free snapshot must contain the active Allocation bytes");
  CheckEq(*failed_snapshot.allocated_bytes, record.allocation_bytes,
          "Active Allocation bytes after rejected Free");

  fake_host::Get().current_device_index = kDeviceIndex;
  CheckEq(Runtime::Instance().Free(view).error_code, ErrorCode::Ok,
          "Free retry error code after restoring the Runtime-bound device");
  CheckEq(fake_host::Get().free_calls, 1U, "CANN free calls after successful retry");
  const auto successful_snapshot = Runtime::Instance().DebugState();
  Check(successful_snapshot.allocated_count.has_value(),
        "Successful Free snapshot must contain the active Allocation count");
  CheckEq(*successful_snapshot.allocated_count, 0U, "Allocated count after successful retry");
  CheckEq(Runtime::Instance().Shutdown().error_code, ErrorCode::Ok, "Runtime shutdown error code");
}

void test_runtime_initialize_failure_allows_retry() {
  fake_host::Reset();
  fake_host::Get().initialize_status = fake_host::Error(ErrorCode::CannError, "aclshmemx_init_attr failed", 507335);
  const Status status =
    Runtime::Instance().Initialize(RootWorldInfo{0, 1}, DefaultConfig(), runtime::kDefaultBootstrapEndpoint);
  CheckEq(status.error_code, ErrorCode::CannError, "Runtime initialization failure error code");
  const auto snapshot = Runtime::Instance().DebugState();
  CheckEq(snapshot.state, State::Uninitialized,
          "CANN initialization failure must roll the Runtime back to Uninitialized");
  Check(!snapshot.root.has_value(), "Initialization failure must not retain Root state");
  Check(!snapshot.device_identity.has_value(), "Initialization failure must not retain device state");
  Check(!snapshot.config.has_value(), "Initialization failure must not retain Config state");
  Check(!snapshot.allocated_count.has_value(), "Initialization failure must not retain Allocation count state");
  Check(!snapshot.allocated_bytes.has_value(), "Initialization failure must not retain Allocation byte state");
  Check(!snapshot.max_allocated_bytes.has_value(),
        "Initialization failure must not retain max active Allocation bytes state");
  Check(snapshot.latest_failure.has_value(), "Initialization failure must be available to DFX");
  CheckEq(snapshot.latest_failure->operation, DfxOperation::Initialize, "Latest initialization failure operation");
  CheckEq(snapshot.latest_failure->phase, DfxPhase::CannCall, "Latest initialization failure phase");

  const auto allocation = Runtime::Instance().Allocate(AllocationSpec{64U, 0U});
  Check(!allocation.ok(), "Uninitialized Runtime must reject Allocation operations");
  CheckEq(allocation.error().error_code, ErrorCode::InvalidState, "Allocation error code in Uninitialized Runtime");
  CheckEq(fake_host::Get().finalize_calls, 0U, "CANN finalize calls after a rolled-back initialization");

  fake_host::Get().initialize_status = Status{};
  const Status retry =
    Runtime::Instance().Initialize(RootWorldInfo{0, 1}, DefaultConfig(), runtime::kDefaultBootstrapEndpoint);
  CheckEq(retry.error_code, ErrorCode::Ok, "Initialization retry error code");
  CheckEq(Runtime::Instance().DebugState().state, State::Ready, "Runtime state after the initialization retry");
  CheckEq(fake_host::Get().initialize_calls, 2U, "CANN initialize calls across the failure and the retry");
  CheckEq(Runtime::Instance().Shutdown().error_code, ErrorCode::Ok, "Runtime shutdown error code after the retry");
}

void test_runtime_shutdown_uninitialized_idempotent() {
  InitializeReadyRuntime();
  CheckEq(Runtime::Instance().Shutdown().error_code, ErrorCode::Ok, "Ready Runtime shutdown error code");
  CheckEq(Runtime::Instance().DebugState().state, State::Uninitialized,
          "Clean shutdown must return Runtime to Uninitialized for another lifecycle");
  CheckEq(fake_host::Get().finalize_calls, 1U, "CANN finalize calls after Ready Runtime shutdown");

  CheckEq(Runtime::Instance().Shutdown().error_code, ErrorCode::Ok,
          "Shutdown while Uninitialized must be a local no-op");
  CheckEq(fake_host::Get().finalize_calls, 1U, "CANN finalize calls after repeated Runtime shutdown");
}

}  // namespace hyper_parallel::multicore::shmem::tests
