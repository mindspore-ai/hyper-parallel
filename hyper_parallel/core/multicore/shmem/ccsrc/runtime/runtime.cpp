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

#include <optional>
#include <string>
#include <utility>

#include "cann/host.h"
#include "runtime/log.h"

#define HP_SM_RECORD_FAILURE(operation, phase, root_rank, error)           \
  do {                                                                     \
    RecordFailure(operation, phase, root_rank, error, __FILE__, __LINE__); \
  } while (false)

#define HP_SM_FAIL_INITIALIZATION(phase, root_rank, error) \
  FailInitialization(phase, root_rank, error, __FILE__, __LINE__)

namespace hyper_parallel::multicore::shmem::runtime {
namespace {

constexpr int32_t kUnavailableRootRank = -1;

Status MakeError(ErrorCode error_code, std::string message) {
  return Status{error_code, std::nullopt, std::move(message)};
}

const char *StateName(State state) {
  switch (state) {
    case State::Uninitialized:
      return "Uninitialized";
    case State::Ready:
      return "Ready";
    case State::Finalized:
      return "Finalized";
  }
  return "Unknown";
}

const char *DataEngineName(DataEngine engine) {
  switch (engine) {
    case DataEngine::Mte:
      return "mte";
  }
  return "unknown";
}

DfxPhase PhaseForDeviceStatus(const Status &status) {
  return status.error_code == ErrorCode::CannError || status.error_code == ErrorCode::Timeout ? DfxPhase::CannCall
                                                                                              : DfxPhase::Validation;
}

}  // namespace

Runtime &Runtime::Instance() {
  static Runtime instance;
  return instance;
}

Status Runtime::Initialize(const RootWorldInfo &root, const Config &config,
                           std::string_view effective_bootstrap_endpoint) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (state_ != State::Uninitialized) {
    const Status error =
      MakeError(ErrorCode::InvalidState, "Runtime state must be Uninitialized for initialization, but got state=" +
                                           std::string(StateName(state_)));
    HP_SM_RECORD_FAILURE(DfxOperation::Initialize, DfxPhase::Validation, root.root_rank, error);
    return error;
  }

  const Status validation = ValidateInitialInput(root, config, effective_bootstrap_endpoint);
  if (validation.error_code != ErrorCode::Ok) {
    return HP_SM_FAIL_INITIALIZATION(DfxPhase::Validation, root.root_rank, validation);
  }
  root_.emplace(root);

  const auto device_identity = cann::host::query_device_identity();
  if (!device_identity.ok()) {
    return HP_SM_FAIL_INITIALIZATION(DfxPhase::CannCall, root.root_rank, device_identity.error());
  }
  device_identity_.emplace(device_identity.value());
  if (device_identity_->model == DeviceModel::kAscend950) {
    const Status error = MakeError(
      ErrorCode::UnsupportedCapability,
      "the first Runtime release supports Ascend 910B and 910C, but got soc_name=" + device_identity_->soc_name);
    return HP_SM_FAIL_INITIALIZATION(DfxPhase::Validation, root.root_rank, error);
  }

  allocations_.emplace(device_identity_->device_index, next_allocation_id_);
  config_.emplace(config);

  const cann::InitOptions options{root.root_rank,         root.root_size,     config.heap_size_bytes,
                                  config.timeout_seconds, config.data_engine, effective_bootstrap_endpoint};
  HP_SM_LOG_DEBUG(root.root_rank, "op=Initialize call=aclshmemx_init_attr enter");
  const Status initialize_status = cann::host::initialize(options);
  if (initialize_status.error_code != ErrorCode::Ok) {
    return HP_SM_FAIL_INITIALIZATION(DfxPhase::CannCall, root.root_rank, initialize_status);
  }
  HP_SM_LOG_DEBUG(root.root_rank, "op=Initialize call=aclshmemx_init_attr ok");

  latest_failure_.reset();
  state_ = State::Ready;
  HP_SM_LOG_INFO(root.root_rank, "op=Initialize ok root=%d/%d heap=%llu timeout=%u engine=%s endpoint=%.*s",
                 root.root_rank, root.root_size, static_cast<unsigned long long>(config.heap_size_bytes),
                 config.timeout_seconds, DataEngineName(config.data_engine),
                 static_cast<int>(effective_bootstrap_endpoint.size()), effective_bootstrap_endpoint.data());
  return Status{};
}

Result<int32_t> Runtime::RootRank() const {
  std::lock_guard<std::mutex> lock(mutex_);
  const Status state_status = RequireReady();
  if (state_status.error_code != ErrorCode::Ok) {
    return Result<int32_t>::Failure(state_status);
  }
  return Result<int32_t>::Success(root_->root_rank);
}

Result<int32_t> Runtime::RootSize() const {
  std::lock_guard<std::mutex> lock(mutex_);
  const Status state_status = RequireReady();
  if (state_status.error_code != ErrorCode::Ok) {
    return Result<int32_t>::Failure(state_status);
  }
  return Result<int32_t>::Success(root_->root_size);
}

Result<DeviceModel> Runtime::QueryDeviceModel() const {
  std::lock_guard<std::mutex> lock(mutex_);
  const Status state_status = RequireReady();
  if (state_status.error_code != ErrorCode::Ok) {
    return Result<DeviceModel>::Failure(state_status);
  }
  return Result<DeviceModel>::Success(device_identity_->model);
}

Status Runtime::ValidateShutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  const int32_t root_rank = RootRankForDfx();
  const Status state_status = RequireReady();
  if (state_status.error_code != ErrorCode::Ok) {
    HP_SM_RECORD_FAILURE(DfxOperation::Shutdown, DfxPhase::Validation, root_rank, state_status);
    return state_status;
  }
  const Status device_status = ValidateCurrentDeviceLocked();
  if (device_status.error_code != ErrorCode::Ok) {
    HP_SM_RECORD_FAILURE(DfxOperation::Shutdown, PhaseForDeviceStatus(device_status), root_rank, device_status);
    return device_status;
  }
  if (allocations_->allocated_count() != 0) {
    const Status error =
      MakeError(ErrorCode::InvalidState,
                "All symmetric Allocations must be freed before Runtime shutdown, but got allocated_count=" +
                  std::to_string(allocations_->allocated_count()) +
                  ", allocated_bytes=" + std::to_string(allocations_->allocated_bytes()));
    HP_SM_RECORD_FAILURE(DfxOperation::Shutdown, DfxPhase::Validation, root_rank, error);
    return error;
  }
  return Status{};
}

Result<AllocationRecord> Runtime::Allocate(const AllocationSpec &spec) {
  std::lock_guard<std::mutex> lock(mutex_);
  const int32_t root_rank = RootRankForDfx();

  const Status state_status = RequireReady();
  if (state_status.error_code != ErrorCode::Ok) {
    HP_SM_RECORD_FAILURE(DfxOperation::Allocate, DfxPhase::Validation, root_rank, state_status);
    return Result<AllocationRecord>::Failure(state_status);
  }
  const Status validation = allocations_->Validate(spec);
  if (validation.error_code != ErrorCode::Ok) {
    HP_SM_RECORD_FAILURE(DfxOperation::Allocate, DfxPhase::Validation, root_rank, validation);
    return Result<AllocationRecord>::Failure(validation);
  }
  const Status device_status = ValidateCurrentDeviceLocked();
  if (device_status.error_code != ErrorCode::Ok) {
    HP_SM_RECORD_FAILURE(DfxOperation::Allocate, PhaseForDeviceStatus(device_status), root_rank, device_status);
    return Result<AllocationRecord>::Failure(device_status);
  }

  const bool has_explicit_alignment = spec.alignment_bytes != 0;
  const char *allocation_call = has_explicit_alignment ? "aclshmem_align" : "aclshmem_malloc";
  HP_SM_LOG_DEBUG(root_rank, "op=Allocate call=%s enter bytes=%llu alignment=%llu", allocation_call,
                  static_cast<unsigned long long>(spec.bytes), static_cast<unsigned long long>(spec.alignment_bytes));
  const auto allocation = has_explicit_alignment ? cann::host::aligned_allocate(spec.alignment_bytes, spec.bytes)
                                                 : cann::host::allocate(spec.bytes);
  if (!allocation.ok()) {
    HP_SM_RECORD_FAILURE(DfxOperation::Allocate, DfxPhase::CannCall, root_rank, allocation.error());
    return Result<AllocationRecord>::Failure(allocation.error());
  }

  const AllocationRecord record = allocations_->Register(spec, allocation.value());
  HP_SM_LOG_DEBUG(root_rank, "op=Allocate call=%s ok allocation_id=%llu", allocation_call,
                  static_cast<unsigned long long>(record.allocation_id));
  return Result<AllocationRecord>::Success(record);
}

Status Runtime::Free(const AllocationView &buffer) {
  std::lock_guard<std::mutex> lock(mutex_);
  const int32_t root_rank = RootRankForDfx();

  const Status state_status = RequireReady();
  if (state_status.error_code != ErrorCode::Ok) {
    HP_SM_RECORD_FAILURE(DfxOperation::Free, DfxPhase::Validation, root_rank, state_status);
    return state_status;
  }
  const auto record = allocations_->MatchForFree(buffer);
  if (!record.ok()) {
    HP_SM_RECORD_FAILURE(DfxOperation::Free, DfxPhase::Validation, root_rank, record.error());
    return record.error();
  }

  const Status device_status = ValidateCurrentDeviceLocked();
  if (device_status.error_code != ErrorCode::Ok) {
    HP_SM_RECORD_FAILURE(DfxOperation::Free, PhaseForDeviceStatus(device_status), root_rank, device_status);
    return device_status;
  }

  HP_SM_LOG_DEBUG(root_rank, "op=Free call=aclshmem_free enter allocation_id=%llu",
                  static_cast<unsigned long long>(buffer.allocation_id));
  cann::host::free_memory(record.value().allocation_base);
  allocations_->Remove(buffer.allocation_id);
  HP_SM_LOG_DEBUG(root_rank, "op=Free call=aclshmem_free ok allocation_id=%llu",
                  static_cast<unsigned long long>(buffer.allocation_id));
  return Status{};
}

Result<AllocationRecord> Runtime::ResolveAllocation(const AllocationView &buffer) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const Status state_status = RequireReady();
  if (state_status.error_code != ErrorCode::Ok) {
    return Result<AllocationRecord>::Failure(state_status);
  }
  return allocations_->Resolve(buffer);
}

Status Runtime::Barrier(const StreamView &stream) {
  std::lock_guard<std::mutex> lock(mutex_);
  const int32_t root_rank = RootRankForDfx();

  const Status state_status = RequireReady();
  if (state_status.error_code != ErrorCode::Ok) {
    HP_SM_RECORD_FAILURE(DfxOperation::Barrier, DfxPhase::Validation, root_rank, state_status);
    return state_status;
  }
  if (stream.native_handle == 0 || stream.device_index != device_identity_->device_index) {
    const Status error =
      MakeError(ErrorCode::InvalidArgument,
                "barrier Stream must have a nonzero handle on the Runtime-bound device, but got native_handle=" +
                  std::to_string(stream.native_handle) + ", device_index=" + std::to_string(stream.device_index) +
                  ", bound device_index=" + std::to_string(device_identity_->device_index));
    HP_SM_RECORD_FAILURE(DfxOperation::Barrier, DfxPhase::Validation, root_rank, error);
    return error;
  }

  HP_SM_LOG_DEBUG(root_rank, "op=Barrier call=aclshmemx_barrier_on_stream enter");
  cann::host::barrier_on_stream(stream);
  HP_SM_LOG_DEBUG(root_rank, "op=Barrier call=aclshmemx_barrier_on_stream ok enqueue_only=true");
  return Status{};
}

DfxSnapshot Runtime::DebugState() const {
  std::lock_guard<std::mutex> lock(mutex_);
  DfxSnapshot snapshot{};
  snapshot.state = state_;
  snapshot.root = root_;
  snapshot.device_identity = device_identity_;
  snapshot.config = config_;
  snapshot.latest_failure = latest_failure_;
  if (allocations_.has_value()) {
    snapshot.allocated_count = allocations_->allocated_count();
    snapshot.allocated_bytes = allocations_->allocated_bytes();
    snapshot.max_allocated_bytes = allocations_->max_allocated_bytes();
    snapshot.active_allocations = allocations_->ActiveRecords();
  }
  return snapshot;
}

Status Runtime::Shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (state_ == State::Uninitialized || state_ == State::Finalized) {
    return Status{};
  }
  if (state_ != State::Ready) {
    const Status error = MakeError(ErrorCode::InvalidState, "Runtime state must be Ready for shutdown, but got state=" +
                                                              std::string(StateName(state_)));
    HP_SM_RECORD_FAILURE(DfxOperation::Shutdown, DfxPhase::Validation, RootRankForDfx(), error);
    return error;
  }

  const int32_t root_rank = RootRankForDfx();
  const uint64_t allocated_count = allocations_->allocated_count();
  const uint64_t allocated_bytes = allocations_->allocated_bytes();
  if (allocated_count != 0) {
    const Status error =
      MakeError(ErrorCode::InvalidState,
                "All symmetric Allocations must be freed before Runtime shutdown, but got allocated_count=" +
                  std::to_string(allocated_count) + ", allocated_bytes=" + std::to_string(allocated_bytes));
    HP_SM_RECORD_FAILURE(DfxOperation::Shutdown, DfxPhase::Validation, root_rank, error);
    return error;
  }

  HP_SM_LOG_DEBUG(root_rank, "op=Shutdown call=aclshmem_finalize enter");
  const Status finalize_status = cann::host::finalize();
  if (finalize_status.error_code != ErrorCode::Ok) {
    state_ = State::Finalized;
    HP_SM_RECORD_FAILURE(DfxOperation::Shutdown, DfxPhase::CannCall, root_rank, finalize_status);
    return finalize_status;
  }
  HP_SM_LOG_DEBUG(root_rank, "op=Shutdown call=aclshmem_finalize ok");
  next_allocation_id_ = allocations_->next_allocation_id();
  HP_SM_LOG_INFO(root_rank, "op=Shutdown ok mode=finalize active_allocations_at_shutdown=0 active_bytes_at_shutdown=0");
  allocations_.reset();
  device_identity_.reset();
  root_.reset();
  config_.reset();
  latest_failure_.reset();
  state_ = State::Uninitialized;
  return Status{};
}

Status Runtime::ValidateInitialInput(const RootWorldInfo &root, const Config &config,
                                     std::string_view effective_bootstrap_endpoint) const {
  if (root.root_size <= 0 || root.root_rank < 0 || root.root_rank >= root.root_size) {
    return MakeError(ErrorCode::InvalidArgument,
                     "root_rank must be in [0, root_size) and root_size must be positive, but got root_rank=" +
                       std::to_string(root.root_rank) + ", root_size=" + std::to_string(root.root_size));
  }
  if (config.heap_size_bytes == 0 || config.timeout_seconds == 0 || config.data_engine != DataEngine::Mte) {
    return MakeError(ErrorCode::InvalidConfig,
                     "Runtime Config requires positive heap/timeout values and the MTE data engine");
  }
  if (config.bootstrap_endpoint_base.empty() || config.bootstrap_endpoint_base.size() > kMaxBootstrapEndpointBytes) {
    return MakeError(ErrorCode::InvalidConfig, "bootstrap_endpoint_base must contain between 1 and " +
                                                 std::to_string(kMaxBootstrapEndpointBytes) + " bytes, but got size=" +
                                                 std::to_string(config.bootstrap_endpoint_base.size()));
  }
  if (effective_bootstrap_endpoint.empty() || effective_bootstrap_endpoint.size() > kMaxBootstrapEndpointBytes) {
    return MakeError(ErrorCode::InvalidArgument,
                     "effective_bootstrap_endpoint must contain between 1 and " +
                       std::to_string(kMaxBootstrapEndpointBytes) +
                       " bytes, but got size=" + std::to_string(effective_bootstrap_endpoint.size()));
  }
  return Status{};
}

Status Runtime::ValidateCurrentDeviceLocked() const {
  const auto current_device = cann::host::query_current_device_index();
  if (!current_device.ok()) {
    return current_device.error();
  }
  if (current_device.value() != device_identity_->device_index) {
    return MakeError(ErrorCode::InvalidArgument,
                     "Current ACL device must match the Runtime-bound device, but got current device_index=" +
                       std::to_string(current_device.value()) +
                       ", bound device_index=" + std::to_string(device_identity_->device_index));
  }
  return Status{};
}

Status Runtime::RequireReady() const {
  if (state_ != State::Ready) {
    return MakeError(ErrorCode::InvalidState,
                     "Runtime state must be Ready for this operation, but got state=" + std::string(StateName(state_)));
  }
  return Status{};
}

int32_t Runtime::RootRankForDfx() const noexcept { return root_.has_value() ? root_->root_rank : kUnavailableRootRank; }

void Runtime::RecordFailure(DfxOperation operation, DfxPhase phase, int32_t root_rank, const Status &error,
                            const char *file, int line) {
  latest_failure_.emplace(DfxFailure{operation, phase, root_rank, error});
  log::Failure(file, line, operation, phase, root_rank, error);
}

Status Runtime::FailInitialization(DfxPhase phase, int32_t root_rank, const Status &error, const char *file, int line) {
  // Roll back the partially initialized lifecycle so the caller can retry: CANN rolls its own initialization
  // failure back, so no CANN finalize is needed or allowed here.
  allocations_.reset();
  config_.reset();
  root_.reset();
  device_identity_.reset();
  RecordFailure(DfxOperation::Initialize, phase, root_rank, error, file, line);
  return error;
}

}  // namespace hyper_parallel::multicore::shmem::runtime

#undef HP_SM_FAIL_INITIALIZATION
#undef HP_SM_RECORD_FAILURE
