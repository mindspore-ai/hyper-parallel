/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/Optional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cann/host.h"
#include "ops/include/shmem_kernel.h"
#include "runtime/log.h"
#include "runtime/runtime.h"
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace py = pybind11;

namespace hyper_parallel::multicore::shmem::bindings::torch {
namespace {

using cann::CompareOp;
using cann::SignalOp;
using ::hyper_parallel::multicore::shmem::DeviceModel;
using runtime::AllocationRecord;
using runtime::AllocationSpec;
using runtime::AllocationView;
using runtime::DataEngine;
using runtime::DfxOperation;
using runtime::DfxPhase;
using runtime::DfxSnapshot;
using runtime::ErrorCode;
using runtime::Result;
using runtime::RootWorldInfo;
using runtime::Runtime;
using runtime::State;
using runtime::Status;
using runtime::StreamView;

constexpr std::string_view kErrorPrefix = "[HYPER_PARALLEL_SHMEM_ERROR]";
constexpr std::string_view kAcquireHint = "Call shmem.acquire() before using SHMEM capabilities.";

std::string_view ToString(ErrorCode error_code) {
  switch (error_code) {
    case ErrorCode::Ok:
      return "OK";
    case ErrorCode::InvalidState:
      return "INVALID_STATE";
    case ErrorCode::InvalidConfig:
      return "INVALID_CONFIG";
    case ErrorCode::InvalidArgument:
      return "INVALID_ARGUMENT";
    case ErrorCode::UnsupportedCapability:
      return "UNSUPPORTED_CAPABILITY";
    case ErrorCode::DoubleFree:
      return "DOUBLE_FREE";
    case ErrorCode::CannError:
      return "CANN_ERROR";
    case ErrorCode::Timeout:
      return "TIMEOUT";
  }
  return "UNKNOWN_ERROR";
}

std::string_view ToString(State state) {
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

std::string_view ToString(DeviceModel model) {
  switch (model) {
    case DeviceModel::kAscend910B:
      return "ascend_910b";
    case DeviceModel::kAscend910C:
      return "ascend_910c";
    case DeviceModel::kAscend950:
      return "ascend_950";
  }
  return "unknown";
}

std::string_view ToString(DataEngine engine) {
  switch (engine) {
    case DataEngine::Mte:
      return "mte";
  }
  return "Unknown";
}

std::string_view ToString(DfxOperation operation) {
  switch (operation) {
    case DfxOperation::Initialize:
      return "initialize";
    case DfxOperation::Allocate:
      return "allocate";
    case DfxOperation::Free:
      return "free";
    case DfxOperation::Barrier:
      return "barrier";
    case DfxOperation::Shutdown:
      return "shutdown";
  }
  return "unknown";
}

std::string_view ToString(DfxPhase phase) {
  switch (phase) {
    case DfxPhase::Validation:
      return "Validation";
    case DfxPhase::CannCall:
      return "CannCall";
  }
  return "Unknown";
}

std::string_view PhaseForStatus(const Status &status, DfxPhase fallback_phase) {
  if (status.error_code == ErrorCode::CannError || status.error_code == ErrorCode::Timeout) {
    return "CannCall";
  }
  return ToString(fallback_phase);
}

[[noreturn]] void ThrowStatus(const Status &status, std::string_view operation, DfxPhase fallback_phase) {
  std::ostringstream message;
  message << kErrorPrefix << " error_code=" << ToString(status.error_code) << " operation=" << operation
          << " phase=" << PhaseForStatus(status, fallback_phase) << " cann_error_code=";
  if (status.cann_error_code.has_value()) {
    message << *status.cann_error_code;
  } else {
    message << "None";
  }
  message << " message=" << status.message;
  throw std::runtime_error(message.str());
}

[[noreturn]] void ThrowStatus(const Status &status, DfxOperation operation, DfxPhase fallback_phase) {
  ThrowStatus(status, ToString(operation), fallback_phase);
}

void RequireOk(Status status, DfxOperation operation, DfxPhase fallback_phase = DfxPhase::Validation) {
  if (status.error_code != ErrorCode::Ok) {
    ThrowStatus(status, operation, fallback_phase);
  }
}

template <typename Operation>
[[noreturn]] void ThrowRuntimeStatus(const Status &status, Operation operation, DfxPhase fallback_phase) {
  if (status.error_code == ErrorCode::InvalidState) {
    Status actionable_status = status;
    actionable_status.message += "; ";
    actionable_status.message += kAcquireHint;
    ThrowStatus(actionable_status, operation, fallback_phase);
  }
  ThrowStatus(status, operation, fallback_phase);
}

template <typename T, typename Operation>
T UnwrapRuntime(Result<T> result, Operation operation, DfxPhase fallback_phase = DfxPhase::Validation) {
  if (!result.ok()) {
    ThrowRuntimeStatus(result.error(), operation, fallback_phase);
  }
  return result.value();
}

template <typename Operation>
void RequireRuntimeOk(Status status, Operation operation, DfxPhase fallback_phase = DfxPhase::Validation) {
  if (status.error_code != ErrorCode::Ok) {
    ThrowRuntimeStatus(status, operation, fallback_phase);
  }
}

template <typename T>
T UnwrapOperation(Result<T> result, std::string_view operation, DfxPhase fallback_phase = DfxPhase::Validation) {
  if (!result.ok()) {
    ThrowStatus(result.error(), operation, fallback_phase);
  }
  return result.value();
}

uintptr_t StreamHandle(const c10_npu::NPUStream &stream) {
  // stream() drains torch_npu's task queue before exposing the ACL stream. Direct CANN launches must not use
  // stream(false), which can overtake Torch operations already queued on the same stream.
  // ABI dependency: callers reinterpret_cast this handle back to aclrtStream, which relies on torch_npu's
  // stream representation being exactly an aclrtStream. If torch_npu changes that representation, every
  // direct-ACL entry (Barrier, AllGather) must be revisited.
  return reinterpret_cast<uintptr_t>(stream.stream());
}

uintptr_t CurrentStreamHandle(int32_t device_index) { return StreamHandle(c10_npu::getCurrentNPUStream(device_index)); }

std::string RequireString(const py::handle &value, std::string_view argument_name) {
  if (!PyUnicode_Check(value.ptr())) {
    throw py::type_error(std::string(argument_name) + " must be a string");
  }
  return py::cast<std::string>(value);
}

SignalOp ParseSignalOp(const py::handle &value) {
  const std::string operation = RequireString(value, "operation");
  if (operation == "set") {
    return SignalOp::Set;
  }
  if (operation == "add") {
    return SignalOp::Add;
  }
  throw py::value_error("operation must be one of: set, add; got " + operation);
}

CompareOp ParseCompareOp(const py::handle &value) {
  const std::string comparison = RequireString(value, "comparison");
  if (comparison == "eq") {
    return CompareOp::Equal;
  }
  if (comparison == "ne") {
    return CompareOp::NotEqual;
  }
  if (comparison == "gt") {
    return CompareOp::Greater;
  }
  if (comparison == "ge") {
    return CompareOp::GreaterEqual;
  }
  if (comparison == "lt") {
    return CompareOp::Less;
  }
  if (comparison == "le") {
    return CompareOp::LessEqual;
  }
  throw py::value_error("comparison must be one of: eq, ne, gt, ge, lt, le; got " + comparison);
}

class StorageAssociations final {
 public:
  enum class State : uint8_t {
    Active,
    Released,
  };

  struct Record {
    uint64_t allocation_id;
    uintptr_t allocation_base;
    uint64_t allocation_bytes;
    int32_t root_rank;
    State state;
  };

  static StorageAssociations &Instance() {
    static StorageAssociations associations;
    return associations;
  }

  bool Insert(c10::StorageImpl *storage, Record record) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto [position, inserted] = records_.emplace(storage, record);
    return inserted || position->second.allocation_id == record.allocation_id;
  }

  std::optional<uint64_t> Find(c10::StorageImpl *storage) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto position = records_.find(storage);
    if (position == records_.end()) {
      return std::nullopt;
    }
    return position->second.allocation_id;
  }

  void MarkReleased(c10::StorageImpl *storage, uint64_t allocation_id) noexcept {
    if (storage == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto position = records_.find(storage);
    if (position != records_.end() && position->second.allocation_id == allocation_id) {
      position->second.state = State::Released;
    }
  }

  std::optional<Record> Remove(c10::StorageImpl *storage, uint64_t allocation_id) noexcept {
    if (storage == nullptr) {
      return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto position = records_.find(storage);
    if (position == records_.end() || position->second.allocation_id != allocation_id) {
      return std::nullopt;
    }
    const Record record = position->second;
    records_.erase(position);
    return record;
  }

 private:
  StorageAssociations() = default;

  mutable std::mutex mutex_;
  std::unordered_map<c10::StorageImpl *, Record> records_;
};

struct StorageAssociationToken {
  explicit StorageAssociationToken(uint64_t id) : allocation_id(id) {}

  std::atomic<c10::StorageImpl *> storage{nullptr};
  uint64_t allocation_id;
};

int32_t CurrentDeviceIndex() { return static_cast<int32_t>(c10_npu::current_device()); }

uint64_t CalculateAllocationBytes(const std::vector<int64_t> &shape, c10::ScalarType dtype) {
  uint64_t element_count = 1;
  for (std::size_t dimension_index = 0; dimension_index < shape.size(); ++dimension_index) {
    const int64_t dimension = shape[dimension_index];
    if (dimension < 0) {
      throw std::runtime_error("shape dimension " + std::to_string(dimension_index) +
                               " must be non-negative, but got " + std::to_string(dimension));
    }
    const uint64_t extent = static_cast<uint64_t>(dimension);
    if (extent != 0 && element_count > std::numeric_limits<uint64_t>::max() / extent) {
      throw std::runtime_error("shape element count overflows uint64_t");
    }
    element_count *= extent;
  }

  const uint64_t element_bytes = static_cast<uint64_t>(at::elementSize(dtype));
  if (element_count != 0 && element_count > std::numeric_limits<uint64_t>::max() / element_bytes) {
    throw std::runtime_error("Tensor byte count overflows uint64_t");
  }
  return element_count * element_bytes;
}

uint64_t NormalizeAlignment(const std::optional<int64_t> &alignment) {
  if (!alignment.has_value()) {
    return 0;
  }
  if (*alignment <= 0) {
    throw std::runtime_error("alignment must be positive, but got " + std::to_string(*alignment));
  }
  return static_cast<uint64_t>(*alignment);
}

AllocationView TensorView(const at::Tensor &tensor) {
  if (!tensor.defined()) {
    throw std::runtime_error("Tensor must be defined");
  }
  if (tensor.device().type() != c10::DeviceType::PrivateUse1) {
    throw std::runtime_error("Tensor must reside on an NPU, but got device=" + tensor.device().str());
  }
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("Tensor or view must represent one contiguous byte range");
  }
  if (tensor.data_ptr() == nullptr) {
    throw std::runtime_error("Tensor data pointer must not be null");
  }

  c10::StorageImpl *storage = tensor.storage().unsafeGetStorageImpl();
  const auto allocation_id = StorageAssociations::Instance().Find(storage);
  if (!allocation_id.has_value()) {
    throw std::runtime_error("Tensor Storage is not associated with a Runtime symmetric Allocation");
  }

  const int64_t element_count = tensor.numel();
  const uint64_t element_bytes = static_cast<uint64_t>(tensor.element_size());
  if (element_count < 0 ||
      static_cast<uint64_t>(element_count) > std::numeric_limits<uint64_t>::max() / element_bytes) {
    throw std::runtime_error("Tensor view byte count overflows uint64_t");
  }
  return AllocationView{*allocation_id, reinterpret_cast<uintptr_t>(tensor.data_ptr()),
                        static_cast<uint64_t>(element_count) * element_bytes,
                        static_cast<int32_t>(tensor.device().index())};
}

void Initialize(int32_t root_rank, int32_t root_size) {
  const auto config = runtime::LoadConfigFromEnvironment();
  if (!config.ok()) {
    ThrowStatus(config.error(), DfxOperation::Initialize, DfxPhase::Validation);
  }
  RequireOk(Runtime::Instance().Initialize(RootWorldInfo{root_rank, root_size}, config.value(),
                                           config.value().bootstrap_endpoint_base),
            DfxOperation::Initialize);
}

at::Tensor Empty(const std::vector<int64_t> &shape, c10::ScalarType dtype, const std::optional<int64_t> &alignment) {
  const c10::Device device(c10::DeviceType::PrivateUse1, CurrentDeviceIndex());
  const uint64_t allocation_bytes = CalculateAllocationBytes(shape, dtype);
  const uint64_t alignment_bytes = NormalizeAlignment(alignment);
  const auto record = UnwrapRuntime(Runtime::Instance().Allocate(AllocationSpec{allocation_bytes, alignment_bytes}),
                                    DfxOperation::Allocate);
  const auto root_rank_result = Runtime::Instance().RootRank();
  const int32_t root_rank = root_rank_result.ok() ? root_rank_result.value() : -1;

  auto token = std::make_shared<StorageAssociationToken>(record.allocation_id);
  auto deleter = [token](void *) {
    const auto association =
      StorageAssociations::Instance().Remove(token->storage.load(std::memory_order_acquire), token->allocation_id);
    if (association.has_value() && association->state == StorageAssociations::State::Active) {
      runtime::log::Line(
        runtime::log::Level::Error, association->root_rank, __FILE__, __LINE__,
        "op=StorageRelease orphaned=true allocation_id=%llu allocation_base=0x%llx allocation_bytes=%llu "
        "msg=\"The last Tensor Storage reference was dropped without shmem.free(). The Allocation remains active "
        "and cannot be freed through the Tensor API. Call shmem.free(tensor) before overwriting or dropping its "
        "last reference. The final shmem.release() will fail; restart the process to recover.\"",
        static_cast<unsigned long long>(association->allocation_id),
        static_cast<unsigned long long>(association->allocation_base),
        static_cast<unsigned long long>(association->allocation_bytes));
    }
  };
  at::Tensor tensor =
    at_npu::native::from_blob(reinterpret_cast<void *>(record.allocation_base), shape, std::move(deleter),
                              at::TensorOptions().dtype(dtype).device(device), c10::optional<c10::Device>(device));
  c10::StorageImpl *storage = tensor.storage().unsafeGetStorageImpl();
  token->storage.store(storage, std::memory_order_release);
  if (!StorageAssociations::Instance().Insert(
        storage, StorageAssociations::Record{record.allocation_id, record.allocation_base, record.allocation_bytes,
                                             root_rank, StorageAssociations::State::Active})) {
    ThrowStatus(Status{ErrorCode::InvalidState, std::nullopt,
                       "Tensor Storage is already associated with another symmetric Allocation"},
                DfxOperation::Allocate, DfxPhase::Validation);
  }
  return tensor;
}

/**
 * @brief Collectively release the complete symmetric Allocation backing tensor.
 * @post Success invalidates tensor and every alias of its Storage.
 * @note Tensor metadata and the Storage association remain intact so repeated SHMEM free is diagnosed.
 * @note Arbitrary Torch use after free remains caller error and is not guarded here.
 */
void Free(const at::Tensor &tensor) {
  const AllocationView view = TensorView(tensor);
  c10::StorageImpl *storage = tensor.storage().unsafeGetStorageImpl();
  RequireRuntimeOk(Runtime::Instance().Free(view), DfxOperation::Free);
  StorageAssociations::Instance().MarkReleased(storage, view.allocation_id);
}

/**
 * @brief Enqueue a Root WORLD barrier on the current NPU Stream and optionally wait for that exact Stream.
 *
 *
 * With blocking=false, return only confirms direct CANN enqueue on the ACL Stream; the barrier does not enter
 *
 * torch_npu's task queue and device completion is not Host-visible.
 */
void Barrier(bool blocking) {
  const int32_t device_index = CurrentDeviceIndex();
  const auto stream = c10_npu::getCurrentNPUStream(device_index);
  const uintptr_t stream_handle = StreamHandle(stream);
  RequireRuntimeOk(Runtime::Instance().Barrier(StreamView{stream_handle, device_index}), DfxOperation::Barrier);
  if (blocking) {
    py::gil_scoped_release release;
    stream.synchronize();
  }
}

void ValidateContiguousNpuTensor(const at::Tensor &tensor, std::string_view argument_name) {
  const std::string name(argument_name);
  if (!tensor.defined()) {
    throw std::runtime_error(name + " must be a defined Tensor");
  }
  if (tensor.device().type() != c10::DeviceType::PrivateUse1) {
    throw std::runtime_error(name + " must reside on an NPU, but got device=" + tensor.device().str());
  }
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(name + " must be contiguous");
  }
}

uint64_t TensorBytes(const at::Tensor &tensor, std::string_view argument_name) {
  const int64_t element_count = tensor.numel();
  const uint64_t element_bytes = static_cast<uint64_t>(tensor.element_size());
  if (element_count < 0 ||
      static_cast<uint64_t>(element_count) > std::numeric_limits<uint64_t>::max() / element_bytes) {
    throw std::runtime_error(std::string(argument_name) + " byte count overflows uint64_t");
  }
  return static_cast<uint64_t>(element_count) * element_bytes;
}

uintptr_t CheckedRangeEnd(uintptr_t begin, uint64_t bytes, std::string_view argument_name) {
  const uintptr_t max_address = std::numeric_limits<uintptr_t>::max();
  if (bytes > static_cast<uint64_t>(max_address - begin)) {
    throw std::runtime_error(std::string(argument_name) + " address range overflows uintptr_t");
  }
  return begin + static_cast<uintptr_t>(bytes);
}

bool AddressRangesOverlap(uintptr_t lhs_begin, uint64_t lhs_bytes, uintptr_t rhs_begin, uint64_t rhs_bytes) {
  const uintptr_t lhs_end = CheckedRangeEnd(lhs_begin, lhs_bytes, "output");
  const uintptr_t rhs_end = CheckedRangeEnd(rhs_begin, rhs_bytes, "input");
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

int32_t ValidateRootPe(int64_t pe, std::string_view argument_name, std::string_view operation) {
  const int32_t root_size = UnwrapRuntime(Runtime::Instance().RootSize(), operation);
  if (pe < 0 || pe >= root_size) {
    throw py::value_error(std::string(argument_name) + " must be in [0, " + std::to_string(root_size) + "), got " +
                          std::to_string(pe));
  }
  return static_cast<int32_t>(pe);
}

int32_t ValidateSignalValue(int64_t value) {
  if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max()) {
    throw py::value_error("signal value must fit int32, got " + std::to_string(value));
  }
  return static_cast<int32_t>(value);
}

StreamView PrepareOneSidedStream(const c10::Device &device, std::string_view operation) {
  const int32_t device_index = static_cast<int32_t>(device.index());
  const int32_t current_device = UnwrapOperation(cann::host::query_current_device_index(), operation);
  if (current_device != device_index) {
    ThrowStatus(Status{ErrorCode::InvalidArgument, std::nullopt,
                       "current ACL device does not match Tensor device: current=" + std::to_string(current_device) +
                         ", tensor=" + std::to_string(device_index)},
                operation, DfxPhase::Validation);
  }
  return StreamView{CurrentStreamHandle(device_index), device_index};
}

void ResolveSymmetricTensor(const at::Tensor &tensor, std::string_view operation) {
  static_cast<void>(UnwrapRuntime(Runtime::Instance().ResolveAllocation(TensorView(tensor)), operation));
}

void Put(const at::Tensor &remote_dst, const at::Tensor &local_src, int64_t target_pe) {
  constexpr std::string_view kOperation = "put";
  ValidateContiguousNpuTensor(remote_dst, "remote_dst");
  ValidateContiguousNpuTensor(local_src, "local_src");
  if (remote_dst.device() != local_src.device()) {
    throw std::runtime_error("remote_dst and local_src must reside on the same NPU");
  }
  const uint64_t bytes = TensorBytes(remote_dst, "remote_dst");
  if (bytes != TensorBytes(local_src, "local_src")) {
    throw std::runtime_error("remote_dst and local_src must have the same byte count");
  }
  const int32_t root_pe = ValidateRootPe(target_pe, "target_pe", kOperation);
  ResolveSymmetricTensor(remote_dst, kOperation);
  if (bytes == 0) {
    return;
  }
  const StreamView stream = PrepareOneSidedStream(remote_dst.device(), kOperation);
  HP_SM_LOG_DEBUG(UnwrapRuntime(Runtime::Instance().RootRank(), kOperation),
                  "op=put enqueue target_pe=%d bytes=%llu remote_addr=%p local_addr=%p", root_pe,
                  static_cast<unsigned long long>(bytes), remote_dst.data_ptr(), local_src.data_ptr());
  cann::host::put_on_stream(reinterpret_cast<uintptr_t>(remote_dst.data_ptr()),
                            reinterpret_cast<uintptr_t>(local_src.data_ptr()), bytes, root_pe, stream);
}

void Get(const at::Tensor &local_dst, const at::Tensor &remote_src, int64_t source_pe) {
  constexpr std::string_view kOperation = "get";
  ValidateContiguousNpuTensor(local_dst, "local_dst");
  ValidateContiguousNpuTensor(remote_src, "remote_src");
  if (local_dst.device() != remote_src.device()) {
    throw std::runtime_error("local_dst and remote_src must reside on the same NPU");
  }
  const uint64_t bytes = TensorBytes(local_dst, "local_dst");
  if (bytes != TensorBytes(remote_src, "remote_src")) {
    throw std::runtime_error("local_dst and remote_src must have the same byte count");
  }
  const int32_t root_pe = ValidateRootPe(source_pe, "source_pe", kOperation);
  ResolveSymmetricTensor(remote_src, kOperation);
  if (bytes == 0) {
    return;
  }
  const StreamView stream = PrepareOneSidedStream(local_dst.device(), kOperation);
  HP_SM_LOG_DEBUG(UnwrapRuntime(Runtime::Instance().RootRank(), kOperation),
                  "op=get enqueue source_pe=%d bytes=%llu local_addr=%p remote_addr=%p", root_pe,
                  static_cast<unsigned long long>(bytes), local_dst.data_ptr(), remote_src.data_ptr());
  cann::host::get_on_stream(reinterpret_cast<uintptr_t>(local_dst.data_ptr()),
                            reinterpret_cast<uintptr_t>(remote_src.data_ptr()), bytes, root_pe, stream);
}

void ValidateSignalTensor(const at::Tensor &signal, std::string_view operation) {
  ValidateContiguousNpuTensor(signal, "signal");
  if (signal.scalar_type() != c10::ScalarType::Int || signal.numel() != 1) {
    throw std::runtime_error("signal must contain exactly one int32 element");
  }
  ResolveSymmetricTensor(signal, operation);
}

void Signal(const at::Tensor &remote_signal, int64_t value, int64_t target_pe, const py::object &operation) {
  constexpr std::string_view kOperation = "signal";
  const SignalOp signal_op = ParseSignalOp(operation);
  const int32_t signal_value = ValidateSignalValue(value);
  ValidateSignalTensor(remote_signal, kOperation);
  const int32_t root_pe = ValidateRootPe(target_pe, "target_pe", kOperation);
  const StreamView stream = PrepareOneSidedStream(remote_signal.device(), kOperation);
  HP_SM_LOG_DEBUG(UnwrapRuntime(Runtime::Instance().RootRank(), kOperation),
                  "op=signal enqueue target_pe=%d value=%d signal_op=%d remote_addr=%p", root_pe, signal_value,
                  static_cast<int>(signal_op), remote_signal.data_ptr());
  cann::host::signal_on_stream(reinterpret_cast<uintptr_t>(remote_signal.data_ptr()), signal_value, signal_op, root_pe,
                               stream);
}

void WaitSignal(const at::Tensor &signal, int64_t value, const py::object &comparison) {
  constexpr std::string_view kOperation = "wait_signal";
  const CompareOp compare_op = ParseCompareOp(comparison);
  const int32_t signal_value = ValidateSignalValue(value);
  ValidateSignalTensor(signal, kOperation);
  const StreamView stream = PrepareOneSidedStream(signal.device(), kOperation);
  HP_SM_LOG_DEBUG(UnwrapRuntime(Runtime::Instance().RootRank(), kOperation),
                  "op=wait_signal enqueue value=%d compare_op=%d local_addr=%p", signal_value,
                  static_cast<int>(compare_op), signal.data_ptr());
  cann::host::wait_signal_on_stream(reinterpret_cast<uintptr_t>(signal.data_ptr()), compare_op, signal_value, stream);
}

void AllGather(const at::Tensor &output, const at::Tensor &input) {
  constexpr std::string_view kOperation = "all_gather";

  const int32_t root_rank = UnwrapRuntime(Runtime::Instance().RootRank(), kOperation);
  const int32_t root_size = UnwrapRuntime(Runtime::Instance().RootSize(), kOperation);
  const DeviceModel device_model = UnwrapRuntime(Runtime::Instance().QueryDeviceModel(), kOperation);
  if (device_model != DeviceModel::kAscend910B) {
    ThrowStatus(
      Status{ErrorCode::UnsupportedCapability, std::nullopt, "all_gather currently supports only Ascend 910B"},
      kOperation, DfxPhase::Validation);
  }

  ValidateContiguousNpuTensor(output, "output");
  ValidateContiguousNpuTensor(input, "input");
  if (output.device() != input.device()) {
    throw std::runtime_error("output and input must reside on the same NPU, but got output=" + output.device().str() +
                             ", input=" + input.device().str());
  }
  if (output.scalar_type() != input.scalar_type()) {
    throw std::runtime_error("output and input must have the same dtype");
  }

  const int64_t input_elements = input.numel();
  if (root_size <= 0 || input_elements > std::numeric_limits<int64_t>::max() / root_size) {
    throw std::runtime_error("expected all_gather output element count overflows int64_t");
  }
  const int64_t expected_output_elements = input_elements * root_size;
  if (output.numel() != expected_output_elements) {
    throw std::runtime_error(
      "output.numel() must equal input.numel() * root_size, but got output=" + std::to_string(output.numel()) +
      ", input=" + std::to_string(input_elements) + ", root_size=" + std::to_string(root_size));
  }

  const uint64_t input_bytes = TensorBytes(input, "input");
  const uint64_t output_bytes = TensorBytes(output, "output");
  if (input_bytes == 0) {
    // A zero-byte gather has no data movement or synchronization work to enqueue.
    return;
  }
  if (output.data_ptr() == nullptr || input.data_ptr() == nullptr) {
    throw std::runtime_error("nonempty all_gather Tensors must have valid storage");
  }
  const AllocationView output_view = TensorView(output);
  static_cast<void>(UnwrapRuntime(Runtime::Instance().ResolveAllocation(output_view), kOperation));
  if (AddressRangesOverlap(reinterpret_cast<uintptr_t>(output.data_ptr()), output_bytes,
                           reinterpret_cast<uintptr_t>(input.data_ptr()), input_bytes)) {
    throw std::runtime_error("output and input byte ranges must not overlap");
  }
  const int32_t device_index = static_cast<int32_t>(output.device().index());
  c10::DeviceGuard guard(output.device());
  const uintptr_t stream_handle = CurrentStreamHandle(device_index);
  const StreamView stream_view{stream_handle, device_index};
  RequireRuntimeOk(Runtime::Instance().Barrier(stream_view), kOperation);
  const aclError launch_error =
    ShmemKernel::aclshmem_all_gather(reinterpret_cast<aclrtStream>(stream_handle), output.data_ptr(), input.data_ptr(),
                                     input_bytes, root_rank, root_size);
  if (launch_error != ACL_SUCCESS) {
    ThrowStatus(
      Status{ErrorCode::CannError, static_cast<int32_t>(launch_error), "all_gather kernel launch failed"},
      kOperation, DfxPhase::CannCall);
  }
  RequireRuntimeOk(Runtime::Instance().Barrier(stream_view), kOperation);
}

void ValidateShutdown() {
  RequireOk(Runtime::Instance().ValidateShutdown(), DfxOperation::Shutdown, DfxPhase::Validation);
}

py::dict FailureDict(const runtime::DfxFailure &failure) {
  py::dict result;
  result["operation"] = ToString(failure.operation);
  result["phase"] = ToString(failure.phase);
  result["root_rank"] = failure.root_rank;
  result["error_code"] = ToString(failure.status.error_code);
  result["cann_error_code"] =
    failure.status.cann_error_code.has_value() ? py::cast(*failure.status.cann_error_code) : py::none();
  result["message"] = failure.status.message;
  return result;
}

py::dict DebugState() {
  const DfxSnapshot snapshot = Runtime::Instance().DebugState();
  py::dict result;
  result["state"] = ToString(snapshot.state);
  result["root_rank"] = snapshot.root.has_value() ? py::cast(snapshot.root->root_rank) : py::none();
  result["root_size"] = snapshot.root.has_value() ? py::cast(snapshot.root->root_size) : py::none();
  result["device_index"] =
    snapshot.device_identity.has_value() ? py::cast(snapshot.device_identity->device_index) : py::none();
  result["device_model"] =
    snapshot.device_identity.has_value() ? py::cast(ToString(snapshot.device_identity->model)) : py::none();
  result["soc_name"] = snapshot.device_identity.has_value() ? py::cast(snapshot.device_identity->soc_name) : py::none();
  if (snapshot.config.has_value()) {
    py::dict config;
    config["heap_size_bytes"] = snapshot.config->heap_size_bytes;
    config["timeout_seconds"] = snapshot.config->timeout_seconds;
    config["data_engine"] = ToString(snapshot.config->data_engine);
    config["bootstrap_endpoint_base"] = snapshot.config->bootstrap_endpoint_base;
    result["config"] = std::move(config);
  } else {
    result["config"] = py::none();
  }
  result["allocated_count"] =
    snapshot.allocated_count.has_value() ? py::cast(*snapshot.allocated_count) : py::none();
  result["allocated_bytes"] =
    snapshot.allocated_bytes.has_value() ? py::cast(*snapshot.allocated_bytes) : py::none();
  result["remaining_bytes"] = snapshot.config.has_value() && snapshot.allocated_bytes.has_value()
                                ? py::cast(snapshot.config->heap_size_bytes - *snapshot.allocated_bytes)
                                : py::none();
  result["max_allocated_bytes"] = snapshot.max_allocated_bytes.has_value()
                                    ? py::cast(*snapshot.max_allocated_bytes)
                                    : py::none();
  if (snapshot.active_allocations.has_value()) {
    py::list allocations;
    for (const AllocationRecord &record : *snapshot.active_allocations) {
      py::dict entry;
      entry["allocation_id"] = py::cast(record.allocation_id);
      entry["allocation_base"] = py::cast(record.allocation_base);
      entry["allocation_bytes"] = py::cast(record.allocation_bytes);
      allocations.append(std::move(entry));
    }
    result["active_allocations"] = std::move(allocations);
  } else {
    result["active_allocations"] = py::none();
  }
  result["latest_failure"] =
    snapshot.latest_failure.has_value() ? py::object(FailureDict(*snapshot.latest_failure)) : py::none();
  return result;
}

}  // namespace
}  // namespace hyper_parallel::multicore::shmem::bindings::torch

PYBIND11_MODULE(hyper_parallel_shmem_torch, module) {
  using namespace hyper_parallel::multicore::shmem::bindings::torch;

  module.doc() = "Private Torch binding for the Hyper-Parallel SHMEM Runtime";
  module.def("_initialize", &Initialize, py::arg("root_rank"), py::arg("root_size"));
  module.def("_empty", &Empty, py::arg("shape"), py::arg("dtype"), py::arg("alignment") = std::nullopt);
  module.def("_free", &Free, py::arg("tensor"));
  module.def("_barrier", &Barrier, py::kw_only(), py::arg("blocking") = true);
  module.def("_put", &Put, py::arg("remote_dst"), py::arg("local_src"), py::arg("target_pe"));
  module.def("_get", &Get, py::arg("local_dst"), py::arg("remote_src"), py::arg("source_pe"));
  module.def("_signal", &Signal, py::arg("remote_signal"), py::arg("value"), py::arg("target_pe"), py::kw_only(),
             py::arg("operation") = "set");
  module.def("_wait_signal", &WaitSignal, py::arg("signal"), py::arg("value"), py::kw_only(),
             py::arg("comparison") = "eq");
  module.def("_all_gather", &AllGather, py::arg("output"), py::arg("input"));
  module.def("_debug_state", &DebugState);
  module.def("_validate_shutdown", &ValidateShutdown);
  module.def("_shutdown", [] { RequireOk(Runtime::Instance().Shutdown(), DfxOperation::Shutdown); });
}
