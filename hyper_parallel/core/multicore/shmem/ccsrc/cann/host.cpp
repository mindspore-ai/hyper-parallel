/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann/host.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>

#include "shmem.h"

namespace hyper_parallel::multicore::shmem::cann::host {
namespace {

constexpr uint32_t kOptionalAttrVersionMajor = 1U;
constexpr uint32_t kOptionalAttrVersionMajorShift = 16U;
constexpr int32_t kInitOptionalAttrVersion = static_cast<int32_t>(
  (kOptionalAttrVersionMajor << kOptionalAttrVersionMajorShift) + sizeof(aclshmem_init_optional_attr_t));
constexpr int32_t kInvalidDeviceIndex = -1;
constexpr int32_t kNoSocketFd = -1;
constexpr uint64_t kDefaultInstanceId = 0U;
constexpr std::size_t kCannEndpointCapacity = ACLSHMEM_MAX_IP_PORT_LEN;
constexpr std::size_t kCannMaxEndpointBytes = kCannEndpointCapacity - 1U;

static_assert(runtime::kMaxBootstrapEndpointBytes == kCannMaxEndpointBytes,
              "Runtime endpoint limit must match the locked CANN SHMEM ABI");
static_assert(sizeof(std::size_t) >= sizeof(uint64_t), "CANN SHMEM allocation requires a 64-bit size_t");

runtime::Status Success() { return {}; }

runtime::Status InvalidArgument(std::string message) {
  return runtime::Status{runtime::ErrorCode::InvalidArgument, std::nullopt, std::move(message)};
}

bool IsTimeoutError(int32_t error_code) {
  return error_code == ACLSHMEM_TIMEOUT_ERROR || error_code == ACLSHMEM_INNER_TIMEOUT;
}

runtime::Status CannFailure(std::string_view operation, int32_t error_code) {
  const runtime::ErrorCode category =
    IsTimeoutError(error_code) ? runtime::ErrorCode::Timeout : runtime::ErrorCode::CannError;
  return runtime::Status{category, error_code,
                         std::string(operation) + " failed with CANN error code " + std::to_string(error_code)};
}

runtime::Status AclFailure(std::string_view operation, int32_t error_code) {
  return runtime::Status{runtime::ErrorCode::CannError, error_code,
                         std::string(operation) + " failed with ACL error code " + std::to_string(error_code)};
}

runtime::Status CannFailureWithoutCode(std::string_view operation) {
  return runtime::Status{runtime::ErrorCode::CannError, std::nullopt,
                         std::string(operation) + " failed without an available CANN error code"};
}

bool StartsWith(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

runtime::Result<DeviceModel> NormalizeDeviceModel(const char *soc_name) {
  if (soc_name == nullptr || soc_name[0] == '\0') {
    return runtime::Result<DeviceModel>::Failure(runtime::Status{
      runtime::ErrorCode::UnsupportedCapability, std::nullopt, "aclrtGetSocName returned an empty SoC name"});
  }

  constexpr std::string_view kAscend910BNames[]{"Ascend910B1", "Ascend910B2", "Ascend910B2C",
                                                "Ascend910B3", "Ascend910B4", "Ascend910B4-1"};
  constexpr std::string_view kAscend910CNames[]{"Ascend910_9362", "Ascend910_9363", "Ascend910_9372", "Ascend910_9381",
                                                "Ascend910_9382", "Ascend910_9391", "Ascend910_9392"};

  const std::string_view name(soc_name);
  if (std::find(std::cbegin(kAscend910BNames), std::cend(kAscend910BNames), name) != std::cend(kAscend910BNames)) {
    return runtime::Result<DeviceModel>::Success(DeviceModel::kAscend910B);
  }
  if (std::find(std::cbegin(kAscend910CNames), std::cend(kAscend910CNames), name) != std::cend(kAscend910CNames)) {
    return runtime::Result<DeviceModel>::Success(DeviceModel::kAscend910C);
  }
  if (StartsWith(name, "Ascend950")) {
    return runtime::Result<DeviceModel>::Success(DeviceModel::kAscend950);
  }
  return runtime::Result<DeviceModel>::Failure(
    runtime::Status{runtime::ErrorCode::UnsupportedCapability, std::nullopt,
                    "unsupported SoC name returned by aclrtGetSocName: " + std::string(name)});
}

runtime::Result<data_op_engine_type_t> DataEngineMask(runtime::DataEngine data_engine) {
  switch (data_engine) {
    case runtime::DataEngine::Mte:
      return runtime::Result<data_op_engine_type_t>::Success(ACLSHMEM_DATA_OP_MTE);
  }
  return runtime::Result<data_op_engine_type_t>::Failure(
    InvalidArgument("data_engine must be a supported Runtime DataEngine value"));
}

int32_t CannSignalOp(SignalOp operation) {
  switch (operation) {
    case SignalOp::Set:
      return ACLSHMEM_SIGNAL_SET;
    case SignalOp::Add:
      return ACLSHMEM_SIGNAL_ADD;
  }
  return ACLSHMEM_SIGNAL_SET;
}

int32_t CannCompareOp(CompareOp comparison) {
  switch (comparison) {
    case CompareOp::Equal:
      return ACLSHMEM_CMP_EQ;
    case CompareOp::NotEqual:
      return ACLSHMEM_CMP_NE;
    case CompareOp::Greater:
      return ACLSHMEM_CMP_GT;
    case CompareOp::GreaterEqual:
      return ACLSHMEM_CMP_GE;
    case CompareOp::Less:
      return ACLSHMEM_CMP_LT;
    case CompareOp::LessEqual:
      return ACLSHMEM_CMP_LE;
  }
  return ACLSHMEM_CMP_EQ;
}

}  // namespace

runtime::Status initialize(const InitOptions &options) {
  if (options.effective_bootstrap_endpoint.empty() ||
      options.effective_bootstrap_endpoint.size() > kCannMaxEndpointBytes) {
    return InvalidArgument("effective_bootstrap_endpoint must contain between 1 and " +
                           std::to_string(kCannMaxEndpointBytes) +
                           " bytes, but got size=" + std::to_string(options.effective_bootstrap_endpoint.size()));
  }

  const auto data_engine_mask = DataEngineMask(options.data_engine);
  if (!data_engine_mask.ok()) {
    return data_engine_mask.error();
  }

  const int32_t tls_status = aclshmemx_set_conf_store_tls(false, nullptr, 0);
  if (tls_status != ACLSHMEM_SUCCESS) {
    return CannFailure("aclshmemx_set_conf_store_tls", tls_status);
  }

  aclshmemx_init_attr_t attributes{};
  attributes.my_pe = options.root_rank;
  attributes.n_pes = options.root_size;
  std::memcpy(attributes.ip_port, options.effective_bootstrap_endpoint.data(),
              options.effective_bootstrap_endpoint.size());
  attributes.ip_port[options.effective_bootstrap_endpoint.size()] = '\0';
  attributes.local_mem_size = options.heap_size_bytes;
  attributes.option_attr.version = kInitOptionalAttrVersion;
  attributes.option_attr.data_op_engine_type = data_engine_mask.value();
  attributes.option_attr.shm_init_timeout = options.timeout_seconds;
  attributes.option_attr.shm_create_timeout = options.timeout_seconds;
  attributes.option_attr.control_operation_timeout = options.timeout_seconds;
  attributes.option_attr.sockFd = kNoSocketFd;
  attributes.comm_args = nullptr;
  attributes.instance_id = kDefaultInstanceId;

  const int32_t init_status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
  return init_status == ACLSHMEM_SUCCESS ? Success() : CannFailure("aclshmemx_init_attr", init_status);
}

runtime::Status finalize() {
  const int32_t status = aclshmem_finalize();
  return status == ACLSHMEM_SUCCESS ? Success() : CannFailure("aclshmem_finalize", status);
}

runtime::Result<uintptr_t> allocate(uint64_t bytes) {
  void *allocation = aclshmem_malloc(static_cast<std::size_t>(bytes));
  if (allocation == nullptr) {
    return runtime::Result<uintptr_t>::Failure(CannFailureWithoutCode("aclshmem_malloc"));
  }
  return runtime::Result<uintptr_t>::Success(reinterpret_cast<uintptr_t>(allocation));
}

runtime::Result<uintptr_t> aligned_allocate(uint64_t alignment_bytes, uint64_t bytes) {
  void *allocation = aclshmem_align(static_cast<std::size_t>(alignment_bytes), static_cast<std::size_t>(bytes));
  if (allocation == nullptr) {
    return runtime::Result<uintptr_t>::Failure(CannFailureWithoutCode("aclshmem_align"));
  }
  return runtime::Result<uintptr_t>::Success(reinterpret_cast<uintptr_t>(allocation));
}

void free_memory(uintptr_t allocation_base) { aclshmem_free(reinterpret_cast<void *>(allocation_base)); }

void barrier_on_stream(const runtime::StreamView &stream) {
  aclshmemx_barrier_on_stream(ACLSHMEM_TEAM_WORLD, reinterpret_cast<aclrtStream>(stream.native_handle));
}

void put_on_stream(uintptr_t remote_dst, uintptr_t local_src, uint64_t bytes, int32_t target_pe,
                   const runtime::StreamView &stream) {
  aclshmemx_putmem_on_stream(reinterpret_cast<void *>(remote_dst), reinterpret_cast<void *>(local_src),
                             static_cast<std::size_t>(bytes), target_pe,
                             reinterpret_cast<aclrtStream>(stream.native_handle));
}

void get_on_stream(uintptr_t local_dst, uintptr_t remote_src, uint64_t bytes, int32_t source_pe,
                   const runtime::StreamView &stream) {
  aclshmemx_getmem_on_stream(reinterpret_cast<void *>(local_dst), reinterpret_cast<void *>(remote_src),
                             static_cast<std::size_t>(bytes), source_pe,
                             reinterpret_cast<aclrtStream>(stream.native_handle));
}

void signal_on_stream(uintptr_t remote_signal, int32_t value, SignalOp operation, int32_t target_pe,
                      const runtime::StreamView &stream) {
  aclshmemx_signal_op_on_stream(reinterpret_cast<int32_t *>(remote_signal), value, CannSignalOp(operation), target_pe,
                                reinterpret_cast<aclrtStream>(stream.native_handle));
}

void wait_signal_on_stream(uintptr_t local_signal, CompareOp comparison, int32_t value,
                           const runtime::StreamView &stream) {
  aclshmemx_signal_wait_until_on_stream(reinterpret_cast<int32_t *>(local_signal), CannCompareOp(comparison), value,
                                        reinterpret_cast<aclrtStream>(stream.native_handle));
}

runtime::Result<int32_t> query_current_device_index() {
  int32_t current_device = kInvalidDeviceIndex;
  const aclError device_status = aclrtGetDevice(&current_device);
  if (device_status != ACL_SUCCESS) {
    // aclrtGetDevice only fails when this thread has no active device (e.g. SHMEM initialize before any
    // torch.npu.set_device), so attach the actionable guidance instead of relying on a specific raw code.
    runtime::Status failure = AclFailure("aclrtGetDevice", static_cast<int32_t>(device_status));
    failure.message += ": no active NPU device on this thread; set the current NPU device "
                       "(torch.npu.set_device) before initializing SHMEM";
    return runtime::Result<int32_t>::Failure(std::move(failure));
  }
  return runtime::Result<int32_t>::Success(current_device);
}

runtime::Result<DeviceIdentity> query_device_identity() {
  const auto current_device = query_current_device_index();
  if (!current_device.ok()) {
    return runtime::Result<DeviceIdentity>::Failure(current_device.error());
  }

  const char *soc_name = aclrtGetSocName();
  const auto model = NormalizeDeviceModel(soc_name);
  if (!model.ok()) {
    return runtime::Result<DeviceIdentity>::Failure(model.error());
  }
  return runtime::Result<DeviceIdentity>::Success(DeviceIdentity{current_device.value(), model.value(), soc_name});
}

}  // namespace hyper_parallel::multicore::shmem::cann::host
