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

#include <cstdint>
#include <cstring>
#include <string>

#include "fakes/fake_acl_cann.h"
#include "test_support.h"

namespace hyper_parallel::multicore::shmem::tests {
namespace {

using runtime::DataEngine;
using runtime::ErrorCode;

constexpr int32_t kRootRank = 2;
constexpr int32_t kRootSize = 8;
constexpr uint64_t kHeapBytes = 4U * 1024U * 1024U;
constexpr uint32_t kTimeoutSeconds = 37U;
constexpr std::string_view kEndpoint = "tcp://127.0.0.1:8662";

cann::InitOptions Options() { return {kRootRank, kRootSize, kHeapBytes, kTimeoutSeconds, DataEngine::Mte, kEndpoint}; }

}  // namespace

void test_host_init_attr_fields_and_order() {
  fake_acl_cann::Reset();
  const auto status = cann::host::initialize(Options());
  const auto &state = fake_acl_cann::Get();
  CheckEq(status.error_code, ErrorCode::Ok, "Host initialization error code");
  CheckEq(state.calls.size(), 2U, "CANN initialization call count");
  CheckEq(state.calls[0], std::string("tls"), "First CANN initialization call");
  CheckEq(state.calls[1], std::string("init"), "Second CANN initialization call");
  CheckEq(state.init_mode, ACLSHMEMX_INIT_WITH_DEFAULT, "CANN initialization mode");
  CheckEq(state.init_attributes.my_pe, kRootRank, "CANN initialization Root rank");
  CheckEq(state.init_attributes.n_pes, kRootSize, "CANN initialization Root size");
  CheckEq(state.init_attributes.local_mem_size, kHeapBytes, "CANN initialization Heap capacity");
  CheckEq(state.init_attributes.option_attr.data_op_engine_type, ACLSHMEM_DATA_OP_MTE,
          "CANN initialization data engine mask");
  CheckEq(state.init_attributes.option_attr.shm_init_timeout, kTimeoutSeconds, "CANN SHMEM initialization timeout");
  CheckEq(state.init_attributes.option_attr.shm_create_timeout, kTimeoutSeconds, "CANN SHMEM creation timeout");
  CheckEq(state.init_attributes.option_attr.control_operation_timeout, kTimeoutSeconds,
          "CANN control-operation timeout");
  CheckEq(state.init_attributes.option_attr.sockFd, -1, "CANN external socket descriptor");
  CheckEq(state.init_attributes.comm_args, nullptr, "CANN optional communication arguments");
  CheckEq(state.init_attributes.instance_id, 0U, "CANN instance ID");
  CheckEq(std::string(state.init_attributes.ip_port), kEndpoint, "CANN effective bootstrap endpoint");
  const int32_t expected_version = static_cast<int32_t>((1U << 16U) + sizeof(aclshmem_init_optional_attr_t));
  CheckEq(state.init_attributes.option_attr.version, expected_version, "CANN optional-attribute ABI version");
}

void test_host_timeout_and_cann_error_mapping() {
  fake_acl_cann::Reset();
  fake_acl_cann::Get().init_result = ACLSHMEM_TIMEOUT_ERROR;
  auto status = cann::host::initialize(Options());
  CheckEq(status.error_code, ErrorCode::Timeout, "CANN timeout Runtime error code");
  Check(status.cann_error_code.has_value(), "CANN timeout must retain its original code");
  CheckEq(*status.cann_error_code, ACLSHMEM_TIMEOUT_ERROR, "CANN timeout original error code");

  fake_acl_cann::Reset();
  constexpr int32_t kCannFailure = 4099;
  fake_acl_cann::Get().init_result = kCannFailure;
  status = cann::host::initialize(Options());
  CheckEq(status.error_code, ErrorCode::CannError, "Non-timeout CANN Runtime error code");
  Check(status.cann_error_code.has_value(), "Non-timeout CANN failure must retain its original code");
  CheckEq(*status.cann_error_code, kCannFailure, "Non-timeout CANN original error code");
}

void test_host_alloc_nullptr_and_free_passthrough() {
  fake_acl_cann::Reset();
  constexpr uint64_t kBytes = 4096U;
  constexpr uint64_t kAlignment = 512U;

  auto allocation = cann::host::allocate(kBytes);
  Check(allocation.ok(), "Ordinary CANN allocation must succeed");
  CheckEq(allocation.value(), reinterpret_cast<uintptr_t>(fake_acl_cann::Get().malloc_result),
          "Ordinary allocation address");
  CheckEq(fake_acl_cann::Get().malloc_bytes, kBytes, "Ordinary allocation byte count");

  auto aligned = cann::host::aligned_allocate(kAlignment, kBytes);
  Check(aligned.ok(), "Aligned CANN allocation must succeed");
  CheckEq(aligned.value(), reinterpret_cast<uintptr_t>(fake_acl_cann::Get().aligned_malloc_result),
          "Aligned allocation address");
  CheckEq(fake_acl_cann::Get().alignment_bytes, kAlignment, "Aligned allocation alignment");
  CheckEq(fake_acl_cann::Get().aligned_bytes, kBytes, "Aligned allocation byte count");

  const uintptr_t address = allocation.value();
  cann::host::free_memory(address);
  CheckEq(fake_acl_cann::Get().freed_address, reinterpret_cast<void *>(address), "Freed Allocation address");

  fake_acl_cann::Get().malloc_result = nullptr;
  allocation = cann::host::allocate(kBytes);
  Check(!allocation.ok(), "Null ordinary allocation must fail");
  CheckEq(allocation.error().error_code, ErrorCode::CannError, "Null ordinary allocation error code");
  Check(!allocation.error().cann_error_code.has_value(),
        "Null ordinary allocation must not fabricate a raw CANN error code");
  fake_acl_cann::Get().aligned_malloc_result = nullptr;
  aligned = cann::host::aligned_allocate(kAlignment, kBytes);
  Check(!aligned.ok(), "Null aligned allocation must fail");
  CheckEq(aligned.error().error_code, ErrorCode::CannError, "Null aligned allocation error code");
  Check(!aligned.error().cann_error_code.has_value(),
        "Null aligned allocation must not fabricate a raw CANN error code");
}

void test_host_stream_operations_forward_arguments() {
  uint8_t source[32]{};
  uint8_t destination[32]{};
  int32_t signal = 0;
  constexpr int32_t kTargetPe = 3;
  constexpr int32_t kSignalValue = 7;
  auto stream = reinterpret_cast<aclrtStream>(0x40000U);
  const runtime::StreamView stream_view{reinterpret_cast<uintptr_t>(stream), 0};

  fake_acl_cann::Reset();
  cann::host::put_on_stream(reinterpret_cast<uintptr_t>(destination), reinterpret_cast<uintptr_t>(source),
                            sizeof(source), kTargetPe, stream_view);
  auto &state = fake_acl_cann::Get();
  CheckEq(state.calls.back(), std::string("put_on_stream"), "Host Put CANN call");
  CheckEq(state.first_address, static_cast<void *>(destination), "Host Put remote destination");
  CheckEq(state.second_address, static_cast<void *>(source), "Host Put local source");
  CheckEq(state.transfer_bytes, sizeof(source), "Host Put byte count");
  CheckEq(state.target_pe, kTargetPe, "Host Put target Root PE");
  CheckEq(state.stream, stream, "Host Put stream");

  fake_acl_cann::Reset();
  cann::host::get_on_stream(reinterpret_cast<uintptr_t>(destination), reinterpret_cast<uintptr_t>(source),
                            sizeof(source), kTargetPe, stream_view);
  CheckEq(fake_acl_cann::Get().calls.back(), std::string("get_on_stream"), "Host Get CANN call");
  CheckEq(fake_acl_cann::Get().target_pe, kTargetPe, "Host Get source Root PE");

  fake_acl_cann::Reset();
  cann::host::signal_on_stream(reinterpret_cast<uintptr_t>(&signal), kSignalValue, cann::SignalOp::Add, kTargetPe,
                               stream_view);
  CheckEq(fake_acl_cann::Get().signal_operation, ACLSHMEM_SIGNAL_ADD, "Host Signal operation");
  CheckEq(fake_acl_cann::Get().signal_value, kSignalValue, "Host Signal value");

  fake_acl_cann::Reset();
  cann::host::wait_signal_on_stream(reinterpret_cast<uintptr_t>(&signal), cann::CompareOp::LessEqual, kSignalValue,
                                    stream_view);
  CheckEq(fake_acl_cann::Get().signal_comparison, ACLSHMEM_CMP_LE, "Host Signal wait comparison");
}

}  // namespace hyper_parallel::multicore::shmem::tests
