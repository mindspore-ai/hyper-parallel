/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_plane/rma.h"
#include "data_plane/sync.h"

#include <cstdint>

#include "fakes/fake_shmem_device.h"
#include "test_support.h"

namespace hyper_parallel::multicore::shmem::tests {
namespace {

static_assert(cann::device::ToCannCompareOp(data_plane::CompareOp::Equal) == ACLSHMEM_CMP_EQ);
static_assert(cann::device::ToCannCompareOp(data_plane::CompareOp::NotEqual) == ACLSHMEM_CMP_NE);
static_assert(cann::device::ToCannCompareOp(data_plane::CompareOp::Greater) == ACLSHMEM_CMP_GT);
static_assert(cann::device::ToCannCompareOp(data_plane::CompareOp::GreaterEqual) == ACLSHMEM_CMP_GE);
static_assert(cann::device::ToCannCompareOp(data_plane::CompareOp::Less) == ACLSHMEM_CMP_LT);
static_assert(cann::device::ToCannCompareOp(data_plane::CompareOp::LessEqual) == ACLSHMEM_CMP_LE);

}  // namespace

void test_put_get_forward_once() {
  fake_device::Reset();
  uint8_t source[64]{};
  uint8_t destination[64]{};
  constexpr uint32_t kBytes = sizeof(source);
  constexpr int32_t kTargetPe = 3;

  data_plane::put(destination, source, kBytes, kTargetPe);
  data_plane::get(destination, source, kBytes, kTargetPe);
  const auto &calls = fake_device::Calls();
  CheckEq(calls.size(), 2U, "CANN call count after one nonzero Put and Get");
  CheckEq(calls[0].kind, fake_device::CallKind::Put, "First CANN call kind");
  CheckEq(calls[0].first_address, static_cast<void *>(destination), "Put remote destination");
  CheckEq(calls[0].second_address, static_cast<void *>(source), "Put local source");
  CheckEq(calls[0].bytes, kBytes, "Put byte count");
  CheckEq(calls[0].target_pe, kTargetPe, "Put Root target PE");
  CheckEq(calls[1].kind, fake_device::CallKind::Get, "Second CANN call kind");
  CheckEq(calls[1].first_address, static_cast<void *>(destination), "Get local destination");
  CheckEq(calls[1].second_address, static_cast<void *>(source), "Get remote source");
  CheckEq(calls[1].bytes, kBytes, "Get byte count");
  CheckEq(calls[1].target_pe, kTargetPe, "Get Root target PE");
}

void test_put_signal_orders_data_before_signal() {
  fake_device::Reset();
  uint8_t source[32]{};
  uint8_t destination[32]{};
  int32_t signal = 0;
  constexpr int32_t kTargetPe = 2;
  constexpr int32_t kSignalValue = 7;

  data_plane::put_signal(destination, source, sizeof(source), &signal, kSignalValue, data_plane::SignalOp::Set,
                         kTargetPe);
  const auto &calls = fake_device::Calls();
  CheckEq(calls.size(), 2U, "CANN call count after Put+Signal");
  CheckEq(calls[0].kind, fake_device::CallKind::Put, "First Put+Signal CANN call kind");
  CheckEq(calls[1].kind, fake_device::CallKind::Signal, "Second Put+Signal CANN call kind");
  CheckEq(calls[1].operation, ACLSHMEM_SIGNAL_SET, "Put+Signal operation");
  CheckEq(calls[1].value, kSignalValue, "Put+Signal value");
  CheckEq(calls[1].target_pe, kTargetPe, "Put+Signal Root target PE");
}

void test_signal_compare_mapping() {
  constexpr data_plane::CompareOp kCompare = data_plane::CompareOp::GreaterEqual;
  constexpr int32_t kExpectedRawCompare = ACLSHMEM_CMP_GE;
  constexpr int32_t kExpectedValue = 5;
  constexpr int32_t kObservedValue = 11;
  int32_t signal = 0;
  fake_device::Reset();
  fake_device::SetObservedSignal(kObservedValue);
  const int32_t observed = data_plane::signal_wait(&signal, kCompare, kExpectedValue);
  const auto &calls = fake_device::Calls();
  CheckEq(observed, kObservedValue, "Signal wait observed value");
  CheckEq(calls.size(), 1U, "CANN call count after Signal wait");
  CheckEq(calls[0].kind, fake_device::CallKind::Wait, "Signal wait CANN call kind");
  CheckEq(calls[0].first_address, static_cast<void *>(&signal), "Signal wait local address");
  CheckEq(calls[0].operation, kExpectedRawCompare, "Signal wait comparison operation");
  CheckEq(calls[0].value, kExpectedValue, "Signal wait expected value");
}

}  // namespace hyper_parallel::multicore::shmem::tests
