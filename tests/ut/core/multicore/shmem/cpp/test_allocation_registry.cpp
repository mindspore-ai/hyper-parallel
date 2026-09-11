/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/allocation.h"

#include <cstdint>

#include "test_support.h"

namespace hyper_parallel::multicore::shmem::tests {
namespace {

using runtime::AllocationRecord;
using runtime::AllocationRegistry;
using runtime::AllocationSpec;
using runtime::AllocationView;
using runtime::ErrorCode;

constexpr int32_t kDeviceIndex = 2;
constexpr uint64_t kFirstAllocationId = 1U;
constexpr uintptr_t kAllocationBase = 0x10000U;

AllocationView CompleteView(const AllocationRecord &record) {
  return {record.allocation_id, record.allocation_base, record.allocation_bytes, kDeviceIndex};
}

}  // namespace

void test_registry_match_for_free_requires_complete() {
  AllocationRegistry registry(kDeviceIndex, kFirstAllocationId);
  const AllocationRecord record = registry.Register(AllocationSpec{256U, 0U}, kAllocationBase);

  const auto complete = registry.MatchForFree(CompleteView(record));
  Check(complete.ok(), "The complete Allocation must match for free");

  const auto subview = registry.MatchForFree({record.allocation_id, kAllocationBase + 32U, 224U, kDeviceIndex});
  Check(!subview.ok(), "A subview must not match for free");
  CheckEq(subview.error().error_code, ErrorCode::InvalidArgument, "Subview free-match error code");
  const auto wrong_size = registry.MatchForFree({record.allocation_id, kAllocationBase, 128U, kDeviceIndex});
  Check(!wrong_size.ok(), "A partial byte range must not match for free");
  CheckEq(wrong_size.error().error_code, ErrorCode::InvalidArgument, "Partial-range free-match error code");
  const auto wrong_device = registry.MatchForFree({record.allocation_id, kAllocationBase, 256U, kDeviceIndex + 1});
  Check(!wrong_device.ok(), "A view on another device must not match for free");
  CheckEq(wrong_device.error().error_code, ErrorCode::InvalidArgument, "Wrong-device free-match error code");
}

void test_registry_double_free_detected() {
  AllocationRegistry registry(kDeviceIndex, kFirstAllocationId);
  const AllocationRecord allocation_a = registry.Register(AllocationSpec{128U, 0U}, kAllocationBase);
  const AllocationView old_view = CompleteView(allocation_a);
  registry.Remove(allocation_a.allocation_id);

  const AllocationRecord allocation_b = registry.Register(AllocationSpec{128U, 0U}, kAllocationBase);
  Check(allocation_b.allocation_id != allocation_a.allocation_id,
        "A reused address must receive a new Allocation identity");
  const auto old_free = registry.MatchForFree(old_view);
  const auto old_resolve = registry.Resolve(old_view);
  Check(!old_free.ok(), "A released identity must remain rejected for free after address reuse");
  CheckEq(old_free.error().error_code, ErrorCode::DoubleFree,
          "Released Allocation free error code after address reuse");
  Check(!old_resolve.ok(), "A released identity must not resolve a newer Allocation at the same address");
  CheckEq(old_resolve.error().error_code, ErrorCode::DoubleFree,
          "Released Allocation resolve error code after address reuse");

  const auto new_free = registry.MatchForFree(CompleteView(allocation_b));
  Check(new_free.ok(), "Address reuse must not invalidate the newer Allocation identity");
  registry.Remove(allocation_b.allocation_id);
}

void test_registry_resolve_allows_subview() {
  AllocationRegistry registry(kDeviceIndex, kFirstAllocationId);
  const AllocationRecord record = registry.Register(AllocationSpec{128U, 0U}, kAllocationBase);

  const auto subview = registry.Resolve({record.allocation_id, kAllocationBase + 32U, 64U, kDeviceIndex});
  Check(subview.ok(), "A contiguous subview must resolve to its complete Allocation");
  CheckEq(subview.value().allocation_id, record.allocation_id, "Resolved Allocation identity for a subview");

  const auto beyond_end = registry.Resolve({record.allocation_id, kAllocationBase + 120U, 16U, kDeviceIndex});
  Check(!beyond_end.ok(), "A subview extending beyond the Allocation must be rejected");
  CheckEq(beyond_end.error().error_code, ErrorCode::InvalidArgument, "Beyond-end subview error code");
  const auto before_base = registry.Resolve({record.allocation_id, kAllocationBase - 1U, 1U, kDeviceIndex});
  Check(!before_base.ok(), "A subview before the Allocation base must be rejected without address wraparound");
  CheckEq(before_base.error().error_code, ErrorCode::InvalidArgument, "Before-base subview error code");
}

void test_registry_diagnostics_are_ordered_and_track_high_watermark() {
  AllocationRegistry registry(kDeviceIndex, kFirstAllocationId);
  const AllocationRecord allocation_a = registry.Register(AllocationSpec{64U, 0U}, kAllocationBase);
  const AllocationRecord allocation_b = registry.Register(AllocationSpec{32U, 0U}, kAllocationBase + 128U);
  CheckEq(registry.max_allocated_bytes(), 96U, "Allocation high watermark at peak usage");

  registry.Remove(allocation_a.allocation_id);
  const AllocationRecord allocation_c = registry.Register(AllocationSpec{16U, 0U}, kAllocationBase + 256U);
  const auto active_records = registry.ActiveRecords();
  CheckEq(active_records.size(), 2U, "Active diagnostic record count");
  CheckEq(active_records[0].allocation_id, allocation_b.allocation_id, "First active diagnostic Allocation identity");
  CheckEq(active_records[1].allocation_id, allocation_c.allocation_id, "Second active diagnostic Allocation identity");
  CheckEq(registry.max_allocated_bytes(), 96U, "Allocation high watermark after usage decreases");
}

}  // namespace hyper_parallel::multicore::shmem::tests
