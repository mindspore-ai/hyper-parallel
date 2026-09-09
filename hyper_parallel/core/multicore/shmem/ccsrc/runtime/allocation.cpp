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

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace hyper_parallel::multicore::shmem::runtime {
namespace {

Status MakeError(ErrorCode error_code, std::string message) {
  return Status{error_code, std::nullopt, std::move(message)};
}

bool IsPowerOfTwo(uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

}  // namespace

AllocationRegistry::AllocationRegistry(int32_t device_index, uint64_t next_allocation_id)
    : device_index_(device_index), next_allocation_id_(next_allocation_id) {}

Status AllocationRegistry::Validate(const AllocationSpec &spec) const {
  if (spec.bytes == 0) {
    return MakeError(ErrorCode::InvalidArgument, "Allocation bytes must be greater than zero");
  }
  if (spec.alignment_bytes != 0 && !IsPowerOfTwo(spec.alignment_bytes)) {
    return MakeError(ErrorCode::InvalidArgument,
                     "Allocation alignment_bytes must be zero or a power of two, but got alignment_bytes=" +
                       std::to_string(spec.alignment_bytes));
  }
  if (next_allocation_id_ == std::numeric_limits<uint64_t>::max()) {
    return MakeError(ErrorCode::InvalidState, "Allocation identity space is exhausted");
  }
  return Status{};
}

AllocationRecord AllocationRegistry::Register(const AllocationSpec &spec, uintptr_t allocation_base) {
  AllocationRecord record{next_allocation_id_, allocation_base, spec.bytes, spec.alignment_bytes};
  const auto inserted = records_.emplace(record.allocation_id, record);
  if (!inserted.second) {
    throw std::logic_error("Allocation identity is already active");
  }

  ++next_allocation_id_;
  allocated_bytes_ += spec.bytes;
  max_allocated_bytes_ = std::max(max_allocated_bytes_, allocated_bytes_);
  return record;
}

Result<AllocationRecord> AllocationRegistry::MatchForFree(const AllocationView &buffer) const {
  const auto result = Find(buffer.allocation_id);
  if (!result.ok()) {
    return result;
  }

  const AllocationRecord &record = result.value();
  if (buffer.device_index != device_index_ || buffer.view_data != record.allocation_base ||
      buffer.view_bytes != record.allocation_bytes) {
    return Result<AllocationRecord>::Failure(MakeError(
      ErrorCode::InvalidArgument,
      "Free requires the complete active Allocation, but got view_data=" + std::to_string(buffer.view_data) +
        ", view_bytes=" + std::to_string(buffer.view_bytes) + ", device_index=" + std::to_string(buffer.device_index) +
        ", Runtime device_index=" + std::to_string(device_index_)));
  }
  return result;
}

Result<AllocationRecord> AllocationRegistry::Resolve(const AllocationView &buffer) const {
  const auto result = Find(buffer.allocation_id);
  if (!result.ok()) {
    return result;
  }

  const AllocationRecord &record = result.value();
  if (buffer.device_index != device_index_ || buffer.view_data < record.allocation_base) {
    return Result<AllocationRecord>::Failure(MakeError(
      ErrorCode::InvalidArgument,
      "Buffer view must be a contiguous range within its active Allocation, but got view_data=" +
        std::to_string(buffer.view_data) + ", view_bytes=" + std::to_string(buffer.view_bytes) + ", device_index=" +
        std::to_string(buffer.device_index) + ", Runtime device_index=" + std::to_string(device_index_)));
  }

  const uint64_t offset = static_cast<uint64_t>(buffer.view_data - record.allocation_base);
  if (offset > record.allocation_bytes || buffer.view_bytes > record.allocation_bytes - offset) {
    return Result<AllocationRecord>::Failure(
      MakeError(ErrorCode::InvalidArgument,
                "Buffer view must be a contiguous range within its active Allocation, but got offset_bytes=" +
                  std::to_string(offset) + ", view_bytes=" + std::to_string(buffer.view_bytes) +
                  ", allocation_bytes=" + std::to_string(record.allocation_bytes)));
  }
  return result;
}

void AllocationRegistry::Remove(uint64_t allocation_id) {
  const auto it = records_.find(allocation_id);
  if (it == records_.end()) {
    throw std::logic_error("Removing an inactive Allocation identity");
  }
  allocated_bytes_ -= it->second.allocation_bytes;
  records_.erase(it);
}

uint64_t AllocationRegistry::allocated_count() const noexcept { return static_cast<uint64_t>(records_.size()); }

uint64_t AllocationRegistry::allocated_bytes() const noexcept { return allocated_bytes_; }

uint64_t AllocationRegistry::max_allocated_bytes() const noexcept { return max_allocated_bytes_; }

uint64_t AllocationRegistry::next_allocation_id() const noexcept { return next_allocation_id_; }

std::vector<AllocationRecord> AllocationRegistry::ActiveRecords() const {
  std::vector<AllocationRecord> records;
  records.reserve(records_.size());
  for (const auto &entry : records_) {
    records.push_back(entry.second);
  }
  std::sort(records.begin(), records.end(),
            [](const AllocationRecord &left, const AllocationRecord &right) {
              return left.allocation_id < right.allocation_id;
            });
  return records;
}

Result<AllocationRecord> AllocationRegistry::Find(uint64_t allocation_id) const {
  const auto it = records_.find(allocation_id);
  if (it != records_.end()) {
    return Result<AllocationRecord>::Success(it->second);
  }
  if (allocation_id != 0 && allocation_id < next_allocation_id_) {
    return Result<AllocationRecord>::Failure(
      MakeError(ErrorCode::DoubleFree,
                "Allocation identity has already been released: allocation_id=" + std::to_string(allocation_id)));
  }
  return Result<AllocationRecord>::Failure(MakeError(
    ErrorCode::InvalidArgument,
    "Allocation identity must be issued by this Runtime, but got allocation_id=" + std::to_string(allocation_id)));
}

}  // namespace hyper_parallel::multicore::shmem::runtime
