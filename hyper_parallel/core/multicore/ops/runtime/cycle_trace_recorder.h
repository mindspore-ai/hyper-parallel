/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file cycle_trace_recorder.h
 * @brief Reusable per-core cycle interval recorder for TaskDesc-based megakernels.
 */

#ifndef MULTICORE_SCHEDULER_CYCLE_TRACE_RECORDER_H
#define MULTICORE_SCHEDULER_CYCLE_TRACE_RECORDER_H

#include "kernel_operator.h"
#include "runtime_config.hpp"

using namespace AscendC;  // NOLINT(build/namespaces)

struct CycleTraceCoreHeader {
  uint64_t entryCycle;
  uint32_t recordCount;
  uint32_t droppedCount;
  uint32_t coreType;
  uint32_t blockId;
  uint32_t recordCapacity;
  uint32_t reserved[9];
};

struct CycleTraceRecord {
  uint64_t startCycle;
  uint64_t endCycle;
  uint32_t descId;
  uint32_t taskId;
  uint32_t stageTaskIndex;
  uint32_t ownerId;
};

static_assert(sizeof(CycleTraceCoreHeader) == 64, "Unexpected cycle trace core header size");
static_assert(sizeof(CycleTraceRecord) == 32, "Unexpected cycle trace record size");

class CycleTraceRecorder {
 public:
  static constexpr uint32_t MAX_RECORDS_PER_CORE = 256;

  __aicore__ inline void Init(uint32_t worker_id, GM_ADDR profile_buffer, uint32_t aic_record_capacity,
                             uint32_t aiv_record_capacity) {
    aic_record_capacity =
      aic_record_capacity > MAX_RECORDS_PER_CORE ? MAX_RECORDS_PER_CORE : aic_record_capacity;
    aiv_record_capacity =
      aiv_record_capacity > MAX_RECORDS_PER_CORE ? MAX_RECORDS_PER_CORE : aiv_record_capacity;
    uint32_t aic_stride_bytes =
      sizeof(CycleTraceCoreHeader) + aic_record_capacity * sizeof(CycleTraceRecord);
    uint32_t aiv_stride_bytes =
      sizeof(CycleTraceCoreHeader) + aiv_record_capacity * sizeof(CycleTraceRecord);
#ifdef __DAV_C220_CUBE__
    uint32_t core_type = 1;
    record_capacity_ = aic_record_capacity;
    GM_ADDR profile_slot_ptr = profile_buffer + worker_id * aic_stride_bytes;
#else
    uint32_t core_type = 2;
    record_capacity_ = aiv_record_capacity;
    GM_ADDR profile_slot_ptr = profile_buffer + NUM_WORKERS_CUBE * aic_stride_bytes +
                               worker_id * aiv_stride_bytes;
#endif
    profile_header_ = reinterpret_cast<__gm__ CycleTraceCoreHeader *>(profile_slot_ptr);
    record_count_ = 0;
    profile_header_->entryCycle = Now();
    profile_header_->recordCount = 0;
    profile_header_->droppedCount = 0;
    profile_header_->coreType = core_type;
    profile_header_->blockId = worker_id;
    profile_header_->recordCapacity = record_capacity_;
    for (uint32_t i = 0; i < 9; ++i) {
      profile_header_->reserved[i] = 0;
    }
  }

  __aicore__ inline uint64_t Now() const {
    PipeBarrier<PIPE_ALL>();
    return static_cast<uint64_t>(AscendC::GetSystemCycle());
  }

  __aicore__ inline void Record(uint32_t desc_id, uint32_t task_id, uint32_t stage_task_index, uint32_t owner_id,
                                uint64_t start_cycle, uint64_t end_cycle) {
    if (record_count_ >= record_capacity_) {
      profile_header_->droppedCount = profile_header_->droppedCount + 1;
      return;
    }
    GM_ADDR record_ptr = reinterpret_cast<GM_ADDR>(profile_header_) + sizeof(CycleTraceCoreHeader) +
                         record_count_ * sizeof(CycleTraceRecord);
    __gm__ CycleTraceRecord *record = reinterpret_cast<__gm__ CycleTraceRecord *>(record_ptr);
    record->startCycle = start_cycle;
    record->endCycle = end_cycle;
    record->descId = desc_id;
    record->taskId = task_id;
    record->stageTaskIndex = stage_task_index;
    record->ownerId = owner_id;
    PipeBarrier<PIPE_ALL>();
    record_count_++;
    profile_header_->recordCount = record_count_;
    PipeBarrier<PIPE_ALL>();
  }

 private:
  __gm__ CycleTraceCoreHeader *profile_header_ = nullptr;
  uint32_t record_count_ = 0;
  uint32_t record_capacity_ = 0;
};

#endif  // MULTICORE_SCHEDULER_CYCLE_TRACE_RECORDER_H
