/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file worker_kernel.h
 * @brief Shared CRTP base class for AIC/AIV scheduling loop and event synchronization.
 *
 * Each op derives KernelWorker : KernelWorkerBase<KernelWorker> and supplies:
 *   - static constexpr uint32_t TILING_IDX  — index into input_list for tiling params
 *   - static constexpr uint32_t EVENT_IDX   — index into input_list for all_event_counters
 *   - void ExecuteComputeKernel(TaskDesc)    — op-specific task dispatch switch
 *
 * Compute kernels (ExecuteMatmul, ExecuteShmemPutMem, etc.) are NOT part of this class;
 * they belong in each op's worker_kernel.cpp.
 */

#ifndef MULTICORE_SCHEDULER_WORKER_KERNEL_H
#define MULTICORE_SCHEDULER_WORKER_KERNEL_H

#include "kernel_operator.h"
#include "runtime_config.hpp"

using AscendC::TPipe;
using AscendC::TBuf;
using AscendC::LocalTensor;
using AscendC::GlobalTensor;
using AscendC::SyncFunc;
using AscendC::DataCopy;
using AscendC::DataCacheCleanAndInvalid;
using AscendC::PipeBarrier;
using AscendC::PIPE_ALL;
using AscendC::CacheLine;
using AscendC::DcciDst;

template <typename Derived>
class KernelWorkerBase {
 public:
  __aicore__ inline KernelWorkerBase() {}

  __aicore__ inline void Init(uint32_t worker_id, __gm__ uint8_t *runtimeConfigPtr, GM_ADDR *input_list) {
    this->worker_id_ = worker_id;
    this->runtimeConfigPtr = runtimeConfigPtr;

    this->tpipe_.InitBuffer(tBuf, DISPATCH_TOKEN_UB_SIZE);
    tpipe_.Destroy();

    all_event_counters.SetGlobalBuffer((__gm__ int32_t *)(input_list[Derived::EVENT_IDX]), MAX_EVENT_NUM);

    all_event_num_triggers.SetGlobalBuffer((__gm__ int32_t *)(this->runtimeConfigPtr + getAllEventNumTriggersOffset()),
                                           MAX_EVENT_NUM);

    vector_task_indexs.SetGlobalBuffer((__gm__ int32_t *)(this->runtimeConfigPtr + getVectorTaskIndexsOffset()),
                                       TASK_TYPE_INDEX_NUM);

    cube_task_indexs.SetGlobalBuffer((__gm__ int32_t *)(this->runtimeConfigPtr + getCubeTaskIndexsOffset()),
                                     TASK_TYPE_INDEX_NUM);

    atomic_add_values.SetGlobalBuffer((__gm__ int32_t *)(this->runtimeConfigPtr + getAtomicAddValuesOffset()), 8);

    this->input_list = input_list;
    this->task_num = getTaskNum(this->runtimeConfigPtr);
    this->vector_task_num = getTaskIndexNumByTaskType(this->runtimeConfigPtr, TaskAiCoreType::TASK_AICORE_VECTOR);
    this->cube_task_num = getTaskIndexNumByTaskType(this->runtimeConfigPtr, TaskAiCoreType::TASK_AICORE_CUBE);
    this->core_num = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 5);
    this->vector_num = this->core_num * 2;
  }

  __aicore__ inline void Process() {
#ifdef __DAV_C220_CUBE__
    uint32_t block_idx = this->worker_id_;
    do {
      if (block_idx >= this->cube_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTask(task_index);
      block_idx = block_idx + this->core_num;
    } while (1);
#else
    if (this->worker_id_ % 2 == 0) {
      return;
    }
    uint32_t block_idx = this->worker_id_ / 2;
    do {
      if (block_idx >= this->vector_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTask(task_index);
      uint32_t half_num = this->vector_num / 2;
      block_idx = block_idx + half_num;
    } while (1);
#endif
  }

 protected:
  __aicore__ inline TaskId GetTaskIndex(uint32_t task_id) {
#ifdef __DAV_C220_CUBE__
    return cube_task_indexs.GetValue(task_id);
#else
    return vector_task_indexs.GetValue(task_id);
#endif
  }

  __aicore__ inline void AtomicAddForAllEventCounters(uint32_t event_index) {
#ifdef __DAV_C220_CUBE__
    this->localSet = tBuf.GetWithOffset<int32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopy(this->localSet, this->atomic_add_values, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    AscendC::SetAtomicAdd<int32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopy(this->all_event_counters[event_index], this->localSet, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    AscendC::SetAtomicNone();
    tBuf.FreeTensor(this->localSet);
#else
    this->localSet = tBuf.GetWithOffset<int32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
    this->localSet.SetValue(0, 1);
    for (int32_t i = 1; i < EXP_TOKEN_COUNT_FLAG_CNT; ++i) {
      this->localSet.SetValue(i, 0);
    }
    AscendC::SetAtomicAdd<int32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(this->all_event_counters[event_index], this->localSet, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    AscendC::SetAtomicNone();
    tBuf.FreeTensor(this->localSet);
#endif
  }

  __aicore__ inline void ExecuteTask(TaskId task_id) {
    TaskDesc task_desc;
    getTaskDesc(this->runtimeConfigPtr, &(task_desc), task_id);
    if (task_desc.dependent_event != EVENT_INVALID_ID) {
      WaitForDependency(task_desc.dependent_event);
    }
    static_cast<Derived *>(this)->ExecuteComputeKernel(task_desc);
    if (task_desc.task_type != TASK_SHMEM_PUT_MEM_SINGAL) {
      TriggerEvent(task_desc.trigger_event);
    }
  }

  __aicore__ inline void WaitForDependency(uint32_t event_index) {
#ifdef __DAV_C220_CUBE__
    int32_t needed = all_event_num_triggers.GetValue(event_index);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      all_event_counters[event_index]);
    int32_t current = all_event_counters.GetValue(event_index);
    PipeBarrier<PIPE_ALL>();
    int64_t systemCycleBefore = AscendC::GetSystemCycle();
    do {
      if (current >= needed) {
        break;
      }
      int64_t systemCycleAfter = AscendC::GetSystemCycle();
      int64_t GetBlockNumCycle = systemCycleAfter - systemCycleBefore;
      int64_t CycleToTimeBase = 50;
      int64_t GetBlockNumTime = GetBlockNumCycle / CycleToTimeBase;
      if (GetBlockNumTime > 50) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
          all_event_counters[event_index]);
        current = all_event_counters.GetValue(event_index);
        systemCycleBefore = AscendC::GetSystemCycle();
      }
    } while (1);
#else
    int32_t needed = all_event_num_triggers.GetValue(event_index);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      all_event_counters[event_index]);
    int32_t current = all_event_counters.GetValue(event_index);
    PipeBarrier<PIPE_ALL>();
    int64_t systemCycleBefore = AscendC::GetSystemCycle();
    do {
      if (current >= needed) {
        break;
      }
      int64_t systemCycleAfter = AscendC::GetSystemCycle();
      int64_t GetBlockNumCycle = systemCycleAfter - systemCycleBefore;
      int64_t CycleToTimeBase = 50;
      int64_t GetBlockNumTime = GetBlockNumCycle / CycleToTimeBase;
      if (GetBlockNumTime > 150) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
          all_event_counters[event_index]);
        current = all_event_counters.GetValue(event_index);
        systemCycleBefore = AscendC::GetSystemCycle();
      }
    } while (1);
#endif
  }

  __aicore__ inline void TriggerEvent(uint32_t event_index) { AtomicAddForAllEventCounters(event_index); }

  uint32_t worker_id_ = 0;
  uint32_t task_num = 0;
  int32_t vector_task_num = 0;
  int32_t cube_task_num = 0;

  TPipe tpipe_;
#ifdef __DAV_C220_CUBE__
  TBuf<AscendC::TPosition::A1> tBuf;
#else
  TBuf<AscendC::TPosition::VECOUT> tBuf;
#endif
  LocalTensor<int32_t> localSet;

  GlobalTensor<int32_t> all_event_counters;
  GlobalTensor<int32_t> all_event_num_triggers;
  GlobalTensor<int32_t> atomic_add_values;
  GlobalTensor<int32_t> vector_task_indexs;
  GlobalTensor<int32_t> cube_task_indexs;

  __gm__ uint8_t *runtimeConfigPtr;
  GM_ADDR *input_list = nullptr;
  int64_t core_num = 0;
  int64_t vector_num = 0;
};

#endif  // MULTICORE_SCHEDULER_WORKER_KERNEL_H
