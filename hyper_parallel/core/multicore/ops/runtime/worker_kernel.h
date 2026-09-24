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
 *   - static constexpr uint32_t PROFILE_IDX — index into input_list for the ordinary profile buffer
 *
 * Compute kernels (ExecuteMatmul, ExecuteShmemPutMem, etc.) are NOT part of this class;
 * they belong in each op's worker_kernel.cpp.
 */

#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_WORKER_KERNEL_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_WORKER_KERNEL_H_

#include "kernel_operator.h"
#include "shmem.h"
#include "runtime_config.hpp"
#include "cycle_trace_recorder.h"

using namespace AscendC;  // NOLINT(build/namespaces)

namespace MulticoreRuntime {

constexpr uint32_t VECTOR_WORKER_STRIDE = 2;
constexpr int64_t EVENT_REFRESH_TIME_UNIT_CYCLES = 50;
constexpr int64_t CUBE_EVENT_REFRESH_INTERVAL_UNITS = 50;     // 2,500 system cycles.
constexpr int64_t VECTOR_EVENT_REFRESH_INTERVAL_UNITS = 150;  // 7,500 system cycles.
// The selected SwiGLU tiling uses baseRowLen=19; smaller dynamic tails must lower it.
constexpr int64_t SWIGLU_DYNAMIC_TAIL_BASE_ROW_LIMIT = 19;
constexpr int64_t READY_SIGNAL_RADIX = 2;

enum class VectorWorkerPolicy : uint32_t {
  LEGACY_ODD = 0,
  ALL = 1,
};

enum class RuntimeStorageBoundsPolicy : uint32_t {
  NATIVE_TILING = 0,
  GRAPH_OWNED = 1,
};

template <typename Derived, VectorWorkerPolicy WorkerPolicy = VectorWorkerPolicy::LEGACY_ODD,
          RuntimeStorageBoundsPolicy StorageBoundsPolicy = RuntimeStorageBoundsPolicy::NATIVE_TILING>
class KernelWorkerBase {
 public:
  __aicore__ inline KernelWorkerBase() {}

  static constexpr uint32_t PROFILE_DESC_WAIT_DEPENDENCY = 0x10000;
  static constexpr uint32_t PROFILE_DESC_TRIGGER_EVENT = 0x10007;
  static constexpr uint32_t PROFILE_DESC_TASK_TYPE_BASE = 0x20000;

  __aicore__ inline void Init(uint32_t worker_id, __gm__ uint8_t *runtimeConfigPtr, GM_ADDR *input_list) {
    this->worker_id_ = worker_id;
    this->runtimeConfigPtr = runtimeConfigPtr;

    uint64_t runtime_bytes = (1ULL << 32) - 1;
    uint64_t event_bytes = getRuntimeEventCapacity(runtimeConfigPtr) * INT32_T_SIZE;
    uint32_t ep_size = 1;
    if constexpr (StorageBoundsPolicy == RuntimeStorageBoundsPolicy::NATIVE_TILING) {
      runtime_bytes = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 6);
      event_bytes = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 7);
      ep_size = static_cast<uint32_t>(getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1));
    }
    if (!isRuntimeStorageValid(runtimeConfigPtr, runtime_bytes, event_bytes, ep_size)) {
      AscendC::Trap();
    }
    this->runtime_task_capacity = getRuntimeTaskCapacity(runtimeConfigPtr);
    this->runtime_event_capacity = getRuntimeEventCapacity(runtimeConfigPtr);
    all_event_counters.SetGlobalBuffer((__gm__ int32_t *)(input_list[Derived::EVENT_IDX]), runtime_event_capacity);

    all_event_num_triggers.SetGlobalBuffer((__gm__ int32_t *)(runtimeConfigPtr + getAllEventNumTriggersOffset()),
                                           runtime_event_capacity);
    vector_task_indexs.SetGlobalBuffer(
      (__gm__ int32_t *)(runtimeConfigPtr + getVectorTaskIndexsOffset(runtimeConfigPtr)), runtime_task_capacity);
    cube_task_indexs.SetGlobalBuffer((__gm__ int32_t *)(runtimeConfigPtr + getCubeTaskIndexsOffset(runtimeConfigPtr)),
                                     runtime_task_capacity);
    atomic_add_values.SetGlobalBuffer((__gm__ int32_t *)(runtimeConfigPtr + getAtomicAddValuesOffset(runtimeConfigPtr)),
                                      ATOMIC_ADD_VALUE_LEN);

    this->input_list = input_list;
    this->task_num = getTaskNum(this->runtimeConfigPtr);
    this->vector_task_num = getTaskIndexNumByTaskType(this->runtimeConfigPtr, TaskAiCoreType::TASK_AICORE_VECTOR);
    this->cube_task_num = getTaskIndexNumByTaskType(this->runtimeConfigPtr, TaskAiCoreType::TASK_AICORE_CUBE);
    // RuntimeConfig is operator-independent; native tiling buffers are not.
    // Reading a fixed native-tiling offset here corrupts the round-robin stride
    // for operators whose sixth field is not the AIC count (for example mHC).
    this->vector_num = getNumWorkers(this->runtimeConfigPtr);
    this->core_num = this->vector_num / 2;
    this->active_vector_num = WorkerPolicy == VectorWorkerPolicy::ALL ? this->vector_num : this->vector_num / 2;
  }

  __aicore__ inline void Process() {
    ReadyHandshakeMeta meta;
    if (LoadReadyHandshakeMeta(&meta)) {
      WaitForDeviceReady(meta);
    }
    bool profile_enabled = isCycleProfileEnabled(this->runtimeConfigPtr);
    if (profile_enabled) {
      ProcessProfiled();
      return;
    }
    ProcessFast();
  }

 protected:
  __aicore__ inline void ProcessFast() {
#ifdef __DAV_C220_CUBE__
    uint32_t block_idx = this->worker_id_;
    do {
      if (block_idx >= this->cube_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTaskFast(task_index);
      block_idx = block_idx + this->core_num;
    } while (1);
#else
    uint32_t block_idx = GetVectorScheduleIndex();
    if (block_idx == INVALID_VECTOR_SCHEDULE_INDEX) {
      return;
    }
    do {
      if (block_idx >= this->vector_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTaskFast(task_index);
      block_idx = block_idx + this->active_vector_num;
    } while (1);
#endif
  }

  __aicore__ inline void ProcessProfiled() {
    CycleTraceRecorder cycle_trace_recorder;
    cycle_trace_recorder.Init(this->worker_id_, this->input_list[Derived::PROFILE_IDX],
                              getAicProfileRecordCapacity(this->runtimeConfigPtr),
                              getAivProfileRecordCapacity(this->runtimeConfigPtr));
#ifdef __DAV_C220_CUBE__
    uint32_t block_idx = this->worker_id_;
    do {
      if (block_idx >= this->cube_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTaskProfiled(task_index, cycle_trace_recorder);
      block_idx = block_idx + this->core_num;
    } while (1);
#else
    uint32_t block_idx = GetVectorScheduleIndex();
    if (block_idx == INVALID_VECTOR_SCHEDULE_INDEX) {
      return;
    }
    do {
      if (block_idx >= this->vector_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTaskProfiled(task_index, cycle_trace_recorder);
      block_idx = block_idx + this->active_vector_num;
    } while (1);
#endif
  }

  __aicore__ inline bool LoadReadyHandshakeMeta(ReadyHandshakeMeta *meta) {
    getReadyHandshakeMeta(this->runtimeConfigPtr, meta);
    return meta->ready_event != 0;
  }

  __aicore__ inline void PublishDeviceReady(const ReadyHandshakeMeta &meta) {
#ifndef __DAV_C220_CUBE__
    int64_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    int64_t rank = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 0) % ep;
    constexpr uint32_t ready_stride = DATA_CACHE_LINE_SIZE / INT32_T_SIZE;
    __gm__ int32_t *ready = (__gm__ int32_t *)(input_list[Derived::EVENT_IDX]) + runtime_event_capacity;

    GlobalTensor<int32_t> ready_state;
    ready_state.SetGlobalBuffer(ready, static_cast<uint32_t>((ep + 1) * ready_stride));
    uint32_t generation_index = static_cast<uint32_t>(ep) * ready_stride;
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      ready_state[generation_index]);
    int32_t generation = ready_state.GetValue(generation_index) + 1;
    ready_state.SetValue(generation_index, generation);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      ready_state[generation_index]);
    PipeBarrier<PIPE_ALL>();

    uint32_t round = 0;
    for (int64_t distance = 1; distance < ep; distance *= READY_SIGNAL_RADIX, ++round) {
      int64_t target = (rank + distance) % ep;
      __gm__ int32_t *round_ready = ready + round * ready_stride;
      aclshmemx_signal_op(round_ready, generation, ACLSHMEM_SIGNAL_SET, static_cast<int>(target));
      aclshmem_signal_wait_until(round_ready, ACLSHMEM_CMP_GE, generation);
    }
    TriggerEvent(meta.ready_event);
#endif
  }

  __aicore__ inline void WaitForDeviceReady(const ReadyHandshakeMeta &meta) {
#ifdef __DAV_C220_CUBE__
    WaitForDependency(meta.ready_event);
#else
    if (this->worker_id_ % VECTOR_WORKER_STRIDE == 0) {
      return;
    }
    if (this->worker_id_ == 1) {
      PublishDeviceReady(meta);
    }
    WaitForDependency(meta.ready_event);
#endif
  }

  __aicore__ inline TaskId GetTaskIndex(uint32_t task_id) {
#ifdef __DAV_C220_CUBE__
    return cube_task_indexs.GetValue(task_id);
#else
    return vector_task_indexs.GetValue(task_id);
#endif
  }

  __aicore__ inline uint32_t GetVectorScheduleIndex() {
    if constexpr (WorkerPolicy == VectorWorkerPolicy::ALL) {
      return this->worker_id_ < this->active_vector_num ? this->worker_id_ : INVALID_VECTOR_SCHEDULE_INDEX;
    }
    return this->worker_id_ % 2 == 0 ? INVALID_VECTOR_SCHEDULE_INDEX : this->worker_id_ / 2;
  }

  __aicore__ inline void AtomicAddForAllEventCounters(uint32_t event_index) {
    TPipe eventPipe;
#ifdef __DAV_C220_CUBE__
    TBuf<AscendC::TPosition::A1> eventBuffer;
#else
    TBuf<AscendC::TPosition::VECOUT> eventBuffer;
#endif
    eventPipe.InitBuffer(eventBuffer, DISPATCH_TOKEN_UB_SIZE);
    LocalTensor<int32_t> localSet = eventBuffer.GetWithOffset<int32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
#ifdef __DAV_C220_CUBE__
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopy(localSet, this->atomic_add_values, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    AscendC::SetAtomicAdd<int32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopy(this->all_event_counters[event_index], localSet, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    AscendC::SetAtomicNone();
#else
    localSet.SetValue(0, 1);
    for (int32_t i = 1; i < EXP_TOKEN_COUNT_FLAG_CNT; ++i) {
      localSet.SetValue(i, 0);
    }
    AscendC::SetAtomicAdd<int32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(this->all_event_counters[event_index], localSet, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    AscendC::SetAtomicNone();
#endif
    eventBuffer.FreeTensor(localSet);
    eventPipe.Destroy();
  }

  __aicore__ inline void ExecuteTaskFast(TaskId task_id) {
    if (task_id >= runtime_task_capacity) {
      AscendC::Trap();
    }
    TaskDesc task_desc;
    getTaskDesc(this->runtimeConfigPtr, &(task_desc), task_id);
    if ((task_desc.dependent_event != EVENT_INVALID_ID && task_desc.dependent_event >= runtime_event_capacity) ||
        static_cast<uint64_t>(task_desc.trigger_event) + ATOMIC_ADD_VALUE_LEN > runtime_event_capacity) {
      AscendC::Trap();
    }
    if (task_desc.dependent_event != EVENT_INVALID_ID) {
      WaitForDependency(task_desc.dependent_event, task_desc.extra_value_2);
    }
    static_cast<Derived *>(this)->ExecuteComputeKernel(task_desc);
    if (task_desc.task_type != TaskType::TASK_SHMEM_PUT_MEM_SIGNAL) {
      TriggerEvent(task_desc.trigger_event);
    }
  }

  __aicore__ inline void ExecuteTaskProfiled(TaskId task_id, CycleTraceRecorder &cycle_trace_recorder) {
    if (task_id >= runtime_task_capacity) {
      AscendC::Trap();
    }
    TaskDesc task_desc;
    getTaskDesc(this->runtimeConfigPtr, &(task_desc), task_id);
    if ((task_desc.dependent_event != EVENT_INVALID_ID && task_desc.dependent_event >= runtime_event_capacity) ||
        static_cast<uint64_t>(task_desc.trigger_event) + ATOMIC_ADD_VALUE_LEN > runtime_event_capacity) {
      AscendC::Trap();
    }
    uint32_t owner_id = getTaskProfileOwnerId(this->runtimeConfigPtr, task_id);
    if (task_desc.dependent_event != EVENT_INVALID_ID) {
      uint64_t wait_start_cycle = cycle_trace_recorder.Now();
      WaitForDependency(task_desc.dependent_event, task_desc.extra_value_2);
      uint64_t wait_end_cycle = cycle_trace_recorder.Now();
      cycle_trace_recorder.Record(Derived::PROFILE_DESC_WAIT_DEPENDENCY, task_id, task_desc.task_index, owner_id,
                                  wait_start_cycle, wait_end_cycle);
    }
    uint32_t profile_desc_id = getTaskProfileDescId(this->runtimeConfigPtr, task_id);
    if (profile_desc_id == PROFILE_DESC_INVALID_ID) {
      profile_desc_id = PROFILE_DESC_TASK_TYPE_BASE + static_cast<uint32_t>(task_desc.task_type);
    }
    uint64_t compute_start_cycle = cycle_trace_recorder.Now();
    static_cast<Derived *>(this)->ExecuteComputeKernel(task_desc);
    uint64_t compute_end_cycle = cycle_trace_recorder.Now();
    cycle_trace_recorder.Record(profile_desc_id, task_id, task_desc.task_index, owner_id, compute_start_cycle,
                                compute_end_cycle);
    if (task_desc.task_type != TaskType::TASK_SHMEM_PUT_MEM_SIGNAL) {
      uint64_t trigger_start_cycle = cycle_trace_recorder.Now();
      TriggerEvent(task_desc.trigger_event);
      uint64_t trigger_end_cycle = cycle_trace_recorder.Now();
      cycle_trace_recorder.Record(Derived::PROFILE_DESC_TRIGGER_EVENT, task_id, task_desc.task_index, owner_id,
                                  trigger_start_cycle, trigger_end_cycle);
    }
  }

  __aicore__ inline void WaitForDependency(uint32_t event_index, uint32_t poll_interval_us = 0) {
#ifdef __DAV_C220_CUBE__
    constexpr uint32_t default_poll_interval_us = CUBE_EVENT_REFRESH_INTERVAL_UNITS;
#else
    constexpr uint32_t default_poll_interval_us = VECTOR_EVENT_REFRESH_INTERVAL_UNITS;
#endif
    if (poll_interval_us == 0) {
      poll_interval_us = default_poll_interval_us;
    }
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
      int64_t CycleToTimeBase = EVENT_REFRESH_TIME_UNIT_CYCLES;
      int64_t GetBlockNumTime = GetBlockNumCycle / CycleToTimeBase;
      if (GetBlockNumTime > poll_interval_us) {
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
      int64_t CycleToTimeBase = EVENT_REFRESH_TIME_UNIT_CYCLES;
      int64_t GetBlockNumTime = GetBlockNumCycle / CycleToTimeBase;
      if (GetBlockNumTime > poll_interval_us) {
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
  uint32_t runtime_task_capacity = 0;
  uint32_t runtime_event_capacity = 0;
  int32_t vector_task_num = 0;
  int32_t cube_task_num = 0;

  GlobalTensor<int32_t> all_event_counters;
  GlobalTensor<int32_t> all_event_num_triggers;
  GlobalTensor<int32_t> atomic_add_values;
  GlobalTensor<int32_t> vector_task_indexs;
  GlobalTensor<int32_t> cube_task_indexs;

  __gm__ uint8_t *runtimeConfigPtr;
  GM_ADDR *input_list = nullptr;
  int64_t core_num = 0;
  int64_t vector_num = 0;
  uint32_t active_vector_num = 0;
  static constexpr uint32_t INVALID_VECTOR_SCHEDULE_INDEX = 0xFFFFFFFF;
};

}  // namespace MulticoreRuntime

#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_WORKER_KERNEL_H_
