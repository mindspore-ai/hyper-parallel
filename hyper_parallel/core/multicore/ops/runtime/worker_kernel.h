/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
#include "runtime_config.hpp"
#include "cycle_trace_recorder.h"

#include "get_mem.h"

using namespace AscendC;  // NOLINT(build/namespaces)

template <typename Derived>
class KernelWorkerBase {
 public:
  __aicore__ inline KernelWorkerBase() {}

  static constexpr uint32_t PROFILE_DESC_WAIT_DEPENDENCY = 0x10000;
  static constexpr uint32_t PROFILE_DESC_TRIGGER_EVENT = 0x10007;
  static constexpr uint32_t PROFILE_DESC_TASK_TYPE_BASE = 0x20000;

  __aicore__ inline void Init(uint32_t worker_id, __gm__ uint8_t *runtimeConfigPtr, GM_ADDR *input_list) {
    this->worker_id_ = worker_id;
    this->runtimeConfigPtr = runtimeConfigPtr;

    uint64_t runtime_bytes = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 6);
    uint64_t event_bytes = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 7);
    uint32_t ep_size = static_cast<uint32_t>(getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1));
    uint64_t local_experts = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 2) / ep_size;
    if (!isRuntimeStorageValid(runtimeConfigPtr, runtime_bytes, event_bytes, ep_size, local_experts)) {
      AscendC::Trap();
    }
    this->runtime_task_capacity = getRuntimeTaskCapacity(runtimeConfigPtr);
    this->runtime_event_capacity = getRuntimeEventCapacity(runtimeConfigPtr);
#ifdef __DAV_C220_CUBE__
    // The worker's scratch region is fixed for the lifetime of this task graph.
    this->grouped_matmul_group_list_offset_ = getGroupedMatmulGroupListOffsetById(runtimeConfigPtr, worker_id);
#endif
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
    this->core_num = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 5);
    this->vector_num = this->core_num * 2;
  }

  __aicore__ inline void Process() {
    ReadyHandshakeMeta meta;
    bool has_ready = LoadReadyHandshakeMeta(&meta);
    pull_protocol_ = meta.completion_event != 0;
    if (has_ready) {
      WaitForDeviceReady(meta);
    }
    bool profile_enabled = isCycleProfileEnabled(this->runtimeConfigPtr);
    if (profile_enabled) {
      ProcessProfiled();
    } else {
      ProcessFast();
    }
    if (meta.completion_event != 0) {
      CompleteDeviceReads(meta);
    }
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
    if (this->worker_id_ % 2 == 0) {
      return;
    }
    uint32_t block_idx = this->worker_id_ / 2;
    do {
      if (block_idx >= this->vector_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTaskFast(task_index);
      uint32_t half_num = this->vector_num / 2;
      block_idx = block_idx + half_num;
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
    if (this->worker_id_ % 2 == 0) {
      return;
    }
    uint32_t block_idx = this->worker_id_ / 2;
    do {
      if (block_idx >= this->vector_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTaskProfiled(task_index, cycle_trace_recorder);
      uint32_t half_num = this->vector_num / 2;
      block_idx = block_idx + half_num;
    } while (1);
#endif
  }

  __aicore__ inline bool LoadReadyHandshakeMeta(ReadyHandshakeMeta *meta) {
    getReadyHandshakeMeta(this->runtimeConfigPtr, meta);
    return meta->ready_event != 0;
  }

  __aicore__ inline void PublishDeviceReady(const ReadyHandshakeMeta &meta, bool completion = false) {
#ifndef __DAV_C220_CUBE__
    int64_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    int64_t rank = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 0) % ep;
    constexpr uint32_t ready_stride = DATA_CACHE_LINE_SIZE / INT32_T_SIZE;
    __gm__ int32_t *ready = (__gm__ int32_t *)(input_list[Derived::EVENT_IDX]) + runtime_event_capacity;
    if (completion) {
      ready += (ep + 1) * ready_stride;
    }

    GlobalTensor<int32_t> ready_state;
    ready_state.SetGlobalBuffer(ready, static_cast<uint32_t>((ep + 1) * ready_stride));
    uint32_t generation_index = static_cast<uint32_t>(ep) * ready_stride;
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      ready_state[generation_index]);
    int32_t previous_generation = ready_state.GetValue(generation_index);
    if (previous_generation == INT32_MAX) {
      AscendC::Trap();
    }
    int32_t generation = previous_generation + 1;
    ready_state.SetValue(generation_index, generation);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      ready_state[generation_index]);
    PipeBarrier<PIPE_ALL>();

    uint32_t round = 0;
    for (int64_t distance = 1; distance < ep; distance *= 2, ++round) {
      int64_t target = (rank + distance) % ep;
      __gm__ int32_t *round_ready = ready + round * ready_stride;
      aclshmemx_signal_op(round_ready, generation, ACLSHMEM_SIGNAL_SET, static_cast<int>(target));
      aclshmem_signal_wait_until(round_ready, ACLSHMEM_CMP_GE, generation);
    }
    if (!completion) {
      TriggerEvent(meta.ready_event);
    }
#endif
  }

  __aicore__ inline void CompleteDeviceReads(const ReadyHandshakeMeta &meta) {
#ifndef __DAV_C220_CUBE__
    if (this->worker_id_ % 2 == 0) {
      return;
    }
#endif
    TriggerEvent(meta.completion_event);
#ifndef __DAV_C220_CUBE__
    if (this->worker_id_ == 1) {
      WaitForDependency(meta.completion_event);
      if (meta.ready_event != 0) {
        PublishDeviceReady(meta, true);
      }
    }
#endif
  }

  template <typename T>
  __aicore__ inline void ExecuteShmemGetMem(const TaskDesc &task) {
    int64_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    int64_t experts = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 2);
    int64_t hidden = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 3);
    int64_t tiles = task.task_split_num / experts;
    int64_t source_pe = task.task_index / ((experts / ep) * tiles);
    int64_t size =
      reinterpret_cast<__gm__ int32_t *>(input_list[task.inputs[3].input_position])[task.inputs[3].base_ptr_offset];
    int64_t start = (task.task_index % tiles) * static_cast<int64_t>(task.task_split_value) * hidden;
    if (start >= size) {
      return;
    }
    int64_t tile_elements = static_cast<int64_t>(task.task_split_value) * hidden;
    int64_t elements = size - start < tile_elements ? size - start : tile_elements;
    int64_t destination_offset =
      reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[0].input_position])[task.inputs[0].base_ptr_offset];
    int64_t source_offset =
      reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[2].input_position])[task.inputs[2].base_ptr_offset];
    PullToLocal<T>(input_list[task.outputs[0].input_position] + (destination_offset + start) * sizeof(T),
                   input_list[task.inputs[1].input_position] + (source_offset + start) * sizeof(T), elements,
                   source_pe);
  }

  __aicore__ inline void WaitForDeviceReady(const ReadyHandshakeMeta &meta) {
#ifdef __DAV_C220_CUBE__
    WaitForDependency(meta.ready_event);
#else
    if (this->worker_id_ % 2 == 0) {
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
      WaitForDependency(task_desc.dependent_event);
    }
    static_cast<Derived *>(this)->ExecuteComputeKernel(task_desc);
    if (task_desc.task_type != TASK_SHMEM_PUT_MEM_SIGNAL) {
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
      WaitForDependency(task_desc.dependent_event);
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
    if (task_desc.task_type != TASK_SHMEM_PUT_MEM_SIGNAL) {
      uint64_t trigger_start_cycle = cycle_trace_recorder.Now();
      TriggerEvent(task_desc.trigger_event);
      uint64_t trigger_end_cycle = cycle_trace_recorder.Now();
      cycle_trace_recorder.Record(Derived::PROFILE_DESC_TRIGGER_EVENT, task_id, task_desc.task_index, owner_id,
                                  trigger_start_cycle, trigger_end_cycle);
    }
  }

  __aicore__ inline void WaitForDependency(uint32_t event_index) {
    // Pull producers signal locally after MTE3 completion. Observe short-lived
    // GMM/SwiGLU dependencies promptly while retaining a bounded refresh rate.
#ifdef __DAV_C220_CUBE__
    int64_t poll_interval_us = pull_protocol_ ? 10 : 50;
#else
    int64_t poll_interval_us = pull_protocol_ ? 30 : 150;
#endif
    constexpr int64_t cycles_per_us = 50;
    int32_t needed = all_event_num_triggers.GetValue(event_index);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      all_event_counters[event_index]);
    int32_t current = all_event_counters.GetValue(event_index);
    PipeBarrier<PIPE_ALL>();
    int64_t previous_cycle = AscendC::GetSystemCycle();
    while (current < needed) {
      int64_t elapsed_cycles = AscendC::GetSystemCycle() - previous_cycle;
      if (elapsed_cycles / cycles_per_us > poll_interval_us) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
          all_event_counters[event_index]);
        current = all_event_counters.GetValue(event_index);
        previous_cycle = AscendC::GetSystemCycle();
      }
    }
  }

  __aicore__ inline void TriggerEvent(uint32_t event_index) { AtomicAddForAllEventCounters(event_index); }

  bool pull_protocol_ = false;
  uint32_t worker_id_ = 0;
  uint32_t grouped_matmul_group_list_offset_ = 0;
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
};

#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_WORKER_KERNEL_H_
