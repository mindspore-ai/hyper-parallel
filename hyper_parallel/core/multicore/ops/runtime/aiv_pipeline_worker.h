/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef MULTICORE_SCHEDULER_AIV_PIPELINE_WORKER_H
#define MULTICORE_SCHEDULER_AIV_PIPELINE_WORKER_H

#include "cycle_trace_recorder.h"
#include "kernel_operator.h"
#include "runtime_config.hpp"

namespace MulticoreRuntime {

/**
 * @brief Broadcast one ordered AIV task sequence to every active worker.
 *
 * Derived must provide ExecuteComputeKernel(TaskType, uint32_t). The second
 * argument is the descriptor's tiling-data offset; row ownership comes from
 * worker_id and remains unchanged across the complete sequence.
 */
template <typename Derived>
class AivPipelineWorkerBase {
 public:
  __aicore__ inline void Init(uint32_t worker_id, __gm__ uint8_t *runtime_config, GM_ADDR profile_buffer) {
    worker_id_ = worker_id;
    runtime_config_ = runtime_config;
    profile_buffer_ = profile_buffer;
    task_capacity_ = getRuntimeTaskCapacity(runtime_config);
    stage_count_ = static_cast<uint32_t>(
      getTaskIndexNumByTaskType(runtime_config, TaskAiCoreType::TASK_AICORE_VECTOR));
    task_ids_ = reinterpret_cast<__gm__ int32_t *>(runtime_config + getVectorTaskIndexsOffset(runtime_config));
  }

  __aicore__ inline void Process() {
    if (isCycleProfileEnabled(runtime_config_)) {
      ProcessProfiled();
      return;
    }
    ProcessFast();
  }

 protected:
  __aicore__ inline void ProcessFast() {
    for (uint32_t stage = 0; stage < stage_count_; ++stage) {
      const uint32_t task_id = LoadTaskId(stage);
      ExecuteTask(task_id);
    }
  }

  __aicore__ inline void ProcessProfiled() {
    CycleTraceRecorder recorder;
    recorder.Init(worker_id_, profile_buffer_, getAicProfileRecordCapacity(runtime_config_),
                  getAivProfileRecordCapacity(runtime_config_));
    for (uint32_t stage = 0; stage < stage_count_; ++stage) {
      const uint32_t task_id = LoadTaskId(stage);
      const uint64_t start_cycle = recorder.Now();
      ExecuteTask(task_id);
      const uint64_t end_cycle = recorder.Now();
      recorder.Record(getTaskProfileDescId(runtime_config_, task_id), task_id, worker_id_, worker_id_, start_cycle,
                      end_cycle);
    }
  }

  __aicore__ inline uint32_t LoadTaskId(uint32_t stage) const {
    const int32_t task_id = task_ids_[stage];
    if (task_id < 0 || static_cast<uint32_t>(task_id) >= task_capacity_) {
      AscendC::Trap();
    }
    return static_cast<uint32_t>(task_id);
  }

  __aicore__ inline void ExecuteTask(uint32_t task_id) {
    const TaskType task_type = getTaskType(runtime_config_, task_id);
    const uint32_t tiling_offset = getTaskTilingDataOffset(runtime_config_, task_id);
    static_cast<Derived *>(this)->ExecuteComputeKernel(task_type, tiling_offset);
  }

  uint32_t worker_id_ = 0;
  uint32_t task_capacity_ = 0;
  uint32_t stage_count_ = 0;
  __gm__ uint8_t *runtime_config_ = nullptr;
  __gm__ int32_t *task_ids_ = nullptr;
  GM_ADDR profile_buffer_ = nullptr;
};

}  // namespace MulticoreRuntime

#endif  // MULTICORE_SCHEDULER_AIV_PIPELINE_WORKER_H
