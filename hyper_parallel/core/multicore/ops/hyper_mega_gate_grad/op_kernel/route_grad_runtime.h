/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef HYPER_MEGA_GATE_ROUTE_GRAD_RUNTIME_H
#define HYPER_MEGA_GATE_ROUTE_GRAD_RUNTIME_H

#include "runtime/aiv_pipeline_worker.h"

template <typename TilingData>
__aicore__ inline bool IsRouteGradRuntimeValid(__gm__ uint8_t *runtime, const TilingData &tiling) {
  const uint32_t task_capacity = MulticoreRuntime::getRuntimeTaskCapacity(runtime);
  const uint64_t required_runtime_bytes = static_cast<uint64_t>(MulticoreRuntime::getVectorTaskIndexsOffset(runtime)) +
                                          static_cast<uint64_t>(task_capacity) * sizeof(int32_t);
  const uint64_t required_profile_bytes =
    static_cast<uint64_t>(MulticoreRuntime::NUM_WORKERS_CUBE) *
      (sizeof(MulticoreRuntime::CycleTraceCoreHeader) +
       MulticoreRuntime::getAicProfileRecordCapacity(runtime) * sizeof(MulticoreRuntime::CycleTraceRecord)) +
    static_cast<uint64_t>(MulticoreRuntime::NUM_WORKERS_VECTOR) *
      (sizeof(MulticoreRuntime::CycleTraceCoreHeader) +
       MulticoreRuntime::getAivProfileRecordCapacity(runtime) * sizeof(MulticoreRuntime::CycleTraceRecord));
  return tiling.activeAivWorkerCount > 0 && tiling.activeAivWorkerCount <= tiling.aivWorkerSlotCapacity &&
         tiling.activeAivWorkerCount <= MulticoreRuntime::getRuntimeWorkerCount(runtime) && tiling.stageCount > 0 &&
         MulticoreRuntime::getTaskNum(runtime) == tiling.stageCount &&
         MulticoreRuntime::getTaskIndexNumByTaskType(runtime, MulticoreRuntime::TaskAiCoreType::TASK_AICORE_VECTOR) ==
           static_cast<int32_t>(tiling.stageCount) &&
         required_runtime_bytes <= tiling.runtimeConfigBytes &&
         (!MulticoreRuntime::isCycleProfileEnabled(runtime) || required_profile_bytes <= tiling.profileBufferBytes);
}

#endif  // HYPER_MEGA_GATE_ROUTE_GRAD_RUNTIME_H
