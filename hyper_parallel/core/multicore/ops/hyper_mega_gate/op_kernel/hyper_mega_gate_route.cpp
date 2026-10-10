/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "kernel_operator.h"
#include "route_pipeline.h"
#include "runtime/aiv_pipeline_worker.h"

using namespace AscendC;  // NOLINT(build/namespaces)

namespace {

constexpr uint32_t kRouteStageCount = 10;

class RoutePipelineWorker : public MulticoreRuntime::AivPipelineWorkerBase<RoutePipelineWorker> {
 public:
  __aicore__ inline void Init(uint32_t worker_id, __gm__ uint8_t *runtime_config, GM_ADDR profile_buffer,
                              __gm__ float *logits, __gm__ float *text_bias, __gm__ float *vision_bias,
                              __gm__ bool *image_mask, __gm__ float *routing_weights, __gm__ int64_t *expert_indices,
                              __gm__ float *route_scores, __gm__ float *selected_scores,
                              __gm__ float *normalization_denominator, GM_ADDR user_workspace,
                              const HyperMegaGateRouteTilingData &tiling_data, uint64_t first_row, uint32_t row_count) {
    MulticoreRuntime::AivPipelineWorkerBase<RoutePipelineWorker>::Init(worker_id, runtime_config, profile_buffer);
    route_.Init(logits, text_bias, vision_bias, image_mask, tiling_data.useVisionBias != 0, routing_weights,
                expert_indices, route_scores, selected_scores, normalization_denominator, user_workspace,
                tiling_data.expertCount, tiling_data.topK, tiling_data.routedScalingFactor, tiling_data.batchRows,
                tiling_data.expertAlign, tiling_data.kAlign, tiling_data.sharedTmpBytes, tiling_data.scoreAOffset,
                tiling_data.topkValuesOffset, tiling_data.indicesI32Offset, tiling_data.rowSumOffset,
                &tiling_data.topkTiling, first_row, row_count);
  }

  __aicore__ inline void ExecuteComputeKernel(MulticoreRuntime::TaskType task_type, uint32_t tiling_offset) {
    const uint32_t stage =
      static_cast<uint32_t>(task_type) - static_cast<uint32_t>(MulticoreRuntime::TaskType::TASK_GATE_SOFTPLUS);
    if (stage != tiling_offset || stage >= kRouteStageCount) {
      Trap();
    }
    switch (task_type) {
      case MulticoreRuntime::TaskType::TASK_GATE_SOFTPLUS:
        route_.Softplus();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_SQRT:
        route_.SqrtScore();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_ADD_BIAS:
        route_.AddBias();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_TOPK:
        route_.TopKScore();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_GATHER:
        route_.GatherScore();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_REDUCE_SUM:
        route_.ReduceSelected();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_ADD_EPSILON:
        route_.AddEpsilon();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_DIV:
        route_.DivideSelected();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_MUL_SCALE:
        route_.ScaleSelected();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_CAST_INDEX:
        route_.CastIndices();
        break;
      default:
        Trap();
    }
  }

 private:
  HyperMegaGate::RoutePipeline route_;
};

}  // namespace

extern "C" __global__ __aicore__ void hyper_mega_gate_route(GM_ADDR logits, GM_ADDR text_bias, GM_ADDR vision_bias,
                                                            GM_ADDR image_mask, GM_ADDR runtime_config,
                                                            GM_ADDR profile_buffer, GM_ADDR routing_weights,
                                                            GM_ADDR expert_indices, GM_ADDR route_scores,
                                                            GM_ADDR selected_scores, GM_ADDR normalization_denominator,
                                                            GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(tiling_data, tiling);
  const uint32_t worker_id = GetBlockIdx();
  auto *runtime = reinterpret_cast<__gm__ uint8_t *>(runtime_config);
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
  if (tiling_data.activeAivWorkerCount == 0 || worker_id >= tiling_data.activeAivWorkerCount ||
      tiling_data.activeAivWorkerCount > tiling_data.aivWorkerSlotCapacity ||
      tiling_data.activeAivWorkerCount > MulticoreRuntime::getRuntimeWorkerCount(runtime) ||
      tiling_data.stageCount != kRouteStageCount || MulticoreRuntime::getTaskNum(runtime) != kRouteStageCount ||
      MulticoreRuntime::getTaskIndexNumByTaskType(runtime, MulticoreRuntime::TaskAiCoreType::TASK_AICORE_VECTOR) !=
        static_cast<int32_t>(kRouteStageCount) ||
      required_runtime_bytes > tiling_data.runtimeConfigBytes ||
      (MulticoreRuntime::isCycleProfileEnabled(runtime) && required_profile_bytes > tiling_data.profileBufferBytes)) {
    Trap();
  }

  const uint64_t first_row = static_cast<uint64_t>(worker_id) * tiling_data.rowsPerWorker;
  const uint64_t remaining = first_row < static_cast<uint64_t>(tiling_data.tokenCount)
                               ? static_cast<uint64_t>(tiling_data.tokenCount) - first_row
                               : 0;
  const uint32_t row_count =
    static_cast<uint32_t>(remaining < tiling_data.rowsPerWorker ? remaining : tiling_data.rowsPerWorker);
  GM_ADDR user_workspace = GetUserWorkspace(workspace);
  if (user_workspace == nullptr) {
    Trap();
  }

  RoutePipelineWorker worker;
  worker.Init(worker_id, runtime, profile_buffer, reinterpret_cast<__gm__ float *>(logits),
              reinterpret_cast<__gm__ float *>(text_bias), reinterpret_cast<__gm__ float *>(vision_bias),
              reinterpret_cast<__gm__ bool *>(image_mask), reinterpret_cast<__gm__ float *>(routing_weights),
              reinterpret_cast<__gm__ int64_t *>(expert_indices), reinterpret_cast<__gm__ float *>(route_scores),
              reinterpret_cast<__gm__ float *>(selected_scores),
              reinterpret_cast<__gm__ float *>(normalization_denominator), user_workspace, tiling_data, first_row,
              row_count);
  worker.Process();
}
