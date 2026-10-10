/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "kernel_operator.h"
#include "route_grad_pipeline.h"
#include "route_grad_runtime.h"

using namespace AscendC;  // NOLINT(build/namespaces)

namespace {

constexpr uint32_t kGreaterOneStageCount = 11;
constexpr uint32_t kOneStageCount = 2;
constexpr MulticoreRuntime::TaskType kGreaterOneStages[kGreaterOneStageCount] = {
  MulticoreRuntime::TaskType::TASK_GATE_GRAD_MULS_SCALE,
  MulticoreRuntime::TaskType::TASK_GATE_GRAD_BROADCAST_DENOMINATOR,
  MulticoreRuntime::TaskType::TASK_GATE_GRAD_NEG,
  MulticoreRuntime::TaskType::TASK_GATE_GRAD_DIV_SELECTED,
  MulticoreRuntime::TaskType::TASK_GATE_GRAD_DIV_SELECTED_RATIO,
  MulticoreRuntime::TaskType::TASK_GATE_GRAD_MUL_CROSS,
  MulticoreRuntime::TaskType::TASK_GATE_GRAD_DIV_DIRECT,
  MulticoreRuntime::TaskType::TASK_GATE_GRAD_REDUCE_SUM,
  MulticoreRuntime::TaskType::TASK_GATE_GRAD_BROADCAST_ROW_SUM,
  MulticoreRuntime::TaskType::TASK_GATE_GRAD_ADD_SELECTED,
  MulticoreRuntime::TaskType::TASK_GATE_GRAD_ZEROS,
};
constexpr MulticoreRuntime::TaskType kOneStages[kOneStageCount] = {
  MulticoreRuntime::TaskType::TASK_GATE_GRAD_MULS_SCALE,
  MulticoreRuntime::TaskType::TASK_GATE_GRAD_ZEROS,
};

class RouteGradWorker : public MulticoreRuntime::AivPipelineWorkerBase<RouteGradWorker> {
 public:
  __aicore__ inline void Init(uint32_t worker_id, __gm__ uint8_t *runtime_config, GM_ADDR profile_buffer,
                              __gm__ float *selected_scores, __gm__ float *denominator, __gm__ float *grad_weights,
                              __gm__ float *selected_score_grad, __gm__ float *zero_score_grad, GM_ADDR workspace,
                              const HyperMegaGateRouteGradTilingData &tiling) {
    MulticoreRuntime::AivPipelineWorkerBase<RouteGradWorker>::Init(worker_id, runtime_config, profile_buffer);
    top_k_ = tiling.topK;
    pipeline_.Init(worker_id, selected_scores, denominator, grad_weights, selected_score_grad, zero_score_grad,
                   workspace, tiling);
  }

  __aicore__ inline void ExecuteComputeKernel(MulticoreRuntime::TaskType task_type, uint32_t stage_id) {
    const uint32_t stage_count = top_k_ == 1 ? kOneStageCount : kGreaterOneStageCount;
    if (stage_id >= stage_count) {
      Trap();
    }
    const MulticoreRuntime::TaskType expected = top_k_ == 1 ? kOneStages[stage_id] : kGreaterOneStages[stage_id];
    if (task_type != expected) {
      Trap();
    }
    switch (task_type) {
      case MulticoreRuntime::TaskType::TASK_GATE_GRAD_MULS_SCALE:
        pipeline_.MulsScaleGrad();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_GRAD_BROADCAST_DENOMINATOR:
        pipeline_.BroadcastDenominator();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_GRAD_NEG:
        pipeline_.NegScaledGrad();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_GRAD_DIV_SELECTED:
        pipeline_.DivSelected();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_GRAD_DIV_SELECTED_RATIO:
        pipeline_.DivSelectedRatio();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_GRAD_MUL_CROSS:
        pipeline_.MulCross();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_GRAD_DIV_DIRECT:
        pipeline_.DivDirect();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_GRAD_REDUCE_SUM:
        pipeline_.ReduceCrossTerm();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_GRAD_BROADCAST_ROW_SUM:
        pipeline_.BroadcastRowSum();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_GRAD_ADD_SELECTED:
        pipeline_.AddSelectedGrad();
        break;
      case MulticoreRuntime::TaskType::TASK_GATE_GRAD_ZEROS:
        pipeline_.ZerosLike();
        break;
      default:
        Trap();
    }
  }

 private:
  HyperMegaGate::RouteGradPipeline pipeline_;
  uint32_t top_k_ = 0;
};

}  // namespace

/**
 * Execute the Route normalization gradient for one fixed token-row shard.
 *
 * Every active AIV block owns disjoint rows. Stage outputs are complete in GM
 * before the block advances to the next RuntimeConfig descriptor.
 */
extern "C" __global__ __aicore__ void hyper_mega_gate_route_grad(
  GM_ADDR selected_scores, GM_ADDR normalization_denominator, GM_ADDR grad_routing_weights, GM_ADDR route_scores,
  GM_ADDR expert_indices, GM_ADDR runtime_config, GM_ADDR profile_buffer, GM_ADDR selected_score_grad,
  GM_ADDR zero_score_grad, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(tiling_data, tiling);
  const uint32_t worker_id = GetBlockIdx();
  auto *runtime = reinterpret_cast<__gm__ uint8_t *>(runtime_config);
  const uint32_t expected_stages = tiling_data.topK == 1 ? kOneStageCount : kGreaterOneStageCount;
  if (worker_id >= tiling_data.activeAivWorkerCount || tiling_data.stageCount != expected_stages ||
      !IsRouteGradRuntimeValid(runtime, tiling_data)) {
    Trap();
  }
  GM_ADDR user_workspace = GetUserWorkspace(workspace);
  if (user_workspace == nullptr) {
    Trap();
  }
  (void)route_scores;
  (void)expert_indices;
  RouteGradWorker worker;
  worker.Init(worker_id, runtime, profile_buffer, reinterpret_cast<__gm__ float *>(selected_scores),
              reinterpret_cast<__gm__ float *>(normalization_denominator),
              reinterpret_cast<__gm__ float *>(grad_routing_weights),
              reinterpret_cast<__gm__ float *>(selected_score_grad), reinterpret_cast<__gm__ float *>(zero_score_grad),
              user_workspace, tiling_data);
  worker.Process();
}
