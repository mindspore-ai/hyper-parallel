/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "aclnn_hyper_mega_gate_route_grad.h"

#include <cmath>

#include "aclnn_kernels/common/op_error_check.h"
#include "hyper_mega_gate_route_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

namespace {

bool HasShape(const aclTensor *tensor, int64_t rows, int64_t columns) {
  return tensor != nullptr && tensor->GetViewShape().GetDimNum() == 2 && tensor->GetViewShape().GetDim(0) == rows &&
         tensor->GetViewShape().GetDim(1) == columns;
}

bool IsFp32(const aclTensor *tensor) { return tensor != nullptr && tensor->GetDataType() == op::DataType::DT_FLOAT; }

bool IsRuntimeResource(const aclTensor *tensor) {
  return tensor != nullptr && tensor->GetViewShape().GetDimNum() == 1 &&
         tensor->GetDataType() == op::DataType::DT_UINT8;
}

}  // namespace

extern "C" {

aclnnStatus aclnnHyperMegaGateRouteGradGetWorkspaceSize(
  const aclTensor *logits, const aclTensor *route_scores, const aclTensor *selected_scores,
  const aclTensor *normalization_denominator, const aclTensor *expert_indices, const aclTensor *grad_routing_weights,
  const aclTensor *runtime_config, const aclTensor *profile_buffer, const aclTensor *grad_logits, int64_t top_k,
  double routed_scaling_factor, uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnHyperMegaGateRouteGrad,
                 DFX_IN(logits, route_scores, selected_scores, normalization_denominator, expert_indices,
                        grad_routing_weights, runtime_config, profile_buffer, top_k, routed_scaling_factor),
                 DFX_OUT(grad_logits));
  CHECK_RET(logits != nullptr && logits->GetViewShape().GetDimNum() == 2, ACLNN_ERR_PARAM_INVALID);
  const int64_t tokens = logits->GetViewShape().GetDim(0);
  const int64_t expert_count = logits->GetViewShape().GetDim(1);
  CHECK_RET(tokens > 0 && expert_count > 0 && top_k > 0 && top_k <= expert_count &&
              HasShape(route_scores, tokens, expert_count) && HasShape(selected_scores, tokens, top_k) &&
              HasShape(normalization_denominator, tokens, 1) && HasShape(expert_indices, tokens, top_k) &&
              HasShape(grad_routing_weights, tokens, top_k) && HasShape(grad_logits, tokens, expert_count) &&
              IsRuntimeResource(runtime_config) && IsRuntimeResource(profile_buffer) &&
              std::isfinite(routed_scaling_factor),
            ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(IsFp32(logits) && IsFp32(route_scores) && IsFp32(selected_scores) && IsFp32(normalization_denominator) &&
              expert_indices->GetDataType() == op::DataType::DT_INT64 && IsFp32(grad_routing_weights) &&
              IsFp32(grad_logits),
            ACLNN_ERR_PARAM_INVALID);
  auto unique_executor = CREATE_EXECUTOR();
  auto *executor_ptr = unique_executor.get();
  CHECK_RET(executor_ptr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  const auto *result = l0op::HyperMegaGateRouteGrad(
    logits, route_scores, selected_scores, normalization_denominator, expert_indices, grad_routing_weights,
    runtime_config, profile_buffer, top_k, static_cast<float>(routed_scaling_factor), grad_logits, executor_ptr);
  CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
  *workspaceSize = unique_executor->GetWorkspaceSize();
  unique_executor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnHyperMegaGateRouteGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnHyperMegaGateRouteGrad);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

}  // extern "C"
