/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "aclnn_hyper_mega_gate_route.h"

#include <algorithm>
#include <cmath>

#include "aclnn_kernels/common/op_error_check.h"
#include "hyper_mega_gate_route.h"
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

bool IsBias(const aclTensor *tensor, int64_t expert_count) {
  return tensor != nullptr && tensor->GetViewShape().GetDimNum() == 1 &&
         tensor->GetViewShape().GetDim(0) == expert_count && IsFp32(tensor);
}

bool IsImageMask(const aclTensor *tensor, int64_t elements) {
  return tensor != nullptr && tensor->GetViewShape().GetDimNum() == 1 && tensor->GetViewShape().GetDim(0) == elements &&
         tensor->GetDataType() == op::DataType::DT_BOOL;
}

}  // namespace

extern "C" {

aclnnStatus aclnnHyperMegaGateRouteGetWorkspaceSize(const aclTensor *logits, const aclTensor *text_bias,
                                                    const aclTensor *vision_bias, const aclTensor *image_mask,
                                                    const aclTensor *runtime_config, const aclTensor *profile_buffer,
                                                    const aclTensor *routing_weights, const aclTensor *expert_indices,
                                                    const aclTensor *route_scores, const aclTensor *selected_scores,
                                                    const aclTensor *normalization_denominator, int64_t top_k,
                                                    double routed_scaling_factor, bool use_vision_bias,
                                                    uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnHyperMegaGateRoute,
                 DFX_IN(logits, text_bias, vision_bias, image_mask, runtime_config, profile_buffer, top_k,
                        routed_scaling_factor, use_vision_bias),
                 DFX_OUT(routing_weights, expert_indices, route_scores, selected_scores, normalization_denominator));
  CHECK_RET(logits != nullptr && logits->GetViewShape().GetDimNum() == 2, ACLNN_ERR_PARAM_INVALID);
  const int64_t tokens = logits->GetViewShape().GetDim(0);
  const int64_t expert_count = logits->GetViewShape().GetDim(1);
  CHECK_RET(tokens > 0 && expert_count > 0 && top_k > 0 && top_k <= expert_count &&
              std::isfinite(routed_scaling_factor) && IsFp32(logits) && IsBias(text_bias, expert_count) &&
              IsBias(vision_bias, expert_count) && IsImageMask(image_mask, use_vision_bias ? tokens : 1) &&
              IsRuntimeResource(runtime_config) && IsRuntimeResource(profile_buffer) &&
              HasShape(routing_weights, tokens, top_k) && IsFp32(routing_weights) &&
              HasShape(expert_indices, tokens, top_k) && expert_indices->GetDataType() == op::DataType::DT_INT64 &&
              HasShape(route_scores, tokens, expert_count) && IsFp32(route_scores) &&
              HasShape(selected_scores, tokens, top_k) && IsFp32(selected_scores) &&
              HasShape(normalization_denominator, tokens, 1) && IsFp32(normalization_denominator),
            ACLNN_ERR_PARAM_INVALID);

  auto unique_executor = CREATE_EXECUTOR();
  auto *executor_ptr = unique_executor.get();
  CHECK_RET(executor_ptr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  const auto outputs =
    l0op::HyperMegaGateRoute(logits, text_bias, vision_bias, image_mask, runtime_config, profile_buffer,
                             routing_weights, expert_indices, route_scores, selected_scores, normalization_denominator,
                             top_k, static_cast<float>(routed_scaling_factor), use_vision_bias, executor_ptr);
  CHECK_RET(std::all_of(outputs.begin(), outputs.end(), [](const aclTensor *output) { return output != nullptr; }),
            ACLNN_ERR_INNER_NULLPTR);
  *workspaceSize = unique_executor->GetWorkspaceSize();
  unique_executor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnHyperMegaGateRoute(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnHyperMegaGateRoute);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

}  // extern "C"
