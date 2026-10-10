/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "hyper_mega_gate_route_grad.h"

#include "hyper_mega_gate_route_grad_post.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

namespace {

const aclTensor *EnsureKernelMatrixMetadata(const aclTensor *tensor, int64_t rows, int64_t columns,
                                            aclOpExecutor *executor) {
  CHECK_RET(tensor != nullptr && executor != nullptr && rows > 0 && columns > 0, nullptr);
  const auto &view_shape = tensor->GetViewShape();
  CHECK_RET(view_shape.GetDimNum() == 2 && view_shape.GetDim(0) == rows && view_shape.GetDim(1) == columns, nullptr);

  const auto &storage_shape = tensor->GetStorageShape();
  if (storage_shape.GetDimNum() == 2 && storage_shape.GetDim(0) == rows && storage_shape.GetDim(1) == columns) {
    return tensor;
  }

  // Nested ACLNN launch can preserve the logical view while flattening the
  // storage/origin metadata. Restore only that metadata; CreateView does not
  // copy data or schedule another kernel.
  CHECK_RET(storage_shape.GetShapeSize() >= view_shape.GetShapeSize(), nullptr);
  auto *matrix_view = executor->CreateView(tensor, view_shape, tensor->GetViewOffset());
  CHECK_RET(matrix_view != nullptr, nullptr);
  matrix_view->SetStorageShape(view_shape);
  matrix_view->SetOriginalShape(view_shape);
  return matrix_view;
}

}  // namespace

namespace l0op {

OP_TYPE_REGISTER(HyperMegaGateRouteGrad);

RouteGradOutputs HyperMegaGateRouteGradKernel(const aclTensor *selected_scores,
                                              const aclTensor *normalization_denominator,
                                              const aclTensor *grad_routing_weights, const aclTensor *route_scores,
                                              const aclTensor *expert_indices, const aclTensor *runtime_config,
                                              const aclTensor *profile_buffer, int64_t top_k,
                                              float routed_scaling_factor, aclOpExecutor *executor) {
  L0_DFX(HyperMegaGateRouteGradKernel, selected_scores, normalization_denominator, grad_routing_weights, route_scores,
         expert_indices, runtime_config, profile_buffer, top_k, routed_scaling_factor);
  RouteGradOutputs outputs{nullptr, nullptr};
  CHECK_RET(selected_scores != nullptr && normalization_denominator != nullptr && grad_routing_weights != nullptr &&
              route_scores != nullptr && expert_indices != nullptr && runtime_config != nullptr &&
              profile_buffer != nullptr && executor != nullptr && top_k > 0,
            outputs);
  const auto &selected_shape = selected_scores->GetViewShape();
  const auto &score_shape = route_scores->GetViewShape();
  CHECK_RET(selected_shape.GetDimNum() == 2 && score_shape.GetDimNum() == 2, outputs);
  const int64_t token_count = selected_shape.GetDim(0);
  const int64_t expert_count = score_shape.GetDim(1);
  CHECK_RET(
    token_count > 0 && expert_count > 0 && selected_shape.GetDim(1) == top_k && score_shape.GetDim(0) == token_count,
    outputs);

  const auto *selected_scores_matrix = EnsureKernelMatrixMetadata(selected_scores, token_count, top_k, executor);
  const auto *denominator_matrix = EnsureKernelMatrixMetadata(normalization_denominator, token_count, 1, executor);
  const auto *grad_weights_matrix = EnsureKernelMatrixMetadata(grad_routing_weights, token_count, top_k, executor);
  const auto *route_scores_matrix = EnsureKernelMatrixMetadata(route_scores, token_count, expert_count, executor);
  const auto *expert_indices_matrix = EnsureKernelMatrixMetadata(expert_indices, token_count, top_k, executor);
  CHECK_RET(selected_scores_matrix != nullptr && denominator_matrix != nullptr && grad_weights_matrix != nullptr &&
              route_scores_matrix != nullptr && expert_indices_matrix != nullptr,
            outputs);

  auto *selected_score_grad = executor->AllocTensor(selected_shape, op::DataType::DT_FLOAT);
  auto *zero_score_grad = executor->AllocTensor(score_shape, op::DataType::DT_FLOAT);
  CHECK_RET(selected_score_grad != nullptr && zero_score_grad != nullptr, outputs);
  selected_score_grad->SetStorageShape(selected_shape);
  selected_score_grad->SetOriginalShape(selected_shape);
  zero_score_grad->SetStorageShape(score_shape);
  zero_score_grad->SetOriginalShape(score_shape);
  const auto status =
    ADD_TO_LAUNCHER_LIST_AICORE(HyperMegaGateRouteGrad,
                                OP_INPUT(selected_scores_matrix, denominator_matrix, grad_weights_matrix,
                                         route_scores_matrix, expert_indices_matrix, runtime_config, profile_buffer),
                                OP_OUTPUT(selected_score_grad, zero_score_grad), OP_ATTR(top_k, routed_scaling_factor));
  CHECK_RET(status == ACLNN_SUCCESS, outputs);
  outputs.selected_score_grad = selected_score_grad;
  outputs.zero_score_grad = zero_score_grad;
  return outputs;
}

const aclTensor *HyperMegaGateRouteGrad(const aclTensor *logits, const aclTensor *route_scores,
                                        const aclTensor *selected_scores, const aclTensor *normalization_denominator,
                                        const aclTensor *expert_indices, const aclTensor *grad_routing_weights,
                                        const aclTensor *runtime_config, const aclTensor *profile_buffer, int64_t top_k,
                                        float routed_scaling_factor, const aclTensor *output, aclOpExecutor *executor) {
  L0_DFX(HyperMegaGateRouteGrad, logits, route_scores, selected_scores, normalization_denominator, expert_indices,
         grad_routing_weights, top_k, routed_scaling_factor);
  CHECK_RET(logits != nullptr && route_scores != nullptr && selected_scores != nullptr &&
              normalization_denominator != nullptr && expert_indices != nullptr && grad_routing_weights != nullptr &&
              executor != nullptr,
            nullptr);

  const auto route_grad = HyperMegaGateRouteGradKernel(selected_scores, normalization_denominator, grad_routing_weights,
                                                       route_scores, expert_indices, runtime_config, profile_buffer,
                                                       top_k, routed_scaling_factor, executor);
  CHECK_RET(route_grad.selected_score_grad != nullptr && route_grad.zero_score_grad != nullptr, nullptr);
  const auto *linear_indices = HyperMegaGateLinearIndex(expert_indices, route_grad.zero_score_grad, executor);
  CHECK_RET(linear_indices != nullptr, nullptr);
  const auto *score_grad = HyperMegaGateScatterSelectedGrad(route_grad.zero_score_grad, linear_indices,
                                                            route_grad.selected_score_grad, executor);
  CHECK_RET(score_grad != nullptr, nullptr);
  const auto *doubled_scores = HyperMegaGateDoubleRouteScores(route_scores, executor);
  CHECK_RET(doubled_scores != nullptr, nullptr);
  const auto *sqrt_input_grad = HyperMegaGateSqrtInputGrad(score_grad, doubled_scores, executor);
  CHECK_RET(sqrt_input_grad != nullptr, nullptr);
  return HyperMegaGateSoftplusV2Grad(sqrt_input_grad, logits, output, executor);
}

}  // namespace l0op
