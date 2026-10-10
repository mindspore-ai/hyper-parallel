/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "hyper_mega_gate_route.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

namespace {

const aclTensor *EnsureKernelMatrixMetadata(const aclTensor *tensor, aclOpExecutor *executor) {
  CHECK_RET(tensor != nullptr && executor != nullptr, nullptr);
  const auto &view_shape = tensor->GetViewShape();
  CHECK_RET(view_shape.GetDimNum() == 2, nullptr);
  const auto &storage_shape = tensor->GetStorageShape();
  if (storage_shape == view_shape && tensor->GetOriginalShape() == view_shape) {
    return tensor;
  }
  CHECK_RET(storage_shape.GetShapeSize() >= view_shape.GetShapeSize(), nullptr);
  auto *matrix_view = executor->CreateView(tensor, view_shape, tensor->GetViewOffset());
  CHECK_RET(matrix_view != nullptr, nullptr);
  matrix_view->SetStorageShape(view_shape);
  matrix_view->SetOriginalShape(view_shape);
  return matrix_view;
}

const aclTensor *EnsureKernelVectorMetadata(const aclTensor *tensor, aclOpExecutor *executor) {
  CHECK_RET(tensor != nullptr && executor != nullptr, nullptr);
  const auto &view_shape = tensor->GetViewShape();
  CHECK_RET(view_shape.GetDimNum() == 1, nullptr);
  const auto &storage_shape = tensor->GetStorageShape();
  if (storage_shape == view_shape && tensor->GetOriginalShape() == view_shape) {
    return tensor;
  }
  CHECK_RET(storage_shape.GetShapeSize() >= view_shape.GetShapeSize(), nullptr);
  auto *vector_view = executor->CreateView(tensor, view_shape, tensor->GetViewOffset());
  CHECK_RET(vector_view != nullptr, nullptr);
  vector_view->SetStorageShape(view_shape);
  vector_view->SetOriginalShape(view_shape);
  return vector_view;
}

}  // namespace

namespace l0op {

OP_TYPE_REGISTER(HyperMegaGateRoute);

const std::array<const aclTensor *, 5> HyperMegaGateRoute(
  const aclTensor *logits, const aclTensor *text_bias, const aclTensor *vision_bias, const aclTensor *image_mask,
  const aclTensor *runtime_config, const aclTensor *profile_buffer, const aclTensor *routing_weights,
  const aclTensor *expert_indices, const aclTensor *route_scores, const aclTensor *selected_scores,
  const aclTensor *normalization_denominator, int64_t top_k, float routed_scaling_factor, bool use_vision_bias,
  aclOpExecutor *executor) {
  L0_DFX(HyperMegaGateRoute, logits, text_bias, vision_bias, image_mask, runtime_config, profile_buffer,
         routing_weights, expert_indices, route_scores, selected_scores, normalization_denominator, top_k,
         routed_scaling_factor, use_vision_bias);
  auto *weights_out = const_cast<aclTensor *>(routing_weights);
  auto *indices_out = const_cast<aclTensor *>(expert_indices);
  auto *route_scores_out = const_cast<aclTensor *>(route_scores);
  auto *selected_scores_out = const_cast<aclTensor *>(selected_scores);
  auto *denominator_out = const_cast<aclTensor *>(normalization_denominator);
  const auto *kernel_logits = EnsureKernelMatrixMetadata(logits, executor);
  const auto *kernel_image_mask = EnsureKernelVectorMetadata(image_mask, executor);
  if (kernel_logits == nullptr || kernel_image_mask == nullptr) {
    return {nullptr, nullptr, nullptr, nullptr, nullptr};
  }
  const auto status = ADD_TO_LAUNCHER_LIST_AICORE(
    HyperMegaGateRoute,
    OP_INPUT(kernel_logits, text_bias, vision_bias, kernel_image_mask, runtime_config, profile_buffer),
    OP_OUTPUT(weights_out, indices_out, route_scores_out, selected_scores_out, denominator_out),
    OP_ATTR(top_k, routed_scaling_factor, use_vision_bias));
  if (status != ACLNN_SUCCESS) {
    return {nullptr, nullptr, nullptr, nullptr, nullptr};
  }
  return {weights_out, indices_out, route_scores_out, selected_scores_out, denominator_out};
}

}  // namespace l0op
