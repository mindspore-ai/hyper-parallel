/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "hyper_mega_gate_route_grad_post.h"

#include <string>

#include "linear_index/linear_index.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

namespace l0op {

OP_TYPE_REGISTER(ScatterElementsV2);
OP_TYPE_REGISTER(SoftplusV2Grad);
OP_TYPE_REGISTER(Muls);
OP_TYPE_REGISTER(RealDiv);

namespace {

const aclTensor *EnsureKernelTensorMetadata(const aclTensor *tensor, aclOpExecutor *executor) {
  CHECK_RET(tensor != nullptr && executor != nullptr, nullptr);
  const auto &view_shape = tensor->GetViewShape();
  const auto &storage_shape = tensor->GetStorageShape();
  CHECK_RET(view_shape.GetDimNum() == 2, nullptr);
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

aclTensor *ResolveOutput(const aclTensor *reference, const aclTensor *output, aclOpExecutor *executor) {
  CHECK_RET(reference != nullptr && executor != nullptr, nullptr);
  if (output != nullptr) {
    CHECK_RET(output->GetViewShape() == reference->GetViewShape() && output->GetDataType() == reference->GetDataType(),
              nullptr);
    return const_cast<aclTensor *>(EnsureKernelTensorMetadata(output, executor));
  }
  auto *result = executor->AllocTensor(reference->GetViewShape(), reference->GetDataType());
  CHECK_RET(result != nullptr, nullptr);
  result->SetStorageShape(reference->GetViewShape());
  result->SetOriginalShape(reference->GetViewShape());
  return result;
}

}  // namespace

const aclTensor *HyperMegaGateLinearIndex(const aclTensor *expert_indices, const aclTensor *zero_score_grad,
                                          aclOpExecutor *executor) {
  L0_DFX(HyperMegaGateLinearIndex, expert_indices, zero_score_grad);
  CHECK_RET(expert_indices != nullptr && zero_score_grad != nullptr && executor != nullptr, nullptr);
  CHECK_RET(expert_indices->GetDataType() == op::DataType::DT_INT64, nullptr);
  const auto *indices_matrix = EnsureKernelTensorMetadata(expert_indices, executor);
  const auto *zero_score_grad_matrix = EnsureKernelTensorMetadata(zero_score_grad, executor);
  CHECK_RET(indices_matrix != nullptr && zero_score_grad_matrix != nullptr, nullptr);
  return l0op::LinearIndex(indices_matrix, zero_score_grad_matrix, 1, false, executor);
}

const aclTensor *HyperMegaGateScatterSelectedGrad(const aclTensor *zero_score_grad,
                                                  const aclTensor *expert_indices_int32,
                                                  const aclTensor *selected_score_grad, aclOpExecutor *executor) {
  L0_DFX(HyperMegaGateScatterSelectedGrad, zero_score_grad, expert_indices_int32, selected_score_grad);
  CHECK_RET(zero_score_grad != nullptr && expert_indices_int32 != nullptr && selected_score_grad != nullptr &&
              executor != nullptr,
            nullptr);
  const auto *zero_score_grad_matrix = EnsureKernelTensorMetadata(zero_score_grad, executor);
  const auto *expert_indices_matrix = EnsureKernelTensorMetadata(expert_indices_int32, executor);
  const auto *selected_score_grad_matrix = EnsureKernelTensorMetadata(selected_score_grad, executor);
  CHECK_RET(
    zero_score_grad_matrix != nullptr && expert_indices_matrix != nullptr && selected_score_grad_matrix != nullptr,
    nullptr);
  auto *score_grad = const_cast<aclTensor *>(zero_score_grad_matrix);
  const std::string reduction = "add";
  const auto status = ADD_TO_LAUNCHER_LIST_AICORE(
    ScatterElementsV2, OP_INPUT(zero_score_grad_matrix, expert_indices_matrix, selected_score_grad_matrix),
    OP_OUTPUT(score_grad), OP_ATTR(1, reduction, true));
  CHECK_RET(status == ACLNN_SUCCESS, nullptr);
  return score_grad;
}

const aclTensor *HyperMegaGateDoubleRouteScores(const aclTensor *route_scores, aclOpExecutor *executor) {
  L0_DFX(HyperMegaGateDoubleRouteScores, route_scores);
  CHECK_RET(route_scores != nullptr && executor != nullptr && route_scores->GetDataType() == op::DataType::DT_FLOAT,
            nullptr);
  const auto *route_scores_matrix = EnsureKernelTensorMetadata(route_scores, executor);
  CHECK_RET(route_scores_matrix != nullptr, nullptr);
  auto *doubled_scores = ResolveOutput(route_scores_matrix, nullptr, executor);
  CHECK_RET(doubled_scores != nullptr, nullptr);
  const auto status =
    ADD_TO_LAUNCHER_LIST_AICORE(Muls, OP_INPUT(route_scores_matrix), OP_OUTPUT(doubled_scores), OP_ATTR(2.0F));
  CHECK_RET(status == ACLNN_SUCCESS, nullptr);
  return doubled_scores;
}

const aclTensor *HyperMegaGateSqrtInputGrad(const aclTensor *score_grad, const aclTensor *doubled_scores,
                                            aclOpExecutor *executor) {
  L0_DFX(HyperMegaGateSqrtInputGrad, score_grad, doubled_scores);
  CHECK_RET(score_grad != nullptr && doubled_scores != nullptr && executor != nullptr &&
              score_grad->GetViewShape() == doubled_scores->GetViewShape() &&
              score_grad->GetDataType() == op::DataType::DT_FLOAT &&
              doubled_scores->GetDataType() == op::DataType::DT_FLOAT,
            nullptr);
  const auto *score_grad_matrix = EnsureKernelTensorMetadata(score_grad, executor);
  const auto *doubled_scores_matrix = EnsureKernelTensorMetadata(doubled_scores, executor);
  CHECK_RET(score_grad_matrix != nullptr && doubled_scores_matrix != nullptr, nullptr);
  auto *sqrt_input_grad = ResolveOutput(score_grad_matrix, nullptr, executor);
  CHECK_RET(sqrt_input_grad != nullptr, nullptr);
  const auto status = ADD_TO_LAUNCHER_LIST_AICORE(RealDiv, OP_INPUT(score_grad_matrix, doubled_scores_matrix),
                                                  OP_OUTPUT(sqrt_input_grad));
  CHECK_RET(status == ACLNN_SUCCESS, nullptr);
  return sqrt_input_grad;
}

const aclTensor *HyperMegaGateSoftplusV2Grad(const aclTensor *sqrt_input_grad, const aclTensor *logits,
                                             const aclTensor *output, aclOpExecutor *executor) {
  L0_DFX(HyperMegaGateSoftplusV2Grad, sqrt_input_grad, logits, output);
  CHECK_RET(sqrt_input_grad != nullptr && logits != nullptr && executor != nullptr, nullptr);
  CHECK_RET(sqrt_input_grad->GetViewShape() == logits->GetViewShape() &&
              sqrt_input_grad->GetDataType() == op::DataType::DT_FLOAT &&
              logits->GetDataType() == op::DataType::DT_FLOAT,
            nullptr);
  const auto *sqrt_input_grad_matrix = EnsureKernelTensorMetadata(sqrt_input_grad, executor);
  const auto *logits_matrix = EnsureKernelTensorMetadata(logits, executor);
  CHECK_RET(sqrt_input_grad_matrix != nullptr && logits_matrix != nullptr, nullptr);
  auto *result = ResolveOutput(sqrt_input_grad_matrix, output, executor);
  CHECK_RET(result != nullptr, nullptr);
  const auto status = ADD_TO_LAUNCHER_LIST_AICORE(SoftplusV2Grad, OP_INPUT(sqrt_input_grad_matrix, logits_matrix),
                                                  OP_OUTPUT(result), OP_ATTR(1.0F, 20.0F));
  CHECK_RET(status == ACLNN_SUCCESS, nullptr);
  return result;
}

}  // namespace l0op
