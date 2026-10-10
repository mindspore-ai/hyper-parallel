/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "register/op_impl_registry.h"

namespace ops {
namespace {

ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const auto *logits = context->GetInputShape(0);
  const auto *text_bias = context->GetInputShape(1);
  const auto *vision_bias = context->GetInputShape(2);
  const auto *image_mask = context->GetInputShape(3);
  auto *weights = context->GetOutputShape(0);
  auto *indices = context->GetOutputShape(1);
  auto *route_scores = context->GetOutputShape(2);
  auto *selected_scores = context->GetOutputShape(3);
  auto *denominator = context->GetOutputShape(4);
  if (logits == nullptr || text_bias == nullptr || vision_bias == nullptr || image_mask == nullptr ||
      weights == nullptr || indices == nullptr || route_scores == nullptr || selected_scores == nullptr ||
      denominator == nullptr || logits->GetDimNum() != 2 || text_bias->GetDimNum() != 1 ||
      vision_bias->GetDimNum() != 1 || image_mask->GetDimNum() != 1 || text_bias->GetDim(0) != logits->GetDim(1) ||
      vision_bias->GetDim(0) != logits->GetDim(1) || context->GetAttrs() == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const int64_t *top_k = context->GetAttrs()->GetAttrPointer<int64_t>(0);
  const bool *use_vision_bias = context->GetAttrs()->GetAttrPointer<bool>(2);
  if (top_k == nullptr || use_vision_bias == nullptr || *top_k <= 0 || *top_k > logits->GetDim(1) ||
      image_mask->GetDim(0) != (*use_vision_bias ? logits->GetDim(0) : 1)) {
    return ge::GRAPH_FAILED;
  }
  const gert::Shape selected({logits->GetDim(0), *top_k});
  *weights = selected;
  *indices = selected;
  *route_scores = *logits;
  *selected_scores = selected;
  *denominator = gert::Shape({logits->GetDim(0), 1});
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
  context->SetOutputDataType(0, ge::DT_FLOAT);
  context->SetOutputDataType(1, ge::DT_INT64);
  context->SetOutputDataType(2, ge::DT_FLOAT);
  context->SetOutputDataType(3, ge::DT_FLOAT);
  context->SetOutputDataType(4, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_INFERSHAPE(HyperMegaGateRoute).InferShape(InferShape).InferDataType(InferDataType);

}  // namespace ops
