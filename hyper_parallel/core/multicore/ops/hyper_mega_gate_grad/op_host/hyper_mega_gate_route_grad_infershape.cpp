/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "register/op_impl_registry.h"

namespace ops {
namespace {

ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const auto *selected_scores = context->GetInputShape(0);
  const auto *route_scores = context->GetInputShape(3);
  auto *selected_score_grad = context->GetOutputShape(0);
  auto *zero_score_grad = context->GetOutputShape(1);
  if (selected_scores == nullptr || route_scores == nullptr || selected_score_grad == nullptr ||
      zero_score_grad == nullptr || selected_scores->GetDimNum() != 2 || route_scores->GetDimNum() != 2) {
    return ge::GRAPH_FAILED;
  }
  *selected_score_grad = *selected_scores;
  *zero_score_grad = *route_scores;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
  context->SetOutputDataType(0, ge::DT_FLOAT);
  context->SetOutputDataType(1, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_INFERSHAPE(HyperMegaGateRouteGrad).InferShape(InferShape).InferDataType(InferDataType);

}  // namespace ops
