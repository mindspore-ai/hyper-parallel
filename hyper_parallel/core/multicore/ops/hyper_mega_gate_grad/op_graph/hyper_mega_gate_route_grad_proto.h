/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef OPS_HYPER_MEGA_GATE_ROUTE_GRAD_PROTO_H
#define OPS_HYPER_MEGA_GATE_ROUTE_GRAD_PROTO_H

#include "graph/operator_reg.h"

namespace ge {

REG_OP(HyperMegaGateRouteGrad)
  .INPUT(selected_scores, TensorType({DT_FLOAT}))
  .INPUT(normalization_denominator, TensorType({DT_FLOAT}))
  .INPUT(grad_routing_weights, TensorType({DT_FLOAT}))
  .INPUT(route_scores, TensorType({DT_FLOAT}))
  .INPUT(expert_indices, TensorType({DT_INT64}))
  .INPUT(runtime_config, TensorType({DT_UINT8}))
  .INPUT(profile_buffer, TensorType({DT_UINT8}))
  .OUTPUT(selected_score_grad, TensorType({DT_FLOAT}))
  .OUTPUT(zero_score_grad, TensorType({DT_FLOAT}))
  .REQUIRED_ATTR(top_k, Int)
  .REQUIRED_ATTR(routed_scaling_factor, Float)
  .OP_END_FACTORY_REG(HyperMegaGateRouteGrad)

}  // namespace ge

#endif  // OPS_HYPER_MEGA_GATE_ROUTE_GRAD_PROTO_H
