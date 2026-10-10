/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef OPS_HYPER_MEGA_GATE_ROUTE_PROTO_H
#define OPS_HYPER_MEGA_GATE_ROUTE_PROTO_H

#include "graph/operator_reg.h"

namespace ge {

REG_OP(HyperMegaGateRoute)
  .INPUT(logits, TensorType({DT_FLOAT}))
  .INPUT(text_bias, TensorType({DT_FLOAT}))
  .INPUT(vision_bias, TensorType({DT_FLOAT}))
  .INPUT(image_mask, TensorType({DT_BOOL}))
  .INPUT(runtime_config, TensorType({DT_UINT8}))
  .INPUT(profile_buffer, TensorType({DT_UINT8}))
  .OUTPUT(routing_weights, TensorType({DT_FLOAT}))
  .OUTPUT(expert_indices, TensorType({DT_INT64}))
  .OUTPUT(route_scores, TensorType({DT_FLOAT}))
  .OUTPUT(selected_scores, TensorType({DT_FLOAT}))
  .OUTPUT(normalization_denominator, TensorType({DT_FLOAT}))
  .REQUIRED_ATTR(top_k, Int)
  .REQUIRED_ATTR(routed_scaling_factor, Float)
  .REQUIRED_ATTR(use_vision_bias, Bool)
  .OP_END_FACTORY_REG(HyperMegaGateRoute)

}  // namespace ge

#endif  // OPS_HYPER_MEGA_GATE_ROUTE_PROTO_H
