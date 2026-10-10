/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef L0OP_HYPER_MEGA_GATE_ROUTE_GRAD_H
#define L0OP_HYPER_MEGA_GATE_ROUTE_GRAD_H

#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"

namespace l0op {

struct RouteGradOutputs {
  const aclTensor *selected_score_grad;
  const aclTensor *zero_score_grad;
};

/**
 * Append the token-row Router gradient kernel.
 *
 * The kernel preserves the golden normalization-gradient order and returns
 * the selected gradient and zero initialized score-gradient base.
 */
RouteGradOutputs HyperMegaGateRouteGradKernel(const aclTensor *selected_scores,
                                              const aclTensor *normalization_denominator,
                                              const aclTensor *grad_routing_weights, const aclTensor *route_scores,
                                              const aclTensor *expert_indices, const aclTensor *runtime_config,
                                              const aclTensor *profile_buffer, int64_t top_k,
                                              float routed_scaling_factor, aclOpExecutor *executor);

/**
 * @brief Append RouteGrad and its production CANN post-processing kernels.
 *
 * All inputs are borrowed until executor launch and must use contiguous
 * current-device storage. output is caller-owned FP32 [tokens, experts]
 * storage. Completion follows the executor's stream order.
 *
 * @return output on success, otherwise nullptr.
 */
const aclTensor *HyperMegaGateRouteGrad(const aclTensor *logits, const aclTensor *route_scores,
                                        const aclTensor *selected_scores, const aclTensor *normalization_denominator,
                                        const aclTensor *expert_indices, const aclTensor *grad_routing_weights,
                                        const aclTensor *runtime_config, const aclTensor *profile_buffer, int64_t top_k,
                                        float routed_scaling_factor, const aclTensor *output, aclOpExecutor *executor);

}  // namespace l0op

#endif  // L0OP_HYPER_MEGA_GATE_ROUTE_GRAD_H
