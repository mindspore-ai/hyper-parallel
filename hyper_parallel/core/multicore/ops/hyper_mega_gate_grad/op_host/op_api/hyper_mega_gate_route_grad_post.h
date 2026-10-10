/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef L0OP_HYPER_MEGA_GATE_ROUTE_GRAD_POST_H
#define L0OP_HYPER_MEGA_GATE_ROUTE_GRAD_POST_H

#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"

namespace l0op {

/** Convert axis-local expert indices into production CANN ScatterElementsV2 indices. */
const aclTensor *HyperMegaGateLinearIndex(const aclTensor *expert_indices, const aclTensor *zero_score_grad,
                                          aclOpExecutor *executor);

/** Scatter selected gradients with production CANN add reduction semantics. */
const aclTensor *HyperMegaGateScatterSelectedGrad(const aclTensor *zero_score_grad,
                                                  const aclTensor *expert_indices_int32,
                                                  const aclTensor *selected_score_grad, aclOpExecutor *executor);

/** Double route scores with the production CANN Muls kernel. */
const aclTensor *HyperMegaGateDoubleRouteScores(const aclTensor *route_scores, aclOpExecutor *executor);

/** Divide score gradients by doubled route scores with the production CANN RealDiv kernel. */
const aclTensor *HyperMegaGateSqrtInputGrad(const aclTensor *score_grad, const aclTensor *doubled_scores,
                                            aclOpExecutor *executor);

/** Append the production CANN SoftplusV2Grad kernel with beta=1 and threshold=20. */
const aclTensor *HyperMegaGateSoftplusV2Grad(const aclTensor *sqrt_input_grad, const aclTensor *logits,
                                             const aclTensor *output, aclOpExecutor *executor);

}  // namespace l0op

#endif  // L0OP_HYPER_MEGA_GATE_ROUTE_GRAD_POST_H
