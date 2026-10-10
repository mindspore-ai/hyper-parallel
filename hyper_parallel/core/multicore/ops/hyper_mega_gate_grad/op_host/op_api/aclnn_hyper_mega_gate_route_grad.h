/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef ACLNN_HYPER_MEGA_GATE_ROUTE_GRAD_H
#define ACLNN_HYPER_MEGA_GATE_ROUTE_GRAD_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Prepare the sqrt-softplus Route gradient on the current device.
 *
 * Input tensors are borrowed until the returned executor has been launched.
 * All data tensors are contiguous two-dimensional tensors on one device:
 * logits and route_scores are FP32 [tokens, experts]; selected_scores,
 * grad_routing_weights and INT64 expert_indices are [tokens, top_k];
 * normalization_denominator is FP32 [tokens, 1]. runtime_config and
 * profile_buffer are borrowed contiguous UINT8 vectors on the same device.
 * grad_logits is caller-owned contiguous FP32 [tokens, experts] storage.
 *
 * @param top_k Number of selected experts in [1, experts].
 * @param routed_scaling_factor Finite scale used by the forward Route.
 * @param workspaceSize Receives the required temporary workspace in bytes.
 * @param executor Receives the prepared executor on success.
 * @return ACLNN_SUCCESS on success, or an ACLNN parameter/executor error.
 */
ACLNN_API aclnnStatus aclnnHyperMegaGateRouteGradGetWorkspaceSize(
  const aclTensor *logits, const aclTensor *route_scores, const aclTensor *selected_scores,
  const aclTensor *normalization_denominator, const aclTensor *expert_indices, const aclTensor *grad_routing_weights,
  const aclTensor *runtime_config, const aclTensor *profile_buffer, const aclTensor *grad_logits, int64_t top_k,
  double routed_scaling_factor, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief Enqueue a prepared Route gradient operation on stream.
 *
 * workspace is caller-owned storage of at least workspaceSize bytes. The call
 * only enqueues work; output completion follows stream order.
 *
 * @return ACLNN_SUCCESS when enqueue succeeds, otherwise an executor error.
 */
ACLNN_API aclnnStatus aclnnHyperMegaGateRouteGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_HYPER_MEGA_GATE_ROUTE_GRAD_H
