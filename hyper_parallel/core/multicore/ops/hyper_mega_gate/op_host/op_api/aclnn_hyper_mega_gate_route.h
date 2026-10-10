/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef ACLNN_HYPER_MEGA_GATE_ROUTE_H
#define ACLNN_HYPER_MEGA_GATE_ROUTE_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Prepare one FP32 sqrt-softplus Route operation.
 *
 * Inputs are borrowed until the returned executor has been launched. logits
 * is contiguous FP32 [tokens, experts]; text_bias and vision_bias are FP32
 * [experts]. image_mask is contiguous BOOL [tokens] when use_vision_bias is
 * true and BOOL [1] otherwise. runtime_config and profile_buffer are borrowed
 * contiguous UINT8 vectors on the same device. All five outputs are
 * caller-owned contiguous tensors: routing_weights, selected_scores and
 * expert_indices are [tokens, top_k], route_scores is [tokens, experts], and
 * normalization_denominator is [tokens, 1]. expert_indices is INT64 and the
 * remaining data tensors are FP32.
 *
 * @param top_k Number of selected experts in [1, experts].
 * @param routed_scaling_factor Finite scale applied after normalization.
 * @param use_vision_bias Select per-token text or vision correction bias.
 * @param workspaceSize Receives the required temporary workspace in bytes.
 * @param executor Receives the prepared executor on success.
 * @return ACLNN_SUCCESS on success, or an ACLNN parameter/executor error.
 */
ACLNN_API aclnnStatus aclnnHyperMegaGateRouteGetWorkspaceSize(
  const aclTensor *logits, const aclTensor *text_bias, const aclTensor *vision_bias, const aclTensor *image_mask,
  const aclTensor *runtime_config, const aclTensor *profile_buffer, const aclTensor *routing_weights,
  const aclTensor *expert_indices, const aclTensor *route_scores, const aclTensor *selected_scores,
  const aclTensor *normalization_denominator, int64_t top_k, double routed_scaling_factor, bool use_vision_bias,
  uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief Enqueue a prepared Route operation on stream.
 *
 * workspace is caller-owned storage of at least workspaceSize bytes. The call
 * only enqueues work; output completion follows stream order.
 *
 * @return ACLNN_SUCCESS when enqueue succeeds, otherwise an executor error.
 */
ACLNN_API aclnnStatus aclnnHyperMegaGateRoute(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_HYPER_MEGA_GATE_ROUTE_H
