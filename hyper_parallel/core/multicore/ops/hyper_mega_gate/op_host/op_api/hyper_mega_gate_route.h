/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef L0OP_HYPER_MEGA_GATE_ROUTE_H
#define L0OP_HYPER_MEGA_GATE_ROUTE_H

#include <array>

#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Append one FP32 sqrtsoftplus Route operation to an aclnn executor.
 * @param logits Borrowed contiguous [tokens, experts] FP32 scores.
 * @param text_bias Borrowed contiguous [experts] FP32 text selection bias.
 * @param vision_bias Borrowed contiguous [experts] FP32 vision selection bias.
 * @param image_mask Borrowed contiguous BOOL token mask: [tokens] in vision mode, otherwise [1].
 * @param runtime_config Borrowed UINT8 ten-stage broadcast schedule.
 * @param profile_buffer Borrowed UINT8 cycle buffer, unread on the fast path.
 * @param routing_weights Caller-owned [tokens, top_k] FP32 output.
 * @param expert_indices Caller-owned [tokens, top_k] INT64 output.
 * @param route_scores Caller-owned [tokens, experts] FP32 backward state.
 * @param selected_scores Caller-owned [tokens, top_k] FP32 backward state.
 * @param normalization_denominator Caller-owned [tokens, 1] FP32 backward state.
 * @param top_k Number of experts selected for each row.
 * @param routed_scaling_factor Finite scale applied after normalization.
 * @param use_vision_bias Select per-token text or vision correction bias.
 * @param executor Current-stream executor that owns the appended operation.
 * @return The caller-owned outputs on success; null entries on failure.
 */
const std::array<const aclTensor *, 5> HyperMegaGateRoute(
  const aclTensor *logits, const aclTensor *text_bias, const aclTensor *vision_bias, const aclTensor *image_mask,
  const aclTensor *runtime_config, const aclTensor *profile_buffer, const aclTensor *routing_weights,
  const aclTensor *expert_indices, const aclTensor *route_scores, const aclTensor *selected_scores,
  const aclTensor *normalization_denominator, int64_t top_k, float routed_scaling_factor, bool use_vision_bias,
  aclOpExecutor *executor);

}  // namespace l0op

#endif  // L0OP_HYPER_MEGA_GATE_ROUTE_H
