/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file multicore_moe_ffn_grad.cpp
 * \brief
 */
#include "multicore_moe_ffn_grad.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"

namespace l0op {
OP_TYPE_REGISTER(MulticoreMoeFfnGrad);

const std::array<const aclTensor *, 7> MulticoreMoeFfnGrad(
  const aclTensor *dispatch_target, const aclTensor *dispatch_target_off, const aclTensor *dy,
  const aclTensor *dispatch_src_off, const aclTensor *dispatch_size, const aclTensor *hidden,
  const aclTensor *hidden_dw,
  const aclTensor *weight,  // w2
  const aclTensor *y,       // act_grad_y
  const aclTensor *gate, const aclTensor *grad_gate, const aclTensor *w1, const aclTensor *gate_dx,
  const aclTensor *grad_x, const aclTensor *combine_target_off, const aclTensor *combine_src_off,
  const aclTensor *combine_size, const aclTensor *permute_out, const aclTensor *gate_dw, const aclTensor *group_list,
  const aclTensor *act_grad_tiling, const aclTensor *gate_grad_tiling, const aclTensor *w2_grad_tiling,
  const aclTensor *w1_grad_tiling, const aclTensor *swiglu_grad_tiling, const aclTensor *gmm_workspace,
  const aclTensor *swiglu_grad_workspace, const aclTensor *runtime_config, const aclTensor *all_event_counters,
  int64_t rankId, int64_t ep, int64_t expert_num, int64_t hidden_size, int64_t seq_size, aclOpExecutor *executor) {
  L0_DFX(MulticoreMoeFfnGrad, dispatch_target, dispatch_target_off, dy, dispatch_src_off, dispatch_size, hidden,
         hidden_dw, weight, y, gate, grad_gate, w1, gate_dx, grad_x, combine_target_off, combine_src_off, combine_size,
         permute_out, gate_dw, group_list, act_grad_tiling, gate_grad_tiling, w2_grad_tiling, w1_grad_tiling,
         swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace, runtime_config, all_event_counters, rankId, ep,
         expert_num, hidden_size, seq_size);
  auto dispatch_target_out = const_cast<aclTensor *>(dispatch_target);
  auto hidden_dw_out = const_cast<aclTensor *>(hidden_dw);
  auto y_out = const_cast<aclTensor *>(y);
  auto grad_gate_out = const_cast<aclTensor *>(grad_gate);
  auto gate_dx_out = const_cast<aclTensor *>(gate_dx);
  auto grad_x_out = const_cast<aclTensor *>(grad_x);
  auto gate_dw_out = const_cast<aclTensor *>(gate_dw);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
    MulticoreMoeFfnGrad,
    OP_INPUT(dispatch_target, dispatch_target_off, dy, dispatch_src_off, dispatch_size, hidden, hidden_dw, weight, y,
             gate, grad_gate, w1, gate_dx, grad_x, combine_target_off, combine_src_off, combine_size, permute_out,
             gate_dw, group_list, act_grad_tiling, gate_grad_tiling, w2_grad_tiling, w1_grad_tiling, swiglu_grad_tiling,
             gmm_workspace, swiglu_grad_workspace, runtime_config, all_event_counters),
    OP_OUTPUT(dispatch_target_out, hidden_dw_out, y_out, grad_gate_out, gate_dx_out, grad_x_out, gate_dw_out),
    OP_ATTR(rankId, ep, expert_num, hidden_size, seq_size));
  if (ret != ACL_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MulticoreMoeFfnGrad ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return {dispatch_target_out, hidden_dw_out, y_out, grad_gate_out, gate_dx_out, grad_x_out, gate_dw_out};
  }
  return {dispatch_target_out, hidden_dw_out, y_out, grad_gate_out, gate_dx_out, grad_x_out, gate_dw_out};
}
}  // namespace l0op
