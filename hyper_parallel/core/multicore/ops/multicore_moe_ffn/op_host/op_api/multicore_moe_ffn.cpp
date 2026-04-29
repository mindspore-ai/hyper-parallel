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
 * \file multicore_moe_ffn.cpp
 * \brief
 */
#include "multicore_moe_ffn.h"
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
OP_TYPE_REGISTER(MulticoreMoeFfn);

const std::array<const aclTensor *, 5> MulticoreMoeFfn(
  const aclTensor *dispatch_target, const aclTensor *dispatch_target_off, const aclTensor *dispatch_src,
  const aclTensor *dispatch_src_off, const aclTensor *dispatch_size, const aclTensor *weight,
  const aclTensor *up_proj_glist, const aclTensor *y, const aclTensor *swiglu_out, const aclTensor *down_proj_weight,
  const aclTensor *down_proj_glist, const aclTensor *down_proj_y, const aclTensor *combine_target,
  const aclTensor *combine_target_off, const aclTensor *combine_src_off, const aclTensor *combine_size,
  const aclTensor *gmm_workspace, const aclTensor *up_proj_tiling, const aclTensor *swiglu_tiling,
  const aclTensor *down_proj_tiling, const aclTensor *runtime_config, const aclTensor *all_event_counters,
  int64_t rankId, int64_t ep, int64_t expert_num, int64_t hidden_size, int64_t seq_size, aclOpExecutor *executor) {
  L0_DFX(MulticoreMoeFfn, dispatch_target, dispatch_target_off, dispatch_src, dispatch_src_off, dispatch_size, weight,
         up_proj_glist, y, swiglu_out, down_proj_weight, down_proj_glist, down_proj_y, combine_target,
         combine_target_off, combine_src_off, combine_size, gmm_workspace, up_proj_tiling, swiglu_tiling,
         down_proj_tiling, runtime_config, all_event_counters, rankId, ep, expert_num, hidden_size, seq_size);
  auto dispatch_target_out = const_cast<aclTensor *>(dispatch_target);
  auto y_out = const_cast<aclTensor *>(y);
  auto swiglu_out_out = const_cast<aclTensor *>(swiglu_out);
  auto down_proj_y_out = const_cast<aclTensor *>(down_proj_y);
  auto combine_target_out = const_cast<aclTensor *>(combine_target);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
    MulticoreMoeFfn,
    OP_INPUT(dispatch_target, dispatch_target_off, dispatch_src, dispatch_src_off, dispatch_size, weight, up_proj_glist,
             y, swiglu_out, down_proj_weight, down_proj_glist, down_proj_y, combine_target, combine_target_off,
             combine_src_off, combine_size, gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling,
             runtime_config, all_event_counters),
    OP_OUTPUT(dispatch_target_out, y_out, swiglu_out_out, down_proj_y_out, combine_target_out),
    OP_ATTR(rankId, ep, expert_num, hidden_size, seq_size));
  if (ret != ACL_SUCCESS) {
    return {dispatch_target_out, y_out, swiglu_out_out, down_proj_y_out, combine_target_out};
  }
  return {dispatch_target_out, y_out, swiglu_out_out, down_proj_y_out, combine_target_out};
}
}  // namespace l0op
