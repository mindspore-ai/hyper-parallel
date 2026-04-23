/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_multicore_moe_ffn.h"
#include "multicore_moe_ffn.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMulticoreMoeFfnGetWorkspaceSize(
  const aclTensor *dispatch_target, const aclTensor *dispatch_target_off, const aclTensor *dispatch_src,
  const aclTensor *dispatch_src_off, const aclTensor *dispatch_size, const aclTensor *weight,
  const aclTensor *up_proj_glist, const aclTensor *y, const aclTensor *swiglu_out, const aclTensor *down_proj_weight,
  const aclTensor *down_proj_glist, const aclTensor *down_proj_y, const aclTensor *combine_target,
  const aclTensor *combine_target_off, const aclTensor *combine_src_off, const aclTensor *combine_size,
  const aclTensor *gmm_workspace, const aclTensor *up_proj_tiling, const aclTensor *swiglu_tiling,
  const aclTensor *down_proj_tiling, const aclTensor *runtime_config, const aclTensor *all_event_counters,
  int64_t rankId, int64_t ep, int64_t expert_num, int64_t hidden_size, int64_t seq_size, uint64_t *workspaceSize,
  aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(
    aclnnMulticoreMoeFfn,
    DFX_IN(dispatch_target, dispatch_target_off, dispatch_src, dispatch_src_off, dispatch_size, weight, up_proj_glist,
           y, swiglu_out, down_proj_weight, down_proj_glist, down_proj_y, combine_target, combine_target_off,
           combine_src_off, combine_size, gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling,
           runtime_config, all_event_counters, rankId, ep, expert_num, hidden_size, seq_size),
    DFX_OUT(dispatch_target, y, swiglu_out, down_proj_y, combine_target));
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  auto dispatch_target_cont = l0op::Contiguous(dispatch_target, uniqueExecutor.get());
  auto dispatch_target_off_cont = l0op::Contiguous(dispatch_target_off, uniqueExecutor.get());
  auto dispatch_src_cont = l0op::Contiguous(dispatch_src, uniqueExecutor.get());
  auto dispatch_src_off_cont = l0op::Contiguous(dispatch_src_off, uniqueExecutor.get());
  auto dispatch_size_cont = l0op::Contiguous(dispatch_size, uniqueExecutor.get());
  auto weight_cont = l0op::Contiguous(weight, uniqueExecutor.get());
  auto up_proj_glist_cont = l0op::Contiguous(up_proj_glist, uniqueExecutor.get());
  auto y_cont = l0op::Contiguous(y, uniqueExecutor.get());
  auto swiglu_out_cont = l0op::Contiguous(swiglu_out, uniqueExecutor.get());
  auto down_proj_weight_cont = l0op::Contiguous(down_proj_weight, uniqueExecutor.get());
  auto down_proj_glist_cont = l0op::Contiguous(down_proj_glist, uniqueExecutor.get());
  auto down_proj_y_cont = l0op::Contiguous(down_proj_y, uniqueExecutor.get());
  auto combine_target_cont = l0op::Contiguous(combine_target, uniqueExecutor.get());
  auto combine_target_off_cont = l0op::Contiguous(combine_target_off, uniqueExecutor.get());
  auto combine_src_off_cont = l0op::Contiguous(combine_src_off, uniqueExecutor.get());
  auto combine_size_cont = l0op::Contiguous(combine_size, uniqueExecutor.get());
  auto gmm_workspace_cont = l0op::Contiguous(gmm_workspace, uniqueExecutor.get());
  auto up_proj_tiling_cont = l0op::Contiguous(up_proj_tiling, uniqueExecutor.get());
  auto swiglu_tiling_cont = l0op::Contiguous(swiglu_tiling, uniqueExecutor.get());
  auto down_proj_tiling_cont = l0op::Contiguous(down_proj_tiling, uniqueExecutor.get());
  auto runtime_config_cont = l0op::Contiguous(runtime_config, uniqueExecutor.get());
  auto all_event_counters_cont = l0op::Contiguous(all_event_counters, uniqueExecutor.get());
  CHECK_RET(dispatch_target_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(dispatch_target_off_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(dispatch_src_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(dispatch_src_off_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(dispatch_size_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(weight_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(up_proj_glist_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(y_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(swiglu_out_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(down_proj_weight_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(down_proj_glist_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(down_proj_y_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(combine_target_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(combine_target_off_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(combine_src_off_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(combine_size_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(gmm_workspace_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(up_proj_tiling_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(swiglu_tiling_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(down_proj_tiling_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(runtime_config_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(all_event_counters_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto fwd_output = l0op::MulticoreMoeFfn(
    dispatch_target, dispatch_target_off, dispatch_src, dispatch_src_off, dispatch_size, weight, up_proj_glist, y,
    swiglu_out, down_proj_weight, down_proj_glist, down_proj_y, combine_target, combine_target_off, combine_src_off,
    combine_size, gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling, runtime_config, all_event_counters,
    rankId, ep, expert_num, hidden_size, seq_size, uniqueExecutor.get());
  CHECK_RET(fwd_output != nullptr, ACLNN_ERR_INNER_NULLPTR);
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMulticoreMoeFfn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMulticoreMoeFfn);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
