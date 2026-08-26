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
#include "aclnn_hyper_mega_moe.h"
#include "hyper_mega_moe.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnHyperMegaMoeGetWorkspaceSize(
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
    aclnnHyperMegaMoe,
    DFX_IN(dispatch_target, dispatch_target_off, dispatch_src, dispatch_src_off, dispatch_size, weight, up_proj_glist,
           y, swiglu_out, down_proj_weight, down_proj_glist, down_proj_y, combine_target, combine_target_off,
           combine_src_off, combine_size, gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling,
           runtime_config, all_event_counters, rankId, ep, expert_num, hidden_size, seq_size),
    DFX_OUT(dispatch_target, y, swiglu_out, down_proj_y, combine_target));
  auto uniqueExecutor = CREATE_EXECUTOR();
  aclOpExecutor *executorPtr = uniqueExecutor.get();
  CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // Create contiguous tensors in a batch and check for null pointers.
#define MAKE_CONTIGUOUS_CHECK(tensor)                         \
  auto tensor##_cont = l0op::Contiguous(tensor, executorPtr); \
  CHECK_RET(tensor##_cont != nullptr, ACLNN_ERR_INNER_NULLPTR)

  MAKE_CONTIGUOUS_CHECK(dispatch_target);
  MAKE_CONTIGUOUS_CHECK(dispatch_target_off);
  MAKE_CONTIGUOUS_CHECK(dispatch_src);
  MAKE_CONTIGUOUS_CHECK(dispatch_src_off);
  MAKE_CONTIGUOUS_CHECK(dispatch_size);
  MAKE_CONTIGUOUS_CHECK(weight);
  MAKE_CONTIGUOUS_CHECK(up_proj_glist);
  MAKE_CONTIGUOUS_CHECK(y);
  MAKE_CONTIGUOUS_CHECK(swiglu_out);
  MAKE_CONTIGUOUS_CHECK(down_proj_weight);
  MAKE_CONTIGUOUS_CHECK(down_proj_glist);
  MAKE_CONTIGUOUS_CHECK(down_proj_y);
  MAKE_CONTIGUOUS_CHECK(combine_target);
  MAKE_CONTIGUOUS_CHECK(combine_target_off);
  MAKE_CONTIGUOUS_CHECK(combine_size);
  MAKE_CONTIGUOUS_CHECK(gmm_workspace);
  MAKE_CONTIGUOUS_CHECK(up_proj_tiling);
  MAKE_CONTIGUOUS_CHECK(swiglu_tiling);
  MAKE_CONTIGUOUS_CHECK(down_proj_tiling);
  MAKE_CONTIGUOUS_CHECK(runtime_config);
  MAKE_CONTIGUOUS_CHECK(all_event_counters);

#undef MAKE_CONTIGUOUS_CHECK

  auto fwd_output = l0op::HyperMegaMoe(
    dispatch_target, dispatch_target_off, dispatch_src, dispatch_src_off, dispatch_size, weight, up_proj_glist, y,
    swiglu_out, down_proj_weight, down_proj_glist, down_proj_y, combine_target, combine_target_off, combine_src_off,
    combine_size, gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling, runtime_config, all_event_counters,
    rankId, ep, expert_num, hidden_size, seq_size, uniqueExecutor.get());
  bool fwd_output_success = std::all_of(fwd_output.begin(), fwd_output.end(), [](const aclTensor* ptr) {
    return ptr != nullptr;
  });
  CHECK_RET(fwd_output_success, ACLNN_ERR_INNER_NULLPTR);
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnHyperMegaMoe(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnHyperMegaMoe);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
