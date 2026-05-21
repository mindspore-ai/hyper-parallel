/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MEGA_MOE_
#define OP_API_INC_MEGA_MOE_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnMegaMoeGetWorkspaceSize(
  const aclTensor *dispatch_target, const aclTensor *dispatch_target_off, const aclTensor *dispatch_src,
  const aclTensor *dispatch_src_off, const aclTensor *dispatch_size, const aclTensor *weight,
  const aclTensor *up_proj_glist, const aclTensor *y, const aclTensor *swiglu_out, const aclTensor *down_proj_weight,
  const aclTensor *down_proj_glist, const aclTensor *down_proj_y, const aclTensor *combine_target,
  const aclTensor *combine_target_off, const aclTensor *combine_src_off, const aclTensor *combine_size,
  const aclTensor *gmm_workspace, const aclTensor *up_proj_tiling, const aclTensor *swiglu_tiling,
  const aclTensor *down_proj_tiling, const aclTensor *runtime_config, const aclTensor *all_event_counters,
  int64_t rankId, int64_t ep, int64_t expert_num, int64_t hidden_size, int64_t seq_size, uint64_t *workspaceSize,
  aclOpExecutor **executor);

/**
 */
ACLNN_API aclnnStatus aclnnMegaMoe(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MEGA_MOE_
