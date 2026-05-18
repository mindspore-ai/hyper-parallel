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
 * \file mega_moe.h
 * \brief
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_MEGA_MOE_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_MEGA_MOE_H_

#include "opdev/op_executor.h"

namespace l0op {

const std::array<const aclTensor *, 5> MegaMoe(
  const aclTensor *dispatch_target, const aclTensor *dispatch_target_off, const aclTensor *dispatch_src,
  const aclTensor *dispatch_src_off, const aclTensor *dispatch_size, const aclTensor *weight,
  const aclTensor *up_proj_glist, const aclTensor *y, const aclTensor *swiglu_out, const aclTensor *down_proj_weight,
  const aclTensor *down_proj_glist, const aclTensor *down_proj_y, const aclTensor *combine_target,
  const aclTensor *combine_target_off, const aclTensor *combine_src_off, const aclTensor *combine_size,
  const aclTensor *gmm_workspace, const aclTensor *up_proj_tiling, const aclTensor *swiglu_tiling,
  const aclTensor *down_proj_tiling, const aclTensor *runtime_config, const aclTensor *all_event_counters,
  int64_t rankId, int64_t ep, int64_t expert_num, int64_t hidden_size, int64_t seq_size, aclOpExecutor *executor);
}  // namespace l0op

#endif  // PTA_NPU_OP_API_INC_LEVEL0_OP_MEGA_MOE_H_
