/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file multicore_moe_ffn.cpp
 */

#include "kernel_operator.h"
#include "worker_kernel.cpp"
#include "multicore_moe_ffn_tiling_key.h"

using namespace AscendC;

template <int D_T_A, int D_T_B, int D_T_Y, int TRANS_A, int TRANS_B, int GROUP_LIST_TYPE, int IS_STATIC_TILING_API,
          int A8W4_KERNEL_TEMPLATE, int A16W8_KERNEL_TEMPLATE, int AIV_AIC_RATIO>
__global__ __aicore__ void multicore_moe_ffn(
  GM_ADDR dispatch_target, GM_ADDR dispatch_target_off, GM_ADDR dispatch_src, GM_ADDR dispatch_src_off,
  GM_ADDR dispatch_size, GM_ADDR weight, GM_ADDR up_proj_glist, GM_ADDR y, GM_ADDR swiglu_out, GM_ADDR down_proj_weight,
  GM_ADDR down_proj_glist, GM_ADDR down_proj_y, GM_ADDR combine_target, GM_ADDR combine_target_off,
  GM_ADDR combine_src_off, GM_ADDR combine_size, GM_ADDR gmm_workspace, GM_ADDR up_proj_tiling, GM_ADDR swiglu_tiling,
  GM_ADDR down_proj_tiling, GM_ADDR runtime_config, GM_ADDR all_event_counters, GM_ADDR dispatch_target_ref,
  GM_ADDR y_ref, GM_ADDR swiglu_out_ref, GM_ADDR down_proj_y_ref, GM_ADDR combine_target_ref, GM_ADDR workspace,
  GM_ADDR tiling) {
  if (GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM && IS_STATIC_TILING_API == 0 &&
      A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE) {
    if constexpr (TRANS_A == 0 && TRANS_B == 0 && AIV_AIC_RATIO == GROUPED_MATMUL_CUBE_ONLY) {
      GM_ADDR input_list[] = {dispatch_target,   dispatch_target_off,
                              dispatch_src,      dispatch_src_off,
                              dispatch_size,     weight,
                              up_proj_glist,     y,
                              swiglu_out,        down_proj_weight,
                              down_proj_glist,   down_proj_y,
                              combine_target,    combine_target_off,
                              combine_src_off,   combine_size,
                              gmm_workspace,     up_proj_tiling,
                              swiglu_tiling,     down_proj_tiling,
                              runtime_config,    nullptr,
                              workspace,         tiling,
                              all_event_counters};
      uint32_t idx = GetBlockIdx();
      worker_kernel(idx, runtime_config, input_list);
    } else if constexpr (TRANS_A == 0 && TRANS_B == 1 && AIV_AIC_RATIO == GROUPED_MATMUL_CUBE_ONLY) {
    } else if constexpr (TRANS_A == 1 && AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_1) {
    }
  }
}
