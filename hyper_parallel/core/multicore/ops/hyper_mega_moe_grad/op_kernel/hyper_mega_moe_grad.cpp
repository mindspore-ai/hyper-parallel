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
 * @file hyper_mega_moe_grad.cpp
 */

#include "kernel_operator.h"
#include "worker_kernel.cpp"
#include "hyper_mega_moe_grad_tiling_key.h"

using namespace AscendC;

template <int D_T_A, int D_T_B, int D_T_Y, int TRANS_A, int TRANS_B, int GROUP_LIST_TYPE, int IS_STATIC_TILING_API,
          int A8W4_KERNEL_TEMPLATE, int A16W8_KERNEL_TEMPLATE, int AIV_AIC_RATIO>
__global__ __aicore__ void hyper_mega_moe_grad(
  GM_ADDR dispatch_target, GM_ADDR dispatch_target_off, GM_ADDR dy, GM_ADDR dispatch_src_off, GM_ADDR dispatch_size,
  GM_ADDR hidden, GM_ADDR hidden_dw, GM_ADDR weight, GM_ADDR y, GM_ADDR gate, GM_ADDR grad_gate, GM_ADDR w1,
  GM_ADDR gate_dx, GM_ADDR grad_x, GM_ADDR combine_target_off, GM_ADDR combine_src_off, GM_ADDR combine_size,
  GM_ADDR permute_out, GM_ADDR gate_dw, GM_ADDR group_list, GM_ADDR act_grad_tiling, GM_ADDR gate_grad_tiling,
  GM_ADDR w1_grad_tiling, GM_ADDR w2_grad_tiling, GM_ADDR swiglu_grad_tiling, GM_ADDR gmm_workspace,
  GM_ADDR swiglu_grad_workspace, GM_ADDR runtime_config, GM_ADDR all_event_counters, GM_ADDR dispatch_target_ref,
  GM_ADDR hidden_dw_ref, GM_ADDR y_ref, GM_ADDR grad_gate_ref, GM_ADDR gate_dx_ref, GM_ADDR grad_x_ref,
  GM_ADDR gate_dw_ref, GM_ADDR workspace, GM_ADDR tiling) {
  if (GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM && IS_STATIC_TILING_API == 0 &&
      A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE) {
    if constexpr (TRANS_A == 0 && TRANS_B == 0 && AIV_AIC_RATIO == GROUPED_MATMUL_CUBE_ONLY) {
      GM_ADDR input_list[] = {dispatch_target,
                              dispatch_target_off,
                              dy,
                              dispatch_src_off,
                              dispatch_size,
                              hidden,  // 5
                              hidden_dw,
                              weight,
                              y,
                              gate,
                              grad_gate,  // 10
                              w1,
                              gate_dx,
                              grad_x,
                              combine_target_off,
                              combine_src_off,  // 15
                              combine_size,
                              permute_out,
                              gate_dw,
                              group_list,
                              act_grad_tiling,  // 20
                              gate_grad_tiling,
                              w1_grad_tiling,
                              w2_grad_tiling,
                              swiglu_grad_tiling,
                              gmm_workspace,  // 25
                              swiglu_grad_workspace,
                              runtime_config,
                              nullptr,
                              workspace,
                              tiling,  // 30
                              all_event_counters};
      uint32_t idx = GetBlockIdx();
      worker_kernel(idx, runtime_config, input_list);
    } else if constexpr (TRANS_A == 0 && TRANS_B == 1 && AIV_AIC_RATIO == GROUPED_MATMUL_CUBE_ONLY) {
    } else if constexpr (TRANS_A == 1 && AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_1) {
    }
  }
}
