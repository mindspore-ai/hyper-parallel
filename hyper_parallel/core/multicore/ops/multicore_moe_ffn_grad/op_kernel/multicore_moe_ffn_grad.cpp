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
 * @file add_custom.cpp
 */

#include "kernel_operator.h"
#include "worker_kernel.cpp"
#include "multicore_moe_ffn_grad_tiling_key.h"

using namespace AscendC;

template <int D_T_A, int D_T_B, int D_T_Y, int TRANS_A, int TRANS_B, int GROUP_LIST_TYPE,
          int IS_STATIC_TILING_API, int A8W4_KERNEL_TEMPLATE, int A16W8_KERNEL_TEMPLATE, int AIV_AIC_RATIO>
__global__ __aicore__ void multicore_moe_ffn_grad(GM_ADDR dispatch_target,
                                            GM_ADDR dispatch_target_off,
                                            GM_ADDR dy,
                                            GM_ADDR dispatch_src_off,
                                            GM_ADDR dispatch_size,
                                            GM_ADDR hidden,
                                            GM_ADDR hidden_dw,
                                            GM_ADDR weight,
                                            GM_ADDR y,
                                            GM_ADDR gate,
                                            GM_ADDR grad_gate,
                                            GM_ADDR w1,
                                            GM_ADDR gate_dx,
                                            GM_ADDR grad_x,
                                            GM_ADDR combine_target_off,
                                            GM_ADDR combine_src_off,
                                            GM_ADDR combine_size,
                                            GM_ADDR permute_out,
                                            GM_ADDR gate_dw,
                                            GM_ADDR group_list,
                                            GM_ADDR act_grad_tiling,
                                            GM_ADDR gate_grad_tiling,
                                            GM_ADDR w2_grad_tiling,
                                            GM_ADDR w1_grad_tiling,
                                            GM_ADDR swiglu_grad_tiling,
                                            GM_ADDR gmm_workspace,
                                            GM_ADDR swiglu_grad_workspace,
                                            GM_ADDR runtime_config,
                                            GM_ADDR all_event_counters,
                                            GM_ADDR dispatch_target_ref,
                                            GM_ADDR hidden_dw_ref,
                                            GM_ADDR y_ref,
                                            GM_ADDR grad_gate_ref,
                                            GM_ADDR gate_dx_ref,
                                            GM_ADDR grad_x_ref,
                                            GM_ADDR gate_dw_ref,
                                            GM_ADDR workspace,
                                            GM_ADDR tiling)
{
  if (GROUP_LIST_TYPE != GROUPED_MATMUL_GROUP_LIST_TYPE_SPARSEM &&
      IS_STATIC_TILING_API == 0 &&
      A8W4_KERNEL_TEMPLATE == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE) {
          if constexpr (TRANS_A == 0 && TRANS_B == 0 && AIV_AIC_RATIO == GROUPED_MATMUL_CUBE_ONLY) {
              GM_ADDR input_list[] = {dispatch_target, dispatch_target_off, dy, dispatch_src_off, dispatch_size, //4
                                      hidden, hidden_dw, weight, y, gate, grad_gate, w1, //11
                                      gate_dx, grad_x, //13
                                      combine_target_off, combine_src_off, combine_size, permute_out, //17
                                      gate_dw, group_list, //19
                                      act_grad_tiling, gate_grad_tiling, w2_grad_tiling, w1_grad_tiling,//23
                                      swiglu_grad_tiling,
                                      gmm_workspace, swiglu_grad_workspace,//26
                                      runtime_config, nullptr,
                                      workspace, tiling, all_event_counters};
              uint32_t idx = GetBlockIdx();
              worker_kernel(idx, runtime_config, input_list);
          } else if constexpr (TRANS_A == 0 && TRANS_B == 1 && AIV_AIC_RATIO == GROUPED_MATMUL_CUBE_ONLY) {
          } else if constexpr (TRANS_A == 1 && AIV_AIC_RATIO == GROUPED_MATMUL_AIV_AIC_RATIO_1) {
          }
  }
}
