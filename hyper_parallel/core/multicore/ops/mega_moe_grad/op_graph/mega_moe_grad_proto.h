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
 * \file mega_moe_grad_proto.h
 * \brief
 */
#ifndef OPS_MEGA_MOE_GRAD_PROTO_H_
#define OPS_MEGA_MOE_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
REG_OP(MegaMoeGrad)
  .INPUT(dispatch_target, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(dispatch_target_off, TensorType({ge::DT_INT64, ge::DT_INT64}))
  .INPUT(dy, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(dispatch_src_off, TensorType({ge::DT_INT64, ge::DT_INT64}))
  .INPUT(dispatch_size, TensorType({ge::DT_INT32, ge::DT_INT32}))
  .INPUT(hidden, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(hidden_dw, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(weight, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))  // w2
  .INPUT(y, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))       // act_grad_y
  .INPUT(gate, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(grad_gate, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(w1, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(gate_dx, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(grad_x, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(combine_target_off, TensorType({ge::DT_INT64, ge::DT_INT64}))
  .INPUT(combine_src_off, TensorType({ge::DT_INT64, ge::DT_INT64}))
  .INPUT(combine_size, TensorType({ge::DT_INT32, ge::DT_INT32}))
  .INPUT(permute_out, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(gate_dw, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(group_list, TensorType({ge::DT_INT64, ge::DT_INT64}))
  .INPUT(act_grad_tiling, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .INPUT(gate_grad_tiling, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .INPUT(w1_grad_tiling, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .INPUT(w2_grad_tiling, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .INPUT(swiglu_grad_tiling, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .INPUT(gmm_workspace, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .INPUT(swiglu_grad_workspace, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .INPUT(runtime_config, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .INPUT(all_event_counters, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .OUTPUT(dispatch_target, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .OUTPUT(hidden_dw, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .OUTPUT(y, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))  // act_grad_y
  .OUTPUT(grad_gate, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .OUTPUT(gate_dx, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .OUTPUT(grad_x, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .OUTPUT(gate_dw, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .ATTR(rank_id, Int, 0)
  .ATTR(ep, Int, 0)
  .ATTR(expert_num, Int, 0)
  .ATTR(hidden_size, Int, 0)
  .ATTR(seq_size, Int, 0)
  .OP_END_FACTORY_REG(MegaMoeGrad)
}  // namespace ge

#endif  // OPS_MEGA_MOE_GRAD_PROTO_H_
