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
 * \file multicore_moe_ffn_proto.h
 * \brief
 */
#ifndef OPS_MULTICORE_MOE_FFN_PROTO_H_
#define OPS_MULTICORE_MOE_FFN_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
REG_OP(MulticoreMoeFfn)
  .INPUT(dispatch_target, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(dispatch_target_off, TensorType({ge::DT_INT64, ge::DT_INT64}))
  .INPUT(dispatch_src, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(dispatch_src_off, TensorType({ge::DT_INT64, ge::DT_INT64}))
  .INPUT(dispatch_size, TensorType({ge::DT_INT32, ge::DT_INT32}))
  .INPUT(weight, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(up_proj_glist, TensorType({ge::DT_INT64, ge::DT_INT64}))
  .INPUT(y, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(swiglu_out, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(down_proj_weight, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(down_proj_glist, TensorType({ge::DT_INT64, ge::DT_INT64}))
  .INPUT(down_proj_y, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(combine_target, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .INPUT(combine_target_off, TensorType({ge::DT_INT64, ge::DT_INT64}))
  .INPUT(combine_src_off, TensorType({ge::DT_INT64, ge::DT_INT64}))
  .INPUT(combine_size, TensorType({ge::DT_INT32, ge::DT_INT32}))
  .INPUT(gmm_workspace, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .INPUT(up_proj_tiling, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .INPUT(swiglu_tiling, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .INPUT(down_proj_tiling, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .INPUT(runtime_config, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .INPUT(all_event_counters, TensorType({ge::DT_UINT8, ge::DT_UINT8}))
  .OUTPUT(dispatch_target, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .OUTPUT(y, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .OUTPUT(swiglu_out, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .OUTPUT(down_proj_y, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .OUTPUT(combine_target, TensorType({ge::DT_FLOAT16, ge::DT_BF16}))
  .ATTR(rank_id, Int, 0)
  .ATTR(ep, Int, 0)
  .ATTR(expert_num, Int, 0)
  .ATTR(hidden_size, Int, 0)
  .ATTR(seq_size, Int, 0)
  .OP_END_FACTORY_REG(MulticoreMoeFfn)
}  // namespace ge

#endif  // OPS_MULTICORE_MOE_FFN_PROTO_H_
