/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swi_glu_grad.cpp
 * \brief
 */
#include "swi_glu_grad_float.hpp"
#include "swi_glu_grad_bf16.hpp"
#include "swi_glu_grad_single.hpp"

using namespace AscendC;
extern "C" inline __aicore__ void swi_glu_grad(GM_ADDR gradout_gm, GM_ADDR input_gm, GM_ADDR output_gm,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
    // DT_FLOAT = 0,            // float type
    // DT_FLOAT16 = 1,          // fp16 type
    // DT_BF16 = 27,            // bf16 type
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tempTilingGm, tiling);
    if constexpr (std::is_same_v<DTYPE_DY, bfloat16_t>) {
      if (tempTilingGm.isDoubleBuffer == 1) {
        SwiGluGradSingle <SwiGluGradBF16<bfloat16_t, float, bfloat16_t, 2>, bfloat16_t, bfloat16_t> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      } else {
        SwiGluGradSingle <SwiGluGradBF16<bfloat16_t, float, bfloat16_t, 1>, bfloat16_t, bfloat16_t> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      }
    } else {
      if (tempTilingGm.isDoubleBuffer == 1) {
        SwiGluGradSingle <SwiGluGradBF16<half, float, half, 2>, half, half> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      } else {
        SwiGluGradSingle <SwiGluGradBF16<half, float, half, 1>, half, half> op;
        op.Init(gradout_gm, input_gm, output_gm, tiling);
        op.Process();
      }
    }
}
