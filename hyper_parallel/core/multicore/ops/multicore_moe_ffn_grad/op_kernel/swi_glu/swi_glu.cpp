/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swi_glu.cpp
 * \brief
 */
#ifndef OPP_MEGA_KERNEL_SWI_GLU_CPP
#define OPP_MEGA_KERNEL_SWI_GLU_CPP
#include "swi_glu_impl.hpp"
#include "swi_glu_bf16.hpp"
#include "swi_glu_single.hpp"

using namespace AscendC;
extern "C" inline __aicore__ void swi_glu(GM_ADDR input_gm, GM_ADDR output_gm,
                                              GM_ADDR workspace, GM_ADDR tiling) {

  GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tempTilingGm, tiling);

  if constexpr (std::is_same_v<DTYPE_DY, bfloat16_t>) {
    if (tempTilingGm.isDoubleBuffer == 1) {
      SwigluSingle<SwigluVectorBF16<bfloat16_t, float, bfloat16_t, 2>, bfloat16_t, bfloat16_t> op;
      op.Init(input_gm, nullptr, output_gm, tiling);
      op.Process();
    } else {
      SwigluSingle<SwigluVectorBF16<bfloat16_t, float, bfloat16_t, 2>, bfloat16_t, bfloat16_t> op;
      op.Init(input_gm, nullptr, output_gm, tiling);
      op.Process();
    }
  } else {
    if (tempTilingGm.isDoubleBuffer == 1) {
      SwigluSingle<SwigluVectorBF16<half, float, half, 2>, half, half> op;
      op.Init(input_gm, nullptr, output_gm, tiling);
      op.Process();
    } else {
      SwigluSingle<SwigluVectorBF16<half, float, half, 1>, half, half> op;
      op.Init(input_gm, nullptr, output_gm, tiling);
      op.Process();
    }
  }
}
#endif  // OPP_MEGA_KERNEL_SWI_GLU_CPP