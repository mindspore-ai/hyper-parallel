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
 * \file swi_glu_bf16.hpp
 * \brief
 */
#ifndef OPP_SWI_GLU_BF16_HPP
#define OPP_SWI_GLU_BF16_HPP
#include "kernel_operator.h"

using namespace AscendC;
template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
class SwigluVectorBF16 {
public:
    __aicore__ inline SwigluVectorBF16() {}
    __aicore__ inline ~SwigluVectorBF16() {}

protected:
    __aicore__ inline void InitUbBuffer(uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);

    float beta = 1.0;
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueA;
    TQue<QuePosition::VECIN, bufferNum> inQueueB;
    TQue<QuePosition::VECOUT, bufferNum> outQueueC;

    TBuf<TPosition::VECCALC> inputTempBuffer;
    TBuf<TPosition::VECCALC> outputTempBuffer;

    GlobalTensor<inType> aGm;
    GlobalTensor<inType> bGm;
    GlobalTensor<outType> cGm;
};

  template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
  __aicore__ inline void SwigluVectorBF16<inType, calcType, outType, bufferNum>::InitUbBuffer(uint64_t tileLength) {
      // pipe alloc memory to queue, the unit is Bytes
      pipe.InitBuffer(inQueueA, bufferNum, tileLength * sizeof(inType));
      pipe.InitBuffer(inQueueB, bufferNum, tileLength * sizeof(inType));
      pipe.InitBuffer(outQueueC, bufferNum, tileLength * sizeof(outType));

      pipe.InitBuffer(inputTempBuffer, tileLength * sizeof(calcType));
      pipe.InitBuffer(outputTempBuffer, tileLength * sizeof(calcType));
  }

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwigluVectorBF16<inType, calcType, outType, bufferNum>::Compute(uint64_t curTileLen)
{
    LocalTensor<inType> aLocal_ = inQueueA.template DeQue<inType>();
    LocalTensor<outType> cLocal_ = outQueueC.template AllocTensor<outType>();

    LocalTensor<calcType> aLocal = inputTempBuffer.Get<calcType>();
    LocalTensor<calcType> cLocal = outputTempBuffer.Get<calcType>();
    Cast(aLocal, aLocal_, RoundMode::CAST_NONE, curTileLen);
    PipeBarrier<PIPE_V>();
    inQueueA.template FreeTensor(aLocal_);

    Muls(cLocal, aLocal, beta, curTileLen);
    PipeBarrier<PIPE_V>();
    Exp(cLocal, cLocal, curTileLen);
    PipeBarrier<PIPE_V>();
    Adds(cLocal, cLocal, calcType(1.0), curTileLen);
    PipeBarrier<PIPE_V>();

    Div(cLocal, aLocal, cLocal, curTileLen);
    PipeBarrier<PIPE_V>();

    LocalTensor<inType> bLocal_ = inQueueB.template DeQue<inType>();

    LocalTensor<calcType> bLocal = inputTempBuffer.Get<calcType>();
    Cast(bLocal, bLocal_, RoundMode::CAST_NONE, curTileLen);
    PipeBarrier<PIPE_V>();
    inQueueB.template FreeTensor(bLocal_);

    Mul(cLocal, cLocal, bLocal, curTileLen);
    PipeBarrier<PIPE_V>();

    Cast(cLocal_, cLocal, RoundMode::CAST_RINT, curTileLen);
    PipeBarrier<PIPE_V>();
    // enque the output tensor to VECOUT queue
    outQueueC.template EnQue<outType>(cLocal_);
    // free input tensors for reuse
}
#endif  // OPP_SWI_GLU_BF16_HPP
