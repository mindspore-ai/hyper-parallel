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
 * \file swi_glu_grad_bf16.hpp
 * \brief
 */
#ifndef OPP_SWI_GLU_GRAD_BF16_HPP
#define OPP_SWI_GLU_GRAD_BF16_HPP
#include "kernel_operator.h"

using namespace AscendC;
template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
class SwiGluGradBF16 {
  public:
    __aicore__ inline SwiGluGradBF16() {}

  protected:
    __aicore__ inline void InitUbBuffer(uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);
    __aicore__ inline void ComputeSigLocal(uint64_t curTileLen);
    __aicore__ inline void ComputeGradN(uint64_t curTileLen);

    calcType beta = 1.0;
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueA;
    TQue<QuePosition::VECIN, bufferNum> inQueueB;
    TQue<QuePosition::VECIN, bufferNum> inQueueL;
    TQue<QuePosition::VECOUT, bufferNum> outQueueM;
    TQue<QuePosition::VECOUT, bufferNum> outQueueN;
    TBuf<TPosition::VECCALC> tmpQueue;
    TBuf<TPosition::VECCALC> sigQueue;
    LocalTensor<calcType> tempLocal;
    LocalTensor<calcType> sigLocal;
    LocalTensor<calcType> aLocal;
    LocalTensor<calcType> nLocal;
    LocalTensor<calcType> lLocal;

    TBuf<TPosition::VECCALC> aTempBuffer;
    TBuf<TPosition::VECCALC> lTempBuffer;
    TBuf<TPosition::VECCALC> outputTempBuffer;

    GlobalTensor<inType> aGm;
    GlobalTensor<inType> bGm;
    GlobalTensor<inType> lGm;
    GlobalTensor<outType> mGm;
    GlobalTensor<outType> nGm;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    RoundMode downcastRoundMode_ = RoundMode::CAST_RINT;
#else
    RoundMode downcastRoundMode_ = RoundMode::CAST_NONE;
#endif
};

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::InitUbBuffer(uint64_t tileLength)
{
    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueA, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(inQueueB, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(inQueueL, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(outQueueM, bufferNum, tileLength * sizeof(outType)); // The length must be an integer multiple of 32
    pipe.InitBuffer(outQueueN, bufferNum, tileLength * sizeof(outType)); // The length must be an integer multiple of 32

    pipe.InitBuffer(tmpQueue, tileLength * sizeof(calcType));
    pipe.InitBuffer(sigQueue, tileLength * sizeof(calcType));

    pipe.InitBuffer(aTempBuffer, tileLength * sizeof(calcType));
    pipe.InitBuffer(lTempBuffer, tileLength * sizeof(calcType));
    pipe.InitBuffer(outputTempBuffer, tileLength * sizeof(calcType));
    tempLocal = tmpQueue.Get<calcType>();
    sigLocal = sigQueue.Get<calcType>();
    aLocal = aTempBuffer.Get<calcType>();
    nLocal = outputTempBuffer.Get<calcType>();
    lLocal = lTempBuffer.Get<calcType>();
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::Compute(uint64_t tileLength)
{
    // tbuf::templocaltensor
    // deque input tensors from VECIN queue
    //calc sigLocal
    ComputeSigLocal(tileLength);
    //----------------N
    Mul(nLocal, sigLocal, aLocal, tileLength);
    ComputeGradN(tileLength);

    //----------------M
    Muls(tempLocal, sigLocal, (calcType)(-1.0), tileLength);
    PipeBarrier<PIPE_V>();
    Adds(tempLocal, tempLocal, (calcType)(1.0), tileLength);
    PipeBarrier<PIPE_V>();

    auto& mLocal = nLocal;
    Mul(mLocal, sigLocal, tempLocal, tileLength);
    PipeBarrier<PIPE_V>();
    Mul(mLocal, mLocal, aLocal, tileLength);
    PipeBarrier<PIPE_V>();

    Muls(mLocal, mLocal, -beta, tileLength);
    PipeBarrier<PIPE_V>();
    Add(mLocal, mLocal, sigLocal, tileLength);

    LocalTensor<inType> bLocal_ = inQueueB.template DeQue<inType>();
    auto& bLocal = aLocal;
    Cast(bLocal, bLocal_, RoundMode::CAST_NONE, tileLength);
    PipeBarrier<PIPE_V>();

    Mul(mLocal, mLocal, lLocal, tileLength);
    PipeBarrier<PIPE_V>();

    LocalTensor<outType> mLocal_ = outQueueM.template AllocTensor<outType>();

    Mul(mLocal, mLocal, bLocal, tileLength);
    inQueueB.template FreeTensor(bLocal_);

    Cast(mLocal_, mLocal, downcastRoundMode_, tileLength);
    PipeBarrier<PIPE_V>();
    outQueueM.template EnQue<outType>(mLocal_);
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::ComputeSigLocal(uint64_t tileLength)
{
    // tbuf::templocaltensor
    // deque input tensors from VECIN queue
    //calc sigLocal
    LocalTensor<inType> aLocal_ = inQueueA.template DeQue<inType>(); //input a

    Cast(aLocal, aLocal_, RoundMode::CAST_NONE, tileLength);
    PipeBarrier<PIPE_V>();
    inQueueA.template FreeTensor(aLocal_);

    Muls(sigLocal, aLocal, beta, tileLength);
    PipeBarrier<PIPE_V>();
    Exp(sigLocal, sigLocal, tileLength);
    PipeBarrier<PIPE_V>();
    Adds(sigLocal, sigLocal, (calcType)(1.0), tileLength);
    PipeBarrier<PIPE_V>();
    Duplicate<calcType>(tempLocal, (calcType)(1.0), tileLength);
    PipeBarrier<PIPE_V>();
    Div(sigLocal, tempLocal, sigLocal, tileLength);
    PipeBarrier<PIPE_V>();
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::ComputeGradN(uint64_t tileLength)
{
    LocalTensor<inType> lLocal_ = inQueueL.template DeQue<inType>(); // input l
    Cast(lLocal, lLocal_, RoundMode::CAST_NONE, tileLength);
    PipeBarrier<PIPE_V>();
    inQueueL.template FreeTensor(lLocal_);

    LocalTensor<outType> nLocal_ = outQueueN.template AllocTensor<outType>(); // lb

    Mul(nLocal, nLocal, lLocal, tileLength);
    PipeBarrier<PIPE_V>();

    Cast(nLocal_, nLocal, downcastRoundMode_, tileLength); // todo nLocal最后使用位置
    PipeBarrier<PIPE_V>();
    outQueueN.template EnQue<outType>(nLocal_);
}
#endif // OPP_SWI_GLU_GRAD_BF16_HPP
