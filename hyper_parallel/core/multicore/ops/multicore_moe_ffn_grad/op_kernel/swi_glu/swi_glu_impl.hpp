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
 * \file swi_glu_impl.hpp
 * \brief
 */
#ifndef OPP_SWI_GLU_IMPL_HPP
#define OPP_SWI_GLU_IMPL_HPP
#include "kernel_operator.h"

using namespace AscendC;
template<typename inType, typename outType, uint16_t bufferNum>
class SwigluVector {
  public:
    __aicore__ inline SwigluVector() {}
    __aicore__ inline ~SwigluVector() {}
  protected:
    __aicore__ inline void InitUbBuffer(uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);
    float beta = 1.0;
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueA;
    TQue<QuePosition::VECIN, bufferNum> inQueueB;
    TQue<QuePosition::VECOUT, bufferNum> outQueueC;
    TBuf<TPosition::VECCALC> tmpQueue;

    GlobalTensor<inType> aGm;
    GlobalTensor<inType> bGm;
    GlobalTensor<outType> cGm;
};

template<typename inType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwigluVector<inType, outType, bufferNum>::InitUbBuffer(uint64_t tileLength) {
    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueA, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(inQueueB, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(outQueueC, bufferNum, tileLength * sizeof(outType));
    pipe.InitBuffer(tmpQueue, tileLength * sizeof(float));
    LocalTensor<float> tempLocal = tmpQueue.Get<float>();
    Duplicate<float>(tempLocal, (float)(1.0), tileLength);
}

template<typename inType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwigluVector<inType, outType, bufferNum>::Compute(uint64_t curTileLen)
{
    LocalTensor<inType> aLocal = inQueueA.template DeQue<inType>();
    LocalTensor<outType> cLocal = outQueueC.template AllocTensor<outType>();
    PipeBarrier<PIPE_V>();
    Muls(cLocal, aLocal, beta, curTileLen);
    PipeBarrier<PIPE_V>();
    Exp(cLocal, cLocal, curTileLen);
    PipeBarrier<PIPE_V>();
    Adds(cLocal, cLocal, (outType)(1.0), curTileLen);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> tempLocal = tmpQueue.Get<float>();
    PipeBarrier<PIPE_V>();
    Div(cLocal, tempLocal, cLocal, curTileLen);
    PipeBarrier<PIPE_V>();
    Mul(cLocal, cLocal, aLocal, curTileLen);
    PipeBarrier<PIPE_V>();
    inQueueA.template FreeTensor(aLocal);

    LocalTensor<inType> bLocal = inQueueB.template DeQue<inType>();
    PipeBarrier<PIPE_V>();
    Mul(cLocal, cLocal, bLocal, curTileLen);
    PipeBarrier<PIPE_V>();
    inQueueB.template FreeTensor(bLocal);
    // enque the output tensor to VECOUT queue
    outQueueC.template EnQue<outType>(cLocal);
    // free input tensors for reuse
}
#endif // OPP_SWI_GLU_IMPL_HPP
