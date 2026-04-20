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
 * \file swi_glu_grad_single.hpp
 * \brief
 */
#ifndef OPP_SWI_GLU_GRAD_SINGLE_HPP
#define OPP_SWI_GLU_GRAD_SINGLE_HPP
#include "glu_tiling_kernel.hpp"

using namespace AscendC;
template<typename ParentClass, typename inType, typename outType>
class SwiGluGradSingle : public ParentClass {
public:
    __aicore__ inline SwiGluGradSingle() = default;
    __aicore__ inline ~SwiGluGradSingle() = default;
    __aicore__ inline void Init(GM_ADDR grad_gm, GM_ADDR input_gm, GM_ADDR output_gm, GM_ADDR tiling_gm)
    {
        singleTiling.GetTilingAndOffset(tiling_gm, sizeof(inType));
        InitGmBuffer(grad_gm, input_gm, output_gm);
        this->InitUbBuffer(singleTiling.tileLength);
    }

    __aicore__ inline void Process()
    {
        if (singleTiling.is32BAligned == 1) {
            SWIGLU_SINGLE_PROCESS(singleTiling);
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        else {
            SWIGLU_SINGLE_PROCESS_NON32BALIGNED(singleTiling);
        }
#endif
    }
protected:
    __aicore__ inline void InitGmBuffer(GM_ADDR grad_gm, GM_ADDR input_gm, GM_ADDR output_gm)
    {
        // get start index for current core, core parallel
        this->beta = -1.0f;
        this->aGm.SetGlobalBuffer((__gm__ inType*)input_gm, singleTiling.totalBlockLen);
        this->lGm.SetGlobalBuffer((__gm__ inType*)grad_gm, singleTiling.totalBlockLen / 2);

        this->mGm.SetGlobalBuffer((__gm__ outType*)output_gm, singleTiling.totalBlockLen);
    }

    __aicore__ inline void CopyIn(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams splitCopyinParams = {SwiGluCopyParam.splitVecCopyParam.blockCount,
                                            SwiGluCopyParam.splitVecCopyParam.blockLen,
                                            SwiGluCopyParam.splitVecCopyParam.stride,
                                            0};
        DataCopyParams indepCopyinParams = {SwiGluCopyParam.indepVecCopyParam.blockCount,
                                            SwiGluCopyParam.indepVecCopyParam.blockLen,
                                            SwiGluCopyParam.indepVecCopyParam.stride,
                                            0};

        // Copy A
        LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        DataCopy(aLocal, this->aGm[offsetParam.splitVecGmOffset1], splitCopyinParams);
        this->inQueueA.template EnQue(aLocal);
        // Copy L
        LocalTensor<inType> lLocal = this->inQueueL.template AllocTensor<inType>();
        DataCopy(lLocal, this->lGm[offsetParam.indepVecGmoffset], indepCopyinParams);
        this->inQueueL.template EnQue(lLocal);
        // Copy B
        LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        DataCopy(bLocal, this->aGm[offsetParam.splitVecGmOffset2], splitCopyinParams);
        this->inQueueB.template EnQue(bLocal);
    }

    __aicore__ inline void CopyOut(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams splitCopyoutParams = {SwiGluCopyParam.splitVecCopyParam.blockCount,
                                             SwiGluCopyParam.splitVecCopyParam.blockLen,
                                             0,
                                             SwiGluCopyParam.splitVecCopyParam.stride};

        // deque output tensor from VECOUT queue
        LocalTensor<outType> mLocal = this->outQueueM.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(this->mGm[offsetParam.splitVecGmOffset1], mLocal, splitCopyoutParams);

        // free output tensor for reuse
        this->outQueueM.template FreeTensor(mLocal);

        // deque output tensor from VECOUT queue
        LocalTensor<outType> nLocal = this->outQueueN.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(this->mGm[offsetParam.splitVecGmOffset2], nLocal, splitCopyoutParams);

        // free output tensor for reuse
        this->outQueueN.template FreeTensor(nLocal);
    }

    __aicore__ inline void CopyIn_Non32BAligned(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams splitCopyinParams = {SwiGluCopyParam.splitVecCopyParam.blockCount,
                                            SwiGluCopyParam.splitVecCopyParam.blockLen,
                                            SwiGluCopyParam.splitVecCopyParam.stride,
                                            0};
        DataCopyParams indepCopyinParams = {SwiGluCopyParam.indepVecCopyParam.blockCount,
                                            SwiGluCopyParam.indepVecCopyParam.blockLen,
                                            SwiGluCopyParam.indepVecCopyParam.stride,
                                            0};
        DataCopyPadParams copyPadParams = {false, 0, 0, 0};
        // Copy A
        LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        DataCopyPad(aLocal, this->aGm[offsetParam.splitVecGmOffset1], splitCopyinParams, copyPadParams);
        this->inQueueA.template EnQue(aLocal);
        // Copy L
        LocalTensor<inType> lLocal = this->inQueueL.template AllocTensor<inType>();
        DataCopyPad(lLocal, this->lGm[offsetParam.indepVecGmoffset], indepCopyinParams, copyPadParams);
        this->inQueueL.template EnQue(lLocal);
        // Copy B
        LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        DataCopyPad(bLocal, this->aGm[offsetParam.splitVecGmOffset2], splitCopyinParams, copyPadParams);
        this->inQueueB.template EnQue(bLocal);
    }

    __aicore__ inline void CopyOut_Non32BAligned(SwiGluSingleTileOffsetParam &offsetParam, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
    {
        DataCopyParams splitCopyoutParams = {SwiGluCopyParam.splitVecCopyParam.blockCount,
                                             SwiGluCopyParam.splitVecCopyParam.blockLen,
                                             0,
                                             SwiGluCopyParam.splitVecCopyParam.stride};

        // deque output tensor from VECOUT queue
        LocalTensor<outType> mLocal = this->outQueueM.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopyPad(this->mGm[offsetParam.splitVecGmOffset1], mLocal, splitCopyoutParams);
        // free output tensor for reuse
        this->outQueueM.template FreeTensor(mLocal);

        // deque output tensor from VECOUT queue
        LocalTensor<outType> nLocal = this->outQueueN.template DeQue<outType>();
        // copy progress_th tile from local tensor to global tensor
        DataCopyPad(this->mGm[offsetParam.splitVecGmOffset2], nLocal, splitCopyoutParams);
        // free output tensor for reuse
        this->outQueueN.template FreeTensor(nLocal);
    }

private:
    SwigluSingleTilingKernel singleTiling;
};
#endif // OPP_SWI_GLU_GRAD_SINGLE_HPP
