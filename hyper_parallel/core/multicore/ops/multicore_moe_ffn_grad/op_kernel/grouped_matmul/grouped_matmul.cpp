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
 * \file grouped_matmul.cpp
 * \brief
 */
#include "grouped_matmul_utils.h"
#include "grouped_matmul.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "grouped_matmul_pre_tiling.h"
#endif


using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MATMUL;

#ifndef FORMAT_FRACTAL_NZ
    #define FORMAT_FRACTAL_NZ
#endif

namespace {
#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
constexpr CubeFormat wFormat = CubeFormat::NZ;
constexpr MatmulConfig matmulCFG = NZ_CFG_MDL;
#else
constexpr CubeFormat wFormat = CubeFormat::ND;
constexpr MatmulConfig matmulCFG = CFG_MDL;
#endif

#if defined(GMM_ANTI_QUANT_A8W4_MSD)
constexpr MatmulConfig A8W4_GMM_CFG_MDL = GetNormalConfig();
constexpr auto GetMmCFG() {
    auto CFG = CFG_MDL;
    CFG.isPartialOutput = true;
    return CFG;
}
constexpr MatmulConfig A8W4_GMM_CFG_MDL_NEW = GetMmCFG();
#endif
}

template <bool trans = false>
using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_WEIGHT, trans>;

template <bool trans = false>
using xTypeMSD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_WEIGHT, trans>;

template <bool trans = false>
using weightType = MatmulType<AscendC::TPosition::GM, wFormat, DTYPE_WEIGHT, trans>;

template <bool trans = false>
using weightTypeMSD = MatmulType<AscendC::TPosition::GM, wFormat, DTYPE_WEIGHT, trans>;

using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, MM_DTYPE_Y>;

using yTypeMSD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;

// using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_WEIGHT>;
using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

namespace {
    __aicore__ inline static constexpr MatmulApiStaticTiling GetGmmMatmulApiTiling(bool isND2NZ, bool transB) {
        MatmulConfig conf = GenGmmConf(isND2NZ);
        MatmulApiStaticTiling staticTilingTmp;
        if (transB) {
            staticTilingTmp = GetMatmulApiTiling<xType<false>, weightType<true>, yType, biasType>(conf);
        } else {
            staticTilingTmp = GetMatmulApiTiling<xType<false>, weightType<false>, yType, biasType>(conf);
        }
        staticTilingTmp.depthA1 = STATIC_TILING_DEPTH_A1_B1;
        staticTilingTmp.depthB1 = STATIC_TILING_DEPTH_A1_B1;
        staticTilingTmp.stepM = 1;
        staticTilingTmp.stepN = 1;
        staticTilingTmp.stepKa = STATIC_TILING_STEP_KA_KB;
        staticTilingTmp.stepKb = STATIC_TILING_STEP_KA_KB;
        staticTilingTmp.dbL0A = DOUBLE_BUFFER_L0A_L0B;
        staticTilingTmp.dbL0B = DOUBLE_BUFFER_L0A_L0B;
        staticTilingTmp.dbL0C = 1;
        return staticTilingTmp;
    }
#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
    constexpr bool isWeightNZ = true;
#else
    constexpr bool isWeightNZ = false;
#endif
    constexpr static auto staticCFG = GetGmmMatmulApiTiling(isWeightNZ, false);
    constexpr static auto staticCFGtransB = GetGmmMatmulApiTiling(isWeightNZ, true);
} // namespace

#define GMM_CUBE_IMP(processClass, transA, transB, sync, cfg)                                                      \
    do {                                                                                                           \
        if ASCEND_IS_AIV {                                                                                         \
            return;                                                                                                \
        }                                                                                                          \
        using matmulType = MMImplType<xType<transA>, weightType<transB>, yType, biasType, cfg>;                    \
        matmulType::MT mm;                                                                                         \
        GET_TILING_DATA_MEMBER(GMMTilingData, gmmBaseParams, gmmBaseParams_, tiling);                              \
        GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, mmTilingData_, tiling);                                \
        GET_TILING_DATA_MEMBER_ADDR(GMMTilingData, gmmArray, gmmArrayAddr_, tiling);                               \
        mm.SetSubBlockIdx(0);                                                                                      \
        mm.Init(&mmTilingData_, &tPipe);                                                                           \
        GMMCompute<matmulType, sync> computeOp(mm);                                                                \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale,  \
                       y, user1,  &gmmBaseParams_, &mmTilingData_, &tPipe);                                        \
        processClass<decltype(computeOp)> op(computeOp);                                                           \
        op.Init(&gmmBaseParams_, &mmTilingData_, gmmArrayAddr_, groupList, tiling);                                \
        op.Process();                                                                                              \
    } while (0)

// template <int D_T_A, int D_T_B, int D_T_Y, int TRANS_A, int TRANS_B, int GROUP_LIST_TYPE,
//           int IS_STATIC_TILING_API, int A8W4_KERNEL_TEMPLATE, int A16W8_KERNEL_TEMPLATE, int AIV_AIC_RATIO>
extern "C" inline __aicore__ void grouped_matmul(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                                     GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                                     GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                                     GM_ADDR workspace, GM_ADDR tiling, bool trans_a, bool trans_b) {
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    GM_ADDR user1 = GetUserWorkspace(workspace);
    if (trans_a) {
        GMM_CUBE_IMP(GMMProcess, true, false, false, matmulCFG);
    } else if (trans_b) {
        GMM_CUBE_IMP(GMMProcess, false, true, false, matmulCFGUnitFlag);
    } else {
        GMM_CUBE_IMP(GMMProcess, false, false, false, matmulCFGUnitFlag);
    }
}
