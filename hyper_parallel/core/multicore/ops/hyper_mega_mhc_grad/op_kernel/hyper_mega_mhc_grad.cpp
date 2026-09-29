/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "kernel_operator.h"
#include "hyper_mega_mhc_grad_tiling_key.h"
#include "worker_kernel.cpp"

using namespace AscendC;

template <int8_t MHC_GRAD_MODE>
__global__ __aicore__ void hyper_mega_mhc_grad(
    GM_ADDR gradHinPlaceholder, GM_ADDR gradHPost, GM_ADDR gradHRes, GM_ADDR x,
    GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR previousPre,
    GM_ADDR hcBeforeNorm, GM_ADDR invRms, GM_ADDR sumOut, GM_ADDR normOut,
    GM_ADDR gradCurrentPre, GM_ADDR mixedInput, GM_ADDR rmsRstd, GM_ADDR normWeight,
    GM_ADDR directGradX, GM_ADDR previousResidual, GM_ADDR previousOutput,
    GM_ADDR previousPost, GM_ADDR previousResidualMix, GM_ADDR runtimeConfig,
    GM_ADDR allEventCounters, GM_ADDR profileBuffer, GM_ADDR gradResidual,
    GM_ADDR gradPhi, GM_ADDR gradAlpha, GM_ADDR gradBias, GM_ADDR gradPreviousOutput,
    GM_ADDR gradPreviousPre, GM_ADDR gradPreviousPost, GM_ADDR gradPreviousResidual,
    GM_ADDR gradNormWeight, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MhcPreSinkhornBackwardArch22TilingData);
    if (workspace == nullptr || runtimeConfig == nullptr || allEventCounters == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    GM_ADDR inputList[] = {
        gradHinPlaceholder, gradHPost, gradHRes, x, phi, alpha, bias, previousPre,
        hcBeforeNorm, invRms, sumOut, normOut, gradCurrentPre, mixedInput, rmsRstd,
        normWeight, directGradX, previousResidual, previousOutput, previousPost,
        previousResidualMix, runtimeConfig, allEventCounters, profileBuffer,
        gradResidual, gradPhi, gradAlpha, gradBias, gradPreviousOutput,
        gradPreviousPre, gradPreviousPost, gradPreviousResidual, gradNormWeight,
        userWorkspace, tiling,
    };
    worker_kernel(GetBlockIdx(), runtimeConfig, inputList);

    (void)MHC_GRAD_MODE;
}
