/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "hyper_mega_mhc_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

namespace l0op {
OP_TYPE_REGISTER(HyperMegaMhcGrad);

const std::array<const aclTensor *, 9> HyperMegaMhcGrad(
    const aclTensor *gradHinPlaceholder, const aclTensor *gradHPost, const aclTensor *gradHRes,
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
    const aclTensor *previousPre, const aclTensor *hcBeforeNorm, const aclTensor *invRms,
    const aclTensor *sumOut, const aclTensor *normOut, const aclTensor *gradCurrentPre,
    const aclTensor *mixedInput, const aclTensor *rmsRstd, const aclTensor *normWeight,
    const aclTensor *directGradX, const aclTensor *previousResidual, const aclTensor *previousOutput,
    const aclTensor *previousPost, const aclTensor *previousResidualMix, const aclTensor *runtimeConfig,
    const aclTensor *allEventCounters, const aclTensor *profileBuffer, const aclTensor *gradResidual,
    const aclTensor *gradPhi, const aclTensor *gradAlpha, const aclTensor *gradBias,
    const aclTensor *gradPreviousOutput, const aclTensor *gradPreviousPre,
    const aclTensor *gradPreviousPost, const aclTensor *gradPreviousResidual,
    const aclTensor *gradNormWeight, double hcEps, aclOpExecutor *executor)
{
    L0_DFX(
        HyperMegaMhcGrad, gradHinPlaceholder, gradHPost, gradHRes, x, phi, alpha, bias,
        previousPre, hcBeforeNorm, invRms, sumOut, normOut, gradCurrentPre, mixedInput,
        rmsRstd, normWeight, directGradX, previousResidual, previousOutput, previousPost,
        previousResidualMix, runtimeConfig, allEventCounters, profileBuffer, gradResidual,
        gradPhi, gradAlpha, gradBias, gradPreviousOutput, gradPreviousPre, gradPreviousPost,
        gradPreviousResidual, gradNormWeight, hcEps);
    auto *gradResidualOut = const_cast<aclTensor *>(gradResidual);
    auto *gradPhiOut = const_cast<aclTensor *>(gradPhi);
    auto *gradAlphaOut = const_cast<aclTensor *>(gradAlpha);
    auto *gradBiasOut = const_cast<aclTensor *>(gradBias);
    auto *gradPreviousOutputOut = const_cast<aclTensor *>(gradPreviousOutput);
    auto *gradPreviousPreOut = const_cast<aclTensor *>(gradPreviousPre);
    auto *gradPreviousPostOut = const_cast<aclTensor *>(gradPreviousPost);
    auto *gradPreviousResidualOut = const_cast<aclTensor *>(gradPreviousResidual);
    auto *gradNormWeightOut = const_cast<aclTensor *>(gradNormWeight);
    ADD_TO_LAUNCHER_LIST_AICORE(
        HyperMegaMhcGrad,
        OP_INPUT(
            gradHinPlaceholder, gradHPost, gradHRes, x, phi, alpha, bias, previousPre,
            hcBeforeNorm, invRms, sumOut, normOut, gradCurrentPre, mixedInput, rmsRstd,
            normWeight, directGradX, previousResidual, previousOutput, previousPost,
            previousResidualMix, runtimeConfig, allEventCounters, profileBuffer),
        OP_OUTPUT(
            gradResidualOut, gradPhiOut, gradAlphaOut, gradBiasOut, gradPreviousOutputOut,
            gradPreviousPreOut, gradPreviousPostOut, gradPreviousResidualOut, gradNormWeightOut),
        OP_ATTR(static_cast<float>(hcEps)));
    return {
        gradResidualOut, gradPhiOut, gradAlphaOut, gradBiasOut, gradPreviousOutputOut,
        gradPreviousPreOut, gradPreviousPostOut, gradPreviousResidualOut, gradNormWeightOut};
}
}  // namespace l0op
