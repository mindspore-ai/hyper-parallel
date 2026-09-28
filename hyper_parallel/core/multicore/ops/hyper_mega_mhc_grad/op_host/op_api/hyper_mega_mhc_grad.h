/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef LEVEL0_OP_HYPER_MEGA_MHC_GRAD_H_
#define LEVEL0_OP_HYPER_MEGA_MHC_GRAD_H_

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
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
    const aclTensor *gradNormWeight, double hcEps, aclOpExecutor *executor);
}  // namespace l0op
#endif  // LEVEL0_OP_HYPER_MEGA_MHC_GRAD_H_
