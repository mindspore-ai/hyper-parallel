/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef ACLNN_HYPER_MEGA_MHC_GRAD_H_
#define ACLNN_HYPER_MEGA_MHC_GRAD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
ACLNN_API aclnnStatus aclnnHyperMegaMhcGradGetWorkspaceSize(
    const aclTensor *gradHinPlaceholder, const aclTensor *gradHPost, const aclTensor *gradHRes,
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
    const aclTensor *previousPre, const aclTensor *hcBeforeNorm, const aclTensor *invRms,
    const aclTensor *sumOut, const aclTensor *normOut, const aclTensor *gradCurrentPre,
    const aclTensor *mixedInput, const aclTensor *rmsRstd, const aclTensor *normWeight,
    const aclTensor *directGradX, const aclTensor *previousResidual, const aclTensor *previousOutput,
    const aclTensor *previousPost, const aclTensor *previousResidualMix, const aclTensor *runtimeConfig,
    const aclTensor *allEventCounters, const aclTensor *profileBuffer, aclTensor *gradResidual,
    aclTensor *gradPhi, aclTensor *gradAlpha, aclTensor *gradBias, aclTensor *gradPreviousOutput,
    aclTensor *gradPreviousPre, aclTensor *gradPreviousPost, aclTensor *gradPreviousResidual,
    aclTensor *gradNormWeight, double hcEps, uint64_t *workspaceSize, aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnHyperMegaMhcGrad(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif  // ACLNN_HYPER_MEGA_MHC_GRAD_H_
