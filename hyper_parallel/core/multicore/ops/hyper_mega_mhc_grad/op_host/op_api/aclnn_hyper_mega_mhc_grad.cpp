/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <algorithm>
#include "aclnn_hyper_mega_mhc_grad.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "hyper_mega_mhc_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnHyperMegaMhcGradGetWorkspaceSize(
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
    aclTensor *gradNormWeight, double hcEps, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(
        aclnnHyperMegaMhcGrad,
        DFX_IN(
            gradHinPlaceholder, gradHPost, gradHRes, x, phi, alpha, bias, previousPre,
            hcBeforeNorm, invRms, sumOut, normOut, gradCurrentPre, mixedInput, rmsRstd,
            normWeight, directGradX, previousResidual, previousOutput, previousPost,
            previousResidualMix, runtimeConfig, allEventCounters, profileBuffer, hcEps),
        DFX_OUT(
            gradResidual, gradPhi, gradAlpha, gradBias, gradPreviousOutput, gradPreviousPre,
            gradPreviousPost, gradPreviousResidual, gradNormWeight));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

#define MAKE_CONTIGUOUS(name) \
    const aclTensor *name##Contiguous = l0op::Contiguous(name, uniqueExecutor.get()); \
    CHECK_RET(name##Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR)
    MAKE_CONTIGUOUS(gradHinPlaceholder);
    MAKE_CONTIGUOUS(gradHPost);
    MAKE_CONTIGUOUS(gradHRes);
    MAKE_CONTIGUOUS(x);
    MAKE_CONTIGUOUS(phi);
    MAKE_CONTIGUOUS(alpha);
    MAKE_CONTIGUOUS(bias);
    MAKE_CONTIGUOUS(previousPre);
    MAKE_CONTIGUOUS(hcBeforeNorm);
    MAKE_CONTIGUOUS(invRms);
    MAKE_CONTIGUOUS(sumOut);
    MAKE_CONTIGUOUS(normOut);
    MAKE_CONTIGUOUS(gradCurrentPre);
    MAKE_CONTIGUOUS(mixedInput);
    MAKE_CONTIGUOUS(rmsRstd);
    MAKE_CONTIGUOUS(normWeight);
    MAKE_CONTIGUOUS(directGradX);
    MAKE_CONTIGUOUS(previousResidual);
    MAKE_CONTIGUOUS(previousOutput);
    MAKE_CONTIGUOUS(previousPost);
    MAKE_CONTIGUOUS(previousResidualMix);
    MAKE_CONTIGUOUS(runtimeConfig);
    MAKE_CONTIGUOUS(allEventCounters);
    MAKE_CONTIGUOUS(profileBuffer);
#undef MAKE_CONTIGUOUS

    const auto outputs = l0op::HyperMegaMhcGrad(
        gradHinPlaceholderContiguous, gradHPostContiguous, gradHResContiguous, xContiguous,
        phiContiguous, alphaContiguous, biasContiguous, previousPreContiguous,
        hcBeforeNormContiguous, invRmsContiguous, sumOutContiguous, normOutContiguous,
        gradCurrentPreContiguous, mixedInputContiguous, rmsRstdContiguous, normWeightContiguous,
        directGradXContiguous, previousResidualContiguous, previousOutputContiguous,
        previousPostContiguous, previousResidualMixContiguous, runtimeConfigContiguous,
        allEventCountersContiguous, profileBufferContiguous, gradResidual, gradPhi, gradAlpha,
        gradBias, gradPreviousOutput, gradPreviousPre, gradPreviousPost, gradPreviousResidual,
        gradNormWeight, hcEps, uniqueExecutor.get());
    CHECK_RET(
        std::all_of(outputs.begin(), outputs.end(), [](const aclTensor *value) { return value != nullptr; }),
        ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHyperMegaMhcGrad(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnHyperMegaMhcGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
