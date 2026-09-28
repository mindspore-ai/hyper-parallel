/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <algorithm>
#include "aclnn_hyper_mega_mhc.h"
#include "aclnn_kernels/contiguous.h"
#include "hyper_mega_mhc.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnHyperMegaMhcGetWorkspaceSize(
  const aclTensor *previousOutput, const aclTensor *residual, const aclTensor *previousPreMix,
  const aclTensor *previousPostMix, const aclTensor *previousResidualMix, const aclTensor *phi, const aclTensor *alpha,
  const aclTensor *bias, const aclTensor *normWeight, const aclTensor *runtimeConfig, const aclTensor *allEventCounters,
  const aclTensor *profileBuffer, aclTensor *newResidual, aclTensor *nextPreMix, aclTensor *nextPostMix,
  aclTensor *nextResidualMix, aclTensor *blockInput, aclTensor *hcBeforeNorm, aclTensor *invRms, aclTensor *sumOut,
  aclTensor *normOut, aclTensor *mixedInput, aclTensor *rmsRstd, double hcEps, double normEps, int64_t numIters,
  bool needBackward, uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(
    aclnnHyperMegaMhc,
    DFX_IN(previousOutput, residual, previousPreMix, previousPostMix, previousResidualMix, phi, alpha, bias, normWeight,
           runtimeConfig, allEventCounters, profileBuffer, hcEps, normEps, numIters, needBackward),
    DFX_OUT(newResidual, nextPreMix, nextPostMix, nextResidualMix, blockInput, hcBeforeNorm, invRms, sumOut, normOut,
            mixedInput, rmsRstd));
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

#define MAKE_CONTIGUOUS(name)                                                       \
  const aclTensor *name##Contiguous = l0op::Contiguous(name, uniqueExecutor.get()); \
  CHECK_RET(name##Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR)

  MAKE_CONTIGUOUS(previousOutput);
  MAKE_CONTIGUOUS(residual);
  MAKE_CONTIGUOUS(previousPreMix);
  MAKE_CONTIGUOUS(previousPostMix);
  MAKE_CONTIGUOUS(previousResidualMix);
  MAKE_CONTIGUOUS(phi);
  MAKE_CONTIGUOUS(alpha);
  MAKE_CONTIGUOUS(bias);
  MAKE_CONTIGUOUS(normWeight);
  MAKE_CONTIGUOUS(runtimeConfig);
  MAKE_CONTIGUOUS(allEventCounters);
  MAKE_CONTIGUOUS(profileBuffer);

#undef MAKE_CONTIGUOUS

  const auto outputs = l0op::HyperMegaMhc(
    residualContiguous, phiContiguous, alphaContiguous, biasContiguous, previousOutputContiguous,
    previousPreMixContiguous, previousPostMixContiguous, previousResidualMixContiguous, normWeightContiguous,
    runtimeConfigContiguous, allEventCountersContiguous, profileBufferContiguous, newResidual, nextPreMix, nextPostMix,
    nextResidualMix, blockInput, hcBeforeNorm, invRms, sumOut, normOut, mixedInput, rmsRstd, 4, numIters, hcEps,
    normEps, needBackward, uniqueExecutor.get());
  CHECK_RET(std::all_of(outputs.begin(), outputs.end(), [](const aclTensor *value) { return value != nullptr; }),
            ACLNN_ERR_INNER_NULLPTR);
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnHyperMegaMhc(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnHyperMegaMhc);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
