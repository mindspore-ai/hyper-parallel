/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef OP_API_INC_HYPER_MEGA_MHC_H_
#define OP_API_INC_HYPER_MEGA_MHC_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
ACLNN_API aclnnStatus aclnnHyperMegaMhcGetWorkspaceSize(
  const aclTensor *previousOutput, const aclTensor *residual, const aclTensor *previousPreMix,
  const aclTensor *previousPostMix, const aclTensor *previousResidualMix, const aclTensor *phi, const aclTensor *alpha,
  const aclTensor *bias, const aclTensor *normWeight, const aclTensor *runtimeConfig, const aclTensor *allEventCounters,
  const aclTensor *profileBuffer, aclTensor *newResidual, aclTensor *nextPreMix, aclTensor *nextPostMix,
  aclTensor *nextResidualMix, aclTensor *blockInput, double hcEps, double normEps, int64_t numIters, bool needBackward,
  uint64_t *workspaceSize, aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnHyperMegaMhc(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif
