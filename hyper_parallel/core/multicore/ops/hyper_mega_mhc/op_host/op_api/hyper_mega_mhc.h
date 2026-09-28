/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_HYPER_MEGA_MHC_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_HYPER_MEGA_MHC_H_

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 11> HyperMegaMhc(
  const aclTensor *residual, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
  const aclTensor *previousOutput, const aclTensor *previousPreMix, const aclTensor *previousPostMix,
  const aclTensor *previousResidualMix, const aclTensor *normWeight, const aclTensor *runtimeConfig,
  const aclTensor *allEventCounters, const aclTensor *profileBuffer, const aclTensor *newResidual,
  const aclTensor *nextPreMix, const aclTensor *nextPostMix, const aclTensor *nextResidualMix,
  const aclTensor *blockInput, const aclTensor *hcBeforeNorm, const aclTensor *invRms, const aclTensor *sumOut,
  const aclTensor *normOut, const aclTensor *mixedInput, const aclTensor *rmsRstd, int64_t hcMult, int64_t numIters,
  double hcEps, double normEps, bool needBackward, aclOpExecutor *executor);
}  // namespace l0op
#endif
