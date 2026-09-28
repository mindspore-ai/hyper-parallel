/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "hyper_mega_mhc.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

namespace l0op {
OP_TYPE_REGISTER(HyperMegaMhc);

const std::array<const aclTensor *, 11> HyperMegaMhc(
  const aclTensor *residual, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
  const aclTensor *previousOutput, const aclTensor *previousPreMix, const aclTensor *previousPostMix,
  const aclTensor *previousResidualMix, const aclTensor *normWeight, const aclTensor *runtimeConfig,
  const aclTensor *allEventCounters, const aclTensor *profileBuffer, const aclTensor *newResidual,
  const aclTensor *nextPreMix, const aclTensor *nextPostMix, const aclTensor *nextResidualMix,
  const aclTensor *blockInput, const aclTensor *hcBeforeNorm, const aclTensor *invRms, const aclTensor *sumOut,
  const aclTensor *normOut, const aclTensor *mixedInput, const aclTensor *rmsRstd, int64_t hcMult, int64_t numIters,
  double hcEps, double normEps, bool needBackward, aclOpExecutor *executor) {
  L0_DFX(HyperMegaMhc, residual, phi, alpha, bias, previousOutput, previousPreMix, previousPostMix, previousResidualMix,
         normWeight, runtimeConfig, allEventCounters, profileBuffer, newResidual, nextPreMix, nextPostMix,
         nextResidualMix, blockInput, hcBeforeNorm, invRms, sumOut, normOut, mixedInput, rmsRstd, hcMult, numIters,
         hcEps, normEps, needBackward);
  auto *newResidualOut = const_cast<aclTensor *>(newResidual);
  auto *nextPreMixOut = const_cast<aclTensor *>(nextPreMix);
  auto *nextPostMixOut = const_cast<aclTensor *>(nextPostMix);
  auto *nextResidualMixOut = const_cast<aclTensor *>(nextResidualMix);
  auto *blockInputOut = const_cast<aclTensor *>(blockInput);
  auto *hcBeforeNormOut = const_cast<aclTensor *>(hcBeforeNorm);
  auto *invRmsOut = const_cast<aclTensor *>(invRms);
  auto *sumOutOut = const_cast<aclTensor *>(sumOut);
  auto *normOutOut = const_cast<aclTensor *>(normOut);
  auto *mixedInputOut = const_cast<aclTensor *>(mixedInput);
  auto *rmsRstdOut = const_cast<aclTensor *>(rmsRstd);
  ADD_TO_LAUNCHER_LIST_AICORE(
    HyperMegaMhc,
    OP_INPUT(residual, phi, alpha, bias, previousOutput, previousPreMix, previousPostMix, previousResidualMix,
             normWeight, runtimeConfig, allEventCounters, profileBuffer, newResidual, nextPreMix, nextPostMix,
             nextResidualMix, blockInput, hcBeforeNorm, invRms, sumOut, normOut, mixedInput, rmsRstd),
    OP_OUTPUT(newResidualOut, nextPreMixOut, nextPostMixOut, nextResidualMixOut, blockInputOut, hcBeforeNormOut,
              invRmsOut, sumOutOut, normOutOut, mixedInputOut, rmsRstdOut),
    OP_ATTR(hcMult, numIters, hcEps, normEps, needBackward));
  return {newResidualOut, nextPreMixOut, nextPostMixOut, nextResidualMixOut, blockInputOut, hcBeforeNormOut,
          invRmsOut,      sumOutOut,     normOutOut,     mixedInputOut,      rmsRstdOut};
}
}  // namespace l0op
