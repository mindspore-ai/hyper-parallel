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

const std::array<const aclTensor *, 5> HyperMegaMhc(
  const aclTensor *residual, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
  const aclTensor *previousOutput, const aclTensor *previousPreMix, const aclTensor *previousPostMix,
  const aclTensor *previousResidualMix, const aclTensor *normWeight, const aclTensor *runtimeConfig,
  const aclTensor *allEventCounters, const aclTensor *profileBuffer, const aclTensor *newResidual,
  const aclTensor *nextPreMix, const aclTensor *nextPostMix, const aclTensor *nextResidualMix,
  const aclTensor *blockInput, int64_t hcMult, int64_t numIters, double hcEps, double normEps, bool needBackward,
  aclOpExecutor *executor) {
  L0_DFX(HyperMegaMhc, residual, phi, alpha, bias, previousOutput, previousPreMix, previousPostMix, previousResidualMix,
         normWeight, runtimeConfig, allEventCounters, profileBuffer, newResidual, nextPreMix, nextPostMix,
         nextResidualMix, blockInput, hcMult, numIters, hcEps, normEps, needBackward);
  auto *newResidualOut = const_cast<aclTensor *>(newResidual);
  auto *nextPreMixOut = const_cast<aclTensor *>(nextPreMix);
  auto *nextPostMixOut = const_cast<aclTensor *>(nextPostMix);
  auto *nextResidualMixOut = const_cast<aclTensor *>(nextResidualMix);
  auto *blockInputOut = const_cast<aclTensor *>(blockInput);
  ADD_TO_LAUNCHER_LIST_AICORE(
    HyperMegaMhc,
    OP_INPUT(residual, phi, alpha, bias, previousOutput, previousPreMix, previousPostMix, previousResidualMix,
             normWeight, runtimeConfig, allEventCounters, profileBuffer, newResidual, nextPreMix, nextPostMix,
             nextResidualMix, blockInput),
    OP_OUTPUT(newResidualOut, nextPreMixOut, nextPostMixOut, nextResidualMixOut, blockInputOut),
    OP_ATTR(hcMult, numIters, hcEps, normEps, needBackward));
  return {newResidualOut, nextPreMixOut, nextPostMixOut, nextResidualMixOut, blockInputOut};
}
}  // namespace l0op
