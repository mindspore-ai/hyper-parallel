/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

/*! \file hyper_mega_mhc.cpp */
#include "kernel_operator.h"
#include "hyper_mega_mhc_tiling_key.h"
#include "worker_kernel.cpp"

using namespace AscendC;

template <int8_t MHC_SPLIT_MODE>
__global__ __aicore__ void hyper_mega_mhc(GM_ADDR residual, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                          GM_ADDR previousOutput, GM_ADDR previousPreMix, GM_ADDR previousPostMix,
                                          GM_ADDR previousResidualMix, GM_ADDR normWeight, GM_ADDR runtimeConfig,
                                          GM_ADDR allEventCounters, GM_ADDR profileBuffer, GM_ADDR newResidualOut,
                                          GM_ADDR nextPreMixOut, GM_ADDR nextPostMixOut, GM_ADDR nextResidualMixOut,
                                          GM_ADDR blockInputOut, GM_ADDR newResidual, GM_ADDR nextPreMix,
                                          GM_ADDR nextPostMix, GM_ADDR nextResidualMix, GM_ADDR blockInput,
                                          GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  if (workspace == nullptr || runtimeConfig == nullptr || allEventCounters == nullptr) {
    return;
  }
  GM_ADDR userWorkspace = GetUserWorkspace(workspace);
  if (userWorkspace == nullptr) {
    return;
  }
  GM_ADDR inputList[] = {
    previousOutput, residual,   previousPreMix, previousPostMix, previousResidualMix, phi,
    alpha,          bias,       normWeight,     runtimeConfig,   allEventCounters,    profileBuffer,
    newResidual,    nextPreMix, nextPostMix,    nextResidualMix, blockInput,          userWorkspace,
    tiling,
  };
  worker_kernel(GetBlockIdx(), runtimeConfig, inputList);

  (void)MHC_SPLIT_MODE;
  (void)newResidualOut;
  (void)nextPreMixOut;
  (void)nextPostMixOut;
  (void)nextResidualMixOut;
  (void)blockInputOut;
}
