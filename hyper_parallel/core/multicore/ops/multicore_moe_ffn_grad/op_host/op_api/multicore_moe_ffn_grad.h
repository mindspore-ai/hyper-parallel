/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file multicore_moe_ffn_grad.h
 * \brief
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_MULTICORE_MOE_FFN_GRAD_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_MULTICORE_MOE_FFN_GRAD_H_

#include "opdev/op_executor.h"

namespace l0op {

const std::array<const aclTensor *, 7> MulticoreMoeFfnGrad(
    const aclTensor* dispatch_target,
    const aclTensor* dispatch_target_off,
    const aclTensor* dy,
    const aclTensor* dispatch_src_off,
    const aclTensor* dispatch_size,
    const aclTensor* hidden,
    const aclTensor* hidden_dw,
    const aclTensor* weight,         // w2
    const aclTensor* y,              // act_grad_y
    const aclTensor* gate,
    const aclTensor* grad_gate,
    const aclTensor* w1,
    const aclTensor* gate_dx,
    const aclTensor* grad_x,
    const aclTensor* combine_target_off,
    const aclTensor* combine_src_off,
    const aclTensor* combine_size,
    const aclTensor* permute_out,
    const aclTensor* gate_dw,
    const aclTensor* group_list,
    const aclTensor* act_grad_tiling,
    const aclTensor* gate_grad_tiling,
    const aclTensor* w2_grad_tiling,
    const aclTensor* w1_grad_tiling,
    const aclTensor* swiglu_grad_tiling,
    const aclTensor* gmm_workspace,
    const aclTensor* swiglu_grad_workspace,
    const aclTensor* runtime_config,
    const aclTensor* all_event_counters,
    int64_t rankId,
    int64_t ep,
    int64_t expert_num,
    int64_t hidden_size,
    int64_t seq_size,
    aclOpExecutor* executor);
}

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_MULTICORE_MOE_FFN_GRAD_H_
