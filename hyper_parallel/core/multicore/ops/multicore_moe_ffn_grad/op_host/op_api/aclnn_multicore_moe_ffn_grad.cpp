/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_multicore_moe_ffn_grad.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "multicore_moe_ffn_grad.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_multicore_moe_ffn_grad.h"

#define RT_MEMORY_POLICY_HUGE_PAGE_FIRST_P2P (0x2000u)
#define RT_MEMORY_POLICY_HUGE_PAGE_ONLY_P2P (0x4000u)
#define RT_MEMORY_POLICY_DEFAULT_PAGE_ONLY_P2P (0x8000u)
#define RT_MEMORY_POLICY_HUGE1G_PAGE_ONLY (0x10000u)
#define RT_MEMORY_POLICY_HUGE1G_PAGE_ONLY_P2P (0x20000u)

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

aclTensor* CreateContiguousTensorList(const aclTensor *tensorList, aclOpExecutor *executor)
{
    op::Shape shape;
    const aclTensor *inputTensor = tensorList;
    op::Shape viewShape = inputTensor->GetViewShape();
    uint32_t viewShapeDimsNum = viewShape.GetDimNum();
    shape.SetScalar();
    // 2: the second last dimension; in for-loops, it indicates dimensions before the second last remain unchanged.
    for (uint32_t i = 0; i < viewShapeDimsNum - 2; ++i) {
        shape.AppendDim(viewShape.GetDim(i));
    }
    // viewShapeDimsNum - 1, the dim value of the last dim. viewShapeDimsNum - 2, the dim value of the second last
    // dim.
    shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 1));
    shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 2)); // 2:the second last dim.
    aclTensor *tensor =
        executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset()); // use executor to create tensor
    tensor->SetStorageFormat(inputTensor->GetStorageFormat());
    return tensor;
}

static void SetTransposedTensorListContiguous(MulticoreMoeFfnGradParams &params, aclOpExecutor *executorPtr)
{
    aclTensor* hidden = CreateContiguousTensorList(params.hidden, executorPtr);
    params.hidden = hidden;

    aclTensor* w1 = CreateContiguousTensorList(params.w1, executorPtr);
    params.w1 = w1;

    aclTensor* permute_out = CreateContiguousTensorList(params.permute_out, executorPtr);
    params.permute_out = permute_out;

    aclTensor* weight = CreateContiguousTensorList(params.weight, executorPtr);
    params.weight = weight;
}

bool IsTransposeLastTwoDims(const aclTensor *tensor)
{
    auto shape = tensor->GetViewShape();
    int64_t dim1 = shape.GetDimNum() - 1;
    int64_t dim2 = shape.GetDimNum() - 2;
    auto strides = tensor->GetViewStrides();
    if (strides[dim2] == 1 && strides[dim1] == shape.GetDim(dim2)) {
        int64_t tmpNxD = shape.GetDim(dim1) * shape.GetDim(dim2);
        for (int64_t batchDim = shape.GetDimNum() - 3; batchDim >= 0; batchDim--) {
            if (strides[batchDim] != tmpNxD) {
                return false;
            }
            tmpNxD *= shape.GetDim(batchDim);
        }
        return true;
    }
    return false;
}

ACLNN_API aclnnStatus aclnnMulticoreMoeFfnGradGetWorkspaceSize(
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
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMulticoreMoeFfnGrad,
                   DFX_IN(dispatch_target, dispatch_target_off, dy, dispatch_src_off, dispatch_size,
                          hidden, hidden_dw, weight, y, gate, grad_gate, w1,
                          gate_dx, grad_x, combine_target_off, combine_src_off, combine_size,
                          permute_out, gate_dw, group_list,
                          act_grad_tiling, gate_grad_tiling, w2_grad_tiling, w1_grad_tiling,
                          swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace,
                          runtime_config, all_event_counters, rankId, ep, expert_num, hidden_size, seq_size),
                   DFX_OUT(dispatch_target, hidden_dw, y, grad_gate, gate_dx, grad_x, gate_dw));

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor *executorPtr = uniqueExecutor.get();
    CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    MulticoreMoeFfnGradParams params{hidden, w1, permute_out, weight};
    SetTransposedTensorListContiguous(params, executorPtr);

    auto dispatch_target_cont = l0op::Contiguous(dispatch_target, executorPtr);
    auto dispatch_target_off_cont = l0op::Contiguous(dispatch_target_off, executorPtr);
    auto dy_cont = l0op::Contiguous(dy, executorPtr);
    auto dispatch_src_off_cont = l0op::Contiguous(dispatch_src_off, executorPtr);
    auto dispatch_size_cont = l0op::Contiguous(dispatch_size, executorPtr);
    auto hidden_cont = l0op::Contiguous(params.hidden, executorPtr);
    auto hidden_dw_cont = l0op::Contiguous(hidden_dw, executorPtr);
    auto weight_cont = l0op::Contiguous(params.weight, executorPtr);
    auto y_cont = l0op::Contiguous(y, executorPtr);
    auto gate_cont = l0op::Contiguous(gate, executorPtr);
    auto grad_gate_cont = l0op::Contiguous(grad_gate, executorPtr);
    auto w1_cont = l0op::Contiguous(params.w1, executorPtr);
    auto gate_dx_cont = l0op::Contiguous(gate_dx, executorPtr);
    auto grad_x_cont = l0op::Contiguous(grad_x, executorPtr);
    auto combine_target_off_cont = l0op::Contiguous(combine_target_off, executorPtr);
    auto combine_src_off_cont = l0op::Contiguous(combine_src_off, executorPtr);
    auto combine_size_cont = l0op::Contiguous(combine_size, executorPtr);
    auto permute_out_cont = l0op::Contiguous(params.permute_out, executorPtr);
    auto gate_dw_cont = l0op::Contiguous(gate_dw, executorPtr);
    auto group_list_cont = l0op::Contiguous(group_list, executorPtr);
    auto act_grad_tiling_cont = l0op::Contiguous(act_grad_tiling, executorPtr);
    auto gate_grad_tiling_cont = l0op::Contiguous(gate_grad_tiling, executorPtr);
    auto w2_grad_tiling_cont = l0op::Contiguous(w2_grad_tiling, executorPtr);
    auto w1_grad_tiling_cont = l0op::Contiguous(w1_grad_tiling, executorPtr);
    auto swiglu_grad_tiling_cont = l0op::Contiguous(swiglu_grad_tiling, executorPtr);
    auto gmm_workspace_cont = l0op::Contiguous(gmm_workspace, executorPtr);
    auto swiglu_grad_workspace_cont = l0op::Contiguous(swiglu_grad_workspace, executorPtr);
    auto runtime_config_cont = l0op::Contiguous(runtime_config, executorPtr);
    auto all_event_counters_cont = l0op::Contiguous(all_event_counters, executorPtr);
    CHECK_RET(dispatch_target_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dispatch_target_off_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dy_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dispatch_src_off_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dispatch_size_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(hidden_dw_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(y_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gate_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(grad_gate_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gate_dx_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(grad_x_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(combine_target_off_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(combine_src_off_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(combine_size_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gate_dw_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(group_list_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(act_grad_tiling_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gate_grad_tiling_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(w2_grad_tiling_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(w1_grad_tiling_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(swiglu_grad_tiling_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gmm_workspace_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(swiglu_grad_workspace_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(runtime_config_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(all_event_counters_cont != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outTensor = l0op::MulticoreMoeFfnGrad(
        dispatch_target, dispatch_target_off, dy, dispatch_src_off, dispatch_size,
        params.hidden, hidden_dw, params.weight, y, gate, grad_gate, params.w1,
        gate_dx, grad_x, combine_target_off, combine_src_off, combine_size,
        params.permute_out, gate_dw, group_list,
        act_grad_tiling, gate_grad_tiling, w2_grad_tiling, w1_grad_tiling, swiglu_grad_tiling,
        gmm_workspace, swiglu_grad_workspace, runtime_config, all_event_counters,
        rankId, ep, expert_num, hidden_size, seq_size,
        executorPtr);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMulticoreMoeFfnGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMulticoreMoeFfnGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
