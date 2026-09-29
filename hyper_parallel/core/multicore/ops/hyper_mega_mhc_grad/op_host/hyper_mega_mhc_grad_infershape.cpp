/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <cstddef>
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t OUTPUT_COUNT = 9;
constexpr size_t OUTPUT_INPUT_SHAPE_MAP[OUTPUT_COUNT] = {
    17,  // grad_residual <- previous_residual
    4,   // grad_phi <- phi
    5,   // grad_alpha <- alpha
    6,   // grad_bias <- bias
    18,  // grad_previous_output <- previous_output
    7,   // grad_previous_pre <- previous_pre
    19,  // grad_previous_post <- previous_post
    20,  // grad_previous_residual <- previous_residual_mix
    15,  // grad_norm_weight <- norm_weight
};
}  // namespace

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t outputIndex = 0; outputIndex < OUTPUT_COUNT; ++outputIndex) {
        const gert::Shape *inputShape = context->GetInputShape(
            OUTPUT_INPUT_SHAPE_MAP[outputIndex]);
        gert::Shape *outputShape = context->GetOutputShape(outputIndex);
        if (inputShape == nullptr || outputShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *outputShape = *inputShape;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HyperMegaMhcGrad).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ops
