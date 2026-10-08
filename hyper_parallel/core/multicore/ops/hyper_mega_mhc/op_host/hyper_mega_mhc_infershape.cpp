/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

/*! \file hyper_mega_mhc_infershape.cpp */
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext *) { return ge::GRAPH_SUCCESS; }

static ge::graphStatus InferDataType(gert::InferDataTypeContext *) { return ge::GRAPH_SUCCESS; }

IMPL_OP_INFERSHAPE(HyperMegaMhc).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ops
