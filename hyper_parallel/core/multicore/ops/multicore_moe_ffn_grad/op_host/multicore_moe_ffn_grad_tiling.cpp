/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file multicore_moe_ffn.cpp
 */
#include "multicore_moe_ffn_grad_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include "tiling_base/tiling_key.h"
#include "tiling_base/tiling_base.h"
#include "../op_kernel/multicore_moe_ffn_grad_tiling_key.h"

namespace optiling {
const uint64_t BLOCK_SIZE = 32;
const uint64_t BUFFER_NUM = 2;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNumAic();
    context->SetBlockDim(coreNum);
    // OP_LOGE(context->GetNodeName(), "coreNum:%d", coreNum);

    auto input0Desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    if (input0DType == ge::DT_FLOAT16) {
        uint64_t tilingKey_ = GET_TPL_TILING_KEY(static_cast<uint32_t>(1),
                                                static_cast<uint32_t>(1),
                                                static_cast<uint32_t>(1),
                                                0UL,
                                                0UL,
                                                static_cast<uint32_t>(0),
                                                static_cast<uint32_t>(0),
                                                static_cast<uint32_t>(0),
                                                static_cast<uint32_t>(0),
                                                static_cast<uint32_t>(0));
        context->SetTilingKey(tilingKey_);
    } else if (input0DType == ge::DT_BF16) {
        uint64_t tilingKey_ = GET_TPL_TILING_KEY(static_cast<uint32_t>(27),
                                                static_cast<uint32_t>(27),
                                                static_cast<uint32_t>(27),
                                                0UL,
                                                0UL,
                                                static_cast<uint32_t>(0),
                                                static_cast<uint32_t>(0),
                                                static_cast<uint32_t>(0),
                                                static_cast<uint32_t>(0),
                                                static_cast<uint32_t>(0));
        context->SetTilingKey(tilingKey_);
    } else {
        OP_LOGE(context->GetNodeName(), "Input0[gradients]:%s is only support float16, bfloat16",
            Ops::Base::ToString(static_cast<ge::DataType>(input0DType)).c_str());
        return ge::GRAPH_FAILED;
    }

    MulticoreMoeFfnGradTilingData tiling;
    auto attr = context->GetAttrs();
    if (attr != nullptr) {
        tiling.set_rankId((*(attr->GetAttrPointer<int64_t>(0))));
        tiling.set_ep((*(attr->GetAttrPointer<int64_t>(1))));
        tiling.set_expertNum((*(attr->GetAttrPointer<int64_t>(2))));
        tiling.set_hiddenSize((*(attr->GetAttrPointer<int64_t>(3))));
        tiling.set_seqSize((*(attr->GetAttrPointer<int64_t>(4))));
        tiling.set_coreNum(static_cast<int64_t>(coreNum));
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 99615232;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareTilingFunc(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

struct MulticoreMoeFfnGradCompileInfo {
};

IMPL_OP_OPTILING(MulticoreMoeFfnGrad)
    .Tiling(TilingFunc)
    .TilingParse<MulticoreMoeFfnGradCompileInfo>(TilingPrepareTilingFunc);

}

