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
 * @file hyper_mega_moe_tiling.cpp
 */
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include "log/log.h"
#include "hyper_mega_moe_tiling.h"
#include "../op_kernel/hyper_mega_moe_tiling_key.h"

namespace optiling {
const uint64_t BLOCK_SIZE = 32;
const uint64_t BUFFER_NUM = 2;
constexpr int64_t MAX_EXPERT_NUM_PER_RANK = 16;
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  OP_CHECK_NULL_WITH_CONTEXT(context, context);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto coreNum = ascendcPlatform.GetCoreNumAic();
  if (coreNum == 0) {
    OP_LOGE(context->GetNodeName(), "The number of AIC cores must be positive.");
    return ge::GRAPH_FAILED;
  }
  context->SetBlockDim(coreNum);

  auto input0Desc = context->GetInputDesc(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, input0Desc);
  ge::DataType input0DType = input0Desc->GetDataType();
  if (input0DType == ge::DT_FLOAT16) {
    uint64_t tilingKey_ = GET_TPL_TILING_KEY(
      static_cast<uint32_t>(1), static_cast<uint32_t>(1), static_cast<uint32_t>(1), 0UL, 0UL, static_cast<uint32_t>(0),
      static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0));
    context->SetTilingKey(tilingKey_);
  } else if (input0DType == ge::DT_BF16) {
    uint64_t tilingKey_ =
      GET_TPL_TILING_KEY(static_cast<uint32_t>(27), static_cast<uint32_t>(27), static_cast<uint32_t>(27), 0UL, 0UL,
                         static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0),
                         static_cast<uint32_t>(0), static_cast<uint32_t>(0));
    context->SetTilingKey(tilingKey_);
  } else {
    OP_LOGE(context->GetNodeName(), "Input0[dispatch_target]:%s is only support float16, bfloat16",
            Ops::Base::ToString(static_cast<ge::DataType>(input0DType)).c_str());
    return ge::GRAPH_FAILED;
  }

  HyperMegaMoeTilingData tiling;
  auto attr = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attr);
  const int64_t *rankIdAttr = attr->GetAttrPointer<int64_t>(0);
  const int64_t *epAttr = attr->GetAttrPointer<int64_t>(1);
  const int64_t *expertNumAttr = attr->GetAttrPointer<int64_t>(2);
  const int64_t *hiddenSizeAttr = attr->GetAttrPointer<int64_t>(3);
  const int64_t *seqSizeAttr = attr->GetAttrPointer<int64_t>(4);
  OP_CHECK_NULL_WITH_CONTEXT(context, rankIdAttr);
  OP_CHECK_NULL_WITH_CONTEXT(context, epAttr);
  OP_CHECK_NULL_WITH_CONTEXT(context, expertNumAttr);
  OP_CHECK_NULL_WITH_CONTEXT(context, hiddenSizeAttr);
  OP_CHECK_NULL_WITH_CONTEXT(context, seqSizeAttr);

  const int64_t rankId = *rankIdAttr;
  const int64_t ep = *epAttr;
  const int64_t expertNum = *expertNumAttr;
  const int64_t hiddenSize = *hiddenSizeAttr;
  const int64_t seqSize = *seqSizeAttr;
  if (rankId < 0 || ep <= 0 || expertNum <= 0 || expertNum % ep != 0 ||
      expertNum / ep > MAX_EXPERT_NUM_PER_RANK || hiddenSize <= 0 || seqSize <= 0) {
    OP_LOGE(context->GetNodeName(),
            "Invalid topology: rankId=%ld, ep=%ld, expertNum=%ld, hiddenSize=%ld, seqSize=%ld.", rankId, ep,
            expertNum, hiddenSize, seqSize);
    return ge::GRAPH_FAILED;
  }
  tiling.set_rankId(rankId);
  tiling.set_ep(ep);
  tiling.set_expertNum(expertNum);
  tiling.set_hiddenSize(hiddenSize);
  tiling.set_seqSize(seqSize);
  tiling.set_coreNum(static_cast<int64_t>(coreNum));

  auto rawTilingData = context->GetRawTilingData();
  OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
  tiling.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
  rawTilingData->SetDataSize(tiling.GetDataSize());

  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
  currentWorkspace[0] = 95420928;
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareTilingFunc(gert::TilingParseContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

struct HyperMegaMoeCompileInfo {};

IMPL_OP_OPTILING(HyperMegaMoe).Tiling(TilingFunc).TilingParse<HyperMegaMoeCompileInfo>(TilingPrepareTilingFunc);

}  // namespace optiling
