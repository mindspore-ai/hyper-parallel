/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "mhc_pre_sinkhorn_backward_tiling/arch22/mhc_pre_sinkhorn_backward_arch22_tiling.h"
#include "mhc_pre_sinkhorn_backward/arch22/mhc_pre_sinkhorn_backward_data_arch22.h"
#include "mhc_pre_sinkhorn_backward_tiling/mhc_pre_sinkhorn_backward_tiling.h"
#include "../op_kernel/hyper_mega_mhc_grad_tiling_key.h"
#include "register/op_def_registry.h"

namespace optiling {
namespace {
constexpr uint64_t ALIGN_BYTES = 512;
constexpr int64_t INPUT_X_INDEX = 3;
constexpr int64_t MAX_RMS_HIDDEN_SIZE = 5760;

uint64_t AlignUp(uint64_t value)
{
    return (value + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
}

ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    ge::graphStatus status = TilingMhcPreSinkhornBackwardArch22(context);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    const auto *xShape = context->GetInputShape(INPUT_X_INDEX);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &shape = xShape->GetStorageShape();
    if (shape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    uint64_t tokens = shape.GetDim(0) * shape.GetDim(1);
    uint64_t streams = shape.GetDim(2);
    uint64_t hiddenSize = shape.GetDim(3);
    if (hiddenSize > MAX_RMS_HIDDEN_SIZE || tokens < 40) {
        return ge::GRAPH_FAILED;
    }
    uint64_t hcMix = streams * streams + 2 * streams;
    uint64_t nativeUserWorkspace = tokens * hcMix * sizeof(float);
    nativeUserWorkspace += 2 * tokens * streams * hiddenSize * sizeof(float);
    uint64_t additional = AlignUp(nativeUserWorkspace) - nativeUserWorkspace;
    additional += AlignUp(tokens * hiddenSize * sizeof(uint16_t));
    additional += AlignUp(tokens * streams * hiddenSize * sizeof(uint16_t));
    additional += AlignUp(tokens * sizeof(float));
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] += additional;
    context->SetTilingKey(GET_TPL_TILING_KEY(HYPER_MEGA_MHC_GRAD_DEFAULT));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(HyperMegaMhcGrad)
    .Tiling(TilingFunc)
    .TilingParse<MhcPreSinkhornBackwardCompileInfo>(TilingPrepare);
}  // namespace optiling
