/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 *
 * PyTorch TORCH_LIBRARY_IMPL for hyper_parallel::mega_moe.
 *
 * NPU implementation calls aclnnMegaMoe via EXEC_NPU_CMD macro
 * (from op_plugin/utils/op_api_common.h), which handles:
 *   1. at::Tensor -> aclTensor conversion
 *   2. aclnnMegaMoeGetWorkspaceSize() call + workspace alloc
 *   3. aclnnMegaMoe(workspace, size, executor, stream) call
 *
 * Meta implementation is a no-op (all output tensors are pre-allocated by caller).
 */
#include <torch/library.h>
#include <tuple>
#include "op_plugin/utils/op_api_common.h"

namespace {

// ---------------------------------------------------------------------------
// NPU device implementation
// ---------------------------------------------------------------------------
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
mega_moe_npu(
    at::Tensor& dispatch_target,           // pos  0 — inplace write, declared return
    const at::Tensor& dispatch_target_off,  // pos  1
    const at::Tensor& dispatch_src,        // pos  2
    const at::Tensor& dispatch_src_off,    // pos  3
    const at::Tensor& dispatch_size,       // pos  4
    const at::Tensor& up_proj_weight,      // pos  5
    const at::Tensor& up_proj_glist,       // pos  6
    at::Tensor& up_proj_y,                 // pos  7 — inplace write, declared return
    at::Tensor& swiglu_out,                // pos  8 — inplace write, declared return
    const at::Tensor& down_proj_weight,    // pos  9
    const at::Tensor& down_proj_glist,     // pos 10
    at::Tensor& down_proj_y,               // pos 11 — inplace write, declared return
    at::Tensor& combine_target,            // pos 12 — inplace write, declared return
    const at::Tensor& combine_target_off,  // pos 13
    const at::Tensor& combine_src_off,     // pos 14
    const at::Tensor& combine_size,        // pos 15
    const at::Tensor& gmm_workspace,       // pos 16
    const at::Tensor& up_proj_tiling,      // pos 17
    const at::Tensor& swiglu_tiling,       // pos 18
    const at::Tensor& down_proj_tiling,    // pos 19
    const at::Tensor& runtime_config,      // pos 20
    const at::Tensor& all_event_counters,  // pos 21
    int64_t rank_id,
    int64_t ep,
    int64_t expert_num,
    int64_t hidden_size,
    int64_t seq_size) {
    EXEC_NPU_CMD_EXT(aclnnMegaMoe,
        dispatch_target, dispatch_target_off,
        dispatch_src, dispatch_src_off, dispatch_size,
        up_proj_weight, up_proj_glist,
        up_proj_y, swiglu_out,
        down_proj_weight, down_proj_glist, down_proj_y,
        combine_target, combine_target_off, combine_src_off, combine_size,
        gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling,
        runtime_config, all_event_counters,
        rank_id, ep, expert_num, hidden_size, seq_size);
    return std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>(
        dispatch_target, up_proj_y, swiglu_out, down_proj_y, combine_target);
}

// ---------------------------------------------------------------------------
// Meta device implementation (shape inference only, no compute)
// All output buffers are pre-allocated by caller — nothing to do here.
// ---------------------------------------------------------------------------
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
mega_moe_meta(
    at::Tensor& dispatch_target,
    const at::Tensor& /*dispatch_target_off*/,
    const at::Tensor& /*dispatch_src*/,
    const at::Tensor& /*dispatch_src_off*/,
    const at::Tensor& /*dispatch_size*/,
    const at::Tensor& /*up_proj_weight*/,
    const at::Tensor& /*up_proj_glist*/,
    at::Tensor& up_proj_y,
    at::Tensor& swiglu_out,
    const at::Tensor& /*down_proj_weight*/,
    const at::Tensor& /*down_proj_glist*/,
    at::Tensor& down_proj_y,
    at::Tensor& combine_target,
    const at::Tensor& /*combine_target_off*/,
    const at::Tensor& /*combine_src_off*/,
    const at::Tensor& /*combine_size*/,
    const at::Tensor& /*gmm_workspace*/,
    const at::Tensor& /*up_proj_tiling*/,
    const at::Tensor& /*swiglu_tiling*/,
    const at::Tensor& /*down_proj_tiling*/,
    const at::Tensor& /*runtime_config*/,
    const at::Tensor& /*all_event_counters*/,
    int64_t /*rank_id*/,
    int64_t /*ep*/,
    int64_t /*expert_num*/,
    int64_t /*hidden_size*/,
    int64_t /*seq_size*/) {
    return std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>(
        dispatch_target, up_proj_y, swiglu_out, down_proj_y, combine_target);
}

}  // namespace

// NPU (Ascend) device
TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
    m.impl("mega_moe", &mega_moe_npu);
}

// Meta device (for shape tracing without actual compute)
TORCH_LIBRARY_IMPL(hyper_parallel, Meta, m) {
    m.impl("mega_moe", &mega_moe_meta);
}
