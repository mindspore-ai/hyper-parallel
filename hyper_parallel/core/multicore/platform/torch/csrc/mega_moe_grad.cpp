/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 *
 * PyTorch TORCH_LIBRARY_IMPL for hyper_parallel::mega_moe_grad.
 *
 * NPU implementation calls aclnnMegaMoeGrad via EXEC_NPU_CMD.
 * Meta implementation is a no-op (output tensors pre-allocated by caller).
 */
#include <torch/library.h>
#include <tuple>
#include "op_plugin/utils/op_api_common.h"

namespace {

using BwdReturn = std::tuple<
    at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
    at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>;

// ---------------------------------------------------------------------------
// NPU device implementation
// ---------------------------------------------------------------------------
BwdReturn mega_moe_grad_npu(
    at::Tensor& dispatch_target,              // pos  0 — inplace write, declared return
    const at::Tensor& dispatch_target_off,    // pos  1
    const at::Tensor& dy,                     // pos  2
    const at::Tensor& dispatch_src_off,       // pos  3
    const at::Tensor& dispatch_size,          // pos  4
    const at::Tensor& hidden,                 // pos  5
    at::Tensor& hidden_dw,                    // pos  6 — inplace write, declared return
    const at::Tensor& w2,                     // pos  7
    at::Tensor& act_grad_y,                   // pos  8 — inplace write, declared return
    const at::Tensor& gate,                   // pos  9
    at::Tensor& grad_gate,                    // pos 10 — inplace write, declared return
    const at::Tensor& w1,                     // pos 11
    at::Tensor& gate_dx,                      // pos 12 — inplace write, declared return
    at::Tensor& grad_x,                       // pos 13 — inplace write, declared return
    const at::Tensor& combine_target_off,     // pos 14
    const at::Tensor& combine_src_off,        // pos 15
    const at::Tensor& combine_size,           // pos 16
    at::Tensor& permute_out,                  // pos 17 — inplace write (not a declared return)
    at::Tensor& gate_dw,                      // pos 18 — inplace write, declared return
    const at::Tensor& group_list,             // pos 19
    const at::Tensor& act_grad_tiling,        // pos 20
    const at::Tensor& gate_grad_tiling,       // pos 21
    const at::Tensor& w1_grad_tiling,         // pos 22
    const at::Tensor& w2_grad_tiling,         // pos 23
    const at::Tensor& swiglu_grad_tiling,     // pos 24
    const at::Tensor& gmm_workspace,          // pos 25
    const at::Tensor& swiglu_grad_workspace,  // pos 26
    const at::Tensor& runtime_config,         // pos 27
    const at::Tensor& all_event_counters,     // pos 28
    int64_t rank_id,
    int64_t ep,
    int64_t expert_num,
    int64_t hidden_size,
    int64_t seq_size) {
    EXEC_NPU_CMD_EXT(aclnnMegaMoeGrad,
        dispatch_target, dispatch_target_off, dy, dispatch_src_off, dispatch_size,
        hidden, hidden_dw, w2, act_grad_y, gate, grad_gate, w1, gate_dx, grad_x,
        combine_target_off, combine_src_off, combine_size,
        permute_out, gate_dw, group_list,
        act_grad_tiling, gate_grad_tiling, w1_grad_tiling, w2_grad_tiling,
        swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace,
        runtime_config, all_event_counters,
        rank_id, ep, expert_num, hidden_size, seq_size);
    return BwdReturn(dispatch_target, hidden_dw, act_grad_y, grad_gate,
                     gate_dx, grad_x, permute_out, gate_dw);
}

// ---------------------------------------------------------------------------
// Meta device implementation
// ---------------------------------------------------------------------------
BwdReturn mega_moe_grad_meta(
    at::Tensor& dispatch_target,
    const at::Tensor& /*dispatch_target_off*/,
    const at::Tensor& /*dy*/,
    const at::Tensor& /*dispatch_src_off*/,
    const at::Tensor& /*dispatch_size*/,
    const at::Tensor& /*hidden*/,
    at::Tensor& hidden_dw,
    const at::Tensor& /*w2*/,
    at::Tensor& act_grad_y,
    const at::Tensor& /*gate*/,
    at::Tensor& grad_gate,
    const at::Tensor& /*w1*/,
    at::Tensor& gate_dx,
    at::Tensor& grad_x,
    const at::Tensor& /*combine_target_off*/,
    const at::Tensor& /*combine_src_off*/,
    const at::Tensor& /*combine_size*/,
    at::Tensor& permute_out,
    at::Tensor& gate_dw,
    const at::Tensor& /*group_list*/,
    const at::Tensor& /*act_grad_tiling*/,
    const at::Tensor& /*gate_grad_tiling*/,
    const at::Tensor& /*w1_grad_tiling*/,
    const at::Tensor& /*w2_grad_tiling*/,
    const at::Tensor& /*swiglu_grad_tiling*/,
    const at::Tensor& /*gmm_workspace*/,
    const at::Tensor& /*swiglu_grad_workspace*/,
    const at::Tensor& /*runtime_config*/,
    const at::Tensor& /*all_event_counters*/,
    int64_t /*rank_id*/,
    int64_t /*ep*/,
    int64_t /*expert_num*/,
    int64_t /*hidden_size*/,
    int64_t /*seq_size*/) {
    return BwdReturn(dispatch_target, hidden_dw, act_grad_y, grad_gate,
                     gate_dx, grad_x, permute_out, gate_dw);
}

}  // namespace

// NPU (Ascend) device
TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
    m.impl("mega_moe_grad", &mega_moe_grad_npu);
}

// Meta device
TORCH_LIBRARY_IMPL(hyper_parallel, Meta, m) {
    m.impl("mega_moe_grad", &mega_moe_grad_meta);
}
