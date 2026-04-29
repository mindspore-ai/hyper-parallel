/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 *
 * PyTorch out-of-tree operator registration for MoE-FFN operators.
 * Registers into the 'hyper_parallel' namespace — does NOT modify aten:: or op-plugin.
 *
 * After building (setup.py build_ext --inplace), import with:
 *   import hyper_parallel.core.multicore.platform.torch  # loads this .so
 * Or call directly after import:
 *   torch.ops.hyper_parallel.moe_ffn_fwd(...)
 *   torch.ops.hyper_parallel.moe_ffn_bwd(...)
 */
#include <torch/library.h>
#include <torch/extension.h>

TORCH_LIBRARY(hyper_parallel, m) {
    // -------------------------------------------------------------------------
    // moe_ffn_fwd: MoE-FFN forward operator (wraps aclnnMulticoreMoeFfn)
    //
    // Output buffers written in-place:
    //   dispatch_target, up_proj_y, swiglu_out, down_proj_y, combine_target.
    // All buffers must be pre-allocated by the caller.
    // -------------------------------------------------------------------------
    m.def(
        "moe_ffn_fwd("
        "  Tensor(a!) dispatch_target,"
        "  Tensor dispatch_target_off,"
        "  Tensor dispatch_src,"
        "  Tensor dispatch_src_off,"
        "  Tensor dispatch_size,"
        "  Tensor up_proj_weight,"
        "  Tensor up_proj_glist,"
        "  Tensor(b!) up_proj_y,"
        "  Tensor(c!) swiglu_out,"
        "  Tensor down_proj_weight,"
        "  Tensor down_proj_glist,"
        "  Tensor(d!) down_proj_y,"
        "  Tensor(e!) combine_target,"
        "  Tensor combine_target_off,"
        "  Tensor combine_src_off,"
        "  Tensor combine_size,"
        "  Tensor gmm_workspace,"
        "  Tensor up_proj_tiling,"
        "  Tensor swiglu_tiling,"
        "  Tensor down_proj_tiling,"
        "  Tensor runtime_config,"
        "  Tensor all_event_counters,"
        "  int rank_id,"
        "  int ep,"
        "  int expert_num,"
        "  int hidden_size,"
        "  int seq_size"
        ") -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!))");


    // -------------------------------------------------------------------------
    // moe_ffn_bwd: MoE-FFN backward operator (wraps aclnnMulticoreMoeFfnGrad)
    //
    // Output buffers written in-place:
    //   dispatch_target, hidden_dw, act_grad_y, grad_gate,
    //   gate_dx, grad_x, permute_out, gate_dw.
    // All buffers must be pre-allocated by the caller.
    // -------------------------------------------------------------------------
    m.def(
        "moe_ffn_bwd("
        "  Tensor(a!) dispatch_target,"
        "  Tensor dispatch_target_off,"
        "  Tensor dy,"
        "  Tensor dispatch_src_off,"
        "  Tensor dispatch_size,"
        "  Tensor hidden,"
        "  Tensor(b!) hidden_dw,"
        "  Tensor w2,"
        "  Tensor(c!) act_grad_y,"
        "  Tensor gate,"
        "  Tensor(d!) grad_gate,"
        "  Tensor w1,"
        "  Tensor(e!) gate_dx,"
        "  Tensor(f!) grad_x,"
        "  Tensor combine_target_off,"
        "  Tensor combine_src_off,"
        "  Tensor combine_size,"
        "  Tensor(g!) permute_out,"
        "  Tensor(h!) gate_dw,"
        "  Tensor group_list,"
        "  Tensor act_grad_tiling,"
        "  Tensor gate_grad_tiling,"
        "  Tensor w2_grad_tiling,"
        "  Tensor w1_grad_tiling,"
        "  Tensor swiglu_grad_tiling,"
        "  Tensor gmm_workspace,"
        "  Tensor swiglu_grad_workspace,"
        "  Tensor runtime_config,"
        "  Tensor all_event_counters,"
        "  int rank_id,"
        "  int ep,"
        "  int expert_num,"
        "  int hidden_size,"
        "  int seq_size"
        ") -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!), Tensor(f!), Tensor(g!), Tensor(h!))");
}

// Python module entry point — required for `import hyper_parallel_multicore_moe_ffn_pta`.
// Use an explicit name rather than TORCH_EXTENSION_NAME to avoid depending on
// the build system injecting -DTORCH_EXTENSION_NAME (not all NpuExtension
// versions guarantee this macro is defined).
// TORCH_LIBRARY_IMPL registrations in moe_ffn_fwd.cpp / moe_ffn_bwd.cpp fire
// via static initializers when the .so is loaded.
PYBIND11_MODULE(hyper_parallel_multicore_moe_ffn_pta, m) {}
