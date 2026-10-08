/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * PyTorch out-of-tree schemas for shifted Single-Pass mHC.
 */
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(hyper_parallel, m) {
  m.def(
    "cann_mhc_pre_sinkhorn("
    "  Tensor x, Tensor phi, Tensor alpha, Tensor bias,"
    "  Tensor(a!) hin, Tensor(b!) h_post, Tensor(c!) h_res, Tensor(d!) h_pre,"
    "  Tensor(e!) hc_before_norm, Tensor(f!) inv_rms,"
    "  Tensor(g!) sum_out, Tensor(h!) norm_out,"
    "  float hc_eps, float norm_eps, int num_iters"
    ") -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!),"
    " Tensor(e!), Tensor(f!), Tensor(g!), Tensor(h!))");

  m.def(
    "mega_mhc("
    "  Tensor previous_output, Tensor residual, Tensor previous_pre_mix,"
    "  Tensor previous_post_mix, Tensor previous_residual_mix,"
    "  Tensor phi, Tensor alpha, Tensor bias, Tensor norm_weight,"
    "  Tensor runtime_config, Tensor all_event_counters, Tensor profile_buffer,"
    "  Tensor(a!) new_residual, Tensor(b!) next_pre_mix,"
    "  Tensor(c!) next_post_mix, Tensor(d!) next_residual_mix,"
    "  Tensor(e!) block_input,"
    "  Tensor(f!) hc_before_norm, Tensor(g!) inv_rms,"
    "  Tensor(h!) sum_out, Tensor(i!) norm_out,"
    "  Tensor(j!) mixed_input, Tensor(k!) rms_rstd,"
    "  float hc_eps, float norm_eps, int num_iters, bool need_backward"
    ") -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!),"
    " Tensor(f!), Tensor(g!), Tensor(h!), Tensor(i!), Tensor(j!), Tensor(k!))");

  m.def(
    "mega_mhc_grad("
    "  Tensor grad_hin_placeholder, Tensor grad_h_post, Tensor grad_h_res,"
    "  Tensor x, Tensor phi, Tensor alpha, Tensor bias, Tensor previous_pre,"
    "  Tensor hc_before_norm, Tensor inv_rms, Tensor sum_out, Tensor norm_out,"
    "  Tensor grad_current_pre, Tensor mixed_input, Tensor rms_rstd,"
    "  Tensor norm_weight, Tensor direct_grad_x, Tensor previous_residual,"
    "  Tensor previous_output, Tensor previous_post, Tensor previous_residual_mix,"
    "  Tensor runtime_config, Tensor all_event_counters, Tensor profile_buffer,"
    "  Tensor(a!) grad_residual, Tensor(b!) grad_phi, Tensor(c!) grad_alpha,"
    "  Tensor(d!) grad_bias, Tensor(e!) grad_previous_output,"
    "  Tensor(f!) grad_previous_pre, Tensor(g!) grad_previous_post,"
    "  Tensor(h!) grad_previous_residual, Tensor(i!) grad_norm_weight, float hc_eps"
    ") -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!),"
    " Tensor(f!), Tensor(g!), Tensor(h!), Tensor(i!))");
}
