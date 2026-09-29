/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * PyTorch NPU adapters for the CANN mHC Golden and HyperMegaMhc.
 */
#include <torch/library.h>
#include <tuple>
#include "op_plugin/include/npu_cpp_extension.h"

namespace {

using EightTensorRefs = std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                                   at::Tensor &, at::Tensor &>;
using FiveTensorRefs = std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &>;
using NineTensorRefs = std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
                                  at::Tensor &, at::Tensor &, at::Tensor &>;

EightTensorRefs cann_mhc_pre_sinkhorn_npu(const at::Tensor &x, const at::Tensor &phi, const at::Tensor &alpha,
                                          const at::Tensor &bias, at::Tensor &hin, at::Tensor &h_post,
                                          at::Tensor &h_res, at::Tensor &h_pre, at::Tensor &hc_before_norm,
                                          at::Tensor &inv_rms, at::Tensor &sum_out, at::Tensor &norm_out, double hc_eps,
                                          double norm_eps, int64_t num_iters) {
  int64_t hc_mult = 4;
  bool need_backward = true;
  EXEC_NPU_CMD_EXT(aclnnMhcPreSinkhorn, x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, need_backward, hin,
                   h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out);
  return EightTensorRefs(hin, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out);
}

EightTensorRefs cann_mhc_pre_sinkhorn_meta(const at::Tensor & /*x*/, const at::Tensor & /*phi*/,
                                           const at::Tensor & /*alpha*/, const at::Tensor & /*bias*/, at::Tensor &hin,
                                           at::Tensor &h_post, at::Tensor &h_res, at::Tensor &h_pre,
                                           at::Tensor &hc_before_norm, at::Tensor &inv_rms, at::Tensor &sum_out,
                                           at::Tensor &norm_out, double /*hc_eps*/, double /*norm_eps*/,
                                           int64_t /*num_iters*/) {
  return EightTensorRefs(hin, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out);
}

FiveTensorRefs mega_mhc_npu(const at::Tensor &previous_output, const at::Tensor &residual,
                            const at::Tensor &previous_pre_mix, const at::Tensor &previous_post_mix,
                            const at::Tensor &previous_residual_mix, const at::Tensor &phi, const at::Tensor &alpha,
                            const at::Tensor &bias, const at::Tensor &norm_weight, const at::Tensor &runtime_config,
                            const at::Tensor &all_event_counters, const at::Tensor &profile_buffer,
                            at::Tensor &new_residual, at::Tensor &next_pre_mix, at::Tensor &next_post_mix,
                            at::Tensor &next_residual_mix, at::Tensor &block_input, double hc_eps, double norm_eps,
                            int64_t num_iters, bool need_backward) {
  EXEC_NPU_CMD_EXT(aclnnHyperMegaMhc, previous_output, residual, previous_pre_mix, previous_post_mix,
                   previous_residual_mix, phi, alpha, bias, norm_weight, runtime_config, all_event_counters,
                   profile_buffer, new_residual, next_pre_mix, next_post_mix, next_residual_mix, block_input, hc_eps,
                   norm_eps, num_iters, need_backward);
  return FiveTensorRefs(new_residual, next_pre_mix, next_post_mix, next_residual_mix, block_input);
}

FiveTensorRefs mega_mhc_meta(const at::Tensor & /*previous_output*/, const at::Tensor & /*residual*/,
                             const at::Tensor & /*previous_pre_mix*/, const at::Tensor & /*previous_post_mix*/,
                             const at::Tensor & /*previous_residual_mix*/, const at::Tensor & /*phi*/,
                             const at::Tensor & /*alpha*/, const at::Tensor & /*bias*/,
                             const at::Tensor & /*norm_weight*/, const at::Tensor & /*runtime_config*/,
                             const at::Tensor & /*all_event_counters*/, const at::Tensor & /*profile_buffer*/,
                             at::Tensor &new_residual, at::Tensor &next_pre_mix, at::Tensor &next_post_mix,
                             at::Tensor &next_residual_mix, at::Tensor &block_input, double /*hc_eps*/,
                             double /*norm_eps*/, int64_t /*num_iters*/, bool /*need_backward*/) {
  return FiveTensorRefs(new_residual, next_pre_mix, next_post_mix, next_residual_mix, block_input);
}

NineTensorRefs mega_mhc_grad_npu(const at::Tensor &grad_hin_placeholder, const at::Tensor &grad_h_post,
                                 const at::Tensor &grad_h_res, const at::Tensor &x, const at::Tensor &phi,
                                 const at::Tensor &alpha, const at::Tensor &bias, const at::Tensor &previous_pre,
                                 const at::Tensor &hc_before_norm, const at::Tensor &inv_rms, const at::Tensor &sum_out,
                                 const at::Tensor &norm_out, const at::Tensor &grad_current_pre,
                                 const at::Tensor &mixed_input, const at::Tensor &rms_rstd,
                                 const at::Tensor &norm_weight, const at::Tensor &direct_grad_x,
                                 const at::Tensor &previous_residual, const at::Tensor &previous_output,
                                 const at::Tensor &previous_post, const at::Tensor &previous_residual_mix,
                                 const at::Tensor &runtime_config, const at::Tensor &all_event_counters,
                                 const at::Tensor &profile_buffer, at::Tensor &grad_residual, at::Tensor &grad_phi,
                                 at::Tensor &grad_alpha, at::Tensor &grad_bias, at::Tensor &grad_previous_output,
                                 at::Tensor &grad_previous_pre, at::Tensor &grad_previous_post,
                                 at::Tensor &grad_previous_residual, at::Tensor &grad_norm_weight, double hc_eps) {
  EXEC_NPU_CMD_EXT(aclnnHyperMegaMhcGrad, grad_hin_placeholder, grad_h_post, grad_h_res, x, phi, alpha, bias,
                   previous_pre, hc_before_norm, inv_rms, sum_out, norm_out, grad_current_pre, mixed_input, rms_rstd,
                   norm_weight, direct_grad_x, previous_residual, previous_output, previous_post, previous_residual_mix,
                   runtime_config, all_event_counters, profile_buffer, grad_residual, grad_phi, grad_alpha, grad_bias,
                   grad_previous_output, grad_previous_pre, grad_previous_post, grad_previous_residual,
                   grad_norm_weight, hc_eps);
  return NineTensorRefs(grad_residual, grad_phi, grad_alpha, grad_bias, grad_previous_output, grad_previous_pre,
                        grad_previous_post, grad_previous_residual, grad_norm_weight);
}

NineTensorRefs mega_mhc_grad_meta(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                  const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                  const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                  const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                  const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                  const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                  at::Tensor &grad_residual, at::Tensor &grad_phi, at::Tensor &grad_alpha,
                                  at::Tensor &grad_bias, at::Tensor &grad_previous_output,
                                  at::Tensor &grad_previous_pre, at::Tensor &grad_previous_post,
                                  at::Tensor &grad_previous_residual, at::Tensor &grad_norm_weight, double) {
  return NineTensorRefs(grad_residual, grad_phi, grad_alpha, grad_bias, grad_previous_output, grad_previous_pre,
                        grad_previous_post, grad_previous_residual, grad_norm_weight);
}

}  // namespace

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("cann_mhc_pre_sinkhorn", &cann_mhc_pre_sinkhorn_npu);
  m.impl("mega_mhc", &mega_mhc_npu);
  m.impl("mega_mhc_grad", &mega_mhc_grad_npu);
}

TORCH_LIBRARY_IMPL(hyper_parallel, Meta, m) {
  m.impl("cann_mhc_pre_sinkhorn", &cann_mhc_pre_sinkhorn_meta);
  m.impl("mega_mhc", &mega_mhc_meta);
  m.impl("mega_mhc_grad", &mega_mhc_grad_meta);
}
