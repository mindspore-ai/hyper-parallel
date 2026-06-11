/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <tuple>
#include <vector>
#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {
std::tuple<ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor> GenResultTensors(const ms::Tensor &x,
                                                                            const ms::Tensor &phi,
                                                                            const ms::Tensor &alpha,
                                                                            const ms::Tensor &bias) {
  auto grad_x = ms::Tensor(x.data_type(), x.shape());
  auto grad_phi = ms::Tensor(phi.data_type(), phi.shape());
  auto grad_alpha = ms::Tensor(alpha.data_type(), alpha.shape());
  auto grad_bias = ms::Tensor(bias.data_type(), bias.shape());
  return std::make_tuple(std::move(grad_x), std::move(grad_phi), std::move(grad_alpha), std::move(grad_bias));
}
}  // namespace

std::vector<ms::Tensor> npu_mhc_pre_clamp_sinkhorn_backward(
  const ms::Tensor &grad_h_in, const ms::Tensor &grad_h_post, const ms::Tensor &grad_h_res, const ms::Tensor &x,
  const ms::Tensor &phi, const ms::Tensor &alpha, const ms::Tensor &bias, const ms::Tensor &h_pre,
  const ms::Tensor &hc_before_norm, const ms::Tensor &inv_rms, const ms::Tensor &sum_out, const ms::Tensor &norm_out,
  const ms::Tensor &h_res_logits, double hc_eps, double clamp_min, double clamp_max) {
  auto [grad_x, grad_phi, grad_alpha, grad_bias] = GenResultTensors(x, phi, alpha, bias);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MhcPreClampSinkhornBackward");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMhcPreClampSinkhornBackward, grad_h_in, grad_h_post, grad_h_res, x, phi,
                                          alpha, bias, h_pre, hc_before_norm, inv_rms, sum_out, norm_out, h_res_logits,
                                          hc_eps, clamp_min, clamp_max, grad_x, grad_phi, grad_alpha, grad_bias));
  runner->Run({grad_h_in, grad_h_post, grad_h_res, x, phi, alpha, bias, h_pre, hc_before_norm, inv_rms, sum_out,
               norm_out, h_res_logits},
              {grad_x, grad_phi, grad_alpha, grad_bias});
  return {grad_x, grad_phi, grad_alpha, grad_bias};
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("npu_mhc_pre_clamp_sinkhorn_backward", PYBOOST_CALLER(4, custom::npu_mhc_pre_clamp_sinkhorn_backward));
}
}  // namespace custom
