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
#include <optional>
#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {
std::tuple<ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor> GenResultTensors(
    const ms::Tensor &x, const ms::Tensor &phi, const ms::Tensor &alpha) {
    auto grad_x = ms::Tensor(x.data_type(), x.shape());
    auto grad_phi = ms::Tensor(phi.data_type(), phi.shape());
    auto grad_alpha = ms::Tensor(alpha.data_type(), alpha.shape());
    // grad_bias shape [n!+2n] derived from phi.dim0 (backward aclnn takes no bias input).
    auto grad_bias = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{phi.shape()[0]});
    return std::make_tuple(std::move(grad_x), std::move(grad_phi), std::move(grad_alpha),
                           std::move(grad_bias));
}
}  // namespace

// NOTE: backward aclnn takes NO bias (grad_bias is derived internally from
// h_mix + grad_h_*). gamma is optional: None passes through as nullptr to
// aclnn (gamma ignored, (void)gamma). When gamma is nullptr, gradGamma is also
// nullptr (aclnn skips grad_gamma computation). value_or(ms::Tensor()) yields
// an empty tensor for TensorToDevice (no-op when gamma is absent).
std::vector<ms::Tensor> npu_mhc_pre_cmhc_backward(
    const ms::Tensor &grad_h_in, const ms::Tensor &grad_h_post, const ms::Tensor &grad_h_res,
    const ms::Tensor &x, const ms::Tensor &phi, const ms::Tensor &alpha,
    const ms::Tensor &h_pre, const ms::Tensor &h_mix, const ms::Tensor &inv_rms,
    const ms::Tensor &h_post, const std::optional<ms::Tensor> &gamma_opt,
    const ms::Tensor &perms, const ms::Tensor &coeff, double hc_eps) {
    auto [grad_x, grad_phi, grad_alpha, grad_bias] = GenResultTensors(x, phi, alpha);
    auto gamma = gamma_opt.value_or(ms::Tensor());
    const std::optional<ms::Tensor> grad_gamma_opt;  // nullopt — gamma absent, skip grad_gamma
    ms::TensorToDevice(x, phi, alpha, grad_h_in, grad_h_post, grad_h_res, inv_rms, h_mix, h_pre,
                       h_post, gamma, perms, coeff);
    ms::TensorAllocate({grad_x, grad_phi, grad_alpha, grad_bias});
    MS_DISPATCH_ACLNN(aclnnMhcPreCmhcBackward, x, phi, alpha, grad_h_in, grad_h_post, grad_h_res,
                      inv_rms, h_mix, h_pre, h_post, gamma_opt, perms, coeff, hc_eps, grad_x,
                      grad_phi, grad_alpha, grad_bias, grad_gamma_opt);
    return {grad_x, grad_phi, grad_alpha, grad_bias};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
    m.def("npu_mhc_pre_cmhc_backward", PYBOOST_CALLER(4, custom::npu_mhc_pre_cmhc_backward));
}
}  // namespace custom
