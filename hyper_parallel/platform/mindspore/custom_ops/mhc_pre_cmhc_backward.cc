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

#include <cstdio>
#include <tuple>
#include <vector>
#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {
std::tuple<ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor> GenResultTensors(
    const ms::Tensor &x, const ms::Tensor &phi, const ms::Tensor &alpha) {
    auto grad_x = ms::Tensor(x.data_type(), x.shape());
    auto grad_phi = ms::Tensor(phi.data_type(), phi.shape());
    auto grad_alpha = ms::Tensor(alpha.data_type(), alpha.shape());
    // grad_bias shape [n!+2n] derived from phi.dim0 (backward aclnn takes no bias input).
    auto grad_bias = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{phi.shape()[0]});
    // grad_gamma shape [n, d] derived from x.
    const auto &x_shape = x.shape();
    const bool is_bsnd = x_shape.size() == 4;
    const int64_t n = is_bsnd ? x_shape[2] : x_shape[1];
    const int64_t d = is_bsnd ? x_shape[3] : x_shape[2];
    auto grad_gamma = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{n, d});
    return std::make_tuple(std::move(grad_x), std::move(grad_phi), std::move(grad_alpha),
                           std::move(grad_bias), std::move(grad_gamma));
}
}  // namespace

// NOTE: backward aclnn takes NO bias (grad_bias is derived internally from
// h_mix + grad_h_*). gamma is not a Python input either (kernel (void)gamma);
// pass std::nullopt to SetLaunchFunc so aclnn treats gammaOptional as nullptr.
// gamma is accepted as a real parameter (created as ones in Python when absent).
// Included in Run()'s input list for device memory and lifetime management.
std::vector<ms::Tensor> npu_mhc_pre_cmhc_backward(
    const ms::Tensor &grad_h_in, const ms::Tensor &grad_h_post, const ms::Tensor &grad_h_res,
    const ms::Tensor &x, const ms::Tensor &phi, const ms::Tensor &alpha,
    const ms::Tensor &h_pre, const ms::Tensor &h_mix, const ms::Tensor &inv_rms,
    const ms::Tensor &h_post, const ms::Tensor &gamma,
    const ms::Tensor &perms, const ms::Tensor &coeff, double hc_eps) {
    auto [grad_x, grad_phi, grad_alpha, grad_bias, grad_gamma] = GenResultTensors(x, phi, alpha);
    std::optional<ms::Tensor> gamma_opt = gamma;
    auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MhcPreCmhcBackward");
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(
        aclnnMhcPreCmhcBackward, x, phi, alpha, grad_h_in, grad_h_post, grad_h_res, inv_rms,
        h_mix, h_pre, h_post, gamma_opt, perms, coeff, hc_eps, grad_x, grad_phi, grad_alpha,
        grad_bias, grad_gamma));
    // fprintf(stderr, "[CMHC] backward kernel before Run()\n"); fflush(stderr);
    runner->Run({x, phi, alpha, grad_h_in, grad_h_post, grad_h_res, inv_rms, h_mix, h_pre, h_post,
                 gamma, perms, coeff},
                {grad_x, grad_phi, grad_alpha, grad_bias, grad_gamma});
    // fprintf(stderr, "[CMHC] backward kernel after Run() OK\n"); fflush(stderr);
    return {grad_x, grad_phi, grad_alpha, grad_bias, grad_gamma};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
    m.def("npu_mhc_pre_cmhc_backward", PYBOOST_CALLER(5, custom::npu_mhc_pre_cmhc_backward));
}
}  // namespace custom
