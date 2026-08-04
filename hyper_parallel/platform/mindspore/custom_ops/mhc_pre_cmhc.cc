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
#include <string>
#include <tuple>
#include <vector>
#include "ms_extension/all.h"
#include "module.h"

namespace custom {

namespace {
std::string ShapeStr(const ms::Tensor &t) {
    const auto &s = t.shape();
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i > 0) r += ",";
        r += std::to_string(s[i]);
    }
    r += "]";
    return r;
}
}  // namespace
namespace {
int64_t Factorial(int64_t n) {
    int64_t r = 1;
    for (int64_t i = 2; i <= n; ++i) {
        r *= i;
    }
    return r;
}

using MhcPreCmhcOutputs =
    std::tuple<ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor>;

MhcPreCmhcOutputs GenResultTensors(const ms::Tensor &x) {
    const auto &x_shape = x.shape();
    const bool is_bsnd = x_shape.size() == 4;
    const int64_t leading0 = x_shape[0];
    const int64_t leading1 = is_bsnd ? x_shape[1] : 0;
    const int64_t n = is_bsnd ? x_shape[2] : x_shape[1];
    const int64_t d = is_bsnd ? x_shape[3] : x_shape[2];
    const int64_t n_perm = Factorial(n);         // n!
    const int64_t fusion_size = n_perm + 2 * n;  // n! + 2n
    const int64_t nn = n * n;                    // n^2

    std::vector<int64_t> h_in_shape;
    std::vector<int64_t> h_post_shape;
    std::vector<int64_t> h_res_shape;
    std::vector<int64_t> inv_rms_shape;
    std::vector<int64_t> h_mix_shape;
    std::vector<int64_t> h_pre_shape;
    std::vector<int64_t> coeff_shape;

    if (is_bsnd) {
        h_in_shape = {leading0, leading1, d};
        h_post_shape = {leading0, leading1, n};
        h_res_shape = {leading0, leading1, nn};
        inv_rms_shape = {leading0, leading1};
        h_mix_shape = {leading0, leading1, fusion_size};
        h_pre_shape = {leading0, leading1, n};
        coeff_shape = {leading0, leading1, n_perm};
    } else {
        h_in_shape = {leading0, d};
        h_post_shape = {leading0, n};
        h_res_shape = {leading0, nn};
        inv_rms_shape = {leading0};
        h_mix_shape = {leading0, fusion_size};
        h_pre_shape = {leading0, n};
        coeff_shape = {leading0, n_perm};
    }

    auto h_in = ms::Tensor(x.data_type(), h_in_shape);
    auto h_post = ms::Tensor(ms::TypeId::kNumberTypeFloat32, h_post_shape);
    auto h_res = ms::Tensor(ms::TypeId::kNumberTypeFloat32, h_res_shape);
    auto inv_rms = ms::Tensor(ms::TypeId::kNumberTypeFloat32, inv_rms_shape);
    auto h_mix = ms::Tensor(ms::TypeId::kNumberTypeFloat32, h_mix_shape);
    auto h_pre = ms::Tensor(ms::TypeId::kNumberTypeFloat32, h_pre_shape);
    auto coeff = ms::Tensor(ms::TypeId::kNumberTypeFloat32, coeff_shape);

    return std::make_tuple(std::move(h_in), std::move(h_post), std::move(h_res), std::move(inv_rms),
                           std::move(h_mix), std::move(h_pre), std::move(coeff));
}
}  // namespace

// gamma is accepted from Python (created as ones when absent). The kernel
// applies (void)gamma (equivalent to gamma=1). gamma is included in Run()'s
// input list so the runtime manages its device memory and lifetime across
// async execution.
std::vector<ms::Tensor> npu_mhc_pre_cmhc(const ms::Tensor &x, const ms::Tensor &phi,
                                         const ms::Tensor &alpha, const ms::Tensor &bias,
                                         const ms::Tensor &perm_mats,
                                         const ms::Tensor &gamma,
                                         double hc_eps, double norm_eps) {
    auto [h_in, h_post, h_res, inv_rms, h_mix, h_pre, coeff] = GenResultTensors(x);
    std::optional<ms::Tensor> gamma_opt = gamma;
    auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MhcPreCmhc");
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMhcPreCmhc, x, phi, alpha, bias, gamma_opt, perm_mats,
                                             norm_eps, hc_eps, h_in, h_post, h_res, inv_rms, h_mix,
                                             h_pre, coeff));
    // fprintf(stderr, "[CMHC] kernel before Run(): x=%s dtype=%d phi=%s alpha=%s bias=%s perm=%s gamma=%d\n",
    //         ShapeStr(x).c_str(), static_cast<int>(x.data_type()),
    //         ShapeStr(phi).c_str(), ShapeStr(alpha).c_str(), ShapeStr(bias).c_str(),
    //         ShapeStr(perm_mats).c_str(), gamma_opt.has_value() ? 1 : 0);
    // fflush(stderr);
    runner->Run({x, phi, alpha, bias, gamma, perm_mats},
                {h_in, h_post, h_res, inv_rms, h_mix, h_pre, coeff});
    // fprintf(stderr, "[CMHC] kernel after Run() OK\n"); fflush(stderr);
    return {h_in, h_post, h_res, inv_rms, h_mix, h_pre, coeff};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
    m.def("npu_mhc_pre_cmhc", PYBOOST_CALLER(7, custom::npu_mhc_pre_cmhc));
}
}  // namespace custom
