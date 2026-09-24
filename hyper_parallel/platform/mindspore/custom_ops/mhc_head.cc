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
using MhcHeadOutputs = std::tuple<ms::Tensor, ms::Tensor, ms::Tensor>;

// Derive the three forward outputs from x [s,b,nH] and weight [n,nH].
//   out     [s,b,H]   (H = nH / n, same dtype as x)
//   rms_inv [s,b,1]   FP32 (optional, cached for backward)
//   mixes   [s,b,n]   FP32 (optional, cached for backward)
MhcHeadOutputs GenResultTensors(const ms::Tensor &x, const ms::Tensor &weight,
                                const ms::Tensor &hc_base, const ms::Tensor &hc_scale) {
    const auto &x_shape = x.shape();
    const auto &w_shape = weight.shape();
    const int64_t s = x_shape[0];
    const int64_t b = x_shape[1];
    const int64_t n = w_shape[0];
    const int64_t n_h = w_shape[1];
    const int64_t h = n_h / n;

    auto out = ms::Tensor(x.data_type(), std::vector<int64_t>{s, b, h});
    auto rms_inv = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{s, b, 1});
    auto mixes = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{s, b, n});
    return std::make_tuple(std::move(out), std::move(rms_inv), std::move(mixes));
}
}  // namespace

// MhcHead forward: collapse n residual streams [s,b,nH] into [s,b,H] via
// linear projection -> RMSNorm -> sigmoid gated weighted sum.
// NOTE: MS_DISPATCH_ACLNN argument order MUST match aclnn_mhc_head.cpp exactly
// (normEps before hcEps). The .cc function signature uses (hc_eps, norm_eps)
// to stay consistent with the mhc_pre_cmhc convention, but the dispatch below
// reorders to (norm_eps, hc_eps) per the aclnn ABI.
std::vector<ms::Tensor> npu_mhc_head(const ms::Tensor &x, const ms::Tensor &weight,
                                     const ms::Tensor &hc_base, const ms::Tensor &hc_scale,
                                     double hc_eps, double norm_eps) {
    auto [out, rms_inv, mixes] = GenResultTensors(x, weight, hc_base, hc_scale);
    ms::TensorToDevice(x, weight, hc_base, hc_scale);
    ms::TensorAllocate({out, rms_inv, mixes});
    MS_DISPATCH_ACLNN(aclnnMhcHead, x, weight, hc_base, hc_scale, norm_eps, hc_eps, out, rms_inv,
                      mixes);
    return {out, rms_inv, mixes};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
    m.def("npu_mhc_head", PYBOOST_CALLER(3, custom::npu_mhc_head));
}
}  // namespace custom
