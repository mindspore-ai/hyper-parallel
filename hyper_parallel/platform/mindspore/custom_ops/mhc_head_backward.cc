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
using MhcHeadBackwardOutputs = std::tuple<ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor>;

// Derive the four gradient outputs from the forward inputs.
//   grad_x      [s,b,nH]  same dtype as x
//   grad_weight [n,nH]    FP32
//   grad_scale  [1]       FP32
//   grad_base   [n]       FP32
MhcHeadBackwardOutputs GenResultTensors(const ms::Tensor &x, const ms::Tensor &weight,
                                        const ms::Tensor &hc_base, const ms::Tensor &hc_scale) {
    auto grad_x = ms::Tensor(x.data_type(), x.shape());
    auto grad_weight = ms::Tensor(weight.data_type(), weight.shape());
    auto grad_scale = ms::Tensor(ms::TypeId::kNumberTypeFloat32, hc_scale.shape());
    auto grad_base = ms::Tensor(ms::TypeId::kNumberTypeFloat32, hc_base.shape());
    return std::make_tuple(std::move(grad_x), std::move(grad_weight), std::move(grad_scale),
                           std::move(grad_base));
}
}  // namespace

// MhcHead backward: given grad_output and forward caches (rms_inv, mixes),
// compute grad_x / grad_weight / grad_scale / grad_base.
// NOTE: backward aclnn takes only hcEps (no normEps) — rms_inv is already
// cached so RMSNorm is not recomputed. MS_DISPATCH_ACLNN argument order MUST
// match aclnn_mhc_head_backward.cpp exactly.
std::vector<ms::Tensor> npu_mhc_head_backward(const ms::Tensor &x, const ms::Tensor &weight,
                                              const ms::Tensor &hc_base, const ms::Tensor &hc_scale,
                                              const ms::Tensor &grad_output, const ms::Tensor &rms_inv,
                                              const ms::Tensor &mixes, double hc_eps) {
    auto [grad_x, grad_weight, grad_scale, grad_base] = GenResultTensors(x, weight, hc_base, hc_scale);
    ms::TensorToDevice(x, weight, hc_base, hc_scale, grad_output, rms_inv, mixes);
    ms::TensorAllocate({grad_x, grad_weight, grad_scale, grad_base});
    MS_DISPATCH_ACLNN(aclnnMhcHeadBackward, x, weight, hc_base, hc_scale, grad_output, rms_inv, mixes,
                      hc_eps, grad_x, grad_weight, grad_scale, grad_base);
    return {grad_x, grad_weight, grad_scale, grad_base};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
    m.def("npu_mhc_head_backward", PYBOOST_CALLER(4, custom::npu_mhc_head_backward));
}
}  // namespace custom
