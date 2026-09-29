/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <vector>
#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {
// Output shape: same as x but dim axis is halved. dim is normalized to [0, rank).
ms::Tensor GenResultTensor(const ms::Tensor &x, int64_t dim) {
    const auto &x_shape = x.shape();
    const int64_t rank = static_cast<int64_t>(x_shape.size());
    int64_t norm_dim = dim < 0 ? dim + rank : dim;
    std::vector<int64_t> out_shape = x_shape;
    out_shape[norm_dim] = x_shape[norm_dim] / 2;
    return ms::Tensor(x.data_type(), out_shape);
}
}  // namespace

// SiTU-GLU fused activation. activate_left is pinned to true by the Python
// wrapper (matches mindformers SiTUGLU's chunk(x, 2, dim) gate=front semantics).
std::vector<ms::Tensor> npu_situ_glu(const ms::Tensor &x, int64_t dim, double beta,
                                     double linear_beta, bool activate_left) {
    auto out = GenResultTensor(x, dim);
    ms::TensorToDevice(x);
    ms::TensorAllocate({out});
    MS_DISPATCH_ACLNN(aclnnSituGlu, x, dim, beta, linear_beta, activate_left, out);
    return {out};
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
    m.def("npu_situ_glu", PYBOOST_CALLER(1, custom::npu_situ_glu));
}
}  // namespace custom
