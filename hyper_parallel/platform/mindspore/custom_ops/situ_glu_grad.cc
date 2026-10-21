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

// Backward: grad_x shape == x shape. aclnnSituGluGrad(gradY, x, dim, beta,
// linearBeta, activateLeft, out). Inputs gradY and x must be contiguous
// (aclnn constraint); the Python DFunction runs _ensure_contiguous on both
// before calling.
std::vector<ms::Tensor> npu_situ_glu_grad(const ms::Tensor &grad_y, const ms::Tensor &x,
                                          int64_t dim, double beta, double linear_beta,
                                          bool activate_left) {
    auto grad_x = ms::Tensor(x.data_type(), x.shape());
    ms::TensorToDevice(grad_y, x);
    ms::TensorAllocate({grad_x});
    MS_DISPATCH_ACLNN(aclnnSituGluGrad, grad_y, x, dim, beta, linear_beta, activate_left, grad_x);
    return {grad_x};
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
    m.def("npu_situ_glu_grad", PYBOOST_CALLER(1, custom::npu_situ_glu_grad));
}
}  // namespace custom
