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

#include <vector>
#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {
ms::Tensor GenResultTensor(const ms::Tensor &x) { return ms::Tensor(x.data_type(), x.shape()); }
}  // namespace

std::vector<ms::Tensor> npu_mhc_post(const ms::Tensor &x, const ms::Tensor &h_res, const ms::Tensor &h_out,
                                     const ms::Tensor &h_post) {
  auto out = GenResultTensor(x);
  ms::TensorToDevice(x, h_res, h_out, h_post);
  ms::TensorAllocate(out);
  MS_DISPATCH_ACLNN(aclnnMhcPost, x, h_res, h_out, h_post, out);
  return {out};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) { m.def("npu_mhc_post", PYBOOST_CALLER(1, custom::npu_mhc_post)); }
}  // namespace custom
