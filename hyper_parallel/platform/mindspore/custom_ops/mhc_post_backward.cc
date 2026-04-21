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
                                                                            const ms::Tensor &h_res,
                                                                            const ms::Tensor &h_out,
                                                                            const ms::Tensor &h_post) {
  auto grad_x = ms::Tensor(x.data_type(), x.shape());
  auto grad_h_res = ms::Tensor(h_res.data_type(), h_res.shape());
  auto grad_h_out = ms::Tensor(h_out.data_type(), h_out.shape());
  auto grad_h_post = ms::Tensor(h_post.data_type(), h_post.shape());
  return std::make_tuple(std::move(grad_x), std::move(grad_h_res), std::move(grad_h_out), std::move(grad_h_post));
}
}  // namespace

std::vector<ms::Tensor> npu_mhc_post_backward(const ms::Tensor &grad_y, const ms::Tensor &x,
                                              const ms::Tensor &h_res, const ms::Tensor &h_out,
                                              const ms::Tensor &h_post) {
  auto [grad_x, grad_h_res, grad_h_out, grad_h_post] = GenResultTensors(x, h_res, h_out, h_post);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MhcPostBackward");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMhcPostBackward, grad_y, x, h_res, h_out, h_post, grad_x, grad_h_res,
                                          grad_h_out, grad_h_post));
  runner->Run({grad_y, x, h_res, h_out, h_post}, {grad_x, grad_h_res, grad_h_out, grad_h_post});
  return {grad_x, grad_h_res, grad_h_out, grad_h_post};
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("npu_mhc_post_backward", PYBOOST_CALLER(4, custom::npu_mhc_post_backward));
}
}  // namespace custom
