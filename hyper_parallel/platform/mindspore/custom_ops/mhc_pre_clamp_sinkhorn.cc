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
using MhcPreClampSinkhornOutputs =
  std::tuple<ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor>;

MhcPreClampSinkhornOutputs GenResultTensors(const ms::Tensor &x, int64_t num_iters) {
  const auto &x_shape = x.shape();
  const bool is_bsnd = x_shape.size() == 4;
  const int64_t leading0 = x_shape[0];
  const int64_t leading1 = is_bsnd ? x_shape[1] : 0;
  const int64_t n = is_bsnd ? x_shape[2] : x_shape[1];
  const int64_t c = is_bsnd ? x_shape[3] : x_shape[2];
  const int64_t fusion_size = n * n + 2 * n;

  std::vector<int64_t> h_in_shape;
  std::vector<int64_t> h_post_shape;
  std::vector<int64_t> h_res_shape;
  std::vector<int64_t> h_pre_shape;
  std::vector<int64_t> hc_before_norm_shape;
  std::vector<int64_t> inv_rms_shape;
  std::vector<int64_t> sum_out_shape;
  std::vector<int64_t> norm_out_shape;
  std::vector<int64_t> h_res_logits_shape;

  if (is_bsnd) {
    h_in_shape = {leading0, leading1, c};
    h_post_shape = {leading0, leading1, n};
    h_res_shape = {leading0, leading1, n * n};
    h_pre_shape = h_post_shape;
    hc_before_norm_shape = {leading0, leading1, fusion_size};
    inv_rms_shape = {leading0, leading1, 1};
    sum_out_shape = {2 * num_iters, leading0, leading1, n};
    norm_out_shape = {2 * num_iters, leading0, leading1, n, n};
    h_res_logits_shape = norm_out_shape;
  } else {
    h_in_shape = {leading0, c};
    h_post_shape = {leading0, n};
    h_res_shape = {leading0, n * n};
    h_pre_shape = h_post_shape;
    hc_before_norm_shape = {leading0, fusion_size};
    inv_rms_shape = {leading0, 1};
    sum_out_shape = {2 * num_iters, leading0, n};
    norm_out_shape = {2 * num_iters, leading0, n, n};
    h_res_logits_shape = norm_out_shape;
  }

  auto h_in = ms::Tensor(x.data_type(), h_in_shape);
  auto h_post = ms::Tensor(ms::TypeId::kNumberTypeFloat32, h_post_shape);
  auto h_res = ms::Tensor(ms::TypeId::kNumberTypeFloat32, h_res_shape);
  auto h_pre = ms::Tensor(ms::TypeId::kNumberTypeFloat32, h_pre_shape);
  auto hc_before_norm = ms::Tensor(ms::TypeId::kNumberTypeFloat32, hc_before_norm_shape);
  auto inv_rms = ms::Tensor(ms::TypeId::kNumberTypeFloat32, inv_rms_shape);
  auto sum_out = ms::Tensor(ms::TypeId::kNumberTypeFloat32, sum_out_shape);
  auto norm_out = ms::Tensor(ms::TypeId::kNumberTypeFloat32, norm_out_shape);
  auto h_res_logits = ms::Tensor(ms::TypeId::kNumberTypeFloat32, h_res_logits_shape);

  return std::make_tuple(std::move(h_in), std::move(h_post), std::move(h_res), std::move(h_pre),
                         std::move(hc_before_norm), std::move(inv_rms), std::move(sum_out), std::move(norm_out),
                         std::move(h_res_logits));
}
}  // namespace

std::vector<ms::Tensor> npu_mhc_pre_clamp_sinkhorn(const ms::Tensor &x, const ms::Tensor &phi,
                                                   const ms::Tensor &alpha, const ms::Tensor &bias,
                                                   int64_t hc_mult, int64_t num_iters, double hc_eps,
                                                   double norm_eps, bool out_flag, double clamp_min,
                                                   double clamp_max) {
  auto [h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out, h_res_logits] =
    GenResultTensors(x, num_iters);
  int hc_mult_value = static_cast<int>(hc_mult);
  int num_iters_value = static_cast<int>(num_iters);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MhcPreClampSinkhorn");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMhcPreClampSinkhorn, x, phi, alpha, bias, hc_mult_value,
                                          num_iters_value, hc_eps, norm_eps, out_flag, clamp_min, clamp_max, h_in,
                                          h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out,
                                          h_res_logits));
  runner->Run({x, phi, alpha, bias},
              {h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out, h_res_logits});
  return {h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out, h_res_logits};
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("npu_mhc_pre_clamp_sinkhorn", PYBOOST_CALLER(9, custom::npu_mhc_pre_clamp_sinkhorn));
}
}  // namespace custom
