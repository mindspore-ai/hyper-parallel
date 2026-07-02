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
std::tuple<ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor>
GenResultTensors(const ms::Tensor &x, int64_t num_iters) {
  const auto &x_shape = x.shape();
  const int64_t bs = x_shape[0];
  const int64_t seq_len = x_shape[1];
  const int64_t n = x_shape[2];
  const int64_t c = x_shape[3];
  const int64_t fusion_size = n * n + 2 * n;

  auto h_in = ms::Tensor(x.data_type(), std::vector<int64_t>{bs, seq_len, c});
  auto h_post = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{bs, seq_len, n});
  auto h_res = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{bs, seq_len, n * n});
  auto h_pre_shape = std::vector<int64_t>{bs, seq_len, n};
  auto hc_before_norm_shape = std::vector<int64_t>{bs, seq_len, fusion_size};
  auto inv_rms_shape = std::vector<int64_t>{bs, seq_len, 1};
  auto sum_out_shape = std::vector<int64_t>{2 * num_iters, bs, seq_len, n};
  auto norm_out_shape =
    std::vector<int64_t>{2 * num_iters, bs, seq_len, n, n};
  auto h_pre = ms::Tensor(ms::TypeId::kNumberTypeFloat32, h_pre_shape);
  auto hc_before_norm = ms::Tensor(ms::TypeId::kNumberTypeFloat32, hc_before_norm_shape);
  auto inv_rms = ms::Tensor(ms::TypeId::kNumberTypeFloat32, inv_rms_shape);
  auto sum_out = ms::Tensor(ms::TypeId::kNumberTypeFloat32, sum_out_shape);
  auto norm_out = ms::Tensor(ms::TypeId::kNumberTypeFloat32, norm_out_shape);

  return std::make_tuple(std::move(h_in), std::move(h_post), std::move(h_res), std::move(h_pre),
                         std::move(hc_before_norm), std::move(inv_rms), std::move(sum_out), std::move(norm_out));
}
}  // namespace

std::vector<ms::Tensor> npu_mhc_pre_sinkhorn(const ms::Tensor &x, const ms::Tensor &phi, const ms::Tensor &alpha,
                                            const ms::Tensor &bias, int64_t hc_mult, int64_t num_iters,
                                            double hc_eps, double norm_eps, bool out_flag) {
  auto [h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out] =
    GenResultTensors(x, num_iters);
  int hc_mult_value = static_cast<int>(hc_mult);
  int num_iters_value = static_cast<int>(num_iters);
  ms::TensorToDevice(x, phi, alpha, bias);
  ms::TensorAllocate({h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out});
  MS_DISPATCH_ACLNN(aclnnMhcPreSinkhorn, x, phi, alpha, bias, hc_mult_value, num_iters_value, hc_eps, norm_eps,
                    out_flag, h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out);
  return {h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("npu_mhc_pre_sinkhorn", PYBOOST_CALLER(8, custom::npu_mhc_pre_sinkhorn));
}
}  // namespace custom
