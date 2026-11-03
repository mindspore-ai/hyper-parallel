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

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {

using IntArrayArg = std::pair<std::optional<std::vector<int64_t>>, bool>;

IntArrayArg OptionalIntArray(const std::optional<std::vector<int64_t>> &value) { return {value, true}; }

void ValidateShapeAccess(const ms::Tensor &q) {
  const size_t rank = q.shape().size();
  if (rank != 3U && rank != 4U) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_bwd: q rank must be 3 or 4.";
  }
}

}  // namespace

std::vector<ms::Tensor> npu_chunk_kda_bwd(
  const ms::Tensor &q, const ms::Tensor &k, const ms::Tensor &v, const ms::Tensor &beta, const ms::Tensor &gk,
  const ms::Tensor &aqk, const ms::Tensor &akk, const ms::Tensor &w, const ms::Tensor &qg, const ms::Tensor &kg,
  const ms::Tensor &v_new, const ms::Tensor &h, const ms::Tensor &d_o, const std::optional<ms::Tensor> &raw_g_opt,
  const std::optional<ms::Tensor> &a_log_opt, const std::optional<ms::Tensor> &dt_bias_opt,
  const std::optional<std::vector<int64_t>> &cu_seqlens_opt,
  const std::optional<std::vector<int64_t>> &chunk_indices_opt, double scale, int64_t chunk_size, bool safe_gate,
  bool use_gate_in_kernel, double lower_bound, const std::optional<std::string> &layout_opt) {
  const std::string layout = layout_opt.value_or("BNSD");
  ValidateShapeAccess(q);
  const bool varlen = q.shape().size() == 3U;
  if (!use_gate_in_kernel || !safe_gate || !raw_g_opt.has_value() || !a_log_opt.has_value() ||
      !dt_bias_opt.has_value()) {
    MS_LOG(EXCEPTION) << "Current KDA backward requires fused safe gate with raw gate, A_log and dt_bias.";
  }

  ms::Tensor dq(q.data_type(), q.shape());
  ms::Tensor dk(k.data_type(), k.shape());
  ms::Tensor dv(v.data_type(), v.shape());
  ms::Tensor db(ms::TypeId::kNumberTypeFloat32, beta.shape());
  ms::Tensor dg(raw_g_opt.value().data_type(), raw_g_opt.value().shape());
  const int64_t heads = varlen ? q.shape()[0] : q.shape()[1];
  ms::Tensor d_a(ms::TypeId::kNumberTypeFloat32, {heads});
  ms::Tensor d_bias(ms::TypeId::kNumberTypeFloat32, {heads, 128});

  ms::TensorToDevice(q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h, d_o, raw_g_opt.value_or(ms::Tensor()),
                     a_log_opt.value_or(ms::Tensor()), dt_bias_opt.value_or(ms::Tensor()));
  ms::TensorAllocate({dq, dk, dv, db, dg, d_a, d_bias});

  const std::optional<ms::Tensor> empty = std::nullopt;
  const std::optional<ms::Tensor> d_a_opt = use_gate_in_kernel ? std::optional<ms::Tensor>(d_a) : empty;
  const std::optional<ms::Tensor> d_bias_opt = dt_bias_opt.has_value() ? std::optional<ms::Tensor>(d_bias) : empty;
  MS_DISPATCH_ACLNN(aclnnChunkKdaBwd, q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h, d_o, raw_g_opt, a_log_opt,
                    dt_bias_opt, empty, empty, OptionalIntArray(cu_seqlens_opt), OptionalIntArray(chunk_indices_opt),
                    scale, chunk_size, safe_gate, use_gate_in_kernel, lower_bound, true, true, false,
                    empty, empty, empty, layout, false, false, dq, dk, dv, db,
                    dg, empty, d_a_opt, d_bias_opt);
  return {dq, dk, dv, db, dg, d_a, d_bias};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) { m.def("npu_chunk_kda_bwd", PYBOOST_CALLER(7, custom::npu_chunk_kda_bwd)); }

}  // namespace custom
