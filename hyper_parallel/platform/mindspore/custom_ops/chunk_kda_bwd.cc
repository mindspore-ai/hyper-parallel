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

void ValidateCanonicalInputRanks(const ms::Tensor &q, const ms::Tensor &k, const ms::Tensor &v, const ms::Tensor &beta,
                                 const ms::Tensor &gk, const ms::Tensor &aqk, size_t token_rank) {
  const auto q_shape = q.shape();
  const bool token_shapes_valid = q_shape.size() == token_rank && k.shape() == q_shape &&
                                  v.shape().size() == token_rank && beta.shape().size() == token_rank - 1 &&
                                  gk.shape().size() == token_rank && aqk.shape().size() == token_rank;
  if (!token_shapes_valid) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_bwd: invalid canonical dense/varlen tensor ranks.";
  }
}

void ValidateCanonicalIntermediateRanks(const ms::Tensor &aqk, const ms::Tensor &akk, const ms::Tensor &w,
                                        const ms::Tensor &qg, const ms::Tensor &kg, const ms::Tensor &v_new,
                                        const ms::Tensor &h, const ms::Tensor &d_o, size_t token_rank,
                                        size_t state_rank) {
  const bool shapes_valid = akk.shape() == aqk.shape() && w.shape().size() == token_rank &&
                            qg.shape().size() == token_rank && kg.shape().size() == token_rank &&
                            v_new.shape().size() == token_rank && d_o.shape().size() == token_rank &&
                            h.shape().size() == state_rank;
  if (!shapes_valid) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_bwd: invalid canonical dense/varlen tensor ranks.";
  }
}

void ValidateCanonicalDimensions(const ms::Tensor &q, const ms::Tensor &v, const ms::Tensor &aqk, int64_t chunk_size,
                                 bool varlen) {
  const auto q_shape = q.shape();
  const size_t head_axis = varlen ? 0U : 1U;
  const size_t token_axis = varlen ? 1U : 2U;
  const size_t dim_axis = varlen ? 2U : 3U;
  const int64_t heads = q_shape[head_axis];
  const int64_t tokens = q_shape[token_axis];
  const bool dimensions_valid = q_shape[dim_axis] == 128 && v.shape()[dim_axis] == 128 &&
                                v.shape()[head_axis] == heads && v.shape()[token_axis] == tokens &&
                                aqk.shape()[dim_axis] == chunk_size;
  if (!dimensions_valid) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_bwd: canonical kernel inputs require equal heads and K=V=128.";
  }
}

void CheckCanonicalShape(const ms::Tensor &q, const ms::Tensor &k, const ms::Tensor &v, const ms::Tensor &beta,
                         const ms::Tensor &gk, const ms::Tensor &aqk, const ms::Tensor &akk, const ms::Tensor &w,
                         const ms::Tensor &qg, const ms::Tensor &kg, const ms::Tensor &v_new, const ms::Tensor &h,
                         const ms::Tensor &d_o, int64_t chunk_size, bool varlen) {
  if (chunk_size != 64) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_bwd: only chunk_size=64 is supported.";
  }
  const size_t token_rank = varlen ? 3U : 4U;
  const size_t state_rank = varlen ? 4U : 5U;
  ValidateCanonicalInputRanks(q, k, v, beta, gk, aqk, token_rank);
  ValidateCanonicalIntermediateRanks(aqk, akk, w, qg, kg, v_new, h, d_o, token_rank, state_rank);
  ValidateCanonicalDimensions(q, v, aqk, chunk_size, varlen);
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
  if (layout != "BNSD" && layout != "NTD") {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_bwd: bridge expects canonical BNSD or NTD input.";
  }
  if (cu_seqlens_opt.has_value() != chunk_indices_opt.has_value()) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_bwd: cu_seqlens and chunk_indices must be supplied together.";
  }
  if (use_gate_in_kernel && (!raw_g_opt.has_value() || !a_log_opt.has_value())) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_bwd: raw_g and a_log are required for raw gate backward.";
  }
  const bool varlen = layout == "NTD";
  CheckCanonicalShape(q, k, v, beta, gk, aqk, akk, w, qg, kg, v_new, h, d_o, chunk_size, varlen);

  ms::Tensor dq(ms::TypeId::kNumberTypeFloat32, q.shape());
  ms::Tensor dk(ms::TypeId::kNumberTypeFloat32, k.shape());
  ms::Tensor dv(v.data_type(), v.shape());
  ms::Tensor db(ms::TypeId::kNumberTypeFloat32, beta.shape());
  ms::Tensor dg(ms::TypeId::kNumberTypeFloat32, gk.shape());
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
                    scale, chunk_size, safe_gate, use_gate_in_kernel, lower_bound, true, true, false, dq, dk, dv, db,
                    dg, empty, d_a_opt, d_bias_opt);
  return {dq, dk, dv, db, dg, d_a, d_bias};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) { m.def("npu_chunk_kda_bwd", PYBOOST_CALLER(7, custom::npu_chunk_kda_bwd)); }

}  // namespace custom
