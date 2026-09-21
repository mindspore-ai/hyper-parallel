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

int64_t CountChunks(const std::optional<std::vector<int64_t>> &cu_seqlens, int64_t tokens, int64_t chunk_size) {
  if (!cu_seqlens.has_value()) {
    return (tokens + chunk_size - 1) / chunk_size;
  }
  const auto &lengths = cu_seqlens.value();
  int64_t chunks = 0;
  for (size_t index = 1; index < lengths.size(); ++index) {
    chunks += (lengths[index] - lengths[index - 1] + chunk_size - 1) / chunk_size;
  }
  return chunks;
}

void ValidateForwardOptions(const std::string &layout, int64_t chunk_size,
                            const std::optional<std::vector<int64_t>> &cu_seqlens_opt,
                            const std::optional<std::vector<int64_t>> &chunk_indices_opt) {
  if (layout != "BNSD" && layout != "NTD") {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_fwd: bridge expects canonical BNSD or NTD input.";
  }
  if (chunk_size != 64 && chunk_size != 128) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_fwd: chunk_size must be 64 or 128.";
  }
  if (cu_seqlens_opt.has_value() != chunk_indices_opt.has_value()) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_fwd: cu_seqlens and chunk_indices must be supplied together.";
  }
}

void ValidateForwardRanks(const ms::Tensor &q, const ms::Tensor &k, const ms::Tensor &v, const ms::Tensor &g,
                          const ms::Tensor &beta, const std::string &layout) {
  const size_t expected_rank = layout == "NTD" ? 3U : 4U;
  if (q.shape().size() != expected_rank || k.shape() != q.shape() || v.shape().size() != expected_rank ||
      g.shape().size() != expected_rank || beta.shape().size() != expected_rank - 1) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_fwd: input ranks do not match canonical layout.";
  }
}

void ValidateFeatureDimensions(int64_t key_dim, int64_t value_dim, int64_t heads, int64_t value_heads) {
  if (key_dim < 16 || key_dim > 256 || key_dim % 16 != 0 || value_dim < 16 || value_dim > 256 || value_dim % 16 != 0 ||
      heads <= 0 || value_heads < heads || value_heads > 128 || value_heads % heads != 0) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_fwd: require 0 < H <= HV <= 128, HV % H == 0, and K/V in "
                         "[16, 256] with multiples of 16.";
  }
}

void ValidateVarlenDimensions(bool rank3, const std::optional<std::vector<int64_t>> &cu_seqlens_opt, int64_t batch) {
  if (rank3 && !cu_seqlens_opt.has_value()) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_fwd: rank-3 NTD input requires variable-length metadata.";
  }
  if (!rank3 && cu_seqlens_opt.has_value() && batch != 1) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_fwd: rank-4 variable-length input requires B=1.";
  }
}

void ValidateForwardInputShapes(const std::vector<int64_t> &v_shape, const ms::Tensor &g, const ms::Tensor &beta,
                                bool rank3, int64_t batch, int64_t tokens, int64_t key_dim, int64_t value_heads,
                                int64_t value_dim) {
  const std::vector<int64_t> value_shape = rank3 ? std::vector<int64_t>({value_heads, tokens, value_dim})
                                                 : std::vector<int64_t>({batch, value_heads, tokens, value_dim});
  const std::vector<int64_t> gate_shape = rank3 ? std::vector<int64_t>({value_heads, tokens, key_dim})
                                                : std::vector<int64_t>({batch, value_heads, tokens, key_dim});
  const std::vector<int64_t> beta_shape =
    rank3 ? std::vector<int64_t>({value_heads, tokens}) : std::vector<int64_t>({batch, value_heads, tokens});
  if (v_shape != value_shape || g.shape() != gate_shape || beta.shape() != beta_shape) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_fwd: v/g/beta shapes do not match canonical layout.";
  }
}

}  // namespace

std::vector<ms::Tensor> npu_chunk_kda_fwd(
  const ms::Tensor &q, const ms::Tensor &k, const ms::Tensor &v, const ms::Tensor &g, const ms::Tensor &beta,
  double scale, int64_t chunk_size, const std::optional<std::string> &layout_opt,
  const std::optional<bool> &output_final_state_opt, bool safe_gate, double lower_bound, bool use_gate_in_kernel,
  const std::optional<ms::Tensor> &a_log_opt, const std::optional<ms::Tensor> &dt_bias_opt,
  const std::optional<ms::Tensor> &initial_state_opt, const std::optional<std::vector<int64_t>> &cu_seqlens_opt,
  const std::optional<std::vector<int64_t>> &chunk_indices_opt, bool state_v_first) {
  (void)output_final_state_opt;
  const std::string layout = layout_opt.value_or("BNSD");
  ValidateForwardOptions(layout, chunk_size, cu_seqlens_opt, chunk_indices_opt);

  const auto q_shape = q.shape();
  const auto v_shape = v.shape();
  const bool rank3 = q_shape.size() == 3U;
  ValidateForwardRanks(q, k, v, g, beta, layout);

  const int64_t batch = rank3 ? 1 : q_shape[0];
  const int64_t heads = rank3 ? q_shape[0] : q_shape[1];
  const int64_t tokens = rank3 ? q_shape[1] : q_shape[2];
  const int64_t key_dim = rank3 ? q_shape[2] : q_shape[3];
  const int64_t value_heads = rank3 ? v_shape[0] : v_shape[1];
  const int64_t value_dim = rank3 ? v_shape[2] : v_shape[3];
  ValidateFeatureDimensions(key_dim, value_dim, heads, value_heads);
  ValidateVarlenDimensions(rank3, cu_seqlens_opt, batch);
  ValidateForwardInputShapes(v_shape, g, beta, rank3, batch, tokens, key_dim, value_heads, value_dim);
  const std::vector<int64_t> value_shape = rank3 ? std::vector<int64_t>({value_heads, tokens, value_dim})
                                                 : std::vector<int64_t>({batch, value_heads, tokens, value_dim});
  const std::vector<int64_t> gate_shape = rank3 ? std::vector<int64_t>({value_heads, tokens, key_dim})
                                                : std::vector<int64_t>({batch, value_heads, tokens, key_dim});

  const int64_t sequence_count =
    cu_seqlens_opt.has_value() ? static_cast<int64_t>(cu_seqlens_opt.value().size()) - 1 : batch;
  const int64_t chunk_count = CountChunks(cu_seqlens_opt, tokens, chunk_size);
  const std::vector<int64_t> matrix_shape = rank3 ? std::vector<int64_t>({value_heads, tokens, chunk_size})
                                                  : std::vector<int64_t>({batch, value_heads, tokens, chunk_size});
  const std::vector<int64_t> attn_shape = rank3 ? std::vector<int64_t>({tokens, value_heads, value_dim})
                                                : std::vector<int64_t>({batch, tokens, value_heads, value_dim});
  const std::vector<int64_t> state_dims =
    state_v_first ? std::vector<int64_t>({value_dim, key_dim}) : std::vector<int64_t>({key_dim, value_dim});
  const std::vector<int64_t> h_shape =
    rank3 ? std::vector<int64_t>({chunk_count, value_heads, state_dims[0], state_dims[1]})
          : std::vector<int64_t>({batch, chunk_count, value_heads, state_dims[0], state_dims[1]});

  ms::Tensor attn_out(v.data_type(), attn_shape);
  ms::Tensor final_state(ms::TypeId::kNumberTypeFloat32, {sequence_count, value_heads, state_dims[0], state_dims[1]});
  ms::Tensor gk(ms::TypeId::kNumberTypeFloat32, gate_shape);
  ms::Tensor aqk(q.data_type(), matrix_shape);
  ms::Tensor akk(q.data_type(), matrix_shape);
  ms::Tensor w(q.data_type(), gate_shape);
  ms::Tensor u(v.data_type(), value_shape);
  ms::Tensor qg(q.data_type(), gate_shape);
  ms::Tensor kg(q.data_type(), gate_shape);
  ms::Tensor v_new(v.data_type(), value_shape);
  ms::Tensor h(q.data_type(), h_shape);

  ms::TensorToDevice(q, k, v, g, beta, a_log_opt.value_or(ms::Tensor()), dt_bias_opt.value_or(ms::Tensor()),
                     initial_state_opt.value_or(ms::Tensor()));
  ms::TensorAllocate({attn_out, final_state, gk, aqk, akk, w, u, qg, kg, v_new, h});
  MS_DISPATCH_ACLNN(aclnnChunkKdaFwd, q, k, v, g, beta, a_log_opt, dt_bias_opt, initial_state_opt,
                    OptionalIntArray(cu_seqlens_opt), OptionalIntArray(chunk_indices_opt), layout, scale, chunk_size,
                    safe_gate, lower_bound, use_gate_in_kernel, state_v_first, attn_out, final_state, gk, aqk, akk, w,
                    u, qg, kg, v_new, h);
  return {attn_out, final_state, gk, aqk, akk, w, u, qg, kg, v_new, h};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) { m.def("npu_chunk_kda_fwd", PYBOOST_CALLER(11, custom::npu_chunk_kda_fwd)); }

}  // namespace custom
