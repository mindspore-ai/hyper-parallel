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

#include <initializer_list>
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

std::vector<int64_t> BatchShape(bool rank3, int64_t batch, std::initializer_list<int64_t> dimensions) {
  std::vector<int64_t> shape;
  if (!rank3) {
    shape.push_back(batch);
  }
  shape.insert(shape.end(), dimensions.begin(), dimensions.end());
  return shape;
}

std::vector<int64_t> StateShape(bool rank3, int64_t batch, int64_t chunks, int64_t heads,
                                const std::vector<int64_t> &state_dims) {
  if (rank3) {
    return {chunks, heads, state_dims[0], state_dims[1]};
  }
  return {batch, chunks, heads, state_dims[0], state_dims[1]};
}

int64_t SequenceCount(const std::optional<std::vector<int64_t>> &cu_seqlens_opt, int64_t batch) {
  return cu_seqlens_opt.has_value() ? static_cast<int64_t>(cu_seqlens_opt.value().size()) - 1 : batch;
}

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

void ValidateOutputShapeInputs(const ms::Tensor &q, const ms::Tensor &v, int64_t chunk_size,
                               const std::optional<std::vector<int64_t>> &cu_seqlens_opt) {
  const size_t rank = q.shape().size();
  if ((rank != 3U && rank != 4U) || v.shape().size() != rank) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_fwd: q and v must have the same rank of 3 or 4.";
  }
  if (chunk_size <= 0) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_fwd: chunk_size must be greater than 0.";
  }
  if (cu_seqlens_opt.has_value() && cu_seqlens_opt.value().empty()) {
    MS_LOG(EXCEPTION) << "npu_chunk_kda_fwd: cu_seqlens must not be empty.";
  }
}

struct ChunkKdaFwdOutputs {
  ms::Tensor attn_out;
  ms::Tensor final_state;
  ms::Tensor gk;
  ms::Tensor aqk;
  ms::Tensor akk;
  ms::Tensor w;
  ms::Tensor u;
  ms::Tensor qg;
  ms::Tensor kg;
  ms::Tensor v_new;
  ms::Tensor h;
  ms::Tensor q_hat;
  ms::Tensor k_hat;
  ms::Tensor q_rstd;
  ms::Tensor k_rstd;
  ms::Tensor beta_eff;
};

ChunkKdaFwdOutputs AllocateOutputs(const ms::Tensor &q, const ms::Tensor &v,
                                   const std::optional<std::vector<int64_t>> &cu_seqlens_opt, int64_t chunk_size,
                                   bool state_v_first, bool use_qk_l2norm_in_kernel, bool use_beta_sigmoid_in_kernel) {
  const auto q_shape = q.shape();
  const auto v_shape = v.shape();
  const bool rank3 = q_shape.size() == 3U;
  const int64_t batch = rank3 ? 1 : q_shape[0];
  const int64_t heads = rank3 ? q_shape[0] : q_shape[1];
  const int64_t tokens = rank3 ? q_shape[1] : q_shape[2];
  const int64_t key_dim = q_shape.back();
  const int64_t value_heads = rank3 ? v_shape[0] : v_shape[1];
  const int64_t value_dim = v_shape.back();
  const std::vector<int64_t> value_shape = BatchShape(rank3, batch, {value_heads, tokens, value_dim});
  const std::vector<int64_t> gate_shape = BatchShape(rank3, batch, {value_heads, tokens, key_dim});
  const int64_t sequence_count = SequenceCount(cu_seqlens_opt, batch);
  const int64_t chunk_count = CountChunks(cu_seqlens_opt, tokens, chunk_size);
  const std::vector<int64_t> matrix_shape = BatchShape(rank3, batch, {value_heads, tokens, chunk_size});
  const std::vector<int64_t> attn_shape = BatchShape(rank3, batch, {tokens, value_heads, value_dim});
  const std::vector<int64_t> state_dims =
    state_v_first ? std::vector<int64_t>({value_dim, key_dim}) : std::vector<int64_t>({key_dim, value_dim});
  const std::vector<int64_t> h_shape = StateShape(rank3, batch, chunk_count, value_heads, state_dims);
  std::vector<int64_t> q_aux_shape = q_shape;
  std::vector<int64_t> rstd_aux_shape = BatchShape(rank3, batch, {heads, tokens});
  std::vector<int64_t> beta_aux_shape = BatchShape(rank3, batch, {value_heads, tokens});
  if (!use_qk_l2norm_in_kernel) {
    q_aux_shape = {0};
    rstd_aux_shape = {0};
  }
  if (!use_beta_sigmoid_in_kernel) {
    beta_aux_shape = {0};
  }

  return {
    ms::Tensor(v.data_type(), attn_shape),
    ms::Tensor(ms::TypeId::kNumberTypeFloat32, {sequence_count, value_heads, state_dims[0], state_dims[1]}),
    ms::Tensor(ms::TypeId::kNumberTypeFloat32, gate_shape),
    ms::Tensor(q.data_type(), matrix_shape),
    ms::Tensor(q.data_type(), matrix_shape),
    ms::Tensor(q.data_type(), gate_shape),
    ms::Tensor(v.data_type(), value_shape),
    ms::Tensor(q.data_type(), gate_shape),
    ms::Tensor(q.data_type(), gate_shape),
    ms::Tensor(v.data_type(), value_shape),
    ms::Tensor(q.data_type(), h_shape),
    ms::Tensor(q.data_type(), q_aux_shape),
    ms::Tensor(q.data_type(), q_aux_shape),
    ms::Tensor(ms::TypeId::kNumberTypeFloat32, rstd_aux_shape),
    ms::Tensor(ms::TypeId::kNumberTypeFloat32, rstd_aux_shape),
    ms::Tensor(ms::TypeId::kNumberTypeFloat32, beta_aux_shape),
  };
}

std::vector<ms::Tensor> CommonOutputs(ChunkKdaFwdOutputs &outputs) {
  return {outputs.attn_out, outputs.final_state, outputs.gk, outputs.aqk,   outputs.akk, outputs.w,
          outputs.u,        outputs.qg,          outputs.kg, outputs.v_new, outputs.h};
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

  ValidateOutputShapeInputs(q, v, chunk_size, cu_seqlens_opt);
  auto outputs = AllocateOutputs(q, v, cu_seqlens_opt, chunk_size, state_v_first, false, false);

  ms::TensorToDevice(q, k, v, g, beta, a_log_opt.value_or(ms::Tensor()), dt_bias_opt.value_or(ms::Tensor()),
                     initial_state_opt.value_or(ms::Tensor()));
  auto common_outputs = CommonOutputs(outputs);
  ms::TensorAllocate(common_outputs);
  MS_DISPATCH_ACLNN(aclnnChunkKdaFwd, q, k, v, g, beta, a_log_opt, dt_bias_opt, initial_state_opt,
                    OptionalIntArray(cu_seqlens_opt), OptionalIntArray(chunk_indices_opt), layout, scale, chunk_size,
                    safe_gate, lower_bound, use_gate_in_kernel, state_v_first, outputs.attn_out, outputs.final_state,
                    outputs.gk, outputs.aqk, outputs.akk, outputs.w, outputs.u, outputs.qg, outputs.kg, outputs.v_new,
                    outputs.h);
  return common_outputs;
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) { m.def("npu_chunk_kda_fwd", PYBOOST_CALLER(11, custom::npu_chunk_kda_fwd)); }

std::vector<ms::Tensor> npu_chunk_kda_fwd_v2(
  const ms::Tensor &q, const ms::Tensor &k, const ms::Tensor &v, const ms::Tensor &g, const ms::Tensor &beta,
  double scale, int64_t chunk_size, const std::optional<std::string> &layout_opt,
  const std::optional<bool> &output_final_state_opt, bool safe_gate, double lower_bound, bool use_gate_in_kernel,
  const std::optional<ms::Tensor> &a_log_opt, const std::optional<ms::Tensor> &dt_bias_opt,
  const std::optional<ms::Tensor> &initial_state_opt, const std::optional<std::vector<int64_t>> &cu_seqlens_opt,
  const std::optional<std::vector<int64_t>> &chunk_indices_opt, bool state_v_first, double epsilon,
  bool use_qk_l2norm_in_kernel, bool use_beta_sigmoid_in_kernel, bool allow_neg_eigval, bool use_exp2) {
  const std::string layout = layout_opt.value_or("BNSD");
  ValidateOutputShapeInputs(q, v, chunk_size, cu_seqlens_opt);
  auto outputs = AllocateOutputs(q, v, cu_seqlens_opt, chunk_size, state_v_first, use_qk_l2norm_in_kernel,
                                 use_beta_sigmoid_in_kernel);
  auto all_outputs = CommonOutputs(outputs);
  all_outputs.insert(all_outputs.end(),
                     {outputs.q_hat, outputs.k_hat, outputs.q_rstd, outputs.k_rstd, outputs.beta_eff});

  ms::TensorToDevice(q, k, v, g, beta, a_log_opt.value_or(ms::Tensor()), dt_bias_opt.value_or(ms::Tensor()),
                     initial_state_opt.value_or(ms::Tensor()));
  ms::TensorAllocate(all_outputs);

  const std::optional<ms::Tensor> final_state_opt =
    output_final_state_opt.value_or(false) ? std::optional<ms::Tensor>(outputs.final_state) : std::nullopt;
  const std::optional<ms::Tensor> q_hat_opt =
    use_qk_l2norm_in_kernel ? std::optional<ms::Tensor>(outputs.q_hat) : std::nullopt;
  const std::optional<ms::Tensor> k_hat_opt =
    use_qk_l2norm_in_kernel ? std::optional<ms::Tensor>(outputs.k_hat) : std::nullopt;
  const std::optional<ms::Tensor> q_rstd_opt =
    use_qk_l2norm_in_kernel ? std::optional<ms::Tensor>(outputs.q_rstd) : std::nullopt;
  const std::optional<ms::Tensor> k_rstd_opt =
    use_qk_l2norm_in_kernel ? std::optional<ms::Tensor>(outputs.k_rstd) : std::nullopt;
  const std::optional<ms::Tensor> beta_eff_opt =
    use_beta_sigmoid_in_kernel ? std::optional<ms::Tensor>(outputs.beta_eff) : std::nullopt;
  MS_DISPATCH_ACLNN(aclnnChunkKdaFwdV2, q, k, v, g, beta, a_log_opt, dt_bias_opt, initial_state_opt,
                    OptionalIntArray(cu_seqlens_opt), OptionalIntArray(chunk_indices_opt), layout, scale, chunk_size,
                    safe_gate, lower_bound, use_gate_in_kernel, state_v_first, epsilon, use_qk_l2norm_in_kernel,
                    use_beta_sigmoid_in_kernel, allow_neg_eigval, use_exp2, outputs.attn_out, final_state_opt,
                    outputs.gk, outputs.aqk, outputs.akk, outputs.w, outputs.u, outputs.qg, outputs.kg, outputs.v_new,
                    outputs.h, q_hat_opt, k_hat_opt, q_rstd_opt, k_rstd_opt, beta_eff_opt);
  return all_outputs;
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) { m.def("npu_chunk_kda_fwd_v2", PYBOOST_CALLER(16, custom::npu_chunk_kda_fwd_v2)); }

}  // namespace custom
