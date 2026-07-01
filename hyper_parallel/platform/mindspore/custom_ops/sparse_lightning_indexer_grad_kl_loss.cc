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
#include <tuple>
#include <vector>
#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {
ms::Tensor GetTensorOrEmpty(const std::optional<ms::Tensor> &opt_tensor) {
  return opt_tensor.has_value() ? opt_tensor.value() : ms::Tensor();
}

static std::tuple<ShapeVector, ShapeVector, ShapeVector, ShapeVector> InferShape(const ShapeVector &query_index_shape,
                                                                                 const ShapeVector &key_index_shape,
                                                                                 const ShapeVector &weights_shape) {
  ShapeVector loss_shape{1};
  return {query_index_shape, key_index_shape, weights_shape, loss_shape};
}

static std::tuple<ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor> GenResultTensors(const ms::Tensor &query_index,
                                                                                   const ms::Tensor &key_index,
                                                                                   const ms::Tensor &weights) {
  auto [dqi_shape, dki_shape, dw_shape, loss_shape] =
    InferShape(query_index.shape(), key_index.shape(), weights.shape());

  ms::Tensor d_query_index(query_index.data_type(), dqi_shape);
  ms::Tensor d_key_index(key_index.data_type(), dki_shape);
  ms::Tensor d_weights(weights.data_type(), dw_shape);
  ms::Tensor loss(ms::TypeId::kNumberTypeFloat32, loss_shape);
  return {std::move(d_query_index), std::move(d_key_index), std::move(d_weights), std::move(loss)};
}
}  // namespace

std::vector<ms::Tensor> npu_sparse_lightning_indexer_grad_kl_loss(
  const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &query_index, const ms::Tensor &key_index,
  const ms::Tensor &weights, const ms::Tensor &sparse_indices, const ms::Tensor &softmax_max,
  const ms::Tensor &softmax_sum, double scale_value, const std::optional<ms::Tensor> &query_rope,
  const std::optional<ms::Tensor> &key_rope, const std::optional<std::vector<int64_t>> &actual_seq_qlen,
  const std::optional<std::vector<int64_t>> &actual_seq_klen, const std::optional<std::string> &layout,
  const std::optional<int64_t> &sparse_mode, const std::optional<int64_t> &pre_tokens,
  const std::optional<int64_t> &next_tokens) {
  auto [d_query_index, d_key_index, d_weights, loss] = GenResultTensors(query_index, key_index, weights);

  auto query_rope_t = GetTensorOrEmpty(query_rope);
  auto key_rope_t = GetTensorOrEmpty(key_rope);
  auto actual_seq_qlen_pair = std::make_pair(actual_seq_qlen, true);
  auto actual_seq_klen_pair = std::make_pair(actual_seq_klen, true);
  std::string layout_str = layout.value_or("BSND");
  int64_t sparse_mode_val = sparse_mode.value_or(3);
  constexpr int64_t default_max = 9223372036854775807;
  int64_t pre_tokens_val = pre_tokens.value_or(default_max);
  int64_t next_tokens_val = next_tokens.value_or(default_max);
  bool deterministic = true;

  ms::TensorToDevice(query, key, query_index, key_index, weights, sparse_indices, softmax_max, softmax_sum,
                     query_rope_t, key_rope_t);
  ms::TensorAllocate({d_query_index, d_key_index, d_weights, loss});
  MS_DISPATCH_ACLNN(aclnnSparseLightningIndexerGradKLLoss, query, key, query_index, key_index, weights, sparse_indices,
                    softmax_max, softmax_sum, query_rope, key_rope, actual_seq_qlen_pair, actual_seq_klen_pair,
                    scale_value, layout_str, sparse_mode_val, pre_tokens_val, next_tokens_val, deterministic,
                    d_query_index, d_key_index, d_weights, loss);
  return {d_query_index, d_key_index, d_weights, loss};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("npu_sparse_lightning_indexer_grad_kl_loss",
        PYBOOST_CALLER(4, custom::npu_sparse_lightning_indexer_grad_kl_loss));
}
}  // namespace custom
