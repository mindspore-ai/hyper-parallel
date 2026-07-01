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

#include <string>
#include <tuple>
#include <vector>
#include <optional>
#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {
std::tuple<ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor> GenResultTensor(const ms::Tensor &query_index,
                                                                           const ms::Tensor &key_index,
                                                                           const ms::Tensor &weights) {
  auto d_query_index = ms::Tensor(query_index.data_type(), query_index.shape());
  auto d_key_index = ms::Tensor(key_index.data_type(), key_index.shape());
  auto d_weights = ms::Tensor(weights.data_type(), weights.shape());
  auto loss = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>({1}));
  return std::make_tuple(std::move(d_query_index), std::move(d_key_index), std::move(d_weights), std::move(loss));
}
}  // namespace

std::vector<ms::Tensor> npu_dense_lightning_indexer_grad_kl_loss(
  const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &query_index, const ms::Tensor &key_index,
  const ms::Tensor &weights, const ms::Tensor &softmax_max, const ms::Tensor &softmax_sum,
  const ms::Tensor &softmax_max_index, const ms::Tensor &softmax_sum_index, double scale_value,
  const std::optional<ms::Tensor> &query_rope_opt, const std::optional<ms::Tensor> &key_rope_opt,
  const std::optional<std::vector<int64_t>> &actual_seq_qlen_opt,
  const std::optional<std::vector<int64_t>> &actual_seq_klen_opt, const std::optional<std::string> &layout_opt,
  const std::optional<int64_t> &sparse_mode_opt, const std::optional<int64_t> &pre_tokens_opt,
  const std::optional<int64_t> &next_tokens_opt) {
  auto [d_query_index, d_key_index, d_weights, loss] = GenResultTensor(query_index, key_index, weights);

  auto query_rope = query_rope_opt.value_or(ms::Tensor());
  auto key_rope = key_rope_opt.value_or(ms::Tensor());
  auto actual_seq_qlen = std::make_pair(actual_seq_qlen_opt, true);
  auto actual_seq_klen = std::make_pair(actual_seq_klen_opt, true);
  std::string layout = layout_opt.value_or("BSND");
  constexpr int64_t default_max = 9223372036854775807;
  int64_t sparse_mode = sparse_mode_opt.value_or(3);
  int64_t pre_tokens = pre_tokens_opt.value_or(default_max);
  int64_t next_tokens = next_tokens_opt.value_or(default_max);

  ms::TensorToDevice(query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index,
                     softmax_sum_index, query_rope, key_rope);
  ms::TensorAllocate({d_query_index, d_key_index, d_weights, loss});

  MS_DISPATCH_ACLNN(aclnnDenseLightningIndexerGradKLLoss, query, key, query_index, key_index, weights, softmax_max,
                    softmax_sum, softmax_max_index, softmax_sum_index, query_rope_opt, key_rope_opt, actual_seq_qlen,
                    actual_seq_klen, scale_value, layout, sparse_mode, pre_tokens, next_tokens, d_query_index,
                    d_key_index, d_weights, loss);
  return {d_query_index, d_key_index, d_weights, loss};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("npu_dense_lightning_indexer_grad_kl_loss",
        PYBOOST_CALLER(4, custom::npu_dense_lightning_indexer_grad_kl_loss));
}
}  // namespace custom
