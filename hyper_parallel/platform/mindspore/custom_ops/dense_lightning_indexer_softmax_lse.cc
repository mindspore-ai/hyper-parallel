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
std::tuple<ms::Tensor, ms::Tensor> GenResultTensors(const ms::Tensor &query_index, const ms::Tensor &key_index,
                                                    const std::string &layout) {
  std::vector<int64_t> out_shape;
  if (layout == "TND") {
    out_shape = {key_index.shape()[1], query_index.shape()[0]};
  } else {
    out_shape = {query_index.shape()[0], key_index.shape()[2], query_index.shape()[1]};
  }
  ms::Tensor softmax_max_out(ms::TypeId::kNumberTypeFloat32, out_shape);
  ms::Tensor softmax_sum_out(ms::TypeId::kNumberTypeFloat32, out_shape);
  return {std::move(softmax_max_out), std::move(softmax_sum_out)};
}
}  // namespace

std::vector<ms::Tensor> npu_dense_lightning_indexer_softmax_lse(
  const ms::Tensor &query_index, const ms::Tensor &key_index, const ms::Tensor &weight,
  const std::optional<std::vector<int64_t>> &actual_seq_qlen,
  const std::optional<std::vector<int64_t>> &actual_seq_klen, const std::optional<std::string> &layout,
  const std::optional<int64_t> &sparse_mode, const std::optional<int64_t> &pre_tokens,
  const std::optional<int64_t> &next_tokens) {
  std::string layout_str = layout.value_or("BSND");
  int64_t sparse_mode_val = sparse_mode.value_or(3);
  int64_t pre_tokens_val = pre_tokens.value_or(9223372036854775807LL);
  int64_t next_tokens_val = next_tokens.value_or(9223372036854775807LL);

  auto actual_seq_qlen_pair = std::make_pair(actual_seq_qlen, true);
  auto actual_seq_klen_pair = std::make_pair(actual_seq_klen, true);

  auto [out0, out1] = GenResultTensors(query_index, key_index, layout_str);

  ms::TensorToDevice(query_index, key_index, weight);
  ms::TensorAllocate({out0, out1});
  MS_DISPATCH_ACLNN(aclnnDenseLightningIndexerSoftmaxLse, query_index, key_index, weight, actual_seq_qlen_pair,
                    actual_seq_klen_pair, layout_str, sparse_mode_val, pre_tokens_val, next_tokens_val, out0, out1);
  return {out0, out1};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("npu_dense_lightning_indexer_softmax_lse", PYBOOST_CALLER(2, custom::npu_dense_lightning_indexer_softmax_lse));
}
}  // namespace custom
