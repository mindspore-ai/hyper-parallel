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

/**
 * TND varlen FlashAttention through aclnn **V4**, whose only added knob we need is
 * ``softmaxOutLayout`` / ``softmaxInLayout``.
 *
 * Why this exists at all: under TND the stock ``aclnnFlashAttentionVarLenScore`` (v1, which is
 * what MindSpore's ``FlashAttentionScore`` primitive dispatches) emits the softmax statistics in
 * **NTD** order -- per document, head-major: ``[doc0 head0 l0 rows, doc0 head1 l0 rows, ...]``.
 * Every consumer that wants ``(N, T)`` therefore has to re-pack them using ``actual_seq_len``,
 * which means it has to know **where this chunk sits in the global sequence**. That single
 * requirement is what makes the statistics fragile under any wrapper that swaps a cell's
 * ``construct`` and calls it per block (CP head-tail fold / load balance): the wrapper hands the
 * consumer a different chunk than it assumes, and the result is silently wrong -- the attention
 * output is unaffected, so only a downstream loss that reads the statistics shows it.
 *
 * With ``softmaxOutLayout = "same_as_input"`` the statistics come out in TND order, so the
 * consumer only transposes and needs no global position at all.
 *
 * Two facts worth keeping in the code rather than in someone's head:
 *   - the string must be the literal ``"same_as_input"``; the tiling does
 *     ``strcmp(softmaxOutLayout, "same_as_input")`` (flash_attention_score_tiling_general.cpp),
 *     so ``"TND"`` is accepted without error and simply does nothing;
 *   - the backward's ``softmaxInLayout`` must agree with how the statistics were produced:
 *     ``"same_as_input"`` when they really are TND, ``""`` when the shape is TND but the data is
 *     still NTD. Getting this pair out of sync is again silent.
 *
 * Ascend 950PR/950DT do not support ``softmaxInLayout`` yet, so callers must keep the re-packing
 * path available rather than deleting it.
 */

#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "ms_extension/all.h"
#include "module.h"

namespace custom {
namespace {
constexpr int64_t kSoftmaxInnerDim = 8;
constexpr double kKeepProb = 1.0;

// Optional inputs must be moved to the device as well: handing ``std::optional`` straight to the
// dispatcher without ``TensorToDevice`` leaves the kernel with an address that is not ready, and
// it surfaces as an aicore exception rather than an argument error.
ms::Tensor EmptyTensor(const std::optional<ms::Tensor> &tensor_opt) { return tensor_opt.value_or(ms::Tensor()); }

// TND: query is (T, N, D); the statistics are (T, N, 8) either way -- only the *order* of the
// elements changes with softmaxOutLayout, not the shape.
std::vector<int64_t> SoftmaxStatShape(const ms::Tensor &query) {
  auto shape = query.shape();
  if (!shape.empty()) {
    shape.back() = kSoftmaxInnerDim;
  }
  return shape;
}

// attention_out follows value's head dim (MLA has q/k at 192 and v at 128).
std::vector<int64_t> AttnOutShape(const ms::Tensor &query, const ms::Tensor &value) {
  auto shape = query.shape();
  const auto v_shape = value.shape();
  if (!shape.empty() && !v_shape.empty()) {
    shape.back() = v_shape.back();
  }
  return shape;
}

std::string SoftmaxLayout(bool tnd_softmax) { return tnd_softmax ? std::string("same_as_input") : std::string(); }

const std::optional<std::vector<int64_t>> kNoIntArray = std::nullopt;
}  // namespace

std::vector<ms::Tensor> npu_flash_attention_varlen_v4(
  const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value,
  const std::optional<ms::Tensor> &atten_mask_opt, const std::vector<int64_t> &actual_seq_qlen,
  const std::vector<int64_t> &actual_seq_kvlen, double scale_value, int64_t head_num, int64_t sparse_mode,
  int64_t pre_tokens, int64_t next_tokens, int64_t inner_precise, bool tnd_softmax_out) {
  std::string layout = "TND";
  std::string softmax_out_layout = SoftmaxLayout(tnd_softmax_out);
  auto softmax_max = ms::Tensor(ms::TypeId::kNumberTypeFloat32, SoftmaxStatShape(query));
  auto softmax_sum = ms::Tensor(ms::TypeId::kNumberTypeFloat32, SoftmaxStatShape(query));
  auto attention_out = ms::Tensor(query.data_type(), AttnOutShape(query, value));

  auto prefix = std::make_pair(kNoIntArray, true);
  auto actual_q = std::make_pair(std::optional<std::vector<int64_t>>(actual_seq_qlen), true);
  auto actual_kv = std::make_pair(std::optional<std::vector<int64_t>>(actual_seq_kvlen), true);
  std::optional<ms::Tensor> none_tensor = std::nullopt;

  auto atten_mask = EmptyTensor(atten_mask_opt);
  ms::TensorToDevice(query, key, value, atten_mask);
  ms::TensorAllocate({softmax_max, softmax_sum, attention_out});
  MS_DISPATCH_ACLNN(aclnnFlashAttentionVarLenScoreV4, query, key, value, none_tensor /*pse*/,
                    none_tensor /*dropMask*/, none_tensor /*paddingMask*/, atten_mask_opt, prefix, actual_q,
                    actual_kv, scale_value, kKeepProb, pre_tokens, next_tokens, head_num, layout, inner_precise,
                    sparse_mode, softmax_out_layout, softmax_max, softmax_sum, none_tensor /*softmaxOut*/,
                    attention_out);
  return {softmax_max, softmax_sum, attention_out};
}

std::vector<ms::Tensor> npu_flash_attention_varlen_grad_v4(
  const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, const ms::Tensor &dy,
  const std::optional<ms::Tensor> &atten_mask_opt, const ms::Tensor &softmax_max, const ms::Tensor &softmax_sum,
  const ms::Tensor &attention_in, const std::vector<int64_t> &actual_seq_qlen,
  const std::vector<int64_t> &actual_seq_kvlen, double scale_value, int64_t head_num, int64_t sparse_mode,
  int64_t pre_tokens, int64_t next_tokens, int64_t inner_precise, bool tnd_softmax_in) {
  std::string layout = "TND";
  std::string softmax_in_layout = SoftmaxLayout(tnd_softmax_in);
  auto dq = ms::Tensor(query.data_type(), query.shape());
  auto dk = ms::Tensor(key.data_type(), key.shape());
  auto dv = ms::Tensor(value.data_type(), value.shape());

  auto prefix = std::make_pair(kNoIntArray, true);
  auto actual_q = std::make_pair(std::optional<std::vector<int64_t>>(actual_seq_qlen), true);
  auto actual_kv = std::make_pair(std::optional<std::vector<int64_t>>(actual_seq_kvlen), true);
  std::optional<ms::Tensor> none_tensor = std::nullopt;

  auto atten_mask = EmptyTensor(atten_mask_opt);
  ms::TensorToDevice(query, key, value, dy, atten_mask, softmax_max, softmax_sum, attention_in);
  ms::TensorAllocate({dq, dk, dv});
  MS_DISPATCH_ACLNN(aclnnFlashAttentionUnpaddingScoreGradV4, query, key, value, dy, none_tensor /*pseShift*/,
                    none_tensor /*dropMask*/, none_tensor /*paddingMask*/, atten_mask_opt, softmax_max, softmax_sum,
                    none_tensor /*softmaxIn*/, attention_in, prefix, actual_q, actual_kv, scale_value, kKeepProb,
                    pre_tokens, next_tokens, head_num, layout, inner_precise, sparse_mode, dq, dk, dv,
                    none_tensor /*dpse*/, softmax_in_layout);
  return {dq, dk, dv};
}

// cppcheck-suppress syntaxError
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("npu_flash_attention_varlen_v4", PYBOOST_CALLER(3, custom::npu_flash_attention_varlen_v4));
  m.def("npu_flash_attention_varlen_grad_v4", PYBOOST_CALLER(3, custom::npu_flash_attention_varlen_grad_v4));
}
}  // namespace custom
