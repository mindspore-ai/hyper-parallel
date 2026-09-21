/*
 * Copyright 2026 Huawei Technologies Co., Ltd.
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
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include <tuple>
#include "cached_op_api.h"

namespace {

using UnpermuteOutputs = std::tuple<at::Tensor &, at::Tensor &>;


void check_unpermute_shapes(const at::Tensor &tokens, const at::Tensor &grad, const at::Tensor &indices,
                            const at::Tensor &probs, const at::Tensor &grad_tokens, const at::Tensor &grad_probs) {
  TORCH_CHECK(tokens.dim() == 2 && grad.dim() == 2 && probs.dim() == 2 && indices.dim() == 1,
              "unpermute requires matrix tokens, gradients and probabilities, and flat indices");
  TORCH_CHECK(tokens.sizes() == grad_tokens.sizes() && probs.sizes() == grad_probs.sizes(),
              "unpermute output shapes must match tokens and probabilities");
  TORCH_CHECK(tokens.size(1) == grad.size(1) && grad.size(0) == probs.size(0) && tokens.size(0) == probs.numel() &&
                indices.numel() == tokens.size(0),
              "unpermute requires all top-k rows without padding or dropped tokens");
}

void check_unpermute_outputs(const at::Tensor &tokens, const at::Tensor &grad, const at::Tensor &indices,
                             const at::Tensor &probs, const at::Tensor &grad_tokens, const at::Tensor &grad_probs) {
  check_unpermute_shapes(tokens, grad, indices, probs, grad_tokens, grad_probs);
  TORCH_CHECK(indices.scalar_type() == at::kInt && probs.scalar_type() == at::kFloat &&
                grad_probs.scalar_type() == probs.scalar_type() && grad_tokens.scalar_type() == tokens.scalar_type() &&
                grad.scalar_type() == tokens.scalar_type(),
              "unpermute requires int32 indices, float32 probabilities and matching token dtypes");
  for (const auto *tensor : {&grad, &indices, &probs, &grad_tokens, &grad_probs}) {
    TORCH_CHECK(tensor->device() == tokens.device(), "unpermute tensors must share one device");
  }
  TORCH_CHECK(tokens.is_contiguous() && indices.is_contiguous() && probs.is_contiguous() &&
                grad_tokens.is_contiguous() && grad_probs.is_contiguous(),
              "unpermute inputs and outputs must be contiguous except grad_output");
  at::assert_no_overlap(grad_tokens, grad_probs);
  for (const auto *input : {&tokens, &grad, &indices, &probs}) {
    at::assert_no_overlap(grad_tokens, *input);
    at::assert_no_overlap(grad_probs, *input);
  }
}

UnpermuteOutputs unpermute_grad_npu(const at::Tensor &tokens, const at::Tensor &grad, const at::Tensor &indices,
                                    const at::Tensor &probs, at::Tensor &grad_tokens, at::Tensor &grad_probs) {
  check_unpermute_outputs(tokens, grad, indices, probs, grad_tokens, grad_probs);
  auto contiguous_grad = grad.contiguous();
  const c10::optional<at::IntArrayRef> restore_shape = c10::nullopt;
  bool padded_mode = false;
  static const hyper_parallel::multicore::CachedOpApi api(
      "aclnnMoeTokenUnpermuteGrad", "aclnnMoeTokenUnpermuteGradGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(
      api, tokens, contiguous_grad, indices, probs, padded_mode, restore_shape, grad_tokens, grad_probs);
  return {grad_tokens, grad_probs};
}

UnpermuteOutputs unpermute_grad_meta(const at::Tensor &tokens, const at::Tensor &grad, const at::Tensor &indices,
                                     const at::Tensor &probs, at::Tensor &grad_tokens, at::Tensor &grad_probs) {
  check_unpermute_outputs(tokens, grad, indices, probs, grad_tokens, grad_probs);
  return {grad_tokens, grad_probs};
}

}  // namespace

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) { m.impl("mega_moe_unpermute_grad_out", &unpermute_grad_npu); }

TORCH_LIBRARY_IMPL(hyper_parallel, Meta, m) { m.impl("mega_moe_unpermute_grad_out", &unpermute_grad_meta); }
