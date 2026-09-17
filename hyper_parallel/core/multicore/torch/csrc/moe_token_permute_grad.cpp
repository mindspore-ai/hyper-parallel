/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Metadata-only adapter for the existing CANN token permutation gradient.
 */
#include <ATen/Functions.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include "op_plugin/include/npu_cpp_extension.h"

namespace {

at::Tensor allocate_token_gradient(const at::Tensor &grad, const at::Tensor &sorted_indices, int64_t num_tokens,
                                   int64_t top_k) {
  TORCH_CHECK(grad.dim() == 2, "Permuted gradients must have shape [T * K, H].");
  TORCH_CHECK(num_tokens >= 0 && top_k > 0 && top_k <= 512, "Invalid token count or Top-K.");
  TORCH_CHECK(grad.size(0) % top_k == 0 && grad.size(0) / top_k == num_tokens,
              "Permuted gradient rows must equal T * K (dropless routing).");
  TORCH_CHECK(sorted_indices.dim() == 1 && sorted_indices.numel() == grad.size(0),
              "Permutation mapping must have T * K entries.");
  TORCH_CHECK(sorted_indices.scalar_type() == at::kInt, "Permutation mapping must be int32.");
  TORCH_CHECK(sorted_indices.device() == grad.device(), "Gradient and mapping devices must match.");
  TORCH_CHECK(
    grad.scalar_type() == at::kBFloat16 || grad.scalar_type() == at::kHalf || grad.scalar_type() == at::kFloat,
    "Gradient must be BF16, FP16 or FP32.");
  return at::empty({num_tokens, grad.size(1)}, grad.options());
}

at::Tensor moe_token_permute_grad_npu(const at::Tensor &grad, const at::Tensor &sorted_indices, int64_t num_tokens,
                                      int64_t top_k) {
  const c10::DeviceGuard device_guard(grad.device());
  auto output = allocate_token_gradient(grad, sorted_indices, num_tokens, top_k);
  if (output.numel() != 0) {
    bool padded_mode = false;
    EXEC_NPU_CMD_EXT(aclnnMoeTokenPermuteGrad, grad, sorted_indices, top_k, padded_mode, output);
  }
  return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_parallel, m) {
  m.def("moe_token_permute_grad(Tensor grad, Tensor sorted_indices, int num_tokens, int top_k) -> Tensor");
}

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) { m.impl("moe_token_permute_grad", &moe_token_permute_grad_npu); }

TORCH_LIBRARY_IMPL(hyper_parallel, Meta, m) { m.impl("moe_token_permute_grad", &allocate_token_gradient); }
