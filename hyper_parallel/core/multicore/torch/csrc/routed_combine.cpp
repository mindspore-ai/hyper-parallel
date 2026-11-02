// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "combine.h"

namespace {

void CheckStorage(const at::Tensor &value, const c10::Device &device) {
  TORCH_CHECK(value.device() == device && value.is_contiguous(),
              "routed combine tensors must be contiguous on the same NPU");
  if (value.numel() == 0) {
    return;
  }
  const int64_t offset = value.storage_offset();
  const uint64_t elements = static_cast<uint64_t>(value.numel());
  const uint64_t capacity = value.storage().nbytes() / value.element_size();
  TORCH_CHECK(offset >= 0 && static_cast<uint64_t>(offset) <= capacity &&
                elements <= capacity - static_cast<uint64_t>(offset),
              "routed combine received invalidated or undersized tensor storage");
  TORCH_CHECK(value.data_ptr() != nullptr, "routed combine received a null nonempty tensor address");
}

bool Overlaps(const at::Tensor &left, const at::Tensor &right) {
  if (left.numel() == 0 || right.numel() == 0) {
    return false;
  }
  if (left.is_alias_of(right)) {
    return true;
  }
  const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left.data_ptr());
  const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right.data_ptr());
  const uint64_t left_bytes = left.numel() * left.element_size();
  const uint64_t right_bytes = right.numel() * right.element_size();
  TORCH_CHECK(left_bytes <= std::numeric_limits<uintptr_t>::max() - left_begin &&
                right_bytes <= std::numeric_limits<uintptr_t>::max() - right_begin,
              "routed combine tensor address range overflows");
  return left_begin < right_begin + right_bytes && right_begin < left_begin + left_bytes;
}

void CheckCombineMatrices(const at::Tensor &values, const at::Tensor &output) {
  TORCH_CHECK(values.device().type() == c10::DeviceType::PrivateUse1,
              "routed combine requires NPU tensors");
  TORCH_CHECK(values.dim() == 2 && output.dim() == 2 && values.size(1) == output.size(1) &&
                values.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16,
              "routed combine requires BF16 values/output matrices with the same hidden width");
}

void CheckCombineExtents(int64_t hidden, int64_t tokens, int64_t top_k, int64_t cores) {
  TORCH_CHECK(hidden > 0 && hidden <= 32768 && hidden % 16 == 0 && top_k > 0 &&
                top_k <= std::numeric_limits<int32_t>::max() && cores > 0 && cores <= 48,
              "routed combine requires H divisible by 16 and at most 32768, positive int32 top-k, and 1..48 cores");
  TORCH_CHECK(tokens <= std::numeric_limits<int64_t>::max() / top_k,
              "routed combine token times top-k extent overflows");
}

void Combine(const at::Tensor &values, const at::Tensor &inverse_order, const std::optional<at::Tensor> &scores,
              at::Tensor output, int64_t top_k, int64_t cores) {
  CheckCombineMatrices(values, output);
  const int64_t hidden = values.size(1);
  const int64_t tokens = output.size(0);
  CheckCombineExtents(hidden, tokens, top_k, cores);
  const int64_t selections = tokens * top_k;
  TORCH_CHECK(selections <= std::numeric_limits<int64_t>::max() / hidden,
              "routed combine selection times hidden extent overflows");
  TORCH_CHECK(values.size(0) == selections && inverse_order.scalar_type() == at::kLong &&
                ((inverse_order.dim() == 1 && inverse_order.size(0) == selections) ||
                 (inverse_order.dim() == 2 && inverse_order.sizes() == at::IntArrayRef({tokens, top_k}))),
              "routed combine inverse order must be int64 [N*K] or [N,K] and values must contain N*K rows");
  std::vector<at::Tensor> inputs{values, inverse_order};
  if (scores.has_value()) {
    TORCH_CHECK(scores->scalar_type() == at::kBFloat16 && scores->sizes() == at::IntArrayRef({tokens, top_k}),
                "routed combine scores must be BF16 [N,K]");
    inputs.push_back(*scores);
  }
  CheckStorage(output, values.device());
  for (const auto &value : inputs) {
    CheckStorage(value, values.device());
    TORCH_CHECK(!Overlaps(output, value), "routed combine output must not alias an input");
  }
  TORCH_CHECK(c10_npu::current_device() == values.device().index(),
              "routed combine current NPU device must match the tensor device");
  const auto stream = c10_npu::getCurrentNPUStream(values.device().index());
  inputs.push_back(output);
  for (const auto &value : inputs) {
    if (value.numel() > 0) {
      c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(), stream);
    }
  }
  if (tokens == 0) {
    return;
  }
  // The caller joins producer streams before this call. Captured tensors keep
  // queued host arguments alive; recordStream protects the device consumers.
  at_npu::native::OpCommand::RunOpApi("RoutedExpertsCombine", [=]() -> int {
    return RoutedExpertsCombine(stream.stream(false), values.data_ptr(), inverse_order.data_ptr(),
      scores.has_value() ? scores->data_ptr() : nullptr, output.data_ptr(), tokens, selections,
      static_cast<int32_t>(hidden), static_cast<int32_t>(top_k), static_cast<int32_t>(cores));
  });
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_parallel, m) {
  m.def("routed_experts_combine(Tensor values, Tensor inverse_order, Tensor? scores, Tensor(a!) output, "
        "int top_k, int cores=48) -> ()");
}

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("routed_experts_combine", &Combine);
}
