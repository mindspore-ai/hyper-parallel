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

#include <array>
#include <cstdint>
#include <limits>

#include "combine_backward.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace {

void CheckStorage(const at::Tensor &value, const c10::Device &device) {
  TORCH_CHECK(value.device() == device && value.is_contiguous(),
              "routed combine backward tensors must be contiguous on the same NPU");
  if (value.numel() == 0) {
    return;
  }
  const int64_t offset = value.storage_offset();
  const uint64_t elements = static_cast<uint64_t>(value.numel());
  const uint64_t capacity = value.storage().nbytes() / value.element_size();
  TORCH_CHECK(offset >= 0 && static_cast<uint64_t>(offset) <= capacity &&
                elements <= capacity - static_cast<uint64_t>(offset),
              "routed combine backward received invalidated or undersized tensor storage");
  TORCH_CHECK(value.data_ptr() != nullptr, "routed combine backward received a null nonempty tensor address");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(value.data_ptr()) % value.element_size() == 0,
              "routed combine backward tensor address is not aligned to its element size");
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
  const uint64_t left_bytes = static_cast<uint64_t>(left.numel()) * left.element_size();
  const uint64_t right_bytes = static_cast<uint64_t>(right.numel()) * right.element_size();
  TORCH_CHECK(left_bytes <= std::numeric_limits<uintptr_t>::max() - left_begin &&
                right_bytes <= std::numeric_limits<uintptr_t>::max() - right_begin,
              "routed combine backward tensor address range overflows");
  return left_begin < right_begin + right_bytes && right_begin < left_begin + left_bytes;
}

void CheckCombineBackwardMatrices(const at::Tensor &dy, const at::Tensor &values, const at::Tensor &scores) {
  TORCH_CHECK(dy.device().type() == c10::DeviceType::PrivateUse1 && dy.dim() == 2 && values.dim() == 2 &&
                scores.dim() == 2 && dy.scalar_type() == at::kBFloat16 && values.scalar_type() == at::kBFloat16 &&
                (scores.scalar_type() == at::kBFloat16 || scores.scalar_type() == at::kFloat),
              "routed combine backward requires BF16 NPU dy [N,H], values [N*K,H], and BF16/FP32 scores [N,K]");
}

void CheckCombineBackwardExtents(int64_t hidden, int64_t tokens, int64_t top_k, int64_t cores) {
  TORCH_CHECK(hidden > 0 && hidden <= 32768 && hidden % 16 == 0 && top_k > 0 &&
                top_k <= std::numeric_limits<int32_t>::max() && cores > 0 && cores <= 48,
              "routed combine backward requires H divisible by 16 and at most 32768, "
              "positive int32 K, and 1..48 cores");
  TORCH_CHECK(tokens <= std::numeric_limits<int64_t>::max() / top_k,
              "routed combine backward token times top-k extent overflows");
}

void CheckCombineBackwardAliases(const std::array<at::Tensor, 6> &arguments) {
  for (const size_t writable : {4U, 5U}) {
    for (size_t index = 0; index < arguments.size(); ++index) {
      TORCH_CHECK(writable == index || !Overlaps(arguments[writable], arguments[index]),
                  "routed combine backward outputs must not alias any other argument");
    }
  }
}

void CombineBackward(const at::Tensor &dy, const at::Tensor &values, const at::Tensor &order,
                      const at::Tensor &scores, at::Tensor dv, at::Tensor ds, int64_t cores) {
  CheckCombineBackwardMatrices(dy, values, scores);
  const int64_t tokens = dy.size(0);
  const int64_t hidden = dy.size(1);
  const int64_t top_k = scores.size(1);
  CheckCombineBackwardExtents(hidden, tokens, top_k, cores);
  const int64_t selections = tokens * top_k;
  TORCH_CHECK(selections <= std::numeric_limits<int64_t>::max() / hidden &&
                values.sizes() == at::IntArrayRef({selections, hidden}) && scores.size(0) == tokens &&
                order.dim() == 1 && order.scalar_type() == at::kLong && order.numel() == selections,
              "routed combine backward requires values [N*K,H] and int64 order [N*K] mapping into scores [N,K]");
  TORCH_CHECK(dv.scalar_type() == at::kBFloat16 && dv.sizes() == values.sizes() &&
                ds.scalar_type() == scores.scalar_type() && ds.sizes() == scores.sizes(),
              "routed combine backward outputs must be BF16 dv [N*K,H] and matching-dtype ds [N,K]");
  const std::array<at::Tensor, 6> arguments{dy, values, order, scores, dv, ds};
  for (const auto &argument : arguments) {
    CheckStorage(argument, dy.device());
  }
  CheckCombineBackwardAliases(arguments);
  TORCH_CHECK(c10_npu::current_device() == dy.device().index(),
              "routed combine backward current NPU device must match the tensor device");
  const auto stream = c10_npu::getCurrentNPUStream(dy.device().index());
  for (const auto &argument : arguments) {
    if (argument.numel() > 0) {
      c10_npu::NPUCachingAllocator::recordStream(argument.storage().data_ptr(), stream);
    }
  }
  if (tokens == 0) {
    return;
  }
  // The plan owns order's bijection and joins all producer streams before use.
  // Captured tensors retain queued host arguments; recordStream covers device use.
  at_npu::native::OpCommand::RunOpApi("RoutedExpertsCombineBackward", [=]() -> int {
    return RoutedExpertsCombineBackward(stream.stream(false), dy.data_ptr(), values.data_ptr(), order.data_ptr(),
      scores.data_ptr(), dv.data_ptr(), ds.data_ptr(), tokens, selections, static_cast<int32_t>(hidden),
      static_cast<int32_t>(top_k), static_cast<int32_t>(cores), scores.scalar_type() == at::kFloat);
  });
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_parallel, m) {
  m.def("routed_experts_combine_backward(Tensor dy, Tensor values, Tensor order, Tensor scores, "
        "Tensor(a!) dv, Tensor(b!) ds, int cores=48) -> ()");
}

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("routed_experts_combine_backward", &CombineBackward);
}
