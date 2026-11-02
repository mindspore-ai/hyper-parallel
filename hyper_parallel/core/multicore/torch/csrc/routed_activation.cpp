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
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include <array>
#include <cmath>
#include <limits>

#include "activation.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "cached_op_api.h"

namespace {

void CheckActivation(const at::Tensor &input, const at::Tensor &output, bool backward) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 && input.dim() == 2 &&
                input.size(1) > 0 && input.size(1) % 32 == 0,
              "routed activation requires a packed NPU gate/up matrix");
  TORCH_CHECK(output.sizes() == at::IntArrayRef({input.size(0), backward ? input.size(1) : input.size(1) / 2}),
              "routed activation output must have the exact forward/backward extent");
  at::assert_no_overlap(input, output);
}

void Record(const at::Tensor &value, const at::Tensor &anchor) {
  TORCH_CHECK(value.device() == anchor.device() && value.scalar_type() == at::kBFloat16 && value.is_contiguous(),
              "routed activation tensors must be contiguous BF16 on one NPU");
  if (value.numel()) {
    const auto capacity = value.storage().nbytes() / value.element_size();
    TORCH_CHECK(value.data_ptr() != nullptr && value.storage_offset() >= 0 &&
                  static_cast<uint64_t>(value.storage_offset()) <= capacity &&
                  static_cast<uint64_t>(value.numel()) <= capacity - value.storage_offset(),
                "routed activation storage has been invalidated or undersized");
    c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(),
                                              c10_npu::getCurrentNPUStream(anchor.device().index()));
  }
}

void Forward(const at::Tensor &input, at::Tensor output) {
  CheckActivation(input, output, false);
  const c10_npu::NPUGuard guard(input.device());
  Record(input, input);
  Record(output, input);
  if (input.numel()) {
    // Use the baseline's rounding while keeping activation storage in the admitted frame.
    int64_t dimension = -1;
    HYPER_EXEC_NPU_CMD_EXT(aclnnSwiGlu, input, dimension, output);
  }
}

void Backward(const at::Tensor &gradient, const at::Tensor &input, at::Tensor output) {
  CheckActivation(input, output, true);
  TORCH_CHECK(gradient.sizes() == at::IntArrayRef({input.size(0), input.size(1) / 2}),
              "routed activation derivative has an invalid extent");
  at::assert_no_overlap(gradient, output);
  const c10_npu::NPUGuard guard(input.device());
  Record(input, input);
  Record(gradient, input);
  Record(output, input);
  if (input.numel()) {
    int64_t dimension = -1;
    HYPER_EXEC_NPU_CMD_EXT(aclnnSwiGluGrad, gradient, input, dimension, output);
  }
}

void CheckClipped(const at::Tensor &input, const at::Tensor &output, double limit, bool backward) {
  CheckActivation(input, output, backward);
  TORCH_CHECK(input.size(0) <= (1LL << 20) && input.size(1) / 2 <= 32768 &&
                std::isfinite(limit) && limit > 0 && limit <= std::numeric_limits<float>::max(),
              "clipped routed activation requires at most 2**20 rows, inner width at most 32768, "
              "and a finite positive float32 limit");
}

void ClippedForward(const at::Tensor &input, at::Tensor output, double limit) {
  CheckClipped(input, output, limit, false);
  const c10_npu::NPUGuard guard(input.device());
  Record(input, input);
  Record(output, input);
  if (input.numel()) {
    const auto stream = c10_npu::getCurrentNPUStream(input.device().index());
    at_npu::native::OpCommand::RunOpApi("RoutedExpertsClippedSwiGlu", [=]() -> int {
      return RoutedExpertsClippedSwiGlu(stream.stream(false), input.data_ptr(), nullptr,
                                        output.data_ptr(), input.size(0), input.size(1) / 2,
                                        static_cast<float>(limit), 48);
    });
  }
}

void ClippedBackward(const at::Tensor &gradient, const at::Tensor &input, at::Tensor output, double limit) {
  CheckClipped(input, output, limit, true);
  TORCH_CHECK(gradient.sizes() == at::IntArrayRef({input.size(0), input.size(1) / 2}),
              "clipped routed activation derivative has an invalid extent");
  at::assert_no_overlap(gradient, output);
  const c10_npu::NPUGuard guard(input.device());
  Record(input, input);
  Record(gradient, input);
  Record(output, input);
  if (input.numel()) {
    const auto stream = c10_npu::getCurrentNPUStream(input.device().index());
    at_npu::native::OpCommand::RunOpApi("RoutedExpertsClippedSwiGluBackward", [=]() -> int {
      return RoutedExpertsClippedSwiGlu(stream.stream(false), input.data_ptr(), gradient.data_ptr(),
                                        output.data_ptr(), input.size(0), input.size(1) / 2,
                                        static_cast<float>(limit), 48);
    });
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_parallel, m) {
  m.def("routed_experts_swiglu(Tensor input, Tensor(a!) output) -> ()");
  m.def("routed_experts_swiglu_backward(Tensor gradient, Tensor input, Tensor(a!) output) -> ()");
  m.def("routed_experts_clipped_swiglu(Tensor input, Tensor(a!) output, float limit) -> ()");
  m.def("routed_experts_clipped_swiglu_backward(Tensor gradient, Tensor input, Tensor(a!) output, float limit) -> ()");
}

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("routed_experts_swiglu", &Forward);
  m.impl("routed_experts_swiglu_backward", &Backward);
  m.impl("routed_experts_clipped_swiglu", &ClippedForward);
  m.impl("routed_experts_clipped_swiglu_backward", &ClippedBackward);
}
