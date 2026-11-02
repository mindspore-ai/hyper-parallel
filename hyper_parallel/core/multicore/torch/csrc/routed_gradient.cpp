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
#include <limits>
#include <tuple>

#include "gradient.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace routed = hyper_parallel::multicore::routed_experts;

namespace {

void CheckWidth(int64_t width) {
  TORCH_CHECK(width > 0 && width <= routed::kGradientMaxWidth && width % routed::kGradientAlignment == 0,
              "routed gradient widths must be positive multiples of 16 no larger than 32768");
}

std::tuple<at::Tensor, int64_t> GradientConfig(int64_t features, int64_t cols, int64_t max_tokens,
                                            int64_t tile_features, int64_t tile_cols) {
  CheckWidth(features);
  CheckWidth(cols);
  TORCH_CHECK(max_tokens > 0 && max_tokens <= routed::kGradientMaxTokens && tile_features > 0 &&
                tile_features <= routed::kGradientMaxTile && tile_cols > 0 && tile_cols <= routed::kGradientMaxTile &&
                tile_features % routed::kGradientAlignment == 0 && tile_cols % routed::kGradientAlignment == 0,
              "invalid routed gradient reduction extent or tile dimensions");
  const auto config_bytes = static_cast<int64_t>(RoutedExpertsGradientConfigBytes());
  auto config = at::empty({config_bytes}, at::TensorOptions().device(at::kCPU).dtype(at::kByte));
  uint64_t written = 0, workspace_bytes = 0;
  const int status = RoutedExpertsGradientConfig(config.data_ptr(), config.numel(), features, cols, max_tokens,
                                                 tile_features, tile_cols, &written, &workspace_bytes);
  TORCH_CHECK(status == 0 && written == static_cast<uint64_t>(config_bytes) && workspace_bytes == 0,
              "independent routed gradient tiling failed: ", status);
  return {config, 0};
}

void CheckGradientMatrices(const at::Tensor &input, const at::Tensor &grad_output, const at::Tensor &output) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 && input.dim() == 2 && grad_output.dim() == 2 &&
                input.size(0) == grad_output.size(0) && input.size(0) <= routed::kGradientMaxTokens &&
                input.scalar_type() == at::kBFloat16 && grad_output.scalar_type() == at::kBFloat16,
              "routed gradient inputs must be BF16 NPU matrices with identical packed token extents");
  CheckWidth(input.size(1));
  CheckWidth(grad_output.size(1));
  TORCH_CHECK(output.dim() == 3 && output.scalar_type() == at::kFloat && output.size(1) == input.size(1) &&
                output.size(2) == grad_output.size(1) && output.size(0) <= std::numeric_limits<int32_t>::max(),
              "routed gradient output must be FP32 [active_experts, features, cols]");
}

void CheckGradientMetadata(const at::Tensor &input, const at::Tensor &boundaries, const at::Tensor &tasks,
                           const at::Tensor &output, const at::Tensor &config, int64_t cores) {
  TORCH_CHECK(boundaries.dim() == 1 && boundaries.scalar_type() == at::kLong &&
                boundaries.numel() == output.size(0) + 1 && tasks.dim() == 2 &&
                tasks.size(1) == routed::kGradientTaskFields && tasks.scalar_type() == at::kLong &&
                tasks.size(0) <= std::numeric_limits<int32_t>::max(),
              "routed gradient requires int64 boundaries [active_experts+1] and canonical tasks [tiles,7]");
  TORCH_CHECK(config.dim() == 1 && config.scalar_type() == at::kByte &&
                config.numel() == static_cast<int64_t>(RoutedExpertsGradientConfigBytes()) && cores > 0 && cores <= 24,
              "routed gradient has an invalid configuration extent or cube core count");
  TORCH_CHECK((output.size(0) == 0) == (input.size(0) == 0) &&
                (tasks.size(0) == 0) == (output.size(0) == 0),
              "routed gradient requires only nonempty active experts and complete output coverage");
}

void Gradient(const at::Tensor &input, const at::Tensor &grad_output, const at::Tensor &boundaries,
              const at::Tensor &tasks, at::Tensor output, const at::Tensor &config, int64_t cores) {
  CheckGradientMatrices(input, grad_output, output);
  CheckGradientMetadata(input, boundaries, tasks, output, config, cores);
  const c10_npu::NPUGuard device_guard(input.device());
  const auto stream = c10_npu::getCurrentNPUStream(input.device().index());
  const std::array<at::Tensor, 6> values{input, grad_output, boundaries, tasks, output, config};
  for (const auto &value : values) {
    TORCH_CHECK(value.device() == input.device() && value.is_contiguous(),
                "routed gradient arguments must be contiguous on one NPU");
    if (value.numel() > 0) {
      TORCH_CHECK(value.data_ptr() != nullptr && value.storage().nbytes() >=
                    (value.storage_offset() + value.numel()) * value.element_size(),
                  "routed gradient received an invalidated or undersized storage");
      c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(), stream);
    }
  }
  for (const size_t index : {0U, 1U, 2U, 3U, 5U}) {
    TORCH_CHECK(output.numel() == 0 || values[index].numel() == 0 || !output.is_alias_of(values[index]),
                "routed gradient output storage must not overlap any input argument");
  }
  for (const size_t index : {0U, 1U, 4U}) {
    TORCH_CHECK(values[index].numel() == 0 || reinterpret_cast<uintptr_t>(values[index].data_ptr()) % 32 == 0,
                "routed gradient matrix addresses must be aligned to 32 bytes");
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(config.data_ptr()) % alignof(routed::GradientConfigHeader) == 0,
              "routed gradient configuration address has an invalid alignment");
  if (tasks.size(0) == 0) {
    return;
  }
  at_npu::native::OpCommand::RunOpApi("RoutedExpertsGradient", [=]() -> int {
    return RoutedExpertsGradient(stream.stream(false), input.data_ptr(), grad_output.data_ptr(), boundaries.data_ptr(),
      tasks.data_ptr(), output.data_ptr(), config.data_ptr(), input.size(0), input.size(1), grad_output.size(1),
      output.size(0), tasks.size(0), cores);
  });
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_parallel, m) {
  m.def("routed_experts_gradient_config(int features, int cols, int max_tokens, int tile_features=128, "
        "int tile_cols=256) -> (Tensor, int)");
  m.impl("routed_experts_gradient_config", &GradientConfig);
  m.def("routed_experts_gradient(Tensor input, Tensor grad_output, Tensor boundaries, Tensor tasks, "
        "Tensor(a!) output, Tensor config, int cores=24) -> ()");
}

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("routed_experts_gradient", &Gradient);
}
