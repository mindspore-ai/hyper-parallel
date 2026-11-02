// Copyright 2026 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <limits>
#include <vector>
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "forward.h"
#include <tuple>

namespace {

void CheckTensor(const at::Tensor &value, const at::Tensor &anchor, at::ScalarType dtype) {
  TORCH_CHECK(value.device() == anchor.device() && value.is_contiguous() && value.scalar_type() == dtype,
              "routed forward requires contiguous tensors of the expected dtype on one device");
  if (!value.numel()) {
    return;
  }
  const auto capacity = value.storage().nbytes() / value.element_size();
  TORCH_CHECK(value.storage_offset() >= 0 && static_cast<uint64_t>(value.storage_offset()) <= capacity &&
                static_cast<uint64_t>(value.numel()) <= capacity - value.storage_offset() &&
                reinterpret_cast<uintptr_t>(value.data_ptr()) % 32 == 0,
              "routed forward storage is undersized or unaligned");
}

void CheckAliases(const std::vector<at::Tensor> &values, std::initializer_list<size_t> writers) {
  for (const auto writer : writers) {
    for (size_t index = 0; index < values.size(); ++index) {
      if (writer == index || !values[writer].numel() || !values[index].numel()) {
        continue;
      }
      const auto start = reinterpret_cast<uintptr_t>(values[writer].data_ptr());
      const auto other = reinterpret_cast<uintptr_t>(values[index].data_ptr());
      const auto bytes = values[writer].numel() * values[writer].element_size();
      const auto other_bytes = values[index].numel() * values[index].element_size();
      const bool overlap = start <= other ? other - start < bytes : start - other < other_bytes;
      TORCH_CHECK(!values[writer].is_alias_of(values[index]) && !overlap, "routed forward output aliases an operand");
    }
  }
}

void *OptionalPointer(const std::optional<at::Tensor> &value) {
  return value ? value->data_ptr() : nullptr;
}

void CheckReadiness(const std::optional<at::Tensor> &input_ready,
                    const std::optional<at::Tensor> &input_expected,
                    const std::optional<at::Tensor> &output_ready, const at::Tensor &anchor,
                    int64_t experts, int64_t tasks, int64_t cores, std::vector<at::Tensor> &values) {
  TORCH_CHECK(input_ready.has_value() == input_expected.has_value(), "readiness inputs must occur together");
  for (const auto *flags : {&input_ready, &output_ready}) {
    if (!*flags) {
      continue;
    }
    CheckTensor(**flags, anchor, at::kInt);
    const auto &value = **flags;
    TORCH_CHECK(value.dim() == 3 && value.size(0) > 0 && value.size(0) <= 48 && value.size(1) == experts &&
                  value.size(2) == 16 && (!value.numel() || reinterpret_cast<uintptr_t>(value.data_ptr()) % 64 == 0),
                "invalid expert readiness cells");
    if (flags == &output_ready) {
      TORCH_CHECK(value.size(0) == cores, "output readiness must contain one cell per C producer");
    }
    values.push_back(value);
  }
  if (input_expected) {
    CheckTensor(*input_expected, anchor, at::kInt);
    TORCH_CHECK(input_expected->sizes() == at::IntArrayRef({input_ready->size(0), experts}) ||
                  input_expected->sizes() == at::IntArrayRef({input_ready->size(0), tasks}),
                "invalid readiness expectation matrix");
    values.push_back(*input_expected);
  }
}

void Record(const std::vector<at::Tensor> &values, c10_npu::NPUStream stream) {
  for (const auto &value : values) {
    if (value.numel()) {
      c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(), stream);
    }
  }
}

void CheckControls(const at::Tensor &tasks, const at::Tensor &mailboxes, const at::Tensor &z,
                   int64_t cores, int64_t tile_rows, int64_t watchdog) {
  CheckTensor(tasks, z, at::kLong);
  CheckTensor(mailboxes, z, at::kInt);
  TORCH_CHECK(tasks.dim() == 2 && tasks.size(1) == 3 && tasks.size(0) <= (1LL << 20), "invalid tile task table");
  TORCH_CHECK(cores > 0 && cores <= 24 && (tile_rows == 128 || tile_rows == 256) && watchdog > 0 &&
                mailboxes.sizes() == at::IntArrayRef({2, cores, 16}) &&
                reinterpret_cast<uintptr_t>(mailboxes.data_ptr()) % 64 == 0,
              "invalid tile mailbox, core count, chunk size, or watchdog");
  TORCH_CHECK(tasks.size(0) == 0 || tasks.size(0) >= cores, "idle cube cores must be removed from the launch");
}

std::tuple<at::Tensor, int64_t> Config(int64_t tile_rows, int64_t hidden, bool transpose) {
  TORCH_CHECK(tile_rows == 128 && hidden == 2048, "unsupported forward pipeline tiling dimensions");
  auto result = at::empty({static_cast<int64_t>(RoutedExpertsForwardConfigBytes())},
                          at::TensorOptions().dtype(at::kByte));
  const int status = RoutedExpertsForwardConfig(result.data_ptr(), result.numel(), tile_rows, hidden, transpose);
  TORCH_CHECK(status == 0, "routed forward tiling failed: ", status);
  return {result, 0};
}

int64_t ConfigBytes() {
  return static_cast<int64_t>(RoutedExpertsForwardConfigBytes());
}

void CheckWeights(at::TensorList weights, const at::Tensor &table, const at::Tensor &x,
                  int64_t rows, int64_t columns, bool transposed, std::vector<at::Tensor> &values) {
  // Tables are trusted prepared artifacts rebound from these references in the same phase.
  // Verifying device contents here would introduce an unwanted device-to-host synchronization.
  CheckTensor(table, x, at::kLong);
  TORCH_CHECK(table.dim() == 1 && table.numel() == static_cast<int64_t>(weights.size()),
              "forward pointer table must cover current weight references");
  for (const auto &weight : weights) {
    TORCH_CHECK(weight.dim() == 2 && weight.sizes() == at::IntArrayRef({rows, columns}),
                "forward weight has an invalid logical shape");
    const auto physical = transposed ? weight.transpose(0, 1) : weight;
    CheckTensor(physical, x, at::kBFloat16);
    values.push_back(weight);
  }
}

std::vector<at::Tensor> CheckCube(const at::Tensor &x, at::TensorList gate, at::TensorList down,
    const at::Tensor &gate_table, const at::Tensor &down_table, const at::Tensor &tasks, const at::Tensor &z,
    const at::Tensor &activation, const at::Tensor &y, const at::Tensor &config, const at::Tensor &mailboxes,
    int64_t tile_rows, bool transpose, int64_t cores, int64_t watchdog,
    const std::optional<at::Tensor> &input_ready, const std::optional<at::Tensor> &input_expected,
    const std::optional<at::Tensor> &output_ready) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.dim() == 2 && x.size(0) <= (1LL << 20) &&
                x.size(1) == 2048 && z.sizes() == at::IntArrayRef({x.size(0), 1536}) && y.sizes() == x.sizes() &&
                gate.size() <= 128 && gate.size() == down.size(), "invalid forward pipeline matrix shapes");
  TORCH_CHECK(tile_rows == 128 && activation.sizes() == at::IntArrayRef({cores, tile_rows, 768}),
              "invalid forward activation tile shape");
  for (const auto &value : {x, z, activation, y}) {
    CheckTensor(value, x, at::kBFloat16);
  }
  CheckTensor(config, x, at::kByte);
  TORCH_CHECK(config.dim() == 1 && config.numel() == static_cast<int64_t>(RoutedExpertsForwardConfigBytes()),
              "invalid forward configuration");
  CheckControls(tasks, mailboxes, x, cores, tile_rows, watchdog);
  std::vector<at::Tensor> values{x, tasks, z, activation, y, config, mailboxes, gate_table, down_table};
  CheckWeights(gate, gate_table, x, 2048, 1536, transpose, values);
  CheckWeights(down, down_table, x, 768, 2048, transpose, values);
  const size_t output_index = values.size() + static_cast<size_t>(input_ready.has_value());
  CheckReadiness(input_ready, input_expected, output_ready, x, gate.size(), tasks.size(0), cores, values);
  CheckAliases(values, {2, 3, 4, 6});
  if (output_ready) {
    CheckAliases(values, {output_index});
  }
  TORCH_CHECK(tasks.size(0) > 0 || x.size(0) == 0, "nonempty inputs require forward tasks");
  return values;
}

std::vector<at::Tensor> CheckVector(const at::Tensor &z, const at::Tensor &activation,
    const at::Tensor &tasks, const at::Tensor &mailboxes, int64_t experts, int64_t tile_rows,
    int64_t cores, int64_t vector_cores, int64_t watchdog) {
  TORCH_CHECK(z.device().type() == c10::DeviceType::PrivateUse1 && z.dim() == 2 && z.size(0) <= (1LL << 20) &&
                z.size(1) == 1536 && experts >= 0 && experts <= 128 && tile_rows == 128 &&
                activation.sizes() == at::IntArrayRef({cores, tile_rows, 768}) &&
                vector_cores > 0 && vector_cores <= 8 && vector_cores <= cores,
              "invalid forward activation service shapes or core counts");
  CheckTensor(z, z, at::kBFloat16);
  CheckTensor(activation, z, at::kBFloat16);
  CheckControls(tasks, mailboxes, z, cores, tile_rows, watchdog);
  std::vector<at::Tensor> values{z, activation, tasks, mailboxes};
  CheckAliases(values, {1, 3});
  TORCH_CHECK(tasks.size(0) > 0 || z.size(0) == 0, "nonempty inputs require forward tasks");
  return values;
}

void Validate(const at::Tensor &x, at::TensorList gate, at::TensorList down,
    const at::Tensor &gate_table, const at::Tensor &down_table, const at::Tensor &tasks, const at::Tensor &z,
    const at::Tensor &activation, const at::Tensor &y, const at::Tensor &config, const at::Tensor &mailboxes,
    int64_t tile_rows, bool transpose, int64_t cores, int64_t watchdog) {
  CheckCube(x, gate, down, gate_table, down_table, tasks, z, activation, y, config, mailboxes,
            tile_rows, transpose, cores, watchdog, std::nullopt, std::nullopt, std::nullopt);
  CheckVector(z, activation, tasks, mailboxes, gate.size(), tile_rows, cores, std::min<int64_t>(8, cores), watchdog);
}

void Cube(const at::Tensor &x, at::TensorList gate, at::TensorList down,
    const at::Tensor &gate_table, const at::Tensor &down_table, const at::Tensor &tasks, at::Tensor z,
    at::Tensor activation, at::Tensor y, const at::Tensor &config, at::Tensor mailboxes,
    int64_t tile_rows, bool transpose, int64_t cores, int64_t watchdog,
    const std::optional<at::Tensor> &input_ready, const std::optional<at::Tensor> &input_expected,
    const std::optional<at::Tensor> &output_ready) {
  auto keepalive = CheckCube(x, gate, down, gate_table, down_table, tasks, z, activation, y, config, mailboxes,
                             tile_rows, transpose, cores, watchdog, input_ready, input_expected, output_ready);
  const c10_npu::NPUGuard guard(x.device());
  if (tasks.size(0) == 0) {
    return;
  }
  const auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  Record(keepalive, stream);
  const int64_t experts = gate.size();
  at_npu::native::OpCommand::RunOpApi("RoutedExpertsForwardCube", [=]() -> int {
    static_cast<void>(keepalive);
    return RoutedExpertsForwardCube(stream.stream(false), x.data_ptr(), gate_table.data_ptr(),
      down_table.data_ptr(), tasks.data_ptr(), z.data_ptr(), activation.data_ptr(), y.data_ptr(),
      config.data_ptr(), mailboxes.data_ptr(), x.size(0), experts, 2048, 768, tasks.size(0),
      tile_rows, transpose, cores, watchdog, OptionalPointer(input_ready), OptionalPointer(input_expected),
      OptionalPointer(output_ready), input_ready ? input_ready->size(0) : 0,
      input_expected ? input_expected->size(1) : 0);
  });
}

void Vector(const at::Tensor &z, at::Tensor activation, const at::Tensor &tasks, at::Tensor mailboxes,
            int64_t experts, int64_t tile_rows, int64_t cores, int64_t vector_cores, int64_t watchdog,
            double clamp_limit) {
  TORCH_CHECK(std::isfinite(clamp_limit) && clamp_limit >= 0 &&
              clamp_limit <= std::numeric_limits<float>::max(),
              "forward activation clamp limit must be zero or a finite positive float32 value");
  const auto keepalive = CheckVector(z, activation, tasks, mailboxes, experts, tile_rows,
                                     cores, vector_cores, watchdog);
  const c10_npu::NPUGuard guard(z.device());
  if (tasks.size(0) == 0) {
    return;
  }
  const auto stream = c10_npu::getCurrentNPUStream(z.device().index());
  Record(keepalive, stream);
  at_npu::native::OpCommand::RunOpApi("RoutedExpertsForwardVector", [=]() -> int {
    static_cast<void>(keepalive);
    return RoutedExpertsForwardVector(stream.stream(false), z.data_ptr(), activation.data_ptr(), tasks.data_ptr(),
      mailboxes.data_ptr(), z.size(0), experts, 768, tasks.size(0), tile_rows, cores, vector_cores, watchdog,
      static_cast<float>(clamp_limit));
  });
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_parallel, library) {
  library.def("routed_experts_forward_config_bytes() -> int", &ConfigBytes);
  library.def("routed_experts_forward_config(int tile_rows, int hidden, bool transpose) -> (Tensor, int)", &Config);
  library.def("routed_experts_forward_validate(Tensor x, Tensor[] gate, Tensor[] down, Tensor gate_table, "
              "Tensor down_table, Tensor tasks, Tensor z, Tensor activation, Tensor y, Tensor config, "
              "Tensor mailboxes, int tile_rows, bool transpose, int cores, int watchdog) -> ()", &Validate);
  library.def("routed_experts_forward_cube(Tensor x, Tensor[] gate, Tensor[] down, Tensor gate_table, "
              "Tensor down_table, Tensor tasks, Tensor(a!) z, Tensor(b!) activation, Tensor(c!) y, "
              "Tensor config, Tensor(d!) mailboxes, int tile_rows, bool transpose, int cores, int watchdog, "
              "Tensor? input_ready=None, Tensor? input_expected=None, Tensor(e!)? output_ready=None) -> ()", &Cube);
  library.def("routed_experts_forward_vector(Tensor z, Tensor(a!) activation, Tensor tasks, Tensor(b!) mailboxes, "
              "int experts, int tile_rows, int cores, int vector_cores, int watchdog, "
              "float clamp_limit=0.0) -> ()", &Vector);
}
