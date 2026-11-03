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

#include <torch/library.h>
#include <ATen/ATen.h>
#include <array>
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "transport.h"

namespace {

void CheckStorage(const at::Tensor &value) {
  if (value.numel() == 0) {
    return;
  }
  const auto capacity = value.storage().nbytes() / value.element_size();
  const auto offset = static_cast<uint64_t>(value.storage_offset());
  TORCH_CHECK(value.data_ptr() != nullptr && value.storage_offset() >= 0 && offset <= capacity &&
                static_cast<uint64_t>(value.numel()) <= capacity - offset,
              "routed argument storage has been invalidated or is undersized");
}

bool StorageOverlaps(const at::Tensor &first, const at::Tensor &second) {
  if (first.numel() == 0 || second.numel() == 0) {
    return false;
  }
  if (first.is_alias_of(second)) {
    return true;
  }
  const auto first_begin = reinterpret_cast<uintptr_t>(first.data_ptr());
  const auto second_begin = reinterpret_cast<uintptr_t>(second.data_ptr());
  const auto first_bytes = static_cast<uint64_t>(first.numel()) * first.element_size();
  const auto second_bytes = static_cast<uint64_t>(second.numel()) * second.element_size();
  return first_begin <= second_begin ? second_begin - first_begin < first_bytes
                                     : first_begin - second_begin < second_bytes;
}

void CheckReadinessAliases(std::vector<at::Tensor> values, std::vector<size_t> writable,
                           const std::optional<at::Tensor> &input_ready,
                           const std::optional<at::Tensor> &input_expected,
                           const std::optional<at::Tensor> &output_ready,
                           const std::optional<at::Tensor> &expert_slots = std::nullopt) {
  for (const auto *value : {&input_ready, &input_expected, &expert_slots}) {
    if (*value) {
      values.push_back(**value);
    }
  }
  if (output_ready) {
    writable.push_back(values.size());
    values.push_back(*output_ready);
  }
  for (const size_t writer : writable) {
    for (size_t argument = 0; argument < values.size(); ++argument) {
      TORCH_CHECK(writer == argument || !StorageOverlaps(values[writer], values[argument]),
                  "routed writable storage must not overlap another argument: writer=", writer,
                  ", argument=", argument);
    }
  }
}

void CheckReadinessShape(const at::Tensor &flags, const at::Tensor &anchor, int64_t experts,
                         int64_t writers, bool output) {
  TORCH_CHECK(flags.device() == anchor.device() && flags.scalar_type() == at::kInt && flags.is_contiguous() &&
                flags.dim() == 3 && flags.size(0) > 0 && flags.size(0) <= INT32_MAX &&
                flags.size(1) == experts && flags.size(2) == 16 &&
                (flags.numel() == 0 || reinterpret_cast<uintptr_t>(flags.data_ptr()) % 64 == 0) &&
                (!output || flags.size(0) == writers), "invalid expert readiness cache lines");
}

// Each producer owns one 64-byte cache line per compact expert.
void CheckReadiness(const std::optional<at::Tensor> &flags, const std::optional<at::Tensor> &expected,
                    const at::Tensor &anchor, int64_t experts, int64_t writers, bool output = false) {
  TORCH_CHECK(output || flags.has_value() == expected.has_value(), "readiness flags/expectations must occur together");
  if (!flags) {
    return;
  }
  CheckReadinessShape(*flags, anchor, experts, writers, output);
  if (expected) {
    TORCH_CHECK(expected->device() == anchor.device() && expected->scalar_type() == at::kInt &&
                  expected->is_contiguous() && expected->sizes() == at::IntArrayRef({flags->size(0), experts}),
                "invalid expert readiness expectations");
  }
  for (const auto *value : {&flags, &expected}) {
    if (*value && (*value)->numel()) {
      CheckStorage(**value);
      c10_npu::NPUCachingAllocator::recordStream((*value)->storage().data_ptr(),
                                                c10_npu::getCurrentNPUStream(anchor.device().index()));
    }
  }
}

void *OptionalPointer(const std::optional<at::Tensor> &value) {
  return value ? value->data_ptr() : nullptr;
}

void CheckGemmMatrices(const at::Tensor &input, const at::Tensor &output) {
  TORCH_CHECK(input.dim() == 2 && output.dim() == 2 && input.size(0) == output.size(0) &&
                input.scalar_type() == at::kBFloat16 && output.scalar_type() == at::kBFloat16,
              "routed GEMM requires matching BF16 input/output row counts");
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 && input.size(0) <= (1LL << 20) &&
                input.size(1) > 0 && input.size(1) <= 32768 && input.size(1) % 16 == 0 &&
                output.size(1) > 0 && output.size(1) <= 32768 && output.size(1) % 16 == 0,
              "routed GEMM dimensions exceed the native contract");
}

void CheckGemmMetadata(const at::Tensor &tasks, const at::Tensor &config, const at::Tensor &workspace, int64_t cores) {
  TORCH_CHECK(cores > 0 && cores <= 24 && tasks.dim() == 2 && tasks.size(1) == 5 &&
                tasks.scalar_type() == at::kLong && workspace.scalar_type() == at::kByte &&
                tasks.size(0) <= (1LL << 30),
              "invalid routed GEMM descriptor or workspace");
  const auto config_bytes = static_cast<int64_t>(RoutedExpertsGemmConfigBytes());
  TORCH_CHECK(config.dim() == 1 && config.scalar_type() == at::kByte && config.numel() == config_bytes,
              "routed GEMM configuration must contain exactly ", config_bytes, " bytes");
}

void CheckTransportMatrices(const at::Tensor &input, const at::Tensor &output) {
  TORCH_CHECK(input.dim() == 2 && output.dim() == 2 && input.size(1) == output.size(1),
              "routed transport requires input/output matrices with the same row width");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && output.scalar_type() == input.scalar_type(),
              "routed transport currently requires BF16 payloads");
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                input.size(1) > 0 && input.size(1) <= 32768 && input.size(1) % 16 == 0,
              "routed transport requires NPU matrices with a supported row width");
}

void CheckTransportTopology(int64_t rank, int64_t ranks, int64_t lanes, int64_t block_rows, int64_t idle_cycles) {
  TORCH_CHECK(ranks > 0 && ranks < (1LL << 30) && rank >= 0 && rank < ranks && lanes > 0 && lanes <= 2 &&
                block_rows > 0 && block_rows <= 1024 && idle_cycles > 0,
              "invalid routed transport topology, capacity or timeout");
}

void CheckTransportTables(const at::Tensor &mapping, const at::Tensor &send_edges, const at::Tensor &receive_edges,
                          const at::Tensor &peer_offsets, const at::Tensor &progress, int64_t ranks,
                          int64_t workers) {
  TORCH_CHECK(send_edges.dim() == 2 && send_edges.size(1) == 6 && receive_edges.dim() == 2 &&
                receive_edges.size(1) == 6 && peer_offsets.sizes() == at::IntArrayRef({ranks, 6}),
              "routed transport descriptors must have six int64 fields");
  for (const auto *value : {&mapping, &send_edges, &receive_edges, &peer_offsets, &progress}) {
    TORCH_CHECK(value->scalar_type() == at::kLong, "routed transport indices must be int64");
  }
  TORCH_CHECK(mapping.dim() == 1, "routed transport mapping must be one-dimensional");
  TORCH_CHECK(progress.dim() == 2 && progress.size(1) == 16 && progress.size(0) >= std::max(ranks, workers) &&
                reinterpret_cast<uintptr_t>(progress.data_ptr()) % 64 == 0,
              "routed transport progress must contain aligned sixteen-int64 rows for every logical owner");
}

void CheckTransportWindow(const at::Tensor &input, const at::Tensor &ring, const at::Tensor &signals,
                          const at::Tensor &headers, int64_t ranks, int64_t lanes, int64_t block_rows) {
  TORCH_CHECK(signals.scalar_type() == at::kInt && signals.numel() == 2 * ranks * lanes * 16 &&
                ring.scalar_type() == input.scalar_type() &&
                ring.numel() == ranks * lanes * block_rows * input.size(1) &&
                headers.scalar_type() == at::kLong &&
                headers.numel() == ranks * lanes * ((block_rows + 8) / 8 * 8),
              "routed transport window or control allocation has an invalid extent");
}

int64_t ReadinessExperts(const std::optional<at::Tensor> &input_ready,
                        const std::optional<at::Tensor> &output_ready) {
  TORCH_CHECK((!input_ready || input_ready->dim() == 3) && (!output_ready || output_ready->dim() == 3),
              "readiness must be a three-dimensional tensor");
  return input_ready ? input_ready->size(1) : output_ready ? output_ready->size(1) : 0;
}

void CheckExpertSlots(const std::optional<at::Tensor> &expert_slots, const at::Tensor &anchor) {
  if (!expert_slots) {
    return;
  }
  TORCH_CHECK(expert_slots->device() == anchor.device() && expert_slots->scalar_type() == at::kLong &&
                expert_slots->is_contiguous() && expert_slots->dim() == 1 && expert_slots->numel() <= (1LL << 20),
              "invalid compact expert slot map");
  CheckStorage(*expert_slots);
  if (expert_slots->numel()) {
    c10_npu::NPUCachingAllocator::recordStream(expert_slots->storage().data_ptr(),
                                              c10_npu::getCurrentNPUStream(anchor.device().index()));
  }
}

int64_t GemmConfigBytes() {
  return static_cast<int64_t>(RoutedExpertsGemmConfigBytes());
}

std::tuple<at::Tensor, int64_t> GemmConfig(int64_t m, int64_t n, int64_t k, bool transpose_b) {
  TORCH_CHECK(m > 0 && m <= 1024 && n > 0 && n <= 32768 && k > 0 && k <= (1LL << 20),
              "invalid routed GEMM tiling dimensions");
  const auto config_bytes = GemmConfigBytes();
  auto result = at::empty({config_bytes}, at::TensorOptions().device(at::kCPU).dtype(at::kByte));
  uint64_t written = 0, workspace_bytes = 0;
  const int status = RoutedExpertsGemmConfig(result.data_ptr(), result.numel(), m, n, k, transpose_b,
                                             &written, &workspace_bytes);
  TORCH_CHECK(status == 0 && written == static_cast<uint64_t>(config_bytes) && workspace_bytes == 0,
              "independent routed GEMM tiling failed: ", status);
  return {result, 0};
}

at::Tensor WeightTable(at::TensorList weights) {
  TORCH_CHECK(!weights.empty(), "a routed weight table requires at least one matrix");
  const auto device = weights.front().device();
  TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "routed weight tables require NPU matrices");
  const c10_npu::NPUGuard guard(device);
  std::vector<int64_t> addresses;
  addresses.reserve(weights.size());
  for (const auto &weight : weights) {
    TORCH_CHECK(weight.device() == device && weight.scalar_type() == at::kBFloat16 && weight.dim() == 2 &&
                  (weight.is_contiguous() || weight.transpose(0, 1).is_contiguous()),
                "routed weight tables require dense BF16 matrices on one NPU");
    CheckStorage(weight);
    addresses.push_back(reinterpret_cast<int64_t>(weight.data_ptr()));
  }
  return at::tensor(addresses, at::TensorOptions().device(at::kCPU).dtype(at::kLong))
      .pin_memory(device).to(device, at::kLong, true, false);
}

void PrepareGemmWeights(const at::Tensor &input, at::TensorList weights, const at::Tensor &output,
                        bool transpose_b, const c10_npu::NPUStream &stream) {
  for (const auto &weight : weights) {
    TORCH_CHECK(weight.device() == input.device() && weight.scalar_type() == input.scalar_type() &&
                  weight.sizes() == at::IntArrayRef({input.size(1), output.size(1)}) &&
                  (transpose_b ? weight.transpose(0, 1).is_contiguous() : weight.is_contiguous()),
                "routed GEMM weight reference has an invalid layout");
    CheckStorage(weight);
    c10_npu::NPUCachingAllocator::recordStream(weight.storage().data_ptr(), stream);
  }
}

void Gemm(const at::Tensor &input, at::TensorList weights, const at::Tensor &tasks, at::Tensor output,
          const at::Tensor &config, at::Tensor workspace, bool transpose_b, int64_t cores,
          const std::optional<at::Tensor> &input_ready, const std::optional<at::Tensor> &input_expected,
          const std::optional<at::Tensor> &output_ready, int64_t watchdog_cycles,
          const std::optional<at::Tensor> &weight_table) {
  CheckGemmMatrices(input, output);
  CheckGemmMetadata(tasks, config, workspace, cores);
  const c10_npu::NPUGuard guard(input.device());
  const auto stream = c10_npu::getCurrentNPUStream(input.device().index());
  TORCH_CHECK(watchdog_cycles > 0, "readiness watchdog must be bounded and positive");
  CheckReadiness(input_ready, input_expected, input, weights.size(), 0);
  CheckReadiness(output_ready, std::nullopt, input, weights.size(), cores, true);
  const int64_t range_stride_bytes = ((static_cast<int64_t>(weights.size()) * 2 * 8 + 63) / 64) * 64;
  TORCH_CHECK(!input_ready || (workspace.numel() >= cores * range_stride_bytes &&
                               reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 64 == 0),
              "routed GEMM readiness requires cache-line-isolated ranges for each core");
  std::vector<at::Tensor> keepalive(weights.begin(), weights.end());
  PrepareGemmWeights(input, weights, output, transpose_b, stream);
  auto pointers = weight_table ? *weight_table : WeightTable(weights);
  TORCH_CHECK(pointers.scalar_type() == at::kLong && pointers.dim() == 1 &&
                pointers.numel() == static_cast<int64_t>(weights.size()),
              "routed weight table must contain one address per current matrix");
  const std::array<at::Tensor, 6> values{input, tasks, output, config, workspace, pointers};
  for (const auto &value : values) {
    TORCH_CHECK(value.device() == input.device() && value.is_contiguous(),
                "routed GEMM arguments must be contiguous on one NPU");
    if (value.numel()) {
      CheckStorage(value);
      c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(), stream);
    }
  }
  std::vector<at::Tensor> aliases(values.begin(), values.end());
  aliases.insert(aliases.end(), weights.begin(), weights.end());
  CheckReadinessAliases(aliases, {2U, 4U}, input_ready, input_expected, output_ready);
  if (tasks.size(0) == 0) {
    return;
  }
  at_npu::native::OpCommand::RunOpApi("RoutedExpertsGemm", [=]() -> int {
    // Keep the matrix objects alive while the framework queues the host launch.
    static_cast<void>(keepalive);
    return RoutedExpertsGemm(stream.stream(false), input.data_ptr(), pointers.data_ptr(), tasks.data_ptr(),
      output.data_ptr(), config.data_ptr(), workspace.data_ptr(), tasks.size(0), output.size(1), input.size(1),
      transpose_b, cores, OptionalPointer(input_ready), OptionalPointer(input_expected),
      OptionalPointer(output_ready), input_ready ? input_ready->size(0) : 0, weights.size(), watchdog_cycles);
  });
}

void Transport(const at::Tensor &input, at::Tensor output, const at::Tensor &mapping,
                const at::Tensor &send_edges, const at::Tensor &receive_edges, const at::Tensor &peer_offsets,
                at::Tensor ring, at::Tensor signals, at::Tensor progress, int64_t rank, int64_t ranks,
                int64_t lanes, int64_t block_rows, bool gather, int64_t idle_cycles, at::Tensor headers,
                const std::optional<at::Tensor> &input_ready, const std::optional<at::Tensor> &input_expected,
                const std::optional<at::Tensor> &output_ready, const std::optional<at::Tensor> &expert_slots,
                const std::optional<at::Tensor> &dedup_state,
                const std::optional<at::Tensor> &group_metadata, bool reduce, bool publish_each_block,
                double reduce_dense_ratio, double forward_compact_max_unique_ratio, int64_t workers) {
  const c10_npu::NPUGuard guard(input.device());
  CheckTransportMatrices(input, output);
  CheckTransportTopology(rank, ranks, lanes, block_rows, idle_cycles);
  const int32_t capacity = RoutedExpertsVectorCores() - 8;
  workers = workers == 0 ? RoutedExpertsTransportWorkers(static_cast<int32_t>(ranks)) : workers;
  TORCH_CHECK(workers > 0 && workers <= capacity && (workers <= ranks || workers == 2 * ranks),
              "transport worker budget must leave the vector compute service resident");
  CheckTransportTables(mapping, send_edges, receive_edges, peer_offsets, progress, ranks, workers);
  CheckTransportWindow(input, ring, signals, headers, ranks, lanes, block_rows);
  TORCH_CHECK(!(dedup_state && reduce) &&
                (dedup_state.has_value() || reduce) == group_metadata.has_value(),
              "one counted dispatch or reduced return requires target token metadata");
  TORCH_CHECK((!dedup_state && !reduce) || input.size(1) % 32 == 0,
              "device-counted transport requires 64-byte aligned BF16 rows");
  if (dedup_state) {
    TORCH_CHECK(gather && dedup_state->scalar_type() == at::kLong &&
                  dedup_state->sizes() ==
                    at::IntArrayRef({2, ranks, ((std::max<int64_t>(1, input.size(0)) + 7) / 8) * 8}) &&
                  group_metadata->scalar_type() == at::kLong && group_metadata->dim() == 2 &&
                  group_metadata->size(0) == output.size(0) && group_metadata->size(1) == 8,
                "device-counted transport metadata has invalid geometry");
  }
  if (reduce) {
    TORCH_CHECK(!gather &&
                  group_metadata->scalar_type() == at::kLong && group_metadata->dim() == 2 &&
                  group_metadata->size(0) == input.size(0) && group_metadata->size(1) == 8,
                "block return reduction metadata has invalid geometry");
  }
  const int64_t experts = ReadinessExperts(input_ready, output_ready);
  CheckReadiness(input_ready, input_expected, input, experts, 0);
  CheckReadiness(output_ready, std::nullopt, input, experts, ranks, true);
  TORCH_CHECK(!publish_each_block || (output_ready && !reduce),
              "per-block readiness requires a forward output publication");
  TORCH_CHECK(reduce_dense_ratio >= 0.0 && reduce_dense_ratio <= 1.0,
              "dense return ratio must be between zero and one");
  TORCH_CHECK(forward_compact_max_unique_ratio >= 0.0 && forward_compact_max_unique_ratio <= 1.0,
              "forward compact ratio must be between zero and one");
  TORCH_CHECK((!input_ready && !output_ready) || expert_slots.has_value(), "readiness requires expert identities");
  CheckExpertSlots(expert_slots, input);
  const std::array<at::Tensor, 10> values{input, output, mapping, send_edges, receive_edges,
                                        peer_offsets, ring, signals, progress, headers};
  const auto stream = c10_npu::getCurrentNPUStream(input.device().index());
  for (const auto &value : values) {
    TORCH_CHECK(value.device() == input.device() && value.is_contiguous(),
                "routed transport tensors must be contiguous on one NPU");
    CheckStorage(value);
  }
  auto checked = std::vector<at::Tensor>(values.begin(), values.end());
  auto writable = std::vector<size_t>{1U, 6U, 7U, 8U, 9U};
  if (dedup_state) {
    writable.push_back(checked.size());
    checked.push_back(*dedup_state);
  }
  if (group_metadata) {
    if (dedup_state) {
      writable.push_back(checked.size());
    }
    checked.push_back(*group_metadata);
  }
  CheckReadinessAliases(std::move(checked), std::move(writable),
                        input_ready, input_expected, output_ready, expert_slots);
  const std::array<const at::Tensor *, 7> allocated{
    &input, &output, &mapping, &send_edges, &receive_edges, &peer_offsets, &progress};
  for (const auto *value : allocated) {
    if (value->numel() > 0) {
      c10_npu::NPUCachingAllocator::recordStream(value->storage().data_ptr(), stream);
    }
  }
  for (const auto *value : {&dedup_state, &group_metadata}) {
    if (*value && (*value)->numel()) {
      TORCH_CHECK((*value)->device() == input.device() && (*value)->is_contiguous(),
                  "device-counted transport metadata must be contiguous on one NPU");
      CheckStorage(**value);
      TORCH_CHECK(reinterpret_cast<std::uintptr_t>((*value)->data_ptr()) % 64 == 0,
                  "device-counted transport metadata must start on a 64-byte boundary");
      c10_npu::NPUCachingAllocator::recordStream((*value)->storage().data_ptr(), stream);
    }
  }
  const int32_t row_bytes = static_cast<int32_t>(input.size(1) * input.element_size());
  // Capturing tensors keeps queued host arguments alive. The window lease separately
  // protects SHMEM storage, and recordStream protects allocator-backed device users.
  at_npu::native::OpCommand::RunOpApi("RoutedExpertsTransport", [=]() -> int {
    return RoutedExpertsTransport(stream.stream(false), input.data_ptr(), output.data_ptr(), mapping.data_ptr(),
      send_edges.data_ptr(), receive_edges.data_ptr(), peer_offsets.data_ptr(), ring.data_ptr(), signals.data_ptr(),
      progress.data_ptr(), rank, ranks, lanes, block_rows, row_bytes, gather, idle_cycles,
      input.size(0), output.size(0), mapping.numel(), send_edges.size(0), receive_edges.size(0),
      OptionalPointer(input_ready), OptionalPointer(input_expected), OptionalPointer(output_ready),
      OptionalPointer(expert_slots), input_ready ? input_ready->size(0) : 0, experts,
      expert_slots ? expert_slots->numel() : 0, headers.data_ptr(), OptionalPointer(dedup_state),
      OptionalPointer(group_metadata), reduce, publish_each_block, static_cast<float>(reduce_dense_ratio),
      static_cast<float>(forward_compact_max_unique_ratio), static_cast<int32_t>(workers));
  });
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_parallel, m) {
  m.def("routed_experts_gemm_config_bytes() -> int");
  m.impl("routed_experts_gemm_config_bytes", &GemmConfigBytes);
  m.def("routed_experts_gemm_config(int m, int n, int k, bool transpose_b) -> (Tensor, int)");
  m.impl("routed_experts_gemm_config", &GemmConfig);
  m.def("routed_experts_weight_table(Tensor[] weights) -> Tensor");
  m.def("routed_experts_gemm(Tensor input, Tensor[] weights, Tensor tasks, Tensor(a!) output, "
        "Tensor config, Tensor(b!) workspace, bool transpose_b, int cores, "
        "Tensor? input_ready=None, Tensor? input_expected=None, Tensor(c!)? output_ready=None, "
        "int watchdog_cycles=60000000000, Tensor? weight_table=None) -> ()");
  m.def("routed_experts_transport(Tensor input, Tensor(a!) output, Tensor mapping, Tensor send_edges, "
        "Tensor receive_edges, Tensor peer_offsets, Tensor(b!) ring, Tensor(c!) signals, Tensor(d!) progress, "
        "int rank, int ranks, int lanes, int block_rows, bool gather, int idle_cycles, "
        "Tensor(f!) headers, "
        "Tensor? input_ready=None, Tensor? input_expected=None, Tensor(e!)? output_ready=None, "
        "Tensor? expert_slots=None, Tensor(g!)? dedup_state=None, "
        "Tensor(h!)? group_metadata=None, bool reduce=False, bool publish_each_block=False, "
        "float reduce_dense_ratio=1.0, float forward_compact_max_unique_ratio=1.0, int workers=0) -> ()");
}

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("routed_experts_weight_table", &WeightTable);
  m.impl("routed_experts_gemm", &Gemm);
  m.impl("routed_experts_transport", &Transport);
}
