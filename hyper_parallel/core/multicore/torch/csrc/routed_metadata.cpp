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
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>
#include <vector>
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace {

constexpr int64_t kFields = 6;
constexpr int64_t kMaxRows = 1LL << 20;

void CheckHost(const at::Tensor &value, at::ScalarType dtype) {
  TORCH_CHECK(value.device().is_cpu() && value.scalar_type() == dtype && value.is_contiguous(),
              "routed metadata requires contiguous CPU tables of the expected dtype");
  if (value.numel()) {
    const auto capacity = value.storage().nbytes() / value.element_size();
    TORCH_CHECK(value.storage_offset() >= 0 && static_cast<uint64_t>(value.storage_offset()) <= capacity &&
                  static_cast<uint64_t>(value.numel()) <= capacity - value.storage_offset(),
                "routed metadata host storage has been invalidated");
  }
}

struct Blocks {
  at::Tensor table;
  std::vector<int64_t> begin;
  std::vector<int64_t> end;
};

Blocks Pack(const at::Tensor &edges, int64_t ranks, int64_t block_rows, int64_t peer_column,
            int64_t local_rows) {
  CheckHost(edges, at::kLong);
  TORCH_CHECK(edges.dim() == 2 && edges.size(1) == kFields, "route edges require six integer columns");
  const auto *data = edges.const_data_ptr<int64_t>();
  std::vector<int64_t> order(edges.size(0));
  std::iota(order.begin(), order.end(), 0);
  std::vector<int64_t> counts(ranks, 0);
  std::vector<std::pair<int64_t, int64_t>> ranges;
  int64_t rows = 0;
  for (const auto index : order) {
    const auto *edge = data + index * kFields;
    for (int64_t field = 0; field < kFields; ++field) {
      TORCH_CHECK(edge[field] >= 0, "route edge fields must be nonnegative");
    }
    TORCH_CHECK(edge[0] < ranks && edge[1] < ranks && edge[5] > 0 && edge[5] <= local_rows,
                "invalid route peers or row count");
    const auto local_start = edge[peer_column == 1 ? 3 : 4];
    TORCH_CHECK(local_start <= local_rows - edge[5] && rows <= local_rows - edge[5] &&
                  edge[3] <= INT64_MAX - edge[5] && edge[4] <= INT64_MAX - edge[5],
                "route row ranges exceed the admitted extent");
    rows += edge[5];
    ranges.emplace_back(local_start, local_start + edge[5]);
    counts[edge[peer_column]] += (edge[5] - 1) / block_rows + 1;
  }
  TORCH_CHECK(rows == local_rows, "route edges do not cover the admitted local row count");
  std::sort(ranges.begin(), ranges.end());
  int64_t cursor = 0;
  for (const auto &range : ranges) {
    TORCH_CHECK(range.first == cursor, "route ranges overlap or omit local rows");
    cursor = range.second;
  }
  std::stable_sort(order.begin(), order.end(), [&](int64_t left, int64_t right) {
    return data[left * kFields + peer_column] < data[right * kFields + peer_column];
  });
  std::vector<int64_t> begin(ranks), end(ranks);
  std::partial_sum(counts.begin(), counts.end(), end.begin());
  for (int64_t peer = 0; peer < ranks; ++peer) {
    begin[peer] = end[peer] - counts[peer];
  }
  auto table = at::empty({end.back(), kFields}, edges.options());
  auto *packed = table.mutable_data_ptr<int64_t>();
  int64_t block = 0;
  for (const auto index : order) {
    const auto *edge = data + index * kFields;
    for (int64_t offset = 0; offset < edge[5]; offset += block_rows, ++block) {
      auto *row = packed + block * kFields;
      std::copy_n(edge, kFields, row);
      row[3] += offset;
      row[4] += offset;
      row[5] = std::min(block_rows, edge[5] - offset);
    }
  }
  return {table, begin, end};
}

at::Tensor Reverse(const at::Tensor &table) {
  auto output = at::empty_like(table);
  const auto *input = table.const_data_ptr<int64_t>();
  auto *data = output.mutable_data_ptr<int64_t>();
  const std::array<int64_t, kFields> columns{1, 0, 2, 4, 3, 5};
  for (int64_t row = 0; row < table.size(0); ++row) {
    for (int64_t column = 0; column < kFields; ++column) {
      data[row * kFields + column] = input[row * kFields + columns[column]];
    }
  }
  return output;
}

struct ComputeTables {
  at::Tensor tasks;
  at::Tensor expected;
  at::Tensor progress;
  std::vector<int64_t> tile_offsets;
};

ComputeTables Compute(at::IntArrayRef boundaries, int64_t tile_rows) {
  const int64_t experts = boundaries.size();
  std::vector<int64_t> offsets(experts + 1, 0);
  int64_t begin = 0;
  for (int64_t expert = 0; expert < experts; ++expert) {
    TORCH_CHECK(boundaries[expert] > begin && boundaries[expert] <= kMaxRows,
                "compute boundaries must be strictly increasing within the admitted row bound");
    offsets[expert + 1] = offsets[expert] + (boundaries[expert] - begin - 1) / tile_rows + 1;
    begin = boundaries[expert];
  }
  const auto cores = std::min<int64_t>(24, offsets.back());
  auto tasks = at::empty({offsets.back(), 3}, at::TensorOptions().dtype(at::kLong));
  auto expected = at::zeros({cores, experts}, at::TensorOptions().dtype(at::kInt));
  auto progress = at::zeros({cores, experts, 16}, at::TensorOptions().dtype(at::kInt));
  auto *task = tasks.mutable_data_ptr<int64_t>();
  auto *flags = progress.mutable_data_ptr<int32_t>();
  auto *target = expected.mutable_data_ptr<int32_t>();
  begin = 0;
  for (int64_t expert = 0; expert < experts; ++expert) {
    for (int64_t index = offsets[expert]; index < offsets[expert + 1]; ++index) {
      const auto row = begin + (index - offsets[expert]) * tile_rows;
      task[3 * index] = expert;
      task[3 * index + 1] = row;
      task[3 * index + 2] = std::min(tile_rows, boundaries[expert] - row);
    }
    for (int64_t core = 0; core < cores; ++core) {
      auto *cell = flags + (core * experts + expert) * 16;
      const auto first = offsets[expert] + (core - offsets[expert] % cores + cores) % cores;
      const bool active = first < offsets[expert + 1];
      target[core * experts + expert] = active;
      cell[1] = active;
      cell[2] = begin;
      cell[3] = tile_rows;
      cell[4] = 1;
      cell[5] = active ? first - offsets[expert] : 0;
      cell[6] = active ? (offsets[expert + 1] - 1 - first) / cores + 1 : 0;
      cell[7] = active ? first : 0;
      cell[8] = cores;
      cell[9] = boundaries[expert] - begin;
    }
    begin = boundaries[expert];
  }
  return {tasks, expected, progress, offsets};
}

struct ReceiveTables {
  at::Tensor slots;
  at::Tensor expected;
  at::Tensor task_expected;
};

ReceiveTables Receive(const at::Tensor &send, const at::Tensor &receive, at::IntArrayRef active,
                      at::IntArrayRef boundaries, const std::vector<int64_t> &tiles,
                      int64_t ranks, int64_t tile_rows) {
  int64_t count = 0;
  for (const auto &table : {send, receive}) {
    const auto *data = table.const_data_ptr<int64_t>();
    for (int64_t row = 0; row < table.size(0); ++row) {
      TORCH_CHECK(data[row * kFields + 2] < INT64_MAX, "expert ID overflows the slot table extent");
      count = std::max(count, data[row * kFields + 2] + 1);
    }
  }
  auto slots = at::full({count}, -1, at::TensorOptions().dtype(at::kLong));
  auto *slot = slots.mutable_data_ptr<int64_t>();
  for (size_t index = 0; index < active.size(); ++index) {
    TORCH_CHECK(active[index] >= 0 && active[index] < count &&
                  (index == 0 || active[index] > active[index - 1]), "invalid active expert IDs");
    slot[active[index]] = index;
  }
  auto expected = at::zeros({ranks, static_cast<int64_t>(active.size())}, at::TensorOptions().dtype(at::kInt));
  auto task_expected = at::zeros({ranks, tiles.back()}, at::TensorOptions().dtype(at::kInt));
  auto *counts = expected.mutable_data_ptr<int32_t>();
  auto *target = task_expected.mutable_data_ptr<int32_t>();
  std::vector<uint8_t> covered(tiles.back(), 0);
  const auto *blocks = receive.const_data_ptr<int64_t>();
  for (int64_t row = 0; row < receive.size(0); ++row) {
    const auto *block = blocks + row * kFields;
    const auto expert = slot[block[2]];
    TORCH_CHECK(expert >= 0, "incoming block does not name an active expert");
    const auto begin = expert ? boundaries[expert - 1] : 0;
    TORCH_CHECK(block[4] >= begin && block[4] + block[5] <= boundaries[expert],
                "incoming block crosses its expert range");
    const auto ordinal = ++counts[block[0] * active.size() + expert];
    const auto first = tiles[expert] + (block[4] - begin) / tile_rows;
    const auto last = tiles[expert] + (block[4] + block[5] - 1 - begin) / tile_rows;
    for (int64_t tile = first; tile <= last; ++tile) {
      auto &value = target[block[0] * tiles.back() + tile];
      value = std::max(value, ordinal);
      covered[tile] = 1;
    }
  }
  TORCH_CHECK(std::all_of(covered.begin(), covered.end(), [](uint8_t value) { return value != 0; }),
              "incoming blocks do not cover every forward compute tile");
  return {slots, expected, task_expected};
}

std::vector<at::Tensor> Upload(const std::vector<at::Tensor> &tables, c10::Device device) {
  std::vector<int64_t> offsets;
  int64_t bytes = 0;
  for (const auto &table : tables) {
    bytes = (bytes + 63) / 64 * 64;
    offsets.push_back(bytes);
    TORCH_CHECK(table.nbytes() <= static_cast<uint64_t>(INT64_MAX - bytes - 63), "metadata size overflows int64");
    bytes += table.nbytes();
  }
  bytes = (bytes + 63) / 64 * 64;
  auto host = at::empty({bytes}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
  for (size_t index = 0; index < tables.size(); ++index) {
    if (tables[index].numel()) {
      std::memcpy(host.mutable_data_ptr<uint8_t>() + offsets[index], tables[index].const_data_ptr(),
                  tables[index].nbytes());
    }
  }
  auto root = host.to(device, at::kByte, true, false);
  std::map<at::ScalarType, at::Tensor> typed;
  std::vector<at::Tensor> result;
  for (size_t index = 0; index < tables.size(); ++index) {
    const auto &table = tables[index];
    if (!typed.count(table.scalar_type())) {
      typed.emplace(table.scalar_type(), root.view(table.scalar_type()));
    }
    std::vector<int64_t> strides(table.dim(), 1);
    for (int64_t axis = table.dim() - 2; axis >= 0; --axis) {
      strides[axis] = strides[axis + 1] * std::max<int64_t>(table.size(axis + 1), 1);
    }
    result.push_back(at::as_strided(typed.at(table.scalar_type()), table.sizes(), strides,
                                   offsets[index] / table.element_size()));
  }
  return result;
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> Compile(
    const at::Tensor &anchor, const at::Tensor &outgoing, const at::Tensor &incoming,
    at::IntArrayRef active, at::IntArrayRef boundaries, const at::Tensor &config,
    int64_t ranks, int64_t source_rows, int64_t block_rows, int64_t tile_rows) {
  TORCH_CHECK(anchor.device().type() == c10::DeviceType::PrivateUse1 && ranks > 0 && ranks < (1LL << 30) &&
                source_rows >= 0 && source_rows <= kMaxRows && block_rows > 0 && block_rows <= 1024 &&
                tile_rows > 0 && tile_rows <= 1024 && active.size() == boundaries.size(),
              "invalid routed metadata geometry");
  CheckHost(config, at::kByte);
  TORCH_CHECK(config.dim() == 1, "forward configuration must be a CPU byte vector");
  const c10_npu::NPUGuard guard(anchor.device());
  auto compute = Compute(boundaries, tile_rows);
  const auto received_rows = boundaries.empty() ? 0 : boundaries.back();
  auto send = Pack(outgoing, ranks, block_rows, 1, source_rows);
  auto receive = Pack(incoming, ranks, block_rows, 0, received_rows);
  auto ready = Receive(send.table, receive.table, active, boundaries, compute.tile_offsets, ranks, tile_rows);
  auto ranges = at::empty({ranks, 4}, outgoing.options());
  auto *range = ranges.mutable_data_ptr<int64_t>();
  for (int64_t peer = 0; peer < ranks; ++peer) {
    range[peer * 4] = send.begin[peer];
    range[peer * 4 + 1] = send.end[peer];
    range[peer * 4 + 2] = receive.begin[peer];
    range[peer * 4 + 3] = receive.end[peer];
  }
  auto device = Upload({send.table, receive.table, Reverse(receive.table), Reverse(send.table),
                        ready.slots, ready.expected, ready.task_expected, compute.tasks, config, compute.expected},
                       anchor.device());
  return {{send.table, receive.table, ranges, ready.expected, ready.task_expected,
           compute.expected, compute.progress}, device};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_parallel, library) {
  library.def("routed_experts_forward_metadata(Tensor anchor, Tensor outgoing, Tensor incoming, "
              "int[] active, int[] boundaries, Tensor config, int ranks, int source_rows, "
              "int block_rows, int tile_rows) -> (Tensor[], Tensor[])", &Compile);
}
