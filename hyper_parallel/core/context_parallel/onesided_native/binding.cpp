// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "shmem.h"
#include <cstring>
#include <unordered_map>
#include <limits>
namespace state_pull {
void signal(void *, void *, int, int);
void wait(void *, void *, int);
}  // namespace state_pull
namespace {
struct Allocation {
  at::Tensor owner;
  void *ptr;
  size_t bytes;
};
std::unordered_map<c10::StorageImpl *, Allocation> allocations;
bool active = false;
int device = -1, nranks = 0;
void checked(int status, const char *name) { TORCH_CHECK(status == 0, name, " returned ", status); }
void require_active() {
  TORCH_CHECK(active, "state pull runtime is not acquired");
  int current = -1;
  checked(aclrtGetDevice(&current), "aclrtGetDevice");
  TORCH_CHECK(current == device, "state pull device changed");
}
aclrtStream stream() {
  require_active();
  // Drain Torch's host task queue before raw ACL launches on its device stream.
  return c10_npu::getCurrentNPUStream(device).stream(true);
}
void valid_tensor(const at::Tensor &t) {
  require_active();
  TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1 && t.device().index() == device,
              "expected an NPU tensor on the acquired device");
  TORCH_CHECK(t.is_contiguous() && t.numel() > 0, "expected a nonempty contiguous tensor");
}
Allocation &symmetric(const at::Tensor &t) {
  valid_tensor(t);
  auto it = allocations.find(t.storage().unsafeGetStorageImpl());
  TORCH_CHECK(it != allocations.end(), "tensor does not belong to this active symmetric heap");
  auto begin = reinterpret_cast<uintptr_t>(it->second.ptr), ptr = reinterpret_cast<uintptr_t>(t.data_ptr());
  auto bytes = t.nbytes();
  TORCH_CHECK(ptr >= begin && bytes <= it->second.bytes && ptr - begin <= it->second.bytes - bytes,
              "tensor view exceeds its symmetric allocation");
  return it->second;
}
void valid_peer(int peer) { TORCH_CHECK(peer >= 0 && peer < nranks, "invalid peer"); }
void init(int rank, int size, int64_t heap, const std::string &endpoint) {
  TORCH_CHECK(!active && allocations.empty(), "backend already acquired");
  TORCH_CHECK(aclshmemx_init_status() != ACLSHMEM_STATUS_IS_INITIALIZED,
              "another SDK SHMEM runtime is already initialized");
  TORCH_CHECK(size >= 2 && size <= ACLSHMEM_MAX_PES && rank >= 0 && rank < size && heap > 0,
              "invalid SHMEM rank, group size or heap budget");
  checked(aclrtGetDevice(&device), "aclrtGetDevice");
  const char *soc = aclrtGetSocName();
  TORCH_CHECK(soc != nullptr && std::strncmp(soc, "Ascend910B", 10) == 0,
              "this state-pull payload is qualified for Ascend910B (A2) only");
  aclshmemx_uniqueid_t uid{};
  uid.my_pe = rank;
  uid.n_pes = size;
  aclshmemx_init_attr_t attr{};
  attr.my_pe = rank;
  attr.n_pes = size;
  attr.local_mem_size = heap;
  TORCH_CHECK(!endpoint.empty() && endpoint.size() < sizeof(attr.ip_port), "invalid bootstrap URI");
  std::memcpy(attr.ip_port, endpoint.c_str(), endpoint.size() + 1);
  attr.comm_args = &uid;
  attr.option_attr = {(1 << 16) + sizeof(aclshmemx_init_attr_t), ACLSHMEM_DATA_OP_SDMA, DEFAULT_TIMEOUT,
                      DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
  checked(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr), "aclshmemx_init_attr");
  nranks = size;
  active = true;
}
at::Tensor empty(const std::vector<int64_t> &shape, c10::ScalarType dtype) {
  require_active();
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kInt, "research heap supports float32/int32 only");
  uint64_t n = 1;
  for (auto dim : shape) {
    TORCH_CHECK(dim > 0 && n <= std::numeric_limits<uint64_t>::max() / dim, "invalid shape");
    n *= dim;
  }
  TORCH_CHECK(n <= std::numeric_limits<size_t>::max() / 4, "allocation overflow");
  auto bytes = static_cast<size_t>(n * 4);
  void *ptr = aclshmem_malloc(bytes);
  TORCH_CHECK(ptr != nullptr, "aclshmem_malloc failed");
  c10::Device target(c10::DeviceType::PrivateUse1, device);
  auto t = at_npu::native::from_blob(
    ptr, shape, [](void *) {}, at::TensorOptions().dtype(dtype).device(target), c10::optional<c10::Device>(target));
  allocations.emplace(t.storage().unsafeGetStorageImpl(), Allocation{t, ptr, bytes});
  return t;
}
bool can_access(const at::Tensor &t, int peer) {
  symmetric(t);
  valid_peer(peer);
  return aclshmem_ptr(t.data_ptr(), peer) != nullptr;
}
void free_tensor(const at::Tensor &t) {
  auto &record = symmetric(t);
  TORCH_CHECK(t.data_ptr() == record.ptr && t.nbytes() == record.bytes, "free requires the full allocation");
  // Explicit teardown is intentionally synchronous; do not use this per packet.
  checked(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice");
  aclshmem_barrier_all();
  auto *storage = t.storage().unsafeGetStorageImpl();
  aclshmem_free(record.ptr);
  storage->set_data_ptr(at::DataPtr(nullptr, nullptr, nullptr, t.device()));
  storage->set_nbytes(0);
  allocations.erase(storage);
}
void validate_packet_flags(const at::Tensor &ready, const at::Tensor &credit,
                           const std::vector<int> &previous, const std::vector<int64_t> &offsets) {
  const int packets = previous.size();
  TORCH_CHECK(packets > 0 && offsets.size() == packets + 1 && ready.dim() == 2 && ready.size(0) == packets &&
                ready.size(1) == 128 && credit.sizes() == ready.sizes(),
              "flags/packets mismatch");
  TORCH_CHECK(ready.scalar_type() == at::kInt && credit.scalar_type() == at::kInt, "int32 flags required");
}
void validate_packet_layout(const at::Tensor &state, const at::Tensor &slot,
                            const at::Tensor &ready, const at::Tensor &credit,
                            const std::vector<int> &previous, const std::vector<int64_t> &offsets) {
  TORCH_CHECK(state.scalar_type() == at::kFloat && slot.scalar_type() == at::kFloat && state.sizes() == slot.sizes() &&
                state.dim() == 4 && state.size(0) == 1 && state.size(2) == state.size(3),
              "fixed FP32 square state");
  validate_packet_flags(ready, credit, previous, offsets);
  const int packets = previous.size();
  TORCH_CHECK(offsets.front() == 0 && offsets.back() == state.size(1), "packet offsets must cover all heads");
  for (int i = 0; i < packets; ++i)
    TORCH_CHECK(offsets[i] < offsets[i + 1], "packet head ranges must be nonempty and ordered");
}
at::Tensor validate_relay_inputs(const at::Tensor &state, const c10::optional<at::Tensor> &matrix,
                                 const c10::optional<at::Tensor> &incoming, bool dependent, bool receives) {
  at::Tensor transfer;
  if (dependent) {
    valid_tensor(*matrix);
    TORCH_CHECK(matrix->sizes() == state.sizes() && matrix->scalar_type() == at::kFloat, "matrix shape mismatch");
    transfer = *matrix;
  }
  if (receives) {
    TORCH_CHECK(incoming.has_value(), "missing owned H/D");
    valid_tensor(*incoming);
    TORCH_CHECK(incoming->sizes() == state.sizes() && incoming->scalar_type() == at::kFloat, "incoming mismatch");
  }
  return transfer;
}
void relay(const at::Tensor &state, const c10::optional<at::Tensor> &matrix, const c10::optional<at::Tensor> &incoming,
           const at::Tensor &slot, const at::Tensor &ready, const at::Tensor &credit, const std::vector<int> &previous,
           const std::vector<int64_t> &offsets, int epoch, int source, int destination, bool receives, bool sends) {
  valid_tensor(state);
  symmetric(slot);
  symmetric(ready);
  symmetric(credit);
  if (receives) valid_peer(source);
  if (sends) valid_peer(destination);
  TORCH_CHECK(epoch > 0, "epoch must be positive");
  validate_packet_layout(state, slot, ready, credit, previous, offsets);
  const int packets = previous.size();
  const bool dependent = receives && sends && matrix.has_value();
  auto transfer = validate_relay_inputs(state, matrix, incoming, dependent, receives);
  auto read_part = [&](int i) {
    auto dst = incoming->slice(1, offsets[i], offsets[i + 1]);
    auto local = slot.slice(1, offsets[i], offsets[i + 1]);
    auto ready_ptr = ready.data_ptr<int32_t>() + i * 128;
    auto credit_ptr = credit.data_ptr<int32_t>() + i * 128;
    auto current = stream();
    auto remote = aclshmem_ptr(local.data_ptr(), source);
    TORCH_CHECK(remote != nullptr, "peer source not mapped");
    state_pull::wait(current, ready_ptr, epoch);
    checked(
      aclrtMemcpyAsync(dst.data_ptr(), dst.nbytes(), remote, local.nbytes(), ACL_MEMCPY_DEVICE_TO_DEVICE, current),
      "relay pull");
    state_pull::signal(current, credit_ptr, epoch, source);
  };
  for (int i = 0; i < packets; ++i) {
    if (dependent) read_part(i);
    if (!sends) continue;
    auto dst = slot.slice(1, offsets[i], offsets[i + 1]);
    auto local = state.slice(1, offsets[i], offsets[i + 1]);
    auto current = stream();
    if (previous[i]) state_pull::wait(current, credit.data_ptr<int32_t>() + i * 128, previous[i]);
    if (dependent) {
      auto entry = incoming->slice(1, offsets[i], offsets[i + 1]);
      at::matmul_out(dst, transfer.slice(1, offsets[i], offsets[i + 1]), entry);
      at::add_out(dst, dst, local, at::Scalar(1));
    } else {
      checked(aclrtMemcpyAsync(dst.data_ptr(), dst.nbytes(), local.data_ptr(), local.nbytes(),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, current),
              "endpoint source");
    }
    // Matmul/add are still the same public ATen kernels. Drain their host
    // queue before publishing readiness on the same physical NPU stream.
    current = stream();
    state_pull::signal(current, ready.data_ptr<int32_t>() + i * 128, epoch, destination);
  }
  // Real reset: publish the independent outgoing summary before consuming
  // the unrelated incoming document; terminal ranks only consume.
  if (receives && !dependent)
    for (int i = 0; i < packets; ++i) read_part(i);
}
void release() {
  require_active();
  TORCH_CHECK(allocations.empty(), "free all symmetric allocations before release");
  checked(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice");
  aclshmem_barrier_all();
  checked(aclshmem_finalize(), "aclshmem_finalize");
  active = false;
  device = -1;
  nranks = 0;
}
}  // namespace
PYBIND11_MODULE(_hp_state_pull, m) {
  m.def("init", &init);
  m.def("empty", &empty);
  m.def("free", &free_tensor);
  m.def("can_access", &can_access);
  m.def("relay", &relay);
  m.def("release", &release);
}
