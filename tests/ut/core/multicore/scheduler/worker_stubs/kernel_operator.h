/*
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
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
#ifndef MULTICORE_TEST_KERNEL_OPERATOR_H
#define MULTICORE_TEST_KERNEL_OPERATOR_H

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>

#define __aicore__
#define __gm__
using GM_ADDR = uint8_t *;
constexpr int PIPE_ALL = 0;
constexpr int ACLSHMEM_SIGNAL_SET = 0;
constexpr int ACLSHMEM_CMP_GE = 0;

namespace AscendC {
inline thread_local bool atomic_add = false;
inline thread_local void (*barrier_callback)() = nullptr;
inline thread_local void (*compute_callback)(uint32_t) = nullptr;

enum class HardEvent { S_MTE2, MTE2_S, S_MTE3, MTE3_S };
enum class TPosition { A1, VECOUT };
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };

template <typename T>
struct GlobalTensor {
  T *data = nullptr;
  void SetGlobalBuffer(T *pointer, uint32_t) { data = pointer; }
  T GetValue(uint32_t index) const { return __atomic_load_n(data + index, __ATOMIC_SEQ_CST); }
  void SetValue(uint32_t index, T value) { __atomic_store_n(data + index, value, __ATOMIC_SEQ_CST); }
  GlobalTensor operator[](uint32_t index) const { return {data + index}; }
};

template <typename T>
struct LocalTensor {
  T *data;
  void SetValue(uint32_t index, T value) { data[index] = value; }
};

struct TPipe {
  template <typename Buffer>
  void InitBuffer(Buffer &, uint32_t) {}
  uint32_t FetchEventID(HardEvent) { return 0; }
  void Destroy() {}
};

template <TPosition position>
struct TBuf {
  int32_t data[8] = {};
  template <typename T>
  LocalTensor<T> GetWithOffset(uint32_t, uint32_t) {
    return {reinterpret_cast<T *>(data)};
  }
  template <typename T>
  void FreeTensor(LocalTensor<T>) {}
};

inline TPipe *GetTPipePtr() {
  static thread_local TPipe pipe;
  return &pipe;
}
template <HardEvent event>
void SetFlag(uint32_t) {}
template <HardEvent event>
void WaitFlag(uint32_t) {}
template <int pipe>
void PipeBarrier() {}
template <typename T, CacheLine line, DcciDst destination>
void DataCacheCleanAndInvalid(GlobalTensor<T>) {}
template <typename T>
void SetAtomicAdd() {
  atomic_add = true;
}
inline void SetAtomicNone() { atomic_add = false; }
inline int64_t GetSystemCycle() {
  static thread_local auto start = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::steady_clock::now() - start;
  if (elapsed > std::chrono::seconds(10)) {
    throw std::runtime_error("dependency wait timed out");
  }
  std::this_thread::yield();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
}
[[noreturn]] inline void Trap() { throw std::runtime_error("device trap"); }

template <typename T>
void DataCopy(LocalTensor<T> destination, GlobalTensor<T> source, uint32_t count) {
  for (uint32_t index = 0; index < count; ++index) {
    destination.data[index] = source.GetValue(index);
  }
}

template <typename T>
void DataCopy(GlobalTensor<T> destination, LocalTensor<T> source, uint32_t count) {
  for (uint32_t index = 0; index < count; ++index) {
    if (atomic_add) {
      __atomic_fetch_add(destination.data + index, source.data[index], __ATOMIC_SEQ_CST);
    } else {
      destination.SetValue(index, source.data[index]);
    }
  }
}

template <bool is_aiv_only = true>
void SyncAll() {
  static_assert(!is_aiv_only, "MIX barriers must include both AIC and AIV workers");
  barrier_callback();
}
}  // namespace AscendC
using AscendC::GetTPipePtr;

// Ready metadata uses a pre-published local counter in this host simulation.
inline void aclshmemx_signal_op(int32_t *, int32_t, int, int) {}
inline void aclshmem_signal_wait_until(int32_t *, int, int32_t) {}

#endif  // MULTICORE_TEST_KERNEL_OPERATOR_H
