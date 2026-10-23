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
#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#define __aicore__
#define __gm__
constexpr uint32_t EXP_TOKEN_COUNT_FLAG_CNT = 8;
constexpr uint32_t DISPATCH_TOKEN_UB_SIZE = 32;
constexpr int PIPE_ALL = 0;
std::atomic<int> scalar_calls{0};
std::atomic<int> unsupported_copies{0};
#include "event_poll_constants.inc"
int64_t current_cycle = 0;
int64_t release_cycle = 0;
std::vector<int64_t> refresh_cycles;

enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };

namespace AscendC {
enum class TPosition { A1, VECOUT };
enum class HardEvent { S_MTE2, MTE2_S, S_MTE3, MTE3_S };
template <int>
inline void PipeBarrier() {}
template <typename T>
inline T AtomicAdd(T *address, T value) {
  ++scalar_calls;
  return __atomic_fetch_add(address, value, __ATOMIC_SEQ_CST);
}
template <typename T>
inline void SetAtomicAdd() {}
inline void SetAtomicNone() {}
inline int64_t GetSystemCycle() { return ++current_cycle; }
}  // namespace AscendC
using AscendC::PipeBarrier;

template <AscendC::HardEvent>
inline void SyncFunc() {}

template <typename T>
struct GlobalTensor {
  T *data;
  const T *GetPhyAddr() const { return data; }
  T *GetPhyAddr(uint64_t offset) const { return data + offset; }
  GlobalTensor operator[](uint32_t index) const { return {data + index}; }
  T GetValue(uint32_t index) const { return data[index]; }
};

template <typename T, CacheLine, DcciDst>
void DataCacheCleanAndInvalid(GlobalTensor<T> tensor) {
  refresh_cycles.push_back(current_cycle);
  if (current_cycle >= release_cycle) {
    *tensor.data = 1;
  }
}

template <typename T>
struct LocalTensor {
  T *data;
  AscendC::TPosition position;
  void SetValue(int index, T value) { data[index] = value; }
};

template <AscendC::TPosition Position>
struct TBuf {
  std::array<int32_t, 8> data{};
  template <typename T>
  LocalTensor<T> GetWithOffset(uint32_t, uint32_t) { return {data.data(), Position}; }
  void FreeTensor(LocalTensor<int32_t>) {}
};

struct TPipe {
  template <typename Buffer>
  void InitBuffer(Buffer &, uint32_t) {}
  void Destroy() {}
};

template <typename T>
void DataCopy(LocalTensor<T> destination, GlobalTensor<T> source, uint32_t count) {
  for (uint32_t index = 0; index < count; ++index) {
    destination.data[index] = source.data[index];
  }
}

template <typename T>
void DataCopy(GlobalTensor<T> destination, LocalTensor<T> source, uint32_t count) {
  // Match the unsupported L1-to-GM branch in the Ascend 950 SDK.
  if (__NPU_ARCH__ == 3510 && source.position == AscendC::TPosition::A1) {
    ++unsupported_copies;
    return;
  }
  for (uint32_t index = 0; index < count; ++index) {
    __atomic_fetch_add(destination.data + index, source.data[index], __ATOMIC_SEQ_CST);
  }
}

class EventProbe {
 public:
  GlobalTensor<int32_t> all_event_counters;
  GlobalTensor<int32_t> atomic_add_values;
  GlobalTensor<int32_t> all_event_num_triggers{};
  bool pull_protocol_ = false;
  // The test supplies the exact production method, not a reimplemented algorithm.
#include "event_counter_method.inc"
#include "event_poll_method.inc"
};

bool CheckPollIntervals() {
  constexpr int64_t expected_cycles_per_us = __NPU_ARCH__ == 3510 ? 1000 : 50;
#ifdef __DAV_CUBE__
  constexpr uint32_t push_interval_us = 50;
  constexpr uint32_t pull_interval_us = 10;
#else
  constexpr uint32_t push_interval_us = 150;
  constexpr uint32_t pull_interval_us = 30;
#endif
  for (bool pull : {false, true}) {
    for (uint32_t requested_us : {0U, 7U}) {
      const uint32_t interval_us = requested_us == 0 ? (pull ? pull_interval_us : push_interval_us) : requested_us;
      const int64_t expected_refresh_cycle = 1 + (interval_us + 1) * expected_cycles_per_us;
      int32_t counter = 0;
      int32_t needed = 1;
      EventProbe probe{{&counter}, {nullptr}, {&needed}, pull};
      current_cycle = 0;
      release_cycle = expected_refresh_cycle;
      refresh_cycles.clear();
      probe.WaitForDependency(0, requested_us);
      if (refresh_cycles != std::vector<int64_t>{0, expected_refresh_cycle} || counter != needed) {
        std::cerr << "pull=" << pull << " interval_us=" << requested_us
                  << " expected_refresh_cycle=" << expected_refresh_cycle << " actual=";
        for (auto cycle : refresh_cycles) {
          std::cerr << cycle << ",";
        }
        std::cerr << "\n";
        return false;
      }
      // An already completed event must not enter the timed refresh loop.
      current_cycle = 0;
      refresh_cycles.clear();
      probe.WaitForDependency(0, requested_us);
      if (refresh_cycles != std::vector<int64_t>{0}) {
        return false;
      }
    }
  }
  std::cout << "EVENT_POLL_PASS cycles_per_us=" << expected_cycles_per_us << "\n";
  return true;
}

int main() {
  std::array<int32_t, 16> counters;
  counters.fill(991);
  counters[4] = 0;
  std::array<int32_t, 8> increment{1, 0, 0, 0, 0, 0, 0, 0};
  EventProbe probe{{counters.data()}, {increment.data()}};
  std::vector<std::thread> workers;
  for (uint32_t worker = 0; worker < 32; ++worker) {
    workers.emplace_back([&probe]() { probe.AtomicAddForAllEventCounters(4); });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  for (uint32_t index = 0; index < counters.size(); ++index) {
    int32_t expected = index == 4 ? 32 : 991;
    if (counters[index] != expected) {
      std::cerr << "event=" << index << " expected=" << expected << " actual=" << counters[index] << "\n";
      return 1;
    }
  }
  int expected_scalar_calls = 0;
#if defined(__DAV_CUBE__) && __NPU_ARCH__ == 3510
  expected_scalar_calls = 32;
#endif
  if (scalar_calls != expected_scalar_calls || unsupported_copies != 0) {
    return 2;
  }
  std::cout << "EVENT_COUNTER_PASS scalar_calls=" << scalar_calls << "\n";
  return CheckPollIntervals() ? 0 : 3;
}
