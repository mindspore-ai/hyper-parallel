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
#include <cstdint>

#define __aicore__
#define __gm__

// Only the unused synchronization primitive needs a host stand-in. Layout and
// task decoding below compile the production header itself.
namespace AscendC {
enum class HardEvent {};
struct Pipe {
  uint32_t FetchEventID(HardEvent) { return 0; }
};
inline Pipe *GetTPipePtr() { return nullptr; }
template <HardEvent event>
inline void SetFlag(uint32_t) {}
template <HardEvent event>
inline void WaitFlag(uint32_t) {}
}  // namespace AscendC
using AscendC::GetTPipePtr;
#include "hyper_parallel/core/multicore/ops/runtime/runtime_config.hpp"

namespace Runtime = MulticoreRuntime;

static_assert(sizeof(Runtime::TensorDesc) == 64);
static_assert(sizeof(Runtime::TaskDesc) == 576);

extern "C" bool valid_runtime(uint8_t *image, uint64_t bytes, uint64_t event_bytes) {
  return Runtime::isRuntimeStorageValid(image, bytes, event_bytes);
}

extern "C" bool valid_ready_runtime(uint8_t *image, uint64_t bytes, uint64_t event_bytes, uint32_t ep_size) {
  return Runtime::isRuntimeStorageValid(image, bytes, event_bytes, ep_size);
}

extern "C" uint32_t read_ready_event(uint8_t *image) {
  Runtime::ReadyHandshakeMeta meta;
  Runtime::getReadyHandshakeMeta(image, &meta);
  return meta.ready_event;
}

extern "C" uint32_t group_list_offset(uint8_t *image, uint32_t worker_id) {
  return Runtime::getGroupedMatmulGroupListOffsetById(image, worker_id);
}

extern "C" void read_task(uint8_t *image, uint32_t index, Runtime::TaskDesc *result) {
  Runtime::getTaskDesc(image, result, index);
}

extern "C" void read_layout(uint8_t *image, uint32_t *result) {
  result[0] = Runtime::getAllEventNumTriggersOffset();
  result[1] = Runtime::getAllTasksOffset(image);
  result[2] = Runtime::getAllEventsOffset(image);
  result[3] = Runtime::getTaskIndexNumOffset(image);
  result[4] = Runtime::getCubeTaskIndexsOffset(image);
  result[5] = Runtime::getVectorTaskIndexsOffset(image);
  result[6] = Runtime::getMixTaskIndexsOffset(image);
  result[7] = Runtime::getDynamicDataOffset(image);
  result[8] = Runtime::getGroupedMatmulGroupListOffsetById(image, 1);
  result[9] = Runtime::getAtomicAddValuesOffset(image);
  result[10] = Runtime::getRuntimeTaskCapacity(image);
  result[11] = Runtime::getRuntimeEventCapacity(image);
}
