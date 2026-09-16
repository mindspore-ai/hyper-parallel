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

static_assert(sizeof(TensorDesc) == 64);
static_assert(sizeof(TaskDesc) == 576);

extern "C" bool valid_runtime(uint8_t *image, uint64_t bytes, uint64_t event_bytes) {
  return isRuntimeStorageValid(image, bytes, event_bytes);
}

extern "C" bool valid_ready_runtime(uint8_t *image, uint64_t bytes, uint64_t event_bytes, uint32_t ep_size) {
  return isRuntimeStorageValid(image, bytes, event_bytes, ep_size);
}

extern "C" bool valid_expert_runtime(uint8_t *image, uint64_t bytes, uint64_t event_bytes, uint64_t local_experts) {
  return isRuntimeStorageValid(image, bytes, event_bytes, 1, local_experts);
}

extern "C" uint32_t read_ready_event(uint8_t *image) {
  ReadyHandshakeMeta meta;
  getReadyHandshakeMeta(image, &meta);
  return meta.ready_event;
}

extern "C" uint32_t group_list_offset(uint8_t *image, uint32_t worker_id) {
  return getGroupedMatmulGroupListOffsetById(image, worker_id);
}

extern "C" void read_task(uint8_t *image, uint32_t index, TaskDesc *result) { getTaskDesc(image, result, index); }

extern "C" void read_layout(uint8_t *image, uint32_t *result) {
  result[0] = getAllEventNumTriggersOffset();
  result[1] = getAllTasksOffset(image);
  result[2] = getAllEventsOffset(image);
  result[3] = getTaskIndexNumOffset(image);
  result[4] = getCubeTaskIndexsOffset(image);
  result[5] = getVectorTaskIndexsOffset(image);
  result[6] = getMixTaskIndexsOffset(image);
  result[7] = getDynamicDataOffset(image);
  result[8] = getGroupedMatmulGroupListOffsetById(image, 1);
  result[9] = getAtomicAddValuesOffset(image);
  result[10] = getRuntimeTaskCapacity(image);
  result[11] = getRuntimeEventCapacity(image);
}

extern "C" void read_protocol_profile(uint8_t *image, uint32_t *result) {
  ReadyHandshakeMeta meta;
  getReadyHandshakeMeta(image, &meta);
  result[0] = meta.ready_event;
  result[1] = meta.completion_event;
  result[2] = isCycleProfileEnabled(image);
  result[3] = getAicProfileRecordCapacity(image);
  result[4] = getAivProfileRecordCapacity(image);
}
