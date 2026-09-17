/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MULTICORE_SCHEDULER_RUNTIME_CONFIG_HPP
#define MULTICORE_SCHEDULER_RUNTIME_CONFIG_HPP

constexpr uint32_t MAX_TENSOR_DIMS = 4;
constexpr uint32_t MAX_INPUTS_PER_TASK = 4;
constexpr uint32_t MAX_OUTPUTS_PER_TASK = 4;

constexpr uint32_t MIN_EVENT_CAPACITY = 1024;
constexpr uint32_t NUM_WORKERS_VECTOR = 48;
constexpr uint32_t NUM_WORKERS_CUBE = 24;
constexpr uint32_t MAX_GROUP_LIST = 512;
constexpr uint32_t MAX_EXPERT_NUM_PER_RANK = 16;
constexpr uint32_t ATOMIC_ADD_VALUE_LEN = 8;
constexpr uint32_t RUNTIME_PROTOCOL_VERSION = 1;

constexpr uint32_t GROUP_LIST_CACHE_LINE_BYTES = 128;
static_assert(MAX_EXPERT_NUM_PER_RANK * sizeof(int64_t) % GROUP_LIST_CACHE_LINE_BYTES == 0,
              "Grouped-list worker stride must preserve cache-line alignment.");
static_assert((GROUP_LIST_CACHE_LINE_BYTES - 1) + NUM_WORKERS_CUBE * MAX_EXPERT_NUM_PER_RANK * sizeof(int64_t) <=
                MAX_GROUP_LIST * sizeof(int64_t),
              "Grouped-list scratch slots exceed the runtime-config buffer.");

constexpr uint32_t UB_32B_ALIGN = 32;
constexpr uint32_t DATA_CACHE_LINE_SIZE = 64;
constexpr uint32_t EXP_TOKEN_COUNT_FLAG_CNT = UB_32B_ALIGN / sizeof(int32_t);  // 8
constexpr uint32_t DISPATCH_TOKEN_UB_SIZE = 3 * 32;

constexpr uint32_t UINT32_T_SIZE = sizeof(uint32_t);
constexpr uint32_t INT32_T_SIZE = sizeof(int32_t);
constexpr uint32_t INT64_T_SIZE = sizeof(int64_t);

constexpr uint32_t EVENT_INVALID_ID = 0xFFFFFFFF;
constexpr uint32_t PROFILE_DESC_INVALID_ID = 0xFFFFFFFF;

typedef uint32_t TaskId;

enum TaskAiCoreType : uint32_t {
  TASK_AICORE_INVALID = 0,
  TASK_AICORE_CUBE = 1,
  TASK_AICORE_VECTOR = 2,
  TASK_AICORE_MIX = 3,
};

enum TaskType : uint32_t {
  TASK_TERMINATE = 0,
  TASK_BEGIN_TASK_GRAPH = 10,
  // compute task starts from 100
  TASK_ADD_CUSTOM = 101,
  TASK_SWI_GLU = 102,
  TASK_MATMUL = 103,
  TASK_GROUPED_MATMUL = 104,
  TASK_SHMEM_PUT_MEM_SIGNAL = 105,
  TASK_SWI_GLU_GRAD = 106,
  TASK_SHMEM_GET_MEM = 107,
};

enum EventType : uint32_t {
  EVENT_EMPTY = 900,
  EVENT_LAUNCH_TASKS = 901,
  EVENT_LAUNCH_MASSIVE_TASKS = 902,
  EVENT_LAUNCH_DEPENDENT_TASKS = 903,
  EVENT_END_OF_TASK_GRAPH = 910,
  EVENT_TERMINATION = 911,  // TASK_TERMINATE
  EVENT_INVALID = 999,
};

struct TensorDesc {
  uint32_t tensor_type;
  uint32_t num_dims;
  uint32_t dim[MAX_TENSOR_DIMS];
  uint32_t stride[MAX_TENSOR_DIMS];
  uint32_t data_type;
  uint32_t input_position;
  uint32_t base_ptr_offset;
  uint32_t transpose_flag;
  uint32_t dynamic_shape;
  uint32_t dynamic_dim;
};

struct EventDesc {
  EventType event_type;
  uint32_t num_triggers;
  uint32_t first_task_id, last_task_id;
};

enum struct DynamicType : uint32_t {
  DYNAMIC_EMPTY = 0,
  DYNAMIC_DSV3_MOE = 101,
};

struct DynamicData {
  DynamicType dynamic_type;
  uint32_t dynamic_input_position;
  uint32_t dynamic_group_size;
  uint32_t dynamic_max_seq_len;
};

struct TaskDesc {
  TaskType task_type;
  TaskAiCoreType task_aicore_type;
  uint32_t num_inputs, num_outputs;
  uint32_t trigger_event;
  uint32_t dependent_event;
  TensorDesc inputs[MAX_INPUTS_PER_TASK];
  TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
  uint32_t tiling_data_position;
  uint32_t tiling_data_offset;
  uint32_t task_index;
  uint32_t task_split_num;
  uint32_t task_split_value;
  uint32_t extra_value_0;
  uint32_t extra_value_1;
  uint32_t extra_value_2;
  uint32_t profile_desc_id;
  uint32_t profile_owner_id;
};

struct ReadyHandshakeMeta {
  uint32_t ready_event;
  uint32_t completion_event;
};

struct RuntimeHeader {
  uint32_t task_num;
  uint32_t num_workers;
  uint32_t task_capacity;
  uint32_t event_capacity;
  uint32_t ready_event;
  uint32_t cycle_profiling_enabled;
  uint32_t aic_profile_record_capacity;
  uint32_t aiv_profile_record_capacity;
  uint32_t completion_event;
  uint32_t protocol_version;
  uint32_t padding[6];
};

static_assert(sizeof(RuntimeHeader) == 64);
static_assert(sizeof(TensorDesc) == 64);
static_assert(sizeof(TaskDesc) == 576);

__aicore__ inline uint32_t getTaskNum(__gm__ uint8_t *tiling) { return (*(__gm__ uint32_t *)(tiling)); }

__aicore__ inline uint32_t getRuntimeTaskCapacity(__gm__ uint8_t *tiling) {
  return (*(__gm__ uint32_t *)(tiling + 2 * UINT32_T_SIZE));
}

__aicore__ inline uint32_t getRuntimeEventCapacity(__gm__ uint8_t *tiling) {
  return (*(__gm__ uint32_t *)(tiling + 3 * UINT32_T_SIZE));
}

__aicore__ inline void getReadyHandshakeMeta(__gm__ uint8_t *tiling, ReadyHandshakeMeta *meta) {
  meta->ready_event = (*(__gm__ uint32_t *)(tiling + 4 * UINT32_T_SIZE));
  meta->completion_event = (*(__gm__ uint32_t *)(tiling + 8 * UINT32_T_SIZE));
}

__aicore__ inline uint32_t getAllEventNumTriggersOffset() { return sizeof(RuntimeHeader); }

__aicore__ inline uint32_t getAllTasksOffset(__gm__ uint8_t *tiling) {
  return getAllEventNumTriggersOffset() + INT32_T_SIZE * getRuntimeEventCapacity(tiling);
}

__aicore__ inline uint32_t getAllEventsOffset(__gm__ uint8_t *tiling) {
  uint32_t start_size = getAllTasksOffset(tiling);

  uint32_t tensor_desc_size = UINT32_T_SIZE * 8 + UINT32_T_SIZE * MAX_TENSOR_DIMS * 2;
  uint32_t task_desc_size = UINT32_T_SIZE * 6 + MAX_INPUTS_PER_TASK * tensor_desc_size +
                            MAX_OUTPUTS_PER_TASK * tensor_desc_size + UINT32_T_SIZE * 10;

  return start_size + task_desc_size * getRuntimeTaskCapacity(tiling);
}

__aicore__ inline void getTaskDesc(__gm__ uint8_t *tiling, TaskDesc *tilingData, uint32_t index_size) {
  uint32_t tensor_desc_size = UINT32_T_SIZE * 8 + UINT32_T_SIZE * MAX_TENSOR_DIMS * 2;
  uint32_t task_desc_size = UINT32_T_SIZE * 6 + MAX_INPUTS_PER_TASK * tensor_desc_size +
                            MAX_OUTPUTS_PER_TASK * tensor_desc_size + UINT32_T_SIZE * 10;
  uint32_t size = getAllTasksOffset(tiling) + index_size * task_desc_size;

  tilingData->task_type = (*(__gm__ TaskType *)(tiling + size));
  tilingData->task_aicore_type = (*(__gm__ TaskAiCoreType *)(tiling + size + UINT32_T_SIZE));
  tilingData->num_inputs = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 2));
  tilingData->num_outputs = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 3));
  tilingData->trigger_event = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 4));
  tilingData->dependent_event = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 5));
  uint32_t start_size = size + UINT32_T_SIZE * 6;
  for (uint32_t i = 0; i < MAX_INPUTS_PER_TASK; i++) {
    (tilingData->inputs)[i].tensor_type = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->inputs)[i].num_dims = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    for (uint32_t j = 0; j < MAX_TENSOR_DIMS; j++) {
      (tilingData->inputs)[i].dim[j] = (*(__gm__ uint32_t *)(tiling + start_size));
      start_size = start_size + UINT32_T_SIZE;
    }
    for (uint32_t j = 0; j < MAX_TENSOR_DIMS; j++) {
      (tilingData->inputs)[i].stride[j] = (*(__gm__ uint32_t *)(tiling + start_size));
      start_size = start_size + UINT32_T_SIZE;
    }
    (tilingData->inputs)[i].data_type = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->inputs)[i].input_position = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->inputs)[i].base_ptr_offset = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->inputs)[i].transpose_flag = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->inputs)[i].dynamic_shape = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->inputs)[i].dynamic_dim = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
  }
  for (uint32_t i = 0; i < MAX_INPUTS_PER_TASK; i++) {
    (tilingData->outputs)[i].tensor_type = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->outputs)[i].num_dims = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    for (uint32_t j = 0; j < MAX_TENSOR_DIMS; j++) {
      (tilingData->outputs)[i].dim[j] = (*(__gm__ uint32_t *)(tiling + start_size));
      start_size = start_size + UINT32_T_SIZE;
    }
    for (uint32_t j = 0; j < MAX_TENSOR_DIMS; j++) {
      (tilingData->outputs)[i].stride[j] = (*(__gm__ uint32_t *)(tiling + start_size));
      start_size = start_size + UINT32_T_SIZE;
    }
    (tilingData->outputs)[i].data_type = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->outputs)[i].input_position = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->outputs)[i].base_ptr_offset = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->outputs)[i].transpose_flag = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->outputs)[i].dynamic_shape = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->outputs)[i].dynamic_dim = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
  }
  tilingData->tiling_data_position = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size = start_size + UINT32_T_SIZE;
  tilingData->tiling_data_offset = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size = start_size + UINT32_T_SIZE;
  tilingData->task_index = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size = start_size + UINT32_T_SIZE;
  tilingData->task_split_num = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size = start_size + UINT32_T_SIZE;
  tilingData->task_split_value = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size += UINT32_T_SIZE;
  tilingData->extra_value_0 = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size += UINT32_T_SIZE;
  tilingData->extra_value_1 = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size += UINT32_T_SIZE;
  tilingData->extra_value_2 = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size += UINT32_T_SIZE;
  tilingData->profile_desc_id = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size += UINT32_T_SIZE;
  tilingData->profile_owner_id = (*(__gm__ uint32_t *)(tiling + start_size));
}

__aicore__ inline uint32_t getTaskProfileDescId(__gm__ uint8_t *tiling, uint32_t index_size) {
  uint32_t tensor_desc_size = UINT32_T_SIZE * 8 + UINT32_T_SIZE * MAX_TENSOR_DIMS * 2;
  uint32_t task_desc_size = UINT32_T_SIZE * 6 + MAX_INPUTS_PER_TASK * tensor_desc_size +
                            MAX_OUTPUTS_PER_TASK * tensor_desc_size + UINT32_T_SIZE * 10;
  uint32_t size = getAllTasksOffset(tiling) + (index_size + 1) * task_desc_size - UINT32_T_SIZE * 2;
  return (*(__gm__ uint32_t *)(tiling + size));
}

__aicore__ inline uint32_t getTaskProfileOwnerId(__gm__ uint8_t *tiling, uint32_t index_size) {
  uint32_t tensor_desc_size = UINT32_T_SIZE * 8 + UINT32_T_SIZE * MAX_TENSOR_DIMS * 2;
  uint32_t task_desc_size = UINT32_T_SIZE * 6 + MAX_INPUTS_PER_TASK * tensor_desc_size +
                            MAX_OUTPUTS_PER_TASK * tensor_desc_size + UINT32_T_SIZE * 10;
  uint32_t size = getAllTasksOffset(tiling) + (index_size + 1) * task_desc_size - UINT32_T_SIZE;
  return (*(__gm__ uint32_t *)(tiling + size));
}

__aicore__ inline void getEventDesc(__gm__ uint8_t *tiling, EventDesc *tilingData, uint32_t index_size) {
  uint32_t size = getAllEventsOffset(tiling) + index_size * 4 * UINT32_T_SIZE;

  tilingData->event_type = (*(__gm__ EventType *)(tiling + size));
  tilingData->num_triggers = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE));
  tilingData->first_task_id = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 2));
  tilingData->last_task_id = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 3));
}

__aicore__ inline uint32_t getTaskIndexNumOffset(__gm__ uint8_t *tiling) {
  return getAllEventsOffset(tiling) + getRuntimeEventCapacity(tiling) * 4 * UINT32_T_SIZE;
}

__aicore__ inline int32_t getTaskIndexNumByTaskType(__gm__ uint8_t *tiling, TaskAiCoreType task_aicore_type) {
  uint32_t size = getAllEventsOffset(tiling) + getRuntimeEventCapacity(tiling) * 4 * UINT32_T_SIZE;
  if (task_aicore_type == TaskAiCoreType::TASK_AICORE_VECTOR) {
    return (*(__gm__ int32_t *)(tiling + size + INT32_T_SIZE));
  } else if (task_aicore_type == TaskAiCoreType::TASK_AICORE_CUBE) {
    return (*(__gm__ int32_t *)(tiling + size));
  } else {
    return (*(__gm__ int32_t *)(tiling + size + INT32_T_SIZE * 2));
  }
}

__aicore__ inline uint32_t getCubeTaskIndexsOffset(__gm__ uint8_t *tiling) {
  return getTaskIndexNumOffset(tiling) + 4 * INT32_T_SIZE;
}

__aicore__ inline uint32_t getVectorTaskIndexsOffset(__gm__ uint8_t *tiling) {
  return getCubeTaskIndexsOffset(tiling) + getRuntimeTaskCapacity(tiling) * INT32_T_SIZE;
}

__aicore__ inline uint32_t getMixTaskIndexsOffset(__gm__ uint8_t *tiling) {
  return getVectorTaskIndexsOffset(tiling) + getRuntimeTaskCapacity(tiling) * INT32_T_SIZE;
}

__aicore__ inline uint32_t getDynamicDataOffset(__gm__ uint8_t *tiling) {
  return getMixTaskIndexsOffset(tiling) + getRuntimeTaskCapacity(tiling) * INT32_T_SIZE;
}

__aicore__ inline void getDynamicData(__gm__ uint8_t *tiling, DynamicData *tilingData) {
  uint32_t size = getDynamicDataOffset(tiling);
  tilingData->dynamic_type = (*(__gm__ DynamicType *)(tiling + size));
  tilingData->dynamic_input_position = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE));
  tilingData->dynamic_group_size = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 2));
  tilingData->dynamic_max_seq_len = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 3));
}

__aicore__ inline uint32_t getGroupedMatmulGroupListOffset(__gm__ uint8_t *tiling) {
  return getDynamicDataOffset(tiling) + 4 * UINT32_T_SIZE;
}

__aicore__ inline uint32_t getGroupedMatmulGroupListOffsetById(__gm__ uint8_t *tiling, uint32_t worker_id) {
  // Each worker must own complete cache lines when publishing its group list.
  const uint32_t aligned_offset = (getGroupedMatmulGroupListOffset(tiling) + GROUP_LIST_CACHE_LINE_BYTES - 1) /
                                  GROUP_LIST_CACHE_LINE_BYTES * GROUP_LIST_CACHE_LINE_BYTES;
  return aligned_offset + MAX_EXPERT_NUM_PER_RANK * INT64_T_SIZE * worker_id;
}

__aicore__ inline uint32_t getAtomicAddValuesOffset(__gm__ uint8_t *tiling) {
  return getGroupedMatmulGroupListOffset(tiling) + MAX_GROUP_LIST * INT64_T_SIZE;
}

__aicore__ inline bool isCycleProfileEnabled(__gm__ uint8_t *tiling) {
  return (*(__gm__ uint32_t *)(tiling + 5 * UINT32_T_SIZE)) != 0;
}

__aicore__ inline uint32_t getAicProfileRecordCapacity(__gm__ uint8_t *tiling) {
  return (*(__gm__ uint32_t *)(tiling + 6 * UINT32_T_SIZE));
}

__aicore__ inline uint32_t getAivProfileRecordCapacity(__gm__ uint8_t *tiling) {
  return (*(__gm__ uint32_t *)(tiling + 7 * UINT32_T_SIZE));
}

__aicore__ inline int64_t getExtraValueFromTiling(__gm__ uint8_t *tiling, uint32_t index) {
  return (*(__gm__ int64_t *)(tiling + index * INT64_T_SIZE));
}

__aicore__ inline bool isRuntimeHandshakeValid(__gm__ uint8_t *tiling, uint64_t events, uint64_t event_bytes,
                                               uint32_t ep_size) {
  ReadyHandshakeMeta meta;
  getReadyHandshakeMeta(tiling, &meta);
  if (meta.ready_event != 0 &&
      (ep_size <= 1 || static_cast<uint64_t>(meta.ready_event) + ATOMIC_ADD_VALUE_LEN > events ||
       event_bytes < events * INT32_T_SIZE + 2 * (static_cast<uint64_t>(ep_size) + 1) * DATA_CACHE_LINE_SIZE)) {
    return false;
  }
  if (meta.completion_event != 0 && static_cast<uint64_t>(meta.completion_event) + ATOMIC_ADD_VALUE_LEN > events) {
    return false;
  }
  return true;
}

// Check byte and index bounds before reading variable-sized arrays.
__aicore__ inline bool isRuntimeStorageValid(__gm__ uint8_t *tiling, uint64_t runtime_bytes, uint64_t event_bytes,
                                             uint32_t ep_size = 1) {
  if (runtime_bytes < sizeof(RuntimeHeader) || runtime_bytes >= (1ULL << 32)) {
    return false;
  }
  if (*(__gm__ uint32_t *)(tiling + 9 * UINT32_T_SIZE) > RUNTIME_PROTOCOL_VERSION) {
    return false;
  }
  uint64_t capacity = getRuntimeTaskCapacity(tiling);
  uint64_t events = getRuntimeEventCapacity(tiling);
  if (capacity % 16 != 0 || events < MIN_EVENT_CAPACITY || events % 16 != 0 || getTaskNum(tiling) > capacity ||
      event_bytes < events * INT32_T_SIZE) {
    return false;
  }
  if (!isRuntimeHandshakeValid(tiling, events, event_bytes, ep_size)) {
    return false;
  }
  uint64_t required = sizeof(RuntimeHeader) + events * (INT32_T_SIZE + sizeof(EventDesc)) +
                      capacity * (sizeof(TaskDesc) + 3 * INT32_T_SIZE) + 4 * INT32_T_SIZE + sizeof(DynamicData) +
                      MAX_GROUP_LIST * INT64_T_SIZE + ATOMIC_ADD_VALUE_LEN * INT32_T_SIZE;
  if (required >= (1ULL << 32) || runtime_bytes < required) {
    return false;
  }
  __gm__ int32_t *counts = reinterpret_cast<__gm__ int32_t *>(tiling + getTaskIndexNumOffset(tiling));
  for (uint32_t index = 0; index < 3; ++index) {
    if (counts[index] < 0 || static_cast<uint64_t>(counts[index]) > capacity) {
      return false;
    }
  }
  return true;
}

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
  uint32_t eventID = static_cast<uint32_t>(GetTPipePtr()->FetchEventID(event));
  AscendC::SetFlag<event>(eventID);
  AscendC::WaitFlag<event>(eventID);
}

#endif  // MULTICORE_SCHEDULER_RUNTIME_CONFIG_HPP
