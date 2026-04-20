#ifndef MULTICORE_MOE_FFN_GRAD_RUNTIME_HEAD_TEST_KERNEL_CODE
#define MULTICORE_MOE_FFN_GRAD_RUNTIME_HEAD_TEST_KERNEL_CODE

constexpr uint32_t MAX_TENSOR_DIMS = 4;
constexpr uint32_t MAX_INPUTS_PER_TASK = 4;
constexpr uint32_t MAX_OUTPUTS_PER_TASK = 4;

constexpr uint32_t MAX_TASK_NUM = 256*100;
constexpr uint32_t MAX_EVENT_NUM = 1024;
constexpr uint32_t NUM_WORKERS_VECTOR = 48;
constexpr uint32_t NUM_WORKERS_CUBE = 24;
constexpr uint32_t QUEUE_CAPACITY = 100;
constexpr uint32_t TASK_TYPE_INDEX_NUM = 256*100;
constexpr uint32_t MAX_GROUP_LIST = 512;
constexpr uint32_t ATOMIC_ADD_VALUE_LEN = 8;

constexpr uint32_t UB_32B_ALIGN = 32;
constexpr uint32_t EXP_TOKEN_COUNT_FLAG_CNT = UB_32B_ALIGN / sizeof(int32_t);  // 8
constexpr uint32_t DISPATCH_TOKEN_UB_SIZE = 3 * 32;

constexpr uint32_t UINT32_T_SIZE = sizeof(uint32_t);
constexpr uint32_t INT32_T_SIZE = sizeof(int32_t);
constexpr uint32_t INT64_T_SIZE = sizeof(int64_t);

constexpr uint32_t EVENT_INVALID_ID = 0xFFFFFFFF;

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
  TASK_SHMEM_PUT_MEM_SINGAL = 105,
  TASK_SWI_GLU_GRAD = 106,
};

enum EventType : uint32_t {
  EVENT_EMPTY = 900,
  EVENT_LAUNCH_TASKS = 901,
  EVENT_LAUNCH_MASSIVE_TASKS = 902,
  EVENT_LAUNCH_DEPENDENT_TASKS = 903,
  EVENT_END_OF_TASK_GRAPH = 910,
  EVENT_TERMINATION = 911, // TASK_TERMINATE
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
  DYNAMIC_DSV3_MOE_FFN = 101,
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
  uint32_t extra_value_3;
  uint32_t extra_value_4;
};

struct RuntimeConfig {
  uint32_t task_num;
  uint32_t num_workers;
  uint32_t queue_capacity;
  uint32_t config_extra_value;

  int32_t* all_event_num_triggers;

  TaskDesc* all_tasks;
  EventDesc* all_events;

  int32_t* task_index_num; // 3
  int32_t* cube_task_indexs; // TASK_TYPE_INDEX_NUM
  int32_t* vector_task_indexs; // TASK_TYPE_INDEX_NUM
  int32_t* mix_task_indexs; // TASK_TYPE_INDEX_NUM

  DynamicData dynamic_data;
  int64_t* grouped_matmul_group_list;
  int32_t* atomic_add_values;
};

__aicore__ inline uint32_t getTaskNum(__gm__ uint8_t *tiling)
{
  return (*(__gm__ uint32_t *)(tiling));
}

__aicore__ inline uint32_t getAllEventNumTriggersOffset()
{
  return UINT32_T_SIZE * 4;
}

__aicore__ inline uint32_t getAllTasksOffset()
{
  return UINT32_T_SIZE * 4 + INT32_T_SIZE * MAX_EVENT_NUM;
}

__aicore__ inline uint32_t getAllEventsOffset()
{
  uint32_t start_size = getAllTasksOffset();

  uint32_t tensor_desc_size = UINT32_T_SIZE * 8 + UINT32_T_SIZE * MAX_TENSOR_DIMS * 2;
  uint32_t task_desc_size = UINT32_T_SIZE * 6 + MAX_INPUTS_PER_TASK * tensor_desc_size + MAX_OUTPUTS_PER_TASK * tensor_desc_size + UINT32_T_SIZE * 10;

  return start_size + task_desc_size * MAX_TASK_NUM;
}

__aicore__ inline void getTaskDesc(__gm__ uint8_t *tiling, TaskDesc *tilingData, uint32_t index_size)
{
  uint32_t tensor_desc_size = UINT32_T_SIZE * 8 + UINT32_T_SIZE * MAX_TENSOR_DIMS * 2;
  uint32_t task_desc_size = UINT32_T_SIZE * 6 + MAX_INPUTS_PER_TASK * tensor_desc_size + MAX_OUTPUTS_PER_TASK * tensor_desc_size + UINT32_T_SIZE * 10;
  uint32_t size = getAllTasksOffset() + index_size * task_desc_size;

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
}

__aicore__ inline void getEventDesc(__gm__ uint8_t *tiling, EventDesc *tilingData, uint32_t index_size)
{
  uint32_t size = getAllEventsOffset() + index_size * 4 * UINT32_T_SIZE;

  tilingData->event_type = (*(__gm__ EventType *)(tiling + size));
  tilingData->num_triggers = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE));
  tilingData->first_task_id = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 2));
  tilingData->last_task_id = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 3));
}

__aicore__ inline uint32_t getTaskIndexNumOffset()
{
  return getAllEventsOffset() + MAX_EVENT_NUM * 4 * UINT32_T_SIZE;
}

__aicore__ inline int32_t getTaskIndexNumByTaskType(__gm__ uint8_t *tiling, TaskAiCoreType task_aicore_type)
{
  uint32_t size = getAllEventsOffset() + MAX_EVENT_NUM * 4 * UINT32_T_SIZE;
  if (task_aicore_type == TaskAiCoreType::TASK_AICORE_VECTOR) {
    return (*(__gm__ int32_t *)(tiling + size + INT32_T_SIZE));
  } else if (task_aicore_type == TaskAiCoreType::TASK_AICORE_CUBE) {
    return (*(__gm__ int32_t *)(tiling + size));
  } else {
    return (*(__gm__ int32_t *)(tiling + size + INT32_T_SIZE * 2));
  }
}

__aicore__ inline uint32_t getCubeTaskIndexsOffset()
{
  return getTaskIndexNumOffset() + 4 * INT32_T_SIZE;
}

__aicore__ inline uint32_t getVectorTaskIndexsOffset()
{
  return getCubeTaskIndexsOffset() + TASK_TYPE_INDEX_NUM * INT32_T_SIZE;
}

__aicore__ inline uint32_t getMixTaskIndexsOffset()
{
  return getVectorTaskIndexsOffset() + TASK_TYPE_INDEX_NUM * INT32_T_SIZE;
}

__aicore__ inline uint32_t getDynamicDataOffset()
{
  return getMixTaskIndexsOffset() + TASK_TYPE_INDEX_NUM * INT32_T_SIZE;
}

__aicore__ inline void getDynamicData(__gm__ uint8_t *tiling, DynamicData *tilingData)
{
  uint32_t size = getDynamicDataOffset();
  tilingData->dynamic_type = (*(__gm__ DynamicType *)(tiling + size));
  tilingData->dynamic_input_position = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE));
  tilingData->dynamic_group_size = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 2));
  tilingData->dynamic_max_seq_len = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 3));
}

__aicore__ inline uint32_t getGroupedMatmulGroupListOffset()
{
  return getDynamicDataOffset() + 4 * UINT32_T_SIZE;
}

__aicore__ inline uint32_t getGroupedMatmulGroupListOffsetById(__gm__ uint8_t *tiling, uint32_t block_idx)
{
  return getGroupedMatmulGroupListOffset() + 8 * INT64_T_SIZE * block_idx;
}

__aicore__ inline uint32_t getAtomicAddValuesOffset()
{
  return getGroupedMatmulGroupListOffset() + MAX_GROUP_LIST * INT64_T_SIZE;
}

__aicore__ inline int64_t getExtraValueFromTiling(__gm__ uint8_t *tiling, uint32_t index)
{
  return (*(__gm__ int64_t *)(tiling + index * INT64_T_SIZE));
}

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    uint32_t eventID = static_cast<uint32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#endif // MULTICORE_MOE_FFN_GRAD_RUNTIME_HEAD_TEST_KERNEL_CODE
