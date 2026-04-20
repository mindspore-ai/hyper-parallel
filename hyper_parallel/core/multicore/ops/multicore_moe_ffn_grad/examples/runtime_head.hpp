#ifndef MULTICORE_MOE_FFN_GRAD_RUNTIME_HEAD_TEST_CODE
#define MULTICORE_MOE_FFN_GRAD_RUNTIME_HEAD_TEST_CODE

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

// constexpr uint32_t MAX_TASK_NUM = 2560;
// constexpr uint32_t MAX_EVENT_NUM = 256;
// constexpr uint32_t NUM_WORKERS_VECTOR = 48;
// constexpr uint32_t NUM_WORKERS_CUBE = 24;
// constexpr uint32_t QUEUE_CAPACITY = 100;
// constexpr uint32_t TASK_TYPE_INDEX_NUM = 2048;
// constexpr uint32_t MAX_GROUP_LIST = 512;
// constexpr uint32_t ATOMIC_ADD_VALUE_LEN = 8;

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
  EventDesc(void)
      : event_type(EVENT_INVALID), num_triggers(0),
        first_task_id(TASK_TERMINATE), last_task_id(TASK_TERMINATE) {}
  EventDesc(EventType type, int nt, uint32_t f, uint32_t l)
      : event_type(type), num_triggers(nt), first_task_id(f), last_task_id(l) {}
  EventType event_type;
  uint32_t num_triggers;
  uint32_t first_task_id, last_task_id;
};

struct TaskDesc {
  TaskDesc(TaskType t, TaskAiCoreType task_aicore_type)
      : task_type(t), task_aicore_type(task_aicore_type), num_inputs(0), num_outputs(0),
        trigger_event(TASK_TERMINATE), dependent_event(TASK_TERMINATE) {}
  TaskDesc() {}
  TaskType task_type;
  TaskAiCoreType task_aicore_type;
  uint32_t num_inputs, num_outputs;
  uint32_t trigger_event;
  uint32_t dependent_event;
  TensorDesc inputs[MAX_INPUTS_PER_TASK];
  TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
  // uint32_t inputs_transpose[MAX_INPUTS_PER_TASK*4];
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

struct RuntimeConfig {
  uint32_t task_num;
  uint32_t num_workers;
  uint32_t queue_capacity;
  uint32_t config_extra_value;

  // int32_t all_event_counters[MAX_EVENT_NUM]={0};
  int32_t all_event_num_triggers[MAX_EVENT_NUM] = {0};

  TaskDesc all_tasks[MAX_TASK_NUM];
  EventDesc all_events[MAX_EVENT_NUM];
  // TaskDesc all_tasks_cube[MAX_TASK_NUM];

  int32_t task_index_num[4] = {0};
  int32_t cube_task_indexs[TASK_TYPE_INDEX_NUM];
  int32_t vector_task_indexs[TASK_TYPE_INDEX_NUM];
  int32_t mix_task_indexs[TASK_TYPE_INDEX_NUM];

  // uint32_t expert_num_signle_rank;
  // uint32_t grouped_list_shape;
  DynamicData dynamic_data;
  int64_t grouped_matmul_group_list[MAX_GROUP_LIST]={0};
  int32_t atomic_add_values[ATOMIC_ADD_VALUE_LEN]={0};
};

void init_runtime_config(RuntimeConfig* runtimeConfig, uint32_t num_workers) {
  runtimeConfig->num_workers = num_workers;
  runtimeConfig->queue_capacity = QUEUE_CAPACITY;
}

struct TaskSplitValue{
  int64_t tp = 4;
  int64_t ep = 4;
  int64_t seq_size = 8192;
  int64_t all_expert_num = 32;
  int64_t top_k = 8;
  int64_t single_rank_expert_num = all_expert_num / ep;
  int64_t seq_all = (seq_size * ep * top_k) / tp;
  int64_t per_expert_seq = seq_all / single_rank_expert_num;
  int64_t per_rank_seq = seq_all / ep;
  int64_t per_expert_seq_to_other = seq_all / (ep * top_k);

  int64_t alltoall_split_value;
  int64_t alltoall_task_num;
  int64_t gmm_split_value_g4;
  int64_t gmm_task_num_g4;
  int64_t gmm_split_value;
  int64_t gmm_task_num;
  int64_t swiglu_split_value;
  int64_t swiglu_task_num;
  int64_t gmm_split_value_g2;
  int64_t gmm_task_num_g2;
  int64_t alltoall_split_value_a2;
  int64_t alltoall_task_num_a2;
  int64_t gmm_split_value_g3;
  int64_t gmm_task_num_g3;
  int64_t rank_id = 0;
  int64_t pre_pre_event_num = 0;
  int64_t pre_event_num = 0;
  int64_t pre_task_num = 0;
  int64_t pre_cube_task_num = 0;
  int64_t pre_vector_task_num = 0;
  int64_t pre_mix_task_num = 0;
  int64_t all_event_num = 1 + all_expert_num + single_rank_expert_num + single_rank_expert_num + single_rank_expert_num;
};

void init_task_split_value(TaskSplitValue &task_split_value) {
  task_split_value.pre_pre_event_num = 0;
  task_split_value.pre_event_num = 0;
  task_split_value.pre_task_num = 0;
  task_split_value.pre_cube_task_num = 0;
  task_split_value.pre_vector_task_num = 0;
  task_split_value.pre_mix_task_num = 0;
}

#endif // MULTICORE_MOE_FFN_RUNTIME_HEAD_TEST_CODE
