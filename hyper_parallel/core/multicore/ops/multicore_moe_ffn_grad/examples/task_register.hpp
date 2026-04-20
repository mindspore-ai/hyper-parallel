#ifndef MULTICORE_MOE_FFN_GRAD_TASK_REGISTER_TEST_CODE
#define MULTICORE_MOE_FFN_GRAD_TASK_REGISTER_TEST_CODE

#include <cassert>
#include "tiling_context.hpp"
#include "runtime_head.hpp"

using namespace std;

TaskDesc register_add_custom_task(TensorDesc inputs[], uint32_t input_len, TensorDesc outputs[], uint32_t output_len) {
  TaskDesc taskDesc(TaskType::TASK_ADD_CUSTOM, TaskAiCoreType::TASK_AICORE_VECTOR);
  taskDesc.num_inputs = input_len;
  taskDesc.num_outputs = output_len;
  taskDesc.trigger_event = 1;
  taskDesc.dependent_event = 0;
  for (uint32_t i = 0; i < taskDesc.num_inputs; i++) {
    taskDesc.inputs[i] = inputs[i];
  }
  for (uint32_t i = 0; i < taskDesc.num_outputs; i++) {
    taskDesc.outputs[i] = outputs[i];
  }
  return taskDesc;
}

TaskDesc register_swi_glu_task(TensorDesc inputs[], uint32_t input_len, TensorDesc outputs[], uint32_t output_len) {
  TaskDesc taskDesc(TaskType::TASK_SWI_GLU, TaskAiCoreType::TASK_AICORE_VECTOR);
  taskDesc.num_inputs = input_len;
  taskDesc.num_outputs = output_len;
  taskDesc.trigger_event = 1;
  taskDesc.dependent_event = 0;
  for (uint32_t i = 0; i < taskDesc.num_inputs; i++) {
    taskDesc.inputs[i] = inputs[i];
  }
  for (uint32_t i = 0; i < taskDesc.num_outputs; i++) {
    taskDesc.outputs[i] = outputs[i];
  }
  return taskDesc;
}

TaskDesc register_swi_glu_grad_task(TensorDesc inputs[], uint32_t input_len, TensorDesc outputs[], uint32_t output_len) {
  TaskDesc taskDesc(TaskType::TASK_SWI_GLU_GRAD, TaskAiCoreType::TASK_AICORE_VECTOR);
  taskDesc.num_inputs = input_len;
  taskDesc.num_outputs = output_len;
  taskDesc.trigger_event = 1;
  taskDesc.dependent_event = 0;
  for (uint32_t i = 0; i < taskDesc.num_inputs; i++) {
    taskDesc.inputs[i] = inputs[i];
  }
  for (uint32_t i = 0; i < taskDesc.num_outputs; i++) {
    taskDesc.outputs[i] = outputs[i];
  }
  return taskDesc;
}

TaskDesc register_matmul_task(TensorDesc inputs[], uint32_t input_len, TensorDesc outputs[], uint32_t output_len) {
  TaskDesc taskDesc(TaskType::TASK_MATMUL, TaskAiCoreType::TASK_AICORE_CUBE);
  taskDesc.num_inputs = input_len;
  taskDesc.num_outputs = output_len;
  taskDesc.trigger_event = 1;
  taskDesc.dependent_event = 0;
  for (uint32_t i = 0; i < taskDesc.num_inputs; i++) {
    taskDesc.inputs[i] = inputs[i];
  }
  for (uint32_t i = 0; i < taskDesc.num_outputs; i++) {
    taskDesc.outputs[i] = outputs[i];
  }
  return taskDesc;
}

TaskDesc register_gmm_task(TensorDesc inputs[], uint32_t input_len, TensorDesc outputs[], uint32_t output_len) {
  TaskDesc taskDesc(TaskType::TASK_GROUPED_MATMUL, TaskAiCoreType::TASK_AICORE_CUBE);
  taskDesc.num_inputs = input_len;
  taskDesc.num_outputs = output_len;
  taskDesc.trigger_event = 1;
  taskDesc.dependent_event = 0;
  for (uint32_t i = 0; i < taskDesc.num_inputs; i++) {
    taskDesc.inputs[i] = inputs[i];
  }
  for (uint32_t i = 0; i < taskDesc.num_outputs; i++) {
    taskDesc.outputs[i] = outputs[i];
  }
  return taskDesc;
}

TaskDesc register_put_mem_singal_task(TensorDesc inputs[], uint32_t input_len, TensorDesc outputs[], uint32_t output_len) {
  TaskDesc taskDesc(TaskType::TASK_SHMEM_PUT_MEM_SINGAL, TaskAiCoreType::TASK_AICORE_CUBE);
  taskDesc.num_inputs = input_len;
  taskDesc.num_outputs = output_len;
  taskDesc.trigger_event = 1;
  taskDesc.dependent_event = 0;
  for (uint32_t i = 0; i < taskDesc.num_inputs; i++) {
    taskDesc.inputs[i] = inputs[i];
  }
  for (uint32_t i = 0; i < taskDesc.num_outputs; i++) {
    taskDesc.outputs[i] = outputs[i];
  }
  return taskDesc;
}


void task_split_alltoallv(RuntimeConfig* runtimeConfig, vector<int64_t>& param_position, size_t dtypeSize, std::vector<std::vector<int64_t>>& input_shape, std::vector<std::vector<int64_t>>& output_shape, string layout, TaskSplitValue& taskSplitValue) {
  // input: target_offset、 src、src_offset、size
  // output：target
  int task_num = taskSplitValue.alltoall_task_num;
  int per_g_e_num = task_num / taskSplitValue.all_expert_num;
  int event_trigger_num = (taskSplitValue.alltoall_task_num * taskSplitValue.ep) / taskSplitValue.all_expert_num;
  for (int i = 0; i < task_num; i++) {
      TensorDesc inputs[MAX_INPUTS_PER_TASK];
      TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
      uint32_t input_len = input_shape.size();
      uint32_t output_len = output_shape.size();
      for (uint32_t j = 0; j < input_len; j++) {
          TensorDesc tensorDesc;
          tensorDesc.tensor_type = 0;
          tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
          tensorDesc.input_position = param_position[j]; //x
          tensorDesc.base_ptr_offset = 0;
          if (j == 0) {
            tensorDesc.data_type = static_cast<uint32_t>(sizeof(int64_t));
            tensorDesc.base_ptr_offset = i / per_g_e_num;
          } else if (j == 1) {
            tensorDesc.tensor_type = 1;
          } else if (j == 2) {
            tensorDesc.data_type = static_cast<uint32_t>(sizeof(int64_t));
            tensorDesc.base_ptr_offset = i / per_g_e_num;
          } else if (j == 3) {
            tensorDesc.data_type = static_cast<uint32_t>(sizeof(int32_t));
            tensorDesc.base_ptr_offset = i / per_g_e_num;
          } else {
            std::cout<<"task_split_alltoallv ERROR"<<std::endl;
          }
          inputs[j] = tensorDesc;
      }
      for (uint32_t j = 0; j < output_len; j++) {
          TensorDesc tensorDesc;
          tensorDesc.tensor_type = 1;
          tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
          tensorDesc.input_position = param_position[input_len + j]; // out
          tensorDesc.base_ptr_offset = 0;
          tensorDesc.dynamic_shape = 1;
          outputs[j] = tensorDesc;
      }
      TaskDesc taskDesc = register_put_mem_singal_task(inputs, input_len, outputs, output_len);
      taskDesc.task_index = i;
      taskDesc.task_split_num = taskSplitValue.alltoall_task_num;
      taskDesc.task_split_value = taskSplitValue.alltoall_split_value;

      int res = i / per_g_e_num;
      // taskDesc.trigger_event = taskSplitValue.pre_event_num + i + 1; // [1,16]
      // taskDesc.dependent_event = taskSplitValue.pre_pre_event_num + -1 + 1; // [0]
      taskDesc.trigger_event = taskSplitValue.pre_event_num + res + 1; // [1,16]
      runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] = event_trigger_num;
      taskDesc.dependent_event = taskSplitValue.pre_pre_event_num + -1 + 1;

      taskDesc.tiling_data_position = -1;
      taskDesc.tiling_data_offset = 0;

      runtimeConfig->all_tasks[taskSplitValue.pre_task_num + i] = taskDesc;
      runtimeConfig->vector_task_indexs[taskSplitValue.pre_vector_task_num + i] = taskSplitValue.pre_task_num + i;

      std::cout << "alltoallv1 trigger num:" << runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] << ",taskDesc.trigger_event:"<<taskDesc.trigger_event<<",taskDesc.dependent_event:"<<taskDesc.dependent_event<<std::endl;
  }
  runtimeConfig->task_index_num[1] += task_num;

  taskSplitValue.pre_task_num = taskSplitValue.pre_task_num + task_num;
  taskSplitValue.pre_vector_task_num = taskSplitValue.pre_vector_task_num + task_num;
  taskSplitValue.pre_pre_event_num = taskSplitValue.pre_event_num;
  // taskSplitValue.pre_event_num = taskSplitValue.pre_event_num + (int)(task_num / 1);
  taskSplitValue.pre_event_num = taskSplitValue.pre_event_num + taskSplitValue.all_expert_num;
}

void task_split_gmm(RuntimeConfig* runtimeConfig, vector<int64_t>& param_position, size_t dtypeSize, vector<vector<vector<int64_t>>>& input_shape, vector<vector<vector<int64_t>>>& output_shape, string layout, TaskSplitValue& taskSplitValue) {
  int task_num = taskSplitValue.gmm_task_num;
  for (int i = 0; i < task_num; i++) {
      TensorDesc inputs[MAX_INPUTS_PER_TASK];
      TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
      uint32_t input_len = input_shape.size();
      uint32_t output_len = output_shape.size();
      int data_index = i/NUM_WORKERS_CUBE;
      for (uint32_t j = 0; j < input_len; j++) {
          TensorDesc tensorDesc;
          tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
          tensorDesc.input_position = param_position[j]; //x
          tensorDesc.tensor_type = 1;
          tensorDesc.dynamic_shape = 0;
          if (j == 0) {
              tensorDesc.base_ptr_offset = data_index * 4096 * input_shape[j][0][1];
              tensorDesc.dynamic_shape = 1;
              tensorDesc.transpose_flag = 0;
          } else {
              tensorDesc.tensor_type = 1;
              tensorDesc.base_ptr_offset = 0;
              tensorDesc.transpose_flag = 1;
          }
          if (j == 2) {
            tensorDesc.transpose_flag = 0;
            tensorDesc.tensor_type = 0;
            tensorDesc.data_type = static_cast<uint32_t>(sizeof(int64_t));
          }
          // if (tensorDesc.dynamic_shape == 1) {
          //   int output_shape_dim_size = input_shape[j][0].size();
          //   int min_len = output_shape_dim_size < MAX_TENSOR_DIMS ? output_shape_dim_size:MAX_TENSOR_DIMS;
          //   for (int k = 0; k < min_len; k++) {
          //     tensorDesc.dim[k] = input_shape[j][0][k];
          //   }
          // }
          int output_shape_dim_size = input_shape[j][0].size();
          int min_len = output_shape_dim_size < MAX_TENSOR_DIMS ? output_shape_dim_size:MAX_TENSOR_DIMS;
          for (int k = 0; k < min_len; k++) {
            tensorDesc.dim[k] = input_shape[j][0][k];
          }
          inputs[j] = tensorDesc;
      }
      for (uint32_t j = 0; j < output_len; j++) {
          TensorDesc tensorDesc;
          tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
          tensorDesc.input_position = param_position[input_len + j]; // out
          tensorDesc.tensor_type = 1;
          tensorDesc.base_ptr_offset = data_index * 4096 * output_shape[j][0][1];
          tensorDesc.dynamic_shape = 1;
          // if (tensorDesc.dynamic_shape == 1) {
          //   int output_shape_dim_size = output_shape[j][0].size();
          //   int min_len = output_shape_dim_size < MAX_TENSOR_DIMS ? output_shape_dim_size:MAX_TENSOR_DIMS;
          //   for (int k = 0; k < min_len; k++) {
          //     tensorDesc.dim[k] = output_shape[j][0][k];
          //   }
          // }
          int output_shape_dim_size = output_shape[j][0].size();
          int min_len = output_shape_dim_size < MAX_TENSOR_DIMS ? output_shape_dim_size:MAX_TENSOR_DIMS;
          for (int k = 0; k < min_len; k++) {
            tensorDesc.dim[k] = output_shape[j][0][k];
          }
          outputs[j] = tensorDesc;
      }
      TaskDesc taskDesc = register_gmm_task(inputs, input_len, outputs, output_len);
      // taskDesc.inputs_transpose[0] = 0;
      // taskDesc.inputs_transpose[1] = 1;
      taskDesc.task_index = i;
      taskDesc.task_split_num = taskSplitValue.gmm_task_num;
      taskDesc.task_split_value = taskSplitValue.gmm_split_value;


      taskDesc.dependent_event = taskSplitValue.pre_pre_event_num + taskSplitValue.single_rank_expert_num * taskSplitValue.rank_id + data_index + 1; // [9, 16]
      taskDesc.trigger_event = taskSplitValue.pre_event_num + data_index + 1; // [17, 24]
      runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] = NUM_WORKERS_CUBE;
      std::cout << "gmm 1 a, i:" << i << ", taskDesc.dependent_event: " << taskDesc.dependent_event << ",taskDesc.trigger_event:"<<taskDesc.trigger_event<< ", runtimeConfig->all_event_num_triggers[taskDesc.trigger_event]:" << runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] <<std::endl;

      taskDesc.tiling_data_position = 20;
      taskDesc.tiling_data_offset = 0;

      runtimeConfig->all_tasks[taskSplitValue.pre_task_num + i] = taskDesc;
      runtimeConfig->cube_task_indexs[taskSplitValue.pre_cube_task_num + i] = taskSplitValue.pre_task_num + i;
      std::cout << "GMM1 trigger num:" << runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] << ",taskDesc.trigger_event:"<<taskDesc.trigger_event<<",taskDesc.dependent_event:"<<taskDesc.dependent_event<<std::endl;
  }
  runtimeConfig->task_index_num[0] += task_num;
  taskSplitValue.pre_task_num = taskSplitValue.pre_task_num + task_num;
  taskSplitValue.pre_cube_task_num = taskSplitValue.pre_cube_task_num + task_num;
}

void task_split_gmm_g4(RuntimeConfig* runtimeConfig, vector<int64_t>& param_position, size_t dtypeSize, vector<vector<vector<int64_t>>>& input_shape, vector<vector<vector<int64_t>>>& output_shape, string layout, TaskSplitValue& taskSplitValue) {
  int task_num = taskSplitValue.gmm_task_num_g4;
  for (int i = 0; i < task_num; i++) {
      TensorDesc inputs[MAX_INPUTS_PER_TASK];
      TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
      uint32_t input_len = input_shape.size();
      uint32_t output_len = output_shape.size();
      int data_index = i/NUM_WORKERS_CUBE;
      for (uint32_t j = 0; j < input_len; j++) {
          TensorDesc tensorDesc;
          tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
          tensorDesc.input_position = param_position[j]; //x
          tensorDesc.tensor_type = 1;
          tensorDesc.base_ptr_offset = 0;
          tensorDesc.dynamic_shape = 0;
          if (j == 0 || j == 1) {
              tensorDesc.base_ptr_offset = data_index * 4096 * input_shape[j][0][1];
              tensorDesc.dynamic_shape = 1;
              tensorDesc.transpose_flag = 0;
          }
          if (j==0) {
            tensorDesc.transpose_flag = 1;
          }
          if (j == 2) {
            tensorDesc.tensor_type = 0;
            tensorDesc.data_type = static_cast<uint32_t>(sizeof(int64_t));
          }
          // if (tensorDesc.dynamic_shape == 1) {
          //   int output_shape_dim_size = input_shape[j][0].size();
          //   int min_len = output_shape_dim_size < MAX_TENSOR_DIMS ? output_shape_dim_size:MAX_TENSOR_DIMS;
          //   for (int k = 0; k < min_len; k++) {
          //     tensorDesc.dim[k] = input_shape[j][0][k];
          //   }
          // }
          int output_shape_dim_size = input_shape[j][0].size();
          int min_len = output_shape_dim_size < MAX_TENSOR_DIMS ? output_shape_dim_size:MAX_TENSOR_DIMS;
          for (int k = 0; k < min_len; k++) {
            tensorDesc.dim[k] = input_shape[j][0][k];
          }
          inputs[j] = tensorDesc;
      }
      for (uint32_t j = 0; j < output_len; j++) {
          TensorDesc tensorDesc;
          tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
          tensorDesc.input_position = param_position[input_len + j]; // out
          tensorDesc.tensor_type = 1;
          tensorDesc.base_ptr_offset = 0;
          tensorDesc.dynamic_shape = 0;
          // tensorDesc.base_ptr_offset = data_index * 4096 * output_shape[j][0][1];
          outputs[j] = tensorDesc;
      }
      TaskDesc taskDesc = register_gmm_task(inputs, input_len, outputs, output_len);
      // taskDesc.inputs_transpose[0] = 1;
      // taskDesc.inputs_transpose[1] = 0;
      taskDesc.task_index = i;
      taskDesc.task_split_num = taskSplitValue.gmm_task_num_g4;
      taskDesc.task_split_value = taskSplitValue.gmm_split_value_g4;


      taskDesc.dependent_event = taskSplitValue.pre_pre_event_num + taskSplitValue.single_rank_expert_num * taskSplitValue.rank_id + data_index + 1; // [9, 16]
      taskDesc.trigger_event = taskSplitValue.all_event_num; // [17, 24]
      runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] = NUM_WORKERS_CUBE;
      taskDesc.tiling_data_position = 23;
      taskDesc.tiling_data_offset = 0;

      runtimeConfig->all_tasks[taskSplitValue.pre_task_num + i] = taskDesc;
      runtimeConfig->cube_task_indexs[taskSplitValue.pre_cube_task_num + i] = taskSplitValue.pre_task_num + i;

      std::cout << "GMM4 trigger num:" << runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] << ",taskDesc.trigger_event:"<<taskDesc.trigger_event<<",taskDesc.dependent_event:"<<taskDesc.dependent_event<<std::endl;
  }
  runtimeConfig->task_index_num[0] += task_num;
  taskSplitValue.pre_task_num = taskSplitValue.pre_task_num + task_num;
  taskSplitValue.pre_cube_task_num = taskSplitValue.pre_cube_task_num + task_num;

  taskSplitValue.pre_pre_event_num = taskSplitValue.pre_event_num;
  taskSplitValue.pre_event_num = taskSplitValue.pre_event_num + (int)(taskSplitValue.gmm_task_num / NUM_WORKERS_CUBE);
}

void task_split_swiglu_grad(RuntimeConfig* runtimeConfig, vector<int64_t>& param_position, size_t dtypeSize, vector<vector<int64_t>>& input_shape, vector<vector<int64_t>>& output_shape, string layout, TaskSplitValue& taskSplitValue) {
  uint32_t num_triggers = taskSplitValue.gmm_split_value / taskSplitValue.swiglu_split_value;
  int task_num = taskSplitValue.swiglu_task_num;
  int swiglu_event_num = task_num / num_triggers;
  int dynamic_shape = 1;
  if (dynamic_shape) {
    num_triggers = taskSplitValue.per_expert_seq / taskSplitValue.swiglu_split_value; // 64
    task_num = num_triggers * taskSplitValue.single_rank_expert_num; // 64 * 8
    swiglu_event_num = task_num / num_triggers; // 8
    taskSplitValue.swiglu_task_num = task_num; // 64 * 8
  }
  for (int i = 0; i < task_num; i++) {
      TensorDesc inputs[MAX_INPUTS_PER_TASK];
      TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
      uint32_t input_len = input_shape.size();
      uint32_t output_len = output_shape.size();
      for (uint32_t j = 0; j < input_len; j++) {
          TensorDesc tensorDesc;
          tensorDesc.tensor_type = 1;
          tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
          tensorDesc.input_position = static_cast<uint32_t>(param_position[j]);
          tensorDesc.base_ptr_offset = static_cast<uint32_t>(i * input_shape[j][1] * taskSplitValue.swiglu_split_value);
          tensorDesc.dynamic_shape = 1;
          if (tensorDesc.dynamic_shape == 1) {
            int output_shape_dim_size = input_shape[j].size();
            int min_len = output_shape_dim_size < MAX_TENSOR_DIMS ? output_shape_dim_size:MAX_TENSOR_DIMS;
            for (int k = 0; k < min_len; k++) {
              tensorDesc.dim[k] = input_shape[j][k];
              std::cout<<"tensor swiglu dim k:"<<  tensorDesc.dim[k]<<std::endl;
            }
          }
          inputs[j] = tensorDesc;
      }
      for (uint32_t j = 0; j < output_len; j++) {
          TensorDesc tensorDesc;
          tensorDesc.tensor_type = 1;
          tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
          tensorDesc.input_position = static_cast<uint32_t>(param_position[input_len + j]); // out
          tensorDesc.base_ptr_offset = static_cast<uint32_t>(i * output_shape[j][1] * taskSplitValue.swiglu_split_value);
          tensorDesc.dynamic_shape = 1;
          if (tensorDesc.dynamic_shape == 1) {
            int output_shape_dim_size = output_shape[j].size();
            int min_len = output_shape_dim_size < MAX_TENSOR_DIMS ? output_shape_dim_size:MAX_TENSOR_DIMS;
            for (int k = 0; k < min_len; k++) {
              tensorDesc.dim[k] = output_shape[j][k];
            }
          }
          outputs[j] = tensorDesc;
      }
      TaskDesc taskDesc = register_swi_glu_grad_task(inputs, input_len, outputs, output_len);
      taskDesc.task_index = i;
      taskDesc.task_split_num = taskSplitValue.swiglu_task_num;
      taskDesc.task_split_value = taskSplitValue.swiglu_split_value;

      // taskDesc.trigger_event = event_num + 1;
      uint32_t event_index_temp = i / num_triggers;
      // taskDesc.dependent_event = event_index_temp + 1;
      // taskDesc.trigger_event = event_num + 1 + event_index_temp;

      taskDesc.dependent_event = taskSplitValue.pre_pre_event_num + event_index_temp + 1; // [17, 24]
      taskDesc.trigger_event = taskSplitValue.pre_event_num + event_index_temp + 1; // [25, 32]
      runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] = num_triggers;

      taskDesc.tiling_data_position = 24;
      taskDesc.tiling_data_offset = 0;

      runtimeConfig->all_tasks[taskSplitValue.pre_task_num + i] = taskDesc;
      runtimeConfig->vector_task_indexs[taskSplitValue.pre_vector_task_num + i] = taskSplitValue.pre_task_num + i;
      std::cout << "swiglu trigger num:" << runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] << ",taskDesc.trigger_event:"<<taskDesc.trigger_event<<",taskDesc.dependent_event:"<<taskDesc.dependent_event<<std::endl;
  }
  runtimeConfig->task_index_num[1] += task_num;

  taskSplitValue.pre_task_num = taskSplitValue.pre_task_num + task_num;
  taskSplitValue.pre_vector_task_num = taskSplitValue.pre_vector_task_num + task_num;
  taskSplitValue.pre_pre_event_num = taskSplitValue.pre_event_num;
  taskSplitValue.pre_event_num = taskSplitValue.pre_event_num + swiglu_event_num;
}

void task_split_gmm_g2(RuntimeConfig* runtimeConfig, vector<int64_t>& param_position, size_t dtypeSize, vector<vector<vector<int64_t>>>& input_shape, vector<vector<vector<int64_t>>>& output_shape, string layout, TaskSplitValue& taskSplitValue) {
  int task_num = taskSplitValue.gmm_task_num_g2;
  for (int i = 0; i < task_num; i++) {
      TensorDesc inputs[MAX_INPUTS_PER_TASK];
      TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
      uint32_t input_len = input_shape.size();
      uint32_t output_len = output_shape.size();
      int data_index = i/NUM_WORKERS_CUBE;
      for (uint32_t j = 0; j < input_len; j++) {
          TensorDesc tensorDesc;
          tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
          tensorDesc.input_position = param_position[j]; //x
          tensorDesc.tensor_type = 1;
          tensorDesc.dynamic_shape = 0;
          if (j == 0) {
              tensorDesc.base_ptr_offset = data_index * 4096 * input_shape[j][0][1];
              tensorDesc.dynamic_shape = 1;
              tensorDesc.transpose_flag = 0;
          } else {
              tensorDesc.base_ptr_offset = 0;
              tensorDesc.transpose_flag = 1;
          }
          if (j == 2) {
            tensorDesc.transpose_flag = 0;
            tensorDesc.tensor_type = 0;
            tensorDesc.data_type = static_cast<uint32_t>(sizeof(int64_t));
          }
          // if (tensorDesc.dynamic_shape == 1) {
          //   int output_shape_dim_size = input_shape[j][0].size();
          //   int min_len = output_shape_dim_size < MAX_TENSOR_DIMS ? output_shape_dim_size:MAX_TENSOR_DIMS;
          //   for (int k = 0; k < min_len; k++) {
          //     tensorDesc.dim[k] = input_shape[j][0][k];
          //   }
          // }
          int output_shape_dim_size = input_shape[j][0].size();
          int min_len = output_shape_dim_size < MAX_TENSOR_DIMS ? output_shape_dim_size:MAX_TENSOR_DIMS;
          for (int k = 0; k < min_len; k++) {
            tensorDesc.dim[k] = input_shape[j][0][k];
          }
          inputs[j] = tensorDesc;
      }
      for (uint32_t j = 0; j < output_len; j++) {
          TensorDesc tensorDesc;
          tensorDesc.tensor_type = 1;
          tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
          tensorDesc.input_position = param_position[input_len + j]; // out
          tensorDesc.base_ptr_offset = data_index * 4096 * output_shape[j][0][1];
          tensorDesc.dynamic_shape = 1;
          // if (tensorDesc.dynamic_shape == 1) {
          //   int output_shape_dim_size = output_shape[j][0].size();
          //   int min_len = output_shape_dim_size < MAX_TENSOR_DIMS ? output_shape_dim_size:MAX_TENSOR_DIMS;
          //   for (int k = 0; k < min_len; k++) {
          //     tensorDesc.dim[k] = output_shape[j][0][k];
          //   }
          // }
          int output_shape_dim_size = output_shape[j][0].size();
          int min_len = output_shape_dim_size < MAX_TENSOR_DIMS ? output_shape_dim_size:MAX_TENSOR_DIMS;
          for (int k = 0; k < min_len; k++) {
            tensorDesc.dim[k] = output_shape[j][0][k];
          }
          outputs[j] = tensorDesc;
      }
      TaskDesc taskDesc = register_gmm_task(inputs, input_len, outputs, output_len);
      // taskDesc.inputs_transpose[0] = 0;
      // taskDesc.inputs_transpose[1] = 1;
      taskDesc.task_index = i;
      taskDesc.task_split_num = taskSplitValue.gmm_task_num_g2;
      taskDesc.task_split_value = taskSplitValue.gmm_split_value_g2;


      taskDesc.dependent_event = taskSplitValue.pre_pre_event_num + data_index + 1; // [25, 32]
      taskDesc.trigger_event =  taskSplitValue.pre_event_num + data_index + 1; // [33, 40]
      runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] = NUM_WORKERS_CUBE;

      taskDesc.tiling_data_position = 21;
      taskDesc.tiling_data_offset = 0;

      runtimeConfig->all_tasks[taskSplitValue.pre_task_num + i] = taskDesc;
      runtimeConfig->cube_task_indexs[taskSplitValue.pre_cube_task_num + i] = taskSplitValue.pre_task_num + i;
      std::cout << "GMM2 trigger num:" << runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] << ",taskDesc.trigger_event:"<<taskDesc.trigger_event<<",taskDesc.dependent_event:"<<taskDesc.dependent_event<<std::endl;
  }
  runtimeConfig->task_index_num[0] += task_num;

  taskSplitValue.pre_task_num = taskSplitValue.pre_task_num + task_num;
  taskSplitValue.pre_cube_task_num = taskSplitValue.pre_cube_task_num + task_num;
  taskSplitValue.pre_pre_event_num = taskSplitValue.pre_event_num;
  taskSplitValue.pre_event_num = taskSplitValue.pre_event_num + (int)(task_num / NUM_WORKERS_CUBE);
}

void task_split_alltoallv_a2(RuntimeConfig* runtimeConfig, vector<int64_t>& param_position, size_t dtypeSize, std::vector<std::vector<int64_t>>& input_shape, std::vector<std::vector<int64_t>>& output_shape, string layout, TaskSplitValue& taskSplitValue) {
  int task_num = taskSplitValue.alltoall_task_num_a2;
  int per_g_e_num = task_num / taskSplitValue.all_expert_num;
  // input: target_offset、 src、src_offset、size
  // output：target
  for (int i = 0; i < task_num; i++) {
      TensorDesc inputs[MAX_INPUTS_PER_TASK];
      TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
      uint32_t input_len = input_shape.size();
      uint32_t output_len = output_shape.size();
      // int current_depend_event_index = i % taskSplitValue.single_rank_expert_num;
      int current_depend_event_index = i / per_g_e_num;
      current_depend_event_index = current_depend_event_index % taskSplitValue.single_rank_expert_num;
      for (uint32_t j = 0; j < input_len; j++) {
          TensorDesc tensorDesc;
          tensorDesc.tensor_type = 0;
          tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
          tensorDesc.input_position = param_position[j]; //x
          tensorDesc.base_ptr_offset = 0;
          if (j == 0) {
            tensorDesc.data_type = static_cast<uint32_t>(sizeof(int64_t));
            tensorDesc.base_ptr_offset = i / per_g_e_num;
          } else if (j == 1) {
            tensorDesc.dynamic_shape = 1;
            tensorDesc.tensor_type = 1;
          } else if (j == 2) {
            tensorDesc.data_type = static_cast<uint32_t>(sizeof(int64_t));
            tensorDesc.base_ptr_offset = i / per_g_e_num;
          } else if (j == 3) {
            tensorDesc.data_type = static_cast<uint32_t>(sizeof(int32_t));
            tensorDesc.base_ptr_offset = i / per_g_e_num;
          } else {
            std::cout<<"task_split_alltoallv ERROR"<<std::endl;
          }
          inputs[j] = tensorDesc;
      }
      for (uint32_t j = 0; j < output_len; j++) {
          TensorDesc tensorDesc;
          tensorDesc.tensor_type = 1;
          tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
          tensorDesc.input_position = param_position[input_len + j]; // out
          tensorDesc.base_ptr_offset = 0;
          outputs[j] = tensorDesc;
      }
      TaskDesc taskDesc = register_put_mem_singal_task(inputs, input_len, outputs, output_len);
      taskDesc.task_index = i;
      taskDesc.task_split_num = taskSplitValue.alltoall_task_num_a2;
      taskDesc.task_split_value = taskSplitValue.alltoall_split_value_a2;

      taskDesc.dependent_event = taskSplitValue.pre_pre_event_num + current_depend_event_index + 1; // [33, 40]
      taskDesc.trigger_event = taskSplitValue.all_event_num; // [41]
      runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] = task_num;

      taskDesc.tiling_data_position = -1;
      taskDesc.tiling_data_offset = 0;

      runtimeConfig->all_tasks[taskSplitValue.pre_task_num + i] = taskDesc;
      runtimeConfig->vector_task_indexs[taskSplitValue.pre_vector_task_num + i] = taskSplitValue.pre_task_num + i;
      std::cout << "alltoallv2 trigger num:" << runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] << ",taskDesc.trigger_event:"<<taskDesc.trigger_event<<",taskDesc.dependent_event:"<<taskDesc.dependent_event<<std::endl;
  }
  runtimeConfig->task_index_num[1] += task_num;

  taskSplitValue.pre_task_num = taskSplitValue.pre_task_num + task_num;
  taskSplitValue.pre_vector_task_num = taskSplitValue.pre_vector_task_num + task_num;
}

void task_split_gmm_g3(RuntimeConfig* runtimeConfig, vector<int64_t>& param_position, size_t dtypeSize, vector<vector<vector<int64_t>>>& input_shape, vector<vector<vector<int64_t>>>& output_shape, string layout, TaskSplitValue& taskSplitValue) {
  int task_num = taskSplitValue.gmm_task_num_g3;
  for (int i = 0; i < task_num; i++) {
      TensorDesc inputs[MAX_INPUTS_PER_TASK];
      TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
      uint32_t input_len = input_shape.size();
      uint32_t output_len = output_shape.size();
      int data_index = i/NUM_WORKERS_CUBE;
      for (uint32_t j = 0; j < input_len; j++) {
          TensorDesc tensorDesc;
          tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
          tensorDesc.input_position = param_position[j]; //x
          tensorDesc.tensor_type = 1;
          tensorDesc.base_ptr_offset = 0;
          tensorDesc.dynamic_shape = 0;
          if (j == 0 || j == 1) {
              tensorDesc.base_ptr_offset = data_index * 4096 * input_shape[j][0][1];
              tensorDesc.dynamic_shape = 1;
              tensorDesc.transpose_flag = 0;
          }
          if (j == 0) {
            tensorDesc.transpose_flag = 1;
          }
          if (j == 2) {
            tensorDesc.transpose_flag = 0;
            tensorDesc.tensor_type = 0;
            tensorDesc.data_type = static_cast<uint32_t>(sizeof(int64_t));
          }
          // if (tensorDesc.dynamic_shape == 1) {
          //   int output_shape_dim_size = input_shape[j][0].size();
          //   int min_len = output_shape_dim_size < MAX_TENSOR_DIMS ? output_shape_dim_size:MAX_TENSOR_DIMS;
          //   for (int k = 0; k < min_len; k++) {
          //     tensorDesc.dim[k] = input_shape[j][0][k];
          //   }
          // }
          int output_shape_dim_size = input_shape[j][0].size();
          int min_len = output_shape_dim_size < MAX_TENSOR_DIMS ? output_shape_dim_size:MAX_TENSOR_DIMS;
          for (int k = 0; k < min_len; k++) {
            tensorDesc.dim[k] = input_shape[j][0][k];
          }
          inputs[j] = tensorDesc;
      }
      for (uint32_t j = 0; j < output_len; j++) {
          TensorDesc tensorDesc;
          tensorDesc.data_type = static_cast<uint32_t>(dtypeSize);
          tensorDesc.input_position = param_position[input_len + j]; // out
          tensorDesc.tensor_type = 1;
          tensorDesc.base_ptr_offset = 0;
          tensorDesc.dynamic_shape = 0;
          // tensorDesc.base_ptr_offset = data_index * 4096 * output_shape[j][0][1];
          outputs[j] = tensorDesc;
      }
      TaskDesc taskDesc = register_gmm_task(inputs, input_len, outputs, output_len);
      // taskDesc.inputs_transpose[0] = 1;
      // taskDesc.inputs_transpose[1] = 0;
      taskDesc.task_index = i;
      taskDesc.task_split_num = taskSplitValue.gmm_task_num_g3;
      taskDesc.task_split_value = taskSplitValue.gmm_split_value_g3;


      taskDesc.dependent_event = taskSplitValue.pre_pre_event_num + data_index + 1; // [25, 32]
      taskDesc.trigger_event = taskSplitValue.all_event_num; // [33, 40]
      runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] = NUM_WORKERS_CUBE;

      taskDesc.tiling_data_position = 22;
      taskDesc.tiling_data_offset = 0;

      runtimeConfig->all_tasks[taskSplitValue.pre_task_num + i] = taskDesc;
      runtimeConfig->cube_task_indexs[taskSplitValue.pre_cube_task_num + i] = taskSplitValue.pre_task_num + i;
      std::cout << "GMM3 trigger num:" << runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] << ",taskDesc.trigger_event:"<<taskDesc.trigger_event<<",taskDesc.dependent_event:"<<taskDesc.dependent_event<<std::endl;
  }
  runtimeConfig->task_index_num[0] += task_num;

  taskSplitValue.pre_task_num = taskSplitValue.pre_task_num + task_num;
  taskSplitValue.pre_cube_task_num = taskSplitValue.pre_cube_task_num + task_num;
  taskSplitValue.pre_pre_event_num = taskSplitValue.pre_event_num;
  taskSplitValue.pre_event_num = taskSplitValue.pre_event_num + 1;
}

void add_terminate_task(RuntimeConfig* runtimeConfig, TaskSplitValue& taskSplitValue) {
  runtimeConfig->all_event_num_triggers[taskSplitValue.all_event_num] = taskSplitValue.gmm_task_num_g4 + taskSplitValue.gmm_task_num_g3 + taskSplitValue.alltoall_task_num_a2/taskSplitValue.ep * taskSplitValue.ep;
  std::cout << "add_terminate_task taskSplitValue.all_event_num:" << taskSplitValue.all_event_num << ",runtimeConfig->all_event_num_triggers[taskSplitValue.all_event_num]:"<<runtimeConfig->all_event_num_triggers[taskSplitValue.all_event_num]<<std::endl;

  TaskDesc taskDesc;
  taskDesc.task_type = TaskType::TASK_TERMINATE;
  taskDesc.dependent_event = taskSplitValue.pre_pre_event_num + 1;
  taskDesc.trigger_event = taskSplitValue.pre_event_num + 1;
  runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] = 1;

  std::cout << "last task trigger num:" << runtimeConfig->all_event_num_triggers[taskDesc.trigger_event] << ",taskDesc.trigger_event:"<<taskDesc.trigger_event<<",taskDesc.dependent_event:"<<taskDesc.dependent_event<<std::endl;

  runtimeConfig->all_tasks[taskSplitValue.pre_task_num] = taskDesc;
  runtimeConfig->vector_task_indexs[taskSplitValue.pre_vector_task_num] = taskSplitValue.pre_task_num;
  runtimeConfig->task_index_num[1] += 1;

  std::cout << "last task taskSplitValue.pre_task_num:" << taskSplitValue.pre_task_num << ",taskSplitValue.pre_vector_task_num:"<<taskSplitValue.pre_vector_task_num<<std::endl;
}

// void revise_task_queue(RuntimeConfig* runtimeConfig, TaskSplitValue& taskSplitValue) {
//   int32_t vector_task_indexs_temp[TASK_TYPE_INDEX_NUM] = {0};
//   int32_t expert_single = taskSplitValue.single_rank_expert_num;
//   int32_t ep = taskSplitValue.ep;
//   for (int i =0;i<TASK_TYPE_INDEX_NUM;i++) {
//     vector_task_indexs_temp[i] = runtimeConfig->vector_task_indexs[i];
//   }
//   int start = 0;
//   int end = start + taskSplitValue.alltoall_task_num;
//   for (int i =start;i<end;i++) {
//     int value = (i % ep) * expert_single + (int)(i/ep) + (int)(start/ep);
//     runtimeConfig->vector_task_indexs[i] = vector_task_indexs_temp[value];
//     std::cout<<"revise_task_queue 0, start+i:"<<i<< ",runtimeConfig->vector_task_indexs[start + i]:"<<runtimeConfig->vector_task_indexs[i]<<std::endl;
//   }
//   start = taskSplitValue.alltoall_task_num + taskSplitValue.swiglu_task_num;
//   end = start + taskSplitValue.alltoall_task_num_a2;
//   for (int i =start;i<end;i++) {
//     int value = (i % ep) * expert_single + (int)(i/ep) + (int)(start/ep);
//     runtimeConfig->vector_task_indexs[i] = vector_task_indexs_temp[value];
//     std::cout<<"revise_task_queue 1, start+i:"<<i<< ",runtimeConfig->vector_task_indexs[start + i]:"<<runtimeConfig->vector_task_indexs[i]<<std::endl;
//   }
// }

void print_data(RuntimeConfig* runtimeConfig, TaskSplitValue& taskSplitValue) {
  for (int i =0;i<2500;i++) {
    TaskDesc taskDesc = runtimeConfig->all_tasks[i];
    std::cout<<"print_data 0, task_id:"<<i<< ",trigger_event:"<<taskDesc.trigger_event<<",dependent_event:"<<taskDesc.dependent_event<<std::endl;
  }

  for (int i =0;i<2000;i++) {
    int32_t data = runtimeConfig->cube_task_indexs[i];
    TaskDesc taskDesc = runtimeConfig->all_tasks[data];
    std::cout<<"cube print_data 0, task_id:"<<data<< ",trigger_event:"<<taskDesc.trigger_event<<",dependent_event:"<<taskDesc.dependent_event<<std::endl;
  }

  for (int i =0;i<2000;i++) {
    int32_t data = runtimeConfig->vector_task_indexs[i];
    TaskDesc taskDesc = runtimeConfig->all_tasks[data];
    std::cout<<"vector print_data 0, task_id:"<<data<< ",trigger_event:"<<taskDesc.trigger_event<<",dependent_event:"<<taskDesc.dependent_event<<std::endl;
  }
}

// void revise_task_queue(RuntimeConfig* runtimeConfig, TaskSplitValue& taskSplitValue) {
//   int32_t vector_task_indexs_temp[TASK_TYPE_INDEX_NUM] = {0};
//   int32_t expert_single = taskSplitValue.single_rank_expert_num;
//   int32_t ep = taskSplitValue.ep;
//   for (int i =0;i<TASK_TYPE_INDEX_NUM;i++) {
//     vector_task_indexs_temp[i] = runtimeConfig->vector_task_indexs[i];
//   }
//   int start = 0;
//   int end = start + taskSplitValue.alltoall_task_num;
//   int per_g_e_num = taskSplitValue.alltoall_task_num / (expert_single * ep);
//   for (int i =0;i<taskSplitValue.alltoall_task_num;i++) {
//     int x = i / per_g_e_num;
//     int y = i % per_g_e_num;
//     int m = x % ep;
//     int n = x / ep;
//     int z = taskSplitValue.alltoall_task_num / ep;
//     int value = y + m * z + start + n * per_g_e_num;
//     runtimeConfig->vector_task_indexs[i] = vector_task_indexs_temp[value];
//     std::cout<<"revise_task_queue 0, start+i:"<<i<< ",runtimeConfig->vector_task_indexs[start + i]:"<<runtimeConfig->vector_task_indexs[i]<<std::endl;
//   }
//   start = taskSplitValue.alltoall_task_num + taskSplitValue.swiglu_task_num;
//   end = start + taskSplitValue.alltoall_task_num_a2;
//   per_g_e_num = taskSplitValue.alltoall_task_num_a2 / (expert_single * ep);
//   for (int i =0;i<taskSplitValue.alltoall_task_num_a2;i++) {
//     int x = i / per_g_e_num;
//     int y = i % per_g_e_num;
//     int m = x % ep;
//     int n = x / ep;
//     int z = taskSplitValue.alltoall_task_num_a2 / ep;
//     int value = y + m * z + start + n * per_g_e_num;
//     runtimeConfig->vector_task_indexs[start + i] = vector_task_indexs_temp[value];

//     std::cout<<"revise_task_queue 1, start+i:"<<start+i<< ",runtimeConfig->vector_task_indexs[start + i]:"<<runtimeConfig->vector_task_indexs[start + i]<<std::endl;
//   }

//   // print_data(runtimeConfig, taskSplitValue);
// }

void revise_task_queue(RuntimeConfig* runtimeConfig, TaskSplitValue& taskSplitValue) {
  int32_t vector_task_indexs_temp[TASK_TYPE_INDEX_NUM] = {0};
  for (int i =0;i<TASK_TYPE_INDEX_NUM;i++) {
    vector_task_indexs_temp[i] = runtimeConfig->vector_task_indexs[i];
  }
  int32_t single_rank_expert_num = taskSplitValue.single_rank_expert_num;
  int32_t single_expert_task_num = taskSplitValue.alltoall_task_num / taskSplitValue.all_expert_num;
  int32_t ep = taskSplitValue.ep;
  int32_t rank_id = taskSplitValue.rank_id;
  int32_t single_rank_task_num = taskSplitValue.alltoall_task_num / taskSplitValue.ep;
  int32_t start = 0;
  int32_t end = start + taskSplitValue.alltoall_task_num;
  std::vector<int32_t> ep_rank;
  for (int32_t i =0;i<ep;i++) {
    int32_t res = (i+rank_id)%ep;
    ep_rank.push_back(res);
  }
  int32_t index = 0;
  for (int32_t j =0; j< single_rank_expert_num;j++) {
    int32_t j_v = j * single_expert_task_num;
    for (int32_t k =0; k< single_expert_task_num;k++) {
      int32_t k_v = j_v + k;
      for (int32_t i : ep_rank) {
        int32_t i_v = k_v + i * single_rank_task_num;
        runtimeConfig->vector_task_indexs[start + index] = vector_task_indexs_temp[start + i_v];
        index+=1;
      }
    }
  }

  index = 0;
  start = taskSplitValue.alltoall_task_num + taskSplitValue.swiglu_task_num;
  end = start + taskSplitValue.alltoall_task_num_a2;
  for (int32_t j =0; j< single_rank_expert_num;j++) {
    int32_t j_v = j * single_expert_task_num;
    for (int32_t k =0; k< single_expert_task_num;k++) {
      int32_t k_v = j_v + k;
      for (int32_t i : ep_rank) {
        int32_t i_v = k_v + i * single_rank_task_num;
        runtimeConfig->vector_task_indexs[start + index] = vector_task_indexs_temp[start + i_v];
        index+=1;
      }
    }
  }
}

void revise_gmm_task_queue(RuntimeConfig* runtimeConfig, TaskSplitValue& taskSplitValue) {
  int32_t cube_task_indexs_temp[TASK_TYPE_INDEX_NUM] = {0};
  int32_t expert_single = taskSplitValue.single_rank_expert_num;
  int32_t ep = taskSplitValue.ep;
  for (int i =0;i<TASK_TYPE_INDEX_NUM;i++) {
    cube_task_indexs_temp[i] = runtimeConfig->cube_task_indexs[i];
  }
  int start = 0;
  int left = start;
  int right = start + taskSplitValue.gmm_task_num;
  int changes_num = 2;
  for (int i = 0; i < expert_single * changes_num; i++) {
    int index = 1 - i % changes_num;
    int m = i / changes_num;
    for (int j = 0; j < NUM_WORKERS_CUBE; j++) {
      int dst = i * NUM_WORKERS_CUBE + j;
      int src = index * taskSplitValue.gmm_task_num + m * NUM_WORKERS_CUBE + j;
      std::cout<<"revise_gmm_task_queue 1, src:"<<src<< ",src value :"<<cube_task_indexs_temp[src]<<",dst:"<<dst<<", dst value:"<<runtimeConfig->cube_task_indexs[dst]<<std::endl;
      runtimeConfig->cube_task_indexs[dst] = cube_task_indexs_temp[src];
    }
  }
}



void add_dynamic_data(RuntimeConfig* runtimeConfig, TaskSplitValue& taskSplitValue) {
  runtimeConfig->dynamic_data.dynamic_type = DynamicType::DYNAMIC_DSV3_MOE_FFN;
  runtimeConfig->dynamic_data.dynamic_input_position = 19;
  runtimeConfig->dynamic_data.dynamic_group_size = taskSplitValue.single_rank_expert_num;
  runtimeConfig->dynamic_data.dynamic_max_seq_len = -1;
}

#endif // MULTICORE_MOE_FFN_GRAD_TASK_REGISTER_TEST_CODE
