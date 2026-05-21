/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file worker_kernel.cpp
 * @brief MoE-FFN forward — op-specific compute kernels + KernelWorker specialization.
 */

#include "kernel_operator.h"
#include "swi_glu/swi_glu.cpp"
#include "grouped_matmul/grouped_matmul.cpp"
#include "put_mem_signal/put_mem_signal_kernel.cpp"
#include "runtime/worker_kernel.h"

using namespace AscendC;

class KernelWorker : public KernelWorkerBase<KernelWorker> {
 public:
  // input_list layout for mega_moe (forward):
  //   [23] = tiling params  [24] = all_event_counters  [11] = gmm_workspace
  static constexpr uint32_t TILING_IDX   = 23;
  static constexpr uint32_t EVENT_IDX    = 24;
  static constexpr uint32_t WORKSPACE_IDX = 11;

  __aicore__ inline void ExecuteComputeKernel(TaskDesc task_desc) {
    switch (task_desc.task_type) {
      case TASK_BEGIN_TASK_GRAPH:
        break;
      case TASK_MATMUL:
        ExecuteMatmul(task_desc);
        break;
      case TASK_GROUPED_MATMUL:
        ExecuteGroupedMatmul(task_desc);
        break;
      case TASK_SWI_GLU:
        ExecuteSwiglu(task_desc);
        break;
      case TASK_SHMEM_PUT_MEM_SINGAL:
        ExecuteShmemPutMem(task_desc);
        break;
      default:
        break;
    }
  }

 private:
  __aicore__ inline void ExecuteMatmul(const TaskDesc &task_desc) {}

  __aicore__ inline void cacheWriteThrough(__gm__ uint8_t *sourceAddr, int64_t length) {
    __gm__ uint8_t *start =
      (__gm__ uint8_t *)((int64_t)sourceAddr / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    __gm__ uint8_t *end =
      (__gm__ uint8_t *)(((int64_t)sourceAddr + length) / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint32_t i = 0; i <= end - start; i += AscendC::CACHE_LINE_SIZE) {
      AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                        AscendC::DcciDst::CACHELINE_OUT>(global[i]);
    }
  }

  __aicore__ inline void ExecuteSwiglu(const TaskDesc &task_desc) {
    if (task_desc.inputs[0].dynamic_shape == 1 || task_desc.outputs[0].dynamic_shape == 1) {
      DynamicData dynamic_data;
      getDynamicData(this->runtimeConfigPtr, &(dynamic_data));
      if (dynamic_data.dynamic_type == DynamicType::DYNAMIC_DSV3_MOE) {
        GM_ADDR grouped_list = input_list[dynamic_data.dynamic_input_position];
        GlobalTensor<int64_t> grouped_list_tensor;
        int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
        int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
        int64_t expert_num_single_rank = expert_num / ep;
        grouped_list_tensor.SetGlobalBuffer((__gm__ int64_t *)(grouped_list), expert_num_single_rank);
        uint32_t grouped_list_shape = dynamic_data.dynamic_group_size;
        uint32_t per_task_num = task_desc.task_split_num / grouped_list_shape;
        uint32_t data_index = task_desc.task_index / per_task_num;
        uint32_t task_index = task_desc.task_index % per_task_num;
        int64_t start = 0;
        int64_t end = grouped_list_tensor.GetValue(data_index);
        if (data_index != 0) {
          start = grouped_list_tensor.GetValue(data_index - 1);
        }
        int64_t current_seq_start = task_index * task_desc.task_split_value;
        int64_t current_seq_end = (task_index + 1) * task_desc.task_split_value;
        int64_t base_ptr_offset = start + current_seq_start;
        if (base_ptr_offset >= end) {
          return;
        }
        int64_t input_0_offset = task_desc.inputs[0].dynamic_shape == 1
                                   ? base_ptr_offset * task_desc.inputs[0].dim[1] * task_desc.inputs[0].data_type
                                   : task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type;
        int64_t output_0_offset = task_desc.outputs[0].dynamic_shape == 1
                                    ? base_ptr_offset * task_desc.outputs[0].dim[1] * task_desc.outputs[0].data_type
                                    : task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type;
        if (start + current_seq_end <= end) {
          swi_glu(input_list[task_desc.inputs[0].input_position] + input_0_offset,
                  input_list[task_desc.outputs[0].input_position] + output_0_offset, nullptr,
                  input_list[task_desc.tiling_data_position] + task_desc.tiling_data_offset);
        } else {
          GM_ADDR tiling_data_addr = input_list[task_desc.tiling_data_position] + 80 * (AscendC::GetBlockIdx() + 1);
          __gm__ SwiGluTilingData *tilingdata_data = reinterpret_cast<__gm__ SwiGluTilingData *>(tiling_data_addr);
          tilingdata_data->rowLen = end - (start + current_seq_start);
          if (tilingdata_data->rowLen == 0) {
            return;
          }
          if (end - (start + current_seq_start) < 19) {
            tilingdata_data->baseRowLen = end - (start + current_seq_start);
          }
          cacheWriteThrough(tiling_data_addr, 10);
          PipeBarrier<PIPE_ALL>();
          swi_glu(input_list[task_desc.inputs[0].input_position] + input_0_offset,
                  input_list[task_desc.outputs[0].input_position] + output_0_offset, nullptr, tiling_data_addr);
        }
      }
      return;
    }
    swi_glu(input_list[task_desc.inputs[0].input_position] +
              task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type,
            input_list[task_desc.outputs[0].input_position] +
              task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type,
            nullptr, input_list[task_desc.tiling_data_position] + task_desc.tiling_data_offset);
  }

  __aicore__ inline void ExecuteGroupedMatmul(TaskDesc task_desc) {
    if (task_desc.inputs[0].dynamic_shape == 1 || task_desc.inputs[1].dynamic_shape == 1) {
      DynamicData dynamic_data;
      getDynamicData(this->runtimeConfigPtr, &(dynamic_data));
      if (dynamic_data.dynamic_type == DynamicType::DYNAMIC_DSV3_MOE) {
        GM_ADDR grouped_list = input_list[task_desc.inputs[2].input_position] +
                               task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type;
        GlobalTensor<int64_t> grouped_list_tensor;
        int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
        int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
        int64_t expert_num_single_rank = expert_num / ep;
        grouped_list_tensor.SetGlobalBuffer((__gm__ int64_t *)(grouped_list), expert_num_single_rank);

        GM_ADDR grouped_list_real = this->runtimeConfigPtr + getGroupedMatmulGroupListOffsetById(
                                                               this->runtimeConfigPtr, AscendC::GetBlockIdx() * 2);
        GlobalTensor<int64_t> grouped_list_tensor_real;
        grouped_list_tensor_real.SetGlobalBuffer((__gm__ int64_t *)(grouped_list_real), expert_num_single_rank);
        uint32_t data_index = task_desc.task_index / this->core_num;
        int64_t value = grouped_list_tensor.GetValue(data_index);
        int64_t start = 0;
        if (data_index != 0) {
          value = value - grouped_list_tensor.GetValue(data_index - 1);
          start = grouped_list_tensor.GetValue(data_index - 1);
        }
        if (value == 0) {
          return;
        }
        for (uint32_t i = 0; i < expert_num_single_rank; i++) {
          if (i > data_index) {
            grouped_list_tensor_real.SetValue(i, value);
          } else if (i == data_index) {
            grouped_list_tensor_real.SetValue(i, value);
          } else {
            grouped_list_tensor_real.SetValue(i, 0);
          }
        }
        cacheWriteThrough(grouped_list_real, expert_num_single_rank);

        GM_ADDR tiling_data_addr = input_list[task_desc.tiling_data_position] + 2016 * AscendC::GetBlockIdx();
        __gm__ GMMTilingData *tilingdata_data = reinterpret_cast<__gm__ GMMTilingData *>(tiling_data_addr);
        tilingdata_data->gmmBaseParams.m = value;
        tilingdata_data->mmTilingData.M = value;
        tilingdata_data->mmTilingData.singleCoreM = value;
        cacheWriteThrough(tiling_data_addr, 250);
        PipeBarrier<PIPE_ALL>();

        int64_t input_0_offset = task_desc.inputs[0].dynamic_shape == 1
                                   ? start * task_desc.inputs[0].dim[1] * task_desc.inputs[0].data_type
                                   : task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type;
        int64_t output_0_offset = task_desc.outputs[0].dynamic_shape == 1
                                    ? start * task_desc.outputs[0].dim[1] * task_desc.outputs[0].data_type
                                    : task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type;
        grouped_matmul(input_list[task_desc.inputs[0].input_position] + input_0_offset,
                       input_list[task_desc.inputs[1].input_position],
                       nullptr, nullptr, nullptr, nullptr, nullptr, grouped_list_real, nullptr,
                       input_list[task_desc.outputs[0].input_position] + output_0_offset,
                       input_list[WORKSPACE_IDX],
                       tiling_data_addr, false, false);
      }
      return;
    }
  }

  __aicore__ inline void ExecuteShmemPutMem(TaskDesc task_desc) {
    GM_ADDR signal = input_list[EVENT_IDX];

    int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
    int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
    int64_t expert_num_single_rank = expert_num / ep;
    int64_t hidden_size = getExtraValueFromTiling(input_list[TILING_IDX], 3);

    int64_t rank_id = (*(__gm__ int64_t *)(input_list[TILING_IDX])) / ep * ep;
    int64_t single_expert_task_num = task_desc.task_split_num / expert_num;
    int64_t target_pe = rank_id + task_desc.task_index / (expert_num_single_rank * single_expert_task_num);

    int64_t target_offset_ = *((__gm__ int64_t *)(input_list[task_desc.inputs[0].input_position] +
                                                  task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type));
    int64_t src_offset_ = *((__gm__ int64_t *)(input_list[task_desc.inputs[2].input_position] +
                                               task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type));
    int32_t size_ = *((__gm__ int32_t *)(input_list[task_desc.inputs[3].input_position] +
                                         task_desc.inputs[3].base_ptr_offset * task_desc.inputs[3].data_type));
    int32_t send_data_size_ = task_desc.task_split_value * hidden_size;

    int32_t value = task_desc.task_index % single_expert_task_num;
    int32_t start = value * task_desc.task_split_value * hidden_size;
    int32_t end = (value + 1) * task_desc.task_split_value * hidden_size;
    if (start >= size_) {
      send_data_size_ = 0;
    } else {
      if (end > size_) {
        send_data_size_ = size_ - start;
      }
    }
    target_offset_ = target_offset_ + static_cast<int64_t>(start);
    src_offset_ = src_offset_ + static_cast<int64_t>(start);
    send_data_size_ = static_cast<int64_t>(send_data_size_);

    put_mem_signal_kernel(input_list[task_desc.outputs[0].input_position] +
                            task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type,
                          target_offset_,
                          input_list[task_desc.inputs[1].input_position] +
                            task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type,
                          src_offset_,
                          send_data_size_,
                          signal,
                          static_cast<int64_t>(task_desc.trigger_event),
                          1,
                          nullptr,
                          1,
                          target_pe,
                          false);
  }
};

extern "C" inline __aicore__ void worker_kernel(uint32_t worker_id, __gm__ uint8_t *runtimeConfigPtr,
                                                GM_ADDR *input_list) {
  KernelWorker worker;
  worker.Init(worker_id, runtimeConfigPtr, input_list);
  worker.Process();
}
