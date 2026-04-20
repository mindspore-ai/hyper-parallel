/**
 * @file worker_kernel.cpp
 * @brief Worker核函数 - 完整版本
 */

#include "kernel_operator.h"
#include "runtime_config.hpp"

#include "swi_glu/swi_glu.cpp"
#include "swi_glu_grad/swi_glu_grad.cpp"
#include "grouped_matmul/grouped_matmul.cpp"
#include "put_mem_signal/put_mem_signal_kernel.cpp"

using namespace AscendC;

class KernelWorker {
public:
    __aicore__ inline KernelWorker() {}

    __aicore__ inline void Init(uint32_t worker_id, __gm__ uint8_t *runtimeConfigPtr, GM_ADDR *input_list) {
        this->worker_id_ = worker_id;
        this->runtimeConfigPtr = runtimeConfigPtr;

        this->tpipe_.InitBuffer(tBuf, DISPATCH_TOKEN_UB_SIZE);
        tpipe_.Destroy();

        all_event_counters.SetGlobalBuffer((__gm__ int32_t *)(input_list[31]), MAX_EVENT_NUM);

        all_event_num_triggers.SetGlobalBuffer((__gm__ int32_t *)(this->runtimeConfigPtr + getAllEventNumTriggersOffset()), MAX_EVENT_NUM);

        vector_task_indexs.SetGlobalBuffer((__gm__ int32_t *)(this->runtimeConfigPtr + getVectorTaskIndexsOffset()), TASK_TYPE_INDEX_NUM);

        cube_task_indexs.SetGlobalBuffer((__gm__ int32_t *)(this->runtimeConfigPtr + getCubeTaskIndexsOffset()), TASK_TYPE_INDEX_NUM);

        atomic_add_values.SetGlobalBuffer((__gm__ int32_t *)(this->runtimeConfigPtr + getAtomicAddValuesOffset()), 8);

        this->input_list = input_list;

        this->task_num = getTaskNum(this->runtimeConfigPtr);

        this->vector_task_num = getTaskIndexNumByTaskType(this->runtimeConfigPtr, TaskAiCoreType::TASK_AICORE_VECTOR);
        this->cube_task_num = getTaskIndexNumByTaskType(this->runtimeConfigPtr, TaskAiCoreType::TASK_AICORE_CUBE);

        this->core_num = getExtraValueFromTiling(input_list[30], 5);
        this->vector_num = this->core_num * 2;
    }

    __aicore__ inline void Process() {
#ifdef __DAV_C220_CUBE__
        uint32_t block_idx = this->worker_id_;
        do {
            if (block_idx >= this->cube_task_num) {
                return;
            }
            TaskId task_index = GetTaskIndex(block_idx);
            ExecuteTask(task_index);
            block_idx = block_idx + this->core_num; // 1;
        } while (1);
#else
        if (this->worker_id_ % 2 == 0) {
            return;
        }
        uint32_t block_idx = this->worker_id_ / 2;
        do {
            if (block_idx >= this->vector_task_num) {
                return;
            }
            TaskId task_index = GetTaskIndex(block_idx);
            ExecuteTask(task_index);
            uint32_t half_num = this->vector_num / 2;
            block_idx = block_idx + half_num; // this->vector_num; // 1;
        } while (1);
#endif
    }

private:
    __aicore__ inline TaskId GetTaskIndex(uint32_t task_id) {
#ifdef __DAV_C220_CUBE__
        return cube_task_indexs.GetValue(task_id);
#else
        return vector_task_indexs.GetValue(task_id);
#endif
    }

    __aicore__ inline void initLocalSet()
    {
#ifdef __DAV_C220_CUBE__
        this->localSet = tBuf.GetWithOffset<int32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy(this->localSet, this->atomic_add_values, EXP_TOKEN_COUNT_FLAG_CNT);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
#else
        this->localSet = tBuf.GetWithOffset<int32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
        this->localSet.SetValue(0, 1);
        for (int32_t i = 1; i < EXP_TOKEN_COUNT_FLAG_CNT; ++i) {
            this->localSet.SetValue(i, 0);
        }
#endif
    }

    __aicore__ inline void AtomicAddForAllEventCounters(uint32_t event_index)
    {
#ifdef __DAV_C220_CUBE__
        this->localSet = tBuf.GetWithOffset<int32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy(this->localSet, this->atomic_add_values, EXP_TOKEN_COUNT_FLAG_CNT);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        AscendC::SetAtomicAdd<int32_t>();
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy(this->all_event_counters[event_index], this->localSet, EXP_TOKEN_COUNT_FLAG_CNT);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        AscendC::SetAtomicNone();
        tBuf.FreeTensor(this->localSet);
#else
        this->localSet = tBuf.GetWithOffset<int32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
        this->localSet.SetValue(0, 1);
        for (int32_t i = 1; i < EXP_TOKEN_COUNT_FLAG_CNT; ++i) {
            this->localSet.SetValue(i, 0);
        }
        AscendC::SetAtomicAdd<int32_t>();
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopy(this->all_event_counters[event_index], this->localSet, EXP_TOKEN_COUNT_FLAG_CNT);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        AscendC::SetAtomicNone();
        tBuf.FreeTensor(this->localSet);
#endif
    }

    __aicore__ inline void ExecuteTask(TaskId task_id) {
        TaskDesc task_desc;
        getTaskDesc(this->runtimeConfigPtr, &(task_desc), task_id);
        if (task_desc.dependent_event != EVENT_INVALID_ID) {
            WaitForDependency(task_desc.dependent_event);
        }
        ExecuteComputeKernel(task_desc);
        if (task_desc.task_type != TASK_SHMEM_PUT_MEM_SINGAL) {
            TriggerEvent(task_desc.trigger_event);
        }
    }

    __aicore__ inline void WaitForDependency(uint32_t event_index) {
#ifdef __DAV_C220_CUBE__
        int32_t needed = all_event_num_triggers.GetValue(event_index);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(all_event_counters[event_index]);
        int32_t current = all_event_counters.GetValue(event_index);
        int32_t retry_count = 0;
        PipeBarrier<PIPE_ALL>();
        int64_t systemCycleBefore = AscendC::GetSystemCycle();
        do {
            if (current >= needed) {
                break;
            }
            int64_t systemCycleAfter = AscendC::GetSystemCycle();
            int64_t GetBlockNumCycle = systemCycleAfter - systemCycleBefore;
            int64_t CycleToTimeBase = 50;
            int64_t GetBlockNumTime = GetBlockNumCycle/CycleToTimeBase;
            if (GetBlockNumTime > 50) { //50
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(all_event_counters[event_index]);
                current = all_event_counters.GetValue(event_index);
                systemCycleBefore = AscendC::GetSystemCycle();
            }
        } while (1);

#else
        int32_t needed = all_event_num_triggers.GetValue(event_index);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(all_event_counters[event_index]);
        int32_t current = all_event_counters.GetValue(event_index);
        int32_t retry_count = 0;
        PipeBarrier<PIPE_ALL>();

        int64_t systemCycleBefore = AscendC::GetSystemCycle();
        do {
            if (current >= needed) {
                break;
            }
            int64_t systemCycleAfter = AscendC::GetSystemCycle();
            int64_t GetBlockNumCycle = systemCycleAfter - systemCycleBefore;
            int64_t CycleToTimeBase = 50;
            int64_t GetBlockNumTime = GetBlockNumCycle/CycleToTimeBase;
            if (GetBlockNumTime > 150) { //150
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(all_event_counters[event_index]);
                current = all_event_counters.GetValue(event_index);
                systemCycleBefore = AscendC::GetSystemCycle();
            }
        } while (1);
#endif
    }

    __aicore__ inline void dcci_cacheline(GlobalTensor<int32_t>& global_tensor, uint32_t value) {
        __asm__ __volatile__("");
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global_tensor[value]);
        __asm__ __volatile__("");
    }


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

            case TASK_SHMEM_PUT_MEM_SINGAL:
                ExecuteShmemPutMem(task_desc);
                break;

            case TASK_SWI_GLU_GRAD:
                ExecuteSwiGluGrad(task_desc);
                break;

            default:
                break;
        }
    }

    __aicore__ inline void ExecuteSwiGluGrad(TaskDesc task_desc) {
        if (task_desc.inputs[0].dynamic_shape == 1 || task_desc.outputs[0].dynamic_shape == 1) {
            DynamicData dynamic_data;
            getDynamicData(this->runtimeConfigPtr, &(dynamic_data));
            if (dynamic_data.dynamic_type == DynamicType::DYNAMIC_DSV3_MOE_FFN) {
                GM_ADDR grouped_list = input_list[dynamic_data.dynamic_input_position];
                GlobalTensor<int64_t> grouped_list_tensor;
                int64_t ep = getExtraValueFromTiling(input_list[30], 1);
                int64_t expert_num = getExtraValueFromTiling(input_list[30], 2);
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

                if (base_ptr_offset >=end) {
                    return;
                }
                int64_t input_0_offset = task_desc.inputs[0].dynamic_shape == 1?
                                        base_ptr_offset * task_desc.inputs[0].dim[1] * task_desc.inputs[0].data_type :
                                        task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type;
                int64_t input_1_offset = task_desc.inputs[1].dynamic_shape == 1?
                                        base_ptr_offset * task_desc.inputs[1].dim[1] * task_desc.inputs[1].data_type :
                                        task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type;
                int64_t output_0_offset = task_desc.outputs[0].dynamic_shape == 1?
                                        base_ptr_offset * task_desc.outputs[0].dim[1] * task_desc.outputs[0].data_type :
                                        task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type;
                if (start + current_seq_end <= end) {
                    swi_glu_grad(
                        input_list[task_desc.inputs[0].input_position] + input_0_offset,
                        input_list[task_desc.inputs[1].input_position] + input_1_offset,
                        input_list[task_desc.outputs[0].input_position] + output_0_offset,
                        input_list[26], // workspace,
                        input_list[task_desc.tiling_data_position] + task_desc.tiling_data_offset
                    );
                    return;
                } else {
                    GM_ADDR tiling_data_addr = input_list[task_desc.tiling_data_position] + 80 * (AscendC::GetBlockIdx() + 1);
                    __gm__ SwiGluTilingData* tilingdata_data = reinterpret_cast<__gm__ SwiGluTilingData*>(tiling_data_addr);
                    tilingdata_data->rowLen = end - (start + current_seq_start);
                    if (tilingdata_data->rowLen == 0) {
                        return;
                    }
                    if (end - (start + current_seq_start) < 19) {
                        tilingdata_data->baseRowLen = end - (start + current_seq_start);
                    }
                    cacheWriteThrough(tiling_data_addr, 10);
                    PipeBarrier<PIPE_ALL>();

                    swi_glu_grad(
                        input_list[task_desc.inputs[0].input_position] + input_0_offset,
                        input_list[task_desc.inputs[1].input_position] + input_1_offset,
                        input_list[task_desc.outputs[0].input_position] + output_0_offset,
                        input_list[26], // workspace,
                        tiling_data_addr
                    );
                }
            }
            return;
        }
    }

    __aicore__ inline void ExecuteMatmul(TaskDesc task_desc) {

    }

    __aicore__ inline void cacheWriteThrough(__gm__ uint8_t* sourceAddr, int64_t length) {
        __gm__ uint8_t* start =
            (__gm__ uint8_t*)((int64_t)sourceAddr / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
        __gm__ uint8_t* end =
            (__gm__ uint8_t*)(((int64_t)sourceAddr + length) / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
        AscendC::GlobalTensor<uint8_t> global;
        global.SetGlobalBuffer(start);
        for (uint32_t i = 0; i <= end - start; i += AscendC::CACHE_LINE_SIZE) {
            __asm__ __volatile__ ("");
            AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(global[i]);
            __asm__ __volatile__ ("");
        }
    }

    __aicore__ inline void ExecuteGroupedMatmul(TaskDesc task_desc) {
        if (task_desc.inputs[0].dynamic_shape == 1 || task_desc.inputs[1].dynamic_shape == 1) {
            DynamicData dynamic_data;
            getDynamicData(this->runtimeConfigPtr, &(dynamic_data));
            if (dynamic_data.dynamic_type == DynamicType::DYNAMIC_DSV3_MOE_FFN) {
                GM_ADDR grouped_list = input_list[task_desc.inputs[2].input_position] + task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type;
                GlobalTensor<int64_t> grouped_list_tensor;
                int64_t ep = getExtraValueFromTiling(input_list[30], 1);
                int64_t expert_num = getExtraValueFromTiling(input_list[30], 2);
                int64_t expert_num_single_rank = expert_num / ep;
                grouped_list_tensor.SetGlobalBuffer((__gm__ int64_t *)(grouped_list), expert_num_single_rank);

                GM_ADDR grouped_list_real = this->runtimeConfigPtr + getGroupedMatmulGroupListOffsetById(this->runtimeConfigPtr, AscendC::GetBlockIdx() * 2);
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
                for (uint32_t i = 0; i < expert_num_single_rank; i ++) {
                    if (i > data_index) {
                        grouped_list_tensor_real.SetValue(i, value);
                    } else if (i==data_index) {
                        grouped_list_tensor_real.SetValue(i, value);
                    } else {
                        grouped_list_tensor_real.SetValue(i, 0);
                    }
                }
                cacheWriteThrough(grouped_list_real, expert_num_single_rank);

                GM_ADDR tiling_data_addr = input_list[task_desc.tiling_data_position] + 2016 * AscendC::GetBlockIdx();
                __gm__ GMMTilingData* tilingdata_data = reinterpret_cast<__gm__ GMMTilingData*>(tiling_data_addr);
                if (getTransposeData(task_desc.inputs[0].transpose_flag)) {
                    tilingdata_data->mmTilingData.Ka = value;
                    tilingdata_data->mmTilingData.Kb = value;
                    tilingdata_data->mmTilingData.singleCoreK = value;
                } else {
                    tilingdata_data->gmmBaseParams.m = value;
                    tilingdata_data->mmTilingData.M = value;
                    tilingdata_data->mmTilingData.singleCoreM = value;
                }
                cacheWriteThrough(tiling_data_addr, 220);
                PipeBarrier<PIPE_ALL>();

                int64_t input_0_offset = task_desc.inputs[0].dynamic_shape == 1?
                                        start * task_desc.inputs[0].dim[1] * task_desc.inputs[0].data_type:
                                        task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type;
                int64_t input_1_offset = task_desc.inputs[1].dynamic_shape == 1?
                                        start * task_desc.inputs[1].dim[1] * task_desc.inputs[1].data_type :
                                        task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type;
                int64_t output_0_offset = task_desc.outputs[0].dynamic_shape == 1?
                                        start * task_desc.outputs[0].dim[1] * task_desc.outputs[0].data_type :
                                        task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type;

                grouped_matmul(
                    input_list[task_desc.inputs[0].input_position] + input_0_offset, // x
                    input_list[task_desc.inputs[1].input_position] + input_1_offset, // weight,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    grouped_list_real,
                    nullptr,
                    input_list[task_desc.outputs[0].input_position] + output_0_offset, // x
                    input_list[25], // workspace,
                    tiling_data_addr,
                    getTransposeData(task_desc.inputs[0].transpose_flag),
                    getTransposeData(task_desc.inputs[1].transpose_flag)
                );

            }
            return;
        }
    }

    __aicore__ inline bool getTransposeData(uint32_t data) {
        return data == 1;
    }

    __aicore__ inline void ExecuteShmemPutMem(TaskDesc task_desc) {
        GM_ADDR signal = input_list[31];
        int64_t ep = getExtraValueFromTiling(input_list[30], 1);
        int64_t expert_num = getExtraValueFromTiling(input_list[30], 2);
        int64_t expert_num_single_rank = expert_num / ep;
        int64_t hidden_size = getExtraValueFromTiling(input_list[30], 3);

        int64_t rank_id = (*(__gm__ int64_t *)(input_list[30])) / ep * ep;
        int64_t single_expert_task_num = task_desc.task_split_num / expert_num;
        int64_t target_pe = rank_id + task_desc.task_index / (expert_num_single_rank * single_expert_task_num);

        int64_t target_offset_ = *((__gm__ int64_t*)(input_list[task_desc.inputs[0].input_position] + task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type));
        int64_t src_offset_ = *((__gm__ int64_t*)(input_list[task_desc.inputs[2].input_position] + task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type));
        int32_t size_ = *((__gm__ int32_t*)(input_list[task_desc.inputs[3].input_position] + task_desc.inputs[3].base_ptr_offset * task_desc.inputs[3].data_type));

        int32_t send_data_size_ = task_desc.task_split_value * hidden_size;

        int32_t value = task_desc.task_index % single_expert_task_num; // / single_expert_task_num;
        int32_t start = value * task_desc.task_split_value * hidden_size;
        int32_t end = (value+1) * task_desc.task_split_value * hidden_size;
        if (start >= size_) {
            send_data_size_ = 0;
        } else {
            if (end > size_) {
                send_data_size_ = size_ - start;
            }
        }
        target_offset_ = target_offset_ + static_cast<int64_t>(start);
        src_offset_ = src_offset_ + static_cast<int64_t>(start);
        send_data_size_ = send_data_size_;

        put_mem_signal_kernel(
            input_list[task_desc.outputs[0].input_position] + task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type,
            target_offset_,
            input_list[task_desc.inputs[1].input_position] + task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type, // x
            src_offset_,
            static_cast<int64_t>(send_data_size_), //size
            signal, //signal
            static_cast<int64_t>(task_desc.trigger_event), //signal_offset
            1, //signal_value
            nullptr, //workspace
            1, //signal_op
            target_pe, //target_pe
            false);
    }

    __aicore__ inline void TriggerEvent(uint32_t event_index) {
        AtomicAddForAllEventCounters(event_index);
    }

private:
    uint32_t worker_id_;
    uint32_t task_num;

    int32_t vector_task_num;
    int32_t cube_task_num;

    TPipe tpipe_;
#ifdef __DAV_C220_CUBE__
    TBuf<AscendC::TPosition::A1> tBuf;
#else
    TBuf<AscendC::TPosition::VECOUT> tBuf;
#endif
    LocalTensor<int32_t> localSet;

    GlobalTensor<int32_t> all_event_counters;
    GlobalTensor<int32_t> all_event_num_triggers;
    GlobalTensor<int32_t> atomic_add_values;

    GlobalTensor<int32_t> vector_task_indexs;
    GlobalTensor<int32_t> cube_task_indexs;

    __gm__ uint8_t *runtimeConfigPtr;
    GM_ADDR *input_list;
    int64_t core_num;
    int64_t vector_num;
};

extern "C" inline __aicore__ void worker_kernel(
    uint32_t worker_id, __gm__ uint8_t *runtimeConfigPtr, GM_ADDR *input_list) {

    KernelWorker worker;
    worker.Init(worker_id, runtimeConfigPtr, input_list);
    worker.Process();
}
