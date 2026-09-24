/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

/*! \file worker_kernel.cpp */
#include "kernel_operator.h"
#include "hyper_mega_mhc_grad_task_tiling.h"
#include "mhc_post_backward/mhc_post_backward_arch22.h"
#include "mhc_pre_sinkhorn_backward/arch22/mhc_pre_grad_kernel.h"
#define ZERO HP_RMS_NORM_GRAD_ZERO
#include "rms_norm_grad/rms_norm_grad_split_n_high_precision.h"
#undef ZERO
#include "runtime/worker_kernel.h"

using namespace AscendC;
using namespace MulticoreRuntime;

namespace {
constexpr int64_t WORKSPACE_ALIGNMENT = 512;
constexpr uint32_t RMS_BF16_BUFFER_SIZE = 5760;
constexpr uint32_t RMS_BF16_DATA_TYPE = 2;
constexpr int64_t ZERO_COPY_ELEMENTS = 256;

__aicore__ inline int64_t AlignWorkspace(int64_t value) {
  return (value + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
}

__aicore__ inline int64_t MinValue(int64_t lhs, int64_t rhs) { return lhs < rhs ? lhs : rhs; }

__aicore__ inline void SynchronizeMte3ToMte2() {
  event_t eventMte3Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
  SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2);
  WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2);
}

struct TokenTile {
  int64_t offset;
  int32_t count;
};

__aicore__ inline TokenTile ResolveTokenTile(const TaskDesc &taskDesc,
                                             const MhcPreSinkhornBackwardArch22TilingData *tiling) {
  TokenTile tile;
  tile.offset = static_cast<int64_t>(taskDesc.task_index) * taskDesc.task_split_value;
  tile.count =
    static_cast<int32_t>(MinValue(taskDesc.task_split_value, tiling->batchSize * tiling->seqLength - tile.offset));
  return tile;
}

struct WorkspaceAddresses {
  GM_ADDR gradHin;
  GM_ADDR gradXFromPre;
  GM_ADDR gradRsqrt;
};

__aicore__ inline WorkspaceAddresses ResolveWorkspace(GM_ADDR workspace,
                                                      const MhcPreSinkhornBackwardArch22TilingData *tiling) {
  int64_t tokens = tiling->batchSize * tiling->seqLength;
  int64_t hcMix = tiling->n * tiling->n + 2 * tiling->n;
  int64_t offset = tokens * hcMix * sizeof(float);
  offset += 2 * tokens * tiling->n * tiling->c * sizeof(float);
  offset = AlignWorkspace(offset);
  WorkspaceAddresses addresses;
  addresses.gradHin = workspace + offset;
  offset += AlignWorkspace(tokens * tiling->c * sizeof(DTYPE_X));
  addresses.gradXFromPre = workspace + offset;
  offset += AlignWorkspace(tokens * tiling->n * tiling->c * sizeof(DTYPE_X));
  addresses.gradRsqrt = workspace + offset;
  return addresses;
}

__aicore__ inline RmsNormGradTilingData MakeRmsGradTaskTiling(const MhcPreSinkhornBackwardArch22TilingData *tiling,
                                                              uint32_t taskCount) {
  uint32_t ubFactor = RMS_BF16_BUFFER_SIZE / tiling->c;
  ubFactor = ubFactor == 0 ? 1 : ubFactor;

  RmsNormGradTilingData result = {};
  result.row = taskCount;
  result.col = tiling->c;
  result.avg_factor = 1.0f / static_cast<float>(tiling->c);
  result.data_type = RMS_BF16_DATA_TYPE;
  result.block_factor = taskCount;
  result.ub_split_dim = 0;
  result.ub_factor = ubFactor;
  result.core_calc_num = taskCount;
  result.core_calc_tail = 0;
  result.block_dim = 1;
  result.ub_calc_num = ubFactor;
  result.ub_calc_tail = taskCount % ubFactor;
  result.ub_calc_loop = (taskCount + ubFactor - 1) / ubFactor;
  result.ub_calc_tail_num = ubFactor;
  result.ub_calc_tail_tail = result.ub_calc_tail;
  result.ub_calc_tail_loop = result.ub_calc_loop;
  result.fixed_output = 0;
  result.chunk_size = tiling->c;
  result.chunk_num = 1;
  result.chunk_tail = 0;
  result.need_chunk = 0;
  return result;
}

__aicore__ inline MhcPostBackwardTilingDataArch22 MakePostGradTiling(
  const MhcPreSinkhornBackwardArch22TilingData *tiling) {
  uint64_t tokens = tiling->batchSize * tiling->seqLength;
  uint64_t blockChannel = tiling->c < 1024 ? tiling->c : 1024;

  MhcPostBackwardTilingDataArch22 result = {};
  result.singleCoreBS = tokens;
  result.tailBS = tokens;
  result.coreUsed = 1;
  result.frontCore = 1;
  result.tailCore = 0;
  result.dFPostResSize = tokens * tiling->n * tiling->c;
  result.xSize = result.dFPostResSize;
  result.hResSize = tokens * tiling->n * tiling->n;
  result.hOutSize = tokens * tiling->c;
  result.hPostSize = tokens * tiling->n;
  result.channel = tiling->c;
  result.blockChannel = blockChannel;
  result.n = tiling->n;
  result.alignN = (tiling->n * sizeof(float) + 31) / 32 * 32 / sizeof(float);
  result.loopC = tiling->c / blockChannel;
  result.tailC = tiling->c % blockChannel;
  return result;
}
}  // namespace

class KernelWorker
    : public KernelWorkerBase<KernelWorker, VectorWorkerPolicy::ALL, RuntimeStorageBoundsPolicy::GRAPH_OWNED> {
 public:
  static constexpr uint32_t TILING_IDX = 34;
  static constexpr uint32_t EVENT_IDX = 22;
  static constexpr uint32_t PROFILE_IDX = 23;
  static constexpr uint32_t WORKSPACE_IDX = 33;

  __aicore__ inline void ExecuteComputeKernel(TaskDesc taskDesc) {
    switch (taskDesc.task_type) {
      case TaskType::TASK_MHC_GRAD_PREV_A_AND_MAPPING:
        ExecuteMhcGradPrevAAndMapping(taskDesc);
        break;
      case TaskType::TASK_RMS_NORM_GRAD:
        ExecuteRmsNormGrad(taskDesc);
        break;
      case TaskType::TASK_MHC_GRAD_PREV_A:
      case TaskType::TASK_MHC_GRAD_PHI_RMS:
        ExecuteMhcPreGrad(taskDesc);
        break;
      case TaskType::TASK_MHC_GRAD_PREV_X_AND_POST:
        ExecuteMhcGradPrevXAndPost(taskDesc);
        break;
      default:
        break;
    }
  }

 private:
  __aicore__ inline void InitializeAlignedReductionOutput(GM_ADDR address, int64_t elementCount,
                                                          const TaskDesc &taskDesc, LocalTensor<float> &zeros) {
    constexpr int64_t elementsPerBlock = 32 / sizeof(float);
    int64_t blockCount = elementCount / elementsPerBlock;
    int64_t blocksPerTask = (blockCount + taskDesc.task_split_num - 1) / taskDesc.task_split_num;
    int64_t blockOffset = static_cast<int64_t>(taskDesc.task_index) * blocksPerTask;
    int64_t localBlockCount = MinValue(blocksPerTask, blockCount - blockOffset);
    if (localBlockCount <= 0) {
      return;
    }
    GlobalTensor<float> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(address), elementCount);
    int64_t elementOffset = blockOffset * elementsPerBlock;
    int64_t remaining = localBlockCount * elementsPerBlock;
    while (remaining > 0) {
      int64_t count = MinValue(remaining, ZERO_COPY_ELEMENTS);
      DataCopy(output[elementOffset], zeros, count);
      elementOffset += count;
      remaining -= count;
    }
  }

  __aicore__ inline void InitializeReductionOutputs(const MhcPreSinkhornBackwardArch22TilingData *tiling,
                                                    const TaskDesc &taskDesc) {
    int64_t hcMix = tiling->n * tiling->n + 2 * tiling->n;
    if (taskDesc.task_index == 0) {
      GlobalTensor<float> gradAlpha;
      gradAlpha.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input_list[26]), 3);
      gradAlpha.SetValue(0, 0.0f);
      gradAlpha.SetValue(1, 0.0f);
      gradAlpha.SetValue(2, 0.0f);
      DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(gradAlpha);
      PipeBarrier<PIPE_ALL>();
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> zeroBuffer;
    pipe.InitBuffer(zeroBuffer, ZERO_COPY_ELEMENTS * sizeof(float));
    LocalTensor<float> zeros = zeroBuffer.Get<float>();
    Duplicate(zeros, 0.0f, ZERO_COPY_ELEMENTS);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    InitializeAlignedReductionOutput(input_list[25], hcMix * tiling->n * tiling->c, taskDesc, zeros);
    InitializeAlignedReductionOutput(input_list[27], hcMix, taskDesc, zeros);
    InitializeAlignedReductionOutput(input_list[32], tiling->c, taskDesc, zeros);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    pipe.Destroy();
  }

  __aicore__ inline void ExecuteRmsNormGrad(const TaskDesc &taskDesc) {
    if ASCEND_IS_AIV {
      GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornBackwardArch22TilingData, tilingValue, input_list[TILING_IDX]);
      const MhcPreSinkhornBackwardArch22TilingData *tiling = &tilingValue;
      TokenTile tile = ResolveTokenTile(taskDesc, tiling);
      if (tile.count <= 0) {
        return;
      }
      WorkspaceAddresses addresses = ResolveWorkspace(input_list[WORKSPACE_IDX], tiling);
      RmsNormGradTilingData rmsTiling = MakeRmsGradTaskTiling(tiling, tile.count);
      int64_t activationOffset = tile.offset * tiling->c * sizeof(DTYPE_X);
      int64_t rstdOffset = tile.offset * sizeof(float);
      RmsNormGradSplitNHighPrecision<DTYPE_X, DTYPE_NORM_WEIGHT> rmsGrad;
      rmsGrad.InitTask(input_list[0] + activationOffset, input_list[13] + activationOffset, input_list[14] + rstdOffset,
                       input_list[15], addresses.gradHin + activationOffset, input_list[32], &rmsTiling,
                       input_list[WORKSPACE_IDX]);
      rmsGrad.Process();
      rmsGrad.DestroyTaskPipe();
    }
  }

  __aicore__ inline void ExecuteMhcGradPrevAAndMapping(const TaskDesc &taskDesc) {
    if ASCEND_IS_AIV {
      // PrevAAndMapping restores the native MhcPre backward order for one token tile: ComputeGradPre
      // generates dA_(l-1), the shifted handoff replaces its UB slot with upstream dA_l, then mapping
      // backward runs.
      GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornBackwardArch22TilingData, tilingValue, input_list[TILING_IDX]);
      const MhcPreSinkhornBackwardArch22TilingData *tiling = &tilingValue;
      TokenTile tile = ResolveTokenTile(taskDesc, tiling);
      if (tile.count <= 0) {
        return;
      }
      WorkspaceAddresses addresses = ResolveWorkspace(input_list[WORKSPACE_IDX], tiling);
      TPipe pipe;
      MhcPreGradKernel<DTYPE_X, DTYPE_PHI, false> preGrad;
      InitMhcPreGradTask(preGrad, pipe, tiling, addresses);
      preGrad.ProcessGradPrepareTask(tile.offset, tile.count);
      pipe.Destroy();
    }
  }

  __aicore__ inline void InitMhcPreGradTask(MhcPreGradKernel<DTYPE_X, DTYPE_PHI, false> &preGrad, TPipe &pipe,
                                            const MhcPreSinkhornBackwardArch22TilingData *tiling,
                                            const WorkspaceAddresses &addresses) {
    preGrad.InitShiftedTask(input_list[3], input_list[4], input_list[7], input_list[12], input_list[29],
                            addresses.gradHin, input_list[1], input_list[2], input_list[5], input_list[6],
                            input_list[8], input_list[9], input_list[10], input_list[11], addresses.gradXFromPre,
                            input_list[25], input_list[26], input_list[27], addresses.gradRsqrt,
                            input_list[WORKSPACE_IDX], tiling, &pipe);
  }

  __aicore__ inline void ExecuteMhcPreGrad(const TaskDesc &taskDesc) {
    GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornBackwardArch22TilingData, tilingValue, input_list[TILING_IDX]);
    const MhcPreSinkhornBackwardArch22TilingData *tiling = &tilingValue;
    if (taskDesc.task_type == TaskType::TASK_MHC_GRAD_PREV_A) {
      if ASCEND_IS_AIV {
        InitializeReductionOutputs(tiling, taskDesc);
      }
      return;
    }
    TokenTile tile = ResolveTokenTile(taskDesc, tiling);
    if (tile.count <= 0) {
      return;
    }
    if ASCEND_IS_AIC {
      if (taskDesc.task_type != TaskType::TASK_MHC_GRAD_PHI_RMS) {
        return;
      }
    }
    if ASCEND_IS_AIV {
      if (taskDesc.task_type == TaskType::TASK_MHC_GRAD_PHI_RMS) {
        return;
      }
    }
    WorkspaceAddresses addresses = ResolveWorkspace(input_list[WORKSPACE_IDX], tiling);
    TPipe pipe;
    MhcPreGradKernel<DTYPE_X, DTYPE_PHI, false> preGrad;
    if ASCEND_IS_AIC {
      preGrad.mm1_.SetSubBlockIdx(0);
      preGrad.mm1_.Init(&tiling->mm1TilingData, &pipe);
      preGrad.mm2_.SetSubBlockIdx(0);
      preGrad.mm2_.Init(&tiling->mm2TilingData, &pipe);
    }
    InitMhcPreGradTask(preGrad, pipe, tiling, addresses);
    switch (taskDesc.task_type) {
      case TaskType::TASK_MHC_GRAD_PHI_RMS:
        preGrad.ProcessPhiRmsTask(tile.offset, tile.count);
        break;
      default:
        break;
    }
    pipe.Destroy();
  }

  __aicore__ inline void ExecuteMhcGradPrevXAndPost(const TaskDesc &taskDesc) {
    if ASCEND_IS_AIV {
      GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornBackwardArch22TilingData, tilingValue, input_list[TILING_IDX]);
      const MhcPreSinkhornBackwardArch22TilingData *tiling = &tilingValue;
      TokenTile tile = ResolveTokenTile(taskDesc, tiling);
      if (tile.count <= 0) {
        return;
      }
      WorkspaceAddresses addresses = ResolveWorkspace(input_list[WORKSPACE_IDX], tiling);
      TPipe pipe;

      MhcPreGradKernel<DTYPE_X, DTYPE_PHI, false> preGrad;
      InitMhcPreGradTask(preGrad, pipe, tiling, addresses);
      preGrad.ProcessGradPreviousXTask(tile.offset, tile.count);

      SynchronizeMte3ToMte2();
      pipe.Reset();

      MhcPostBackwardTilingDataArch22 postTiling = MakePostGradTiling(tiling);
      KernelMhcPostBackward<DTYPE_X> postGrad;
      postGrad.InitAddTask(addresses.gradXFromPre, input_list[16], input_list[17], input_list[20], input_list[18],
                           input_list[19], input_list[24], input_list[31], input_list[28], input_list[30], postTiling,
                           &pipe, tile.offset, tile.count);
      postGrad.Process();
      pipe.Destroy();
    }
  }
};

extern "C" inline __aicore__ void worker_kernel(uint32_t workerId, __gm__ uint8_t *runtimeConfig, GM_ADDR *inputList) {
  KernelWorker worker;
  worker.Init(workerId, runtimeConfig, inputList);
  worker.Process();
}
