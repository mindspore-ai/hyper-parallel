/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

/*! \file worker_kernel.cpp */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hyper_mega_mhc_task_tiling.h"
#include "mhc_post/mhc_post_arch22.h"
#include "mhc_pre_sinkhorn/mhc_pre_sinkhorn_m_split_core.h"
#include "rms_norm/rms_norm.h"
#include "runtime/worker_kernel.h"

using namespace AscendC;
using namespace MhcPreSinkhorn;
using namespace MulticoreRuntime;

namespace {
constexpr int64_t WORKSPACE_ALIGNMENT = 512;
constexpr int64_t POST_HIDDEN_TILE = 2048;
constexpr int64_t PROJECTION_ROWS_PER_CALL = 64;
constexpr int64_t RMS_NORM_NATIVE_ROW_FACTOR = 64;
constexpr int64_t RMS_NORM_NATIVE_BF16_UB_FACTOR = 12288;

__aicore__ inline int64_t AlignWorkspace(int64_t value) {
  return (value + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
}

__aicore__ inline int64_t CeilDivide(int64_t value, int64_t divisor) { return (value + divisor - 1) / divisor; }

__aicore__ inline int64_t MinValue(int64_t lhs, int64_t rhs) { return lhs < rhs ? lhs : rhs; }

struct TokenTile {
  int64_t offset;
  int64_t count;
};

__aicore__ inline TokenTile ResolveTokenTile(const TaskDesc &taskDesc, const MhcPreSinkhornTilingData *tiling) {
  TokenTile tile;
  tile.offset = static_cast<int64_t>(taskDesc.task_index) * taskDesc.task_split_value;
  tile.count = MinValue(taskDesc.task_split_value, static_cast<int64_t>(tiling->bs) - tile.offset);
  return tile;
}

__aicore__ inline int64_t ResolveXCastSlot(const TaskDesc &taskDesc, const MhcPreSinkhornTilingData *tiling) {
  int64_t rowBytes = tiling->hcMult * tiling->d * sizeof(float);
  int64_t slotBytes = taskDesc.task_split_value * rowBytes;
  int64_t ringSlots = tiling->stage1XCastWsSize / slotBytes;
  return taskDesc.task_index % ringSlots;
}

struct WorkspaceAddresses {
  GM_ADDR hcBeforeNorm;
  GM_ADDR invRms;
  GM_ADDR sumOut;
  GM_ADDR normOut;
  GM_ADDR mixedInput;
  GM_ADDR rmsRstd;
};

__aicore__ inline WorkspaceAddresses ResolveWorkspace(GM_ADDR workspace, const MhcPreSinkhornTilingData *tiling) {
  int64_t offset = tiling->stage1XCastWsSize;
  WorkspaceAddresses addresses;
  addresses.hcBeforeNorm = workspace + offset;
  offset += AlignWorkspace(tiling->bs * tiling->hcMix * sizeof(float));
  addresses.invRms = workspace + offset;
  offset += AlignWorkspace(tiling->bs * sizeof(float));
  addresses.sumOut = workspace + offset;
  offset += AlignWorkspace(2 * tiling->iterTimes * tiling->bs * tiling->hcMult * sizeof(float));
  addresses.normOut = workspace + offset;
  offset += AlignWorkspace(2 * tiling->iterTimes * tiling->bs * tiling->hcMult * tiling->hcMult * sizeof(float));
  addresses.mixedInput = workspace + offset;
  offset += AlignWorkspace(tiling->bs * tiling->d * sizeof(DTYPE_RESIDUAL));
  addresses.rmsRstd = workspace + offset;
  return addresses;
}
}  // namespace

class KernelWorker
    : public KernelWorkerBase<KernelWorker, VectorWorkerPolicy::ALL, RuntimeStorageBoundsPolicy::GRAPH_OWNED> {
 public:
  // Custom input_list layout assembled by hyper_mega_mhc.cpp.
  static constexpr uint32_t TILING_IDX = 18;
  static constexpr uint32_t EVENT_IDX = 10;
  static constexpr uint32_t PROFILE_IDX = 11;
  static constexpr uint32_t WORKSPACE_IDX = 17;

  __aicore__ inline void ExecuteComputeKernel(TaskDesc taskDesc) {
    switch (taskDesc.task_type) {
      case TaskType::TASK_MHC_POST:
        ExecuteMhcPost(taskDesc);
        break;
      case TaskType::TASK_MHC_NORM_CAST:
        ExecuteMhcNormCast(taskDesc);
        break;
      case TaskType::TASK_MHC_PROJECTION:
        ExecuteMhcProjection(taskDesc);
        break;
      case TaskType::TASK_MHC_INPUT_MIX:
        ExecuteMhcInputMix(taskDesc);
        break;
      case TaskType::TASK_MHC_MAPPING:
        ExecuteMhcMapping(taskDesc);
        break;
      case TaskType::TASK_RMS_NORM:
        ExecuteRmsNorm(taskDesc);
        break;
      default:
        break;
    }
  }

 private:
  __aicore__ inline void ExecuteMhcPost(const TaskDesc &taskDesc) {
    if ASCEND_IS_AIV {
      GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornTilingData, tilingValue, input_list[TILING_IDX]);
      const MhcPreSinkhornTilingData *tiling = &tilingValue;
      TokenTile tile = ResolveTokenTile(taskDesc, tiling);
      if (tile.count <= 0) {
        return;
      }
      int64_t hiddenLoops = CeilDivide(tiling->d, POST_HIDDEN_TILE);

      MhcPostTilingData postTiling = {};
      postTiling.n = tiling->hcMult;
      postTiling.d = tiling->d;
      postTiling.usedCoreNum = taskDesc.task_split_num;
      postTiling.normalCoreProcessNum = taskDesc.task_split_value * hiddenLoops;
      postTiling.tailCoreProcessNum = tile.count * hiddenLoops;
      postTiling.bsInner = taskDesc.task_split_value;
      postTiling.bsOuter = taskDesc.task_split_num;
      postTiling.bsTail = tiling->bs - (taskDesc.task_split_num - 1) * taskDesc.task_split_value;
      postTiling.dInner = POST_HIDDEN_TILE;
      postTiling.dOuter = hiddenLoops;
      postTiling.dTail = tiling->d - (hiddenLoops - 1) * POST_HIDDEN_TILE;
      postTiling.dTailAlign = (postTiling.dTail + 15) / 16 * 16;

      TPipe pipe;
      MhcPost::MhcPostKernel<DTYPE_RESIDUAL, 1> post(&pipe, &postTiling);
      post.InitTask(input_list[1], input_list[4], input_list[0], input_list[3], input_list[12], nullptr,
                    taskDesc.task_index);
      post.Process();
      pipe.Destroy();
    }
  }

  __aicore__ inline void ExecuteMhcNormCast(const TaskDesc &taskDesc) {
    if ASCEND_IS_AIV {
      GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornTilingData, tilingValue, input_list[TILING_IDX]);
      const MhcPreSinkhornTilingData *tiling = &tilingValue;
      TokenTile tile = ResolveTokenTile(taskDesc, tiling);
      if (tile.count <= 0) {
        return;
      }
      WorkspaceAddresses addresses = ResolveWorkspace(input_list[WORKSPACE_IDX], tiling);
      int64_t rowBytes = tiling->hcMult * tiling->d * sizeof(float);
      int64_t slot = ResolveXCastSlot(taskDesc, tiling);
      GM_ADDR slotWorkspace = input_list[WORKSPACE_IDX] + slot * taskDesc.task_split_value * rowBytes;

      TPipe pipe;
      MhcPreSinkhornStage1<DTYPE_RESIDUAL> stage1;
      stage1.Init(input_list[12], input_list[5], addresses.invRms, addresses.hcBeforeNorm, input_list[WORKSPACE_IDX],
                  tiling, &pipe);
      stage1.SetNormCastTaskContext(slotWorkspace, tile.offset, tile.count);
      stage1.Process();
      pipe.Destroy();
    }
  }

  __aicore__ inline void ExecuteMhcProjection(const TaskDesc &taskDesc) {
    if ASCEND_IS_AIC {
      GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornTilingData, tilingValue, input_list[TILING_IDX]);
      const MhcPreSinkhornTilingData *tiling = &tilingValue;
      TokenTile tile = ResolveTokenTile(taskDesc, tiling);
      if (tile.count <= 0) {
        return;
      }
      WorkspaceAddresses addresses = ResolveWorkspace(input_list[WORKSPACE_IDX], tiling);
      int64_t slot = ResolveXCastSlot(taskDesc, tiling);
      int64_t slotRowOffset = slot * taskDesc.task_split_value;

      TPipe pipe;
      MhcPreSinkhornStage1<DTYPE_RESIDUAL> stage1;
      stage1.cubeCompute_.mm1_.Init(&tiling->mm1TilingData, &pipe);
      stage1.cubeCompute_.mm1_.SetSubBlockIdx(0);
      stage1.Init(input_list[12], input_list[5], addresses.invRms, addresses.hcBeforeNorm, input_list[WORKSPACE_IDX],
                  tiling, &pipe);
      for (int64_t row = 0; row < tile.count; row += PROJECTION_ROWS_PER_CALL) {
        int64_t rowCount = MinValue(PROJECTION_ROWS_PER_CALL, tile.count - row);
        stage1.cubeCompute_.ProcessMatmulXPhiTask(slotRowOffset + row, tile.offset + row, rowCount);
      }
      pipe.Destroy();
    }
  }

  __aicore__ inline void ExecuteMhcInputMix(const TaskDesc &taskDesc) {
    if ASCEND_IS_AIV {
      GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornTilingData, tilingValue, input_list[TILING_IDX]);
      const MhcPreSinkhornTilingData *tiling = &tilingValue;
      TokenTile tile = ResolveTokenTile(taskDesc, tiling);
      if (tile.count <= 0) {
        return;
      }
      WorkspaceAddresses addresses = ResolveWorkspace(input_list[WORKSPACE_IDX], tiling);
      TPipe pipe;
      MhcPreSinkhornStage2<DTYPE_RESIDUAL> stage2;
      stage2.Init(input_list[12], input_list[6], input_list[7], addresses.mixedInput, input_list[14], input_list[15],
                  input_list[13], addresses.hcBeforeNorm, addresses.invRms, addresses.sumOut, addresses.normOut,
                  input_list[WORKSPACE_IDX], tiling, &pipe);
      stage2.SetTaskContext(input_list[2], taskDesc.task_index, taskDesc.task_split_num, taskDesc.task_split_value);
      stage2.ProcessYTask();
      pipe.Destroy();
    }
  }

  __aicore__ inline void ExecuteMhcMapping(const TaskDesc &taskDesc) {
    if ASCEND_IS_AIV {
      GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornTilingData, tilingValue, input_list[TILING_IDX]);
      const MhcPreSinkhornTilingData *tiling = &tilingValue;
      TokenTile tile = ResolveTokenTile(taskDesc, tiling);
      if (tile.count <= 0) {
        return;
      }
      WorkspaceAddresses addresses = ResolveWorkspace(input_list[WORKSPACE_IDX], tiling);
      TPipe pipe;
      MhcPreSinkhornStage2<DTYPE_RESIDUAL> stage2;
      stage2.Init(input_list[12], input_list[6], input_list[7], addresses.mixedInput, input_list[14], input_list[15],
                  input_list[13], addresses.hcBeforeNorm, addresses.invRms, addresses.sumOut, addresses.normOut,
                  input_list[WORKSPACE_IDX], tiling, &pipe);
      stage2.SetTaskContext(input_list[2], taskDesc.task_index, taskDesc.task_split_num, taskDesc.task_split_value);
      stage2.Process(false, false);
      pipe.Destroy();
    }
  }

  __aicore__ inline void ExecuteRmsNorm(const TaskDesc &taskDesc) {
    if ASCEND_IS_AIV {
      GET_TILING_DATA_WITH_STRUCT(MhcPreSinkhornTilingData, tilingValue, input_list[TILING_IDX]);
      const MhcPreSinkhornTilingData *tiling = &tilingValue;
      TokenTile tile = ResolveTokenTile(taskDesc, tiling);
      if (tile.count <= 0) {
        return;
      }
      WorkspaceAddresses addresses = ResolveWorkspace(input_list[WORKSPACE_IDX], tiling);
      RMSNormTilingData rmsTiling = {};
      rmsTiling.num_row = tiling->bs;
      rmsTiling.num_col = tiling->d;
      rmsTiling.num_col_align = tiling->d;
      rmsTiling.block_factor = taskDesc.task_split_value;
      // Reuse the pinned ops-nn 910B BF16 normal-mode policy. Native
      // tiling uses row_factor=min(64, block_factor) and a 12288-element
      // UB tile when H fits the normal path (including V4.1 H=5120).
      rmsTiling.row_factor = MinValue(RMS_NORM_NATIVE_ROW_FACTOR, taskDesc.task_split_value);
      rmsTiling.ub_factor = RMS_NORM_NATIVE_BF16_UB_FACTOR;
      rmsTiling.epsilon = tiling->normEps;
      rmsTiling.avg_factor = 1.0f / static_cast<float>(tiling->d);
      rmsTiling.is_gemma = 0;

      RmsNorm::KernelRmsNorm<DTYPE_RESIDUAL, DTYPE_RESIDUAL> rmsNorm;
      rmsNorm.InitTask(addresses.mixedInput, input_list[8], input_list[16], addresses.rmsRstd, &rmsTiling,
                       input_list[WORKSPACE_IDX], taskDesc.task_index, taskDesc.task_split_num);
      rmsNorm.Process();
    }
  }
};

extern "C" inline __aicore__ void worker_kernel(uint32_t workerId, __gm__ uint8_t *runtimeConfig, GM_ADDR *inputList) {
  KernelWorker worker;
  worker.Init(workerId, runtimeConfig, inputList);
  worker.Process();
}
