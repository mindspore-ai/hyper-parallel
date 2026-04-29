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
 * @file multicore_moe_ffn_tiling.h
 */
#ifndef MULTICORE_MOE_FFN_TILING_H
#define MULTICORE_MOE_FFN_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint64_t, smallCoreDataNum);
TILING_DATA_FIELD_DEF(uint64_t, bigCoreDataNum);
TILING_DATA_FIELD_DEF(uint64_t, ubPartDataNum);
TILING_DATA_FIELD_DEF(uint64_t, smallCoreTailDataNum);
TILING_DATA_FIELD_DEF(uint64_t, bigCoreTailDataNum);
TILING_DATA_FIELD_DEF(uint64_t, smallCoreLoopNum);
TILING_DATA_FIELD_DEF(uint64_t, bigCoreLoopNum);
TILING_DATA_FIELD_DEF(uint64_t, tailBlockNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TilingDataOp, TilingData)

BEGIN_TILING_DATA_DEF(SwiGluTilingData)
TILING_DATA_FIELD_DEF(uint32_t, is32BAligned);
TILING_DATA_FIELD_DEF(uint32_t, isDoubleBuffer);
TILING_DATA_FIELD_DEF(uint64_t, rowLen);
TILING_DATA_FIELD_DEF(uint64_t, colLen);
TILING_DATA_FIELD_DEF(uint32_t, baseRowLen);
TILING_DATA_FIELD_DEF(uint32_t, baseColLen);
TILING_DATA_FIELD_DEF(uint32_t, activateLeft);
TILING_DATA_FIELD_DEF(uint32_t, biasIsEmpty);
TILING_DATA_FIELD_DEF(uint32_t, quantScaleIsEmpty);
TILING_DATA_FIELD_DEF(uint32_t, activateScaleIsEmpty);
TILING_DATA_FIELD_DEF(uint64_t, swiColLen);
TILING_DATA_FIELD_DEF(uint64_t, perRowLen);
TILING_DATA_FIELD_DEF(uint64_t, modRowLen);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SwiGluTilingDataOp, SwiGluTilingData)

BEGIN_TILING_DATA_DEF(L2cacheUseInfo)
TILING_DATA_FIELD_DEF(uint32_t, l2CacheFlag);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(L2cacheUseInfoOp, L2cacheUseInfo);

BEGIN_TILING_DATA_DEF(L2cacheTilePara)
TILING_DATA_FIELD_DEF(uint32_t, mTileCntL2);
TILING_DATA_FIELD_DEF(uint32_t, nTileCntL2);
TILING_DATA_FIELD_DEF(uint32_t, mTileBlock);
TILING_DATA_FIELD_DEF(uint32_t, nTileBlock);
TILING_DATA_FIELD_DEF(uint32_t, calOrder);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(L2cacheTileParaOp, L2cacheTilePara)

BEGIN_TILING_DATA_DEF(MatMulRunInfo)
TILING_DATA_FIELD_DEF(uint32_t, transA);
TILING_DATA_FIELD_DEF(uint32_t, transB);
TILING_DATA_FIELD_DEF(uint32_t, nd2nzA);
TILING_DATA_FIELD_DEF(uint32_t, nd2nzB);
TILING_DATA_FIELD_DEF(uint32_t, isNzA);
TILING_DATA_FIELD_DEF(uint32_t, isNzB);
TILING_DATA_FIELD_DEF(uint32_t, isHf32);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatMulRunInfoOp, MatMulRunInfo)

BEGIN_TILING_DATA_DEF(MatmulTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
TILING_DATA_FIELD_DEF_STRUCT(L2cacheTilePara, tileL2cacheTiling);
TILING_DATA_FIELD_DEF_STRUCT(MatMulRunInfo, matmulRunInfo);
TILING_DATA_FIELD_DEF_STRUCT(L2cacheUseInfo, l2cacheUseInfo);
TILING_DATA_FIELD_DEF(uint32_t, baseAN);
TILING_DATA_FIELD_DEF(uint32_t, baseAD);
TILING_DATA_FIELD_DEF(uint32_t, baseBN);
TILING_DATA_FIELD_DEF(uint32_t, baseBD);
TILING_DATA_FIELD_DEF(uint32_t, taskNumMSplit);
END_TILING_DATA_DEF;

// REGISTER_TILING_DATA_CLASS(MatMulV3, MatmulTilingData)
REGISTER_TILING_DATA_CLASS(MatmulTilingDataOp, MatmulTilingData)

BEGIN_TILING_DATA_DEF(GMMBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, activeType);
TILING_DATA_FIELD_DEF(uint32_t, ubBaseK);
TILING_DATA_FIELD_DEF(uint32_t, ubBaseN);
TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);
TILING_DATA_FIELD_DEF(uint32_t, singleWeight);
TILING_DATA_FIELD_DEF(uint32_t, singleX);
TILING_DATA_FIELD_DEF(uint32_t, singleY);
TILING_DATA_FIELD_DEF(int32_t, groupType);
TILING_DATA_FIELD_DEF(uint32_t, singleN);     // If sequential write， the value should be zero!
TILING_DATA_FIELD_DEF(uint32_t, quantParam);  // in quant case, PerToken: 1; in antiquant case, represents PerGroupSize
TILING_DATA_FIELD_DEF(uint32_t, groupListType);
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, hasBias);
TILING_DATA_FIELD_DEF(uint64_t, workspaceSize);
TILING_DATA_FIELD_DEF(uint64_t, totalInGroup);   // for A8W4 MSD
TILING_DATA_FIELD_DEF(uint64_t, k);              // for A8W4 MSD
TILING_DATA_FIELD_DEF(uint64_t, n);              // for A8W4 MSD
TILING_DATA_FIELD_DEF(uint64_t, vBaseM);         // for A8W4 MSD
TILING_DATA_FIELD_DEF(uint64_t, parallNum);      // for A8W4 MSD
TILING_DATA_FIELD_DEF(uint64_t, quantGroupNum);  // for A8W4 MSD
TILING_DATA_FIELD_DEF(uint64_t, isPreTiling);
TILING_DATA_FIELD_DEF(uint32_t, withOffset);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMBaseParamsOp, GMMBaseParams)

BEGIN_TILING_DATA_DEF(GMMArray)
TILING_DATA_FIELD_DEF_ARR(int32_t, 128, mList);  // 128 ：MAX_TENSOR_CONT
TILING_DATA_FIELD_DEF_ARR(int32_t, 128, kList);
TILING_DATA_FIELD_DEF_ARR(int32_t, 128, nList);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMArrayOp, GMMArray)

BEGIN_TILING_DATA_DEF(GMMQuantParams)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, activeType);
TILING_DATA_FIELD_DEF(uint32_t, aQuantMode);
TILING_DATA_FIELD_DEF(uint32_t, bQuantMode);
TILING_DATA_FIELD_DEF(uint8_t, singleX);
TILING_DATA_FIELD_DEF(uint8_t, singleW);
TILING_DATA_FIELD_DEF(uint8_t, singleY);
TILING_DATA_FIELD_DEF(int8_t, groupType);
TILING_DATA_FIELD_DEF(uint8_t, groupListType);
TILING_DATA_FIELD_DEF(uint8_t, hasBias);
TILING_DATA_FIELD_DEF(uint16_t, reserved);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMQuantParamsOp, GMMQuantParams)

BEGIN_TILING_DATA_DEF(GMMQuantTilingData)
TILING_DATA_FIELD_DEF_STRUCT(GMMQuantParams, gmmQuantParams);
TILING_DATA_FIELD_DEF_STRUCT(GMMArray, gmmArray);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedMatmul_20000000000, GMMQuantTilingData)
REGISTER_TILING_DATA_CLASS(GroupedMatmul_20000000001, GMMQuantTilingData)
REGISTER_TILING_DATA_CLASS(GroupedMatmul_20000000010, GMMQuantTilingData)
REGISTER_TILING_DATA_CLASS(GroupedMatmul_20000000100, GMMQuantTilingData)
REGISTER_TILING_DATA_CLASS(GroupedMatmul_20000000101, GMMQuantTilingData)
REGISTER_TILING_DATA_CLASS(GroupedMatmul_20000000110, GMMQuantTilingData)
REGISTER_TILING_DATA_CLASS(GroupedMatmul_20000000200, GMMQuantTilingData)
REGISTER_TILING_DATA_CLASS(GroupedMatmul_20000000201, GMMQuantTilingData)
REGISTER_TILING_DATA_CLASS(GroupedMatmul_20000000210, GMMQuantTilingData)

BEGIN_TILING_DATA_DEF(GMMWeightQuantParam)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint64_t, kSize);
TILING_DATA_FIELD_DEF(uint64_t, nSize);
TILING_DATA_FIELD_DEF(uint8_t, singleX);
TILING_DATA_FIELD_DEF(uint8_t, singleWeight);
TILING_DATA_FIELD_DEF(uint8_t, singleY);
TILING_DATA_FIELD_DEF(int8_t, groupType);
TILING_DATA_FIELD_DEF(uint8_t, groupListType);
TILING_DATA_FIELD_DEF(uint8_t, hasBias);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimN);
TILING_DATA_FIELD_DEF(uint8_t, reserved);
TILING_DATA_FIELD_DEF(uint32_t, groupSize);
TILING_DATA_FIELD_DEF(uint32_t, mainBlockSize);
TILING_DATA_FIELD_DEF(uint64_t, mainBlockCount);
TILING_DATA_FIELD_DEF(uint16_t, firstTailBlockSize);
TILING_DATA_FIELD_DEF(uint16_t, secondTailBlockSize);
TILING_DATA_FIELD_DEF(uint16_t, firstTailBlockCount);
TILING_DATA_FIELD_DEF(uint16_t, secondTailBlockCount);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMWeightQuantParamOp, GMMWeightQuantParam)

BEGIN_TILING_DATA_DEF(GMMWeightQuantTilingData)
TILING_DATA_FIELD_DEF_STRUCT(GMMWeightQuantParam, gmmWeightQuantParam);
TILING_DATA_FIELD_DEF_STRUCT(GMMArray, gmmArray);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedMatmul_2000020003000012000, GMMWeightQuantTilingData)
REGISTER_TILING_DATA_CLASS(GroupedMatmul_2000020003000012020, GMMWeightQuantTilingData)
REGISTER_TILING_DATA_CLASS(GroupedMatmul_2000020004000002001, GMMWeightQuantTilingData)
REGISTER_TILING_DATA_CLASS(GroupedMatmul_2000020003000004001, GMMWeightQuantTilingData)

BEGIN_TILING_DATA_DEF(GMMNoQuantBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, singleWeight);
TILING_DATA_FIELD_DEF(uint32_t, singleX);
TILING_DATA_FIELD_DEF(uint32_t, singleY);
TILING_DATA_FIELD_DEF(int32_t, groupType);
TILING_DATA_FIELD_DEF(uint32_t, groupListType);
TILING_DATA_FIELD_DEF(uint32_t, hasBias);
TILING_DATA_FIELD_DEF(uint32_t, mTailCnt);
TILING_DATA_FIELD_DEF(uint32_t, nTailCnt);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMNoQuantBaseParamsOp, GMMNoQuantBaseParams)

BEGIN_TILING_DATA_DEF(GMMNoQuantTilingData)
TILING_DATA_FIELD_DEF_STRUCT(GMMNoQuantBaseParams, gmmNoQuantParam);
TILING_DATA_FIELD_DEF_STRUCT(GMMArray, gmmArray);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedMatmul_10000900009000090000, GMMNoQuantTilingData)
REGISTER_TILING_DATA_CLASS(GroupedMatmul_10000900009000090001, GMMNoQuantTilingData)
REGISTER_TILING_DATA_CLASS(GroupedMatmul_10000900009000090002, GMMNoQuantTilingData)

// for autotiling w4a8
BEGIN_TILING_DATA_DEF(A8W4HPTiling)
TILING_DATA_FIELD_DEF(uint32_t, group_num);
TILING_DATA_FIELD_DEF(int8_t, group_type);
TILING_DATA_FIELD_DEF(uint32_t, required_core_num);
TILING_DATA_FIELD_DEF(float, format_in);
TILING_DATA_FIELD_DEF(float, format_out);
TILING_DATA_FIELD_DEF(uint32_t, numAic);
TILING_DATA_FIELD_DEF(uint32_t, numAiv);
TILING_DATA_FIELD_DEF(uint64_t, szUb);
TILING_DATA_FIELD_DEF(uint64_t, szL0A);
TILING_DATA_FIELD_DEF(uint64_t, szL0C);
TILING_DATA_FIELD_DEF(uint8_t, pattern);
TILING_DATA_FIELD_DEF(uint8_t, kernel_index);
TILING_DATA_FIELD_DEF(uint32_t, splitTimes);
TILING_DATA_FIELD_DEF(int8_t, output_type);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 2, ori_in0_shape);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 2, ori_in1_shape);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 2, ori_out_shape);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, single_core_tiling);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 2, single_core_base_tiling);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 3, splitRecord);
TILING_DATA_FIELD_DEF(uint64_t, workspaceOffset);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(A8W4HPTilingOp, A8W4HPTiling)

BEGIN_TILING_DATA_DEF(GMMTilingData)
TILING_DATA_FIELD_DEF_STRUCT(GMMBaseParams, gmmBaseParams);
TILING_DATA_FIELD_DEF_STRUCT(GMMArray, gmmArray);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
// for autotiling
TILING_DATA_FIELD_DEF_STRUCT(A8W4HPTiling, hpTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GMMTilingDataOp, GMMTilingData)

BEGIN_TILING_DATA_DEF(MulticoreMoeFfnTilingData)
TILING_DATA_FIELD_DEF(int64_t, rankId);
TILING_DATA_FIELD_DEF(int64_t, ep);
TILING_DATA_FIELD_DEF(int64_t, expertNum);
TILING_DATA_FIELD_DEF(int64_t, hiddenSize);
TILING_DATA_FIELD_DEF(int64_t, seqSize);
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF_STRUCT(TilingData, tilingData);
TILING_DATA_FIELD_DEF_STRUCT(SwiGluTilingData, swiGluTilingData);
TILING_DATA_FIELD_DEF_STRUCT(MatmulTilingData, matmulTilingData);
TILING_DATA_FIELD_DEF_STRUCT(GMMTilingData, gmmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MulticoreMoeFfn, MulticoreMoeFfnTilingData)
}  // namespace optiling
#endif  // MULTICORE_MOE_FFN_TILING_H

// BEGIN_TILING_DATA_DEF(Data)
//   TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_EVENT_NUM, queues);
// END_TILING_DATA_DEF;
// REGISTER_TILING_DATA_CLASS(DataOp, Data);

// BEGIN_TILING_DATA_DEF(DataConfig)
//   TILING_DATA_FIELD_DEF_ARR(Data, MAX_TASK_NUM, all_tasks);
// END_TILING_DATA_DEF;
// REGISTER_TILING_DATA_CLASS(DataConfigOp, DataConfig);
