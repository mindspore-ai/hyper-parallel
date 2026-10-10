/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef HYPER_MEGA_GATE_ROUTE_GRAD_TILING_H
#define HYPER_MEGA_GATE_ROUTE_GRAD_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(HyperMegaGateRouteGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, tokenCount);
TILING_DATA_FIELD_DEF(uint32_t, expertCount);
TILING_DATA_FIELD_DEF(uint32_t, topK);
TILING_DATA_FIELD_DEF(float, routedScalingFactor);
TILING_DATA_FIELD_DEF(uint32_t, stageCount);
TILING_DATA_FIELD_DEF(uint32_t, activeAivWorkerCount);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerWorker);
TILING_DATA_FIELD_DEF(uint32_t, aivWorkerSlotCapacity);
TILING_DATA_FIELD_DEF(uint32_t, batchRows);
TILING_DATA_FIELD_DEF(uint32_t, expertAlign);
TILING_DATA_FIELD_DEF(uint32_t, kAlign);
TILING_DATA_FIELD_DEF(uint32_t, sharedTmpBytes);
TILING_DATA_FIELD_DEF(uint64_t, runtimeConfigBytes);
TILING_DATA_FIELD_DEF(uint64_t, profileBufferBytes);
TILING_DATA_FIELD_DEF(uint64_t, kSlot0Offset);
TILING_DATA_FIELD_DEF(uint64_t, kSlot1Offset);
TILING_DATA_FIELD_DEF(uint64_t, kSlot2Offset);
TILING_DATA_FIELD_DEF(uint64_t, kSlot3Offset);
TILING_DATA_FIELD_DEF(uint64_t, kSlot4Offset);
TILING_DATA_FIELD_DEF(uint64_t, crossTermOffset);
TILING_DATA_FIELD_DEF(uint64_t, directTermOffset);
TILING_DATA_FIELD_DEF(uint64_t, broadcastRowSumOffset);
TILING_DATA_FIELD_DEF(uint64_t, rowSumOffset);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HyperMegaGateRouteGrad, HyperMegaGateRouteGradTilingData)

}  // namespace optiling

#endif  // HYPER_MEGA_GATE_ROUTE_GRAD_TILING_H
