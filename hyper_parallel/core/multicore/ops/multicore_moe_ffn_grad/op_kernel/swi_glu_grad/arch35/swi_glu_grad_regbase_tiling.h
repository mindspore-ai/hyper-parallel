/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swi_glu_grad_regbase_tiling.h
 * \brief
 */
#include <cstdint>

struct GluBaseTilingData {
    int64_t rowTotal;
    int64_t colTotal;
    int64_t rowBase;
    int64_t colBase;
    int64_t rowTail;
    int64_t colTail;
    int64_t ubSize;
    int64_t rowTileNum;
    int64_t colTileNum;
    int64_t usedCoreNum;
};