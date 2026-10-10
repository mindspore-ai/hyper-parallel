/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef HYPER_MEGA_MHC_GRAD_TASK_TILING_H_
#define HYPER_MEGA_MHC_GRAD_TASK_TILING_H_

#include "kernel_operator.h"

// Device-only mirror of the pinned ops-nn RmsNormGrad tiling layout.
struct RmsNormGradTilingData {
    uint32_t row;
    uint32_t col;
    float avg_factor;
    uint32_t data_type;
    uint32_t block_factor;
    uint32_t ub_split_dim;
    uint32_t ub_factor;
    uint32_t core_calc_num;
    uint32_t core_calc_tail;
    uint32_t block_dim;
    uint32_t ub_calc_num;
    uint32_t ub_calc_tail;
    uint32_t ub_calc_loop;
    uint32_t ub_calc_tail_num;
    uint32_t ub_calc_tail_tail;
    uint32_t ub_calc_tail_loop;
    uint32_t fixed_output;
    uint32_t chunk_size;
    uint32_t chunk_num;
    uint32_t chunk_tail;
    uint32_t need_chunk;
};

#endif  // HYPER_MEGA_MHC_GRAD_TASK_TILING_H_
