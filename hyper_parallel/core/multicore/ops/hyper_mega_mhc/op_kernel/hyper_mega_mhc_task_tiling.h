/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef HYPER_MEGA_MHC_TASK_TILING_H_
#define HYPER_MEGA_MHC_TASK_TILING_H_

#include "kernel_operator.h"

// Device-only mirror of the pinned ops-nn RMSNorm tiling layout. HyperMegaMhc
// initializes the fields consumed by KernelRmsNorm directly for each token task.
struct RMSNormTilingData {
  uint64_t num_row;
  uint64_t num_col;
  uint64_t num_col_align;
  uint64_t block_factor;
  uint32_t row_factor;
  uint32_t ub_factor;
  uint32_t reduce_mask;
  uint32_t left_num;
  uint32_t last_reduce_mask;
  uint32_t last_left_num;
  uint32_t rstd_size;
  uint32_t ub_loop;
  uint32_t col_buffer_length;
  uint32_t multi_n_num;
  uint32_t is_nddma;
  float epsilon;
  float avg_factor;
  uint8_t is_gemma;
  uint64_t last_block_factor;
  uint64_t row_loop;
  uint64_t last_block_row_loop;
  uint64_t row_tail;
  uint64_t last_block_row_tail;
  uint32_t mul_loop;
  uint32_t mul_tail;
  uint8_t dst_rep_stride;
  uint8_t is_performance;
  uint8_t normal_flag;
};

#endif  // HYPER_MEGA_MHC_TASK_TILING_H_
