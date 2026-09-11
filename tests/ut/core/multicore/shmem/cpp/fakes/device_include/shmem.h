/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#define __gm__
#define __ubuf__
#define __aicore__
#define ACLSHMEM_DEVICE inline

inline constexpr int32_t ACLSHMEM_CMP_EQ = 0;
inline constexpr int32_t ACLSHMEM_CMP_NE = 1;
inline constexpr int32_t ACLSHMEM_CMP_GT = 2;
inline constexpr int32_t ACLSHMEM_CMP_GE = 3;
inline constexpr int32_t ACLSHMEM_CMP_LT = 4;
inline constexpr int32_t ACLSHMEM_CMP_LE = 5;
inline constexpr int32_t ACLSHMEM_SIGNAL_SET = 6;
inline constexpr int32_t ACLSHMEM_SIGNAL_ADD = 7;
inline constexpr int32_t ACLSHMEM_TEAM_WORLD = 0;

void *aclshmem_ptr(void *local_symmetric_address, int32_t target_pe);
void aclshmem_fence();
void aclshmem_putmem(void *remote_dst, void *local_src, uint32_t bytes, int32_t target_pe);
void aclshmem_getmem(void *local_dst, void *remote_src, uint32_t bytes, int32_t target_pe);
void aclshmemx_signal_op(int32_t *remote_signal, int32_t value, int32_t operation, int32_t target_pe);
int32_t aclshmem_signal_wait_until(int32_t *local_signal, int32_t compare, int32_t value);
void aclshmem_barrier(int32_t team);

template <typename T>
void aclshmemx_mte_put_nbi(T *remote_dst, T *local_src, uint8_t *ub_scratch, uint32_t ub_scratch_bytes, uint32_t bytes,
                           int32_t target_pe, uint32_t event_id);

void aclshmemx_mte_quiet();
