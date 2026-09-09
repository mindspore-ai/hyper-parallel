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

#include <cstddef>
#include <cstdint>

using aclError = int32_t;
using aclrtStream = void *;
using data_op_engine_type_t = int32_t;

inline constexpr aclError ACL_SUCCESS = 0;
inline constexpr int32_t ACLSHMEM_SUCCESS = 0;
inline constexpr int32_t ACLSHMEM_TIMEOUT_ERROR = 1001;
inline constexpr int32_t ACLSHMEM_INNER_TIMEOUT = 1002;
inline constexpr int32_t ACLSHMEMX_INIT_WITH_DEFAULT = 1;
inline constexpr data_op_engine_type_t ACLSHMEM_DATA_OP_MTE = 2;
inline constexpr int32_t ACLSHMEM_TEAM_WORLD = 0;
inline constexpr int32_t ACLSHMEM_CMP_EQ = 0;
inline constexpr int32_t ACLSHMEM_CMP_NE = 1;
inline constexpr int32_t ACLSHMEM_CMP_GT = 2;
inline constexpr int32_t ACLSHMEM_CMP_GE = 3;
inline constexpr int32_t ACLSHMEM_CMP_LT = 4;
inline constexpr int32_t ACLSHMEM_CMP_LE = 5;
inline constexpr int32_t ACLSHMEM_SIGNAL_SET = 6;
inline constexpr int32_t ACLSHMEM_SIGNAL_ADD = 7;
inline constexpr std::size_t ACLSHMEM_MAX_IP_PORT_LEN = 64U;

struct aclshmem_init_optional_attr_t {
  int32_t version;
  data_op_engine_type_t data_op_engine_type;
  uint32_t shm_init_timeout;
  uint32_t shm_create_timeout;
  uint32_t control_operation_timeout;
  int32_t sockFd;
};

struct aclshmemx_init_attr_t {
  int32_t my_pe;
  int32_t n_pes;
  char ip_port[ACLSHMEM_MAX_IP_PORT_LEN];
  uint64_t local_mem_size;
  aclshmem_init_optional_attr_t option_attr;
  void *comm_args;
  uint64_t instance_id;
};

int32_t aclshmemx_set_conf_store_tls(bool enabled, void *config, uint32_t config_size);
int32_t aclshmemx_init_attr(int32_t mode, aclshmemx_init_attr_t *attributes);
int32_t aclshmem_finalize();
void *aclshmem_malloc(std::size_t bytes);
void *aclshmem_align(std::size_t alignment_bytes, std::size_t bytes);
void aclshmem_free(void *allocation);
void aclshmemx_barrier_on_stream(int32_t team, aclrtStream stream);
void aclshmemx_putmem_on_stream(void *remote_dst, void *local_src, std::size_t bytes, int32_t target_pe,
                                aclrtStream stream);
void aclshmemx_getmem_on_stream(void *local_dst, void *remote_src, std::size_t bytes, int32_t source_pe,
                                aclrtStream stream);
void aclshmemx_signal_op_on_stream(int32_t *remote_signal, int32_t value, int32_t operation, int32_t target_pe,
                                   aclrtStream stream);
void aclshmemx_signal_wait_until_on_stream(int32_t *signal, int32_t comparison, int32_t value, aclrtStream stream);
aclError aclrtGetDevice(int32_t *device_index);
const char *aclrtGetSocName();
