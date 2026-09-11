/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fake_acl_cann.h"

#include <cstring>

namespace hyper_parallel::multicore::shmem::tests::fake_acl_cann {
namespace {

State state;

}  // namespace

State &Get() { return state; }

void Reset() { state = State{}; }

}  // namespace hyper_parallel::multicore::shmem::tests::fake_acl_cann

int32_t aclshmemx_set_conf_store_tls(bool, void *, uint32_t) {
  auto &state = hyper_parallel::multicore::shmem::tests::fake_acl_cann::Get();
  state.calls.emplace_back("tls");
  return state.tls_result;
}

int32_t aclshmemx_init_attr(int32_t mode, aclshmemx_init_attr_t *attributes) {
  auto &state = hyper_parallel::multicore::shmem::tests::fake_acl_cann::Get();
  state.calls.emplace_back("init");
  state.init_mode = mode;
  std::memcpy(&state.init_attributes, attributes, sizeof(*attributes));
  return state.init_result;
}

int32_t aclshmem_finalize() {
  auto &state = hyper_parallel::multicore::shmem::tests::fake_acl_cann::Get();
  state.calls.emplace_back("finalize");
  return state.finalize_result;
}

void *aclshmem_malloc(std::size_t bytes) {
  auto &state = hyper_parallel::multicore::shmem::tests::fake_acl_cann::Get();
  state.calls.emplace_back("malloc");
  state.malloc_bytes = bytes;
  return state.malloc_result;
}

void *aclshmem_align(std::size_t alignment_bytes, std::size_t bytes) {
  auto &state = hyper_parallel::multicore::shmem::tests::fake_acl_cann::Get();
  state.calls.emplace_back("align");
  state.alignment_bytes = alignment_bytes;
  state.aligned_bytes = bytes;
  return state.aligned_malloc_result;
}

void aclshmem_free(void *allocation) {
  auto &state = hyper_parallel::multicore::shmem::tests::fake_acl_cann::Get();
  state.calls.emplace_back("free");
  state.freed_address = allocation;
}

void aclshmemx_barrier_on_stream(int32_t, aclrtStream) {
  hyper_parallel::multicore::shmem::tests::fake_acl_cann::Get().calls.emplace_back("barrier");
}

void aclshmemx_putmem_on_stream(void *remote_dst, void *local_src, std::size_t bytes, int32_t target_pe,
                                aclrtStream stream) {
  auto &state = hyper_parallel::multicore::shmem::tests::fake_acl_cann::Get();
  state.calls.emplace_back("put_on_stream");
  state.first_address = remote_dst;
  state.second_address = local_src;
  state.transfer_bytes = bytes;
  state.target_pe = target_pe;
  state.stream = stream;
}

void aclshmemx_getmem_on_stream(void *local_dst, void *remote_src, std::size_t bytes, int32_t source_pe,
                                aclrtStream stream) {
  auto &state = hyper_parallel::multicore::shmem::tests::fake_acl_cann::Get();
  state.calls.emplace_back("get_on_stream");
  state.first_address = local_dst;
  state.second_address = remote_src;
  state.transfer_bytes = bytes;
  state.target_pe = source_pe;
  state.stream = stream;
}

void aclshmemx_signal_op_on_stream(int32_t *remote_signal, int32_t value, int32_t operation, int32_t target_pe,
                                   aclrtStream stream) {
  auto &state = hyper_parallel::multicore::shmem::tests::fake_acl_cann::Get();
  state.calls.emplace_back("signal_on_stream");
  state.first_address = remote_signal;
  state.signal_value = value;
  state.signal_operation = operation;
  state.target_pe = target_pe;
  state.stream = stream;
}

void aclshmemx_signal_wait_until_on_stream(int32_t *signal, int32_t comparison, int32_t value, aclrtStream stream) {
  auto &state = hyper_parallel::multicore::shmem::tests::fake_acl_cann::Get();
  state.calls.emplace_back("wait_signal_on_stream");
  state.first_address = signal;
  state.signal_comparison = comparison;
  state.signal_value = value;
  state.stream = stream;
}

aclError aclrtGetDevice(int32_t *device_index) {
  auto &state = hyper_parallel::multicore::shmem::tests::fake_acl_cann::Get();
  state.calls.emplace_back("get_device");
  if (state.get_device_result == ACL_SUCCESS) {
    *device_index = state.current_device_index;
  }
  return state.get_device_result;
}

const char *aclrtGetSocName() {
  auto &state = hyper_parallel::multicore::shmem::tests::fake_acl_cann::Get();
  state.calls.emplace_back("get_soc_name");
  return state.soc_name.c_str();
}
