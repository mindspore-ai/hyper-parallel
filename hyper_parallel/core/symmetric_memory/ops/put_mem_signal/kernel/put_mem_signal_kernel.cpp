/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>

#include "kernel_operator.h"
#include "acl/acl.h"

#include "shmem.h"

using namespace AscendC;
inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg{0};
namespace ShmemKernel {

template <typename T>
__aicore__ void CopyGmSingleValueToUb(GM_ADDR gm_addr, T *result) {
  __ubuf__ T *ubAddr = (__ubuf__ T *)(32);
  aclshmemi_copy_gm2ub(ubAddr, (__gm__ T *)gm_addr, sizeof(T));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  *result = *ubAddr;
}

template <typename T>
class PutmemSignalKernel {
 public:
  __aicore__ inline PutmemSignalKernel() {}
  __aicore__ inline void Init(GM_ADDR target, GM_ADDR target_offset, GM_ADDR src, GM_ADDR src_offset, GM_ADDR size,
                              GM_ADDR signal, GM_ADDR signal_offset, GM_ADDR signal_value, int64_t signal_op,
                              int64_t target_pe, bool non_blocking, GM_ADDR Syncmem) {
    // Initialize pointers and parameters
    target_ = reinterpret_cast<__gm__ T *>(target);
    src_ = reinterpret_cast<__gm__ T *>(src);
    signal_ = reinterpret_cast<__gm__ int32_t *>(signal);

    AscendC::PipeBarrier<PIPE_ALL>();
    CopyGmSingleValueToUb<int64_t>(target_offset, &target_offset_);
    CopyGmSingleValueToUb<int64_t>(src_offset, &src_offset_);
    CopyGmSingleValueToUb<int64_t>(size, &size_);
    CopyGmSingleValueToUb<int64_t>(signal_offset, &signal_offset_);
    CopyGmSingleValueToUb<int32_t>(signal_value, &signal_value_);

    signal_op_ = signal_op;
    target_pe_ = target_pe;
    non_blocking_ = non_blocking;

    aiv_idx_ = get_block_idx() * get_subblockdim() + get_subblockid();
    aiv_num_ = get_block_num() * get_subblockdim();

    syncGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(Syncmem), aiv_num_ * 8);
    pipe.InitBuffer(localwork, 1, sizeof(int32_t) * aiv_num_);
  }

  __aicore__ inline void Process() {
    // Perform data transfer
    auto size_per_core = size_ / aiv_num_;
    auto target_ptr = target_ + target_offset_ + aiv_idx_ * size_per_core;
    auto src_ptr = src_ + src_offset_ + aiv_idx_ * size_per_core;
    if (aiv_idx_ == aiv_num_ - 1) {
      size_per_core = size_ - size_per_core * aiv_idx_;
    }
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    __ubuf__ T *ping_buff = reinterpret_cast<__ubuf__ T *>(copy_ub);
    __ubuf__ T *pong_buff = reinterpret_cast<__ubuf__ T *>(copy_ub + copy_ub_size / 2);
    auto ptr = aclshmem_ptr(target_ptr, target_pe_);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);
    uint64_t block_size = copy_ub_size / 2 / sizeof(T) * sizeof(T);
    uint64_t remain = (size_per_core * sizeof(T)) % block_size;

    uint64_t repeat_times = (size_per_core * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint64_t i = 0; i < repeat_times; i++) {
      AscendC::TEventID EVENT_ID = (i & 1) ? EVENT_ID0 : EVENT_ID1;
      __ubuf__ T *buf = (i & 1) ? ping_buff : pong_buff;
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
      aclshmemi_copy_gm2ub(buf, src_ptr + i * repeat_elem, block_size);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
      aclshmemi_copy_ub2gm(remote_ptr + i * repeat_elem, buf, block_size);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
    if (remain > 0) {
      AscendC::TEventID EVENT_ID = (repeat_times & 1) ? EVENT_ID0 : EVENT_ID1;
      __ubuf__ T *buf = (repeat_times & 1) ? ping_buff : pong_buff;
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
      aclshmemi_copy_gm2ub(buf, src_ptr + repeat_times * repeat_elem, remain);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
      aclshmemi_copy_ub2gm(remote_ptr + repeat_times * repeat_elem, buf, remain);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

    if (non_blocking_) {
      // aclshmem_putmem_nbi(target_ptr, src_ptr, size_per_core * sizeof(T), target_pe_);
      aclshmem_fence();
    } else {
      // aclshmem_putmem_nbi(target_ptr, src_ptr, size_per_core * sizeof(T), target_pe_);
      aclshmem_fence();
    }

    if (aiv_num_ > 1) {
      AscendC::LocalTensor<int32_t> workLocal = localwork.AllocTensor<int32_t>();
      AscendC::SyncAll(syncGlobal, workLocal, aiv_num_);
      localwork.FreeTensor(workLocal);
    }
    auto signal_ptr = signal_ + signal_offset_;
    if (aiv_idx_ == 0) {
      aclshmemx_signal_op(signal_ptr, signal_value_, signal_op_ == 1 ? ACLSHMEM_SIGNAL_ADD : ACLSHMEM_SIGNAL_SET,
                          target_pe_);
    }
  }

 private:
  __gm__ T *target_;
  __gm__ T *src_;
  __gm__ int32_t *signal_;

  AscendC::TPipe pipe;

  int64_t target_offset_ = 0;
  int64_t src_offset_ = 0;
  int64_t size_ = 0;
  int64_t signal_offset_ = 0;
  int32_t signal_value_ = 0;
  int64_t signal_op_ = 0;
  int64_t target_pe_ = 0;
  bool non_blocking_ = true;

  uint32_t aiv_idx_ = 0;
  uint32_t aiv_num_ = 0;

  AscendC::GlobalTensor<int32_t> syncGlobal;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> localwork;
};

template <typename T>
__global__ __aicore__ void put_mem_signal_kernel(GM_ADDR target, GM_ADDR target_offset, GM_ADDR src, GM_ADDR src_offset,
                                                 GM_ADDR size, GM_ADDR signal, GM_ADDR signal_offset,
                                                 GM_ADDR signal_value, int64_t signal_op, int64_t target_pe,
                                                 bool non_blocking, GM_ADDR Syncmem) {
  PutmemSignalKernel<T> op;
  op.Init(target, target_offset, src, src_offset, size, signal, signal_offset, signal_value, signal_op, target_pe,
          non_blocking, Syncmem);
  op.Process();
}

void put_mem_signal(uint32_t block_dim, void *stream, uint64_t elementSize, uint8_t *target, uint8_t *target_offset,
                    uint8_t *src, uint8_t *src_offset, uint8_t *size, uint8_t *signal, uint8_t *signal_offset,
                    uint8_t *signal_value, int64_t signal_op, int64_t target_pe, bool non_blocking, uint8_t *Syncmem) {
  if (elementSize == 1) {
    put_mem_signal_kernel<int8_t><<<block_dim, nullptr, stream>>>(target, target_offset, src, src_offset, size, signal,
                                                                  signal_offset, signal_value, signal_op, target_pe,
                                                                  non_blocking, Syncmem);
  } else if (elementSize == 2) {
    put_mem_signal_kernel<int16_t><<<block_dim, nullptr, stream>>>(target, target_offset, src, src_offset, size, signal,
                                                                   signal_offset, signal_value, signal_op, target_pe,
                                                                   non_blocking, Syncmem);
  } else if (elementSize == 4) {
    put_mem_signal_kernel<int32_t><<<block_dim, nullptr, stream>>>(target, target_offset, src, src_offset, size, signal,
                                                                   signal_offset, signal_value, signal_op, target_pe,
                                                                   non_blocking, Syncmem);
  } else if (elementSize == 8) {
    put_mem_signal_kernel<int64_t><<<block_dim, nullptr, stream>>>(target, target_offset, src, src_offset, size, signal,
                                                                   signal_offset, signal_value, signal_op, target_pe,
                                                                   non_blocking, Syncmem);
  } else {
    std::cerr << "Unknown dtype with size" << elementSize << std::endl;
  }
}
}  // namespace ShmemKernel
