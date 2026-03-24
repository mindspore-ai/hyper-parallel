/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "acl/acl.h"

#include "shmem.h"

using namespace AscendC;
namespace ShmemKernel {

template <typename T>
__aicore__ void CopyGmSingleValueToUb(GM_ADDR gm_addr, T *result) {
  __ubuf__ T *ubAddr = (__ubuf__ T *)(32);
  aclshmemi_copy_gm2ub(ubAddr, (__gm__ T *)gm_addr, sizeof(T));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  *result = *ubAddr;
}

class SignalOpKernel {
 public:
  __aicore__ inline SignalOpKernel() {}
  __aicore__ inline void Init(GM_ADDR signal, GM_ADDR signal_offset, GM_ADDR signal_value, int64_t signal_op,
                              int64_t target_pe) {
    // Initialize pointers and parameters
    signal_ = reinterpret_cast<__gm__ int32_t *>(signal);

    AscendC::PipeBarrier<PIPE_ALL>();
    CopyGmSingleValueToUb<int64_t>(signal_offset, &signal_offset_);
    CopyGmSingleValueToUb<int32_t>(signal_value, &signal_value_);

    signal_op_ = signal_op;
    target_pe_ = target_pe;
  }

  __aicore__ inline void Process() {
    auto signal_ptr = signal_ + signal_offset_;
    aclshmemx_signal_op(signal_ptr, signal_value_, signal_op_ == 1 ? ACLSHMEM_SIGNAL_ADD : ACLSHMEM_SIGNAL_SET,
                        target_pe_);
  }

 private:
  __gm__ int32_t *signal_;

  int64_t signal_offset_ = 0;
  int32_t signal_value_ = 0;
  int64_t signal_op_ = 0;
  int64_t target_pe_ = 0;
};

__global__ __aicore__ void signal_op_kernel(GM_ADDR signal, GM_ADDR signal_offset, GM_ADDR signal_value,
                                            int64_t signal_op, int64_t target_pe) {
  SignalOpKernel op;
  op.Init(signal, signal_offset, signal_value, signal_op, target_pe);
  op.Process();
}

void signal_op(uint32_t block_dim, void *stream, uint8_t *signal, uint8_t *signal_offset, uint8_t *signal_value,
               int64_t signal_op, int64_t target_pe) {
  signal_op_kernel<<<block_dim, nullptr, stream>>>(signal, signal_offset, signal_value, signal_op, target_pe);
}
}  // namespace ShmemKernel
