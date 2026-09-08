/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <shmem.h>
#include <shmem_kernel.h>

#include <algorithm>
#include <cstring>
#include <iostream>  // NOLINT(build/include_order)
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "torch/custom_class.h"
#include "torch/types.h"
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace ShmemOps {

aclshmemx_uniqueid_t default_flag_uid;

static int64_t CheckedAllocationBytes(const std::vector<int64_t> &shape, torch::Dtype dtype) {
  TORCH_CHECK(!shape.empty(), "symmetric-memory allocation shape cannot be empty.");
  int64_t total_size = 1;
  for (const int64_t dimension : shape) {
    TORCH_CHECK(dimension > 0, "symmetric-memory dimensions must be positive, got ", dimension, ".");
    TORCH_CHECK(total_size <= std::numeric_limits<int64_t>::max() / dimension,
                "symmetric-memory allocation element count overflowed int64.");
    total_size *= dimension;
  }
  const int64_t element_size = at::elementSize(dtype);
  TORCH_CHECK(total_size <= std::numeric_limits<int64_t>::max() / element_size,
              "symmetric-memory allocation byte count overflowed int64.");
  return total_size * element_size;
}

class Manager : public torch::jit::CustomClassHolder {
 public:
  Manager() : name_("Manager") {}

  std::string get_name() const { return name_; }

  int64_t attr_init(int64_t my_pe, int64_t n_ranks, int64_t local_mem_size, const std::string &ip_port) {
    TORCH_CHECK(n_ranks > 0, "SHMEM rank count must be positive, got ", n_ranks, ".");
    TORCH_CHECK(my_pe >= 0 && my_pe < n_ranks, "SHMEM rank must be in [0, ", n_ranks, "), got ", my_pe, ".");
    TORCH_CHECK(local_mem_size > 0, "SHMEM heap size must be positive, got ", local_mem_size, ".");
    TORCH_CHECK(!ip_port.empty(), "SHMEM endpoint cannot be empty.");
    TORCH_CHECK(ip_port.size() < ACLSHMEM_MAX_IP_PORT_LEN, "SHMEM endpoint exceeds ", ACLSHMEM_MAX_IP_PORT_LEN - 1,
                " bytes: ", ip_port, ".");
    int32_t set_conf_status = aclshmemx_set_conf_store_tls(false, nullptr, 0);
    if (set_conf_status != 0) {
      std::cerr << "Aclshmem set conf store tls failed, error code:" << set_conf_status << std::endl;
      return set_conf_status;
    }
    aclshmemx_init_attr_t attributes{};
    int32_t set_attr_status = test_set_attr(my_pe, n_ranks, local_mem_size,
                                            ip_port.c_str(), default_flag_uid, &attributes);
    if (set_attr_status != 0) {
      std::cerr << "Aclshmem set attr failed, error code:" << set_attr_status << std::endl;
      return set_attr_status;
    }
    int32_t init_status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (init_status != 0) {
      std::cerr << "Aclshmem init failed, error code:" << init_status << std::endl;
      return init_status;
    }
    return 0;
  }

  int64_t finalize() { return aclshmem_finalize(); }

  at::Tensor malloc_tensor(const std::vector<int64_t> &shape, torch::Dtype dtype) {
    const int64_t allocation_bytes = CheckedAllocationBytes(shape, dtype);
    void *symm_ptr = aclshmem_malloc(allocation_bytes);
    TORCH_CHECK(symm_ptr != nullptr, "aclshmem_malloc failed for ", allocation_bytes,
                " bytes. Increase SYMMETRIC_MEMORY_HEAP_SIZE or release unused allocations.");
    // NPU storage metadata is required when invalidating a released allocation.
    return at_npu::native::from_blob(symm_ptr, shape, dtype);
  }

  at::Tensor aligned_malloc_tensor(const std::vector<int64_t> &shape, torch::Dtype dtype,
                                   int64_t alignment) {
    TORCH_CHECK(alignment > 0 && (alignment & (alignment - 1)) == 0,
                "alignment must be a positive power of two, got ", alignment, ".");
    const int64_t allocation_bytes = CheckedAllocationBytes(shape, dtype);
    void *symm_ptr = aclshmem_align(static_cast<size_t>(alignment), allocation_bytes);
    TORCH_CHECK(symm_ptr != nullptr, "aclshmem_align failed for ", allocation_bytes,
                " bytes with ", alignment, "-byte alignment. Increase SYMMETRIC_MEMORY_HEAP_SIZE.");
    return at_npu::native::from_blob(symm_ptr, shape, dtype);
  }

  void free_tensor(const at::Tensor &aclshmem_tensor) {
    TORCH_CHECK(aclshmem_tensor.defined(), "cannot free an undefined symmetric-memory tensor.");
    TORCH_CHECK(aclshmem_tensor.data_ptr() != nullptr,
                "cannot free a symmetric-memory tensor with a null data pointer.");
    void *aclshmem_ptr = const_cast<void *>(aclshmem_tensor.data_ptr());
    aclshmem_free(aclshmem_ptr);
  }

 private:
  int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port,
                        aclshmemx_uniqueid_t default_flag_uid, aclshmemx_init_attr_t *attributes) {
    size_t ip_len = 0;
    if (ip_port != nullptr) {
      ip_len = std::min(
          strlen(ip_port),
          static_cast<size_t>(ACLSHMEM_MAX_IP_PORT_LEN - 1));

      std::copy_n(ip_port, ip_len, attributes->ip_port);
      if (attributes->ip_port[0] == '\0') {
        return ACLSHMEM_INVALID_VALUE;
      }
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(&default_flag_uid);

    return ACLSHMEM_SUCCESS;
  }
  std::string name_;
};

static constexpr uint32_t DEFAULT_BLOCK_DIM = 1;
class Ops : public torch::jit::CustomClassHolder {
 public:
  Ops() : name_("Ops"), block_dim_(DEFAULT_BLOCK_DIM) { fftsAddr_ = util_get_ffts_config(); }

  ~Ops() {
    // fftsAddr_为配置地址（非aclshmem_malloc分配），无需释放；elementSize_为普通变量，无析构逻辑
  }

  std::string get_name() const { return name_; }

  void put_mem(const at::Tensor &target, const at::Tensor &target_offset, const at::Tensor &src,
               const at::Tensor &src_offset, const at::Tensor &size, const int64_t target_pe) {
    elementSize_ = target.element_size();
    void *target_ptr = const_cast<void *>(target.data_ptr());
    void *target_offset_ptr = const_cast<void *>(target_offset.data_ptr());
    void *src_ptr = const_cast<void *>(src.data_ptr());
    void *src_offset_ptr = const_cast<void *>(src_offset.data_ptr());
    void *size_ptr = const_cast<void *>(size.data_ptr());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    ShmemKernel::aclshmem_put_mem(block_dim_, stream, elementSize_, target_ptr, target_offset_ptr, src_ptr,
                                  src_offset_ptr, size_ptr, target_pe, false);
  }

  void put_mem_signal(const at::Tensor &target, const at::Tensor &target_offset, const at::Tensor &src,
                      const at::Tensor &src_offset, const at::Tensor &size, const at::Tensor &signal,
                      const at::Tensor &signal_offset, const at::Tensor &signal_value, const int64_t signal_op,
                      const int64_t target_pe) {
    elementSize_ = target.element_size();
    void *target_ptr = const_cast<void *>(target.data_ptr());
    void *target_offset_ptr = const_cast<void *>(target_offset.data_ptr());
    void *src_ptr = const_cast<void *>(src.data_ptr());
    void *src_offset_ptr = const_cast<void *>(src_offset.data_ptr());
    void *size_ptr = const_cast<void *>(size.data_ptr());
    void *signal_ptr = const_cast<void *>(signal.data_ptr());
    void *signal_offset_ptr = const_cast<void *>(signal_offset.data_ptr());
    void *signal_value_ptr = const_cast<void *>(signal_value.data_ptr());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    ShmemKernel::aclshmem_put_mem_signal(block_dim_, stream, elementSize_, target_ptr, target_offset_ptr, src_ptr,
                                         src_offset_ptr, size_ptr, signal_ptr, signal_offset_ptr, signal_value_ptr,
                                         signal_op, target_pe, false);
  }

  void get_mem(const at::Tensor &target, const at::Tensor &target_offset, const at::Tensor &src,
               const at::Tensor &src_offset, const at::Tensor &size, const int64_t target_pe) {
    elementSize_ = target.element_size();
    void *target_ptr = const_cast<void *>(target.data_ptr());
    void *target_offset_ptr = const_cast<void *>(target_offset.data_ptr());
    void *src_ptr = const_cast<void *>(src.data_ptr());
    void *src_offset_ptr = const_cast<void *>(src_offset.data_ptr());
    void *size_ptr = const_cast<void *>(size.data_ptr());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    ShmemKernel::aclshmem_get_mem(block_dim_, stream, elementSize_, target_ptr, target_offset_ptr, src_ptr,
                                  src_offset_ptr, size_ptr, target_pe, false);
  }

  void signal_op(const at::Tensor &signal, const at::Tensor &signal_offset, const at::Tensor &signal_value,
                 const int64_t signal_op, const int64_t target_pe) {
    void *signal_ptr = const_cast<void *>(signal.data_ptr());
    void *signal_offset_ptr = const_cast<void *>(signal_offset.data_ptr());
    void *signal_value_ptr = const_cast<void *>(signal_value.data_ptr());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    ShmemKernel::aclshmem_signal_op(stream, signal_ptr, signal_offset_ptr, signal_value_ptr, signal_op, target_pe);
  }

  void signal_wait_until(const at::Tensor &depend_target, const at::Tensor &signal, const at::Tensor &signal_offset,
                         const at::Tensor &compare_value, const int64_t compare_op) {
    elementSize_ = depend_target.element_size();
    void *depend_target_ptr = const_cast<void *>(depend_target.data_ptr());
    void *signal_ptr = const_cast<void *>(signal.data_ptr());
    void *signal_offset_ptr = const_cast<void *>(signal_offset.data_ptr());
    void *compare_value_ptr = const_cast<void *>(compare_value.data_ptr());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    ShmemKernel::aclshmem_signal_wait_until(stream, elementSize_, depend_target_ptr, signal_ptr, signal_offset_ptr,
                                            compare_value_ptr, compare_op);
  }

 private:
  std::string name_;
  uint32_t block_dim_;
  uint64_t fftsAddr_;
  uint64_t elementSize_ = 0;
};

}  // namespace ShmemOps

// register class to TorchScript
static auto registry_common = torch::jit::class_<ShmemOps::Manager>("SymmetricMemory", "Manager")
                                .def(torch::jit::init<>())
                                .def("attr_init", &ShmemOps::Manager::attr_init)
                                .def("finalize", &ShmemOps::Manager::finalize)
                                .def("malloc", &ShmemOps::Manager::malloc_tensor)
                                .def("aligned_malloc", &ShmemOps::Manager::aligned_malloc_tensor)
                                .def("free", &ShmemOps::Manager::free_tensor)
                                .def("get_name", &ShmemOps::Manager::get_name);

static auto registry_ops = torch::jit::class_<ShmemOps::Ops>("SymmetricMemory", "Ops")
                             .def(torch::jit::init<>())
                             .def("put_mem", &ShmemOps::Ops::put_mem)
                             .def("get_mem", &ShmemOps::Ops::get_mem)
                             .def("put_mem_signal", &ShmemOps::Ops::put_mem_signal)
                             .def("signal_op", &ShmemOps::Ops::signal_op)
                             .def("signal_wait_until", &ShmemOps::Ops::signal_wait_until)
                             .def("get_name", &ShmemOps::Ops::get_name);
