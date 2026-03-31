/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include <vector>
#include <memory>

#include "framework/module.h"
#include "shmem.h"
#include "shmem_kernel.h"

namespace ms_custom_ops {
class PutMemSignalRunner : public ms::pynative::PyboostRunner {
 public:
  using ms::pynative::PyboostRunner::PyboostRunner;

  virtual ~PutMemSignalRunner() = default;

  void SetTargetPe(const int64_t &target_pe) { this->target_pe_ = target_pe; }
  void SetSignalOp(const int64_t &signal_op) { this->signal_op_ = signal_op; }

  int TypeIdToElementSize(mindspore::TypeId type_id) {
    static const std::unordered_map<mindspore::TypeId, int> type_size_map = {
      {mindspore::kNumberTypeBool, 1},       {mindspore::kNumberTypeInt8, 1},
      {mindspore::kNumberTypeUInt8, 1},      {mindspore::kNumberTypeFloat8E4M3FN, 1},
      {mindspore::kNumberTypeFloat8E5M2, 1}, {mindspore::kNumberTypeHiFloat8, 1},

      {mindspore::kNumberTypeInt16, 2},      {mindspore::kNumberTypeUInt16, 2},
      {mindspore::kNumberTypeFloat16, 2},    {mindspore::kNumberTypeBFloat16, 2},

      {mindspore::kNumberTypeInt, 4},        {mindspore::kNumberTypeUInt, 4},
      {mindspore::kNumberTypeInt32, 4},      {mindspore::kNumberTypeUInt32, 4},
      {mindspore::kNumberTypeFloat, 4},      {mindspore::kNumberTypeFloat32, 4},

      {mindspore::kNumberTypeInt64, 8},      {mindspore::kNumberTypeUInt64, 8},
      {mindspore::kNumberTypeFloat64, 8},    {mindspore::kNumberTypeDouble, 8},
      {mindspore::kNumberTypeComplex, 8},    {mindspore::kNumberTypeComplex64, 8}};

    auto iter = type_size_map.find(type_id);
    if (iter != type_size_map.end()) {
      return iter->second;
    }

    std::cerr << "Error: TypeId " << type_id << " is not a valid Number Type, cannot convert to element_size!"
              << std::endl;
    return -1;
  }

 protected:
  void LaunchKernel() override {
    auto op_name = this->op_name();
    MS_LOG(DEBUG) << "Launch " << op_name << " start";
    uint32_t block_dim_ = 4;
    uint64_t element_size_ = TypeIdToElementSize(this->inputs()[0].data_type());
    void *target_ptr = const_cast<void *>(this->inputs()[0].GetDataPtr());
    void *target_offset_ptr = const_cast<void *>(this->inputs()[1].GetDataPtr());
    void *src_ptr = const_cast<void *>(this->inputs()[2].GetDataPtr());
    void *src_offset_ptr = const_cast<void *>(this->inputs()[3].GetDataPtr());
    void *size_ptr = const_cast<void *>(this->inputs()[4].GetDataPtr());
    void *signal_ptr = const_cast<void *>(this->inputs()[5].GetDataPtr());
    void *signal_offset_ptr = const_cast<void *>(this->inputs()[6].GetDataPtr());
    void *signal_value_ptr = const_cast<void *>(this->inputs()[7].GetDataPtr());
    ShmemKernel::aclshmem_put_mem_signal(block_dim_, this->stream(), element_size_, target_ptr, target_offset_ptr,
                                         src_ptr, src_offset_ptr, size_ptr, signal_ptr, signal_offset_ptr,
                                         signal_value_ptr, this->signal_op_, this->target_pe_, true);
    MS_LOG(DEBUG) << "Launch " << op_name << " end";
  }

 private:
  int64_t target_pe_{0};
  int64_t signal_op_{0};
};

void npu_put_mem_signal(const ms::Tensor &target, const ms::Tensor &target_offset, const ms::Tensor &src,
                        const ms::Tensor &src_offset, const ms::Tensor &size, const ms::Tensor &signal,
                        const ms::Tensor &signal_offset, const ms::Tensor &signal_value, const int64_t signal_op,
                        const int64_t target_pe) {
  auto op_name = "PutMemSignal";
  auto runner = std::make_shared<ms_custom_ops::PutMemSignalRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);
  runner->SetSignalOp(signal_op);
  runner->SetTargetPe(target_pe);
  std::vector<ms::Tensor> inputs = {target, target_offset, src, src_offset, size, signal, signal_offset, signal_value};
  std::vector<ms::Tensor> outputs = {};
  runner->Run(inputs, outputs);
  return;
}
}  // namespace ms_custom_ops

auto pyboost_put_mem_signal(const ms::Tensor &target, const ms::Tensor &target_offset, const ms::Tensor &src,
                            const ms::Tensor &src_offset, const ms::Tensor &size, const ms::Tensor &signal,
                            const ms::Tensor &signal_offset, const ms::Tensor &signal_value, const int64_t signal_op,
                            const int64_t target_pe) {
  return ms::pynative::PyboostRunner::Call<0>(ms_custom_ops::npu_put_mem_signal, target, target_offset, src, src_offset,
                                              size, signal, signal_offset, signal_value, signal_op,
                                              target_pe);  // 0 represent output num = 0
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("put_mem_signal", &pyboost_put_mem_signal, "Put Mem Signal", pybind11::arg("target"),
        pybind11::arg("target_offset"), pybind11::arg("src"), pybind11::arg("src_offset"), pybind11::arg("size"),
        pybind11::arg("signal"), pybind11::arg("signal_offset"), pybind11::arg("signal_value"),
        pybind11::arg("signal_op"), pybind11::arg("target_pe"));
}
