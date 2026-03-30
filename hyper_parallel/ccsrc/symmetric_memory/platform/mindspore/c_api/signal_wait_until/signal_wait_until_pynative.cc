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
class SignalWaitUntilRunner : public ms::pynative::PyboostRunner {
 public:
  using ms::pynative::PyboostRunner::PyboostRunner;

  virtual ~SignalWaitUntilRunner() = default;

  void SetCompareOp(const int64_t &compare_op) { this->compare_op_ = compare_op; }

 protected:
  void LaunchKernel() override {
    auto op_name = this->op_name();
    MS_LOG(DEBUG) << "Launch " << op_name << " start";
    uint64_t element_size_ = 1;
    void *depend_target_ptr = const_cast<void *>(this->inputs()[0].GetDataPtr());
    void *signal_ptr = const_cast<void *>(this->inputs()[1].GetDataPtr());
    void *signal_offset_ptr = const_cast<void *>(this->inputs()[2].GetDataPtr());
    void *compare_value_ptr = const_cast<void *>(this->inputs()[3].GetDataPtr());
    ShmemKernel::aclshmem_signal_wait_until(this->stream(), element_size_, depend_target_ptr, signal_ptr,
                                            signal_offset_ptr, compare_value_ptr, this->compare_op_);
    MS_LOG(DEBUG) << "Launch " << op_name << " end";
  }

 private:
  int64_t compare_op_{0};
};

void npu_signal_wait_until(const ms::Tensor &depend_target, const ms::Tensor &signal, const ms::Tensor &signal_offset,
                           const ms::Tensor &compare_value, const int64_t compare_op) {
  auto op_name = "SignalWaitUntil";
  auto runner = std::make_shared<ms_custom_ops::SignalWaitUntilRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);
  runner->SetCompareOp(compare_op);
  std::vector<ms::Tensor> inputs = {depend_target, signal, signal_offset, compare_value};
  std::vector<ms::Tensor> outputs = {};
  runner->Run(inputs, outputs);
  return;
}
}  // namespace ms_custom_ops

auto pyboost_signal_wait_until(const ms::Tensor &depend_target, const ms::Tensor &signal,
                               const ms::Tensor &signal_offset, const ms::Tensor &compare_value,
                               const int64_t compare_op) {
  return ms::pynative::PyboostRunner::Call<0>(ms_custom_ops::npu_signal_wait_until, depend_target, signal,
                                              signal_offset, compare_value, compare_op);  // 0 represent output num = 0
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("signal_wait_until", &pyboost_signal_wait_until, "Wait Signal Until", pybind11::arg("depend_target"),
        pybind11::arg("signal"), pybind11::arg("signal_offset"), pybind11::arg("compare_value"),
        pybind11::arg("compare_op"));
}
