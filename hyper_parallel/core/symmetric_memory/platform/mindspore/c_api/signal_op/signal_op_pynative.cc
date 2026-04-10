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
class SignalOpRunner : public ms::pynative::PyboostRunner {
 public:
  using ms::pynative::PyboostRunner::PyboostRunner;

  virtual ~SignalOpRunner() = default;

  void SetTargetPe(const int64_t &target_pe) { this->target_pe_ = target_pe; }
  void SetSignalOp(const int64_t &signal_op) { this->signal_op_ = signal_op; }

 protected:
  void LaunchKernel() override {
    auto op_name = this->op_name();
    MS_LOG(DEBUG) << "Launch " << op_name << " start";
    void *signal_ptr = const_cast<void *>(this->inputs()[0].GetDataPtr());
    void *signal_offset_ptr = const_cast<void *>(this->inputs()[1].GetDataPtr());
    void *signal_value_ptr = const_cast<void *>(this->inputs()[2].GetDataPtr());
    ShmemKernel::aclshmem_signal_op(this->stream(), signal_ptr, signal_offset_ptr, signal_value_ptr, this->signal_op_,
                                    this->target_pe_);
    MS_LOG(DEBUG) << "Launch " << op_name << " end";
  }

 private:
  int64_t target_pe_{0};
  int64_t signal_op_{0};
};

void npu_signal_op(const ms::Tensor &signal, const ms::Tensor &signal_offset, const ms::Tensor &signal_value,
                   const int64_t signal_op, const int64_t target_pe) {
  auto op_name = "SignalOp";
  auto runner = std::make_shared<ms_custom_ops::SignalOpRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);
  runner->SetSignalOp(signal_op);
  runner->SetTargetPe(target_pe);
  std::vector<ms::Tensor> inputs = {signal, signal_offset, signal_value};
  std::vector<ms::Tensor> outputs = {};
  runner->Run(inputs, outputs);
  return;
}
}  // namespace ms_custom_ops

auto pyboost_signal_op(const ms::Tensor &signal, const ms::Tensor &signal_offset, const ms::Tensor &signal_value,
                       const int64_t signal_op, const int64_t target_pe) {
  return ms::pynative::PyboostRunner::Call<0>(ms_custom_ops::npu_signal_op, signal, signal_offset, signal_value,
                                              signal_op, target_pe);  // 0 represent output num = 0
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("signal_op", &pyboost_signal_op, "Signal Op", pybind11::arg("signal"), pybind11::arg("signal_offset"),
        pybind11::arg("signal_value"), pybind11::arg("signal_op"), pybind11::arg("target_pe"));
}
