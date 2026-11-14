// Copyright 2026 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/pybind.h>

namespace {

at::Tensor detach_saved_tensor(const at::Tensor& tensor) {
  // Match detach's storage and version-counter aliasing without dispatching
  // an operator for metadata that is private to the saved-tensor hook.
  return torch::autograd::make_variable_non_differentiable_view(
      tensor, tensor, /*allow_tensor_metadata_change=*/false);
}

class SavedNativeValue {
 public:
  SavedNativeValue(const at::Tensor& tensor, int64_t slot_index)
      : tensor_(detach_saved_tensor(tensor)),
        version_(tensor_.unsafeGetTensorImpl()->version_counter().current_version()),
        slot_index_(slot_index) {}

  at::Tensor unpack() const {
    TORCH_CHECK(
        tensor_.unsafeGetTensorImpl()->version_counter().current_version() == version_,
        "a tensor saved for backward was modified in place");
    return tensor_;
  }

  const at::Tensor& tensor() const { return tensor_; }
  uint32_t version() const { return version_; }
  int64_t slot_index() const { return slot_index_; }

 private:
  at::Tensor tensor_;
  uint32_t version_;
  int64_t slot_index_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("detach_saved_tensor", &detach_saved_tensor);
  pybind11::class_<SavedNativeValue>(module, "SavedNativeValue")
      .def(pybind11::init<const at::Tensor&, int64_t>())
      .def("unpack", &SavedNativeValue::unpack)
      .def_property_readonly("tensor", &SavedNativeValue::tensor)
      .def_property_readonly("version", &SavedNativeValue::version)
      .def_property_readonly("slot_index", &SavedNativeValue::slot_index);
}
