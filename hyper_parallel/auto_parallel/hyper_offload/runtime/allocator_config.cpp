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
// ============================================================================
#include <pybind11/pybind11.h>

#if __has_include(<c10/core/AllocatorConfig.h>)
#include <c10/core/AllocatorConfig.h>
#endif

namespace {

pybind11::object expandable_segments() {
#if __has_include(<c10/core/AllocatorConfig.h>)
  return pybind11::bool_(c10::CachingAllocator::AcceleratorAllocatorConfig::use_expandable_segments());
#else
  return pybind11::none();
#endif
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("expandable_segments", &expandable_segments);
}
