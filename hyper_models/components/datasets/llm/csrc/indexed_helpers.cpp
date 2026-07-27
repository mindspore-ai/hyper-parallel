// Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
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

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

void BuildBlendingIndices(py::array_t<int16_t> &dataset_index, py::array_t<int64_t> &dataset_sample_index,
                          const py::array_t<double> &weights, int32_t num_datasets, int64_t size, bool verbose) {
  if (verbose) {
    std::cout << "> building indices for blended datasets ..." << std::endl;
  }

  auto dataset_index_buffer = dataset_index.mutable_unchecked<1>();
  auto dataset_sample_index_buffer = dataset_sample_index.mutable_unchecked<1>();
  auto weight_buffer = weights.unchecked<1>();
  std::vector<int64_t> current_samples(num_datasets, 0);

  for (int64_t sample_index = 0; sample_index < size; ++sample_index) {
    const double sample_position = std::max(static_cast<double>(sample_index), 1.0);
    int64_t maximum_error_index = 0;
    double maximum_error = weight_buffer[0] * sample_position - static_cast<double>(current_samples[0]);
    for (int64_t dataset_id = 1; dataset_id < num_datasets; ++dataset_id) {
      const double sampling_error =
        weight_buffer[dataset_id] * sample_position - static_cast<double>(current_samples[dataset_id]);
      if (sampling_error > maximum_error) {
        maximum_error = sampling_error;
        maximum_error_index = dataset_id;
      }
    }

    dataset_index_buffer[sample_index] = static_cast<int16_t>(maximum_error_index);
    dataset_sample_index_buffer[sample_index] = current_samples[maximum_error_index];
    current_samples[maximum_error_index] += 1;
  }

  if (verbose) {
    std::cout << " > sample ratios:" << std::endl;
    for (int64_t dataset_id = 0; dataset_id < num_datasets; ++dataset_id) {
      const double achieved_ratio = static_cast<double>(current_samples[dataset_id]) / static_cast<double>(size);
      std::cout << "   dataset " << dataset_id << ", input: " << weight_buffer[dataset_id]
                << ", achieved: " << achieved_ratio << std::endl;
    }
  }
}

}  // namespace

PYBIND11_MODULE(_indexed_helpers_cpp, module) {
  module.doc() = "Native indexed Dataset helper functions";
  module.def("build_blending_indices", &BuildBlendingIndices, py::arg("dataset_index"), py::arg("dataset_sample_index"),
             py::arg("weights"), py::arg("num_datasets"), py::arg("size"), py::arg("verbose") = false);
}
