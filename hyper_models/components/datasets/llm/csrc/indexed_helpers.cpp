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
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

template <typename T>
py::array_t<T> BuildSampleIndex(const py::array_t<int32_t> &sequence_lengths,
                                const py::array_t<int32_t> &document_index, int64_t sequence_length,
                                int64_t num_epochs, int64_t num_tokens_per_epoch,
                                bool drop_last_partial_sequence, bool add_extra_token_to_sequence) {
  const int64_t extra_token = static_cast<int64_t>(add_extra_token_to_sequence);
  const int64_t available_tokens = num_epochs * num_tokens_per_epoch - extra_token;
  const int64_t num_samples = drop_last_partial_sequence
                              ? available_tokens / sequence_length
                              : (available_tokens + sequence_length - 1) / sequence_length;
  py::array_t<T> sample_index(
    {static_cast<py::ssize_t>(num_samples + 1), static_cast<py::ssize_t>(2)});

  auto sequence_length_buffer = sequence_lengths.unchecked<1>();
  auto document_index_buffer = document_index.unchecked<1>();
  auto sample_index_buffer = sample_index.template mutable_unchecked<2>();
  int64_t document_position = 0;
  int64_t document_offset = 0;
  sample_index_buffer(0, 0) = static_cast<T>(document_position);
  sample_index_buffer(0, 1) = static_cast<T>(document_offset);

  for (int64_t sample_position = 1; sample_position <= num_samples; ++sample_position) {
    int64_t remaining_length = sequence_length + extra_token;
    while (true) {
      const int32_t document_id = document_index_buffer(document_position);
      const int64_t document_length = sequence_length_buffer(document_id) - document_offset;
      remaining_length -= document_length;
      if (remaining_length <= 0) {
        document_offset += remaining_length + document_length - extra_token;
        break;
      }
      if (document_position == document_index_buffer.shape(0) - 1) {
        if (sample_position != num_samples) {
          throw py::value_error("The final partial sample was reached before the last sample index");
        }
        document_offset = sequence_length_buffer(document_id) - extra_token;
        break;
      }
      ++document_position;
      document_offset = 0;
    }
    sample_index_buffer(sample_position, 0) = static_cast<T>(document_position);
    sample_index_buffer(sample_position, 1) = static_cast<T>(document_offset);
  }
  return sample_index;
}

void BuildBlendingIndices(py::array_t<int16_t> &dataset_index, py::array_t<int64_t> &dataset_sample_index,
                          const py::array_t<double> &weights) {
  auto dataset_index_buffer = dataset_index.mutable_unchecked<1>();
  auto dataset_sample_index_buffer = dataset_sample_index.mutable_unchecked<1>();
  const auto weight_buffer = weights.unchecked<1>();
  const int64_t num_datasets = weight_buffer.shape(0);
  const int64_t size = dataset_index_buffer.shape(0);
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
}

}  // namespace

PYBIND11_MODULE(_indexed_helpers_cpp, module) {
  module.doc() = "Native indexed Dataset helper functions";
  module.def("build_sample_index_int32", &BuildSampleIndex<int32_t>, py::arg("sequence_lengths"),
             py::arg("document_index"), py::arg("sequence_length"), py::arg("num_epochs"),
             py::arg("num_tokens_per_epoch"), py::arg("drop_last_partial_sequence"),
             py::arg("add_extra_token_to_sequence"));
  module.def("build_sample_index_int64", &BuildSampleIndex<int64_t>, py::arg("sequence_lengths"),
             py::arg("document_index"), py::arg("sequence_length"), py::arg("num_epochs"),
             py::arg("num_tokens_per_epoch"), py::arg("drop_last_partial_sequence"),
             py::arg("add_extra_token_to_sequence"));
  module.def("build_blending_indices", &BuildBlendingIndices, py::arg("dataset_index"),
             py::arg("dataset_sample_index"), py::arg("weights"));
}
