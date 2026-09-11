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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace py = pybind11;

namespace {

template <typename T>
py::array_t<T> build_sample_idx(const py::array_t<int32_t> &sequence_lengths,
                               const py::array_t<int32_t> &document_index, int64_t sequence_length,
                               int64_t num_epochs, int64_t num_tokens_per_epoch,
                               bool drop_last_partial_sequence, bool add_extra_token_to_sequence) {
  // Sample index is used for GPT-like datasets whose documents are flattened into
  // a 1-D array. It has shape [number_of_samples + 1, 2], where column 0 contains
  // the index into document_index and column 1 is the starting offset in that document.

  // Consistency checks.
  if (sequence_length <= 0 || num_epochs < 0 || num_tokens_per_epoch < 0) {
    throw py::value_error("Sequence length must be positive and token/epoch counts nonnegative");
  }

  const int64_t extra_token = static_cast<int64_t>(add_extra_token_to_sequence);
  const int64_t available_tokens = num_epochs * num_tokens_per_epoch - extra_token;
  if (available_tokens < 0) {
    throw py::value_error("Token count must cover the extra token");
  }

  const int64_t num_samples =
    drop_last_partial_sequence ? available_tokens / sequence_length
                               : (available_tokens + sequence_length - 1) / sequence_length;
  py::array_t<T> sample_index(
    {static_cast<py::ssize_t>(num_samples + 1), static_cast<py::ssize_t>(2)});

  // Get the pointer access without the checks. Validate document IDs before sampling.
  auto sequence_length_buffer = sequence_lengths.unchecked<1>();
  auto document_index_buffer = document_index.unchecked<1>();
  if (num_samples > 0 && document_index_buffer.shape(0) == 0) {
    throw py::value_error("Document index must not be empty when building samples");
  }

  // Validate once so repeated samples from the same document keep unchecked access.
  int32_t minimum_document_id = 0;
  int32_t maximum_document_id = -1;
  for (py::ssize_t position = 0; position < document_index_buffer.shape(0); ++position) {
    minimum_document_id = std::min(minimum_document_id, document_index_buffer(position));
    maximum_document_id = std::max(maximum_document_id, document_index_buffer(position));
  }
  if (minimum_document_id < 0 || maximum_document_id >= sequence_length_buffer.shape(0)) {
    throw py::value_error("Document ID is outside sequence_lengths");
  }

  auto sample_index_buffer = sample_index.template mutable_unchecked<2>();

  // Start with the first document and no offset.
  int64_t document_position = 0;
  int64_t document_offset = 0;
  sample_index_buffer(0, 0) = static_cast<T>(document_position);
  sample_index_buffer(0, 1) = static_cast<T>(document_offset);

  for (int64_t sample_position = 1; sample_position <= num_samples; ++sample_position) {
    // Start with a fresh sequence.
    int64_t remaining_length = sequence_length + extra_token;
    while (true) {
      // Get the document length and add it to the current sequence.
      const int32_t document_id = document_index_buffer(document_position);
      const int64_t document_length = sequence_length_buffer(document_id) - document_offset;
      remaining_length -= document_length;

      // If we have a full sequence, adjust the offset and leave the loop.
      // Subtract the extra token so consecutive samples overlap for next-token labels.
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

      // Otherwise, start from the beginning of the next document.
      ++document_position;
      document_offset = 0;
    }

    // Record the sequence.
    sample_index_buffer(sample_position, 0) = static_cast<T>(document_position);
    sample_index_buffer(sample_position, 1) = static_cast<T>(document_offset);
  }

  return sample_index;
}

void build_blending_indices(py::array_t<int16_t, 0> &dataset_index,
                            py::array_t<int64_t, 0> &dataset_sample_index,
                            const py::array_t<double> &weights) {
  // Given multiple datasets and a weighting array, build samples following those weights.

  // Get the pointer access without the checks.
  auto dataset_index_buffer = dataset_index.mutable_unchecked<1>();
  auto dataset_sample_index_buffer = dataset_sample_index.mutable_unchecked<1>();
  const auto weight_buffer = weights.unchecked<1>();
  const int64_t num_datasets = weight_buffer.shape(0);
  const int64_t size = dataset_index_buffer.shape(0);

  // Dataset IDs start at zero and must fit in the int16_t output array.
  constexpr int64_t kMaxDatasets =
    static_cast<int64_t>(std::numeric_limits<int16_t>::max()) + 1;
  if (num_datasets <= 0 || num_datasets > kMaxDatasets) {
    throw py::value_error("Weights must be nonempty and dataset IDs must fit in int16_t");
  }
  if (dataset_sample_index_buffer.shape(0) != size) {
    throw py::value_error("Blend output arrays must have the same length");
  }

  // Initialize buffer for number of samples used for each dataset.
  std::vector<int64_t> current_samples(num_datasets, 0);

  for (int64_t sample_index = 0; sample_index < size; ++sample_index) {
    // Determine where the maximum error in sampling is happening.
    const double sample_position = std::max(static_cast<double>(sample_index), 1.0);
    int64_t maximum_error_index = 0;
    double maximum_error =
      weight_buffer[0] * sample_position - static_cast<double>(current_samples[0]);
    for (int64_t dataset_id = 1; dataset_id < num_datasets; ++dataset_id) {
      const double sampling_error =
        weight_buffer[dataset_id] * sample_position - static_cast<double>(current_samples[dataset_id]);
      if (sampling_error > maximum_error) {
        maximum_error = sampling_error;
        maximum_error_index = dataset_id;
      }
    }

    // Populate the indices.
    dataset_index_buffer[sample_index] = static_cast<int16_t>(maximum_error_index);
    dataset_sample_index_buffer[sample_index] = current_samples[maximum_error_index];

    // Update the total samples.
    current_samples[maximum_error_index] += 1;
  }
}

}  // namespace

PYBIND11_MODULE(_indexed_helpers_cpp, module) {
  module.doc() = "Native indexed Dataset helper functions";

  module.def("build_sample_index_int32", &build_sample_idx<int32_t>, py::arg("sequence_lengths"),
             py::arg("document_index"), py::arg("sequence_length"), py::arg("num_epochs"),
             py::arg("num_tokens_per_epoch"), py::arg("drop_last_partial_sequence"),
             py::arg("add_extra_token_to_sequence"));

  module.def("build_sample_index_int64", &build_sample_idx<int64_t>, py::arg("sequence_lengths"),
             py::arg("document_index"), py::arg("sequence_length"), py::arg("num_epochs"),
             py::arg("num_tokens_per_epoch"), py::arg("drop_last_partial_sequence"),
             py::arg("add_extra_token_to_sequence"));

  module.def("build_blending_indices", &build_blending_indices, py::arg("dataset_index").noconvert(),
             py::arg("dataset_sample_index").noconvert(), py::arg("weights"));
}
