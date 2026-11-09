# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Unit tests for Omni sample workload metadata."""

import unittest

import torch

from hyper_parallel.data.omni.metadata import vlm_sample_metadata


class TestVlmSampleMetadata(unittest.TestCase):
    """Validate model-neutral image-text workload extraction."""

    def test_uses_raw_image_grid_and_keeps_sample_unchanged(self) -> None:
        """Feature: Omni workload metadata.
        Description: Extract image grids and token lengths from a raw sample.
        Expectation: Encoder and LLM costs are correct and input tensors remain unchanged.
        """
        sample = {
            "input_ids": torch.arange(8),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 1, 3]], dtype=torch.int64),
            "pixel_values": torch.zeros((7, 4)),
        }
        original = {key: value.clone() for key, value in sample.items()}

        metadata = vlm_sample_metadata(sample)

        self.assertEqual(metadata.pack_tokens, 8)
        self.assertEqual(metadata.cost.encoder, 7)
        self.assertEqual(metadata.cost.llm, 8)
        for key, value in original.items():
            torch.testing.assert_close(sample[key], value)

    def test_rejects_batched_or_mismatched_modality_fields(self) -> None:
        """Feature: Omni metadata validation.
        Description: Provide batched tokens or mismatched image patches.
        Expectation: Invalid samples fail before planning or payload movement.
        """
        sample = {
            "input_ids": torch.ones(8),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
            "pixel_values": torch.zeros((4, 4)),
        }
        with self.assertRaisesRegex(ValueError, "1-D"):
            vlm_sample_metadata({**sample, "input_ids": torch.ones((1, 8))})
        with self.assertRaisesRegex(ValueError, "patch count"):
            vlm_sample_metadata({**sample, "pixel_values": torch.zeros((3, 4))})
