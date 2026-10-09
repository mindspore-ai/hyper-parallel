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
"""Unit tests for model-neutral Omni collation and forwarding."""

import unittest

import torch

from hyper_parallel.data.batching.build_collate_fn import OmniCollator
from hyper_parallel.data.batching.get_batch import OmniParallelBatch
from tests.common.mark_utils import arg_mark


class TestOmniCollator(unittest.TestCase):
    """Omni collation keeps generic model fields without an allowlist."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_collator_stacks_tokens_and_concatenates_modality_values(self):
        """Collate fixed token fields and variable image rows.

        Feature: Generic Omni collation.
        Description: Collate two samples with different image-row counts.
        Expectation: Tokens stack by sample and image rows concatenate in order.
        """
        collator = OmniCollator()

        batch = collator([
            {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([-100, 2]),
                "pixel_values": torch.tensor([[1.0, 2.0]]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([-100, 5]),
                "pixel_values": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
            },
        ])

        torch.testing.assert_close(batch["input_ids"], torch.tensor([[1, 2], [4, 5]]))
        torch.testing.assert_close(
            batch["pixel_values"],
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_batch_runtime_forwards_unknown_model_fields(self):
        """Forward model-specific fields without a framework allowlist.

        Feature: Omni model-input forwarding.
        Description: Split one batch containing an unknown modality route.
        Expectation: The route reaches model inputs while loss metadata stays separate.
        """
        model_inputs, loss_inputs = OmniParallelBatch._split_model_and_loss_inputs({
            "input_ids": torch.tensor([[1, 2]]),
            "labels": torch.tensor([[-100, 2]]),
            "loss_mask": torch.tensor([[False, True]]),
            "custom_modality_route": torch.tensor([[3, 4]]),
        })

        self.assertIn("custom_modality_route", model_inputs)
        self.assertNotIn("loss_mask", model_inputs)
        self.assertEqual(set(loss_inputs), {"labels", "loss_mask"})


if __name__ == "__main__":
    unittest.main()
