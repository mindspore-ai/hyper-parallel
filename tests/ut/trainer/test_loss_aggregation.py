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
"""Token counting across microbatches and modality families."""

import unittest

import torch

from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token


class TestCountLossToken(unittest.TestCase):
    """A batch collection counts the same targets as its individual batches."""

    def test_token_families_accumulate_across_nested_batches(self) -> None:
        """Feature: Multimodal batch token counts.
        Description: Count nested microbatches with overlapping token families.
        Expectation: Each family includes all contributions, independent of batch order.
        """
        batches = [
            {
                "shift_labels": torch.tensor([[1, -100, 3]]),
                "foundation_labels": torch.tensor([4, -100]),
                "image_labels": torch.tensor([1, 2, -100]),
                "audio_labels": torch.tensor([-100, 1]),
                "image_decoder_labels": torch.tensor([1, -100]),
                "image_output_mask": torch.tensor([1, 0]),
            },
            {
                "labels": torch.tensor([[5, 6, 7]]),
                "image_labels": torch.tensor([3, -100]),
                "audio_labels": torch.tensor([2, -100]),
                "image_decoder_labels": torch.tensor([2, 3]),
                "image_output_mask": torch.tensor([0, 1]),
            },
        ]
        expected = {"foundation_tokens": 5, "image_tokens": 3, "audio_tokens": 2, "image_decoder_tokens": 5}
        for collection in (batches, (batches[1], [batches[0]])):
            with self.subTest(collection=type(collection).__name__):
                actual = count_loss_token(collection)
                self.assertEqual({name: value.item() for name, value in actual.items()}, expected)
                for name, value in actual.items():
                    torch.testing.assert_close(value, sum(count_loss_token(batch).get(name, 0) for batch in batches))

    def test_padding_batches_are_skipped(self) -> None:
        """Feature: Padding-only microbatches.
        Description: Mix padding markers with real batches, then count only padding.
        Expectation: Padding contributes nothing and requires no target fields.
        """
        padding = {"padding_flag": True}
        actual = count_loss_token([padding, ({"labels": torch.tensor([[1, 2, -100]])}, padding)])
        self.assertEqual({name: value.item() for name, value in actual.items()},
                         {"foundation_tokens": 1, "image_decoder_tokens": 0})
        empty = count_loss_token([padding, [padding]])
        self.assertEqual({name: value.item() for name, value in empty.items()},
                         {"foundation_tokens": 0, "image_decoder_tokens": 0})

    def test_invalid_batch_type_is_rejected(self) -> None:
        """Feature: Batch traversal contract.
        Description: Include a non-batch object inside a collection.
        Expectation: Invalid entries still raise a type error instead of disappearing.
        """
        with self.assertRaisesRegex(TypeError, "Unsupported batch type"):
            count_loss_token([None])
