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
"""Granularity-based padding for the Kimi Omni transform.

The transform historically padded every sample to the fixed ``max_seq_len``, so
a micro-batch of short samples still paid ``max_seq_len`` worth of compute.
``pad_granularity`` lets the transform stop at the next multiple of a chosen
alignment. Loss semantics are unchanged: padded label slots stay
``IGNORE_INDEX``.

The historical fixed-length micro-batch collator (``VLMCollator(pad_to_batch=True)``)
is superseded by the merged Omni packing path: ``SamplePacker`` concatenates whole
samples up to the token budget and pads only the packed window, so no per-batch
``pad_to_batch`` switch exists. The transform-side padding tested here is what
survives.
"""
# pylint: disable=wrong-import-position

import importlib.util
import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch

from tests.common.mark_utils import arg_mark

# The VLM facade pulls transformers.AutoProcessor, whose lazy import chain ends
# in torchvision; some CI pythons lack liblzma and cannot import it.
_HAS_LZMA = importlib.util.find_spec("_lzma") is not None

from hyper_parallel.data.constants import IGNORE_INDEX


def _sample(seq_len: int) -> dict:
    """Build one pre-collation sample whose text fields are ``seq_len`` long."""
    return {
        "input_ids": torch.arange(seq_len),
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "labels": torch.zeros(seq_len, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(seq_len, dtype=torch.long),
        "pixel_values": torch.randn(3, 4),
    }


@unittest.skipIf(not _HAS_LZMA, "python build lacks liblzma (_lzma)")
class TestPadTarget(unittest.TestCase):
    """The transform stops padding at the granularity when one is configured."""

    @staticmethod
    def _transform(**kwargs):
        from hyper_parallel.data.omni.kimi_transform import KimiVLMChatTransform

        return KimiVLMChatTransform(processor=object(), max_seq_len=8192, **kwargs)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_without_granularity_pads_to_max_seq_len(self):
        """
        Feature: fixed-length padding contract.
        Description: Ask the transform for the pad target without a granularity.
        Expectation: The target is max_seq_len whatever the sample length.
        """
        transform = self._transform()

        self.assertEqual(transform._pad_target(3000), 8192)
        self.assertEqual(transform._pad_target(8192), 8192)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_granularity_rounds_up_and_caps_at_max_seq_len(self):
        """
        Feature: granularity-based padding.
        Description: Ask the transform for the pad target with pad_granularity=128.
        Expectation: The target is the next 128-multiple, capped at max_seq_len.
        """
        transform = self._transform(pad_granularity=128)

        self.assertEqual(transform._pad_target(1), 128)
        self.assertEqual(transform._pad_target(128), 128)
        self.assertEqual(transform._pad_target(129), 256)
        self.assertEqual(transform._pad_target(3000), 3072)
        self.assertEqual(transform._pad_target(8189), 8192)
        self.assertEqual(transform._pad_target(9000), 8192)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_truncate_and_pad_pads_all_sequence_fields(self):
        """
        Feature: sequence-aligned padding of one sample.
        Description: Pad a 300-token sample with pad_granularity=128.
        Expectation: Every sequence field grows to 384 and labels pad with IGNORE_INDEX.
        """
        from hyper_parallel.data.omni.kimi_transform import _SEQ_FIELDS

        transform = self._transform(pad_granularity=128)
        out = transform._truncate_and_pad(_sample(300))

        for field in _SEQ_FIELDS:
            self.assertEqual(int(out[field].shape[0]), 384, field)
        self.assertTrue(torch.equal(out["labels"][:300], torch.zeros(300, dtype=torch.long)))
        self.assertTrue((out["labels"][300:] == IGNORE_INDEX).all())
        self.assertEqual(int(out["pixel_values"].shape[0]), 3)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_bad_granularity(self):
        """
        Feature: padding-option validation.
        Description: Build the transform with a non-positive or non-integer granularity.
        Expectation: ValueError naming a positive integer.
        """
        for bad in (0, -8, 1.5, True):
            with self.subTest(granularity=bad):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self._transform(pad_granularity=bad)


if __name__ == "__main__":
    unittest.main()
