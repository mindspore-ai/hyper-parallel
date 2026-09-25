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
"""``truncate_mode`` for the Kimi VLM transform.

The supervised region (the assistant turn) sits at the *end* of the token
stream, so the historical pure head cut keeps the prompt and images and throws
the whole response away as soon as they fill ``max_seq_len``: the sample reaches
the loss with zero supervised tokens and the step trains on nothing. Measured on
the mock dataset the untruncated stream is 13934 tokens, 5131 of them the
response, and at ``max_seq_len=8192`` head truncation keeps 8192 prompt/image
tokens and 0 supervised ones.

``truncate_mode="proportional"`` (the new default) ports MindSpeed-MM's
``infer_seqlen``: the budget is split between the prompt and the response, so
the response head survives a truncation. ``truncate_mode="head"`` keeps the old
behaviour for configs tuned against it. These tests pin the split arithmetic and
the rebuilt truncation on synthetic samples -- no processor is involved.
"""
# pylint: disable=wrong-import-position

import importlib.util
import os
import unittest
from typing import Optional

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch

from tests.common.mark_utils import arg_mark

# The VLM package facade pulls transformers.AutoProcessor, whose lazy import chain
# ends in torchvision; some CI pythons lack liblzma and cannot import it.
_HAS_LZMA = importlib.util.find_spec("_lzma") is not None

from hyper_parallel.data.constants import IGNORE_INDEX


def _sample(seq_len: int, supervised_from: Optional[int] = None) -> dict:
    """Build one pre-collation sample supervised from ``supervised_from`` on.

    ``input_ids``/``labels`` carry their own position so a truncated slice can be
    traced back to the original stream. ``supervised_from=None`` leaves the whole
    sample unsupervised.
    """
    input_ids = torch.arange(seq_len)
    labels = input_ids.clone()
    labels[:seq_len if supervised_from is None else supervised_from] = IGNORE_INDEX
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(seq_len, dtype=torch.long),
        "labels": labels,
        "mm_token_type_ids": torch.zeros(seq_len, dtype=torch.long),
        "pixel_values": torch.zeros(0, 4),
        "image_grid_thw": torch.zeros((0, 3), dtype=torch.long),
    }


def _run(transform, sample: dict) -> dict:
    """Run ``_truncate_and_pad`` on a copy: it rebinds the sample's fields in place."""
    return transform._truncate_and_pad({key: value.clone() for key, value in sample.items()})


@unittest.skipIf(not _HAS_LZMA, "python build lacks liblzma (_lzma)")
class TestInferSeqlen(unittest.TestCase):
    """The MindSpeed-MM budget split, one case per branch."""

    @staticmethod
    def _infer_seqlen():
        from hyper_parallel.models.kimi_k26.adapter.data.transform_fn import _infer_seqlen

        return _infer_seqlen

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_small_response_keeps_everything_it_needs(self):
        """``target_len * 2 < cutoff_len``: the response is not cut."""
        self.assertEqual(self._infer_seqlen()(900, 100, 1000), (900, 100))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_small_prompt_gives_the_rest_to_the_response(self):
        """``source_len * 2 < cutoff_len``: target gets ``cutoff_len - source_len``."""
        self.assertEqual(self._infer_seqlen()(200, 900, 1000), (200, 800))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_both_large_splits_proportionally(self):
        """Both sides large: the budget is split by size, truncated with ``int()``."""
        # The real mock-dataset shape: 8803 prompt tokens, 5131 response tokens.
        self.assertEqual(self._infer_seqlen()(8803, 5131, 8192), (5176, 3016))
        # Pins the ``int()`` truncation of 8192 * 5131/13934 = 3016.3 -> 3016.
        self.assertEqual(int(8192 * (5131 / 13934)), 3016)


@unittest.skipIf(not _HAS_LZMA, "python build lacks liblzma (_lzma)")
class TestTruncateMode(unittest.TestCase):
    """Truncation keeps the response head under ``"proportional"``."""

    @staticmethod
    def _transform(truncate_mode: str = "proportional", **kwargs):
        from hyper_parallel.models.kimi_k26.adapter.data.transform_fn import KimiVLMChatTransform

        return KimiVLMChatTransform(processor=object(), max_seq_len=600,
                                    truncate_mode=truncate_mode, **kwargs)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_proportional_keeps_the_head_of_the_response(self):
        """Half the sample is supervised: the cut lands inside the response."""
        from hyper_parallel.models.kimi_k26.adapter.data.transform_fn import _infer_seqlen

        transform = self._transform("proportional")
        sample = _sample(1000, supervised_from=500)
        out = _run(transform, sample)

        new_source_len, new_target_len = _infer_seqlen(500, 500, 600)
        self.assertEqual((new_source_len, new_target_len), (300, 300))
        self.assertEqual(int(out["input_ids"].shape[0]), 600)
        self.assertEqual(int((out["labels"] != IGNORE_INDEX).sum()), new_target_len)
        self.assertGreater(int((out["labels"] != IGNORE_INDEX).sum()), 0)

        # The surviving supervised tokens are the *first* new_target_len of the
        # response, not its tail.
        kept = out["labels"][new_source_len:]
        self.assertTrue(torch.equal(kept, sample["labels"][500:500 + new_target_len]))
        self.assertEqual(int(kept[0]), 500)
        self.assertFalse(torch.equal(kept, sample["labels"][-new_target_len:]))
        # Both halves of the rebuilt sample are contiguous slices of the original.
        self.assertTrue(torch.equal(out["input_ids"][:new_source_len], torch.arange(300)))
        self.assertTrue(torch.equal(out["input_ids"][new_source_len:], torch.arange(500, 800)))
        self.assertTrue((out["labels"][600:] == IGNORE_INDEX).all())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_head_mode_reproduces_the_old_cut(self):
        """``"head"`` keeps the leading 600 tokens, exactly as before."""
        sample = _sample(1000, supervised_from=500)
        out = _run(self._transform("head"), sample)

        self.assertEqual(int(out["input_ids"].shape[0]), 600)
        for field in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
            self.assertTrue(torch.equal(out[field][:600], sample[field][:600]), field)
        # The old contract, verbatim: labels[500:600] survive, labels[600:] are gone.
        self.assertEqual(int((out["labels"] != IGNORE_INDEX).sum()), 100)
        self.assertTrue((out["labels"][600:] == IGNORE_INDEX).all())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_head_mode_loses_the_response_when_the_prompt_fills_the_window(self):
        """The bug the proportional mode fixes: 800 prompt + 200 response at 600."""
        sample = _sample(1000, supervised_from=800)

        head_out = _run(self._transform("head"), sample)
        self.assertEqual(int((head_out["labels"] != IGNORE_INDEX).sum()), 0)

        prop_out = _run(self._transform("proportional"), sample)
        self.assertEqual(int((prop_out["labels"] != IGNORE_INDEX).sum()), 200)
        self.assertEqual(int(prop_out["input_ids"].shape[0]), 600)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_no_supervised_token_falls_back_to_head(self):
        """With nothing supervised to preserve the proportional mode is a head cut."""
        sample = _sample(1000)
        out = _run(self._transform("proportional"), sample)

        self.assertEqual(int(out["input_ids"].shape[0]), 600)
        self.assertTrue(torch.equal(out["input_ids"][:600], torch.arange(600)))
        self.assertEqual(int((out["labels"] != IGNORE_INDEX).sum()), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_short_sample_is_untouched_by_either_mode(self):
        """Below ``max_seq_len`` neither mode cuts: only padding is applied."""
        sample = _sample(300, supervised_from=200)
        outputs = [_run(self._transform(mode), sample)
                   for mode in ("proportional", "head")]

        for out in outputs:
            self.assertEqual(int(out["input_ids"].shape[0]), 600)
            self.assertTrue(torch.equal(out["input_ids"][:300], sample["input_ids"]))
            self.assertEqual(int((out["labels"] != IGNORE_INDEX).sum()), 100)
            self.assertEqual(int(out["input_ids"][299]), 299)
            self.assertTrue((out["labels"][300:] == IGNORE_INDEX).all())
        self.assertTrue(torch.equal(outputs[0]["input_ids"], outputs[1]["input_ids"]))
        self.assertTrue(torch.equal(outputs[0]["labels"], outputs[1]["labels"]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mode_validation_and_default(self):
        """Only the two documented strings are accepted; the default preserves loss."""
        self.assertEqual(self._transform().truncate_mode, "proportional")
        self.assertEqual(self._transform("head").truncate_mode, "head")

        for bad in ("bogus", "Head", "proportional ", 1, None):
            with self.subTest(truncate_mode=bad):
                with self.assertRaisesRegex(ValueError, "truncate_mode"):
                    self._transform(bad)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_builder_forwards_the_option(self):
        """The builder passes ``truncate_mode`` through and defaults to proportional."""
        from hyper_parallel.models.kimi_k26.adapter.data.transform_fn import build_kimi_vlm_data_transform

        self.assertEqual(
            build_kimi_vlm_data_transform(
                processor=object(), max_seq_len=600, truncate_mode="head").truncate_mode,
            "head",
        )
        default = build_kimi_vlm_data_transform(processor=object(), max_seq_len=600)
        self.assertEqual(default.truncate_mode, "proportional")
        self.assertEqual(default.max_seq_len, 600)


if __name__ == "__main__":
    unittest.main()
