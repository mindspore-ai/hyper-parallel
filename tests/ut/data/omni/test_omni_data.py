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
"""CPU regression tests for the opt-in multimodal data path."""

"""Qwen processor supervision boundaries on the shared Omni pipeline."""

import unittest
from typing import Any

import torch

from hyper_parallel.data.omni.omni_transform import AutoProcessorTransform
from tests.common.mark_utils import arg_mark

class FakeProcessor:
    """Return stable processor-shaped data without loading any model."""

    def apply_chat_template(self, messages: list, *, add_generation_prompt: bool, **kwargs: Any) -> dict:
        """Expose a masked media prefix and a supervised assistant suffix."""
        del messages, kwargs
        tokens = [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 5]
        return {
            "input_ids": torch.tensor([tokens]),
            "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 0] + ([] if add_generation_prompt else [0, 0])]),
            "pixel_values": torch.ones(2, 3),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }


class BoundaryMergeProcessor(FakeProcessor):
    """Model expanded image tokens and an empty-thinking BPE boundary merge."""

    def __init__(self) -> None:
        """Initialize exact prompt/full token sequences for a boundary merge."""
        self.full_tokens = [1, 2, 2, 3, 271, 4, 271, 5, 6]
        self.prompt_tokens = [1, 2, 2, 3, 198]
        self.full_text = "image<think>\n\n</think>\n\nanswer"
        self.prompt_text = "image<think>\n"
        self.tokenizer = self

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        """Keep special tokens and whitespace visible during boundary checks."""
        del kwargs
        vocabulary = {1: "user", 2: "<image>", 3: "<think>", 198: "\n", 271: "\n\n",
                      4: "</think>", 5: "answer", 6: "<eos>", 7: "different", 8: "\ufffd"}
        return "".join(vocabulary[token] for token in tokens)

    def apply_chat_template(self, messages: list, *, add_generation_prompt: bool, **kwargs: Any) -> Any:
        """Return native sequence-aligned fields or rendered template text."""
        del messages
        if not kwargs.get("tokenize", True):
            return self.prompt_text if add_generation_prompt else self.full_text
        tokens = self.prompt_tokens if add_generation_prompt else self.full_tokens
        return {
            "input_ids": torch.tensor([tokens]),
            "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 1] + [0] * (len(tokens) - 3)]),
            "pixel_values": torch.ones(2, 3),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }


class TestOmniProcessorBoundaries(unittest.TestCase):
    """Preserve native processor tokens while locating assistant supervision."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_processor_labels_and_length(self):
        """Final assistant is supervised; overlong media samples are not sliced."""
        transform = AutoProcessorTransform(max_seq_len=5, processor=FakeProcessor())
        sample = {"messages": [{"role": "user", "content": "image"}, {"role": "assistant", "content": "answer"}]}
        result = transform.encode_sample(sample)
        self.assertEqual(result["labels"].tolist(), [-100, -100, -100, 4, 5])
        self.assertEqual(tuple(result["pixel_values"].shape), (2, 3))
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            AutoProcessorTransform(max_seq_len=4, processor=FakeProcessor()).encode_sample(sample)
        with self.assertRaisesRegex(ValueError, "final assistant"):
            transform.encode_sample({"messages": sample["messages"][:-1]})


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_processor_masks_merged_boundary_without_changing_inputs(self):
        """Empty thinking preserves image expansion and masks the shared newline token."""
        processor = BoundaryMergeProcessor()
        transform = AutoProcessorTransform(max_seq_len=16, processor=processor)
        messages = [{"role": "user", "content": "image"}, {"role": "assistant", "content": "answer"}]
        original = processor.apply_chat_template(messages, add_generation_prompt=False)
        result = transform.encode_sample({"messages": messages})
        self.assertEqual(result["labels"].tolist(), [-100] * 5 + [4, 271, 5, 6])
        for field in ("input_ids", "attention_mask", "mm_token_type_ids"):
            self.assertTrue(torch.equal(result[field], original[field].squeeze(0)))
        for field in ("pixel_values", "image_grid_thw"):
            self.assertTrue(torch.equal(result[field], original[field]))


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_processor_rejects_real_rewrites_and_lossy_decoding(self):
        """Text rewrites and changed expanded tokens must not be hidden by the fallback."""
        sample = {"messages": [{"role": "user"}, {"role": "assistant"}]}
        cases = (("prompt_text", "changed", "text prefix"),
                 ("prompt_tokens", [1, 7, 2, 3, 198], "lossless text"),
                 ("prompt_tokens", [1, 2, 2, 3, 8], "lossless text"))
        for field, value, error in cases:
            with self.subTest(field=field, value=value):
                processor = BoundaryMergeProcessor()
                setattr(processor, field, value)
                with self.assertRaisesRegex(ValueError, error):
                    AutoProcessorTransform(max_seq_len=16, processor=processor).encode_sample(sample)


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_processor_rejects_no_targets_after_boundary_masking(self):
        """A merge consuming all remaining tokens does not create a supervised answer."""
        processor = BoundaryMergeProcessor()
        processor.full_tokens = [1, 2, 2, 3, 271]
        sample = {"messages": [{"role": "user"}, {"role": "assistant"}]}
        with self.assertRaisesRegex(ValueError, "no supervised"):
            AutoProcessorTransform(max_seq_len=16, processor=processor).encode_sample(sample)
