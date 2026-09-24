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
"""Context-parallel input sharding for the Kimi-K2.5/K2.6 VLM text tower.

The CP contract is "replicated vision, sharded text": every CP rank runs the
vision tower on the complete sequence, then the text tower receives only this
rank's ``[cp_rank*L, (cp_rank+1)*L)`` window of ``inputs_embeds`` together with
globally-offset ``position_ids`` and a 4D offset-aware causal+padding mask.
"""
# pylint: disable=wrong-import-position

import os
import unittest
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch
from torch import nn

from tests.common.mark_utils import arg_mark

from hyper_parallel.models.kimi_k26.adapter.distributed.context_parallel import (
    bind_context_parallel,
    build_cp_attention_mask,
)


class _RecordingLanguageModel(nn.Module):
    """Text tower stand-in that records the kwargs it is entered with."""

    def __init__(self) -> None:
        """Initialize an empty call log."""
        super().__init__()
        self.calls = []

    def forward(self, **kwargs: Any) -> Any:
        """Record one call and return the kwargs unchanged."""
        self.calls.append(dict(kwargs))
        return kwargs


class _FakeModel(nn.Module):
    """Minimal ``KimiK25ForConditionalGeneration`` stand-in."""

    def __init__(self, model_type: str = "kimi_k25", with_language_model: bool = True) -> None:
        """Build the ``model.language_model`` nesting the adapter resolves."""
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        inner = nn.Module()
        if with_language_model:
            inner.language_model = _RecordingLanguageModel()
        self.model = inner


class _FakeMesh:
    """Mesh exposing the CP sizes read by the adapter."""

    def __init__(self, cp_size: int, cp_rank: int = 0) -> None:
        """Store the CP topology."""
        self.cp_size = cp_size
        self.cp_rank = cp_rank


def _make_kwargs(seq_len=8, pad_tail=2, batch_size=2):
    """Build the full-sequence language-model kwargs for one VLM forward."""
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    if pad_tail:
        attention_mask[:, seq_len - pad_tail:] = 0
    return {
        "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "inputs_embeds": torch.randn(batch_size, seq_len, 4),
        "attention_mask": attention_mask,
        "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
    }


class TestBuildCpAttentionMask(unittest.TestCase):
    """``build_cp_attention_mask`` matches causal-offset + padding semantics."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mask_is_offset_causal_and_padding_aware(self):
        """Row i of rank r admits j <= r*L + i and un-padded K/V positions."""
        seq_len, cp_size, pad_tail = 8, 2, 2
        local_len = seq_len // cp_size
        attention_mask = torch.ones(2, seq_len, dtype=torch.long)
        attention_mask[:, seq_len - pad_tail:] = 0

        for cp_rank in range(cp_size):
            with self.subTest(cp_rank=cp_rank):
                mask = build_cp_attention_mask(
                    attention_mask,
                    q_len=local_len,
                    kv_len=seq_len,
                    query_offset=cp_rank * local_len,
                    device=torch.device("cpu"),
                )
                self.assertEqual(tuple(mask.shape), (2, 1, local_len, seq_len))
                self.assertEqual(mask.dtype, torch.bool)
                for batch in range(2):
                    for row in range(local_len):
                        for col in range(seq_len):
                            expected = (
                                col <= cp_rank * local_len + row
                                and attention_mask[batch, col].item() == 1
                            )
                            self.assertEqual(
                                bool(mask[batch, 0, row, col].item()), expected
                            )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mask_requires_the_complete_sequence(self):
        """The mask must be a 2D [B, S] tensor covering every rank's K/V."""
        with self.assertRaisesRegex(ValueError, "requires a 2D attention_mask"):
            build_cp_attention_mask(
                None, q_len=4, kv_len=8, query_offset=0, device=torch.device("cpu")
            )
        with self.assertRaisesRegex(ValueError, "expects a 2D"):
            build_cp_attention_mask(
                torch.ones(2, 1, 4, 8), q_len=4, kv_len=8, query_offset=0,
                device=torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "complete sequence"):
            build_cp_attention_mask(
                torch.ones(2, 4), q_len=4, kv_len=8, query_offset=0,
                device=torch.device("cpu"),
            )


class TestBindContextParallel(unittest.TestCase):
    """``bind_context_parallel`` installs the rank-local text-tower input."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp1_is_a_no_op(self):
        """cp_size == 1 leaves the language-model forward untouched."""
        model = _FakeModel()
        language_model = model.model.language_model

        bind_context_parallel(model, _FakeMesh(cp_size=1))

        # A rewritten forward is installed as an instance attribute; the
        # class-level method is what an untouched module resolves.
        self.assertNotIn("forward", vars(language_model))
        self.assertFalse(getattr(language_model, "_hp_kimi_k26_cp_sharded", False))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp2_slices_embeds_and_offsets_positions(self):
        """Rank r receives its own embed window with global position ids."""
        seq_len, cp_size = 8, 2
        local_len = seq_len // cp_size
        for cp_rank in range(cp_size):
            with self.subTest(cp_rank=cp_rank):
                model = _FakeModel()
                recorder = model.model.language_model
                kwargs = _make_kwargs(seq_len=seq_len)
                bind_context_parallel(model, _FakeMesh(cp_size, cp_rank))

                recorder(**kwargs)

                call = recorder.calls[-1]
                self.assertTrue(torch.equal(
                    call["inputs_embeds"],
                    kwargs["inputs_embeds"][:, cp_rank * local_len:(cp_rank + 1) * local_len],
                ))
                expected_positions = torch.arange(
                    cp_rank * local_len, (cp_rank + 1) * local_len
                ).unsqueeze(0).expand(kwargs["inputs_embeds"].shape[0], -1)
                self.assertTrue(torch.equal(call["position_ids"], expected_positions))
                self.assertEqual(tuple(call["attention_mask"].shape), (2, 1, local_len, seq_len))
                self.assertNotIn("input_ids", call)
                self.assertIsNone(call["past_key_values"])
                self.assertFalse(call["use_cache"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_bind_is_idempotent(self):
        """Rebinding a sharded tower keeps the already-installed forward."""
        model = _FakeModel()
        bind_context_parallel(model, _FakeMesh(2, 0))
        installed = vars(model.model.language_model)["forward"]

        bind_context_parallel(model, _FakeMesh(2, 0))

        self.assertIs(vars(model.model.language_model)["forward"], installed)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_forward_requires_inputs_embeds(self):
        """The media scatter must run before the text tower is entered."""
        model = _FakeModel()
        bind_context_parallel(model, _FakeMesh(2, 0))

        with self.assertRaisesRegex(RuntimeError, "requires inputs_embeds"):
            model.model.language_model(input_ids=torch.zeros(2, 8, dtype=torch.long))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_forward_rejects_indivisible_sequence_length(self):
        """cp_size must divide the padded sequence length."""
        model = _FakeModel()
        bind_context_parallel(model, _FakeMesh(3, 0))

        with self.assertRaisesRegex(ValueError, "divisible by cp_size"):
            model.model.language_model(**_make_kwargs(seq_len=8))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_unexpected_model_structure(self):
        """A non-Kimi config or a missing text tower fails fast."""
        with self.assertRaisesRegex(TypeError, r"model_type in \('kimi_k25', 'kimi_k26'\)"):
            bind_context_parallel(_FakeModel(model_type="qwen3"), _FakeMesh(2, 0))
        with self.assertRaisesRegex(TypeError, "model.model.language_model"):
            bind_context_parallel(
                _FakeModel(with_language_model=False), _FakeMesh(2, 0)
            )


if __name__ == "__main__":
    unittest.main()
