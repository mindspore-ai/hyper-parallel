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
"""Unit tests for the DeepSeek V4.1 dry-run example recipe."""
# pylint: disable=protected-access

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from hyper_parallel.components.modules.attention.compressed_sparse_attention2 import (
    SharedCompressedPackedSequence,
)
from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.distributed.expert_parallel.collectives import _EPAllToAllPadded
from hyper_parallel.tools.dryrun.examples import deepseek_v41


@contextlib.contextmanager
def _cpu_checkpoint_autocast_guard():
    """Avoid real aclInit when torch.utils.checkpoint probes NPU autocast on CPU CI."""
    npu = getattr(torch, "npu", None)
    if npu is None or not hasattr(npu, "is_bf16_supported"):
        yield
        return
    with patch.object(npu, "is_bf16_supported", return_value=False):
        yield


class _TinyCausalLM(nn.Module):
    """Minimal causal LM exposing .model and .lm_head for _bind_chunk_loss."""

    def __init__(self, hidden=4, vocab=8):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def model(self, input_ids=None, **_kwargs):
        """Return hidden states matching input_ids shape."""
        hidden = self.proj(input_ids)
        return SimpleNamespace(last_hidden_state=hidden)


class TestDeepSeekV41Example(unittest.TestCase):
    """Unit tests covering DeepSeek V4.1 dry-run replay and builder helpers."""

    def test_balanced_sparse_lookup_and_fake_padded_all_to_all(self):
        """Exercise balanced Engram lookup and padded EP all-to-all replay."""
        embedding = nn.Embedding(8, 4)
        hash_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        with patch.object(
            deepseek_v41, "ep_all_to_all", side_effect=lambda x, _s, _r, _g: x
        ):
            out = deepseek_v41._balanced_sparse_lookup(
                hash_ids,
                embedding,
                ep_group="ep",
                ep_rank=0,
                ep_size=2,
                padded_num_embeddings=16,
            )
        self.assertEqual(out.shape, (2, 4, 4))

        ctx = SimpleNamespace()
        x = torch.ones(4, 3)
        with (
            patch.object(_EPAllToAllPadded, "_pad_and_exchange", return_value=x) as mock_ex,
            patch.object(_EPAllToAllPadded, "_unpad", return_value=x) as mock_unpad,
        ):
            res = deepseek_v41._fake_padded_all_to_all(ctx, x, [2, 2], [2, 2], "ep")
        self.assertIs(res, x)
        self.assertEqual(ctx.pad_to, 2)
        mock_ex.assert_called_once()
        mock_unpad.assert_called_once()

    def test_install_replay_and_csa2_patches(self):
        """Exercise install_deepseek_v41_replay and CSA2 synthetic boundary patches."""
        orig_lookup = EngramModule._sparse_lookup
        orig_forward = _EPAllToAllPadded.forward
        try:
            deepseek_v41.install_deepseek_v41_replay()
            self.assertIs(EngramModule._sparse_lookup, deepseek_v41._balanced_sparse_lookup)
            self.assertIs(_EPAllToAllPadded.forward, deepseek_v41._fake_padded_all_to_all)
        finally:
            EngramModule._sparse_lookup = orig_lookup
            _EPAllToAllPadded.forward = orig_forward

        seq = SimpleNamespace(cu_seq_lens=torch.tensor([0, 2, 5, 10]), global_sequence_length=10)
        self.assertEqual(deepseek_v41._synthetic_document_boundaries(seq), (0, 4, 7, 10))

        deepseek_v41._install_csa2_patches()
        # Idempotent second call
        deepseek_v41._install_csa2_patches()
        with FakeTensorMode():
            fake_seq = SimpleNamespace(
                cu_seq_lens=torch.empty(3, dtype=torch.int64),
                global_sequence_length=8,
                _snapshot={},
                _prepared_starts={},
                local_positions=lambda device: torch.arange(4, device=device).unsqueeze(0),
            )
            bounds = SharedCompressedPackedSequence.document_boundaries(fake_seq)
            self.assertEqual(bounds, (0, 4, 8))
            starts = SharedCompressedPackedSequence.local_segment_starts(fake_seq, torch.device("cpu"))
            self.assertEqual(starts.shape, (1, 4))
            # Cached hit
            cached = SharedCompressedPackedSequence.local_segment_starts(fake_seq, torch.device("cpu"))
            self.assertIs(cached, starts)

    def test_bind_chunk_loss_and_build_text_model(self):
        """Exercise chunked loss forward/backward and build_text_model wiring."""
        model = _TinyCausalLM()
        with self.assertRaisesRegex(ValueError, "loss_chunks must be positive"):
            deepseek_v41._bind_chunk_loss(model, 0)

        deepseek_v41._bind_chunk_loss(model, chunks=2)
        x = torch.randn(1, 4, 4, requires_grad=True)
        labels = torch.tensor([[1, 2, 3, -100]])
        out = model(input_ids=x, labels=labels)
        with _cpu_checkpoint_autocast_guard():
            out.loss.backward()
        self.assertIsNotNone(x.grad)

        with self.assertRaisesRegex(ValueError, "divisible by loss_chunks"):
            model(input_ids=torch.randn(1, 3, 4), labels=torch.tensor([[1, 2, 3]]))

        fake_cfg = SimpleNamespace()
        fake_setup = SimpleNamespace(mesh_context="mesh")
        orig_lookup = EngramModule._sparse_lookup
        orig_forward = _EPAllToAllPadded.forward
        try:
            with (
                patch.object(deepseek_v41, "build_deepseek_v41_validation_config", return_value=fake_cfg),
                patch.object(
                    deepseek_v41.HyperAutoModelForCausalLM, "from_config", return_value=_TinyCausalLM()
                ),
                patch.object(deepseek_v41, "install_moe_replay") as mock_moe,
            ):
                with self.assertRaisesRegex(ValueError, "Unknown recompute_tier"):
                    deepseek_v41.build_text_model("cfg", "engram", recompute_tier="bad")

                built = deepseek_v41.build_text_model(
                    "cfg",
                    "engram",
                    loss_chunks=2,
                    recompute_tier="reuse",
                    distributed_setup=fake_setup,
                )
                self.assertEqual(fake_cfg.v41_model_mode, "full")
                mock_moe.assert_called_once_with(built, "mesh", "balanced")
        finally:
            EngramModule._sparse_lookup = orig_lookup
            _EPAllToAllPadded.forward = orig_forward


if __name__ == "__main__":
    unittest.main()
