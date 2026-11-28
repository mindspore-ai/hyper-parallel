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
"""CPU contracts for DSA MTP Top-K/KV sharing and recomputation."""

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.components.modules import DeepseekSparseAttention, MTPLayer, MultiTokenPrediction
from hyper_parallel.components.modules.attention.deepseek_sparse_attention import deepseek_sparse_attention as dsa
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.cp_deepseek_sparse_attention import (
    CPDeepseekSparseAttentionModule,
)
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.deepseek_sparse_attention import (
    SparseAttentionInputs, SparseAttentionSequenceMetadata,
)


def _config(layer_idx: int, dsa_loss_coeff: float = 0.0) -> SimpleNamespace:
    """Small DSA config with the released MTP flag enabled."""
    return SimpleNamespace(
        hidden_size=8, num_attention_heads=2, q_lora_rank=4, kv_lora_rank=4,
        qk_rope_head_dim=2, qk_nope_head_dim=2, v_head_dim=4,
        index_n_heads=2, index_head_dim=4, index_topk=2, num_hidden_layers=4,
        layer_idx=layer_idx, model_type="deepseek_v32", dsa_loss_coeff=dsa_loss_coeff,
        index_share_for_mtp_iteration=True, rotary_interleaved=False,
    )


class _DSADecoder(nn.Module):
    """Pass per-step MTP metadata directly to a real DSA module."""

    def __init__(self, layer_idx: int, dsa_loss_coeff: float = 0.0) -> None:
        super().__init__()
        self.attn = DeepseekSparseAttention(config=_config(layer_idx, dsa_loss_coeff))

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        output, _ = self.attn(hidden_states, **kwargs)
        return output


def _indexer(index_query, _index_key, _weights, query_ends, _key_ends, topk, **_kwargs):
    """Emit a unique route per query row so position mapping is observable."""
    indexer_calls.append((index_query.shape[0] * index_query.shape[1], topk))
    rows = index_query.shape[0] * index_query.shape[1]
    indices = torch.arange(rows, dtype=torch.int32, device=index_query.device).view(rows, 1, 1)
    indices = indices.expand(-1, 1, topk).contiguous()
    return indices, index_query.flatten(0, 1), index_query.flatten(0, 1), index_query.flatten(0, 1)


def _sparse(query, key, _query_rope, _key_rope, topk, _scale, q_ends, k_ends, **_kwargs):
    """Differentiable stand-in that records the selected routes and source KV."""
    calls.append((query, key, topk, q_ends.detach().clone(), k_ends.detach().clone()))
    return query + key.mean(dim=1, keepdim=True), query.new_zeros(1), query.new_ones(1)


class TestMTPIndexShare(unittest.TestCase):
    """Exercise producer/consumer, packed alignment and checkpoint replay."""

    def setUp(self) -> None:
        torch.manual_seed(19)
        self.embedding = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16, bias=False)
        self.tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
        self.hidden = torch.randn(1, 7, 8, requires_grad=True)

    def _model(self, num_layers: int = 2) -> MultiTokenPrediction:
        return MultiTokenPrediction([
            MTPLayer(8, _DSADecoder(4 + step)) for step in range(num_layers)
        ])

    def _kernels(self):
        global calls, indexer_calls
        calls = []
        indexer_calls = []
        stack = ExitStack()
        stack.enter_context(patch.object(dsa, "dsa_indexer", side_effect=_indexer))
        stack.enter_context(patch.object(dsa, "dsa_sparse_attention", side_effect=_sparse))
        return stack

    def test_later_step_reuses_step_zero_topk_and_kv_and_backpropagates(self) -> None:
        model = self._model()
        producer = model.layers[0].mtp_block.attn
        projection_calls = []
        handle = producer.linear_qkv.register_forward_hook(lambda *_: projection_calls.append(1))
        try:
            with self._kernels():
                output = model(
                    self.hidden[:, :5], self.tokens[:, :5], embedding=self.embedding, head=self.head,
                )
                loss = sum(value.square().mean() for value in output.hidden_states)
                loss.backward()
        finally:
            handle.remove()

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(indexer_calls), 1, "the Indexer should run only in step zero")
        self.assertEqual(len(projection_calls), 1, "consumer should project Q only, not new KV")
        self.assertIs(calls[0][1], calls[1][1], "both steps must use the first step's KV tensor")
        expected_rows = torch.tensor([1, 2, 3], dtype=torch.long)
        torch.testing.assert_close(calls[1][2], calls[0][2].index_select(0, expected_rows))
        self.assertEqual(calls[1][4].tolist(), [4])
        self.assertTrue(torch.isfinite(self.hidden.grad).all())
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        first_grad = model.layers[0].mtp_block.attn.linear_qkv.weight.grad
        later_grad = model.layers[1].mtp_block.attn.linear_qkv.weight.grad
        self.assertIsNotNone(first_grad)
        self.assertGreater(first_grad[4:].abs().sum().item(), 0.0,
                           "consumer loss should reach the producer KV projection")
        if later_grad is not None:
            torch.testing.assert_close(later_grad[4:], torch.zeros_like(later_grad[4:]))

    def test_checkpoint_replay_reuses_first_pass_state_without_reindexing(self) -> None:
        model = self._model()
        with self._kernels():
            output = model(
                self.hidden[:, :5], self.tokens[:, :5], embedding=self.embedding, head=self.head,
                checkpoint_layers=True,
            )
            sum(value.square().mean() for value in output.hidden_states).backward()
        self.assertEqual(len(calls), 4)  # two initial calls plus two checkpoint replays
        self.assertEqual(len(indexer_calls), 2)  # step zero plus replay, never the consumer
        self.assertEqual(sum(call[0].shape[1] == 4 for call in calls), 2)
        self.assertEqual(sum(call[0].shape[1] == 3 for call in calls), 2)
        self.assertTrue(torch.isfinite(self.hidden.grad).all())

    def test_packed_documents_keep_independent_topk_row_alignment(self) -> None:
        model = self._model()
        with self._kernels():
            output = model(
                self.hidden, self.tokens, embedding=self.embedding,
                packed_seq_lengths=(3, 4),
            )
        self.assertEqual([state.shape[1] for state in output.hidden_states], [5, 3])
        self.assertEqual(len(calls), 2)
        expected_rows = torch.tensor([1, 3, 4], dtype=torch.long)
        torch.testing.assert_close(calls[1][2], calls[0][2].index_select(0, expected_rows))
        self.assertEqual(calls[1][3].tolist(), [1, 3])
        self.assertEqual(calls[1][4].tolist(), [2, 5])


    def test_cp_consumer_does_not_gather_the_shared_kv_bank_again(self) -> None:
        module = CPDeepseekSparseAttentionModule(config=_config(4), kv_cp_size=2)
        module._cp_context = SimpleNamespace(size=2, rank=0)
        module.prepare_key_bank = unittest.mock.Mock(side_effect=AssertionError("unexpected KV gather"))
        inputs = SparseAttentionInputs(
            (), (torch.empty(1),) * 4,
            SparseAttentionSequenceMetadata(1, torch.tensor([1]), torch.tensor([1])),
            False, torch.tensor([[[0]]]),
        )
        self.assertIs(module.exchange_attention(inputs), inputs)
        module.prepare_key_bank.assert_not_called()


if __name__ == "__main__":
    unittest.main()
