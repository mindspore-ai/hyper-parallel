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
"""CPU tests of sharing ownership, invocation isolation and autograd replay."""

from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint, DefaultDeviceType

from hyper_parallel.components.modules.attention.deepseek_sparse_attention import (
    deepseek_sparse_attention as impl,
)
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.indexer_share import (
    IndexShareContext, install_indexer_share, resolve_index_share_plan,
)
from hyper_parallel.components.modules import DeepseekSparseAttention
from hyper_parallel.components.modules.attention.deepseek_sparse_attention.cp_deepseek_sparse_attention import (
    CPDeepseekSparseAttentionModule, CPSparseAttentionSequenceMetadata, _gather_key_banks,
)


def config(**overrides):
    """Small native dimensions, independent of Transformers or an accelerator."""
    values = dict(hidden_size=16, num_attention_heads=2, q_lora_rank=8, kv_lora_rank=8,
                  qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
                  index_n_heads=2, index_head_dim=8, index_topk=4, num_hidden_layers=3,
                  dsa_indexer_types=["full", "shared", "shared"], dsa_loss_coeff=0.3,
                  layer_idx=0, rotary_interleaved=False, model_type="deepseek_v32")
    return SimpleNamespace(**(values | overrides))


def select(query, key, weights, *_args, **_kwargs):
    """CPU selection surrogate; assertions separately inspect the reused selection."""
    indices = torch.arange(4, dtype=torch.int32).expand(query.shape[0] * query.shape[1], 1, 4).clone()
    return indices, query.flatten(0, 1), key.flatten(0, 1), weights.flatten(0, 1)


def sparse(query, key, qrope, krope, indices, scale, *_args, **_kwargs):
    """Differentiable CPU attention with detached teacher statistics."""
    del indices
    qfull, kfull = torch.cat((query, qrope), -1), torch.cat((key, krope), -1)
    scores = torch.einsum("bqhd,bkhd->bhqk", qfull, kfull) * scale
    scores = scores.masked_fill(~torch.ones(scores.shape[-2:], dtype=torch.bool).tril(), -torch.inf)
    output = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), key)
    return output, scores.amax(-1).detach(), torch.ones_like(scores[..., 0]).detach()


def dense(module, query, key, value, _lengths, *, need_stats):
    """Dense CPU kernel boundary with real Q/K/V projections."""
    del need_stats
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) * module.scaling
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), value), scores.amax(-1).detach(), scores.sum(-1).detach()


def sparse_loss(query, key, weights, *args, **_kwargs):
    """Known objective with each served teacher's different magnitude."""
    return args[-1] * (1 + args[0].square().mean()) * sum(t.square().mean() for t in (query, key, weights))


def dense_loss(query, key, weights, *args, loss_coefficient, **_kwargs):
    """The same known objective for dense warm-up."""
    return loss_coefficient * (1 + args[0].square().mean()) * sum(
        tensor.square().mean() for tensor in (query, key, weights)
    )


@contextmanager
def kernels():
    """Mock only unavailable accelerator kernels, retaining projections/autograd."""
    with ExitStack() as stack:
        index = stack.enter_context(patch.object(impl, "dsa_indexer", side_effect=select))
        attention = stack.enter_context(patch.object(impl, "dsa_sparse_attention", side_effect=sparse))
        loss = stack.enter_context(patch.object(impl, "dsa_kl_loss", side_effect=sparse_loss))
        stack.enter_context(patch.object(impl, "_dense_flash_attention", side_effect=dense))
        stack.enter_context(patch("hyper_parallel.components.functional.dsa_dense.dsa_dense_indexer_lse",
                                  return_value=(torch.zeros(1), torch.ones(1))))
        stack.enter_context(patch("hyper_parallel.components.functional.dsa_dense.dsa_dense_indexer_kl_loss",
                                  side_effect=dense_loss))
        yield index, attention, loss


class Stack(nn.Module):
    """Native kwargs loop matching the HF model's sharing transport."""

    def __init__(self, *, mode="served", warmup=False, recompute=False, roles=None):
        super().__init__()
        self.config = config(dsa_index_share_loss=mode, dsa_dense_warm_up=warmup)
        if roles is not None:
            self.config.dsa_indexer_types = roles
            self.config.num_hidden_layers = len(roles)
        self.layers = nn.ModuleList([
            DeepseekSparseAttention(config=SimpleNamespace(**(vars(self.config) | {"layer_idx": i})))
            for i in range(self.config.num_hidden_layers)
        ])
        self.recompute = recompute
        install_indexer_share(self)

    def forward(self, hidden, **kwargs):
        """Pass the same invocation carrier to every checkpointed layer."""
        for layer in self.layers:
            if self.recompute:
                result = checkpoint(layer, hidden, use_reentrant=False, **kwargs)
            else:
                result = layer(hidden, **kwargs)
            hidden = hidden + result[0]
        return hidden


class TestIndexerShare(unittest.TestCase):
    """Validate the new behavior without importing NPU kernels."""

    def setUp(self) -> None:
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        torch.manual_seed(42)

    def test_layout_and_partial_group(self) -> None:
        plan = resolve_index_share_plan(config(dsa_indexer_types=None, num_hidden_layers=5, dsa_index_topk_freq=3))
        self.assertEqual(plan.leaders, (0, 0, 0, 3, 3))
        self.assertEqual(plan.group_size(4), 2)
        plan = resolve_index_share_plan(config(dsa_indexer_types=None, num_hidden_layers=6,
                                               dsa_index_topk_freq=3, dsa_index_skip_topk_offset=1))
        self.assertEqual(plan.leaders, (0, 1, 1, 1, 4, 4))
        self.assertEqual(resolve_index_share_plan(config(dsa_index_topk_freq=99)).leaders, (0, 0, 0))


    def test_selection_is_computed_once_and_reused(self) -> None:
        for mode, calls in (("served", 3), ("leader", 1)):
            with self.subTest(mode=mode), kernels() as (index, attention, loss):
                model = Stack(mode=mode)
                model(torch.randn(1, 4, 16)).sum().backward()
                self.assertEqual(index.call_count, 1)
                selections = [call.args[4] for call in attention.call_args_list]
                self.assertTrue(all(torch.equal(selections[0], item) for item in selections))
                self.assertEqual(loss.call_count, calls)
                self.assertTrue(all(call.args[-1] == (0.1 if mode == "served" else 0.3)
                                    or abs(call.args[-1] - 0.1) < 1e-8 for call in loss.call_args_list))
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.layers[0].indexer.parameters()))


    def test_checkpoint_output_and_all_gradients_match(self) -> None:
        for warmup in (False, True):
            for mode in ("leader", "served"):
                with self.subTest(warmup=warmup, mode=mode), kernels():
                    baseline = Stack(mode=mode, warmup=warmup)
                    candidate = Stack(mode=mode, warmup=warmup, recompute=True)
                    candidate.load_state_dict(baseline.state_dict())
                    hidden = torch.randn(1, 4, 16, requires_grad=True)
                    replay = hidden.detach().clone().requires_grad_()
                    expected = baseline(hidden)
                    actual = candidate(replay)
                    expected.square().mean().backward()
                    actual.square().mean().backward()
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(replay.grad, hidden.grad)
                    for (_, param), (_, reference) in zip(candidate.named_parameters(), baseline.named_parameters()):
                        torch.testing.assert_close(param.grad, reference.grad)


    def test_cp_shared_gathers_only_main_banks(self) -> None:
        for leader in (True, False):
            for overlap in (True, False):
                with self.subTest(leader=leader, overlap=overlap):
                    context = MagicMock()
                    tensors = tuple(torch.randn(1, 4, 2, 8, requires_grad=True) for _ in range(7))
                    index_states = tensors[:3] if leader else ()
                    metadata = CPSparseAttentionSequenceMetadata(6, torch.tensor([4]), torch.tensor([6]))
                    context.gather_sequence.side_effect = lambda x, dim: (torch.cat((x, x), dim=dim), None)

                    def launch(x, dim):
                        receipt = MagicMock()
                        receipt.wait.return_value = (torch.cat((x, x), dim=dim), None)
                        return receipt

                    context.launch_gather_sequence.side_effect = launch
                    ready_index, attention = _gather_key_banks(context, overlap, index_states, tensors[3:], metadata)
                    calls = context.launch_gather_sequence if overlap else context.gather_sequence
                    self.assertEqual(calls.call_count, 3 if leader else 2)
                    self.assertEqual(attention[1].shape[1], 6)
                    self.assertEqual(attention[3].shape[1], 6)
                    self.assertIs(attention[0], tensors[3])
                    self.assertEqual(len(ready_index), 3 if leader else 0)
                    (attention[1].sum() + attention[3].sum()).backward()
                    self.assertIsNotNone(tensors[4].grad)
                    self.assertIsNotNone(tensors[6].grad)
