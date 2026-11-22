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
    IndexShareContext, resolve_index_share_plan,
)
from hyper_parallel.components.modules import DeepseekSparseAttention, MTPLayer, MultiTokenPrediction
from hyper_parallel.components.modules.attention.deepseek_sparse_attention import install_indexer_share
from hyper_parallel.distributed.activation_checkpoint import _apply_activation_checkpointing
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


class MTPDecoder(nn.Module):
    """Return the hidden states from a DSA-backed MTP decoder block."""

    def __init__(self, layer_idx, *, share_mtp_indexer=True):
        super().__init__()
        self.attn = DeepseekSparseAttention(config=config(
            layer_idx=layer_idx, mtp_num_layers=2,
            index_share_for_mtp_iteration=share_mtp_indexer,
        ))

    def forward(self, hidden_states, **kwargs):
        output, _ = self.attn(hidden_states, **kwargs)
        return output


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

    def test_invalid_layout(self) -> None:
        for options in ({"dsa_indexer_types": ["shared"] * 3}, {"dsa_indexer_types": ["full"]},
                        {"dsa_index_share_loss": "sum"}, {"dsa_index_share_topk_offload": True},
                        {"dsa_indexer_types": None, "dsa_index_topk_freq": 0},
                        {"dsa_indexer_types": None, "dsa_index_skip_topk_offset": -1},
                        {"index_share_for_mtp_iteration": 1},
                        {"layer_idx": None}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DeepseekSparseAttention(config=config(**options))

    def test_shared_layers_have_no_indexer_parameters(self) -> None:
        model = Stack()
        self.assertIsNotNone(model.layers[0].indexer)
        for layer in model.layers[1:]:
            self.assertIsNone(layer.indexer)
            self.assertFalse(any("indexer" in name for name in layer.state_dict()))
            layer.reset_parameters()
        clone = Stack()
        clone.load_state_dict(model.state_dict(), strict=True)

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

    def test_missing_leader_and_context_fail(self) -> None:
        layer = Stack().layers[1]
        with self.assertRaisesRegex(ValueError, "fresh IndexShareContext"):
            layer(torch.randn(1, 4, 16))
        with self.assertRaisesRegex(ValueError, "run first"):
            layer(torch.randn(1, 4, 16), index_share_context=IndexShareContext())

    def test_stage_and_shape_mismatch_fail(self) -> None:
        model = Stack()
        context = IndexShareContext()
        with kernels():
            model.layers[0](torch.randn(1, 4, 16), index_share_context=context)
            with self.assertRaisesRegex(ValueError, "matching stage"):
                model.layers[1](torch.randn(2, 4, 16), index_share_context=context)
            model.layers[1].dsa_dense_warm_up = True
            with self.assertRaisesRegex(ValueError, "matching stage"):
                model.layers[1](torch.randn(1, 4, 16), index_share_context=context)

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

    def test_overlapping_invocations_backward_in_reverse(self) -> None:
        with kernels():
            reference = Stack(recompute=True)
            actual = Stack(recompute=True)
            actual.load_state_dict(reference.state_dict())
            batches = [torch.randn(1, 4, 16), torch.randn(1, 4, 16)]
            for batch in reversed(batches):
                reference(batch).square().mean().backward()
            outputs = [actual(batch) for batch in batches]
            for output in reversed(outputs):
                output.square().mean().backward()
            for (_, param), (_, expected) in zip(actual.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(param.grad, expected.grad)

    def test_disabled_sharing_retains_every_indexer(self) -> None:
        with kernels() as (index, _, loss):
            model = Stack(roles=["full"] * 3)
            model(torch.randn(1, 4, 16)).sum().backward()
            self.assertEqual(index.call_count, 3)
            self.assertEqual(loss.call_count, 3)
            self.assertTrue(all(layer.indexer is not None for layer in model.layers))

    def test_mtp_iterations_share_parameters_but_compute_separate_selections(self) -> None:
        model = MultiTokenPrediction([
            MTPLayer(16, MTPDecoder(3)),
            MTPLayer(16, MTPDecoder(4)),
        ])
        first = model.layers[0].mtp_block.attn
        second = model.layers[1].mtp_block.attn
        self.assertIs(first.indexer, second.indexer)
        self.assertNotIn("indexer", second._modules)
        self.assertFalse(second._owns_indexer)
        self.assertEqual(sum("indexer.wq_b.weight" in name for name in model.state_dict()), 1)

        reference = MultiTokenPrediction([
            MTPLayer(16, MTPDecoder(3, share_mtp_indexer=False)),
            MTPLayer(16, MTPDecoder(4, share_mtp_indexer=False)),
        ])
        reference.load_state_dict(model.state_dict(), strict=False)
        reference_first = reference.layers[0].mtp_block.attn
        reference_second = reference.layers[1].mtp_block.attn
        reference_second.indexer.load_state_dict(first.indexer.state_dict())

        tokens = torch.tensor([[0, 1, 2, 3, 4]])
        embedding = nn.Embedding(8, 16)
        with kernels() as (index, _, _):
            hidden = torch.randn(1, 5, 16)
            output = model(hidden, tokens, embedding=embedding)
            expected = reference(hidden, tokens, embedding=embedding)
            torch.testing.assert_close(output.hidden_states, expected.hidden_states)
            sum(state.square().mean() for state in output.hidden_states).backward()
            sum(state.square().mean() for state in expected.hidden_states).backward()

        self.assertEqual(index.call_count, 4)
        self.assertEqual([call.args[0].shape[1] for call in index.call_args_list], [4, 3, 4, 3])
        for name, parameter in first.indexer.named_parameters():
            self.assertIsNotNone(parameter.grad)
            expected_gradient = dict(reference_first.indexer.named_parameters())[name].grad
            expected_gradient = expected_gradient + dict(reference_second.indexer.named_parameters())[name].grad
            torch.testing.assert_close(parameter.grad, expected_gradient)

    def test_mtp_indexers_remain_independent_when_disabled(self) -> None:
        model = MultiTokenPrediction([
            MTPLayer(16, MTPDecoder(3, share_mtp_indexer=False)),
            MTPLayer(16, MTPDecoder(4, share_mtp_indexer=False)),
        ])
        first = model.layers[0].mtp_block.attn
        second = model.layers[1].mtp_block.attn
        self.assertIsNot(first.indexer, second.indexer)
        self.assertTrue(second._owns_indexer)

    def test_parallel_sharing_preserves_existing_family_support(self) -> None:
        for topology in ({"tp_size": 2}, {"cp_size": 2}, {"tp_size": 2, "cp_size": 2},
                         {"dp_size": 2}, {"ep_size": 2}):
            with self.subTest(topology=topology):
                layer = DeepseekSparseAttention(
                    config=config(layer_idx=1), context=topology,
                    kv_cp_size=topology.get("cp_size", 1),
                )
                self.assertIsNone(layer.indexer)
                if topology.get("cp_size", 1) > 1:
                    self.assertIsInstance(layer, CPDeepseekSparseAttentionModule)
        with self.assertRaisesRegex(ValueError, "pipeline"):
            DeepseekSparseAttention(config=config(), context={"pp_size": 2})

    def test_install_is_idempotent(self) -> None:
        model = Stack()
        handle = model._hp_index_share_hook
        install_indexer_share(model)
        self.assertIs(model._hp_index_share_hook, handle)
        with kernels():
            with self.assertRaisesRegex(ValueError, "model owns"):
                model(torch.randn(1, 4, 16), index_share_context=IndexShareContext())

    def test_served_gradient_is_mean_of_all_teachers(self) -> None:
        with kernels() as (_, _, loss):
            model = Stack()
            hidden = torch.randn(1, 4, 16, requires_grad=True)
            output = model(hidden)
            calls = loss.call_args_list
            parameters = tuple(model.layers[0].indexer.parameters())
            terms = [sparse_loss(*call.args[:-1], 1.0) for call in calls]
            expected = torch.autograd.grad(0.3 * torch.stack(terms).mean(), parameters, retain_graph=True)
            output.sum().backward()
            for param, gradient in zip(parameters, expected):
                torch.testing.assert_close(param.grad, gradient)
            self.assertTrue(all(not call.args[i].requires_grad for call in calls for i in (3, 4, 8, 9)))

    def test_multiple_groups_checkpoint_replay(self) -> None:
        with kernels():
            reference = Stack(roles=["full", "shared", "full", "shared", "shared"])
            actual = Stack(roles=["full", "shared", "full", "shared", "shared"], recompute=True)
            actual.load_state_dict(reference.state_dict())
            hidden = torch.randn(1, 4, 16)
            reference(hidden).sum().backward()
            actual(hidden).sum().backward()
            for (_, param), (_, expected) in zip(actual.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(param.grad, expected.grad)

    def test_reentrant_training_rejected_and_eval_no_grad_works(self) -> None:
        model = Stack()
        with kernels(), torch.no_grad():
            with self.assertRaisesRegex(ValueError, "non-reentrant"):
                model(torch.randn(1, 4, 16))
            model.eval()(torch.randn(1, 4, 16))

    def test_model_adapter_installs_carrier_and_nonreentrant_regions(self) -> None:
        with kernels():
            reference = Stack()
            actual = Stack()
            actual.load_state_dict(reference.state_dict())
            actual._hp_index_share_hook.remove()
            del actual._hp_index_share_hook
            actual.config.model_type = "deepseek_v32"
            actual.gradient_checkpointing = False
            _apply_activation_checkpointing(actual, "full")
            hidden = torch.randn(1, 4, 16)
            reference(hidden).square().mean().backward()
            actual(hidden).square().mean().backward()
            for (_, param), (_, expected) in zip(actual.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(param.grad, expected.grad)

    def test_meta_shared_layer_materialization(self) -> None:
        with torch.device("meta"):
            layer = DeepseekSparseAttention(config=config(layer_idx=1))
        self.assertIsNone(layer.indexer)
        self.assertTrue(all(p.is_meta for p in layer.parameters()))
        layer.to_empty(device="cpu")
        layer.reset_parameters()
        self.assertTrue(all(torch.isfinite(p).all() for p in layer.parameters()))

    def test_install_rejects_mismatched_layer_config(self) -> None:
        model = Stack()
        model.config.dsa_indexer_types = ["full", "full", "shared"]
        with self.assertRaisesRegex(ValueError, "same Indexer Share plan"):
            install_indexer_share(model)

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
