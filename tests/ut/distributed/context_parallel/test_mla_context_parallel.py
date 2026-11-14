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
"""CPU adapter contracts and real HF replacement/checkpoint coverage for MLA CP."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import yaml
from torch.nn import functional as F

from hyper_parallel.components.modules import (
    MultiLatentAttention, GatedMultiLatentAttention, MultiLatentAttentionModule,
)
from hyper_parallel.components.modules.module_family import get_family_build_info
from hyper_parallel.components import functional
from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion
from hyper_parallel.components.functional.npu_fusion_attention import (
    _prepare_fusion_attention_context, resolve_packed_sequence_lengths,
)
from hyper_parallel.components.modules.attention.multi_latent_attention.cp_multi_latent_attention import (
    MLADimensions, MLACPPlan, _causal_partition, _kv_exchange_ranges,
)
from hyper_parallel.distributed.mesh import DistributedSetup, LogicalDeviceMesh, MeshContext
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.fully_shard.utils import apply_to_tensors
from hyper_parallel.components.modules.attention.multi_latent_attention.cp_mla_backward_overlap import (
    MLAOutputProjectionState,
)
from hyper_parallel.components.modules.attention.gated_multi_latent_attention.cp_gated_mla_backward_overlap import (
    GatedMLAInputState, GatedMLAOutputProjectionState,
)
from hyper_parallel.models.kimi_k3.adapter.mla import KimiMultiLatentAttention
from hyper_parallel.models.module_build import build_module_context, prepare_model_components
from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.config.parallelism import normalize_distributed_setup_overrides
from tests.common.mla_cp_utils import (
    TestCPMesh, make_cp_mla, make_mla, make_deepseek_model, replace_and_load_mla,
)


def _reference_attention(module, query, key, value, attention_mask=None, **kwargs):
    """CPU-only attention oracle for small layout and replacement contract tests."""
    scale = kwargs.get("scaling", module.scaling)
    causal = kwargs.get("is_causal", True)
    lengths = kwargs.get("actual_seq_qlen")
    key_lengths = kwargs.get("actual_seq_kvlen")
    allowed = attention_mask
    if causal and allowed is not None:
        allowed = allowed & torch.ones(query.shape[2], key.shape[2], dtype=torch.bool, device=query.device).tril()
    if lengths is None:
        return F.scaled_dot_product_attention(  # pylint: disable=not-callable
            query, key, value, attn_mask=allowed, dropout_p=0.0,
            is_causal=causal and allowed is None, scale=scale,
        ).transpose(1, 2), None
    batch, sequence = query.shape[0], query.shape[2]
    # Match the NPU TND batch-major token order before applying document ends.
    query, key, value = (tensor.transpose(0, 1).flatten(1, 2).unsqueeze(0) for tensor in (query, key, value))
    outputs = []
    query_begin = key_begin = 0
    for query_end, key_end in zip(lengths, key_lengths):
        query_part = query[:, :, query_begin:query_end]
        key_part = key[:, :, key_begin:key_end]
        value_part = value[:, :, key_begin:key_end]
        # CPU SDPA uses upper-left causality; rectangular CP segments need right alignment.
        rectangular = causal and query_part.shape[2] != key_part.shape[2]
        mask = None
        if rectangular:
            mask = torch.ones(query_part.shape[2], key_part.shape[2], dtype=torch.bool, device=query.device)
            mask = mask.tril(diagonal=key_part.shape[2] - query_part.shape[2])
        outputs.append(F.scaled_dot_product_attention(  # pylint: disable=not-callable
            query_part, key_part, value_part, attn_mask=mask,
            dropout_p=0.0, is_causal=causal and not rectangular, scale=scale,
        ).transpose(1, 2))
        query_begin, key_begin = query_end, key_end
    return torch.cat(outputs, dim=1).reshape(batch, sequence, query.shape[1], value.shape[-1]), None


def _reference_rotary_single(tensor, cos, sin, *, rotary_interleaved, use_fused_rotary_pos_emb):
    """CPU oracle for the native single-tensor RoPE kernel, preserving adjacent pairs."""
    del use_fused_rotary_pos_emb
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    if rotary_interleaved:
        real, imaginary = tensor[..., 0::2], tensor[..., 1::2]
        cos, sin = cos[..., 0::2], sin[..., 0::2]
        return torch.stack((real * cos - imaginary * sin, imaginary * cos + real * sin), dim=-1).flatten(-2)
    first, second = tensor.chunk(2, dim=-1)
    return tensor * cos + torch.cat((-second, first), dim=-1) * sin


class TestMLAContextParallel(unittest.TestCase):
    """Keep focused adapter contracts; distributed score math lives in the ST worker."""

    def test_local_callback_arguments(self) -> None:
        """
        Feature: Model-local MLA CP callbacks.
        Description: Install callable overrides and reject invalid callback arguments.
        Expectation: The CP execution preserves callback identity and rejects invalid values before installation.
        """
        rotary, attention = Mock(), Mock()
        module = make_mla()
        module = make_cp_mla(module, TestCPMesh(singleton=True), "expanded_ulysses",
                       rotary_fn=rotary, attention_interface=attention)
        self.assertIs(module.cp_runtime.rotary_fn, rotary)
        self.assertIs(module.cp_runtime.attention_interface, attention)
        for name in ("rotary_fn", "attention_interface"):
            module = make_mla()
            with self.subTest(name=name), self.assertRaisesRegex(TypeError, name):
                module = make_cp_mla(module, TestCPMesh(singleton=True), "expanded_ulysses", **{name: object()})
            self.assertFalse(hasattr(module, "cp_runtime"))

    def test_packed_boundaries_and_long_causal_mask(self) -> None:
        """
        Feature: Shared packed-sequence metadata and bounded causal masks.
        Description: Resolve aliases and Tensor boundaries, reject conflicts, and prepare S4096 attention.
        Expectation: Valid document ends agree and default causal masks remain 2048 by 2048.
        """
        for carrier in ({"cu_seqlens_q": (0, 3, 12), "cu_seqlens_kv": (0, 3, 12)},
                        SimpleNamespace(cu_seqlens_q=(0, 3, 12), cu_seqlens_kv=(0, 3, 12))):
            options = {"actual_seq_len": (3, 12), "packed_seq_params": carrier}
            self.assertEqual(resolve_packed_sequence_lengths(options, 12, 12), ([3, 12], [3, 12]))
            options["actual_seq_len"] = (4, 12)
            with self.assertRaisesRegex(ValueError, "conflicting"):
                resolve_packed_sequence_lengths(options, 12, 12)
        for lengths in ((0, 12), (3.5, 12), (True, 12)):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                resolve_packed_sequence_lengths({"actual_seq_len": lengths}, 12, 12)
        self.assertEqual(resolve_packed_sequence_lengths({"actual_seq_len": torch.tensor([3, 12])}, 12, 12),
                         ([3, 12], [3, 12]))
        query = torch.empty(1, 2, 4096, 8)
        context = _prepare_fusion_attention_context(torch.nn.Module(), query, query, query, None, {})
        self.assertEqual(context.attention_mask.shape, (2048, 2048))
        self.assertEqual(context.sparse_mode, 2)
        self.assertIsNone(context.valid_rows)

    def test_padding_mask_broadcasts_in_window_fallback(self) -> None:
        """Feature: Padding masks preserve batch and key ownership.

        Description: Exercise window and noncausal fallbacks with Boolean and integer masks.
        Expectation: Broadcast does not confuse batch with query or key axes.
        """
        query = torch.empty(2, 2, 4, 8)
        padding = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]])
        for options in ({"sliding_window": 2}, {"pre_tokens": 2}, {"is_causal": False}):
            for dtype in (torch.bool, torch.int64):
                with self.subTest(options=options, dtype=dtype):
                    context = _prepare_fusion_attention_context(
                        torch.nn.Module(), query, query, query, padding.to(dtype), options,
                    )
                    allowed = padding.bool()[:, None, None, :].expand(-1, 1, 4, -1)
                    if options.get("is_causal", True):
                        causal = torch.ones(4, 4, dtype=torch.bool).tril()
                        if "sliding_window" in options:
                            causal &= torch.ones(4, 4, dtype=torch.bool).triu(diagonal=-1)
                        elif "pre_tokens" in options:
                            causal &= torch.ones(4, 4, dtype=torch.bool).triu(diagonal=-2)
                        allowed = allowed & causal
                    torch.testing.assert_close(context.valid_rows, allowed.any(dim=-1))
                    blocked = ~allowed
                    blocked[..., 0] &= context.valid_rows
                    torch.testing.assert_close(context.attention_mask, blocked)
                    self.assertEqual(context.input_layout, "BNSD")
                    self.assertEqual(context.sparse_mode, 0)
        for options in ({"pre_tokens": 2}, {"sliding_window": 3}):
            without_mask = _prepare_fusion_attention_context(torch.nn.Module(), query, query, query, None, options)
            all_valid = _prepare_fusion_attention_context(
                torch.nn.Module(), query, query, query, torch.ones_like(padding), options,
            )
            torch.testing.assert_close(
                without_mask.attention_mask.expand_as(all_valid.attention_mask), all_valid.attention_mask,
            )

    def test_explicit_mask_preserves_query_axis(self) -> None:
        """Feature: Explicit MLA attention masks preserve the query axis.

        Description: Use a Q/K mask including a fully masked query and B == S.
        Expectation: Mask normalization keeps query semantics and the caller input intact.
        """
        mask = torch.ones(4, 4, dtype=torch.bool).tril()
        mask[1] = False
        original = mask.clone()
        for batch in (1, 2, 4):
            for causal in (False, True):
                for strategy in (None, "expanded_ulysses", "latent_kv_head"):
                    with self.subTest(batch=batch, causal=causal, strategy=strategy):
                        module = make_mla()
                        attention = Mock(side_effect=_reference_attention)
                        if strategy is None:
                            module.attention_interface = attention
                        else:
                            module = make_cp_mla(module, TestCPMesh(singleton=True), strategy,
                                           attention_interface=attention)
                        # Only bypass the device check; run actual MLA projections and mask preparation.
                        with patch("hyper_parallel.components.modules.attention.multi_latent_attention."
                              "cp_multi_latent_attention._validate_state"):
                            module(torch.randn(batch, 4, 12, dtype=torch.float64),
                                   attention_mask=mask, is_causal=causal)
                        args, options = attention.call_args
                        self.assertEqual(args[4].shape, (1, 1, 4, 4))
                        self.assertEqual(args[4].untyped_storage().data_ptr(), mask.untyped_storage().data_ptr())
                        context = _prepare_fusion_attention_context(*args, options)
                        self.assertEqual(context.input_layout, "BNSD")
                        self.assertEqual(context.attention_mask.shape, (1, 1, 4, 4))
                        torch.testing.assert_close(context.valid_rows, mask.any(dim=-1)[None, None])
                        torch.testing.assert_close(mask, original)

    def test_projection_preserves_module_hooks_and_parameter_identity(self) -> None:
        """
        Feature: Head-selective KV projection with managed module hooks.
        Description: Bind managed KV projection before hooks, then project selected heads and backpropagate.
        Expectation: Forward hooks run, parameter identities remain stable, and unused rows receive zero gradient.
        """
        module = make_mla()
        projection = module.kv_b_proj
        identities = [id(param) for param in projection.parameters()]
        module = make_cp_mla(module, TestCPMesh(singleton=True), "latent_kv_head")
        projection = module.kv_b_proj
        calls = []
        pre = projection.register_forward_pre_hook(lambda _module, _args: calls.append("pre"))
        post = projection.register_forward_hook(lambda _module, _args, _out: calls.append("post"))
        inputs = torch.randn(2, 3, 6, dtype=torch.float64)
        expected = torch.nn.functional.linear(  # pylint: disable=not-callable
            inputs, projection.weight[7:21], projection.bias[7:21],
        )
        actual = projection(inputs, mla_head_range=(1, 3))
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertEqual(calls, ["pre", "post"])
        self.assertEqual(identities, [id(param) for param in projection.parameters()])
        self.assertEqual(projection.weight.grad[:7].count_nonzero(), 0)
        pre.remove()
        post.remove()

    def test_invalid_metadata_and_options_fail_before_collectives(self) -> None:
        """
        Feature: MLA CP input validation before communication.
        Description: Supply invalid global boundaries, masks, cache options and local RoPE frequencies.
        Expectation: Each unsupported input raises a descriptive error before any collective executes.
        """
        module = make_mla()
        module = make_cp_mla(module, TestCPMesh(singleton=True), "latent_kv_head")
        inputs = torch.randn(1, 12, 12, dtype=torch.float64)
        cases = (
            ({"actual_seq_len": (3, 9)}, "query token count"),
            ({"attention_mask": torch.zeros(12, 12)}, "boolean"),
            ({"attention_mask": torch.ones(3, 12, dtype=torch.bool)}, "global Q/K"),
            ({"indices": torch.zeros(1)}, "Unsupported"),
            ({"past_key_values": object()}, "KV cache"),
            ({"actual_seq_len": (3, 12), "cu_seq_lens_q": (0, 4, 12)}, "conflicting"),
            ({"position_embeddings": (torch.ones(24, 2), torch.zeros(24, 2))}, "local tokens"),
        )
        with patch("hyper_parallel.components.modules.attention.multi_latent_attention."
                              "cp_multi_latent_attention.mla_all_gather") as collective:
            for kwargs, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        module(inputs, **kwargs)
            collective.assert_not_called()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            module(inputs, actual_seq_len=(0, 12))
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            module(inputs.expand(2, -1, -1), actual_seq_len=(24,))
        with self.assertRaisesRegex(ValueError, "NPU BF16/FP16"):
            module(inputs)
        query_partition = make_mla()
        mesh = unittest.mock.Mock(ndim=1, size=lambda: 8, get_local_rank=lambda: 0)
        query_partition = make_cp_mla(query_partition, mesh, "latent_kv_head", ulysses_degree=1)
        with patch("hyper_parallel.components.modules.attention.multi_latent_attention."
                              "cp_multi_latent_attention.mla_all_gather") as collective:
            for options in ({"is_causal": False}, {"attention_mask": torch.ones(96, 96, dtype=torch.bool)}):
                with self.assertRaisesRegex(ValueError, "causal attention without an explicit mask"):
                    query_partition(inputs, **options)
            collective.assert_not_called()

    def test_plan_and_member_fail_fast(self) -> None:
        """
        Feature: MLA CP plan and class construction contracts.
        Description: Exercise invalid head division, TP layout, dropout and unsupported projections.
        Expectation: Invalid configurations fail and valid TP2 CP2 computes one head per rank.
        """
        dims = MLADimensions(4, 5, 6, 3, 2, 4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            MLACPPlan(dims, 3)
        with self.assertRaisesRegex(ValueError, "divisible"):
            MLACPPlan(dims, 2, tp_degree=4)
        self.assertEqual(MLACPPlan(dims, 2, tp_degree=2).compute_heads, 1)
        self.assertEqual(MLACPPlan(dims, 8, ulysses_degree=1).compute_heads, 4)
        self.assertEqual(MLACPPlan(dims, 8, ulysses_degree=2).all_gather_degree, 4)
        for degree in (0, 3, 16, True, 1.5):
            with self.assertRaisesRegex(ValueError, "ulysses_degree"):
                MLACPPlan(dims, 8, ulysses_degree=degree)
        with self.assertRaisesRegex(ValueError, "TP1"):
            MLACPPlan(dims, 4, tp_degree=2, ulysses_degree=1)
        with self.assertRaisesRegex(ValueError, "overlap must be bool"):
            MLACPPlan(dims, 2, overlap="true")
        with self.assertRaisesRegex(ValueError, "kv_exchange"):
            MLACPPlan(dims, 4, ulysses_degree=1, kv_exchange="invalid")
        with self.assertRaisesRegex(ValueError, "query-partitioned"):
            MLACPPlan(dims, 4, kv_exchange="all_to_all_v")
        module = make_mla()
        self.assertFalse(hasattr(module, "cp_runtime"))
        mesh = TestCPMesh(singleton=True)
        module.attention_dropout = 0.1
        with self.assertRaisesRegex(ValueError, "zero dropout"):
            make_cp_mla(module, mesh, "expanded_ulysses")
        module.attention_dropout = 0.0
        module.q_b_proj = torch.nn.Sequential(module.q_b_proj)
        with self.assertRaisesRegex(ValueError, "plain Linear query"):
            make_cp_mla(module, mesh, "expanded_ulysses")

    def test_causal_query_partitions_match_global_attention(self) -> None:
        """Feature: Causal KV prefixes for AllGather and hybrid MLA.

        Description: Compare single/packed rectangular SDPA and independent batch samples with a global block mask.
        Expectation: Outputs and Q/K/V gradients agree for every query partition degree.
        """
        torch.manual_seed(19)
        for batch, ends in ((1, [16]), (1, [1, 7, 8, 13, 16]), (2, [16]), (4, [16])):
            query = torch.randn(batch, 2, 16, 4, dtype=torch.float64, requires_grad=True)
            key = torch.randn_like(query, requires_grad=True)
            value = torch.randn(batch, 2, 16, 3, dtype=torch.float64, requires_grad=True)
            allowed = torch.zeros(16, 16, dtype=torch.bool)
            for begin, end in zip([0] + ends[:-1], ends):
                allowed[begin:end, begin:end] = torch.ones(end - begin, end - begin, dtype=torch.bool).tril()
            scores = query @ key.transpose(-1, -2)
            expected = (scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ value).transpose(1, 2)
            upstream = torch.randn_like(expected)
            expected_grads = torch.autograd.grad(expected, (query, key, value), upstream, retain_graph=True)
            for degree in (1, 2, 4):
                pieces = []
                for rank, local_query in enumerate(query.chunk(degree, dim=2)):
                    partition = _causal_partition(ends, local_query.shape[2], rank, batch)
                    local_key, local_value = partition.select(key, 2), partition.select(value, 2)
                    query_ends, key_ends = partition.query_lengths, partition.key_lengths
                    self.assertEqual(local_key.untyped_storage().data_ptr(), key.untyped_storage().data_ptr())
                    pieces.append(_reference_attention(
                        SimpleNamespace(scaling=1.0), local_query, local_key, local_value,
                        actual_seq_qlen=query_ends, actual_seq_kvlen=key_ends, is_causal=True)[0])
                actual = torch.cat(pieces, dim=1)
                torch.testing.assert_close(actual, expected)
                actual_grads = torch.autograd.grad(actual, (query, key, value), upstream, retain_graph=True)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad)


    def test_selective_kv_routes_preserve_batch_and_sum_consumers(self) -> None:
        """Feature: Selective causal KV distribution with repeated consumers.

        Description: Reconstruct B2 payloads from peer intervals and compare with an independent causal mask.
        Expectation: Peer splits agree, token order is exact, and repeated consumer gradients sum at their source.
        """
        self.addCleanup(_kv_exchange_ranges.cache_clear)
        for ends in ((16,), (4, 8, 12, 16), (3, 9, 11, 16)):
            document_ids = torch.repeat_interleave(torch.arange(len(ends)), torch.diff(torch.tensor((0,) + ends)))
            allowed = (document_ids[:, None] == document_ids[None, :]).tril()
            for copies in (1, 2):
                source = torch.arange(32, dtype=torch.float64).reshape(2, 16, 1).requires_grad_()
                routes = [_kv_exchange_ranges(ends, 4, 4 * copies, copies, 4, rank) for rank in range(4)]
                actual_loss = expected_loss = 0
                for rank in range(4):
                    received = []
                    for peer in range(4):
                        begin, count = routes[peer][0][rank]
                        self.assertEqual(count, routes[rank][1][peer])
                        received.append(source[:, peer * 4 + begin:peer * 4 + begin + count])
                    actual = torch.cat(received, dim=1)
                    query_begin = rank // copies * 4 * copies
                    needed = allowed[query_begin:query_begin + 4 * copies].any(dim=0)
                    expected = source[:, needed]
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    actual_loss = actual_loss + (rank + 1) * actual.sum()
                    expected_loss = expected_loss + (rank + 1) * expected.sum()
                torch.testing.assert_close(torch.autograd.grad(actual_loss, source)[0],
                                           torch.autograd.grad(expected_loss, source)[0], atol=0, rtol=0)
                local_only = ends == (4, 8, 12, 16) and copies == 1
                self.assertTrue(all(route[2] == local_only for route in routes))


class TestMLAModel(unittest.TestCase):
    """Exercise real decoder objects and the family-specific fused latent rules."""

    def test_family_construction_preserves_parameters_and_nope(self) -> None:
        """Feature: Family conversion preserves source parameter and position contracts.

        Description: Convert ordinary/gated modules with loaded and frozen parameters.
        Expectation: Parameter identity, training state, initialization and NoPE remain correct.
        """
        template = make_deepseek_model()
        for factory, gated in ((MultiLatentAttention, False), (GatedMultiLatentAttention, True),
                               (KimiMultiLatentAttention, True)):
            with self.subTest(gated=gated):
                # Clone before family construction: build provenance is immutable metadata,
                # while this test compares independent parameter owners.
                converted, _ = replace_and_load_mla(deepcopy(template))
                source = converted.model.layers[0].self_attn.eval()
                if gated:
                    source.use_nope = source.use_output_gate = True
                    source.g_proj = torch.nn.Linear(source.config.hidden_size, source.num_heads * source.v_head_dim)
                direct = factory(config=source.config)
                source.q_b_proj.weight.requires_grad_(False)
                source.q_a_layernorm.weight = source.kv_a_layernorm.weight
                rebuilt = factory(module=source)
                self.assertEqual(get_family_build_info(rebuilt).resolved.variant, "local")
                self.assertIsInstance(rebuilt, MultiLatentAttentionModule)
                self.assertFalse(rebuilt.training)
                self.assertEqual(set(direct.state_dict()), set(rebuilt.state_dict()))
                self.assertEqual(rebuilt.make_transforms(), [])
                self.assertEqual(direct.rotary_interleaved, rebuilt.rotary_interleaved)
                self.assertEqual(direct.scaling, rebuilt.scaling)
                for name, parameter in source.named_parameters():
                    self.assertIs(rebuilt.get_parameter(name), parameter, name)
                self.assertFalse(rebuilt.q_b_proj.weight.requires_grad)
                self.assertIs(rebuilt.q_a_layernorm.weight, rebuilt.kv_a_layernorm.weight)
                initialized = {name: value.detach().clone() for name, value in rebuilt.named_parameters()}
                rebuilt.reset_parameters()
                for name, value in rebuilt.named_parameters():
                    torch.testing.assert_close(value, initialized[name], atol=0, rtol=0)
                if gated:
                    tensor = torch.randn(2, 3, 1, 4)
                    self.assertIs(rebuilt.apply_position(tensor, (tensor, tensor)), tensor)
                if gated:
                    with self.assertRaisesRegex(ValueError, "TP1"):
                        factory(module=source, context={"tp_size": 2})
                else:
                    self.assertEqual(factory(module=source, context={"tp_size": 2}).tp_degree, 2)

    def test_local_family_layout_owns_parameters_without_binding_cp(self) -> None:
        """Feature: Local family construction through the real logical layout planner.

        Description: Build both algorithms on meta with a DP axis and inactive CP/TP axes.
        Expectation: All parameters have one owner; CP selection creates concrete classes without runtime groups.
        """
        config = make_deepseek_model().config
        mesh = MeshContext(device_mesh=LogicalDeviceMesh(("dp", "cp", "tp"), (2, 1, 1)), dp_size=2)
        with patch("torch.distributed.new_group", side_effect=AssertionError("unexpected process group")):
            for factory in (MultiLatentAttention, GatedMultiLatentAttention, KimiMultiLatentAttention):
                with torch.device("meta"):
                    module = factory(config=config, mesh=mesh)
                layout = get_family_build_info(module).layout
                self.assertFalse(layout.applied)
                outputs = ("output",) if factory is KimiMultiLatentAttention else ("output", "attention_weights")
                self.assertEqual(layout.sharding.out_names, outputs)
                self.assertEqual(set(layout.sharding.params), dict(module.named_parameters()).keys())
                self.assertTrue(all(parameter.is_meta for parameter in module.parameters()))
                self.assertTrue(all("dp" not in axes for axes in layout.sharding.params.values()))
                self.assertFalse(hasattr(module, "cp_runtime"))
                cp_mesh = MeshContext(device_mesh=LogicalDeviceMesh(("cp", "tp"), (2, 1)), cp_size=2)
                with torch.device("meta"):
                    parallel = factory(config=config, mesh=cp_mesh, strategy="latent_kv_head", overlap=True)
                self.assertEqual(get_family_build_info(parallel).resolved.variant, "cp")
                self.assertEqual(parallel.cp_plan.cp_degree, 2)
                self.assertIsNone(parallel.cp_runtime)
                self.assertNotIn("forward", vars(parallel))
                self.assertNotIn("forward", vars(parallel.kv_b_proj))
                self.assertNotIn("forward", vars(parallel.o_proj))
                with torch.device("meta"):
                    deferred = factory(config=config)
                identities = {name: id(parameter) for name, parameter in deferred.named_parameters()}
                deferred, _ = prepare_model_components(
                    deferred, build_module_context(DistributedSetup(mesh_context=cp_mesh)),
                )
                self.assertEqual(get_family_build_info(deferred).resolved.variant, "cp")
                self.assertEqual(identities, {name: id(parameter) for name, parameter in deferred.named_parameters()})

    def test_projection_states_survive_fsdp_input_mapping(self) -> None:
        """Feature: FSDP preserves output layout values and shared gate-gradient ownership.

        Description: Map ordinary/gated arguments through the actual FSDP tensor visitor.
        Expectation: Rebuilt layouts keep their values; paired gate nodes retain one handoff identity.
        """
        layout = MLAOutputProjectionState(TestCPMesh(singleton=True), 4)
        mapped_layout = apply_to_tensors(lambda tensor: tensor.clone(), layout)
        self.assertIsNot(mapped_layout, layout)
        self.assertIs(mapped_layout.mesh, layout.mesh)
        self.assertEqual(mapped_layout.head_dim, layout.head_dim)
        gate_state = GatedMLAInputState(None, None, True)
        projection_state = GatedMLAOutputProjectionState(TestCPMesh(singleton=True), 4, input_gate=gate_state)
        gate = torch.ones(1, 2, 4)
        mapped = apply_to_tensors(lambda tensor: tensor.clone(), (gate, projection_state))
        self.assertIsNot(mapped[0], gate)
        self.assertIs(mapped[1].input_gate, gate_state)
        gradient = torch.randn_like(gate)
        mapped[1].input_gate.gradient = gradient
        self.assertIs(gate_state.take_gradient(), gradient)
        with self.assertRaisesRegex(RuntimeError, "did not release"):
            gate_state.take_gradient()

    def test_recipe_selects_family_without_forward_rewrite(self) -> None:
        """Feature: The shipped MLA recipe uses one topology-aware family request.

        Description: Resolve the actual YAML fragment and normalize the Trainer setup.
        Expectation: Only algorithm options enter the request; no method rewrite or runtime resources leak in.
        """
        recipe = Path(__file__).resolve().parents[4] / "hyper_parallel/models/deepseek_v3/recipes/mla_ulysses.yaml"
        raw = yaml.safe_load(recipe.read_text(encoding="utf-8"))
        raw.update(model={"_target_": "torch.nn.Identity"},
                   optimizer={"_target_": "torch.optim.AdamW", "lr": 1e-4})
        setup = DistributedSetup(mesh_context=MeshContext(tp_size=2, cp_size=4))
        normalize_distributed_setup_overrides(setup, resolve_config(raw))
        self.assertEqual(len(setup.module_replacement_requests), 1)
        request = setup.module_replacement_requests[0]
        self.assertIs(request.target, MultiLatentAttention)
        self.assertEqual(request.kwargs["strategy"], "expanded_ulysses")
        self.assertFalse({"module", "module_fqn", "context", "mesh"} & set(request.kwargs))
        self.assertFalse(setup.method_replacements)

    def test_family_final_layout_overrides_preserve_tp_ownership(self) -> None:
        """Feature: MLA uses the final explicit layout without bypassing parameter validation.

        Description: Override SP token ports while keeping the owned TP parameter layout.
        Expectation: Valid token layouts reach the plan; incompatible head/parameter sharding is rejected.
        """
        config = make_deepseek_model().config
        mesh = LogicalDeviceMesh(("cp", "tp"), (2, 2))
        with torch.device("meta"):
            module = MultiLatentAttention(config=config, context={"cp_size": 2, "tp_size": 2})
        tokens = {"cp": Shard(1), "tp": Replicate()}
        override = ModuleShardingSpec(
            in_src={"hidden_states": tokens}, in_dst={"hidden_states": tokens},
            out_src={"output": tokens}, out_dst={"output": tokens},
        )
        plan = ShardingPlanner(plan_overrides={"": override}).plan(
            module, mesh, tp_size=2, cp_size=2, sequence_parallel=True,
        )
        self.assertEqual(plan.modules[""].in_src["hidden_states"]["tp"], Replicate())
        self.assertEqual(plan.modules[""].params["q_b_proj.weight"]["tp"], Shard(0))
        self.assertEqual(plan.modules[""].params["o_proj.weight"]["tp"], Shard(1))
        self.assertEqual(plan.parallel_modules[""].binding_kind, "cp")
        parameters = deepcopy(plan.modules[""].params)
        parameters["kv_b_proj.weight"]["tp"] = Replicate()
        with self.assertRaisesRegex(ValueError, "MLA requires"):
            ShardingPlanner(plan_overrides={"": ModuleShardingSpec(params=parameters)}).plan(
                module, mesh, tp_size=2, cp_size=2,
            )
        missing_cp = {"tp": Replicate()}
        with self.assertRaisesRegex(ValueError, "non-TP token ownership"):
            ShardingPlanner(plan_overrides={"": ModuleShardingSpec(
                in_src={"hidden_states": missing_cp}, in_dst={"hidden_states": missing_cp},
            )}).plan(module, mesh, tp_size=2, cp_size=2)

    @patch.dict(functional.__dict__, {"apply_rotary_pos_emb_single": _reference_rotary_single})
    def test_replacement_decoder_and_checkpoint_roundtrip(self) -> None:
        """
        Feature: HF DeepSeek MLA replacement and checkpoint compatibility.
        Description: Replace a real decoder, restore original checkpoint keys, and compare forward and backward.
        Expectation: Checkpoint tensors are exact and logits and gradients match the HF model within tolerance.
        """
        torch.manual_seed(101)
        reference = make_deepseek_model()
        original_state = deepcopy(reference.state_dict())
        parallel, _ = replace_and_load_mla(deepcopy(reference))
        for layer in parallel.model.layers:
            layer.self_attn.attention_interface = _reference_attention
        restored = revert_weight_conversion(parallel, parallel.state_dict())
        self.assertEqual(set(restored), set(original_state))
        for name, value in original_state.items():
            torch.testing.assert_close(restored[name], value, atol=0, rtol=0)
        tokens = torch.randint(0, 128, (2, 16))
        allowed = torch.ones(1, 1, 16, 16, dtype=torch.bool).tril()
        expected = reference(input_ids=tokens, attention_mask=allowed, use_cache=False).logits
        actual = parallel(input_ids=tokens, attention_mask=allowed, use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        target = torch.randn_like(expected)
        (expected * target).mean().backward()
        (actual * target).mean().backward()
        gradients = revert_weight_conversion(
            parallel, {name: parameter.grad for name, parameter in parallel.named_parameters()},
        )
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(gradients[name], parameter.grad, atol=2e-7, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
