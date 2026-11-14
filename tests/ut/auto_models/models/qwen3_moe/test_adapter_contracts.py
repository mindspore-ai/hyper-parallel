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
"""Qwen adapter contracts and class-owned GQA family integration on CPU."""
# These per-case imports keep optional model implementations behind the CPU fixtures.
# pylint: disable=wrong-import-position,import-outside-toplevel

import importlib.util
import inspect
import sys
import types
import unittest
import unittest.mock
from types import SimpleNamespace

import torch
from torch import nn

from hyper_parallel.distributed.expert_parallel import recipes as ep_compute
from hyper_parallel.distributed.recipe_spec import (
    LOCAL_COMPUTE,
)
from hyper_parallel.models.qwen3_moe.adapter.conversion import (
    module_replacement as adapter_replacements,
)
from hyper_parallel.models.qwen3_moe.adapter.runtime import attention as adapter_attention
from hyper_parallel.models.qwen3_moe.adapter.runtime.attention import (
    run_qwen3_moe_flash_attention,
)
from hyper_parallel.models.qwen3_moe.adapter.distributed import (
    expert_parallel as adapter_expert_parallel,
)

_REPLACEMENT_NAMES = (
    "replace_qwen3_moe_rms_norm",
    "replace_qwen3_moe_flash_attention",
    "replace_qwen3_moe_grouped_experts",
)

_MESH_FAMILY = frozenset({"mesh", "tp_mesh", "cp_mesh", "ep_mesh"})


class TestLegacyEntryPointsRemoved(unittest.TestCase):
    """M5 gate: the legacy Qwen files and generic-surface branches are gone."""

    def test_legacy_model_modules_are_deleted(self):
        """``qwen3_moe_fusions`` / ``qwen3_moe_attention_common`` no longer exist.

        Stage 8 (S8c) deleted the whole ``auto_models.components`` package;
        asserting the package root is unresolvable subsumes the former
        per-module checks.
        """
        self.assertIsNone(
            importlib.util.find_spec("hyper_parallel.models.components")
        )

    def test_whole_block_sparse_moe_replacement_is_gone(self):
        """``replace_qwen3_moe_sparse_moe`` is a deliberate breaking removal."""
        self.assertFalse(hasattr(adapter_replacements, "replace_qwen3_moe_sparse_moe"))
        from hyper_parallel.models.qwen3_moe import adapter
        self.assertFalse(hasattr(adapter, "replace_qwen3_moe_sparse_moe"))

    def test_generic_ep_surface_has_no_qwen3_branch(self):
        """The qwen3moe archetype leaves the generic recipes/registry/exports."""
        self.assertFalse(hasattr(ep_compute, "qwen3moe_ep_compute_fn"))
        self.assertNotIn("qwen3moe_topk_router", ep_compute.EP_ARCHETYPES)
        self.assertNotIn("qwen3moe", ep_compute.EP_ARCHETYPE_SUGGESTIONS)
        self.assertNotIn("qwen3_moe", ep_compute.EP_ARCHETYPE_SUGGESTIONS)
        from hyper_parallel.distributed import expert_parallel
        self.assertFalse(hasattr(expert_parallel, "qwen3moe_ep_compute_fn"))

    def test_cp_wrappers_replaced_by_family(self):
        """Feature: No model-specific CP forward closures remain.
        Description: Resolve the deleted wrapper modules without importing them.
        Expectation: CP is selected by the family and both old modules are absent.
        """
        for suffix in ("context_parallel", "context_parallel_async"):
            self.assertIsNone(importlib.util.find_spec(
                "hyper_parallel.models.qwen3_moe.adapter.distributed." + suffix))


class TestReplacementFactoryContracts(unittest.TestCase):
    """``@module_replacement`` factories: mark, signature, single home."""

    def test_replacement_mark_and_signature(self):
        """Each factory carries the replacement mark and the exact signature."""
        for name in _REPLACEMENT_NAMES:
            with self.subTest(factory=name):
                factory = getattr(adapter_replacements, name)
                self.assertIs(getattr(factory, "_hp_module_replacement", None), True)
                parameters = list(inspect.signature(factory).parameters.values())
                self.assertEqual(
                    [param.name for param in parameters],
                    ["module", "module_fqn", "context"],
                )
                for param in parameters:
                    self.assertIs(param.kind, inspect.Parameter.KEYWORD_ONLY)
                    self.assertIs(param.default, inspect.Parameter.empty)

    def test_adapter_package_surface_exports_the_factories(self):
        """The adapter package surface exports the replacement objects."""
        from hyper_parallel.models.qwen3_moe import adapter
        for name in _REPLACEMENT_NAMES:
            with self.subTest(factory=name):
                self.assertIs(getattr(adapter, name), getattr(adapter_replacements, name))


class TestAdapterHomeContracts(unittest.TestCase):
    """Single-home pins: attention contract and lazy spec providers."""

    def test_attention_contract_lives_in_the_adapter(self):
        """The mask/cache contract and the function-form forward are callables."""
        self.assertTrue(callable(adapter_attention.run_qwen3_moe_flash_attention))
        # the function-form forward keeps the historical contract
        self.assertTrue(callable(adapter_attention.qwen3_moe_flash_attention_forward))

    def test_adapter_spec_providers_are_wired(self):
        """The registered qwen3_moe spec resolves its providers lazily."""
        from hyper_parallel.models.registry import get_model_adapter
        spec = get_model_adapter("qwen3_moe")
        self.assertIs(spec.replacements(), adapter_replacements)
        self.assertIs(spec.attention(), adapter_attention)
        self.assertIsNone(spec.context_parallel)
        self.assertIs(spec.expert_parallel(), adapter_expert_parallel)


class TestGQAFamilyContracts(unittest.TestCase):
    """Minimal family promotion, layout and packed-partition regressions."""

    @staticmethod
    def _source(gated=False):
        source = nn.Module()
        source.config = SimpleNamespace(hidden_size=16, num_attention_heads=4,
                                        num_key_value_heads=2, head_dim=4)
        source.q_proj = nn.Linear(16, 32 if gated else 16)
        source.k_proj = nn.Linear(16, 8)
        source.v_proj = nn.Linear(16, 8)
        source.o_proj = nn.Linear(16, 16)
        source.q_norm = nn.LayerNorm(4)
        source.k_norm = nn.LayerNorm(4)
        return source

    def test_promotion_preserves_parameters_and_fixed_representation(self):
        """Feature: Class-owned CP with stable parameter/FSDP boundaries.
        Description: Promote ordinary, split and gated local members to CP2/TP2.
        Expectation: Weights keep identity and overlap never changes their names.
        """
        from hyper_parallel.components.modules import (
            GroupedQueryAttention, SplitGroupedQueryAttention, GatedGroupedQueryAttention,
        )
        from hyper_parallel.components.modules.module_family import get_family_build_info
        for factory in (GroupedQueryAttention, SplitGroupedQueryAttention, GatedGroupedQueryAttention):
            local = factory(module=self._source(factory is GatedGroupedQueryAttention))
            parallel = factory(module=local, context={"cp_size": 2, "tp_size": 2}, overlap=True, ulysses_degree=1)
            self.assertEqual(get_family_build_info(parallel).resolved.variant, "cp")
            self.assertEqual(set(dict(local.named_parameters())), set(dict(parallel.named_parameters())))
            for name, parameter in parallel.named_parameters():
                self.assertIs(parameter, local.get_parameter(name))
            self.assertNotIn("forward", vars(parallel))
            if factory is not SplitGroupedQueryAttention:
                source = self._source(factory is GatedGroupedQueryAttention)
                parallel.num_key_value_heads = 1  # TP binding localizes runtime heads, not checkpoint layout.
                transform = parallel.make_transforms()[0]
                converted = transform.operations[0].convert(
                    dict(source.named_parameters()), transform.source_patterns, transform.target_patterns)
                self.assertEqual(converted["linear_qkv.weight"].shape, local.linear_qkv.weight.shape)
        with self.assertRaisesRegex(ValueError, "divisible by TP"):
            GroupedQueryAttention(module=self._source(), context={"tp_size": 3})

    def test_family_owns_qkv_groups_norm_and_output_bias_layout(self):
        """Feature: Complete TP/SP layout without KV replication.
        Description: Build the final layout for a biased GQA module at TP2 CP2.
        Expectation: QKV rows and output columns shard; per-head Norm and O bias replicate.
        """
        from hyper_parallel.components.modules import GroupedQueryAttention
        from hyper_parallel.components.modules.module_family import FamilyShardingContext, get_family_build_info
        from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
        module = GroupedQueryAttention(module=self._source())
        info = get_family_build_info(module)
        context = FamilyShardingContext(("cp", "tp"), {"cp": 2, "tp": 2}, {}, sequence_parallel=True)
        spec = info.resolved.sharding(module=module, module_fqn="", member=info.resolved, context=context)
        self.assertEqual(spec.params["linear_qkv.weight"]["tp"], Shard(0))
        self.assertEqual(spec.params["o_proj.weight"]["tp"], Shard(1))
        self.assertEqual(spec.params["o_proj.bias"]["tp"], Replicate())
        self.assertEqual(spec.params["q_norm.weight"]["tp"], Replicate())
        self.assertEqual(spec.in_src["hidden_states"]["tp"], Shard(1))

    def test_packed_query_fragments_preserve_document_visibility(self):
        """Feature: Packed AG consumes local Q and correct global document prefixes.
        Description: Include a document crossing a batch boundary with CP2.
        Expectation: Every selected key belongs to the query's own document and causal prefix.
        """
        from hyper_parallel.components.modules.attention.grouped_query_attention.cp_grouped_query_attention import (
            _query_documents,
        )
        lengths = (3, 10, 16)
        ranges, q_lengths, k_lengths = _query_documents(lengths, 2, 8, 4, 1, True)
        self.assertEqual(ranges, ((3, 8), (10, 16)))
        self.assertEqual(q_lengths, (4, 8))
        self.assertEqual(k_lengths, (5, 11))


class TestEpFactoryContract(unittest.TestCase):
    """``@local_compute`` EP factory at its single adapter home."""

    def test_local_compute_mark_and_signature(self):
        """qwen3moe_ep_compute_fn declares module + mesh family, keyword-only."""
        factory = adapter_expert_parallel.qwen3moe_ep_compute_fn
        meta = getattr(factory, "_injection_meta", None)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.kind, LOCAL_COMPUTE)
        self.assertEqual(meta.context, {"module"} | _MESH_FAMILY)
        parameters = list(inspect.signature(factory).parameters.values())
        self.assertEqual(
            [param.name for param in parameters],
            ["module", "mesh", "tp_mesh", "cp_mesh", "ep_mesh", "use_grouped_gemm",
             "fix_router"],
        )
        for param in parameters:
            self.assertIs(param.kind, inspect.Parameter.KEYWORD_ONLY)
        for param in parameters[:-2]:
            self.assertIs(param.default, inspect.Parameter.empty)
        # Both switches are opt-in: an existing recipe keeps the real router and the
        # eager per-expert loop unless it asks otherwise.
        self.assertIs(parameters[-2].default, False)
        self.assertIs(parameters[-1].default, False)

    def test_ep_factory_validates_ep_mesh(self):
        """Calling the factory without an ep_mesh fails with the teaching error."""
        with self.assertRaisesRegex(ValueError, "built without an ep_mesh"):
            adapter_expert_parallel.qwen3moe_ep_compute_fn(
                module=nn.Module(), mesh=None, tp_mesh=None, cp_mesh=None,
                ep_mesh=None,
            )


class _AttentionNpuStub(types.ModuleType):
    """Recording ``torch_npu`` stub for the flash-attention kernel path."""

    def __init__(self):
        super().__init__("torch_npu")
        self.calls = []

    def npu_fusion_attention(self, query, key, value, head_num, input_layout, **kwargs):
        """Record the kernel kwargs and echo the query as attention output."""
        del key, value
        self.calls.append(dict(kwargs, head_num=head_num, input_layout=input_layout))
        return query, None


class TestFlashAttentionMaskBranches(unittest.TestCase):
    """Mask/cache branches of ``run_qwen3_moe_flash_attention`` (Ascend kernel)."""

    def _run(self, query, key, value, attention_mask, module=None, **options):
        """Run the attention helper with a recording ``torch_npu`` stub."""
        stub = _AttentionNpuStub()
        with unittest.mock.patch.dict(sys.modules, {"torch_npu": stub}):
            output, weights = run_qwen3_moe_flash_attention(
                module if module is not None else nn.Module(),
                query,
                key,
                value,
                attention_mask,
                **options,
            )
        return stub, output, weights

    def test_packed_lengths_reach_tnd_kernel(self):
        """Feature: Qwen no longer discards packed metadata.
        Description: Supply two documents to the adapter's common FA delegate.
        Expectation: TND and both cumulative-length arrays reach the kernel unchanged.
        """
        query = torch.randn(1, 2, 4, 8)
        stub, output, weights = self._run(query, query, query, None, actual_seq_len=(1, 4))
        call = stub.calls[0]
        self.assertEqual(call["input_layout"], "TND")
        self.assertEqual(list(call["actual_seq_qlen"]), [1, 4])
        self.assertEqual(list(call["actual_seq_kvlen"]), [1, 4])
        self.assertEqual(output.shape, (1, 4, 2, 8))
        self.assertIsNone(weights)
        stub, _, _ = self._run(query, query, query, None, actual_seq_len=(1, 4), sliding_window=2)
        self.assertEqual(stub.calls[0]["sparse_mode"], 4)
        self.assertEqual(stub.calls[0]["pre_tockens"], 1)

    def test_bool_mask_is_inverted_and_sliced(self):
        """A Transformers-convention bool mask is inverted (True = blocked) for the
        Ascend kernel, sliced to the key length, and selects sparse_mode=0."""
        query = torch.randn(1, 2, 4, 8)
        key = torch.randn(1, 2, 4, 8)
        full_mask = torch.zeros(1, 1, 4, 6, dtype=torch.bool)
        full_mask[..., :2] = True
        stub, _, _ = self._run(query, key, key, full_mask)
        call = stub.calls[0]
        self.assertEqual(call["sparse_mode"], 0)
        self.assertTrue(torch.equal(
            call["atten_mask"], torch.logical_not(full_mask[:, :, :, :4]) | torch.ones(4, 4, dtype=torch.bool).triu(1)
        ))
        # is_causal default + self-attention length match adds next_tockens=0
        self.assertEqual(call["next_tockens"], 0)

    def test_float_mask_is_cast_without_inversion(self):
        """A float (additive-style) mask is cast to bool WITHOUT inversion."""
        query = torch.randn(1, 2, 4, 8)
        float_mask = torch.zeros(1, 1, 4, 4)
        float_mask[..., -1] = float("-inf")
        stub, _, _ = self._run(query, query, query, float_mask)
        call = stub.calls[0]
        self.assertEqual(call["sparse_mode"], 0)
        self.assertTrue(torch.equal(
            call["atten_mask"], float_mask.bool() | torch.ones(4, 4, dtype=torch.bool).triu(1)
        ))

    def test_next_tockens_requires_causal_self_attention(self):
        """next_tockens=0 is withheld for cross-attention or is_causal=False."""
        query = torch.randn(1, 2, 2, 8)
        key = torch.randn(1, 2, 4, 8)
        mask = torch.ones(1, 1, 2, 4, dtype=torch.bool)
        stub, _, _ = self._run(query, key, key, mask)
        self.assertEqual(stub.calls[0]["next_tockens"], 0)
        non_causal = nn.Module()
        non_causal.is_causal = False
        self_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        stub, _, _ = self._run(key, key, key, self_mask, module=non_causal)
        self.assertEqual(stub.calls[0]["next_tockens"], 2147483647)


if __name__ == "__main__":
    unittest.main()
