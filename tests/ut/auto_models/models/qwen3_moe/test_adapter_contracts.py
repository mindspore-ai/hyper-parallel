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
"""Contract pinning for the Qwen3-MoE adapter homes (post-M5).

Post-M5 state (adjust doc §5.1-§5.3, M5): the legacy
``components/models/qwen3_moe_fusions.py`` and
``qwen3_moe_attention_common.py`` are deleted, the whole-block
``replace_qwen3_moe_sparse_moe`` is gone (breaking change — superseded by a
``*.mlp.experts`` replacement onto ``modules.GroupedExperts``), and every
Qwen3-MoE rule lives at exactly one adapter home:

* the ``@module_replacement`` factories in
  ``models/qwen3_moe/adapter/replacements.py`` (declaring the generic
  ``modules.RMSNorm`` / ``modules.GQAAttention`` entries);
* the Qwen mask/cache attention contract in
  ``models/qwen3_moe/adapter/attention.py``;
* the fused ``@inner_wrapper`` CP wrappers in
  ``adapter/distributed/context_parallel.py`` and the async (whole-forward,
  HF-structure) wrappers in ``adapter/distributed/context_parallel_async.py``
  — all returning ``_ForwardRewriteRequest`` (they never assign
  ``target.forward``);
* the ``@local_compute`` EP archetype factory in
  ``adapter/distributed/expert_parallel.py`` on top of the generic
  ``build_ep_compute`` skeleton.

Pinned here (Gate-1: no NPU, no process group):

* the legacy import paths, the generic-surface Qwen3 exports and the
  ``EP_ARCHETYPES["qwen3moe_topk_router"]`` entry are gone;
* marks, signatures, fail-fast validation and the rewrite-request return
  contract of every adapter wrapper/factory;
* the attention mask/cache branches of ``run_qwen3_moe_flash_attention``
  driven through a recording ``torch_npu`` stub.
"""
# pylint: disable=wrong-import-position

import importlib.util
import inspect
import os
import sys
import types
import unittest
import unittest.mock
from types import SimpleNamespace

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch
from torch import nn

from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed.expert_parallel import recipes as ep_compute
from hyper_parallel.distributed.recipe_spec import (
    INNER_WRAPPER,
    LOCAL_COMPUTE,
)
from hyper_parallel.models.qwen3_moe.adapter import (
    attention as adapter_attention,
)
from hyper_parallel.models.qwen3_moe.adapter import (
    replacements as adapter_replacements,
)
from hyper_parallel.models.qwen3_moe.adapter.attention import (
    run_qwen3_moe_flash_attention,
)
from hyper_parallel.models.qwen3_moe.adapter.distributed import (
    context_parallel as adapter_context_parallel,
)
from hyper_parallel.models.qwen3_moe.adapter.distributed import (
    context_parallel_async as adapter_context_parallel_async,
)
from hyper_parallel.models.qwen3_moe.adapter.distributed import (
    expert_parallel as adapter_expert_parallel,
)
from tests.common.mark_utils import arg_mark
from tests.ut.auto_models.distributed.conftest import FakeSubMesh

_REPLACEMENT_NAMES = (
    "replace_qwen3_moe_rms_norm",
    "replace_qwen3_moe_flash_attention",
    "replace_qwen3_moe_grouped_experts",
)

_CP_WRAPPER_NAMES = (
    "qwen3_moe_flash_attention_cp_wrapper",
    "qwen3_moe_flash_attention_cp_mask_wrapper",
    "qwen3_moe_flash_attention_ulysses_cp_wrapper",
)

_ASYNC_CP_WRAPPER_NAMES = (
    "qwen3_moe_async_colossal_cp_wrapper",
    "qwen3_moe_async_ulysses_cp_wrapper",
    "qwen3_moe_async_hybrid_cp_wrapper",
)

_MESH_FAMILY = frozenset({"mesh", "tp_mesh", "cp_mesh", "ep_mesh"})


class TestLegacyEntryPointsRemoved(unittest.TestCase):
    """M5 gate: the legacy Qwen files and generic-surface branches are gone."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_legacy_model_modules_are_deleted(self):
        """``qwen3_moe_fusions`` / ``qwen3_moe_attention_common`` no longer exist.

        Stage 8 (S8c) deleted the whole ``auto_models.components`` package;
        asserting the package root is unresolvable subsumes the former
        per-module checks.
        """
        self.assertIsNone(
            importlib.util.find_spec("hyper_parallel.models.components")
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_whole_block_sparse_moe_replacement_is_gone(self):
        """``replace_qwen3_moe_sparse_moe`` is a deliberate breaking removal."""
        self.assertFalse(hasattr(adapter_replacements, "replace_qwen3_moe_sparse_moe"))
        from hyper_parallel.models.qwen3_moe import adapter
        self.assertFalse(hasattr(adapter, "replace_qwen3_moe_sparse_moe"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_generic_ep_surface_has_no_qwen3_branch(self):
        """The qwen3moe archetype leaves the generic recipes/registry/exports."""
        self.assertFalse(hasattr(ep_compute, "qwen3moe_ep_compute_fn"))
        self.assertNotIn("qwen3moe_topk_router", ep_compute.EP_ARCHETYPES)
        self.assertNotIn("qwen3moe", ep_compute.EP_ARCHETYPE_SUGGESTIONS)
        self.assertNotIn("qwen3_moe", ep_compute.EP_ARCHETYPE_SUGGESTIONS)
        from hyper_parallel.distributed import expert_parallel
        self.assertFalse(hasattr(expert_parallel, "qwen3moe_ep_compute_fn"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_generic_cp_surface_has_no_qwen3_branch(self):
        """The async Qwen3 wrappers leave the generic CP surface."""
        from hyper_parallel.distributed import context_parallel
        from hyper_parallel.distributed.context_parallel import wrappers
        for name in _ASYNC_CP_WRAPPER_NAMES:
            with self.subTest(wrapper=name):
                self.assertFalse(hasattr(context_parallel, name))
                self.assertFalse(hasattr(wrappers, name))
        self.assertFalse(any("qwen3" in name for name in wrappers.INNER_WRAPPER_REGISTRY))


class TestReplacementFactoryContracts(unittest.TestCase):
    """``@module_replacement`` factories: mark, signature, single home."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_adapter_package_surface_exports_the_factories(self):
        """The adapter package surface exports the replacement objects."""
        from hyper_parallel.models.qwen3_moe import adapter
        for name in _REPLACEMENT_NAMES:
            with self.subTest(factory=name):
                self.assertIs(getattr(adapter, name), getattr(adapter_replacements, name))


class TestAdapterHomeContracts(unittest.TestCase):
    """Single-home pins: attention contract and lazy spec providers."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_attention_contract_lives_in_the_adapter(self):
        """The mask/cache contract and the function-form forward are callables."""
        self.assertTrue(callable(adapter_attention.run_qwen3_moe_flash_attention))
        self.assertTrue(callable(adapter_attention._get_compressed_causal_mask))  # pylint: disable=protected-access
        # the function-form forward keeps the historical contract
        self.assertTrue(callable(adapter_attention.qwen3_moe_flash_attention_forward))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_adapter_spec_providers_are_wired(self):
        """The registered qwen3_moe spec resolves its providers lazily."""
        from hyper_parallel.models.registry import get_model_adapter
        spec = get_model_adapter("qwen3_moe")
        self.assertIs(spec.replacements(), adapter_replacements)
        self.assertIs(spec.attention(), adapter_attention)
        self.assertIs(spec.context_parallel(), adapter_context_parallel)
        self.assertIs(spec.expert_parallel(), adapter_expert_parallel)


class TestCpWrapperContracts(unittest.TestCase):
    """Fused ``@inner_wrapper`` CP wrappers: mark, signature, validation."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_inner_wrapper_mark_and_signature(self):
        """Each wrapper declares the full wrapper context and returns a request."""
        for name in _CP_WRAPPER_NAMES:
            with self.subTest(wrapper=name):
                wrapper = getattr(adapter_context_parallel, name)
                meta = getattr(wrapper, "_injection_meta", None)
                self.assertIsNotNone(meta)
                self.assertEqual(meta.kind, INNER_WRAPPER)
                self.assertEqual(meta.context, {"target_module"} | _MESH_FAMILY)
                parameters = list(inspect.signature(wrapper).parameters.values())
                self.assertEqual(
                    [param.name for param in parameters],
                    ["target_module", "mesh", "tp_mesh", "cp_mesh", "ep_mesh"],
                )
                for param in parameters:
                    self.assertIs(param.default, inspect.Parameter.empty)
                # the wrappers return a _ForwardRewriteRequest instead of
                # assigning target.forward (`from __future__ import
                # annotations` stores the annotation as a string)
                self.assertEqual(
                    inspect.signature(wrapper).return_annotation,
                    "_ForwardRewriteRequest",
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_wrappers_require_active_cp_mesh(self):
        """cp_mesh None or size 1 fails fast before any forward is touched."""
        for name in _CP_WRAPPER_NAMES:
            wrapper = getattr(adapter_context_parallel, name)
            with self.subTest(wrapper=name, cp_mesh="none"):
                with self.assertRaisesRegex(ValueError, "requires an active CP mesh"):
                    wrapper(nn.Module(), None, None, None, None)
            with self.subTest(wrapper=name, cp_mesh="size1"):
                size_one = FakeSubMesh("cpu", (1,), ("cp",))
                with self.assertRaisesRegex(ValueError, "requires an active CP mesh"):
                    wrapper(nn.Module(), None, None, size_one, None)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_wrappers_require_flash_attention_forward_first(self):
        """A target without the fused attention forward is rejected."""
        cp_mesh = FakeSubMesh("cpu", (2,), ("cp",))
        target = nn.Module()
        original_forward = target.forward
        for name in (
            "qwen3_moe_flash_attention_cp_wrapper",
            "qwen3_moe_flash_attention_cp_mask_wrapper",
        ):
            with self.subTest(wrapper=name):
                with self.assertRaisesRegex(
                    ValueError, "replace_qwen3_moe_flash_attention first"
                ):
                    getattr(adapter_context_parallel, name)(target, None, None, cp_mesh, None)
        self.assertIs(target.forward.__func__, original_forward.__func__)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_ulysses_wrapper_validates_head_counts(self):
        """Pure Ulysses requires config head counts divisible by the CP degree."""
        cp_mesh = FakeSubMesh("cpu", (2,), ("cp",))
        wrapper = adapter_context_parallel.qwen3_moe_flash_attention_ulysses_cp_wrapper
        with self.assertRaisesRegex(ValueError, "requires target_module.config"):
            wrapper(nn.Module(), None, None, cp_mesh, None)
        target = nn.Module()
        target.config = SimpleNamespace(num_attention_heads=3, num_key_value_heads=2)
        with self.assertRaisesRegex(ValueError, "divisible"):
            wrapper(target, None, None, cp_mesh, None)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_wrappers_return_rewrite_requests_without_mutating_forward(self):
        """The adapter wrappers never assign ``target.forward`` (05 §15.2.3).

        Each wrapper returns one ``_ForwardRewriteRequest`` for the target
        whose companion attribute swaps ``attention_interface`` for the
        CP-orchestrated interface; the target's forward and interface stay
        untouched until the generic forward rewriter commits the request.
        """
        cp_mesh = FakeSubMesh("cpu", (2,), ("cp",))
        for name in _CP_WRAPPER_NAMES:
            with self.subTest(wrapper=name):
                target = nn.Module()
                target.config = SimpleNamespace(
                    num_attention_heads=4, num_key_value_heads=2
                )
                sentinel_interface = lambda *args, **kwargs: None  # noqa: E731
                target.attention_interface = sentinel_interface
                calls = []

                def recording_forward(*args, **kwargs):
                    calls.append((args, kwargs))
                    return "sentinel-output"

                target.forward = recording_forward
                request = getattr(adapter_context_parallel, name)(
                    target, None, None, cp_mesh, None
                )
                self.assertIsInstance(request, _ForwardRewriteRequest)
                self.assertIs(request.target, target)
                self.assertTrue(callable(request.forward))
                self.assertEqual(set(request.companion_attrs), {"attention_interface"})
                self.assertTrue(callable(request.companion_attrs["attention_interface"]))
                # nothing is installed until the rewriter commits the request
                self.assertIs(target.forward, recording_forward)
                self.assertIs(target.attention_interface, sentinel_interface)
                # the returned forward passes through to the original forward
                self.assertIs(request.forward("a", key=1), "sentinel-output")
                self.assertEqual(calls, [(("a",), {"key": 1})])


def _hf_structure_attention():
    """A module carrying the HF-structure attributes the async wrappers need."""
    target = nn.Module()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(target, name, nn.Linear(4, 4, bias=False))
    target.q_norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    target.k_norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    target.head_dim = 4
    target.scaling = 0.5
    target.config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2)
    return target


class TestAsyncCpWrapperContracts(unittest.TestCase):
    """Async (HF-structure) ``@inner_wrapper`` CP wrappers at their M5 home."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_inner_wrapper_mark_and_signature(self):
        """Each async wrapper declares the wrapper context (+ ulysses_degree)."""
        expected_params = {
            "qwen3_moe_async_colossal_cp_wrapper":
                ["target_module", "mesh", "tp_mesh", "cp_mesh", "ep_mesh"],
            "qwen3_moe_async_ulysses_cp_wrapper":
                ["target_module", "mesh", "tp_mesh", "cp_mesh", "ep_mesh"],
            "qwen3_moe_async_hybrid_cp_wrapper":
                ["target_module", "mesh", "tp_mesh", "cp_mesh", "ep_mesh",
                 "ulysses_degree"],
        }
        for name in _ASYNC_CP_WRAPPER_NAMES:
            with self.subTest(wrapper=name):
                wrapper = getattr(adapter_context_parallel_async, name)
                meta = getattr(wrapper, "_injection_meta", None)
                self.assertIsNotNone(meta)
                self.assertEqual(meta.kind, INNER_WRAPPER)
                self.assertEqual(meta.context, {"target_module"} | _MESH_FAMILY)
                parameters = list(inspect.signature(wrapper).parameters.values())
                self.assertEqual(
                    [param.name for param in parameters], expected_params[name]
                )
                for param in parameters:
                    self.assertIs(param.default, inspect.Parameter.empty)
                self.assertEqual(
                    inspect.signature(wrapper).return_annotation,
                    "_ForwardRewriteRequest",
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_async_wrappers_require_active_cp_mesh(self):
        """cp_mesh None or size 1 fails fast before any forward is touched."""
        for name in _ASYNC_CP_WRAPPER_NAMES:
            wrapper = getattr(adapter_context_parallel_async, name)
            extra = (2,) if name == "qwen3_moe_async_hybrid_cp_wrapper" else ()
            with self.subTest(wrapper=name, cp_mesh="none"):
                with self.assertRaisesRegex(ValueError, "requires an active CP mesh"):
                    wrapper(nn.Module(), None, None, None, None, *extra)
            with self.subTest(wrapper=name, cp_mesh="size1"):
                size_one = FakeSubMesh("cpu", (1,), ("cp",))
                with self.assertRaisesRegex(ValueError, "requires an active CP mesh"):
                    wrapper(nn.Module(), None, None, size_one, None, *extra)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_async_wrappers_require_hf_attention_structure(self):
        """A target without q_proj/k_proj/v_proj/q_norm/k_norm is rejected."""
        cp_mesh = FakeSubMesh("cpu", (2,), ("cp",))
        for name in (
            "qwen3_moe_async_colossal_cp_wrapper",
            "qwen3_moe_async_ulysses_cp_wrapper",
        ):
            with self.subTest(wrapper=name):
                # config present so head validation passes; the HF projection
                # structure is what's missing
                target = nn.Module()
                target.config = SimpleNamespace(
                    num_attention_heads=4, num_key_value_heads=2
                )
                with self.assertRaisesRegex(
                    TypeError, "requires an attention module with attributes"
                ):
                    getattr(adapter_context_parallel_async, name)(
                        target, None, None, cp_mesh, None
                    )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_async_ulysses_wrapper_validates_head_counts(self):
        """Async Pure Ulysses requires head counts divisible by the CP degree."""
        cp_mesh = FakeSubMesh("cpu", (2,), ("cp",))
        wrapper = adapter_context_parallel_async.qwen3_moe_async_ulysses_cp_wrapper
        target = _hf_structure_attention()
        target.config = SimpleNamespace(num_attention_heads=3, num_key_value_heads=2)
        with self.assertRaisesRegex(ValueError, "divisible"):
            wrapper(target, None, None, cp_mesh, None)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_async_wrappers_return_rewrite_requests_without_mutating_forward(self):
        """M5: the async wrappers return a request and never assign forward.

        Unlike the fused wrappers there are no companion attributes — the
        request's forward IS the whole replacement forward (the async
        overlap-scheduled pipeline), so nothing passes through to the
        original forward.
        """
        cp_mesh = FakeSubMesh("cpu", (2,), ("cp",))
        for name in (
            "qwen3_moe_async_colossal_cp_wrapper",
            "qwen3_moe_async_ulysses_cp_wrapper",
        ):
            with self.subTest(wrapper=name):
                target = _hf_structure_attention()

                def recording_forward(*args, **kwargs):
                    return "sentinel-output"

                target.forward = recording_forward
                request = getattr(adapter_context_parallel_async, name)(
                    target, None, None, cp_mesh, None
                )
                self.assertIsInstance(request, _ForwardRewriteRequest)
                self.assertIs(request.target, target)
                self.assertTrue(callable(request.forward))
                self.assertFalse(request.companion_attrs)
                # nothing is installed until the rewriter commits the request
                self.assertIs(target.forward, recording_forward)


class TestEpFactoryContract(unittest.TestCase):
    """``@local_compute`` EP factory at its single adapter home."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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
            ["module", "mesh", "tp_mesh", "cp_mesh", "ep_mesh", "use_grouped_gemm"],
        )
        for param in parameters:
            self.assertIs(param.kind, inspect.Parameter.KEYWORD_ONLY)
        for param in parameters[:-1]:
            self.assertIs(param.default, inspect.Parameter.empty)
        self.assertIs(parameters[-1].default, False)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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

    def npu_fusion_attention(self, query, key, value, **kwargs):
        """Record the kernel kwargs and echo the query as attention output."""
        self.calls.append(kwargs)
        return query, None


class TestFlashAttentionMaskBranches(unittest.TestCase):
    """Mask/cache branches of ``run_qwen3_moe_flash_attention`` (Ascend kernel)."""

    def _run(self, query, key, value, attention_mask, module=None):
        stub = _AttentionNpuStub()
        with unittest.mock.patch.dict(sys.modules, {"torch_npu": stub}):
            output, weights = run_qwen3_moe_flash_attention(
                module if module is not None else nn.Module(),
                query,
                key,
                value,
                attention_mask,
            )
        return stub, output, weights

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_none_mask_uses_cached_compressed_causal_mask(self):
        """mask=None selects sparse_mode=2 with the cached 2048 compressed mask."""
        query = torch.randn(1, 2, 4, 8)
        stub, output, weights = self._run(query, query, query, None)
        self.assertEqual(stub.calls[0]["sparse_mode"], 2)
        mask = stub.calls[0]["atten_mask"]
        self.assertEqual(mask.shape, (2048, 2048))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(mask[0, 1])
        self.assertFalse(mask[1, 0])
        self.assertFalse(mask.diagonal().any())
        # per-device cache: a second call reuses the identical mask object
        stub_two, _, _ = self._run(query, query, query, None)
        self.assertIs(stub_two.calls[0]["atten_mask"], mask)
        # the kernel output is transposed back to (B, S, N, D) and weights are None
        self.assertIsNone(weights)
        self.assertTrue(torch.equal(output, query.transpose(1, 2).contiguous()))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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
            call["atten_mask"], torch.logical_not(full_mask[:, :, :, :4])
        ))
        # is_causal default + self-attention length match adds next_tockens=0
        self.assertEqual(call["next_tockens"], 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_float_mask_is_cast_without_inversion(self):
        """A float (additive-style) mask is cast to bool WITHOUT inversion."""
        query = torch.randn(1, 2, 4, 8)
        float_mask = torch.zeros(1, 1, 4, 4)
        float_mask[..., -1] = float("-inf")
        stub, _, _ = self._run(query, query, query, float_mask)
        call = stub.calls[0]
        self.assertEqual(call["sparse_mode"], 0)
        self.assertTrue(torch.equal(
            call["atten_mask"], float_mask.bool()
        ))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_next_tockens_requires_causal_self_attention(self):
        """next_tockens=0 is withheld for cross-attention or is_causal=False."""
        query = torch.randn(1, 2, 2, 8)
        key = torch.randn(1, 2, 4, 8)
        mask = torch.ones(1, 1, 2, 4, dtype=torch.bool)
        stub, _, _ = self._run(query, key, key, mask)
        self.assertNotIn("next_tockens", stub.calls[0])
        non_causal = nn.Module()
        non_causal.is_causal = False
        self_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        stub, _, _ = self._run(key, key, key, self_mask, module=non_causal)
        self.assertNotIn("next_tockens", stub.calls[0])


if __name__ == "__main__":
    unittest.main()
