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
"""Regression tests for observed factories and instance-bound forward state."""

import ast
from collections.abc import Callable
import inspect
from importlib import import_module
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from hyper_parallel.codegen.emit.modeling import _verify_replacement_records
from hyper_parallel.codegen.emit.patches import emission_patches_from_meta
from hyper_parallel.codegen.emit.snapshot_patches import emission_patches_from_snapshot_payload
from hyper_parallel.codegen.inline.component_source import observe_wrapper_source
from hyper_parallel.codegen.inline.framework_spec import observed_constructor_spec
from hyper_parallel.codegen.inline.ir import SourceObservation
from hyper_parallel.codegen.inline.observations import collect_source_observations
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set
from hyper_parallel.codegen.inline.pipeline import _require_inline_coverage
from hyper_parallel.codegen.inline.replacement_pass import build_replacement_patches
from hyper_parallel.codegen.inline.strategy_pass import build_strategy_patches
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.models import registry
from hyper_parallel.models.replacement import module_replacement


class _Attention(nn.Module):
    """Keep one parameter whose identity must survive native replacement."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class _AdaptedAttention(_Attention):
    """Adopt an existing module through a native classmethod factory."""

    @classmethod
    def from_source(cls, source: _Attention, *, module_fqn: str) -> "_AdaptedAttention":
        """Preserve the source parameter without cloning its storage."""
        result = cls()
        result.weight = source.weight
        result.module_fqn = module_fqn
        return result


@module_replacement
def _replace_attention(*, module, module_fqn, context):
    """Use the native factory protocol without consulting build context."""
    del context
    return _AdaptedAttention.from_source(module, module_fqn=module_fqn)


@module_replacement
def _replace_with_context(*, module, module_fqn, context):
    """Require context that cannot be fabricated during source emission."""
    context["prepare"](module)
    return _AdaptedAttention.from_source(module, module_fqn=module_fqn)


class _Bridge:
    """Represent model-bound state with an explicit native construction source."""

    def __init__(self, owner, enabled=True):
        self.owner = owner
        self.enabled = enabled

    def call(self, delegate: Callable[..., object], *args: object, **kwargs: object) -> object:
        """Exercise a saved forward as a callable value rather than a direct call."""
        return delegate(*args, **kwargs)


class _Model(nn.Module):
    """A source host with no pre-existing wrapper state."""

    def __init__(self):
        super().__init__()
        self.marker = object()

    def forward(self, *args: object, **kwargs: object) -> tuple:
        """Expose argument forwarding without a training or loss computation."""
        return self.marker, args, kwargs


def _bind_bridge(model):
    """Publish the state used by an installed native closure."""
    original_forward = model.forward
    bridge = _Bridge(model)

    def forward(*args: object, **kwargs: object) -> object:
        """Delegate through published native state."""
        delegate = original_forward
        return bridge.call(delegate, *args, **kwargs)

    model.bridge = bridge
    model.forward = forward


def _bind_alias(model):
    """Save a method exclusively through a local callable alias."""
    original_forward = model.forward

    def forward(*args: object, **kwargs: object) -> tuple:
        """Preserve a callable alias to the original method."""
        delegate = original_forward
        return delegate(*args, **kwargs)

    model.forward = forward


def _bind_configured_bridge(model, enabled=True):
    """Keep a native constructor option that the forward also captures."""
    original_forward = model.forward
    bridge = _Bridge(model, enabled=enabled)

    def forward(*args: object, **kwargs: object) -> object:
        """Retain the configured native forwarding branch."""
        if enabled:
            return bridge.call(original_forward, *args, **kwargs)
        return original_forward(*args, **kwargs)

    model.bridge = bridge
    model.forward = forward


class TestObservedReplacementFactory(unittest.TestCase):
    """Native observations must not re-execute model replacement providers."""

    def test_classmethod_factory_preserves_parameters_and_aliases(self):
        """Coverage and rendering share the same recorded factory resolution."""
        target = "hyper_parallel.models.fixture.adapter.replacements"
        module = ModuleType(target)
        module.replace_attention = _replace_attention
        provider = Mock(side_effect=lambda model, context: ())
        adapter = SimpleNamespace(model_type="fixture", replacements=provider)
        rule = SourceObservation(
            match=("attn", "alias"), shared_fqns=("attn", "alias"),
            module_type=f"{__name__}._Attention", replace_target=target + ".replace_attention",
            construction={"implementation": f"{__name__}:_AdaptedAttention", "source_fqns": ["attn", "alias"]},
        )
        with (patch.dict(sys.modules, {target: module}),
              patch.object(registry, "get_model_adapter", return_value=adapter)):
            _require_inline_coverage((rule,), "fixture")
            patches = build_replacement_patches((rule,), "fixture")
        provider.assert_not_called()
        source = (
            f"from {__name__} import _Attention\n"
            "class Model:\n"
            "    def __init__(self):\n"
            "        self.attn = _Attention()\n"
            "        self.alias = self.attn\n"
            "        self.original_weight = self.attn.weight\n"
        )
        namespace = {}
        generated = apply_patch_set(source, patches, root_class_name="Model")
        exec(compile(generated, "<factory>", "exec"), namespace)  # pylint: disable=exec-used
        model = namespace["Model"]()
        self.assertIsInstance(model.attn, _AdaptedAttention)
        self.assertIs(model.attn, model.alias)
        self.assertIs(model.attn.weight, model.original_weight)
        self.assertEqual(model.attn.module_fqn, "attn")
        payload = {"source": {}, "replacements": {
            "declarations": [{"rule_id": "native", "replacement_target": f"{__name__}._replace_attention"}],
            "applications": [{"rule_id": "native", "primary_fqn": "attn",
                              "source_type": f"{__name__}:_Attention",
                              "replacement_type": f"{__name__}:_AdaptedAttention"}],
        }}
        records = emission_patches_from_meta(SimpleNamespace(
            emission_patches=emission_patches_from_snapshot_payload(payload),
        ))
        self.assertEqual(_verify_replacement_records(ast.parse(generated), {}, records.module_replacements, None), [])

    def test_unrecorded_context_is_not_replaced_with_an_empty_mapping(self):
        """An observed implementation does not authorize inventing factory context."""
        rule = SourceObservation(
            match=("attn",), module_type=f"{__name__}._Attention",
            replace_target=f"{__name__}._replace_with_context",
            construction={"implementation": f"{__name__}:_AdaptedAttention"},
        )
        self.assertIsNone(observed_constructor_spec(rule))


class TestObservedWrapperBindings(unittest.TestCase):
    """Generated forwards retain native callable and instance-state bindings."""

    @staticmethod
    def _render(bind):
        model = _Model()
        original = model.forward
        bind(model)
        session = CodegenBuildSession()
        session.record_source({"model_type": "fixture"})
        session.mark_not_required("no_parameter_layout")
        session.observe_final_forwards(model, {"": original})
        injections = session.finish()._emission_view()["plan"]["injections"]
        rules = collect_source_observations(SimpleNamespace(injections=injections))
        patches = build_strategy_patches(rules)
        source = apply_patch_set("from torch import nn\n" + inspect.getsource(_Model), patches)
        namespace = {}
        exec(compile(source, "<wrapper>", "exec"), namespace)  # pylint: disable=exec-used
        return namespace["_Model"], source

    def test_saved_forward_alias_is_preserved(self):
        """A saved method used as a value still needs the original source method."""
        generated, _ = self._render(_bind_alias)
        model = generated()
        self.assertEqual(model.forward(3, token=7), (model.marker, (3,), {"token": 7}))

    def test_published_bridge_is_rebuilt_once_for_each_instance(self):
        """Constructor state binds the generated model, never the observed model."""
        generated, source = self._render(_bind_bridge)
        left, right = generated(), generated()
        self.assertIs(left.bridge.owner, left)
        self.assertIs(right.bridge.owner, right)
        self.assertIsNot(left.bridge, right.bridge)
        bridge = left.bridge
        self.assertEqual(left.forward(3, token=7), (left.marker, (3,), {"token": 7}))
        self.assertIs(left.bridge, bridge)
        self.assertIn("self._forward_impl", source)

    def test_unpublished_bridge_remains_an_error(self):
        """Arbitrary closure objects cannot silently become global source bindings."""
        model = _Model()
        original = model.forward
        _bind_bridge(model)
        del model.bridge
        with self.assertRaisesRegex(RuntimeError, "unresolved captured state.*bridge"):
            observe_wrapper_source(model, model.forward, vars(model), original_forward=original)

    def test_native_constructor_literals_are_preserved(self):
        """Both native branches retain their per-instance constructor option."""
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                generated, _ = self._render(lambda model: _bind_configured_bridge(model, enabled))
                model = generated()
                self.assertEqual(model.bridge.enabled, enabled)
                self.assertEqual(model.forward(3, token=7), (model.marker, (3,), {"token": 7}))

    def test_conflicting_instance_initializers_are_rejected(self):
        """One source class cannot silently receive the first instance's state."""
        rules = tuple(SourceObservation(
            match=(name,), inner_wrapper_target="fixture.bind",
            observed_wrapper={
                "host_class": "Host", "body": "return self._forward_impl(value)", "imports": [],
                "initializers": (f"self.enabled = {enabled!r}",),
            },
        ) for name, enabled in (("left", False), ("right", True)))
        with self.assertRaisesRegex(RuntimeError, "conflicting CP bindings"):
            build_strategy_patches(rules)

    def test_qwen_native_state_publication_keeps_parameter_ownership(self):
        """Publishing the bridge adds no registered modules or parameter aliases."""
        adapter = import_module("hyper_parallel.models.qwen3_5_moe.adapter.vlm")
        model = _Model()
        model.config = SimpleNamespace(image_token_id=3)
        model.model = nn.Module()
        model.model.visual = nn.Linear(2, 2)
        model.model.visual.spatial_merge_size = 2
        before = {name: id(value) for name, value in model.named_parameters()}
        adapter.install_qwen_vlm_cp_forward(model)
        after = {name: id(value) for name, value in model.named_parameters()}
        self.assertEqual(after, before)
        self.assertIs(model._hp_qwen_vlm_bridge.model, model)  # pylint: disable=protected-access
        self.assertEqual(model.forward(3, token=7), (model.marker, (3,), {"token": 7}))
