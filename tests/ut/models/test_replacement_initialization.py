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
"""Replacement initialization contracts after meta parameter materialization."""

import unittest
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
import tempfile
from typing import Any
from unittest.mock import patch

from safetensors.torch import save_file
import torch
from torch import nn

from examples.torch.module_family.config import (
    EngramConfig, GateConfig, MHCPreConfig, MegaMhcConfig, PipelinedMhcConfig, RMSNormConfig,
    SharedExpertConfig, SpatialMergeProjectorConfig, SwiGLUMLPConfig,
)
from hyper_parallel.components.modules import (
    Engram, MHCPre, MegaMhc, PipelinedMhc, RMSNorm, SharedExpert, SpatialMergeProjector, SwiGLUMLP,
)
from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.components.modules.linear import Linear
from hyper_parallel.components.modules.moe.grouped_experts import GroupedExperts, GroupedExpertsConfig
from hyper_parallel.components.modules.moe.router import Gate
from hyper_parallel.core.activation_memory.wrapper import CheckpointWrapper
from hyper_parallel.components.modules.common.spatial_merge_projector import SpatialMergeProjectorModule
from hyper_parallel.models._transformers.model_builder import _initialize_model_weights, _materialize_and_load_model
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import _initialize_v41_owned_module
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec, apply_module_replacements, compile_module_replacements, module_replacement,
)
from tests.common.mark_utils import arg_mark


@module_replacement
class _ResetOnlyLinear(nn.Module):
    """Reuse source parameters but omit the checkpoint declaration intentionally."""

    def __init__(self, *, module: nn.Linear, module_fqn: str, context: Mapping[str, Any]) -> None:
        """Retain source parameter identity without calling any reset."""
        super().__init__()
        del module_fqn, context
        self.weight = module.weight
        self.register_parameter("bias", module.bias)
        self.reset_calls = 0
        self.train(module.training)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """Preserve the source Linear call contract."""
        output = input @ self.weight.t()
        return output if self.bias is None else output + self.bias

    def reset_parameters(self) -> None:
        """Record a replacement-owned reset and fill the final storage."""
        self.reset_calls += 1
        nn.init.constant_(self.weight, 2.0)


@module_replacement
class _EmptyTransformsLinear(_ResetOnlyLinear):
    """An inherited custom reset with an explicit identity checkpoint declaration."""

    @staticmethod
    def make_transforms() -> list:
        """Keep canonical checkpoint weights unchanged."""
        return []


@module_replacement
class _InheritedTransformsLinear(_EmptyTransformsLinear):
    """Inherit the checkpoint/reset contract without overriding the model's policy."""


@module_replacement
class _ResetWithInheritedTransformsLinear(_InheritedTransformsLinear):
    """Override reset while retaining the parent's checkpoint declaration."""

    def reset_parameters(self) -> None:
        """Override the reset policy while inheriting the checkpoint declaration."""
        self.reset_calls += 1
        nn.init.constant_(self.weight, 3.0)


def _child_module_cases() -> tuple:
    """Return wrappers whose adopted Torch children retain model-native initialization."""
    assets = {
        "layer_ids": [0], "max_ngram_size": 3, "num_heads": 2, "token_map": list(range(16)),
        "pad_token_id": 0, "primes": [[[5, 7], [11, 13]]], "multipliers": [[1, 3, 5]],
    }
    return (
        (SpatialMergeProjector, SpatialMergeProjectorModule, SpatialMergeProjectorConfig(4, 8, 4)),
        (Engram, EngramModule, EngramConfig(assets, 0, 8, 2, 4, padded_num_embeddings=40)),
    )


class _NativeChildrenModel(nn.Module):
    """Model with an independent initialization policy for Linear/Embedding children."""

    def __init__(self, component: nn.Module) -> None:
        super().__init__()
        self.component = component
        self.native_snapshot = {}
        self.native_calls = 0

    def initialize_weights(self) -> None:
        """Record exactly the parameters initialized through the native model policy."""
        self.native_calls += 1
        for name, module in self.component.named_modules():
            if not isinstance(module, (nn.Linear, nn.Embedding)):
                continue
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            for parameter_name, parameter in module.named_parameters(recurse=False):
                self.native_snapshot[f"{name}.{parameter_name}"] = parameter.detach().clone()


class _NativeInitModel(nn.Module):
    """Represent a model with a native initialization entry point."""

    def __init__(self) -> None:
        """Create a Linear source and count native initialization calls."""
        super().__init__()
        self.projection = nn.Linear(3, 2, bias=False)
        self.native_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the projection."""
        return self.projection(inputs)

    def initialize_weights(self) -> None:
        """Apply a model policy distinguishable from both component reset values."""
        self.native_calls += 1
        for parameter in self.parameters():
            nn.init.constant_(parameter, 1.0)


class _ParentResetBlock(nn.Module):
    """Composite module whose explicit initialization owns its child's final values."""

    def __init__(self, projection: nn.Linear, value: float) -> None:
        super().__init__()
        self.projection = projection
        self.value = value

    def reset_parameters(self) -> None:
        """Apply a parent-owned policy after child defaults."""
        nn.init.constant_(self.projection.weight, self.value)


class TestReplacementInitialization(unittest.TestCase):
    """Validate the declared reset policy without changing ordinary Linear initialization."""

    def test_fallback_preserves_parent_policy_and_shared_module_identity(self) -> None:
        """Visit children before parents and reset shared children only once."""
        for shared in (False, True):
            with self.subTest(shared=shared), torch.device("meta"):
                first = nn.Linear(8, 4, bias=False)
                second = first if shared else nn.Linear(8, 4, bias=False)
                model = nn.Sequential(_ParentResetBlock(first, .125), _ParentResetBlock(second, .25))
            model.to_empty(device="cpu")
            with ExitStack() as stack:
                resets = [stack.enter_context(patch.object(child, "reset_parameters", wraps=child.reset_parameters))
                          for child in model.modules() if isinstance(child, nn.Linear)]
                _initialize_model_weights(model)
                for reset in resets:
                    reset.assert_called_once()
            for index, expected in enumerate((.25 if shared else .125, .25)):
                torch.testing.assert_close(model[index].projection.weight, torch.full((4, 8), expected), rtol=0, atol=0)
            self.assertEqual(model[0].projection is model[1].projection, shared)

    def test_projection_family_defaults_preserve_native_values(self) -> None:
        """MLP and shared-expert resets provide defaults without overwriting the model policy."""
        cases = ((SwiGLUMLP, SwiGLUMLPConfig(8, 12), {}),
                 (SwiGLUMLP, SwiGLUMLPConfig(8, 12), {"state_layout": "w1_w2"}),
                 (SharedExpert, SharedExpertConfig(8, 12), {}))
        for factory, config, options in cases:
            for adopt in (False, True):
                with self.subTest(family=factory.__name__, options=options, adopt=adopt):
                    with torch.device("meta"):
                        block = factory(config=config, **options)
                        if adopt:
                            block = factory(module=block, **options)
                        model = _NativeChildrenModel(block)
                    model.to_empty(device="cpu")
                    _initialize_model_weights(model)
                    for name, value in model.native_snapshot.items():
                        torch.testing.assert_close(model.component.get_parameter(name), value, rtol=0, atol=0)

    def test_activation_wrapper_does_not_repeat_component_reset(self) -> None:
        """A forwarding wrapper does not reset the same child twice during initialization."""
        for factory in (_EmptyTransformsLinear, _InheritedTransformsLinear):
            for native in (True, False):
                with self.subTest(factory=factory.__name__, native=native), torch.device("meta"):
                    model = _NativeInitModel()
                    plan = compile_module_replacements(model, [ModuleReplacementSpec(("projection",), factory)])
                    apply_module_replacements(model, plan)
                    component = model.projection
                    model.projection = CheckpointWrapper(component)
                    if not native:
                        model.initialize_weights = None
                model.to_empty(device="cpu")
                _initialize_model_weights(model)
                self.assertEqual(component.reset_calls, 1)
                expected = 1.0 if native else 2.0
                torch.testing.assert_close(component.weight, torch.full((2, 3), expected), rtol=0, atol=0)

    def test_component_defaults_yield_to_native_policy(self) -> None:
        """Complete mHC, gate and expert defaults survive only when the model does not replace them."""
        cases = (
            (MHCPre, MHCPreConfig(4, 16, use_gamma=True), {},
             {"branch_alpha": .01, "branch_beta": 0., "norm_gamma": 1.}),
            (PipelinedMhc, PipelinedMhcConfig(4, 16), {}, {"base": 0., "scale": .01}),
            (MegaMhc, MegaMhcConfig(16), {}, {"alpha": 1., "bias": 0., "norm_weight": 1.}),
            (Gate, GateConfig(16, 4, 2, vision_enabled=True), {"implementation": "mega"},
             {"bias": 0., "bias_vl": 0.}),
            (GroupedExperts, GroupedExpertsConfig(4, 32, 64, init_method_std=.03, num_layers=8), {}, {}),
        )
        for factory, config, options, constants in cases:
            for native in (False, True):
                with self.subTest(factory=factory.__name__, native=native), torch.random.fork_rng(devices=[]):
                    torch.manual_seed(41)
                    with torch.device("meta"):
                        model = _NativeInitModel()
                        model.projection = factory(config=config, **options)
                        if not native:
                            model.initialize_weights = None
                    model.to_empty(device="cpu")
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.fill_(float("nan"))
                    _initialize_model_weights(model)
                    self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
                    if native:
                        for parameter in model.parameters():
                            torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)
                        continue
                    for name, value in constants.items():
                        parameter = model.projection.get_parameter(name)
                        torch.testing.assert_close(parameter, torch.full_like(parameter, value), rtol=0, atol=0)
                    if factory is GroupedExperts:
                        self.assertAlmostEqual(model.projection.gate_up_proj.std().item(), .03, delta=.001)
                        self.assertAlmostEqual(model.projection.down_proj.std().item(), .0075, delta=.0003)

    def test_v41_pipelined_mhc_preserves_model_initialization(self) -> None:
        """Preserve V4.1's scale=1 and model-specific predictor distribution after defaults."""
        with torch.device("meta"):
            model = _NativeInitModel()
            model.projection = PipelinedMhc(config=PipelinedMhcConfig(4, 16))
        model.to_empty(device="cpu")
        native_values = {}

        def initialize_weights():
            _initialize_v41_owned_module(model.projection, .04)
            native_values.update({name: parameter.detach().clone()
                                  for name, parameter in model.projection.named_parameters()})

        with patch.object(model, "initialize_weights", side_effect=initialize_weights):
            _initialize_model_weights(model)
        torch.testing.assert_close(model.projection.scale, torch.ones(3), rtol=0, atol=0)
        for name, value in native_values.items():
            torch.testing.assert_close(model.projection.get_parameter(name), value, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_custom_reset_requires_make_transforms(self) -> None:
        """Feature: Replacement initialization.
        Description: Reject a reset-only replacement before it changes the model tree.
        Expectation: Custom reset requires make transforms.
        """
        model = nn.Sequential(nn.Linear(3, 2, bias=False))
        source = model[0]
        plan = compile_module_replacements(model, [ModuleReplacementSpec(
            match=("0",), factory=_ResetOnlyLinear,
        )])
        with self.assertRaisesRegex(TypeError, "reset_parameters requires a callable make_transforms"):
            apply_module_replacements(model, plan)
        self.assertIs(model[0], source)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_empty_transforms_enable_materialized_reset(self) -> None:
        """Feature: Replacement initialization.
        Description: Apply component reset defaults before native initialization.
        Expectation: Empty transforms enable materialized reset.
        """
        with torch.device("meta"):
            model = _NativeInitModel()
        source = model.projection
        plan = compile_module_replacements(model, [ModuleReplacementSpec(
            match=("projection",), factory=_EmptyTransformsLinear,
        )])
        apply_module_replacements(model, plan)
        replacement = model.projection
        self.assertIs(replacement.weight, source.weight)
        self.assertTrue(getattr(replacement, "_hp_reset_after_materialization", False))
        self.assertEqual(replacement.reset_calls, 0)
        model.to_empty(device="cpu")
        _initialize_model_weights(model)
        self.assertEqual(model.native_calls, 1)
        self.assertEqual(replacement.reset_calls, 1)
        torch.testing.assert_close(replacement.weight, torch.full((2, 3), 1.0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_inherited_transforms_satisfy_custom_reset_contract(self) -> None:
        """Feature: Replacement initialization.
        Description: Materialize inherited and overridden defaults with and without a native initializer.
        Expectation: Each reset runs once; the native policy owns final values when available.
        """
        for factory, default in ((_InheritedTransformsLinear, 2.0), (_ResetWithInheritedTransformsLinear, 3.0)):
            for native in (False, True):
                with self.subTest(factory=factory.__name__, native=native):
                    self.assertNotIn("make_transforms", vars(factory))
                    with torch.device("meta"):
                        model = _NativeInitModel()
                        if not native:
                            model.initialize_weights = None
                    plan = compile_module_replacements(model, [ModuleReplacementSpec(("projection",), factory)])
                    apply_module_replacements(model, plan)
                    self.assertTrue(getattr(model.projection, "_hp_reset_after_materialization", False))
                    model.to_empty(device="cpu")
                    _initialize_model_weights(model)
                    self.assertEqual(model.native_calls, int(native))
                    self.assertEqual(model.projection.reset_calls, 1)
                    expected = 1.0 if native else default
                    torch.testing.assert_close(model.projection.weight, torch.full((2, 3), expected), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_adopted_children_keep_native_initialization(self) -> None:
        """Feature: Replacement initialization.
        Description: The materialization pipeline never replaces native child initialization with Torch defaults.
        Expectation: Adopted children keep native initialization.
        """
        for factory, implementation, config in _child_module_cases():
            with self.subTest(factory=factory.__name__):
                with torch.device("meta"):
                    source = implementation(config=config)
                    model = _NativeChildrenModel(source)
                source_parameters = dict(source.named_parameters())
                plan = compile_module_replacements(model, [ModuleReplacementSpec(("component",), factory)])
                apply_module_replacements(model, plan, weights_mapping=[])
                for name, parameter in source_parameters.items():
                    self.assertIs(model.component.get_parameter(name), parameter)
                model.to_empty(device="cpu")
                with ExitStack() as stack:
                    resets = [stack.enter_context(patch.object(child, "reset_parameters", wraps=child.reset_parameters))
                              for child in model.component.modules() if isinstance(child, (nn.Linear, nn.Embedding))]
                    _initialize_model_weights(model)
                    for reset in resets:
                        reset.assert_not_called()
                self.assertEqual(model.native_calls, 1)
                for name, value in model.native_snapshot.items():
                    torch.testing.assert_close(model.component.get_parameter(name), value, atol=0, rtol=0)
                for parameter in model.component.parameters(recurse=False):
                    torch.testing.assert_close(parameter, torch.ones_like(parameter), atol=0, rtol=0)
                for parameter in model.parameters():
                    self.assertTrue(torch.isfinite(parameter).all())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fallback_initializes_children_once(self) -> None:
        """Feature: Replacement initialization.
        Description: Direct and replacement builds initialize owned scales and each Torch child once.
        Expectation: Fallback initializes children once.
        """
        for factory, implementation, config in _child_module_cases():
            for replace_source in (False, True):
                with self.subTest(factory=factory.__name__, replacement=replace_source):
                    with torch.device("meta"):
                        constructor = implementation if replace_source else factory
                        model = nn.Sequential(constructor(config=config))
                    if replace_source:
                        plan = compile_module_replacements(model, [ModuleReplacementSpec(("0",), factory)])
                        apply_module_replacements(model, plan, weights_mapping=[])
                    model.to_empty(device="cpu")
                    with ExitStack() as stack:
                        resets = [stack.enter_context(patch.object(
                            child, "reset_parameters", wraps=child.reset_parameters,
                        )) for child in model.modules() if isinstance(child, (nn.Linear, nn.Embedding))]
                        _initialize_model_weights(model)
                        for reset in resets:
                            reset.assert_called_once()
                    for parameter in model[0].parameters(recurse=False):
                        torch.testing.assert_close(parameter, torch.ones_like(parameter), atol=0, rtol=0)
                    for parameter in model.parameters():
                        self.assertTrue(torch.isfinite(parameter).all())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fallback_reset_does_not_repeat_marked_module(self) -> None:
        """Feature: Replacement initialization.
        Description: Run one reset when the root has no native initialization method.
        Expectation: Fallback reset does not repeat marked module.
        """
        with torch.device("meta"):
            model = nn.Sequential(nn.Linear(3, 2, bias=False))
        plan = compile_module_replacements(model, [ModuleReplacementSpec(
            match=("0",), factory=_EmptyTransformsLinear,
        )])
        apply_module_replacements(model, plan)
        model.to_empty(device="cpu")
        _initialize_model_weights(model)
        self.assertEqual(model[0].reset_calls, 1)
        torch.testing.assert_close(model[0].weight, torch.full((2, 3), 2.0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_checkpoint_loading_does_not_reset_marked_module(self) -> None:
        """Feature: Replacement initialization.
        Description: Load real checkpoint values without executing the declared random initialization policy.
        Expectation: Checkpoint loading does not reset marked module.
        """
        with torch.device("meta"):
            model = _NativeInitModel()
            model.defaults = RMSNorm(config=RMSNormConfig(3))
        mapping = []
        plan = compile_module_replacements(model, [ModuleReplacementSpec(
            match=("projection",), factory=_InheritedTransformsLinear,
        )])
        apply_module_replacements(model, plan, weights_mapping=mapping)
        expected = torch.full((2, 3), 7.0)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                model.defaults, "reset_parameters", wraps=model.defaults.reset_parameters) as default_reset:
            save_file({"projection.weight": expected, "defaults.weight": torch.full((3,), 9.)},
                      Path(directory) / "model.safetensors")
            _materialize_and_load_model(
                model, is_meta_device=True, device=torch.device("cpu"), load_base_model=True,
                pretrained_path=directory, weights_mapping=mapping,
            )
        self.assertEqual(model.native_calls, 0)
        self.assertEqual(model.projection.reset_calls, 0)
        default_reset.assert_not_called()
        torch.testing.assert_close(model.defaults.weight, torch.full((3,), 9.))
        torch.testing.assert_close(model.projection.weight, expected)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_inherited_torch_reset_keeps_native_initialization(self) -> None:
        """Feature: Replacement initialization.
        Description: Preserve HF-style native initialization for an ordinary Linear replacement.
        Expectation: Inherited torch reset keeps native initialization.
        """
        with torch.device("meta"):
            model = _NativeInitModel()
        plan = compile_module_replacements(model, [ModuleReplacementSpec(
            match=("projection",), factory=Linear,
        )])
        apply_module_replacements(model, plan)
        self.assertFalse(getattr(model.projection, "_hp_reset_after_materialization", False))
        model.to_empty(device="cpu")
        _initialize_model_weights(model)
        self.assertEqual(model.native_calls, 1)
        torch.testing.assert_close(model.projection.weight, torch.ones(2, 3))
