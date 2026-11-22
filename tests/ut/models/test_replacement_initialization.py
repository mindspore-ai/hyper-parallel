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

from examples.torch.module_family.config import EngramConfig, SpatialMergeProjectorConfig
from hyper_parallel.components.modules import Engram, SpatialMergeProjector
from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.components.modules.linear import Linear
from hyper_parallel.components.modules.common.spatial_merge_projector import SpatialMergeProjectorModule
from hyper_parallel.models._transformers.model_builder import _initialize_model_weights, _materialize_and_load_model
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
    """Inherit both the checkpoint declaration and replacement reset."""


@module_replacement
class _ResetWithInheritedTransformsLinear(_EmptyTransformsLinear):
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
        """Initialize model weights before replacement-specific reset."""
        self.native_calls += 1
        for parameter in self.parameters():
            nn.init.constant_(parameter, 1.0)


class TestReplacementInitialization(unittest.TestCase):
    """Validate the declared reset policy without changing ordinary Linear initialization."""

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
        Description: Use an inherited custom reset after native initialization even with no checkpoint transforms.
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
        torch.testing.assert_close(replacement.weight, torch.full((2, 3), 2.0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_inherited_transforms_satisfy_custom_reset_contract(self) -> None:
        """Feature: Replacement initialization.
        Description: Accept a parent conversion declaration with either an inherited or overridden reset.
        Expectation: Inherited transforms satisfy custom reset contract.
        """
        for factory, expected in ((_InheritedTransformsLinear, 2.0), (_ResetWithInheritedTransformsLinear, 3.0)):
            with self.subTest(factory=factory.__name__):
                self.assertNotIn("make_transforms", vars(factory))
                with torch.device("meta"):
                    model = _NativeInitModel()
                plan = compile_module_replacements(model, [ModuleReplacementSpec(("projection",), factory)])
                apply_module_replacements(model, plan)
                self.assertTrue(getattr(model.projection, "_hp_reset_after_materialization", False))
                model.to_empty(device="cpu")
                _initialize_model_weights(model)
                self.assertEqual(model.projection.reset_calls, 1)
                torch.testing.assert_close(model.projection.weight, torch.full((2, 3), expected))

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
                apply_module_replacements(model, plan)
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
                        apply_module_replacements(model, plan)
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
        mapping = []
        plan = compile_module_replacements(model, [ModuleReplacementSpec(
            match=("projection",), factory=_EmptyTransformsLinear,
        )])
        apply_module_replacements(model, plan, weights_mapping=mapping)
        expected = torch.full((2, 3), 7.0)
        with tempfile.TemporaryDirectory() as directory:
            save_file({"projection.weight": expected}, Path(directory) / "model.safetensors")
            _materialize_and_load_model(
                model, is_meta_device=True, device=torch.device("cpu"), load_base_model=True,
                pretrained_path=directory, weights_mapping=mapping,
            )
        self.assertEqual(model.native_calls, 0)
        self.assertEqual(model.projection.reset_calls, 0)
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
