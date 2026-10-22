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
"""Family configuration through the unchanged CPU module replacement engine."""

import copy
import unittest
from typing import Any, Callable, Mapping, Optional
from unittest.mock import patch

import torch
import yaml
from torch import nn

from hyper_parallel.components.modules import family as registry
from hyper_parallel.components.modules.family import ModuleFamily, SelectorSpec, VariantRule, register_module_family
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements, module_replacement
from hyper_parallel.trainer.config.parallelism import entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import ConfigResolutionError, replace_override_path, resolve_config
from hyper_parallel.trainer.config.target import Target
from tests.common.mark_utils import arg_mark


@module_replacement
class ReferenceLinear(nn.Module):
    """Test-only baseline retaining source parameters, state and call signature."""

    def __init__(
        self,
        *,
        module: nn.Linear,
        module_fqn: str = "",
        context: Optional[Mapping[str, Any]] = None,
        gain: float = 1.0,
    ) -> None:
        """Reuse loaded parameters and preserve the source training mode."""
        super().__init__()
        self.register_parameter("weight", module.weight)
        self.register_parameter("bias", module.bias)
        self.gain = gain
        self.module_fqn = module_fqn
        self.factory_context = context
        self.train(module.training)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the same linear transform with the selected CPU implementation."""
        return nn.functional.linear(input, self.weight, self.bias) * self.gain  # pylint: disable=not-callable


@module_replacement
class MatmulLinear(ReferenceLinear):
    """Test implementation using ordinary CPU matrix multiplication."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the same linear transform with the selected CPU implementation."""
        output = input @ self.weight.t()
        return (output if self.bias is None else output + self.bias) * self.gain


@module_replacement
class AddmmLinear(ReferenceLinear):
    """Test implementation using addmm with the same live parameters."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the same linear transform with the selected CPU implementation."""
        output = input @ self.weight.t() if self.bias is None else torch.addmm(self.bias, input, self.weight.t())
        return output * self.gain


@module_replacement
def ordinary_factory(
    *,
    module: nn.Linear,
    module_fqn: str = "",
    context: Optional[Mapping[str, Any]] = None,
    algorithm: str = "ordinary",
    custom_option: int = 0,
    callback: Optional[Callable] = None,
) -> ReferenceLinear:
    """Existing non-Family factories retain their explicitly declared arguments."""
    result = ReferenceLinear(module=module, module_fqn=module_fqn, context=context)
    result.options = {"custom_option": custom_option, "callback": callback}
    result.algorithm = algorithm
    return result


def _raw_config(**selectors):
    return {
        "model": {"_target_": "torch.nn.Identity"},
        "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
        "plan_overrides": [
            {
                "match": "0",
                "module_type": "torch.nn.Linear",
                "exact_type": True,
                "replace_module": {"_target_": f"{__name__}.ReferenceLinear", **selectors},
            }
        ],
    }


def _install(model, config):
    rules = entries_to_module_replacements(config.plan_overrides)
    plan = compile_module_replacements(model, rules)
    return apply_module_replacements(model, plan, context={"test_context": 7})[0]


class TestFamilyReplacement(unittest.TestCase):
    """Use real configuration, real replacement, and CPU autograd."""

    def setUp(self) -> None:
        """Register only test-owned variants and restore state after each case."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        registry_patch = patch.dict(registry._FAMILIES, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        register_module_family(
            ModuleFamily(
                "linear_test",
                ReferenceLinear,
                {"matmul": f"{__name__}.MatmulLinear", "addmm": f"{__name__}.AddmmLinear"},
                {"algorithm": SelectorSpec(str, "reference"), "fused": SelectorSpec(bool, False)},
                (
                    VariantRule({"algorithm": "matmul", "fused": False}, "matmul"),
                    VariantRule({"algorithm": "matmul", "fused": True}, "addmm"),
                ),
            )
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_yaml_real_replacement_and_autograd(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: All allowed choices preserve Parameter identity, values, mode, output and gradients.
        Expectation: Selection preserves live parameters, frozen state, values, outputs and gradients.
        """
        for selectors, implementation in (
            ({}, ReferenceLinear),
            ({"algorithm": "matmul"}, MatmulLinear),
            ({"algorithm": "matmul", "fused": True}, AddmmLinear),
        ):
            for trainable in (False, True):
                with self.subTest(selectors=selectors, trainable=trainable):
                    config = resolve_config(yaml.safe_load(yaml.safe_dump(_raw_config(**selectors))))
                    source = nn.Linear(4, 3).double().eval()
                    source.requires_grad_(trainable)
                    reference = copy.deepcopy(source)
                    model = nn.Sequential(source)
                    weight, bias = source.weight, source.bias
                    model = _install(model, config)
                    self.assertIs(type(model[0]), implementation)
                    self.assertIs(model[0].weight, weight)
                    self.assertIs(model[0].bias, bias)
                    self.assertFalse(model[0].training)
                    self.assertEqual(model[0].weight.requires_grad, trainable)
                    self.assertEqual(model[0].factory_context["test_context"], 7)
                    inputs = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
                    reference_inputs = inputs.detach().clone().requires_grad_(True)
                    output, expected = model(inputs), reference(reference_inputs)
                    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
                    output.square().sum().backward()
                    expected.square().sum().backward()
                    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
                    if trainable:
                        torch.testing.assert_close(weight.grad, reference.weight.grad)
                        torch.optim.SGD(model.parameters(), lr=0.01, foreach=False, fused=False).step()
                        torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False, fused=False).step()
                        torch.testing.assert_close(weight, reference.weight)
                    else:
                        self.assertIsNone(weight.grad)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_aliases_use_one_selected_instance(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Alias handling remains owned by the unchanged replacement engine.
        Expectation: Aliases point to one replacement with the original Parameter object.
        """
        raw = _raw_config(algorithm="matmul")
        raw["plan_overrides"][0]["match"] = "left"
        model = nn.Module()
        model.left = nn.Linear(4, 3)
        model.right = model.left
        weight = model.left.weight
        _install(model, resolve_config(raw))
        self.assertIs(model.left, model.right)
        self.assertIs(model.left.weight, weight)
        self.assertIsInstance(model.left, MatmulLinear)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_roundtrip_and_copy_keep_explicit_selectors(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Selection changes copy configuration without mutating the original or changing its baseline path.
        Expectation: Copies reselect independently and YAML round trips retain explicit fields.
        """
        original = resolve_config(_raw_config()).plan_overrides[0].replace_module
        alternate = original.replace(algorithm="matmul")
        fused = alternate.replace(fused=True)
        restored = fused.replace(algorithm="reference", fused=False)
        self.assertIs(original.callable, ReferenceLinear)
        self.assertEqual(original.selectors, {})
        for target in (original, alternate, fused, restored):
            raw = _raw_config()
            raw["plan_overrides"][0]["replace_module"] = target.to_dict()
            roundtrip = resolve_config(yaml.safe_load(yaml.safe_dump(raw))).plan_overrides[0].replace_module
            self.assertEqual(roundtrip.to_dict(), target.to_dict())
            self.assertIs(roundtrip.callable, target.callable)
        self.assertEqual(restored.selectors, {"algorithm": "reference", "fused": False})

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_invalid_configuration_fails_before_model_construction(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Unknown combinations/types and ordinary constructor typos fail in configuration.
        Expectation: Malformed selectors and unknown kwargs fail before constructing a module.
        """
        for selectors in (
            {"fused": True},
            {"algorithm": "unknown"},
            {"fused": 1},
            {"algorithm": "matmul", "typo": True},
        ):
            with self.subTest(selectors=selectors), self.assertRaises(ConfigResolutionError):
                resolve_config(_raw_config(**selectors))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_target_overrides_reject_extra_path_segments(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Scalar selectors are terminal path elements; suffixes are not silently discarded.
        Expectation: Overrides reselect correctly and reject child paths on scalar selectors.
        """
        original = resolve_config(_raw_config()).plan_overrides[0].replace_module
        alternate = replace_override_path(original, ["algorithm"], "matmul", path="replace_module")
        self.assertIs(alternate.callable, MatmulLinear)
        with self.assertRaises(ConfigResolutionError):
            replace_override_path(original, ["algorithm", "extra"], "matmul", path="replace_module")
        with self.assertRaises(ConfigResolutionError):
            replace_override_path(original, ["fused"], True, path="replace_module")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_ordinary_replacements_keep_original_target_policy(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Unregistered factories remain ordinary Targets, including same-named parameters.
        Expectation: Non-Family factories retain the existing Target and factory behavior.
        """
        raw = _raw_config()
        raw["plan_overrides"][0]["replace_module"] = {
            "_target_": f"{__name__}.ordinary_factory",
            "algorithm": "not-a-selector",
            "custom_option": 42,
        }
        config = resolve_config(raw)
        self.assertIs(type(config.plan_overrides[0].replace_module), Target)
        model = _install(nn.Sequential(nn.Linear(4, 3)), config)
        self.assertEqual(model[0].algorithm, "not-a-selector")
        self.assertEqual(model[0].options, {"custom_option": 42, "callback": None})

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_programmatic_target_uses_same_family_selection(self) -> None:
        """
        Feature: Module Family construction-time selection.
        Description: Python Target callers receive the same selected implementation without serialization.
        Expectation: Python Targets select the same class while ordinary callbacks preserve identity.
        """
        config = resolve_config(_raw_config())
        config.plan_overrides[0].replace_module = Target(
            ReferenceLinear,
            target_path=f"{__name__}.ReferenceLinear",
            algorithm="matmul",
        )
        model = _install(nn.Sequential(nn.Linear(4, 3)), config)
        self.assertIsInstance(model[0], MatmulLinear)

        def callback(value: object) -> object:
            """Keep object identity through Python Target construction."""
            return value

        config.plan_overrides[0].replace_module = Target(
            ordinary_factory,
            target_path=f"{__name__}.ordinary_factory",
            callback=callback,
        )
        model = _install(nn.Sequential(nn.Linear(4, 3)), config)
        self.assertIs(model[0].options["callback"], callback)
