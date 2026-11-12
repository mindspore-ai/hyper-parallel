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
"""Full-state coordinate migration and resumed optimizer update acceptance."""

from copy import deepcopy
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.optim import optimizer as torch_optimizer

from hyper_parallel.components.checkpoint.conversion_ops import ConcatenateWithSections
from hyper_parallel.components.checkpoint.weight_conversion import Transpose
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from tests.common.mark_utils import arg_mark
from tests.common.training_state import (
    TensorStateTransform, capture_full_training_state, convert_full_training_state, restore_full_training_state,
)


class _OldExperts(nn.Module):
    """Source expert using separate gate, up and down projections."""

    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        """Register the complete expert weight representation.

        Args:
            dtype: Persistent weight dtype.
        """
        super().__init__()
        self.gate = nn.Parameter(torch.arange(6, dtype=dtype).reshape(3, 2) / 20)
        self.up = nn.Parameter(torch.arange(6, dtype=dtype).reshape(3, 2) / 30 + 0.1)
        self.down = nn.Parameter(torch.arange(6, dtype=dtype).reshape(2, 3) / 10)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute the old three-weight expert algorithm.

        Args:
            tokens: Local expert input activations.
        """
        return (torch.nn.functional.silu(tokens @ self.gate.t()) * (tokens @ self.up.t())) @ self.down.t()


class _PackedExperts(nn.Module):
    """Target expert weights using the converted packed projection layout."""

    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        """Register the complete expert weight representation.

        Args:
            dtype: Persistent weight dtype.
        """
        super().__init__()
        self.gate_up = nn.Parameter(torch.zeros(2, 6, dtype=dtype))
        self.down = nn.Parameter(torch.zeros(3, 2, dtype=dtype))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute the equivalent packed expert algorithm.

        Args:
            tokens: Local expert input activations.
        """
        gate, up = (tokens @ self.gate_up).chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate) * up) @ self.down


def _transforms():
    return (
        TensorStateTransform(("gate", "up"), ("gate_up",),
                             (ConcatenateWithSections((3, 3)), Transpose(0, 1))),
        TensorStateTransform(("down",), ("down",), (Transpose(0, 1),)),
    )


def _update(model, optimizer, probability_mode):
    tokens = torch.tensor([[0.3, -0.2], [0.5, 0.1]], dtype=next(model.parameters()).dtype)
    weights = torch.tensor([[0.2], [0.8]], dtype=tokens.dtype, requires_grad=True)
    output = model(tokens * weights) if probability_mode == "input" else model(tokens) * weights
    output.float().square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return output.detach(), weights.grad.detach()


def _optimizer(model, kind, mixed=False):
    constructors = {
        "Adam": lambda params: torch.optim.Adam(params, lr=0.01, foreach=False),
        "AdamW": lambda params: torch.optim.AdamW(params, lr=0.01, amsgrad=True, foreach=False),
        "SGD": lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9, foreach=False),
    }
    base = constructors[kind](model.parameters())
    return Float16OptimizerWithFloat16Params(ChainedOptimizer(model, {"adamw": base}), model) if mixed else base


class TestTrainingState(unittest.TestCase):
    """Test full migration as a mathematical change of parameter coordinates."""

    def setUp(self) -> None:
        """Preserve RNG state and keep CPU optimizer updates independent of accelerators."""
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        hardware_probe = patch.object(
            torch_optimizer, "_get_foreach_kernels_supported_devices",
            side_effect=AssertionError("CPU optimizer tests must not probe accelerator devices"),
        )
        hardware_probe.start()
        self.addCleanup(hardware_probe.stop)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_optimizer_resume_after_merge_transpose_and_inverse(self):
        """Feature: Complete optimizer-state migration.
        Description: Migrate merged/transposed expert weights and resume both routing probability modes.
        Expectation: Outputs, probability gradients, second updates, moments and inverse weights match the source.
        """
        for kind in ("Adam", "AdamW", "SGD"):
            for mode in ("input", "output"):
                with self.subTest(optimizer=kind, probability=mode):
                    old, target = _OldExperts(), _PackedExperts()
                    source_opt, target_opt = _optimizer(old, kind), _optimizer(target, kind)
                    _update(old, source_opt, mode)
                    state = capture_full_training_state(old, source_opt, global_step=1, data_progress={"sample": 8})
                    original = deepcopy(state)
                    migrated = convert_full_training_state(state, _transforms())
                    extra = restore_full_training_state(target, target_opt, migrated)
                    self.assertEqual(extra["global_step"], 1)
                    self.assertEqual(extra["data_progress"], {"sample": 8})
                    actual, actual_probability_grad = _update(target, target_opt, mode)
                    expected, expected_probability_grad = _update(old, source_opt, mode)
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(actual_probability_grad, expected_probability_grad)
                    expected_state = convert_full_training_state(capture_full_training_state(
                        old, source_opt, global_step=2, data_progress={"sample": 16}), _transforms())
                    actual_state = capture_full_training_state(target, target_opt, global_step=2,
                                                               data_progress={"sample": 16})
                    for name, value in expected_state["model"].items():
                        torch.testing.assert_close(actual_state["model"][name], value)
                    for name, values in expected_state["optimizer"]["state"].items():
                        for key, value in values.items():
                            torch.testing.assert_close(actual_state["optimizer"]["state"][name][key], value)
                    reversed_state = convert_full_training_state(
                        migrated, tuple(rule.inverse() for rule in _transforms()))
                    for name, value in original["model"].items():
                        torch.testing.assert_close(reversed_state["model"][name], value)
                        torch.testing.assert_close(state["model"][name], value)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_fp32_master_weights_resume_without_rounding_loss(self):
        """Feature: FP32 main-weight continuation.
        Description: Migrate BF16 expert parameters and their optimizer-owned FP32 main weights.
        Expectation: Restoration preserves FP32 values exactly and the next update matches the original representation.
        """
        old, target = _OldExperts(torch.bfloat16), _PackedExperts(torch.bfloat16)
        source_opt, target_opt = _optimizer(old, "AdamW", True), _optimizer(target, "AdamW", True)
        _update(old, source_opt, "output")
        source = capture_full_training_state(old, source_opt, global_step=1, data_progress={"sample": 2})
        migrated = convert_full_training_state(source, _transforms())
        self.assertTrue(all(value.dtype == torch.float32 for value in migrated["master_weights"].values()))
        restore_full_training_state(target, target_opt, migrated)
        restored = capture_full_training_state(target, target_opt, global_step=1, data_progress={"sample": 2})
        for name, value in migrated["master_weights"].items():
            torch.testing.assert_close(restored["master_weights"][name], value, rtol=0, atol=0)
        _update(old, source_opt, "output")
        _update(target, target_opt, "output")
        expected = convert_full_training_state(capture_full_training_state(old, source_opt, global_step=2,
                                                                         data_progress={}), _transforms())
        actual = capture_full_training_state(target, target_opt, global_step=2, data_progress={})
        for name, value in expected["master_weights"].items():
            torch.testing.assert_close(actual["master_weights"][name], value, rtol=1e-4, atol=2e-5)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_invalid_groups_steps_states_and_pending_gradients_fail(self):
        """Feature: Training-state migration validation.
        Description: Attempt conversion with pending gradients, unknown moments, mismatched steps or parameter groups.
        Expectation: Unsafe conversions fail explicitly without inferring parameter names from unbound saved IDs.
        """
        old = _OldExperts()
        optimizer = _optimizer(old, "AdamW")
        old(torch.ones(1, 2, dtype=torch.float64)).sum().backward()
        with self.assertRaisesRegex(ValueError, "completed step"):
            capture_full_training_state(old, optimizer, global_step=0, data_progress={})
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state = capture_full_training_state(old, optimizer, global_step=1, data_progress={})
        bad = deepcopy(state)
        bad["optimizer"]["state"]["gate"]["row_factor"] = torch.ones(3)
        with self.assertRaisesRegex(ValueError, "identical|unsupported"):
            convert_full_training_state(bad, _transforms())
        bad = deepcopy(state)
        bad["optimizer"]["state"]["gate"]["step"] += 1
        with self.assertRaisesRegex(ValueError, "different optimizer steps"):
            convert_full_training_state(bad, _transforms())
        bad = deepcopy(state)
        bad["optimizer"]["param_groups"].append({**bad["optimizer"]["param_groups"][0], "params": ["up"]})
        bad["optimizer"]["param_groups"][0]["params"].remove("up")
        with self.assertRaisesRegex(ValueError, "different optimizer groups"):
            convert_full_training_state(bad, _transforms())
        bad = deepcopy(state)
        bad["optimizer"]["param_groups"][0]["params"][0] = 0
        with self.assertRaisesRegex(ValueError, "FQNs"):
            convert_full_training_state(bad, _transforms())

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_rng_and_freeze_contract_restore(self):
        """Feature: RNG and frozen-parameter restoration.
        Description: Restore a step-boundary snapshot and then attempt loading an incompatible freeze contract.
        Expectation: All RNG sequences resume exactly and incompatible loading leaves target weights unchanged.
        """
        model = _OldExperts()
        optimizer = _optimizer(model, "SGD")
        state = capture_full_training_state(model, optimizer, global_step=0, data_progress={"epoch": 3})
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_full_training_state(model, optimizer, state)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
        model.up.requires_grad_(False)
        before = dict(model.named_parameters())["gate"].detach().clone()
        with self.assertRaisesRegex(ValueError, "freeze/alias"):
            restore_full_training_state(model, optimizer, state)
        torch.testing.assert_close(model.gate, before)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_named_optimizer_groups_and_partial_alias_conversion(self):
        """Feature: Named optimizer groups and shared parameters.
        Description: Convert named groups, reject a target dtype mismatch and exercise shared-parameter transforms.
        Expectation: Names follow target FQNs; incomplete alias conversion fails and consistent transforms
            preserve aliases.
        """
        model = _OldExperts()
        optimizer = torch.optim.AdamW([{"params": list(model.parameters()),
                                       "param_names": ["gate", "up", "down"]}], lr=0.01, foreach=False)
        _update(model, optimizer, "output")
        state = capture_full_training_state(model, optimizer, global_step=1, data_progress={})
        converted = convert_full_training_state(state, _transforms())
        self.assertEqual(converted["optimizer"]["param_groups"][0]["param_names"], ["gate_up", "down"])
        different_dtype = _PackedExperts(torch.float32)
        target_optimizer = _optimizer(different_dtype, "AdamW")
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            restore_full_training_state(different_dtype, target_optimizer, converted)
        aliased = nn.Module()
        aliased.weight = nn.Parameter(torch.tensor([[1., 2.], [3., 4.]]))
        aliased.tied = aliased.weight
        tied_opt = torch.optim.SGD(aliased.parameters(), lr=0.1, foreach=False)
        state = capture_full_training_state(aliased, tied_opt, global_step=0, data_progress={})
        rule = TensorStateTransform(("weight",), ("transposed",), (Transpose(0, 1),))
        with self.assertRaisesRegex(ValueError, "shared parameter aliases"):
            convert_full_training_state(state, (rule,))
        second = TensorStateTransform(("tied",), ("transposed_tied",), (Transpose(0, 1),))
        result = convert_full_training_state(state, (rule, second))
        self.assertEqual(result["extra_state"]["aliases"], [["transposed", "transposed_tied"]])
