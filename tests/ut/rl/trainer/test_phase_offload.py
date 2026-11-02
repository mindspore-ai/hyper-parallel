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
"""CPU contracts for stage-only training-state residency."""

import copy
import unittest
from types import SimpleNamespace

import torch

from rl.roles.phase_offload import TrainingStateOffload
from rl.trainer import SyncTrainer

from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from tests.common.mark_utils import arg_mark


def build_optimizer(model: torch.nn.Module) -> Float16OptimizerWithFloat16Params:
    """Use the actual shared fp32-main optimizer composition."""
    leaf = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    return Float16OptimizerWithFloat16Params(ChainedOptimizer(model, {"adamw": leaf}), model)


def update(model: torch.nn.Module, optimizer: Float16OptimizerWithFloat16Params) -> None:
    """Run a reproducible CPU optimizer update and clear all gradients."""
    model(torch.ones(2, 3, dtype=torch.bfloat16)).float().square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestPhaseOffload(unittest.TestCase):
    """Preserve optimizer identity, CPU policy, and update semantics."""

    def test_roundtrip_preserves_next_update_and_main_parameter_bindings(self):
        """CPU-offloaded training remains CPU resident and update-identical."""
        torch.manual_seed(31)
        model = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
        reference = copy.deepcopy(model)
        optimizer = build_optimizer(model)
        reference_optimizer = build_optimizer(reference)
        manager = TrainingStateOffload([(model, optimizer)])
        parameters = tuple(model.parameters())
        mains = tuple(optimizer.fp32_from_float16_groups[0])
        update(model, optimizer)
        update(reference, reference_optimizer)
        state = optimizer.chained_optimizers[0].state
        state_ids = {id(tensor) for values in state.values() for tensor in values.values()}
        self.assertEqual(manager.offload(), 0)
        self.assertTrue(manager.offloaded)
        self.assertEqual(manager.offload(), 0)
        self.assertEqual(manager.restore(), 0)
        self.assertFalse(manager.offloaded)
        self.assertEqual(manager.restore(), 0)
        self.assertEqual(tuple(map(id, model.parameters())), tuple(map(id, parameters)))
        self.assertEqual(tuple(map(id, optimizer.fp32_from_float16_groups[0])), tuple(map(id, mains)))
        self.assertEqual(state_ids, {id(t) for values in state.values() for t in values.values()})
        update(model, optimizer)
        update(reference, reference_optimizer)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_states = reference_optimizer.chained_optimizers[0].state.values()
        for actual, expected in zip(state.values(), expected_states):
            for key in actual:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_live_gradient_rejected_before_mutation(self):
        """Offload must never discard unfinished accumulated gradients."""
        model = torch.nn.Linear(3, 2)
        model(torch.ones(1, 3)).sum().backward()
        manager = TrainingStateOffload([(model, None)])
        with self.assertRaisesRegex(RuntimeError, "cleared gradients"):
            manager.offload()
        self.assertFalse(manager.offloaded)
        self.assertIsNotNone(model.weight.grad)


class TestResidentOptimizer(unittest.TestCase):
    """Check residency at publication without real accelerator allocation."""

    def test_residency_allows_scalar_cpu_counter_but_rejects_moment_offload(self) -> None:
        """Tensor moments and master parameters follow the selected device."""
        parameter = torch.nn.Parameter(torch.empty(2, device="meta"))
        optimizer = SimpleNamespace(param_groups=[{"params": [parameter]}], state={parameter: {
            "step": torch.tensor(1.0), "exp_avg": torch.empty(2, device="meta"),
        }})
        SyncTrainer._validate_optimizer_device_residency(optimizer, torch.device("meta"))
        optimizer.state[parameter]["exp_avg"] = torch.zeros(2)
        with self.assertRaisesRegex(RuntimeError, "optimizer state"):
            SyncTrainer._validate_optimizer_device_residency(optimizer, torch.device("meta"))
        optimizer.state[parameter]["exp_avg"] = torch.zeros(())
        with self.assertRaisesRegex(RuntimeError, "optimizer state"):
            SyncTrainer._validate_optimizer_device_residency(optimizer, torch.device("meta"))
        optimizer.param_groups[0]["params"] = [torch.zeros(2)]
        with self.assertRaisesRegex(RuntimeError, "optimizer parameter"):
            SyncTrainer._validate_optimizer_device_residency(optimizer, torch.device("meta"))
