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
"""Auxiliary-loss-free load balancing: the sign of the correction.

The router picks experts by ``sigmoid(gate) + e_score_correction_bias``. The
bias is only useful if it pushes against the observed load -- an over-used
expert must have its bias *decreased*, or the correction reinforces the very
imbalance it exists to undo, which is worse than not updating at all. These
tests pin the direction, the zero-mean centering (which stops the whole vector
drifting while the ranking stays put), and the recompute correction.

The collective is stubbed: what is under test is the arithmetic the ranks agree
on, not the all-reduce that carries it.
"""
# pylint: disable=wrong-import-position

import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.distributed.expert_parallel import experts as ep_experts
from hyper_parallel.trainer.callbacks.expert_bias_callback import (
    ExpertBiasUpdateCallback,
)
from tests.common.mark_utils import arg_mark


def _moe(num_experts: int = 4) -> torch.nn.Module:
    """Build a stand-in MoE module: a router with a bias, and nothing else."""
    gate = torch.nn.Module()
    gate.e_score_correction_bias = torch.nn.Parameter(torch.zeros(num_experts))
    module = torch.nn.Module()
    module.gate = gate
    return module


def _update(module, lr=1e-3, num_recomputations=1, load=None) -> bool:
    """Run one update against a stubbed all-reduce (single-rank world)."""
    if load is not None:
        module._ep_expert_load = torch.as_tensor(load, dtype=torch.float32)  # pylint: disable=protected-access
    with patch.object(
        ep_experts.dist, "all_reduce", side_effect=lambda tensor, group=None: None
    ):
        return ep_experts.update_expert_bias_from_load(
            module, lr=lr, group=None, num_recomputations=num_recomputations
        )


class TestAccumulateExpertLoad(unittest.TestCase):
    """Route counts land in a float32 accumulator of the right width."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_counts_are_the_histogram_of_the_routes(self):
        """A top-2 batch counts every pick, not only the winning expert."""
        module = _moe(4)
        # 2 tokens x top-2: experts [0, 1] and [1, 3].
        topk = torch.tensor([[0, 1], [1, 3]])
        ep_experts.accumulate_expert_load(module, topk, 4)

        self.assertEqual(
            module._ep_expert_load.tolist(),  # pylint: disable=protected-access
            [1.0, 2.0, 0.0, 1.0],
            f"unexpected load histogram: {module._ep_expert_load.tolist()}",  # pylint: disable=protected-access
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_successive_forwards_accumulate(self):
        """Two forwards before one step are summed, not overwritten."""
        module = _moe(3)
        topk = torch.tensor([[0, 2]])
        ep_experts.accumulate_expert_load(module, topk, 3)
        ep_experts.accumulate_expert_load(module, topk, 3)

        self.assertEqual(
            module._ep_expert_load.tolist(),  # pylint: disable=protected-access
            [2.0, 0.0, 2.0],
            f"load must accumulate across forwards: {module._ep_expert_load.tolist()}",  # pylint: disable=protected-access
        )


class TestBiasUpdateDirection(unittest.TestCase):
    """The correction must oppose the load, and leave the scale alone."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_over_used_experts_lose_bias_and_under_used_gain(self):
        """The most-loaded expert must be pushed down, the rest up."""
        module = _moe(4)
        updated = _update(module, lr=1e-3, load=[10.0, 1.0, 1.0, 1.0])

        self.assertTrue(updated, "an update with load and bias present must report a change")
        bias = module.gate.e_score_correction_bias.data
        self.assertLess(
            float(bias[0]), 0.0, f"over-used expert bias must fall, got {float(bias[0])}"
        )
        for index in (1, 2, 3):
            with self.subTest(expert=index):
                self.assertGreater(
                    float(bias[index]),
                    0.0,
                    f"under-used expert {index} bias must rise, got {float(bias[index])}",
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_the_delta_is_zero_mean_so_the_scale_never_drifts(self):
        """Only the ranking moves; the whole vector does not walk away."""
        module = _moe(4)
        _update(module, lr=1e-3, load=[10.0, 1.0, 1.0, 1.0])

        self.assertAlmostEqual(
            float(module.gate.e_score_correction_bias.data.sum()),
            0.0,
            places=6,
            msg=f"bias sum must stay centered, got {float(module.gate.e_score_correction_bias.data.sum())}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_a_perfectly_balanced_load_moves_nothing(self):
        """Equal load everywhere means equal sign, which centers to zero."""
        module = _moe(4)
        _update(module, lr=1e-3, load=[5.0, 5.0, 5.0, 5.0])

        self.assertTrue(
            torch.equal(module.gate.e_score_correction_bias.data, torch.zeros(4)),
            f"a balanced load must leave the bias alone, "
            f"got {module.gate.e_score_correction_bias.data.tolist()}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_repeated_updates_walk_the_bias_towards_the_load(self):
        """The sign rule keeps pushing while the imbalance persists."""
        module = _moe(4)
        for _ in range(5):
            _update(module, lr=1e-3, load=[10.0, 1.0, 1.0, 1.0])

        bias = module.gate.e_score_correction_bias.data
        self.assertLess(
            float(bias[0]),
            -4e-3,
            f"five steps must accumulate the correction, got {float(bias[0])}",
        )
        self.assertGreater(
            float(bias[1]),
            1e-3,
            f"the under-used expert must have gained bias, got {float(bias[1])}",
        )


class TestBiasUpdateBookkeeping(unittest.TestCase):
    """Recompute correction, accumulator reset, and the no-op cases."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_recomputation_divides_the_double_counted_load(self):
        """Activation checkpoint routes every token twice; the load must not care."""
        once, twice = _moe(4), _moe(4)
        _update(once, lr=1e-3, num_recomputations=1, load=[20.0, 2.0, 2.0, 2.0])
        _update(twice, lr=1e-3, num_recomputations=2, load=[40.0, 4.0, 4.0, 4.0])

        self.assertTrue(
            torch.equal(
                once.gate.e_score_correction_bias.data,
                twice.gate.e_score_correction_bias.data,
            ),
            f"recompute correction must cancel: once="
            f"{once.gate.e_score_correction_bias.data.tolist()}, "
            f"twice={twice.gate.e_score_correction_bias.data.tolist()}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_the_accumulator_is_reset_after_each_step(self):
        """A stale accumulator would inflate the next step's statistic."""
        module = _moe(3)
        _update(module, load=[4.0, 0.0, 0.0])

        self.assertEqual(
            module._ep_expert_load.tolist(),  # pylint: disable=protected-access
            [0.0, 0.0, 0.0],
            f"load must be reset after the update: {module._ep_expert_load.tolist()}",  # pylint: disable=protected-access
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_a_module_that_never_routed_is_left_alone(self):
        """Dense layers have no load accumulator and must not raise."""
        module = _moe(4)
        bias = module.gate.e_score_correction_bias.data.clone()

        self.assertFalse(
            _update(module), "a module with no accumulated load must report no update"
        )
        self.assertTrue(
            torch.equal(module.gate.e_score_correction_bias.data, bias),
            f"bias must be untouched, got {module.gate.e_score_correction_bias.data.tolist()}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_a_router_without_the_bias_is_left_alone(self):
        """Routers outside the noaux_tc path carry no correction bias."""
        module = torch.nn.Module()
        module.gate = torch.nn.Module()
        module._ep_expert_load = torch.zeros(4)  # pylint: disable=protected-access

        self.assertFalse(
            _update(module), "a router without a correction bias must report no update"
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_a_load_bias_width_mismatch_is_rejected(self):
        """A silent reshape would correct the wrong expert bank."""
        module = _moe(4)
        with self.assertRaisesRegex(ValueError, "same expert bank"):
            _update(module, load=[1.0, 2.0, 3.0])


class _MeshEntry:
    """Stand-in for the MoE mesh; only the EP group lookup is exercised."""

    def __init__(self, ep_group: Any) -> None:
        """Record the group the EP axis resolves to."""
        self.ep_group = ep_group
        self.requested: list[str] = []

    def get_group(self, name: str) -> Any:
        """Return the recorded group and remember the axis that was asked for."""
        self.requested.append(name)
        return self.ep_group


class _FakeTrainer:
    """Stand-in trainer exposing the model, mesh and activation-checkpoint mode."""

    def __init__(
        self,
        model: torch.nn.Module,
        mesh_entry: Any,
        ac_mode: str = "off",
    ) -> None:
        """Build a trainer view over ``model`` with the given mesh and mode."""
        self.model = model
        self.mesh = SimpleNamespace(fsdp_moe_mesh=mesh_entry)
        self.config = SimpleNamespace(
            activation_checkpoint=SimpleNamespace(mode=ac_mode)
        )


def _routed_model(*loads) -> torch.nn.Module:
    """Build a model of routed MoE layers carrying the given loads."""
    model = torch.nn.Module()
    for index, load in enumerate(loads):
        layer = _moe(len(load))
        layer._ep_expert_load = torch.as_tensor(load, dtype=torch.float32)  # pylint: disable=protected-access
        model.add_module(f"layer{index}", layer)
    return model


def _layers(model: torch.nn.Module) -> list:
    """Return the routed layers of a model built by ``_routed_model``."""
    return list(model.children())


class TestExpertBiasUpdateCallback(unittest.TestCase):
    """The callback is the per-step trigger for the rule above."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_every_routed_layer_is_corrected_on_one_step(self):
        """One step end corrects all routed layers, over the EP group."""
        model = _routed_model([8.0, 0.0], [0.0, 8.0])
        mesh_entry = _MeshEntry(ep_group="ep-group")
        callback = ExpertBiasUpdateCallback(
            _FakeTrainer(model, mesh_entry), lr=1e-3
        )

        with patch.object(
            ep_experts.dist, "all_reduce", side_effect=lambda tensor, group=None: None
        ):
            callback.on_step_end(SimpleNamespace())

        bias = [layer.gate.e_score_correction_bias.data.tolist() for layer in _layers(model)]
        self.assertLess(bias[0][0], 0.0, f"the loaded expert must lose bias, got {bias[0]}")
        self.assertGreater(bias[0][1], 0.0, f"the idle expert must gain bias, got {bias[0]}")
        self.assertGreater(bias[1][0], 0.0, f"layer1's idle expert must gain bias, got {bias[1]}")
        self.assertEqual(
            [layer._ep_expert_load.tolist() for layer in _layers(model)],  # pylint: disable=protected-access
            [[0.0, 0.0], [0.0, 0.0]],
            f"every layer's accumulator must be reset, got "
            f"{[layer._ep_expert_load.tolist() for layer in _layers(model)]}",  # pylint: disable=protected-access
        )
        self.assertEqual(
            mesh_entry.requested,
            ["ep"],
            f"the load must be summed over the EP group, got {mesh_entry.requested}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_activation_checkpoint_halves_the_double_counted_load(self):
        """A recomputed forward routes twice, so the callback divides it out."""
        plain = _routed_model([4.0, 0.0])
        recomputed = _routed_model([8.0, 0.0])
        mesh_entry = _MeshEntry(ep_group="ep-group")

        with patch.object(
            ep_experts.dist, "all_reduce", side_effect=lambda tensor, group=None: None
        ):
            ExpertBiasUpdateCallback(
                _FakeTrainer(plain, mesh_entry, ac_mode="off"), lr=1e-3
            ).on_step_end(SimpleNamespace())
            ExpertBiasUpdateCallback(
                _FakeTrainer(recomputed, mesh_entry, ac_mode="full"), lr=1e-3
            ).on_step_end(SimpleNamespace())

        self.assertTrue(
            torch.equal(
                plain.layer0.gate.e_score_correction_bias.data,
                recomputed.layer0.gate.e_score_correction_bias.data,
            ),
            f"recompute correction must cancel: plain="
            f"{plain.layer0.gate.e_score_correction_bias.data.tolist()}, recomputed="
            f"{recomputed.layer0.gate.e_score_correction_bias.data.tolist()}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_the_callback_is_inert_without_a_step_size(self):
        """A disabled callback costs one attribute read and changes nothing."""
        model = _routed_model([8.0, 0.0])
        bias = model.layer0.gate.e_score_correction_bias.data.clone()
        callback = ExpertBiasUpdateCallback(
            _FakeTrainer(model, _MeshEntry(ep_group="ep-group")), lr=0.0
        )

        callback.on_step_end(SimpleNamespace())

        self.assertTrue(
            torch.equal(model.layer0.gate.e_score_correction_bias.data, bias),
            f"disabled callback changed the bias to "
            f"{model.layer0.gate.e_score_correction_bias.data.tolist()}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_env_var_sets_the_step_size(self):
        """``HP_EXPERT_BIAS_LR`` enables the callback; 0 leaves it disabled."""
        trainer = _FakeTrainer(_routed_model([1.0, 0.0]), _MeshEntry(ep_group=None))

        with patch.dict(os.environ, {"HP_EXPERT_BIAS_LR": "2e-3"}):
            self.assertEqual(
                ExpertBiasUpdateCallback(trainer).lr, 2e-3, "env lr must reach the callback"
            )
        with patch.dict(os.environ, {"HP_EXPERT_BIAS_LR": "0"}):
            self.assertIsNone(
                ExpertBiasUpdateCallback(trainer).lr, "a zero step size must disable the update"
            )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_a_trainer_without_the_moe_mesh_fails_loudly(self):
        """An unresolved EP group must raise instead of summing over nothing."""
        model = _routed_model([8.0, 0.0])
        callback = ExpertBiasUpdateCallback(
            _FakeTrainer(model, _MeshEntry(ep_group="ep-group")), lr=1e-3
        )
        callback.mesh = SimpleNamespace(fsdp_moe_mesh=None)

        with self.assertRaisesRegex(RuntimeError, "fsdp_moe_mesh"):
            callback.on_step_end(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
