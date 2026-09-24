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
"""Trainer-side wiring of the ``optimizer.swap`` configuration section.

``optimizer.swap`` must be a no-op for every existing config and must wrap the
Adam/AdamW leaves the YAML optimizer target builds when it is enabled. The
optimizer object the lr scheduler and the trainer loop see keeps proxying
``param_groups``, ``state_dict`` and ``load_state_dict`` either way.
"""
# pylint: disable=wrong-import-position

import os
import unittest
from unittest import mock

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch import nn

from tests.common.mark_utils import arg_mark

from hyper_parallel.components.optim.builders import AdamW as AdamWBuilder
from hyper_parallel.core.optimizer import ChainedOptimizer, SwapOptimizerConfig
from hyper_parallel.core.optimizer.swap_optimizer import is_swap_optimizer
from hyper_parallel.models.build_options import FSDP2Config, FSDP2MixedPrecisionConfig
from hyper_parallel.core.optimizer import swap_optimizer_base as swap_adapters
from hyper_parallel.trainer import base as trainer_base
from hyper_parallel.trainer.base import BaseTrainer, _to_swap_optimizer_config
from hyper_parallel.trainer.config import (
    OptimizerConfig,
    OptimizerSwapConfig,
    Target,
    TrainerConfig,
)
from hyper_parallel.trainer.config.resolver import ConfigResolutionError, resolve_config

_MARK = {
    "plat_marks": ["cpu_linux", "cpu_macos"],
    "level_mark": "level0",
    "card_mark": "allcards",
    "essential_mark": "essential",
}
_THIS_MODULE = "tests.ut.trainer.test_optimizer_swap"
_ADAMW_TARGET = "hyper_parallel.components.optim.builders.AdamW"


class _Model(nn.Module):
    """Smallest model that gives AdamW two parameter groups."""

    def __init__(self) -> None:
        """Initialize one decaying projection and one non-decaying norm."""
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the projection followed by normalization."""
        return self.norm(self.proj(inputs))


def _no_model(**kwargs) -> None:
    """Model target placeholder: ``_build_optimizer`` never builds the model."""
    del kwargs


def _builder_target(optimizer: torch.optim.Optimizer) -> Target:
    """Return a target whose builder hands back ``optimizer`` unchanged.

    A real optimizer target cannot be asked for the same instance twice, and the
    no-swap contract is about object identity.
    """

    class _Builder:
        """Stand-in for a YAML optimizer target body."""

        def __init__(self, model: nn.Module) -> None:
            """Keep the injected model; the optimizer is the prebuilt one."""
            self.model = model

        def get_optimizer(self) -> torch.optim.Optimizer:
            """Return the optimizer the test prebuilt."""
            return optimizer

    return Target(_Builder, target_path=f"{_THIS_MODULE}._Builder")


def _trainer_config(optimizer_target: Target, swap: OptimizerSwapConfig,
                    fp32_main_params: bool = False) -> TrainerConfig:
    """Build a trainer config holding only the fields the build path reads."""
    fsdp_config = FSDP2Config(mix_precision=FSDP2MixedPrecisionConfig(reduce_dtype="float32"))
    return TrainerConfig(
        model=Target(_no_model, target_path=f"{_THIS_MODULE}._no_model"),
        optimizer=OptimizerConfig(
            target=optimizer_target,
            fp32_main_params=fp32_main_params,
            swap=swap,
        ),
        fsdp_config=fsdp_config,
    )


def _build_optimizer(config: TrainerConfig, model: nn.Module):
    """Run ``BaseTrainer._build_optimizer`` without constructing a trainer."""
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.config = config
    trainer.model = model
    trainer._build_optimizer()
    return trainer.optimizer


def _materialize_adam_state(optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> None:
    """Populate deterministic Adam state without executing an optimizer update."""
    values = torch.arange(1, param.numel() + 1, dtype=param.dtype).view_as(param)
    state = optimizer.state[param]
    state["step"] = torch.tensor(1.0)
    state["exp_avg"] = values.clone()
    state["exp_avg_sq"] = values.square()


class TestOptimizerSwapConfig(unittest.TestCase):
    """Defaults, validation and YAML shape of ``OptimizerSwapConfig``."""

    @arg_mark(**_MARK)
    def test_defaults_match_the_swap_runtime(self):
        """Feature: optimizer.swap defaults.
        Description: an omitted YAML section constructs ``OptimizerSwapConfig()``.
        Expectation: swap stays disabled and every field keeps the runtime default.
        """
        swap = OptimizerSwapConfig()

        self.assertFalse(swap.enabled)
        self.assertEqual(swap.swap_times, 16)
        self.assertEqual(swap.min_numel, 1024)
        self.assertIsNone(swap.state_keys)
        self.assertFalse(swap.include_master_params)
        self.assertIsNone(swap.packed_swap)
        self.assertEqual(OptimizerConfig(
            target=Target(_no_model, target_path=f"{_THIS_MODULE}._no_model"),
        ).swap, swap)

    @arg_mark(**_MARK)
    def test_rejects_non_positive_swap_times(self):
        """Feature: optimizer.swap validation.
        Description: construct the section with a zero or negative partition count.
        Expectation: both fail fast and a single partition is accepted.
        """
        for swap_times in (0, -1):
            with self.subTest(swap_times=swap_times):
                with self.assertRaisesRegex(ValueError, "swap_times"):
                    OptimizerSwapConfig(swap_times=swap_times)

        self.assertEqual(OptimizerSwapConfig(swap_times=1).swap_times, 1)

    @arg_mark(**_MARK)
    def test_rejects_negative_min_numel(self):
        """Feature: optimizer.swap validation.
        Description: construct the section with a negative state-size floor.
        Expectation: it fails fast and zero (swap everything) is accepted.
        """
        with self.assertRaisesRegex(ValueError, "min_numel"):
            OptimizerSwapConfig(min_numel=-1)

        self.assertEqual(OptimizerSwapConfig(min_numel=0).min_numel, 0)

    @arg_mark(**_MARK)
    def test_rejects_unknown_state_keys(self):
        """Feature: optimizer.swap state-key validation.
        Description: request a logical slot the swap runtime does not implement.
        Expectation: it fails fast; the Adam moments stay accepted.
        """
        with self.assertRaisesRegex(ValueError, "state_keys"):
            OptimizerSwapConfig(state_keys=["momentum_buffer"])

        accepted = OptimizerSwapConfig(state_keys=["exp_avg", "exp_avg_sq"])
        self.assertEqual(accepted.state_keys, ["exp_avg", "exp_avg_sq"])

    @arg_mark(**_MARK)
    def test_resolves_from_yaml_and_serializes_back(self):
        """Feature: optimizer.swap YAML section.
        Description: resolve the nested section, then serialize the config back.
        Expectation: every field lands on the dataclass and round-trips.
        """
        swap_node = {
            "enabled": True,
            "swap_times": 4,
            "min_numel": 8,
            "state_keys": ["exp_avg"],
            "include_master_params": False,
            "packed_swap": False,
        }
        config = resolve_config({
            "model": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 2},
            "optimizer": {
                "_target_": _ADAMW_TARGET,
                "no_decay_params": ["bias", "norm", "ln_"],
                "adamw_config": {"adamw_lr": 1.0e-5, "adamw_weight_decay": 0.01},
                "swap": swap_node,
            },
        })

        self.assertEqual(config.optimizer.swap, OptimizerSwapConfig(**swap_node))
        self.assertEqual(config.optimizer.to_dict()["swap"], swap_node)

    @arg_mark(**_MARK)
    def test_rejects_invalid_yaml_swap_values(self):
        """Feature: optimizer.swap YAML validation.
        Description: resolve sections with a bad width, slot name and field name.
        Expectation: each fails at resolve time instead of at the first step.
        """
        root = {
            "model": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 2},
            "optimizer": {
                "_target_": _ADAMW_TARGET,
                "adamw_config": {"adamw_lr": 1.0e-5},
                "swap": {"swap_times": 0},
            },
        }
        with self.assertRaisesRegex(ValueError, "swap_times"):
            resolve_config(root)

        root["optimizer"]["swap"] = {"state_keys": ["momentum_buffer"]}
        with self.assertRaisesRegex(ValueError, "state_keys"):
            resolve_config(root)

        root["optimizer"]["swap"] = {"unknown": 1}
        with self.assertRaisesRegex(ConfigResolutionError, "unknown configuration fields"):
            resolve_config(root)


class TestOptimizerSwapBuildPath(unittest.TestCase):
    """``_build_optimizer`` attaches swap only when it is enabled."""

    def setUp(self) -> None:
        """Give every test its own model."""
        self.model = _Model()

    @arg_mark(**_MARK)
    def test_disabled_keeps_the_built_optimizer_unchanged(self):
        """Feature: optimizer.swap disabled.
        Description: build the trainer optimizer with the default section.
        Expectation: the trainer holds exactly the optimizer the target built.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        config = _trainer_config(_builder_target(optimizer), OptimizerSwapConfig())

        built = _build_optimizer(config, self.model)

        self.assertIs(built, optimizer)
        self.assertFalse(is_swap_optimizer(built))

    @arg_mark(**_MARK)
    def test_enabled_wraps_a_bare_adamw(self):
        """Feature: optimizer.swap with a bare AdamW target.
        Description: build with swap enabled over the fused AdamW the real config uses.
        Expectation: the trainer optimizer is the swap wrapper, the optimizer is inside.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, fused=True)
        config = _trainer_config(_builder_target(optimizer), OptimizerSwapConfig(enabled=True))

        built = _build_optimizer(config, self.model)

        self.assertTrue(is_swap_optimizer(built))
        self.assertIsInstance(built.adapter, swap_adapters.TorchNativeAdamWAdapter)
        self.assertIs(built.optimizer, optimizer)

    @arg_mark(**_MARK)
    def test_enabled_wraps_every_leaf_of_a_chained_optimizer(self):
        """Feature: optimizer.swap with a chained optimizer.
        Description: build the shipped AdamW YAML target, which chains its leaves.
        Expectation: every leaf is swap-wrapped and the chain still lists them.
        """
        target = Target(
            AdamWBuilder,
            target_path=_ADAMW_TARGET,
            no_decay_params=["bias", "norm"],
            adamw_config={"adamw_lr": 1.0e-5, "adamw_weight_decay": 0.01},
        )
        config = _trainer_config(target, OptimizerSwapConfig(enabled=True, min_numel=1))

        built = _build_optimizer(config, self.model)

        self.assertIsInstance(built, ChainedOptimizer)
        self.assertEqual(list(built.optimizers_dict), ["adamw"])
        for name, leaf in built.optimizers_dict.items():
            with self.subTest(name=name):
                self.assertTrue(is_swap_optimizer(leaf), (
                    f"leaf {name!r} was not swap-wrapped: {type(leaf).__name__}"
                ))
        self.assertEqual(list(built.chained_optimizers), list(built.optimizers_dict.values()))

    @arg_mark(**_MARK)
    def test_swap_config_gets_the_yaml_values(self):
        """Feature: swap config translation.
        Description: build with a fully specified section and capture the swap call.
        Expectation: ``SwapOptimizerConfig`` gets every value field by field.
        """
        swap = OptimizerSwapConfig(
            enabled=True,
            swap_times=7,
            min_numel=64,
            state_keys=["exp_avg"],
            include_master_params=False,
            packed_swap=True,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, fused=True)
        config = _trainer_config(_builder_target(optimizer), swap)

        def _passthrough(target_optimizer, swap_config):
            del swap_config
            return target_optimizer

        with mock.patch.object(trainer_base, "swap_optimizer", side_effect=_passthrough) as swap_mock:
            _build_optimizer(config, self.model)

        self.assertEqual(swap_mock.call_count, 1)
        passed = swap_mock.call_args_list[0][0][1]
        self.assertIsInstance(passed, SwapOptimizerConfig)
        self.assertEqual(passed.swap_times, 7)
        self.assertEqual(passed.min_numel, 64)
        self.assertEqual(list(passed.state_keys), ["exp_avg"])
        self.assertFalse(passed.include_master_params)
        self.assertIs(passed.packed_swap, True)

    @arg_mark(**_MARK)
    def test_unspecified_packed_swap_keeps_the_backend_default(self):
        """Feature: swap config translation.
        Description: leave ``packed_swap`` unset and build on the Torch backend.
        Expectation: the backend default applies instead of a hard False.
        """
        config = _trainer_config(
            _builder_target(torch.optim.AdamW(self.model.parameters(), lr=0.01)),
            OptimizerSwapConfig(enabled=True, packed_swap=None),
        )

        swap_config = _to_swap_optimizer_config(config.optimizer.swap)
        built = _build_optimizer(config, self.model)

        self.assertIs(swap_config.packed_swap, True)
        self.assertTrue(built.runtime.packed_enabled)

    @arg_mark(**_MARK)
    def test_wrapped_optimizer_keeps_param_groups_and_checkpoint_api(self):
        """Feature: swap wrapper transparency.
        Description: wrap an AdamW with materialized state and read the public API.
        Expectation: param_groups identity and state_dict/load_state_dict still work.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, fused=True)
        param = next(iter(optimizer.param_groups[0]["params"]))
        _materialize_adam_state(optimizer, param)
        before = optimizer.state_dict()

        config = _trainer_config(
            _builder_target(optimizer),
            OptimizerSwapConfig(enabled=True, min_numel=1),
        )
        built = _build_optimizer(config, self.model)

        self.assertIs(built.param_groups, optimizer.param_groups)
        self.assertEqual(sorted(built.state_dict()), sorted(before))
        self.assertEqual(built.state_dict()["param_groups"], before["param_groups"])
        self.assertTrue(torch.equal(
            built.state_dict()["state"][0]["exp_avg"], before["state"][0]["exp_avg"]))

        built.load_state_dict(before)
        self.assertTrue(torch.equal(built.state[param]["exp_avg"], before["state"][0]["exp_avg"]))

    @arg_mark(**_MARK)
    def test_chained_optimizer_checkpoint_api_works_through_wrapped_leaves(self):
        """Feature: chained optimizer checkpoint API.
        Description: snapshot and restore a swap-wrapped chained optimizer on CPU.
        Expectation: the chained state dict keeps its shape and loads back.
        """
        target = Target(
            AdamWBuilder,
            target_path=_ADAMW_TARGET,
            no_decay_params=["bias", "norm"],
            adamw_config={"adamw_lr": 1.0e-5},
        )
        config = _trainer_config(target, OptimizerSwapConfig(enabled=True, min_numel=1))
        built = _build_optimizer(config, self.model)
        leaf = built.optimizers_dict["adamw"]
        for group in leaf.param_groups:
            for param in group["params"]:
                _materialize_adam_state(leaf.optimizer, param)

        state_dict = built.state_dict()

        self.assertEqual(list(state_dict), ["state", "param_groups"])
        self.assertEqual(len(state_dict["state"]), sum(
            len(group["params"]) for group in built.param_groups))
        built.load_state_dict(state_dict)

    @arg_mark(**_MARK)
    def test_fp32_main_params_keeps_the_mixed_precision_alias_in_sync(self):
        """Feature: optimizer.swap with fp32 main parameters.
        Description: enable both the fp32 main-parameter wrap and swap.
        Expectation: swap wraps the leaves below the wrapper and aliases stay in sync.
        """
        target = Target(
            AdamWBuilder,
            target_path=_ADAMW_TARGET,
            no_decay_params=["bias", "norm"],
            adamw_config={"adamw_lr": 1.0e-5},
        )
        config = _trainer_config(
            target,
            OptimizerSwapConfig(enabled=True, min_numel=1, include_master_params=True),
            fp32_main_params=True,
        )

        built = _build_optimizer(config, self.model)

        self.assertIs(built.chained_optimizers, built.optimizer.chained_optimizers)
        self.assertTrue(is_swap_optimizer(built.optimizers_dict["adamw"]))
        self.assertEqual(len(built.optimizer.param_groups), len(built.param_groups))

if __name__ == "__main__":
    unittest.main()
