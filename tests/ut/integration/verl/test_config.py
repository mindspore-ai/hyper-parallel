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
"""Validate recipe merging and VeRL-owned input configuration."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from tests.ut.integration.verl._contracts import ADAPTERS

ENGINE_ROUTER_REPLAY_CONFIG = SimpleNamespace
HYPER_PARALLEL_ACTOR_CONFIG = ADAPTERS.config.HyperParallelActorConfig
HYPER_PARALLEL_CHECKPOINT_CONFIG = ADAPTERS.config.HyperParallelCheckpointConfig
HYPER_PARALLEL_CRITIC_CONFIG = ADAPTERS.config.HyperParallelCriticConfig
HYPER_PARALLEL_ENGINE_CONFIG = ADAPTERS.config.HyperParallelEngineConfig
validate_and_resolve_hyper_parallel_config = ADAPTERS.config.validate_and_resolve_hyper_parallel_config


_RECIPE = Path(__file__).with_name("native_cpu.yaml")


class TestHyperParallelEngineConfig(unittest.TestCase):
    """Exercise typed native configuration rather than substituting config mocks."""

    def test_resolves_inline_trainer_config_without_recipe_path(self) -> None:
        """Allow the inline config to supply the model and optimizer builders."""
        inline = OmegaConf.to_container(OmegaConf.load(_RECIPE))
        inline_config = HYPER_PARALLEL_ENGINE_CONFIG(trainer_config=inline).trainer_config
        recipe_config = HYPER_PARALLEL_ENGINE_CONFIG(recipe_path=str(_RECIPE)).trainer_config
        self.assertEqual(inline_config.to_dict(), recipe_config.to_dict())

    def test_overrides_and_input_update_preserve_loader_components(self) -> None:
        """Model-path adaptation must preserve target-selected data components."""
        engine = HYPER_PARALLEL_ENGINE_CONFIG(
            recipe_path=str(_RECIPE), trainer_config={"training": {"train_iters": 17}},
        )
        config = engine.trainer_config
        collator = config.dataloader.collate_fn
        updated = validate_and_resolve_hyper_parallel_config(engine, SimpleNamespace(path="model-from-verl"))
        self.assertIs(updated, config, f"updated config identity={id(updated)}, expected={id(config)}")
        self.assertIs(config.dataloader.collate_fn, collator, "input adaptation replaced the collator")
        self.assertEqual(config.training.train_iters, 17, f"train_iters={config.training.train_iters}, expected=17")
        self.assertEqual(config.model.pretrained_model_name_or_path, "model-from-verl", "VeRL path was not applied")
        self.assertTrue(config.dataloader.get_batch.reset_position_ids, "packed samples must reset positions")

    def test_verl_model_overrides_merge_with_native_settings(self) -> None:
        """Support flat and nested VeRL overrides without replacing native loader settings."""
        overrides = {"num_hidden_layers": 3, "text_config": {"hidden_size": 16}, "torch_dtype": "float16"}
        for verl_overrides in (
            overrides, {"model_config": overrides},
            {"model_config": {**overrides, "num_hidden_layers": 5}, **overrides},
        ):
            with self.subTest(verl_overrides=verl_overrides):
                engine = HYPER_PARALLEL_ENGINE_CONFIG(
                    recipe_path=str(_RECIPE),
                    trainer_config={"model": {"config_overrides": {
                        "num_hidden_layers": 8, "text_config": {"vocab_size": 32},
                    }}},
                )
                model = SimpleNamespace(path="model-from-verl", override_config=verl_overrides)
                config = validate_and_resolve_hyper_parallel_config(engine, model)
                self.assertEqual(config.model.config_overrides, {
                    "num_hidden_layers": 3, "text_config": {"vocab_size": 32, "hidden_size": 16},
                })
                self.assertEqual(config.model.torch_dtype, "float32")

    def test_rejects_invalid_engine_settings_at_construction(self) -> None:
        """Check settings that are fixed before the worker binds the engine."""
        cases = (
            ({"strategy": "fsdp2"}, ValueError, "strategy=hyper_parallel"),
            ({"dtype": "float64"}, ValueError, "dtype"),
            ({"router_replay": ENGINE_ROUTER_REPLAY_CONFIG(mode="R2")}, NotImplementedError, "router replay"),
        )
        for options, error, message in cases:
            with self.subTest(options=options):
                with self.assertRaisesRegex(error, message):
                    HYPER_PARALLEL_ENGINE_CONFIG(recipe_path=str(_RECIPE), **options)

    def test_rejects_unsupported_engine_settings_after_binding(self) -> None:
        """Check model settings after VeRL copies them onto the engine."""
        for name, value in (("use_fused_kernels", True), ("use_remove_padding", False)):
            with self.subTest(name=name):
                engine = HYPER_PARALLEL_ENGINE_CONFIG(recipe_path=str(_RECIPE))
                setattr(engine, name, value)
                with self.assertRaisesRegex(NotImplementedError, name):
                    validate_and_resolve_hyper_parallel_config(engine, SimpleNamespace(path="model-from-verl"))

    def test_disables_native_diagnostics_requested_by_recipe(self) -> None:
        """Disable both diagnostics even when the native configuration enables them."""
        engine = HYPER_PARALLEL_ENGINE_CONFIG(
            recipe_path=str(_RECIPE),
            trainer_config={"model_integration": {"mode": "runtime"}, "debug": {"check_fsdp_runtime": True}},
        )
        config = validate_and_resolve_hyper_parallel_config(engine, SimpleNamespace(path="model-from-verl"))
        self.assertEqual(config.model_integration.mode, "off")
        self.assertFalse(config.debug.check_fsdp_runtime)

    def test_reuses_typed_config_and_replaces_changed_builder(self) -> None:
        """Keep a typed recipe intact and discard arguments of a replaced builder."""
        engine = HYPER_PARALLEL_ENGINE_CONFIG(recipe_path=str(_RECIPE))
        typed = HYPER_PARALLEL_ENGINE_CONFIG(trainer_config=engine.trainer_config)
        self.assertEqual(typed.trainer_config.to_dict(), engine.trainer_config.to_dict())

        engine = HYPER_PARALLEL_ENGINE_CONFIG(
            recipe_path=str(_RECIPE),
            trainer_config={"optimizer": {
                "_target_": "hyper_parallel.components.optim.AdamW",
                "adamw_config": {"adamw_lr": 1e-6},
            }},
        )
        optimizer = engine.trainer_config.to_dict()["optimizer"]
        self.assertNotIn("muon_config", optimizer)
        self.assertEqual(optimizer["adamw_config"]["adamw_lr"], 1e-6)

    def test_verl_step_budget_overrides_recipe_for_each_role(self) -> None:
        """Apply a valid VeRL budget to both step fields for Actor and Reference."""
        cases = ((False, 13), (True, 13), (True, -1))
        for forward_only, total_steps in cases:
            with self.subTest(forward_only=forward_only, total_steps=total_steps):
                engine = HYPER_PARALLEL_ENGINE_CONFIG(
                    recipe_path=str(_RECIPE),
                    dtype="float16",
                    forward_only=forward_only,
                    trainer_config={"training": {"train_iters": 3, "lr_scheduler_iters": 7}},
                )
                config = validate_and_resolve_hyper_parallel_config(
                    engine, SimpleNamespace(path="model-from-verl"),
                    optimizer_config=SimpleNamespace(total_training_steps=total_steps),
                )
                expected = (13, 13) if total_steps > 0 else (3, 7)
                self.assertEqual((config.training.train_iters, config.training.lr_scheduler_iters), expected)
                self.assertEqual(config.model_init_dtype, "float16")


class TestHyperParallelRoleConfig(unittest.TestCase):
    """Validate the policy and checkpoint contracts before worker creation."""

    def test_actor_rejects_unsupported_options(self) -> None:
        """Keep unsupported features disabled at actor construction."""
        options = {"rollout_n": 1, "use_dynamic_bsz": True}
        self.assertEqual(HYPER_PARALLEL_ACTOR_CONFIG(**options).strategy, "hyper_parallel")
        with self.assertRaisesRegex(ValueError, "strategy=hyper_parallel"):
            HYPER_PARALLEL_ACTOR_CONFIG(strategy="fsdp2", **options)
        for name in ("calculate_sum_pi_squared", "use_fused_kernels", "use_prefix_grouper", "use_torch_compile"):
            with self.subTest(name=name), self.assertRaisesRegex(NotImplementedError, name):
                HYPER_PARALLEL_ACTOR_CONFIG(**options, **{name: True})

    def test_rejects_unsupported_checkpoint_options(self) -> None:
        """Reject partial state and asynchronous saves."""
        for options, message in (
            ({"save_contents": ["model"]}, "save_contents"),
            ({"load_contents": ["model"]}, "load_contents"),
            ({"async_save": True}, "async_save"),
        ):
            with self.subTest(options=options), self.assertRaisesRegex(NotImplementedError, message):
                HYPER_PARALLEL_CHECKPOINT_CONFIG(**options)

    def test_rejects_critic(self) -> None:
        """Fail before attempting to create a value model."""
        with self.assertRaisesRegex(NotImplementedError, "does not support VeRL critic"):
            HYPER_PARALLEL_CRITIC_CONFIG()
