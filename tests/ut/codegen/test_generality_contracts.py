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
"""Generality contracts for codegen topology, replacements, and isolation."""

from types import SimpleNamespace
import unittest

from tests.ut.codegen.helpers import SourceMLP, _replace_mlp

from hyper_parallel.codegen.build import create_offline_distributed_setup
from hyper_parallel.models._transformers import auto_model
from hyper_parallel.models.build_capture import capture_model_build_plan, is_model_build_capture_active
from hyper_parallel.trainer.config import (
    PlanOverride,
    ReplaceAction,
    Target,
    entries_to_module_replacements,
    entries_to_plan_overrides,
)


def _offline_config(*, tp: int = 1, cp: int = 1, ep: int = 1, pp: int = 1,
                    dp_shard: int = 1, edp_shard: int = 1) -> SimpleNamespace:
    """Build the minimum typed-config shape consumed by offline planning."""
    return SimpleNamespace(
        accelerator=SimpleNamespace(
            tp_size=tp,
            cp_size=cp,
            ep_size=ep,
            pp_size=pp,
            sequence_parallel=tp > 1,
            loss_parallel=False,
        ),
        fsdp_config=SimpleNamespace(
            dp_shard_size=dp_shard,
            edp_shard_size=edp_shard,
        ),
        training=SimpleNamespace(low_precision=None),
        optimizer=SimpleNamespace(fp32_main_params=False),
    )


class TestOfflineTopologyMatrix(unittest.TestCase):
    """Codegen accepts every supported combination without creating devices."""

    def test_legal_parallel_combinations_preserve_requested_degrees(self):
        """Build representative TP/CP/EP/PP/FSDP combinations as pure metadata."""
        cases = (
            (1, 1, 1, 1, 1, 1),
            (2, 1, 1, 1, 1, 1),
            (1, 2, 1, 1, 1, 1),
            (1, 1, 4, 1, 1, 1),
            (2, 2, 1, 1, 1, 1),
            (2, 1, 4, 1, 1, 1),
            (1, 2, 4, 1, 1, 1),
            (2, 2, 4, 1, 1, 1),
            (2, 1, 1, 2, 1, 1),
            (2, 1, 4, 1, 2, 4),
            (4, 2, 8, 2, 2, 4),
        )
        for tp, cp, ep, pp, dp_shard, edp_shard in cases:
            with self.subTest(tp=tp, cp=cp, ep=ep, pp=pp,
                              dp_shard=dp_shard, edp_shard=edp_shard):
                setup = create_offline_distributed_setup(_offline_config(
                    tp=tp,
                    cp=cp,
                    ep=ep,
                    pp=pp,
                    dp_shard=dp_shard,
                    edp_shard=edp_shard,
                ))
                mesh = setup.mesh_context
                self.assertEqual((mesh.tp_size, mesh.cp_size, mesh.ep_size, mesh.pp_size),
                                 (tp, cp, ep, pp))
                self.assertEqual(mesh.dp_shard_size, dp_shard)
                self.assertEqual(mesh.edp_shard_size, edp_shard)
                self.assertEqual(mesh.device_mesh.mesh_dim_names, ("dp", "cp", "tp"))

    def test_non_positive_parallel_degrees_match_trainer_compatibility(self):
        """Normalize non-positive degrees exactly as the native Trainer does."""
        fields = {
            "accelerator.tp_size": ("tp", 0),
            "accelerator.cp_size": ("cp", -1),
            "accelerator.ep_size": ("ep", False),
            "accelerator.pp_size": ("pp", 0),
            "fsdp_config.dp_shard_size": ("dp_shard", 0),
            "fsdp_config.edp_shard_size": ("edp_shard", -2),
        }
        for field, (argument, value) in fields.items():
            with self.subTest(field=field, value=value):
                setup = create_offline_distributed_setup(
                    _offline_config(**{argument: value})
                )
                mesh = setup.mesh_context
                normalized = {
                    "tp": mesh.tp_size,
                    "cp": mesh.cp_size,
                    "ep": mesh.ep_size,
                    "pp": mesh.pp_size,
                    "dp_shard": mesh.dp_shard_size,
                    "edp_shard": mesh.edp_shard_size,
                }[argument]
                self.assertEqual(normalized, 1)
                self.assertEqual(mesh.device_mesh.shape, (1, 1, 1))


class TestStrategyReplacementMatrix(unittest.TestCase):
    """Conditional strategies and replacements compose through one config."""

    def test_cp_ep_and_module_replacement_can_be_declared_together(self):
        """Activate CP and EP sharding entries while compiling an independent replacement."""
        replacement = Target(
            _replace_mlp,
            target_path="tests.ut.codegen.helpers._replace_mlp",
        )
        entries = [
            PlanOverride(
                match="model.layers.*.self_attn",
                when="cp",
                params={"q_proj.weight": {"cp": "replicate"}},
            ),
            PlanOverride(
                match="model.layers.*.mlp",
                when="ep",
                params={"gate.weight": {"ep": "replicate"}},
            ),
            PlanOverride(
                match=["model.layers.0.mlp", "model.layers.1.mlp"],
                replace=ReplaceAction(
                    kind="module",
                    implementation=replacement,
                    source_type="tests.ut.codegen.helpers.SourceMLP",
                ),
            ),
        ]

        overrides = entries_to_plan_overrides(entries, cp_size=2, ep_size=4)
        replacements = entries_to_module_replacements(entries)

        self.assertEqual(
            set(overrides),
            {"model.layers.*.self_attn", "model.layers.*.mlp"},
        )
        self.assertEqual(len(replacements), 1)
        self.assertEqual(
            replacements[0].match,
            ("model.layers.0.mlp", "model.layers.1.mlp"),
        )
        self.assertIs(replacements[0].module_type, SourceMLP)
        self.assertEqual(
            replacements[0].replacement_target,
            "tests.ut.codegen.helpers._replace_mlp",
        )

    def test_inactive_conditional_strategies_do_not_disable_replacement(self):
        """Keep replacement declarations when CP/EP strategy conditions are inactive."""
        replacement = Target(
            _replace_mlp,
            target_path="tests.ut.codegen.helpers._replace_mlp",
        )
        entries = [
            PlanOverride(match="attention", when="cp", params={"weight": {"cp": "replicate"}}),
            PlanOverride(match="moe", when="ep", params={"weight": {"ep": "replicate"}}),
            PlanOverride(
                match="mlp",
                replace=ReplaceAction(
                    kind="module",
                    implementation=replacement,
                    source_type="tests.ut.codegen.helpers.SourceMLP",
                ),
            ),
        ]

        self.assertEqual(entries_to_plan_overrides(entries, cp_size=1, ep_size=1), {})
        self.assertEqual(len(entries_to_module_replacements(entries)), 1)


class TestGenerationFailureIsolation(unittest.TestCase):
    """Codegen keeps native entry points intact during capture."""

    def test_capture_failure_leaves_native_symbols_intact(self):
        """Capture leaves the native AutoModel build and infrastructure factory intact."""
        model_class = auto_model._BaseHyperAutoModelClass  # pylint: disable=protected-access
        original_build = model_class.__dict__["_build_model"]
        original_factory = auto_model.instantiate_infrastructure

        with self.assertRaisesRegex(RuntimeError, "injected build failure"):
            with capture_model_build_plan():
                self.assertIs(model_class.__dict__["_build_model"], original_build)
                self.assertIs(auto_model.instantiate_infrastructure, original_factory)
                raise RuntimeError("injected build failure")

        self.assertFalse(is_model_build_capture_active())
        self.assertIs(model_class.__dict__["_build_model"], original_build)
        self.assertIs(auto_model.instantiate_infrastructure, original_factory)


if __name__ == "__main__":
    unittest.main()
