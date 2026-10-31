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
from unittest import mock

from tests.common.mark_utils import arg_mark
from tests.ut.codegen.helpers import SourceMLP, _replace_mlp

from hyper_parallel.codegen.model_build_overlay import use_codegen_model_build_overlay
from hyper_parallel.codegen.offline_build import create_offline_distributed_setup
from hyper_parallel.codegen.planner_overlay import use_codegen_planner_overlay
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.models._transformers import auto_model
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_invalid_parallel_degrees_name_the_rejected_field(self):
        """Reject zero, negative, boolean, and non-integer degrees without clamping."""
        fields = {
            "accelerator.tp_size": ("tp", 0),
            "accelerator.cp_size": ("cp", -1),
            "accelerator.ep_size": ("ep", False),
            "accelerator.pp_size": ("pp", 1.5),
            "fsdp_config.dp_shard_size": ("dp_shard", 0),
            "fsdp_config.edp_shard_size": ("edp_shard", -2),
        }
        for field, (argument, value) in fields.items():
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field.replace(".", r"\.")):
                    create_offline_distributed_setup(_offline_config(**{argument: value}))


class TestStrategyReplacementMatrix(unittest.TestCase):
    """Conditional strategies and replacements compose through one config."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
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
    """Scoped codegen overlays restore native symbols on every failure path."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_nested_overlays_restore_native_symbols_after_exception(self):
        """Restore AutoModel build and planner factories when generation raises."""
        model_class = auto_model._BaseHyperAutoModelClass  # pylint: disable=protected-access
        original_build = model_class.__dict__["_build_model"]
        original_factory = auto_model.instantiate_infrastructure

        with self.assertRaisesRegex(RuntimeError, "injected emitter failure"):
            with use_codegen_model_build_overlay(CodegenBuildSession()), use_codegen_planner_overlay():
                raise RuntimeError("injected emitter failure")

        self.assertIs(model_class.__dict__["_build_model"], original_build)
        self.assertIs(auto_model.instantiate_infrastructure, original_factory)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="unessential")
    def test_failure_does_not_prevent_a_subsequent_native_factory_call(self):
        """Call the restored native factory after an isolated codegen failure."""
        native_result = (object(), object())
        native_factory = mock.Mock(return_value=native_result)
        with mock.patch.object(auto_model, "instantiate_infrastructure", native_factory):
            with self.assertRaisesRegex(ValueError, "invalid replacement"):
                with use_codegen_planner_overlay():
                    raise ValueError("invalid replacement")
            self.assertEqual(auto_model.instantiate_infrastructure(), native_result)
            native_factory.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
