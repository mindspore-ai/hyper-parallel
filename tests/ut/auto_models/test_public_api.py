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
"""Public API contract snapshots locked before the auto_models migration.

Per the migration plan, the symbols recorded here must keep their names,
``inspect.signature``, dataclass field order/defaults and ``__all__`` across
the directory refactoring. These tests intentionally import the current
(pre-migration) module paths; the same commit that moves a production file
must update the import here, with no signature/field change.
"""
# pylint: disable=wrong-import-position

import dataclasses
import inspect
import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from tests.common.mark_utils import arg_mark


def _field_snapshot(cls):
    """Return ``[(name, default_repr), ...]`` for a dataclass.

    Value defaults are recorded by ``repr``; factory defaults are recorded as
    ``repr`` of one produced value (deterministic for dict/list/placement
    factories used by the sharding configs).
    """
    snapshot = []
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            snapshot.append((f.name, repr(f.default)))
        elif f.default_factory is not dataclasses.MISSING:  # pylint: disable=comparison-with-callable
            snapshot.append((f.name, "<factory> " + repr(f.default_factory())))
        else:
            snapshot.append((f.name, "<required>"))
    return snapshot


class TestShardingConfigContracts(unittest.TestCase):
    """Sharding spec/plan/template dataclass field snapshots."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_module_sharding_spec_fields(self):
        """
        Feature: ModuleShardingSpec public field contract.
        Description: Snapshot the dataclass field order and defaults.
        Expectation: The snapshot matches the supported public contract.
        """
        from hyper_parallel.distributed.recipe_spec import (
            ModuleShardingSpec,
        )

        self.assertEqual(
            _field_snapshot(ModuleShardingSpec),
            [
                ("params", "None"),
                ("in_src", "None"),
                ("in_dst", "None"),
                ("out_src", "None"),
                ("out_dst", "None"),
                ("out_names", "None"),
                ("tp_divide_attrs", "None"),
                ("_tp_local_attr_plan", "None"),
                ("_deferred_bias_params", "()"),
                ("is_boundary", "True"),
                ("region_dispatch", "None"),
                ("inner_target", "None"),
                ("inner_wrapper", "None"),
                ("inner_out_src", "None"),
                ("local_compute_fn", "None"),
                ("_is_terminal", "False"),
                ("_needs_cp_attn", "False"),
                ("_resolved_inner_wrapper", "None"),
                ("_resolved_inner_target", "None"),
                ("_ep_stack", "<factory> {}"),
                ("_ep_size", "0"),
            ],
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sharding_plan_fields(self):
        """
        Feature: ShardingPlan public field contract.
        Description: Snapshot the dataclass field order and defaults.
        Expectation: The snapshot matches the supported public contract.
        """
        from hyper_parallel.distributed.plan import (
            ShardingPlan,
        )

        self.assertEqual(
            _field_snapshot(ShardingPlan),
            [
                ("modules", "<factory> {}"),
                ("sequence_parallel", "True"),
                ("loss_parallel", "False"),
                ("special_handlers", "<factory> {}"),
                ("mesh_dim_names", "()"),
                ("tied_pairs", "<factory> []"),
            ],
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sharding_template_fields(self):
        """
        Feature: ShardingTemplate public field contract.
        Description: Snapshot the dataclass field order and defaults.
        Expectation: The snapshot matches the supported public contract.
        """
        from hyper_parallel.distributed import (
            ShardingTemplate,
        )

        self.assertEqual(
            _field_snapshot(ShardingTemplate),
            [
                ("colwise_placement", "<factory> Shard(dim=0)"),
                ("rowwise_placement", "<factory> Shard(dim=1)"),
                ("norm_placement", "<factory> Replicate()"),
                ("moe_expert_placement", "<factory> Shard(dim=0)"),
                ("sp_in_src", "<factory> {}"),
                ("sp_in_dst", "<factory> {}"),
                ("sp_out_src", "None"),
                ("sp_out_dst", "None"),
                ("nosp_in_src", "<factory> {}"),
                ("nosp_in_dst", "<factory> {}"),
                ("nosp_out_src", "None"),
                ("nosp_out_dst", "None"),
                ("region_dispatch", "None"),
                ("needs_cp_attn", "False"),
            ],
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sharding_planner_signature(self):
        """
        Feature: ShardingPlanner constructor contract.
        Description: Snapshot the public constructor signature.
        Expectation: The signature matches the supported public contract.
        """
        from hyper_parallel.distributed import (
            ShardingPlanner,
        )

        self.assertEqual(
            str(inspect.signature(ShardingPlanner.__init__)),
            "(self, plan_overrides: Optional[Dict[str, "
            "hyper_parallel.distributed.recipe_spec."
            "ModuleShardingSpec]] = None, *, derive: bool = True, "
            "allow_uncovered_params: bool = False) -> None",
        )


class TestTrainerConfigContracts(unittest.TestCase):
    """Trainer-facing config dataclass field snapshots."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_accelerator_config_fields(self):
        """
        Feature: AcceleratorConfig public field contract.
        Description: Snapshot the dataclass field order and defaults.
        Expectation: The snapshot includes the supported pipeline fields.
        """
        from hyper_parallel.trainer.config import AcceleratorConfig

        self.assertEqual(
            _field_snapshot(AcceleratorConfig),
            [
                ("tp_size", "1"),
                ("cp_size", "1"),
                ("ep_size", "1"),
                ("pp_size", "1"),
                ("pp_micro_batch_num", "1"),
                ("pp_schedule", "None"),
                ("pp_vpp", "1"),
                ("pp_layer_split", "None"),
                ("sequence_parallel", "False"),
                ("loss_parallel", "False"),
            ],
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_compile_config_fields(self):
        """
        Feature: CompileConfig public field contract.
        Description: Snapshot the dataclass field order and defaults.
        Expectation: The snapshot matches the supported public contract.
        """
        from hyper_parallel.trainer.config import CompileConfig

        self.assertEqual(
            _field_snapshot(CompileConfig),
            [
                ("enabled", "False"),
                ("mode", "'default'"),
                ("fullgraph", "False"),
                ("dynamic", "False"),
                ("backend", "None"),
                ("options", "None"),
                ("dynamo_cache_size_limit", "256"),
            ],
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_low_precision_config_fields(self):
        """
        Feature: LowPrecisionConfig public field contract.
        Description: Snapshot the dataclass field order and defaults.
        Expectation: The snapshot matches the supported public contract.
        """
        from hyper_parallel.components.quantization.config import (
            LowPrecisionConfig,
        )

        self.assertEqual(
            _field_snapshot(LowPrecisionConfig),
            [
                ("enabled", "False"),
                ("format", "'mxfp8_e4m3'"),
                ("scaling", "'mx_block'"),
            ],
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_trainer_config_module_all(self):
        """
        Feature: Trainer configuration package exports.
        Description: Snapshot the symbols declared by trainer.config.__all__.
        Expectation: The snapshot includes the supported dry-run configuration.
        """
        from hyper_parallel.trainer import config as trainer_config

        self.assertEqual(
            trainer_config.__all__,
            [
                "AcceleratorConfig",
                "ActivationCheckpointConfig",
                "CompileConfig",
                "DataLoaderConfig",
                "DatasetConfig",
                "DebugConfig",
                "DryRunConfig",
                "FSDP2Config",
                "MixedPrecisionConfig",
                "OptimizerConfig",
                "ProfilingConfig",
                "Target",
                "TrainerConfig",
                "TrainingConfig",
                "WandbConfig",
                "save_configs",
                "CheckpointingConfig",
            ],
        )


class TestOptimizerContracts(unittest.TestCase):
    """Optimizer wrapper constructor signature snapshots."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_optimizer_wrapper_signatures(self):
        """
        Feature: Optimizer wrapper constructor contracts.
        Description: Snapshot the public optimizer constructor signatures.
        Expectation: All signatures match their supported public contracts.
        """
        from hyper_parallel.components import optim

        self.assertEqual(
            str(inspect.signature(optim.AdamW.__init__)),
            "(self, adamw_config: dict, model: torch.nn.modules.module.Module, "
            "no_decay_params: Optional[List[str]] = None) -> None",
        )
        self.assertEqual(
            str(inspect.signature(optim.Muon.__init__)),
            "(self, muon_config: dict, adamw_config: dict, "
            "model: torch.nn.modules.module.Module, "
            "extra_adamw_name_keywords: Optional[List[str]] = None, "
            "no_decay_params: Optional[List[str]] = None) -> None",
        )
        self.assertEqual(
            str(inspect.signature(optim.MixedPrecisionOptimizer.__init__)),
            "(self, optimizer: hyper_parallel.core.optimizer.optimizer.ChainedOptimizer, "
            "model: torch.nn.modules.module.Module) -> None",
        )
        self.assertEqual(
            str(inspect.signature(optim.Float16OptimizerWithFloat16Params.__init__)),
            "(self, optimizer: hyper_parallel.core.optimizer.optimizer.ChainedOptimizer, "
            "model: torch.nn.modules.module.Module) -> None",
        )
        self.assertEqual(
            str(inspect.signature(optim.MultiLRScheduler.__init__)),
            "(self, optimizer: Any, lr_decay_style: str, train_iters: int, "
            "lr_config: dict) -> None",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_optimizer_module_all(self):
        """
        Feature: Optimizer package exports.
        Description: Snapshot the optimizer modules' __all__ declarations.
        Expectation: Both snapshots match their supported public contracts.
        """
        from hyper_parallel.components import optim
        from hyper_parallel.components.optim import mixed_precision_optimizer

        self.assertEqual(
            optim.__all__,
            [
                "AdamW",
                "Float16OptimizerWithFloat16Params",
                "MixedPrecisionOptimizer",
                "Muon",
                "MultiLRScheduler",
                "get_adamw_param_groups",
                "get_parameter_names",
                "split_muon_adamw_params",
            ],
        )
        self.assertEqual(
            mixed_precision_optimizer.__all__,
            [
                "FP32_MAIN_PARAM_STATE_KEY",
                "Float16OptimizerWithFloat16Params",
                "MIXED_PRECISION_OPTIMIZER_STATE_KEY",
                "MixedPrecisionOptimizer",
            ],
        )


if __name__ == "__main__":
    unittest.main()
