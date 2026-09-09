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
"""Unit tests for graph-mode integration helpers."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch

from hyper_parallel.compile.pass_config import build_pass_config_from_trainer_config
from hyper_parallel.compile.trainer import GraphTrainer
from hyper_parallel.distributed.compile import _resolve_compile_config
from hyper_parallel.models._transformers.model_builder import instantiate_infrastructure
from hyper_parallel.models.build_options import CompileConfig
from hyper_parallel.trainer.config import (
    AcceleratorConfig,
    OptimizerConfig,
    Target,
    TrainerConfig,
)


def _build_model(*, distributed_setup, width):
    """Return model construction inputs for config assertions."""
    return distributed_setup, width


def _build_optimizer(*, params):
    """Minimal optimizer target unused by these tests."""
    return params


class TestGraphModeIntegration(unittest.TestCase):
    """Test graph-mode selection without the old target-wrapping bridge."""

    def test_graphtrainer_infers_pass_config_from_trainer_config(self):
        """GraphTrainer should infer pass config from trainer topology when omitted."""
        trainer_config = SimpleNamespace(
            accelerator=SimpleNamespace(
                tp_size=2,
                sequence_parallel=True,
                loss_parallel=False,
            ),
            fsdp_config=SimpleNamespace(dp_shard_size=4, edp_shard_size=1),
        )

        trainer = GraphTrainer(
            model=torch.nn.Linear(2, 2),
            train_fn=lambda model, x, y: ((model(x) - y) ** 2).mean(),
            trainer_config=trainer_config,
            device=torch.device("cpu"),
        )

        self.assertTrue(trainer.pass_config.fsdp_enabled)
        self.assertEqual(trainer.pass_config.fsdp_degree, 4)
        self.assertEqual(trainer.pass_config.tp_size, 2)
        self.assertTrue(trainer.pass_config.sequence_parallel)

    def test_graphtrainer_enabled_skips_eager_fsdp2_instantiation(self):
        """Graph-mode selection should keep mesh planning but skip eager FSDP2."""
        distributed_setup = SimpleNamespace(
            mesh_context="mesh",
            strategy_config=object(),
            fp32_main_params=False,
        )

        with patch(
            "hyper_parallel.models._transformers.model_builder._instantiate_fsdp2"
        ) as mock_instantiate:
            sharding_planner, fsdp_manager = instantiate_infrastructure(
                distributed_setup=distributed_setup,
                compile_config=CompileConfig(enabled=True, graphtrainer_enabled=True),
            )

        self.assertIsNotNone(sharding_planner)
        self.assertIsNone(fsdp_manager)
        mock_instantiate.assert_not_called()

    def test_non_graph_mode_preserves_eager_fsdp2_instantiation(self):
        """Non-graph builds should still instantiate eager FSDP2 as before."""
        distributed_setup = SimpleNamespace(
            mesh_context="mesh",
            strategy_config=object(),
            fp32_main_params=False,
        )
        fsdp_manager = SimpleNamespace(config=SimpleNamespace())

        with patch(
            "hyper_parallel.models._transformers.model_builder._instantiate_fsdp2",
            return_value=fsdp_manager,
        ) as mock_instantiate:
            _, built_manager = instantiate_infrastructure(
                distributed_setup=distributed_setup,
                compile_config=CompileConfig(enabled=True, graphtrainer_enabled=False),
            )

        self.assertIs(built_manager, fsdp_manager)
        mock_instantiate.assert_called_once_with(
            config=distributed_setup.strategy_config,
            mesh_context=distributed_setup.mesh_context,
            fp32_main_params=False,
        )

    def test_graphtrainer_enabled_skips_eager_decoder_compile(self):
        """Graph-mode compile selection should not trigger eager layer compile."""
        compile_config, compile_for_execution = _resolve_compile_config(
            CompileConfig(
                enabled=True,
                graphtrainer_enabled=True,
                mode="reduce-overhead",
            ),
            validate_placement=False,
            fsdp2_manager=None,
        )

        self.assertIsNotNone(compile_config)
        self.assertTrue(compile_config.selects_graph_trainer())
        self.assertFalse(
            compile_for_execution,
            f"graph mode should skip eager decoder compile: {compile_config!r}",
        )

    def test_graph_mode_allows_pipeline_parallel_topology(self):
        """Graph trainer selection should bypass the eager compile PP guard."""
        model_target = Target(
            _target_=_build_model,
            target_path="tests.ut.compile.test_graphtrainer_config._build_model",
            width=16,
        )
        optimizer_target = Target(
            _target_=_build_optimizer,
            target_path="tests.ut.compile.test_graphtrainer_config._build_optimizer",
        )

        config = TrainerConfig(
            model=model_target,
            optimizer=OptimizerConfig(target=optimizer_target),
            accelerator=AcceleratorConfig(pp_size=2),
            compile=CompileConfig(enabled=True, graphtrainer_enabled=True),
        )

        self.assertTrue(config.compile.selects_graph_trainer())


    def test_build_pass_config_projects_parallel_topology(self):
        """Trainer topology and inferred FSDP intent are copied to PassConfig."""
        config = SimpleNamespace(
            accelerator=SimpleNamespace(
                tp_size=2,
                sequence_parallel=True,
                loss_parallel=True,
            ),
            fsdp_config=SimpleNamespace(dp_shard_size=4, edp_shard_size=1),
        )

        pass_config = build_pass_config_from_trainer_config(
            config,
            enable_overlap=True,
        )

        self.assertTrue(
            pass_config.fsdp_enabled,
            f"FSDP should be inferred as enabled: got={pass_config!r}",
        )
        self.assertEqual(
            pass_config.fsdp_degree,
            4,
            f"FSDP degree mismatch: expected=4, got={pass_config.fsdp_degree}",
        )
        self.assertEqual(
            pass_config.tp_size, 2, f"TP size mismatch: expected=2, got={pass_config.tp_size}"
        )
        self.assertTrue(
            pass_config.sequence_parallel,
            f"sequence parallel flag mismatch: expected=True, got={pass_config.sequence_parallel}",
        )
        self.assertTrue(
            pass_config.loss_parallel,
            f"loss parallel flag mismatch: expected=True, got={pass_config.loss_parallel}",
        )
        self.assertTrue(
            pass_config.enable_overlap,
            f"overlap flag mismatch: expected=True, got={pass_config.enable_overlap}",
        )


if __name__ == "__main__":
    unittest.main()
