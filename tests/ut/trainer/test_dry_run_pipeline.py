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
"""Unit tests for the primary pipeline dry-run workflow."""
# pylint: disable=protected-access

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.platform.torch.dry_run import DryRunRuntime
from hyper_parallel.trainer.config import (
    DryRunConfig,
    OptimizerConfig,
    Target,
    TrainerConfig,
    TrainingConfig,
)
from hyper_parallel.trainer.dry_run import HyperModelsDryRunRunner, _DryRunTrainingBatch
from hyper_parallel.trainer.dry_run_pipeline import DryRunBoundaryLeaf, DryRunPipelineChunk
from hyper_parallel.trainer.dry_run_pipeline_assembly import (
    _DryRunParallelContext,
    _PreparedPipelineChunk,
    _DryRunStageShardingResult,
    _resolve_boundary_metadata,
    build_pipeline_chunks,
)


def _target(**kwargs: Any) -> dict[str, Any]:
    """Provide a callable placeholder for validation-only configurations."""
    return kwargs


def _pipeline_config(
        pp_vpp: int = 1,
        global_batch_size: int = 4,
        micro_batch_num: int = 4,
) -> TrainerConfig:
    """Build a minimal two-rank pipeline dry-run configuration."""
    config = TrainerConfig(
        model=Target(_target, target_path="tests.model"),
        optimizer=OptimizerConfig(Target(_target, target_path="tests.optimizer")),
        training=TrainingConfig(
            global_batch_size=global_batch_size,
            micro_batch_size=1,
            backend="gloo",
        ),
        dry_run=DryRunConfig(
            sequence_length=8,
            pipeline_stage_builder=Target(_target, target_path="tests.pipeline"),
        ),
    )
    config.accelerator.pp_size = 2
    config.accelerator.pp_vpp = pp_vpp
    config.accelerator.pp_micro_batch_num = micro_batch_num
    return config


class _StageModule(torch.nn.Module):
    """Tiny pipeline chunk used for contract tests."""

    def __init__(self) -> None:
        """Create one trainable projection."""
        super().__init__()
        self.projection = torch.nn.Linear(4, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project one hidden-state tensor."""
        return self.projection(inputs)


class TestDryRunPipelineValidation(unittest.TestCase):
    """Tests for pipeline topology and builder contracts."""

    def test_pipeline_validation_accepts_supported_and_rejects_invalid_topologies(self):
        """Accept supported topology while rejecting invalid batches and parallel mixes."""
        for pp_vpp in (1, 2):
            with self.subTest(pp_vpp=pp_vpp):
                runner = HyperModelsDryRunRunner(
                    _pipeline_config(pp_vpp=pp_vpp),
                    DryRunRuntime(rank=0, world_size=2, local_rank=0),
                )
                runner._validate_config()

        config = _pipeline_config(global_batch_size=3, micro_batch_num=2)
        with self.assertRaisesRegex(ValueError, "DP-local batch size must be divisible"):
            HyperModelsDryRunRunner(config, DryRunRuntime(0, 2, 0))._validate_config()

        config = _pipeline_config()
        config.accelerator.ep_size = 2
        with self.assertRaisesRegex(NotImplementedError, "Pipeline Dry-run supports"):
            HyperModelsDryRunRunner(config, DryRunRuntime(0, 4, 0))._validate_config()

    def test_pipeline_builder_validates_chunks_for_physical_and_virtual_stages(self):
        """Require one correctly indexed, shape-compatible chunk per local stage."""
        for pp_vpp, stage_indices in ((1, (0,)), (2, (0, 2))):
            with self.subTest(pp_vpp=pp_vpp):
                config = _pipeline_config(pp_vpp=pp_vpp)
                chunks = tuple(
                    DryRunPipelineChunk(
                        _StageModule(),
                        layer_start=index,
                        layer_end=index + 1,
                        hidden_size=4,
                        stage_index=stage_index,
                    )
                    for index, stage_index in enumerate(stage_indices)
                )
                config.dry_run.pipeline_stage_builder = SimpleNamespace(
                    build=MagicMock(return_value=chunks)
                )
                context = _DryRunParallelContext(
                    setup=SimpleNamespace(mesh_context=SimpleNamespace(cp_size=1)),
                    root_mesh=MagicMock(),
                    pp_mesh=SimpleNamespace(get_local_rank=lambda: 0),
                )

                result = build_pipeline_chunks(
                    config,
                    MagicMock(),
                    MagicMock(),
                    4,
                    context,
                    torch.float32,
                    (1, 8),
                )

                self.assertEqual(tuple(chunk.stage_index for chunk in result), stage_indices)
                config.dry_run.pipeline_stage_builder.build.assert_called_once()

class TestDryRunPipelineBoundaries(unittest.TestCase):
    """Tests for pipeline layer splitting and boundary metadata."""

    def test_boundary_metadata_resolves_tensor_and_dtensor_wires(self):
        """Resolve planner placements into local Tensor and DTensor metadata."""
        leaves = (
            DryRunBoundaryLeaf((2, 8, 4), torch.float32, True, "block", "hidden", "tensor"),
            DryRunBoundaryLeaf((2, 8, 4), torch.float32, True, "block", "hidden", "dtensor"),
        )
        module_spec = SimpleNamespace(in_dst={"hidden": {"tp": "shard(1)"}})
        plan = SimpleNamespace(mesh_dim_names=("tp",), modules={"block": module_spec})
        sharding = _DryRunStageShardingResult(MagicMock(), None, plan)
        mesh = SimpleNamespace(tp_size=2, cp_size=1, device_mesh=MagicMock())
        fake_layout = MagicMock(name="layout")

        with (
            patch(
                "hyper_parallel.trainer.dry_run_pipeline_assembly.resolve_placements",
                return_value=(MagicMock(),),
            ),
            patch(
                "hyper_parallel.trainer.dry_run_pipeline_assembly.compute_local_shape_and_global_offset",
                return_value=(2, 4, 4),
            ),
            patch(
                "hyper_parallel.trainer.dry_run_pipeline_assembly._active_plan_mesh",
                return_value=MagicMock(),
            ),
            patch(
                "hyper_parallel.trainer.dry_run_pipeline_assembly._build_layout",
                return_value=fake_layout,
            ),
        ):
            metadata = _resolve_boundary_metadata(leaves, sharding, mesh, "in_dst")

        self.assertEqual(metadata[0], [(2, 4, 4), torch.float32, True])
        self.assertEqual(metadata[1], [(2, 4, 4), torch.float32, fake_layout, True])


class TestDryRunPipelineExecution(unittest.TestCase):
    """Tests for one rank-local mocked-transport pipeline step."""

    def test_pipeline_step_runs_and_cleans_temporary_state(self):
        """Execute forward/backward and emit pipeline metadata without real P2P."""
        config = _pipeline_config()
        runner = HyperModelsDryRunRunner(config, DryRunRuntime(0, 2, 0))
        runner._simulation_device = "cpu"
        runner._target_device = "cuda"
        model = _StageModule()
        model.config = SimpleNamespace()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=False)
        base = SimpleNamespace(
            model=model,
            optimizer=optimizer,
            mesh=SimpleNamespace(
                dp_size=1,
                dp_replicate_size=1,
                dp_shard_size=1,
                tp_size=1,
                cp_size=1,
                ep_size=1,
                pp_size=2,
                loss_parallel=False,
            ),
            hsdp_model_parts=[],
            device=torch.device("cpu"),
            model_fwd_context=nullcontext(),
            model_bwd_context=nullcontext(),
        )
        batch = _DryRunTrainingBatch(
            model_inputs={"input_ids": torch.ones(4, 4)},
            loss_inputs={"labels": torch.ones(4, 4, dtype=torch.long)},
            token_counts={"foundation_tokens": 12},
            valid_token_count=12,
            target_tokens_per_rank=(12,),
        )
        chunk = DryRunPipelineChunk(model, 0, 1, 4, stage_index=0)
        prepared = _PreparedPipelineChunk(chunk, [], [])
        stage = MagicMock()
        stage.is_first_stage = True
        stage.is_last_stage = True

        class _Schedule:
            """Run the tiny model while standing in for mocked P2P scheduling."""

            @staticmethod
            def run(input_ids: torch.Tensor, **kwargs: Any) -> list[torch.Tensor]:
                """Execute the local model while transport remains mocked."""
                del kwargs
                loss = model(input_ids).square().mean()
                loss.backward()
                return [loss]

        dependencies = MagicMock()
        dependencies.fake_step_context.return_value = nullcontext()
        dependencies.logical_scope.return_value = nullcontext()
        profile = MagicMock()
        profile.metadata.return_value = {}
        pp_mesh = MagicMock()

        with (
            patch("hyper_parallel.trainer.dry_run.DryRunPipelineStage", return_value=stage),
            patch("hyper_parallel.trainer.dry_run.build_pipeline_schedule", return_value=_Schedule()),
            patch("hyper_parallel.trainer.dry_run.hsdp_sync_stream"),
        ):
            report = runner._execute_pipeline_step(
                base,
                batch,
                profile,
                dependencies,
                (prepared,),
                pp_mesh=pp_mesh,
            )

        stage.set_micro_labels.assert_called_once()
        stage.clear_all_states.assert_called_once_with()
        self.assertEqual(report["metadata"]["local_stage_indices"], [0])
        self.assertEqual(report["metadata"]["pipeline_chunks"][0]["layer_start"], 0)
        self.assertEqual(report["metadata"]["pipeline_mock"]["transport"], "mocked")


if __name__ == "__main__":
    unittest.main()
