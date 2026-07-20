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
"""Unit tests for Torch dry-run report generation and configuration helpers."""
import csv
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from tests.common.mark_utils import arg_mark
from hyper_parallel import fully_shard
from hyper_parallel.core.activation_checkpoint.recompute_state import (
    create_recompute_contexts,
)
from hyper_parallel.models.spec.model_spec import ModelSpec
from hyper_parallel.models.spec.registry import _SPEC_REGISTRY, register_spec
from hyper_parallel.platform.torch.dry_run import (
    TorchDryRunRunner,
    _operator_phase,
    build_memory_csv_rows,
    build_memory_report,
    write_memory_csv,
    write_memory_report,
)
from hyper_parallel.platform.torch.fully_shard.param_group import HSDPParamGroup
from hyper_parallel.platform.torch.platform import TorchPlatform
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config import HyperTrainerConfig


class _TinyBlock(nn.Module):
    """Small transformer-like residual MLP used by the fake FSDP test."""

    def __init__(self) -> None:
        """Build the two linear projections."""
        super().__init__()
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, 16)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply one residual MLP block."""
        return hidden_states + self.linear2(torch.relu(self.linear1(hidden_states)))


class _TinyLanguageModel(nn.Module):
    """Two-layer causal LM that exercises parameters, activations, and CE."""

    def __init__(self) -> None:
        """Build embeddings, residual blocks, and the LM head."""
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=4,
            tie_word_embeddings=False,
        )
        self.embed_tokens = nn.Embedding(64, 16)
        self.layers = nn.ModuleList([_TinyBlock(), _TinyBlock()])
        self.lm_head = nn.Linear(16, 64, bias=False)

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            use_cache: bool = False,
            **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Return a Trainer-compatible loss/logits mapping."""
        del use_cache, kwargs
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
        )
        return {"loss": loss, "logits": logits}


def _build_tiny_model(args: Any) -> nn.Module:
    """Build the fake FSDP integration model."""
    del args
    return _TinyLanguageModel()


def _parallelize_tiny_model(model: nn.Module, mesh: Any, args: Any) -> nn.Module:
    """Apply the same per-layer plus root FSDP pattern as production models."""
    for layer in model.layers:
        fully_shard(
            layer,
            mesh=mesh["fsdp"],
            reshard_after_forward=args.train.accelerator.reshard_after_forward,
        )
    return fully_shard(
        model,
        mesh=mesh["fsdp"],
        reshard_after_forward=args.train.accelerator.reshard_after_forward,
    )


def _parallelize_tiny_model_no_zero_copy(
        model: nn.Module, mesh: Any, args: Any
) -> nn.Module:
    """Apply fused FSDP while explicitly preserving per-parameter storage."""
    for layer in model.layers:
        fully_shard(
            layer,
            mesh=mesh["fsdp"],
            reshard_after_forward=args.train.accelerator.reshard_after_forward,
            comm_fusion=True,
            comm_fusion_zero_copy=False,
        )
    return fully_shard(
        model,
        mesh=mesh["fsdp"],
        reshard_after_forward=args.train.accelerator.reshard_after_forward,
        comm_fusion=True,
        comm_fusion_zero_copy=False,
    )


class _Category(str, Enum):
    PARAMETER = "Parameter"
    ACTIVATION = "Activation"


class _State(str, Enum):
    PEAK_FORWARD = "Peak-Forward"


class _ModuleStats:
    def __init__(self, fqn: str, peak: int) -> None:
        """Create deterministic module statistics for report tests."""
        self.mod_fqn = fqn
        self.parameter_mem = 128
        self.buffer_mem = 16
        self.input_mem = 32
        self.output_mem = 64
        self.local_peak = {"cuda:0": peak}
        self.snapshots = {
            _State.PEAK_FORWARD: [
                {"cuda:0": {_Category.PARAMETER: 128, "Total": peak}}
            ]
        }


class _Tracker:
    def __init__(self) -> None:
        """Create deterministic global and module memory snapshots."""
        self.memory_tracking = {
            "root": _ModuleStats("Model", 800),
            "layer0": _ModuleStats("Model.layers.0", 900),
            "layer1": _ModuleStats("Model.layers.1", 1000),
        }
        self._snapshots = {
            "peak": {
                "cuda:0": {
                    _Category.PARAMETER: 400,
                    _Category.ACTIVATION: 600,
                    "Total": 1000,
                },
                "cpu": {_Category.ACTIVATION: 50, "Total": 50},
            },
            "current": {
                "cuda:0": {_Category.PARAMETER: 400, "Total": 400},
            },
        }

    def get_tracker_snapshot(self, snapshot_type: str) -> Dict[Any, Dict[Any, int]]:
        """Return the requested deterministic tracker snapshot."""
        return self._snapshots[snapshot_type]


class TestDryRunReport(unittest.TestCase):
    """Report output preserves byte precision and filters module hotspots."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_build_report_filters_modules_and_computes_headroom(self):
        """Feature: Dry-run memory report.

        Description: Build a filtered report with a configured memory capacity.
        Expectation: Only target-device bytes determine peaks, headroom, and OOM risk.
        """
        report = build_memory_report(
            tracker=_Tracker(),
            metadata={"rank": 0},
            device_type="cuda",
            module_depth=3,
            top_modules=1,
            device_memory_gib=0.0000005,
        )

        self.assertEqual(report["summary"]["peak_bytes"], 1000, (
            f"Expected CUDA peak 1000, got {report['summary']['peak_bytes']}"
        ))
        self.assertEqual(
            report["summary"]["peak_breakdown_bytes"],
            {"Activation": 600, "Parameter": 400},
            f"Unexpected peak breakdown: {report['summary']}",
        )
        self.assertEqual(
            report["summary"]["current_breakdown_bytes"],
            {"Parameter": 400},
            f"Unexpected current breakdown: {report['summary']}",
        )
        self.assertTrue(report["summary"]["oom_risk"], (
            f"Expected OOM risk for capacity, got {report['summary']}"
        ))
        self.assertEqual(len(report["modules"]), 1, (
            f"Expected one top module, got {report['modules']}"
        ))
        self.assertEqual(
            report["modules"][0]["fqn"],
            "Model.layers.1",
            f"Unexpected hottest module: {report['modules']}",
        )
        self.assertIn(
            "Peak-Forward",
            report["modules"][0]["snapshots"],
            f"Expected forward snapshot, got {report['modules'][0]}",
        )
        self.assertIn("cpu", report["devices"], (
            f"Expected diagnostic CPU entry, got {report['devices']}"
        ))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_build_report_rejects_missing_target_device(self):
        """Feature: Dry-run report validation.

        Description: Build a report without a target-device memory snapshot.
        Expectation: Validation rejects the incomplete snapshot instead of reporting success.
        """
        with self.assertRaisesRegex(ValueError, "observed no 'npu'"):
            build_memory_report(
                tracker=_Tracker(),
                metadata={},
                device_type="npu",
                module_depth=0,
                top_modules=0,
            )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_operator_phase_distinguishes_checkpoint_recompute(self):
        """Feature: Operator phase classification.

        Description: Exercise forward, backward, recompute, and optimizer runtime states.
        Expectation: Each state maps to its distinct CSV phase.
        """
        tracker = SimpleNamespace(
            _in_opt=False,
            _mod_tracker=SimpleNamespace(is_bw=False),
        )
        self.assertEqual(_operator_phase(tracker), "forward")
        tracker._mod_tracker.is_bw = True
        self.assertEqual(_operator_phase(tracker), "backward")

        _, recompute_context = create_recompute_contexts()
        with recompute_context:
            self.assertEqual(_operator_phase(tracker), "recompute")
            tracker._in_opt = True
            self.assertEqual(_operator_phase(tracker), "optimizer")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_write_report_creates_valid_json_without_temp_file(self):
        """Feature: Atomic JSON report output.

        Description: Write a report into a missing nested directory.
        Expectation: The final JSON is valid and no temporary file remains.
        """
        report = {"schema_version": 1, "status": "ok"}
        with tempfile.TemporaryDirectory() as tmp:
            destination = os.path.join(tmp, "nested", "rank_0.json")
            result = write_memory_report(report, destination)
            with open(result, encoding="utf-8") as report_file:
                loaded = json.load(report_file)
            remaining = os.listdir(os.path.dirname(result))

        self.assertEqual(loaded, report, (
            f"Round-tripped report differs: expected={report}, actual={loaded}"
        ))
        self.assertEqual(remaining, ["rank_0.json"], (
            f"Atomic writer left temporary files: {remaining}"
        ))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_csv_rows_match_mindspore_memory_block_shape(self):
        """Feature: MindSpore-compatible memory-block CSV.

        Description: Convert a logical operator storage record into a CSV row.
        Expectation: The row keeps operator data and omits topology and summary fields.
        """
        memory_blocks = [{
            "start_time_stamp": 0,
            "end_time_stamp": 3,
            "device_addr": "0xf00000000000",
            "stream_id": 0,
            "pool_type": "FakeTensorLogicalMemoryPool",
            "size": 500,
            "actual_used_memory": 1000,
            "actual_peak_memory": 1200,
            "file_name": "/workspace/model.py",
            "line_num": 42,
            "type": "Activation",
            "producer_task": 0,
            "task_name": "forward",
            "node_name": "aten.relu.default",
            "graph_name": "Model.layers.1",
            "user_tasks": [1, 2],
            "python_stack": "File:/workspace/model.py;Line:42;Function:forward",
            "is_persistent": 0,
            "is_small": 1,
        }]
        report = build_memory_report(
            tracker=_Tracker(),
            metadata={
                "rank": 1,
                "world_size": 2,
                "device_type": "npu",
                "simulation_device_type": "cuda",
            },
            device_type="cuda",
            module_depth=3,
            top_modules=1,
            device_memory_gib=0.0000005,
            memory_blocks=memory_blocks,
        )
        rows = build_memory_csv_rows(report)

        self.assertEqual(len(rows), 1, f"Expected one storage row, got {rows}")
        self.assertEqual(rows[0]["node_name"], "aten.relu.default", (
            f"Unexpected operator CSV row: {rows[0]}"
        ))
        self.assertEqual(rows[0]["size"], 500, f"Unexpected size: {rows[0]}")
        self.assertEqual(rows[0]["user_tasks"], "{1-2}", (
            f"Unexpected user task encoding: {rows[0]}"
        ))
        self.assertEqual(rows[0]["last_user_task"], 2, (
            f"Unexpected last user task: {rows[0]}"
        ))
        self.assertNotIn("rank", rows[0], f"CSV retained topology data: {rows[0]}")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_write_csv_creates_parseable_file_without_temp_file(self):
        """Feature: Atomic CSV report output.

        Description: Write one memory block into a missing nested directory.
        Expectation: The result is parseable, zero-indexed, and has no temporary file.
        """
        report = build_memory_report(
            tracker=_Tracker(),
            metadata={"rank": 0, "world_size": 2},
            device_type="cuda",
            module_depth=0,
            top_modules=0,
            memory_blocks=[{
                "start_time_stamp": 0,
                "end_time_stamp": 1,
                "device_addr": "0xf00000000000",
                "size": 64,
                "node_name": "aten.empty.memory_format",
                "user_tasks": [],
                "python_stack": "File:/workspace/model.py;Line:1;Function:forward",
            }],
        )
        with tempfile.TemporaryDirectory() as tmp:
            destination = os.path.join(tmp, "nested", "rank_0_memory.csv")
            result = write_memory_csv(report, destination)
            with open(result, encoding="utf-8", newline="") as report_file:
                rows = list(csv.DictReader(report_file))
            remaining = os.listdir(os.path.dirname(result))

        self.assertEqual(len(rows), 1, f"Expected one CSV memory block, got {rows}")
        self.assertEqual(rows[0]["start_time_stamp"], "0", (
            f"Expected zero-based logical timestamp, got {rows[0]}"
        ))
        self.assertEqual(remaining, ["rank_0_memory.csv"], (
            f"Atomic writer left temporary files: {remaining}"
        ))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_memory_analysis_executes_dry_run_csv_contract(self):
        """Feature: Memory visualization compatibility.

        Description: Execute the viewer parser and analysis functions against dry-run CSV data.
        Expectation: Fake pools, lifetimes, quoting, and snapshot conversion satisfy the contract.
        """
        node_binary = shutil.which("node")
        if node_binary is None:
            self.skipTest("Node.js is required for the HTML behavior contract")
        repository_root = Path(__file__).resolve().parents[4]
        test_script = (
            repository_root
            / "tests/ut/platform/torch/memory_analysis_behavior_test.js"
        )
        completed = subprocess.run(
            [node_binary, str(test_script)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            "memory_analysis.html behavior contract failed:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


class TestDryRunWorldSize(unittest.TestCase):
    """Logical world-size inference follows ParallelDims' product contract."""

    @staticmethod
    def _args(
            world_size: Optional[int] = None,
            dp_shard: Optional[int] = 2,
            legacy_dp: Optional[int] = None,
    ) -> Any:
        accelerator = SimpleNamespace(
            dp=legacy_dp,
            dp_replicate=2,
            dp_shard=dp_shard,
            cp=2,
            tp=2,
            pp=1,
            ep=1,
        )
        dry_run = SimpleNamespace(world_size=world_size)
        return SimpleNamespace(
            train=SimpleNamespace(accelerator=accelerator, dry_run=dry_run)
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_explicit_world_size_wins(self):
        """Feature: Dry-run world-size resolution.

        Description: Configure an explicit logical world size.
        Expectation: The runner preserves the explicit value.
        """
        args = self._args(world_size=64, dp_shard=-1)
        actual = TorchDryRunRunner._resolve_world_size(args)
        self.assertEqual(actual, 64, (
            f"Expected explicit world size 64, got {actual}"
        ))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_parallel_product_is_inferred(self):
        """Feature: Dry-run world-size inference.

        Description: Configure fully explicit parallel dimensions without world size.
        Expectation: The runner infers the product of all parallel degrees.
        """
        args = self._args()
        actual = TorchDryRunRunner._resolve_world_size(args)
        self.assertEqual(actual, 16, (
            f"Expected inferred world size 16, got {actual}"
        ))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_auto_dp_requires_world_size(self):
        """Feature: Dry-run world-size validation.

        Description: Leave both data-parallel degree and world size automatic.
        Expectation: The runner requests an explicit world size.
        """
        args = self._args(dp_shard=-1)
        with self.assertRaisesRegex(ValueError, "world_size is required"):
            TorchDryRunRunner._resolve_world_size(args)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_torch_26_is_rejected(self):
        """Feature: Torch version validation.

        Description: Prepare a run with a Torch version lacking required fake tracking.
        Expectation: Validation fails before distributed setup.
        """
        with self.assertRaisesRegex(RuntimeError, "PyTorch >= 2.7"):
            TorchDryRunRunner._check_torch_version(
                SimpleNamespace(__version__="2.6.0")
            )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_pipeline_parallel_is_rejected(self):
        """Feature: Parallel-mode validation.

        Description: Enable pipeline parallelism for a dry-run.
        Expectation: Validation rejects the unsupported schedule.
        """
        args = HyperTrainerConfig()
        args.train.accelerator.pp = 2
        args.train.accelerator.dp_shard = 1
        args.train.dry_run.world_size = 2
        with self.assertRaisesRegex(NotImplementedError, "pipeline parallelism"):
            TorchDryRunRunner(args)._prepare_args(
                SimpleNamespace(__version__="2.7.1")
            )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_unvalidated_parallel_axis_is_rejected(self):
        """Feature: Parallel-mode validation.

        Description: Enable an unvalidated tensor-parallel axis.
        Expectation: Phase 1 rejects the configuration rather than emitting a report.
        """
        args = HyperTrainerConfig()
        args.train.accelerator.tp = 2
        args.train.accelerator.dp_shard = 1
        args.train.dry_run.world_size = 2
        with self.assertRaisesRegex(NotImplementedError, "FSDP .* only"):
            TorchDryRunRunner(args)._prepare_args(
                SimpleNamespace(__version__="2.7.1")
            )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_full_activation_checkpoint_is_allowed(self):
        """Feature: Full activation checkpoint dry-run.

        Description: Enable full per-layer activation recomputation.
        Expectation: Validation accepts the supported checkpoint mode.
        """
        args = HyperTrainerConfig()
        args.train.accelerator.dp_shard = 1
        args.train.dry_run.world_size = 1
        args.train.gradient_checkpointing.activation_checkpoint = "full"
        prepared = TorchDryRunRunner(args)._prepare_args(
            SimpleNamespace(__version__="2.7.1")
        )
        self.assertEqual(
            prepared.train.gradient_checkpointing.activation_checkpoint,
            "full",
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_selective_activation_checkpoint_is_rejected(self):
        """Feature: Selective activation checkpoint validation.

        Description: Enable selective activation recomputation.
        Expectation: Validation fails until selective policies are supported.
        """
        args = HyperTrainerConfig()
        args.train.gradient_checkpointing.activation_checkpoint = "selective"
        with self.assertRaisesRegex(NotImplementedError, "full activation"):
            TorchDryRunRunner(args)._prepare_args(
                SimpleNamespace(__version__="2.7.1")
            )


class TestDryRunTrainerIntegration(unittest.TestCase):
    """BaseTrainer keeps shape setup but skips checkpoint tensor loading."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_post_parallelize_skips_weights_during_dry_run(self):
        """Feature: Dry-run model materialization.

        Description: Post-process a meta model with a configured weights path.
        Expectation: Fake shards are materialized without loading checkpoint tensors.
        """
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.args = SimpleNamespace(
            train=SimpleNamespace(
                init_device="meta",
                dry_run=SimpleNamespace(enabled=True),
            ),
            model=SimpleNamespace(weights_path="/not/read/checkpoint"),
        )
        trainer.model = MagicMock()

        with patch.object(trainer, "_materialize_and_init_shards") as materialize, \
             patch.object(trainer, "_load_weights") as load_weights, \
             patch.object(trainer, "_maybe_downcast_frozen_params") as downcast, \
             patch.object(trainer, "_maybe_cast_trainable_params") as cast:
            trainer._post_parallelize()

        materialize.assert_called_once_with()
        load_weights.assert_not_called()
        downcast.assert_called_once_with()
        cast.assert_called_once_with()
        trainer.model.train.assert_called_once_with()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_torch_device_type_falls_back_to_cuda_without_torch_npu(self):
        """Feature: Torch device-type discovery.

        Description: Query the platform while the torch.npu extension is absent.
        Expectation: CUDA is selected without accessing a missing NPU attribute.
        """
        had_npu = hasattr(torch, "npu")
        saved_npu = getattr(torch, "npu", None)
        if had_npu:
            delattr(torch, "npu")
        try:
            actual = TorchPlatform().device_type()
        finally:
            if had_npu:
                torch.npu = saved_npu
        self.assertEqual(actual, "cuda", (
            f"Expected CUDA fallback without torch.npu, got {actual}"
        ))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_torch_op_overload_name_is_canonical(self):
        """Feature: Torch operator-name normalization.

        Description: Resolve the name of an in-place optimizer overload.
        Expectation: The canonical YAML registry name excludes overload suffixes.
        """
        actual = TorchPlatform.get_op_name(torch.ops.aten.lerp_.Scalar)
        self.assertEqual(actual, "lerp_", (
            f"Expected canonical overload name 'lerp_', got {actual!r}"
        ))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_non_torch_backend_fails_before_distributed_setup(self):
        """Feature: Dry-run backend validation.

        Description: Start the Torch runner with the MindSpore backend configured.
        Expectation: The error report identifies the backend before process-group setup.
        """
        with tempfile.TemporaryDirectory() as output_dir:
            args = HyperTrainerConfig()
            args.train.backend = "mindspore"
            args.train.dry_run.output_dir = output_dir
            with self.assertRaisesRegex(RuntimeError, "supports train.backend='torch'"):
                TorchDryRunRunner(args).run()
            with open(os.path.join(output_dir, "rank_0.json"), encoding="utf-8") as report_file:
                report = json.load(report_file)

        self.assertEqual(report["status"], "error", f"Unexpected report: {report}")
        self.assertEqual(
            report["error"]["type"],
            "NotImplementedError",
            f"Unexpected error report: {report}",
        )
        self.assertFalse(torch.distributed.is_initialized(), (
            "Unsupported backend unexpectedly initialized a process group"
        ))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_error_report_write_failure_preserves_original_error(self):
        """Feature: Dry-run error reporting.

        Description: Fail both execution and the fallback report writer.
        Expectation: The original execution error is preserved and the I/O error is logged.
        """
        args = HyperTrainerConfig()
        args.train.backend = "mindspore"
        with patch(
                "hyper_parallel.platform.torch.dry_run.write_memory_report",
                side_effect=OSError("disk full"),
        ), self.assertLogs(
                "hyper_parallel.platform.torch.dry_run", level="WARNING"
        ) as captured:
            with self.assertRaisesRegex(RuntimeError, "supports train.backend='torch'"):
                TorchDryRunRunner(args).run()

        self.assertTrue(any(
            "disk full" in message for message in captured.output
        ), f"Expected report-write diagnostic, got {captured.output}")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_existing_process_group_is_rejected_and_preserved(self):
        """Feature: Dry-run process-group ownership.

        Description: Invoke the runner while a caller-owned process group exists.
        Expectation: The runner rejects the call and leaves the existing group intact.
        """
        TorchPlatform.init_dry_run_process_group(world_size=1, rank=0)
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                args = HyperTrainerConfig()
                args.train.dry_run.output_dir = output_dir
                with self.assertRaisesRegex(RuntimeError, "before a real or fake"):
                    TorchDryRunRunner(args).run()
            self.assertTrue(torch.distributed.is_initialized(), (
                "Runner destroyed the caller's existing process group"
            ))
        finally:
            torch.distributed.destroy_process_group()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_parallelize_error_writes_error_report_and_cleans_process_group(self):
        """Feature: Dry-run failure cleanup.

        Description: Run a model spec without the required parallelization callback.
        Expectation: An error report is written and the fake process group is destroyed.
        """
        spec_name = "test_tiny_dry_run_without_parallelize"
        register_spec(
            spec_name,
            ModelSpec(name=spec_name, build_model_fn=_build_tiny_model),
        )
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                args = HyperTrainerConfig()
                args.model.name = spec_name
                args.train.accelerator.dp_shard = 2
                args.train.dry_run.enabled = True
                args.train.dry_run.world_size = 2
                args.train.dry_run.output_dir = output_dir

                with self.assertRaisesRegex(RuntimeError, "must define parallelize_fn"):
                    TorchDryRunRunner(args).run()
                with open(os.path.join(output_dir, "rank_0.json"), encoding="utf-8") as report_file:
                    report = json.load(report_file)
        finally:
            _SPEC_REGISTRY.pop(spec_name, None)

        self.assertEqual(report["status"], "error", f"Unexpected report: {report}")
        self.assertEqual(
            report["stage"], "parallelize", f"Unexpected error stage: {report}"
        )
        self.assertFalse(torch.distributed.is_initialized(), (
            "Dry-run failure leaked its fake process group"
        ))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_fake_fsdp_step_runs_on_available_simulation_device(self):
        """Feature: End-to-end FSDP dry-run.

        Description: Execute world-size one and two fake training steps with memory tracking.
        Expectation: Reports contain sharded memory, operator lifetimes, and clean global state.
        """
        # This end-to-end scenario intentionally keeps setup and cross-report
        # invariants together so cleanup is covered by one try/finally scope.
        # lizard forgives(NLOC, CCN)
        spec_name = "test_tiny_dry_run_fsdp"
        register_spec(
            spec_name,
            ModelSpec(
                name=spec_name,
                build_model_fn=_build_tiny_model,
                parallelize_fn=_parallelize_tiny_model,
            ),
        )
        try:
            reports = []
            deterministic_before = torch.are_deterministic_algorithms_enabled()
            deterministic_env_names = (
                "ASCEND_LAUNCH_BLOCKING",
                "CUDA_LAUNCH_BLOCKING",
                "CUBLAS_WORKSPACE_CONFIG",
                "FLASH_ATTENTION_DETERMINISTIC",
                "HCCL_DETERMINISTIC",
                "PYTHONHASHSEED",
            )
            deterministic_env_before = {
                name: os.environ.get(name) for name in deterministic_env_names
            }
            with tempfile.TemporaryDirectory() as output_dir:
                args = HyperTrainerConfig()
                args.model.name = spec_name
                args.data.max_seq_len = 8
                args.train.micro_batch_size = 2
                args.train.global_batch_size = 4
                args.train.init_device = "cpu"
                args.train.accelerator.dp_shard = 2
                args.train.accelerator.comm_fusion = True
                args.train.mixed_precision.enabled = True
                args.train.mixed_precision.param_dtype = "bfloat16"
                args.train.mixed_precision.reduce_dtype = "float32"
                args.train.optimizer.foreach = False
                args.train.optimizer.max_grad_norm = 0
                args.train.debug.deterministic = True
                args.train.dry_run.enabled = True
                args.train.dry_run.world_size = 2
                args.train.dry_run.device_type = "npu"
                args.train.dry_run.output_dir = output_dir

                args.train.accelerator.dp_shard = 1
                args.train.global_batch_size = 2
                args.train.dry_run.world_size = 1
                args.train.dry_run.rank = 0
                world_one_path = TorchDryRunRunner(args).run()
                with open(world_one_path, encoding="utf-8") as report_file:
                    world_one_report = json.load(report_file)

                args.train.accelerator.dp_shard = 2
                args.train.global_batch_size = 4
                args.train.dry_run.world_size = 2
                for rank in (0, 1):
                    args.train.dry_run.rank = rank
                    report_path = TorchDryRunRunner(args).run()
                    with open(report_path, encoding="utf-8") as report_file:
                        reports.append(json.load(report_file))

                args.train.dry_run.rank = 0
                args.train.accelerator.reshard_after_forward = False
                args.train.mixed_precision.enabled = False
                fp32_report_path = TorchDryRunRunner(args).run()
                with open(fp32_report_path, encoding="utf-8") as report_file:
                    fp32_no_reshard_report = json.load(report_file)
                fp32_csv_path = os.path.splitext(fp32_report_path)[0] + "_memory.csv"
                with open(fp32_csv_path, encoding="utf-8", newline="") as report_file:
                    fp32_csv_rows = list(csv.DictReader(report_file))
        finally:
            _SPEC_REGISTRY.pop(spec_name, None)

        npu_handle = getattr(torch, "npu", None)
        expected_simulation_device = (
            "npu"
            if npu_handle is not None and npu_handle.is_available()
            else "cpu"
        )
        for rank, report in enumerate(reports):
            self.assertEqual(report["status"], "ok", (
                f"Expected successful rank {rank} report, got {report}"
            ))
            self.assertGreater(report["summary"]["peak_bytes"], 0, (
                f"Expected a positive fake-device peak, got {report['summary']}"
            ))
            breakdown = report["summary"]["peak_breakdown_bytes"]
            self.assertIn("Parameter", breakdown, f"Missing Parameter: {breakdown}")
            self.assertIn("Activation", breakdown, f"Missing Activation: {breakdown}")
            self.assertIn("Gradient", breakdown, f"Missing Gradient: {breakdown}")
            current_breakdown = report["summary"]["current_breakdown_bytes"]
            self.assertGreater(current_breakdown.get("Parameter", 0), 0, (
                f"Expected a resident FSDP parameter shard, got {current_breakdown}"
            ))
            self.assertGreater(current_breakdown.get("Optstate", 0), 0, (
                f"Expected AdamW state after the optimizer step, got {current_breakdown}"
            ))
            self.assertGreater(
                report["summary"]["peak_bytes"],
                current_breakdown["Parameter"],
                f"Expected all-gather/step peak above resident shard: {report['summary']}",
            )
            persistent_activation_bytes = sum(
                block["size"]
                for block in report["memory_blocks"]
                if block["type"] == "Activation" and block["is_persistent"]
            )
            self.assertLessEqual(
                persistent_activation_bytes,
                current_breakdown.get("Activation", 0),
                "Operator output lifetimes must not retain released activations: "
                f"persistent={persistent_activation_bytes}, current={current_breakdown}",
            )
            persistent_output_bytes = sum(
                block["size"]
                for block in report["memory_blocks"]
                if block["is_persistent"]
            )
            self.assertLessEqual(
                persistent_output_bytes,
                report["summary"]["current_bytes"],
                "Persistent operator outputs must fit in tracker current memory: "
                f"persistent={persistent_output_bytes}, summary={report['summary']}",
            )
            memory_changes = {}
            for block in report["memory_blocks"]:
                start = block["start_time_stamp"]
                memory_changes[start] = memory_changes.get(start, 0) + block["size"]
                if not block["is_persistent"]:
                    end = block["end_time_stamp"]
                    memory_changes[end] = memory_changes.get(end, 0) - block["size"]
            logical_current = 0
            logical_peak = 0
            for task_index in sorted(memory_changes):
                logical_current += memory_changes[task_index]
                logical_peak = max(logical_peak, logical_current)
            self.assertLessEqual(
                logical_peak,
                report["summary"]["peak_bytes"],
                "Operator lifetime peak must fit in the MemTracker peak: "
                f"lifetime={logical_peak}, summary={report['summary']}",
            )
            self.assertEqual(
                report["metadata"]["parallel"]["dp_shard"],
                2,
                f"Unexpected topology metadata: {report['metadata']}",
            )
            self.assertEqual(
                report["metadata"]["rank"], rank,
                f"Unexpected rank metadata: {report['metadata']}",
            )
            self.assertEqual(
                report["metadata"]["device_type"], "npu",
                f"Unexpected target device: {report['metadata']}",
            )
            self.assertEqual(
                report["metadata"]["simulation_device_type"],
                expected_simulation_device,
                f"Unexpected simulation device: {report['metadata']}",
            )
            has_cpu_fallback_caveat = any(
                "simulated with cpu FakeTensors" in limitation
                for limitation in report["limitations"]
            )
            self.assertEqual(
                has_cpu_fallback_caveat,
                expected_simulation_device == "cpu",
                f"Unexpected CPU fallback caveat: {report['limitations']}",
            )

        world_one_parameter = world_one_report["summary"]["current_breakdown_bytes"]["Parameter"]
        world_two_parameter = reports[0]["summary"]["current_breakdown_bytes"]["Parameter"]
        self.assertEqual(world_one_parameter, 2 * world_two_parameter, (
            "Expected FSDP2 local parameter bytes to be half of world-size=1, "
            f"got world1={world_one_parameter}, world2={world_two_parameter}"
        ))
        self.assertEqual(
            fp32_no_reshard_report["status"],
            "ok",
            f"Expected fp32/no-reshard success, got {fp32_no_reshard_report}",
        )
        self.assertFalse(
            fp32_no_reshard_report["metadata"]["reshard_after_forward"],
            f"Expected no-reshard metadata, got {fp32_no_reshard_report['metadata']}",
        )
        self.assertGreater(
            fp32_no_reshard_report["summary"]["current_breakdown_bytes"].get("Parameter", 0),
            0,
            f"Expected resident fp32 parameter shard, got {fp32_no_reshard_report['summary']}",
        )
        memory_blocks = fp32_no_reshard_report["memory_blocks"]
        self.assertGreater(len(memory_blocks), 0, "Expected operator storage records")
        self.assertTrue(any(
            "test_dry_run.py" in block["python_stack"]
            for block in memory_blocks
        ), "Expected live model Python frames in operator records")
        self.assertTrue(any(
            row["node_name"] and row["device_addr"].startswith("0x")
            for row in fp32_csv_rows
        ), f"Expected logical memory-block CSV rows, got {fp32_csv_rows[:5]}")
        self.assertEqual(min(
            int(row["start_time_stamp"]) for row in fp32_csv_rows
        ), 0, "Logical memory-block timestamps did not start at zero")
        self.assertNotIn("record_type", fp32_csv_rows[0], (
            f"CSV retained the old summary/operator schema: {fp32_csv_rows[0]}"
        ))
        self.assertFalse(torch.distributed.is_initialized(), (
            "Successful dry-runs leaked the fake process group"
        ))
        self.assertEqual(
            torch.are_deterministic_algorithms_enabled(),
            deterministic_before,
            "Dry-run changed the caller's deterministic-algorithm setting",
        )
        deterministic_env_after = {
            name: os.environ.get(name) for name in deterministic_env_names
        }
        self.assertEqual(
            deterministic_env_after,
            deterministic_env_before,
            "Dry-run changed deterministic environment variables: "
            f"before={deterministic_env_before}, after={deterministic_env_after}",
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_zero_copy_disabled_preserves_normal_fsdp_storage_path(self):
        """Feature: FSDP zero-copy configuration.

        Description: Run dry-run with communication-fusion zero-copy disabled.
        Expectation: No flat parameter buffer is created and resident parameters remain tracked.
        """
        spec_name = "test_tiny_dry_run_no_zero_copy"
        register_spec(
            spec_name,
            ModelSpec(
                name=spec_name,
                build_model_fn=_build_tiny_model,
                parallelize_fn=_parallelize_tiny_model_no_zero_copy,
            ),
        )
        flat_buffer_calls = []
        original_init_flat_buffer = HSDPParamGroup._init_flat_param_buffer

        def _track_flat_buffer_init(param_group: HSDPParamGroup) -> None:
            flat_buffer_calls.append(param_group.enable_zero_copy)
            original_init_flat_buffer(param_group)

        try:
            with tempfile.TemporaryDirectory() as output_dir:
                args = HyperTrainerConfig()
                args.model.name = spec_name
                args.data.max_seq_len = 8
                args.train.micro_batch_size = 2
                args.train.global_batch_size = 4
                args.train.init_device = "cpu"
                args.train.accelerator.dp_shard = 2
                args.train.optimizer.foreach = False
                args.train.optimizer.max_grad_norm = 0
                args.train.dry_run.enabled = True
                args.train.dry_run.world_size = 2
                args.train.dry_run.output_dir = output_dir

                with patch.object(
                        HSDPParamGroup,
                        "_init_flat_param_buffer",
                        new=_track_flat_buffer_init,
                ):
                    report_path = TorchDryRunRunner(args).run()
                with open(report_path, encoding="utf-8") as report_file:
                    report = json.load(report_file)
        finally:
            _SPEC_REGISTRY.pop(spec_name, None)

        self.assertEqual(report["status"], "ok", f"Unexpected report: {report}")
        self.assertEqual(flat_buffer_calls, [], (
            "Dry-run initialized a flat parameter buffer even though "
            "comm_fusion_zero_copy=False"
        ))
        self.assertGreater(
            report["summary"]["current_breakdown_bytes"].get("Parameter", 0),
            0,
            f"Expected tracked resident parameters, got {report['summary']}",
        )


if __name__ == "__main__":
    unittest.main()
