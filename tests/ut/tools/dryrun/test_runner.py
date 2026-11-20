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
"""Unit tests for the primary LLM dry-run workflow."""
# pylint: disable=protected-access

from dataclasses import dataclass
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torch.nn import functional
from torch._subclasses.fake_tensor import FakeTensorMode

from hyper_parallel.models._transformers import model_builder
from hyper_parallel.tools.dryrun.config import DryRunConfig
from hyper_parallel.tools.dryrun import __main__ as dryrun_main
from hyper_parallel.tools.dryrun.__main__ import _configured_ranks
from hyper_parallel.tools.dryrun.runtime import (
    DryRunBatchMocker,
    DryRunRuntime,
    FakeHostScalarShim,
    _MoERoutingPlans,
    _MoERoutingRuntime,
    derive_tp_target_counts,
)
from hyper_parallel.trainer.config import (
    ActivationCheckpointConfig,
    OptimizerConfig,
    SelectiveCheckpointConfig,
    Target,
    TrainerConfig,
    TrainingConfig,
)
from hyper_parallel.trainer.config.parser import load_training_config
from hyper_parallel.tools.dryrun import examples as dryrun_examples
from hyper_parallel.tools.dryrun.runner import HyperModelsDryRunRunner, _DryRunTrainingBatch


def _target(**kwargs: Any) -> dict[str, Any]:
    """Provide a callable placeholder for validation-only configurations."""
    return kwargs


def _config(dry_run: Optional[DryRunConfig] = None) -> TrainerConfig:
    """Build a minimal Trainer configuration for dry-run tests."""
    return TrainerConfig(
        model=Target(_target, target_path="tests.model", path="unused"),
        optimizer=OptimizerConfig(Target(_target, target_path="tests.optimizer")),
        training=TrainingConfig(global_batch_size=1, micro_batch_size=1, backend="gloo"),
        dry_run=dry_run or DryRunConfig(),
    )


def _named_model(*modules: tuple[str, Any]) -> Any:
    """Expose the named-module surface consumed by value-dependency rules."""
    return SimpleNamespace(named_modules=lambda: [("", SimpleNamespace()), *modules])


def _plans(mode: str = "balanced", **options: Any) -> _MoERoutingPlans:
    """Build routed-expert replay plans for one mode."""
    return _MoERoutingPlans(mode, **options)


class _TinyModel(torch.nn.Module):
    """Minimal differentiable model accepted by the dry-run micro-step."""

    def __init__(self) -> None:
        """Create one parameterized projection."""
        super().__init__()
        self.projection = torch.nn.Linear(4, 4)
        self.config = SimpleNamespace(vocab_size=4)

    def forward(self, inputs: torch.Tensor, use_cache: bool = False) -> Any:
        """Return an object exposing a Trainer-compatible loss."""
        del use_cache
        return SimpleNamespace(loss=self.projection(inputs).square().mean())


class _KwargLoss(torch.nn.Module):
    """Trainer-compatible loss module reading the model-computed loss."""

    def forward(self, *, model_output: Any, labels: Any = None) -> torch.Tensor:
        """Return the loss the tiny model computed."""
        del labels
        return model_output.loss


class TestDryRunConfiguration(unittest.TestCase):
    """Tests for dry-run validation and Trainer entrypoint routing."""

    def test_validate_accepts_accelerator_targets_and_rejects_disabled_mode(self):
        """Accept supported report targets and require dry-run to be enabled."""
        runtime = DryRunRuntime(rank=0, world_size=1, local_rank=0)
        for target_device in ("npu", "cuda"):
            with self.subTest(target_device=target_device):
                runner = HyperModelsDryRunRunner(
                    _config(DryRunConfig(target_device=target_device)),
                    runtime,
                )
                self.assertEqual(runner._validate_config().target_device, target_device)

        runner = HyperModelsDryRunRunner(
            _config(DryRunConfig(enabled=False)),
            runtime,
        )
        with self.assertRaisesRegex(ValueError, "dry_run.enabled must be true"):
            runner._validate_config()

    def test_validate_rejects_invalid_topology_and_unsupported_mutation(self):
        """Reject impossible parallel sizes and unsupported model mutation."""
        runtime = DryRunRuntime(rank=0, world_size=1, local_rank=0)
        config = _config()
        config.accelerator.ep_size = 2
        with self.assertRaisesRegex(ValueError, "expert domain size"):
            HyperModelsDryRunRunner(config, runtime)._validate_config()

        config = _config()
        config.peft = SimpleNamespace()
        with self.assertRaisesRegex(NotImplementedError, "PEFT"):
            HyperModelsDryRunRunner(config, runtime)._validate_config()

        # Offload and swap features simulate: their storage lifecycle is
        # host-ordered (resize_(0) on wait_offload, resize_ + copy on load),
        # so validation accepts them.
        config = _config()
        config.activation_checkpoint = ActivationCheckpointConfig(
            mode="selective",
            selective=SelectiveCheckpointConfig(swap={"self_attn": [0]}),
        )
        config.fsdp_config.enable_offload = True
        config.optimizer.swap.enabled = True
        HyperModelsDryRunRunner(config, runtime)._validate_config()

        config = _config()
        config.accelerator.pp_size = 2
        with self.assertRaisesRegex(
                NotImplementedError,
                "does not support pipeline parallelism",
        ):
            HyperModelsDryRunRunner(
                config,
                DryRunRuntime(rank=0, world_size=2, local_rank=0),
            )._validate_config()

    def test_validate_normalizes_yaml_mapping(self):
        """The trainer config keeps dry-run as a generic mapping until validation."""
        runtime = DryRunRuntime(rank=0, world_size=1, local_rank=0)
        config = _config()
        config.dry_run = {
            "enabled": True,
            "target_device": "npu",
            "ranks": [0],
            "snapshot": {
                "enabled": True,
                "path": "outputs/dry_run",
                "capture_stacks": False,
                "stack_depth": 8,
            },
        }
        runner = HyperModelsDryRunRunner(config, runtime)
        dry_run = runner._validate_config()
        self.assertIsInstance(dry_run, DryRunConfig)
        self.assertFalse(dry_run.snapshot.capture_stacks)
        self.assertEqual(dry_run.snapshot.stack_depth, 8)
        self.assertEqual(
            runner._snapshot_path(dry_run, runtime),
            "outputs/dry_run/rank_0/memory.pickle",
        )

    def test_validate_rejects_rank_outside_world_size(self):
        """Configured ranks must exist in the launched world size."""
        runtime = DryRunRuntime(rank=0, world_size=2, local_rank=0)
        config = _config(DryRunConfig(ranks=[0, 2]))
        with self.assertRaisesRegex(ValueError, "dry_run.ranks"):
            HyperModelsDryRunRunner(config, runtime)._validate_config()

    def test_cli_rank_defaults_and_validation(self):
        """CLI rank selection defaults to rank zero and rejects invalid entries."""
        self.assertEqual(_configured_ranks(DryRunConfig(), 8), [0])
        self.assertEqual(_configured_ranks(DryRunConfig(ranks=[0, 7]), 8), [0, 7])
        with self.assertRaisesRegex(ValueError, "dry_run.ranks"):
            _configured_ranks(DryRunConfig(ranks=[8]), 8)

    def test_validate_steps_default_target_and_requires_torchrun(self):
        """Default target is npu, steps=2, and CLI requires torchrun env."""
        default_cfg = DryRunConfig()
        self.assertEqual(default_cfg.target_device, "npu")
        self.assertEqual(default_cfg.steps, 2)

        for bad_steps in (0, -1, True):
            runner = HyperModelsDryRunRunner(
                _config(DryRunConfig(steps=bad_steps)),
                DryRunRuntime(rank=0, world_size=1, local_rank=0),
            )
            with self.assertRaisesRegex(ValueError, "dry_run.steps"):
                runner._validate_config()

        with patch.object(dryrun_main, "parse_training_args", return_value=_config()), \
                patch.object(dryrun_main, "_torchrun_identity", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "torchrun"):
                dryrun_main.main()

    def test_cli_runs_selected_torchrun_rank_and_skips_unselected(self):
        """Under torchrun, only ranks listed in dry_run.ranks invoke the runner."""
        cfg = _config(DryRunConfig(ranks=[0, 7]))
        ran = []

        class _FakeRunner:
            def __init__(self, config):
                self.config = config

            def run(self):
                ran.append(self.config.dry_run.ranks)

        with patch.object(dryrun_main, "parse_training_args", return_value=cfg), \
                patch.object(dryrun_main, "HyperModelsDryRunRunner", _FakeRunner):
            with patch.object(dryrun_main, "_torchrun_identity", return_value=(1, 8, 1)):
                dryrun_main.main()
            self.assertEqual(ran, [])
            with patch.object(dryrun_main, "_torchrun_identity", return_value=(7, 8, 7)):
                dryrun_main.main()
            self.assertEqual(ran, [[0, 7]])

    def test_builtin_deepseek_v41_and_st_recipes_parse_cleanly(self):
        """Ensure shipped dry-run YAML recipes stay aligned with TrainerConfig schema."""
        example_yaml = Path(dryrun_examples.__file__).resolve().parent / "deepseek_v41.yaml"
        example_cfg = load_training_config(str(example_yaml))
        self.assertEqual(example_cfg.dry_run["steps"], 2)

        st_recipe = Path(__file__).resolve().parents[3] / "torch/tools/dryrun_sync_recipe.yaml"
        if not st_recipe.exists():
            return
        st_template = st_recipe.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            rendered = (
                st_template
                .replace("@MODEL_PATH@", tmp)
                .replace("@ENGRAM_PATH@", tmp)
                .replace("@DATA_PATH@", tmp)
                .replace("@OUTPUT_ROOT@", tmp)
            )
            recipe_path = Path(tmp) / "st_recipe.yaml"
            recipe_path.write_text(rendered, encoding="utf-8")
            st_cfg = load_training_config(str(recipe_path))
        self.assertEqual(st_cfg.dry_run["steps"], 2)


class TestDryRunModelAndData(unittest.TestCase):
    """Tests for meta-model creation and normal-data reuse."""

    def test_dry_run_builds_the_configured_target_with_distributed_setup(self):
        """Use arbitrary configured factories instead of reparsing Hugging Face paths."""
        captured = {}

        def from_config_factory(distributed_setup: Any = None, **kwargs: Any) -> torch.nn.Module:
            """Capture build options and create a deferred meta model."""
            captured.update(kwargs)
            captured["distributed_setup"] = distributed_setup
            captured["deferred"] = getattr(model_builder, "is_model_materialization_deferred")()
            model = torch.nn.Linear(2, 2, device="meta")
            model.config = SimpleNamespace(vocab_size=8)
            return model

        config = _config()
        config.model = Target(
            from_config_factory,
            target_path="tests.from_config_factory",
            config_path="config-only-model",
        )
        setup = SimpleNamespace(marker="setup")
        model = HyperModelsDryRunRunner(config, DryRunRuntime(0, 1, 0))._build_target_model(setup)

        self.assertEqual(model.config.vocab_size, 8)
        self.assertIs(captured["distributed_setup"], setup)
        self.assertEqual(captured["config_path"], "config-only-model")
        self.assertTrue(captured["deferred"])
        self.assertFalse(getattr(model_builder, "is_model_materialization_deferred")())

    def test_next_prepared_batch_counts_tokens_from_the_real_batch(self):
        """Derive token counts from the real CPU batch, not from fake values."""
        labels = torch.tensor([[0, 1, -100, 3]])
        loss_mask = torch.tensor([[1, 1, 0, 1]])
        base = MagicMock()
        base.get_batch.return_value = (
            {"input_ids": torch.tensor([[0, 1, 2, 3]])},
            {"labels": labels, "loss_mask": loss_mask},
        )
        prepared = HyperModelsDryRunRunner._next_prepared_batch(base, iter([object()]))

        self.assertEqual(prepared.token_counts["foundation_tokens"], 2)
        self.assertIs(prepared.loss_inputs["labels"], labels)

    def test_batch_mocker_preserves_structure_and_tp_label_statistics(self):
        """Erase tensor values while retaining batch structure and label counts."""
        labels = torch.tensor([[0, 1, 6, -100]])
        loss_inputs = {
            "labels": labels,
            "loss_mask": torch.tensor([[1, 1, 0]]),
        }
        valid_count, owned = derive_tp_target_counts(loss_inputs, vocab_size=8, tp_size=2)

        with FakeTensorMode() as fake_mode:
            mocked = DryRunBatchMocker(fake_mode, torch.device("cpu")).mock({
                "tokens": torch.ones(1, 4, dtype=torch.int64),
                "nested": [torch.ones(2, dtype=torch.float32)],
            })

        self.assertEqual(valid_count, 2)
        self.assertEqual(owned, (1, 1))
        self.assertEqual(mocked["tokens"].shape, (1, 4))
        self.assertEqual(mocked["tokens"].dtype, torch.int64)
        self.assertEqual(mocked["nested"][0].shape, (2,))


class TestDryRunValueDependencies(unittest.TestCase):
    """Tests for value-dependent branches used by LLM dry-run."""

    def test_moe_routing_supports_balanced_and_explicit_loads(self):
        """Produce deterministic EP traffic for common routing profiles."""
        experts = SimpleNamespace(w1=None, w2=None, w3=None)
        module = SimpleNamespace(num_experts=4, top_k=2, experts=experts, gate=SimpleNamespace())

        balanced_plan = _plans().moe_routing_plan("mlp", module, 2, 0, 4, 2)
        self.assertEqual(balanced_plan.input_split_sizes, (4, 4))

        explicit_plan = _plans(
            "explicit",
            source_expert_loads=[[4, 4, 0, 0], [0, 0, 4, 4]],
        ).moe_routing_plan("mlp", module, 2, 1, 4, 2)
        self.assertEqual(explicit_plan.input_split_sizes, (0, 8))
        self.assertEqual(explicit_plan.output_split_sizes, (0, 8))

    def test_moe_routing_rejects_an_unknown_mode(self):
        """Reject routing modes the replay cannot represent."""
        with self.assertRaisesRegex(ValueError, "replay mode"):
            _plans("unknown")

    def test_local_moe_replay_accepts_model_specific_keyword_arguments(self):
        """Ignore keyword-only model context that does not change routed shapes."""
        module = torch.nn.Module()
        module.experts = SimpleNamespace(w1=None, w2=None, w3=None)
        module.gate = SimpleNamespace()
        model = torch.nn.Module()
        model.mlp = module
        runtime = _MoERoutingRuntime(
            _plans(),
            model,
            SimpleNamespace(ep_size=1, ep_rank=0),
        )
        expected = torch.empty(1, 2, 4)
        runtime._run_moe_forward = MagicMock(return_value=expected)

        self.assertEqual(runtime._ep_group_identity(), (1, 0))
        runtime.install()
        actual = module(torch.empty(1, 2, 4), input_ids=torch.ones(1, 2, dtype=torch.long))

        self.assertIs(actual, expected)
        runtime._run_moe_forward.assert_called_once()

    def test_eager_expert_replay_matches_per_expert_packed_swiglu(self):
        """Replay the non-grouped EP expert path with independent expert activations."""
        experts = SimpleNamespace(
            _ep_apply_gate=lambda gate_up: functional.silu(gate_up.chunk(2, dim=-1)[0])
            * gate_up.chunk(2, dim=-1)[1]
        )
        gate_up = torch.randn(2, 6, 4)
        down = torch.randn(2, 4, 3)
        values = torch.randn(5, 4)
        loads = (2, 3)

        actual = _MoERoutingRuntime._run_eager_experts(
            experts,
            "packed_gate_up",
            (gate_up, down),
            values,
            loads,
        )
        expected = []
        start = 0
        for expert, load in enumerate(loads):
            rows = values[start:start + load]
            packed = functional.linear(rows, gate_up[expert])
            gate, up = packed.chunk(2, dim=-1)
            expected.append(functional.linear(functional.silu(gate) * up, down[expert]))
            start += load

        torch.testing.assert_close(actual, torch.cat(expected))

class TestDryRunExecution(unittest.TestCase):
    """Tests for FakeTensor execution, memory accounting, and reporting."""

    def test_fake_step_runs_forward_backward_optimizer_and_builds_report(self):
        """Drive the real train_step over FakeTensors and report memory."""
        from hyper_parallel.trainer.base import BaseTrainer  # pylint: disable=import-outside-toplevel
        from hyper_parallel.trainer.runtime.model_integration import (  # pylint: disable=import-outside-toplevel
            DisabledModelIntegrationSession,
        )

        runner = HyperModelsDryRunRunner(_config(), DryRunRuntime(0, 1, 0))
        runner._simulation_device = "cpu"
        runner._target_device = "cuda"

        with FakeTensorMode(allow_non_fake_inputs=True):
            model = _TinyModel()
            labels = torch.zeros(1, 4, dtype=torch.long)
            batch = _DryRunTrainingBatch(
                model_inputs={"inputs": torch.empty(1, 4)},
                loss_inputs={"labels": labels},
                token_counts={"foundation_tokens": 4},
                valid_token_count=0,
                target_tokens_per_rank=(0,),
            )
            base = BaseTrainer.__new__(BaseTrainer)
            base.config = runner.config
            base.local_rank = 0
            base.global_rank = 0
            base.world_size = 1
            base.device = torch.device("cpu")
            base.model = model
            base.model_config = model.config
            base.model_parts = [model]
            base.hsdp_model_parts = []
            base.mesh = SimpleNamespace(
                dp_size=1,
                dp_replicate_size=1,
                dp_shard_size=1,
                tp_size=1,
                cp_size=1,
                ep_size=1,
                pp_size=1,
                tp_rank=0,
                dp_rank=0,
                loss_parallel=False,
                sequence_parallel=False,
                dp_cp_mesh=None,
                device_mesh=None,
            )
            base.dp_cp_mesh = None
            base.device_mesh = None
            base.loss_fn = _KwargLoss()
            base.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=False)
            base.lr_scheduler = None
            base.graph_compiler = None
            base.num_micro_batches = 1
            base.model_integration = DisabledModelIntegrationSession()
            base.fsdp_runtime_diagnostics = None
            base.train_dataloader = None
            base._build_training_context()
            base._init_callbacks()
            report = runner._execute_step(base, batch)

        self.assertEqual(report["metadata"]["num_micro_batches"], 1)
        self.assertEqual(report["metadata"]["parallel"]["dp"], 1)
        self.assertGreater(report["summary"]["peak_bytes"], 0)
        self.assertEqual(report["metadata"]["target_device"], "cuda")
        self.assertIs(type(report["metadata"]["torch_version"]), str)
        self.assertTrue(report["device_traces"][0])
        # Model parameters (64B weight + 16B bias seeds, 32B+16B optimizer
        # momentum) survive the step; the real train_step additionally keeps
        # its scalar bookkeeping (token counts, loss accumulators) alive.
        # Assert the parameter seeds remain and transient activations do not.
        self.assertGreaterEqual(report["summary"]["current_bytes"], 96)
        self.assertLess(report["summary"]["current_bytes"], 512)
        self.assertNotIn("modules", report)
        self.assertNotIn("peak_breakdown_bytes", report["summary"])

    def _run_tiny_adamw(self, steps: int) -> dict[str, Any]:
        """Execute ``steps`` simulated steps with lazy AdamW optimizer state."""
        from hyper_parallel.trainer.base import BaseTrainer  # pylint: disable=import-outside-toplevel
        from hyper_parallel.trainer.runtime.model_integration import (  # pylint: disable=import-outside-toplevel
            DisabledModelIntegrationSession,
        )

        runner = HyperModelsDryRunRunner(_config(DryRunConfig(steps=steps)), DryRunRuntime(0, 1, 0))
        runner._simulation_device = "cpu"
        runner._target_device = "cuda"

        with FakeTensorMode(allow_non_fake_inputs=True):
            model = _TinyModel()
            batch = _DryRunTrainingBatch(
                model_inputs={"inputs": torch.empty(8, 4)},
                loss_inputs={"labels": torch.zeros(8, 4, dtype=torch.long)},
                token_counts={"foundation_tokens": 32},
                valid_token_count=0,
                target_tokens_per_rank=(0,),
            )
            base = BaseTrainer.__new__(BaseTrainer)
            base.config = runner.config
            base.local_rank = 0
            base.global_rank = 0
            base.world_size = 1
            base.device = torch.device("cpu")
            base.model = model
            base.model_config = model.config
            base.model_parts = [model]
            base.hsdp_model_parts = []
            base.mesh = SimpleNamespace(
                dp_size=1, dp_replicate_size=1, dp_shard_size=1, tp_size=1,
                cp_size=1, ep_size=1, pp_size=1, tp_rank=0, dp_rank=0,
                loss_parallel=False, sequence_parallel=False,
                dp_cp_mesh=None, device_mesh=None,
            )
            base.dp_cp_mesh = None
            base.device_mesh = None
            from hyper_parallel.core.optimizer.adamw import AdamW  # pylint: disable=import-outside-toplevel
            base.loss_fn = _KwargLoss()
            base.optimizer = AdamW(model.parameters(), lr=1e-3)
            base.lr_scheduler = None
            base.graph_compiler = None
            base.num_micro_batches = 1
            base.model_integration = DisabledModelIntegrationSession()
            base.fsdp_runtime_diagnostics = None
            base.train_dataloader = None
            base._build_training_context()
            base._init_callbacks()
            return runner._execute_step(base, batch)

    def test_multi_step_captures_lazy_optimizer_state_in_steady_state_peak(self):
        """Step 2 overlaps forward/backward peaks with Step 0's lazy AdamW states."""
        report_step1 = self._run_tiny_adamw(steps=1)
        report_step2 = self._run_tiny_adamw(steps=2)
        self.assertEqual(report_step1["metadata"]["steps"], 1)
        self.assertEqual(report_step2["metadata"]["steps"], 2)
        self.assertGreater(
            report_step2["summary"]["peak_bytes"],
            report_step1["summary"]["peak_bytes"],
        )

    def test_fake_host_scalar_shim_preserves_scalar_dtypes(self):
        """Integer and boolean FakeTensors return int/bool from .item() and .tolist()."""
        shim = FakeHostScalarShim()
        with FakeTensorMode(), shim:
            int_t = torch.empty((), dtype=torch.int64)
            vec_t = torch.empty((2,), dtype=torch.int64)
            bool_t = torch.empty((), dtype=torch.bool)
            float_t = torch.empty((), dtype=torch.float32)
            shim.register(int_t, 7)
            shim.register(bool_t, 1)
            shim.register(float_t, 2.5)
            int_val = int_t.item()
            bool_val = bool_t.item()
            float_val = float_t.item()
            vec_list = vec_t.tolist()
        self.assertIs(type(int_val), int)
        self.assertEqual(int_val, 7)
        self.assertIs(type(bool_val), bool)
        self.assertTrue(bool_val)
        self.assertIs(type(float_val), float)
        self.assertEqual(float_val, 2.5)
        self.assertEqual(vec_list, [0, 0])
        self.assertTrue(all(item.__class__ is int for item in vec_list))

    def test_trace_peak_bytes_includes_pre_window_resident_segments(self):
        """trace_peak_bytes adds pre-existing resident bytes inferred from segments."""
        st_file = Path(__file__).resolve().parents[3] / "torch/tools/dryrun_sync_common.py"
        if not st_file.exists():
            self.skipTest("tests/torch/tools/dryrun_sync_common.py is unavailable in UT-only bundle")
        spec = importlib.util.spec_from_file_location("dryrun_sync_common", st_file)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        dryrun_sync_common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dryrun_sync_common)
        snapshot = {
            "segments": [{"active_size": 1000}],
            "device_traces": [[
                {"action": "alloc", "size": 500},
                {"action": "free_completed", "size": 500},
                {"action": "alloc", "size": 200},
            ]],
        }
        # net_delta = +200, end_live = 1000 => start_live = 800, peak = 800 + 500 = 1300
        self.assertEqual(dryrun_sync_common.trace_peak_bytes(snapshot), 1300)


@dataclass
class _MetaBatchSample:
    """Dataclass fixture for _tensor_leaves and _batch_shape_metadata."""

    tensor: torch.Tensor
    note: str


class TestDryRunRunnerCoverage(unittest.TestCase):
    """Comprehensive unit tests for HyperModelsDryRunRunner and CLI entry."""

    def test_validate_config_and_topology_error_paths(self):
        """Exercise static validation errors across config, snapshot, ranks, features, and topology."""
        rt = DryRunRuntime(rank=0, world_size=8, local_rank=0)

        with patch.object(torch, "__version__", "2.5.0"):
            with self.assertRaisesRegex(RuntimeError, "PyTorch >= 2.7"):
                HyperModelsDryRunRunner._check_torch_version()

        cfg = _config()
        cfg.dry_run = None
        with self.assertRaisesRegex(ValueError, "must be configured"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.dry_run = {"unknown_field": 1}
        with self.assertRaisesRegex(ValueError, "Invalid dry_run"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.dry_run = "bad"
        with self.assertRaisesRegex(TypeError, "must resolve to a DryRunConfig"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.dry_run.steps = 0
        with self.assertRaisesRegex(ValueError, "dry_run.steps"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.dry_run.snapshot = "bad"
        with self.assertRaisesRegex(TypeError, "DryRunSnapshotConfig"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.dry_run.snapshot.enabled = True
        cfg.dry_run.snapshot.path = "   "
        with self.assertRaisesRegex(ValueError, "snapshot.path"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.dry_run.snapshot.capture_stacks = "yes"
        with self.assertRaisesRegex(TypeError, "capture_stacks"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.dry_run.snapshot.stack_depth = 0
        with self.assertRaisesRegex(ValueError, "stack_depth"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.dry_run.ranks = [0, 0]
        with self.assertRaisesRegex(ValueError, "dry_run.ranks"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.training.micro_batch_size = 0
        with self.assertRaisesRegex(ValueError, "micro_batch_size"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.activation_checkpoint = SimpleNamespace(mode="unsupported")
        with self.assertRaisesRegex(NotImplementedError, "activation checkpoint"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.hyper_offload.enable = True
        with self.assertRaisesRegex(NotImplementedError, "hyper_offload"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.fsdp_config.activation_checkpointing = True
        with self.assertRaisesRegex(NotImplementedError, "FSDP activation checkpointing"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.peft = object()
        with self.assertRaisesRegex(NotImplementedError, "PEFT"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.compile.enabled = True
        with self.assertRaisesRegex(NotImplementedError, "torch.compile"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.accelerator.pp_size = 2
        with self.assertRaisesRegex(NotImplementedError, "pipeline parallelism"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.dry_run.ranks = [8]
        with self.assertRaisesRegex(ValueError, "smaller than virtual world size"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.accelerator.tp_size = 0
        with self.assertRaisesRegex(ValueError, "positive integers"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.accelerator.tp_size = 3
        with self.assertRaisesRegex(ValueError, "not divisible by TP\\*CP\\*PP"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.fsdp_config.dp_shard_size = 3
        with self.assertRaisesRegex(ValueError, "dp_shard_size"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.accelerator.ep_size = 3
        with self.assertRaisesRegex(ValueError, "ep_size"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.fsdp_config.edp_shard_size = 3
        with self.assertRaisesRegex(ValueError, "edp_shard_size"):
            HyperModelsDryRunRunner(cfg, rt)._validate_config()

        cfg = _config()
        cfg.dry_run.snapshot.enabled = True
        cfg.dry_run.snapshot.path = "/tmp/hp_dryrun_test_snap"
        HyperModelsDryRunRunner(cfg, rt)._validate_config()

    def test_device_resolution_and_simulation_device_selection(self):
        """Exercise _resolve_target_device, _select_simulation_device, and _simulation_torch_device."""
        cfg = _config()
        runner = HyperModelsDryRunRunner(cfg, DryRunRuntime(0, 1, 1))
        cfg.dry_run.target_device = "invalid"
        with patch("hyper_parallel.tools.dryrun.runner.get_device_type", return_value="cuda"):
            self.assertEqual(runner._resolve_target_device(), "cuda")
        cfg.dry_run.target_device = "invalid"
        with patch("hyper_parallel.tools.dryrun.runner.get_device_type", return_value="cpu"):
            with self.assertRaisesRegex(RuntimeError, "requires dry_run.target_device"):
                runner._resolve_target_device()

        with patch.object(torch.version, "cuda", None):
            self.assertEqual(runner._select_simulation_device("cuda"), "cpu")

        fake_cuda = SimpleNamespace(device_count=lambda: 0)
        with (
            patch.object(torch.version, "cuda", "12.0"),
            patch.object(torch, "cuda", fake_cuda),
        ):
            self.assertEqual(runner._select_simulation_device("cuda"), "cpu")

        fake_cuda_err = SimpleNamespace(device_count=MagicMock(side_effect=RuntimeError("boom")))
        with (
            patch.object(torch.version, "cuda", "12.0"),
            patch.object(torch, "cuda", fake_cuda_err),
        ):
            self.assertEqual(runner._select_simulation_device("cuda"), "cpu")

        runner._simulation_device = "cuda"
        self.assertEqual(runner._simulation_torch_device(), torch.device("cuda", 1))
        runner._simulation_device = "cpu"
        self.assertEqual(runner._simulation_torch_device(), torch.device("cpu"))

    def test_fake_pg_mesh_build_prewarm_and_flat_buffer_helpers(self):
        """Exercise fake process group, distributed setup, materialization, and flat buffers."""
        cfg = _config()
        cfg.accelerator.tp_size = 2
        cfg.accelerator.cp_size = 2
        cfg.accelerator.ep_size = 2
        cfg.fsdp_config.dp_shard_size = 2
        cfg.fsdp_config.edp_shard_size = 2
        rt = DryRunRuntime(rank=0, world_size=8, local_rank=0)
        runner = HyperModelsDryRunRunner(cfg, rt)

        if dist.is_initialized():
            dist.destroy_process_group()
        try:
            runner._init_fake_process_group(rt)
            with self.assertRaisesRegex(RuntimeError, "already|before any process group"):
                runner._init_fake_process_group(rt)
            setup = runner._build_distributed_setup(rt, "cpu")
            self.assertEqual(setup.mesh_context.dp_size, 2)
            self.assertEqual(setup.mesh_context.dp_replicate_size, 2)
            with runner._dry_run_fsdp_device():
                pass
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

        plain_model = torch.nn.Linear(2, 2, device="meta")
        with FakeTensorMode(allow_non_fake_inputs=True):
            materialized = runner._materialize_fake_model(plain_model)
            self.assertTrue(hasattr(materialized.weight, "fake_mode"))
            # Second pass hits the fake_parameter-present branch
            materialized = runner._materialize_fake_model(materialized)
            self.assertTrue(hasattr(materialized.weight, "fake_mode"))

        bucket = MagicMock()
        param_group = SimpleNamespace(
            enable_zero_copy=True,
            _init_all_gather_buckets=MagicMock(),
            all_gather_buckets=[bucket],
            device=torch.device("cpu"),
        )
        hsdp_state = SimpleNamespace(lazy_init=MagicMock(), param_group=param_group)
        host_mod = torch.nn.Linear(2, 2)
        host_mod.hsdp_scheduler = SimpleNamespace(hsdp_state=hsdp_state)
        runner._initialize_flat_buffers(host_mod)
        hsdp_state.lazy_init.assert_called_once()
        param_group._init_all_gather_buckets.assert_called_once()
        bucket.init_flat_param_buffer.assert_called_once_with(torch.device("cpu"))

    def test_data_helpers_metadata_and_run_lifecycle(self):
        """Exercise _prepare_data_base, _read_training_batch, _next_prepared_batch, _close_data_iterator, and run()."""
        cfg = _config()
        rt = DryRunRuntime(rank=0, world_size=1, local_rank=0)
        runner = HyperModelsDryRunRunner(cfg, rt)

        # _next_prepared_batch error cases
        bad_base = MagicMock()
        bad_base.get_batch.side_effect = StopIteration()
        with self.assertRaisesRegex(ValueError, "produced no batches"):
            runner._next_prepared_batch(bad_base, iter(()))
        bad_base.get_batch.side_effect = None
        bad_base.get_batch.return_value = ("not_mapping", {})
        with self.assertRaisesRegex(ValueError, "must return model_inputs"):
            runner._next_prepared_batch(bad_base, iter(()))

        # _close_data_iterator
        shutdown = MagicMock()
        runner._data_iterator = SimpleNamespace(_shutdown_workers=shutdown)
        runner._close_data_iterator()
        shutdown.assert_called_once()
        self.assertIsNone(runner._data_iterator)

        # _tensor_leaves & _batch_shape_metadata with dataclass/list/tuple/scalar
        sample = _MetaBatchSample(tensor=torch.ones(2, 3), note="ok")
        leaves = runner._tensor_leaves({"dc": sample, "seq": [torch.zeros(1), (torch.ones(2), 5)]})
        self.assertEqual(len(leaves), 3)
        meta = runner._batch_shape_metadata({"dc": sample, "seq": [torch.zeros(1), "txt"]})
        self.assertEqual(meta["dc"]["tensor"]["shape"], [2, 3])
        self.assertEqual(meta["dc"]["note"], "ok")
        self.assertEqual(meta["seq"][1], "txt")

        # _existing_tensors with chained_optimizers
        opt = MagicMock()
        opt.param_groups = [{"params": [torch.ones(2)]}]
        opt.state = {0: {"exp_avg": torch.zeros(2)}}
        base_stub = SimpleNamespace(
            model=torch.nn.Linear(2, 2),
            optimizer=SimpleNamespace(chained_optimizers=[opt]),
        )
        batch_stub = _DryRunTrainingBatch(
            model_inputs={"x": torch.ones(1)},
            loss_inputs={"y": torch.ones(1)},
            token_counts={},
            valid_token_count=1,
            target_tokens_per_rank=(1,),
        )
        existing = list(runner._existing_tensors(base_stub, batch_stub))
        self.assertGreaterEqual(len(existing), 4)

        # _run_simulation error wrapping (plain error + unsupported operator hint)
        with (
            patch.object(runner, "_init_fake_process_group", side_effect=RuntimeError("boom")),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed during distributed_setup: boom"):
                runner._run_simulation(rt, batch_stub, FakeTensorMode(), None)

        with (
            patch.object(runner, "_init_fake_process_group", side_effect=NotImplementedError("op::missing")),
            patch("hyper_parallel.tools.dryrun.runner.unsupported_operator_hint", return_value="register op"),
        ):
            with self.assertRaisesRegex(RuntimeError, "register op"):
                runner._run_simulation(rt, batch_stub, FakeTensorMode(), None)

        # run() end-to-end orchestration with mocked probe & simulation
        with (
            patch.object(runner, "_resolve_target_device", return_value="npu"),
            patch.object(runner, "_select_simulation_device", return_value="cpu"),
            patch.object(runner, "_probe_training_batch", return_value=batch_stub),
            patch.object(runner, "_run_simulation", return_value={"summary": {"peak_bytes": 123}}),
            patch("hyper_parallel.tools.dryrun.runner.simulate_device_runtime"),
            patch("hyper_parallel.tools.dryrun.runner.release_simulated_stream_holds") as mock_release,
        ):
            result = runner.run()
        self.assertEqual(result["summary"]["peak_bytes"], 123)
        mock_release.assert_called_once()

        # _prepare_data_base & _read_training_batch
        from hyper_parallel.trainer.base import BaseTrainer  # pylint: disable=import-outside-toplevel
        from hyper_parallel.trainer.text_trainer import TextTrainer  # pylint: disable=import-outside-toplevel

        cfg_data = _config()
        cfg_data.training.seed = None
        runner_data = HyperModelsDryRunRunner(cfg_data, runtime=rt)
        mesh_ctx = SimpleNamespace(device_mesh="dm", dp_cp_mesh="dp_cp")
        setup_stub = SimpleNamespace(mesh_context=mesh_ctx)
        model_stub = SimpleNamespace(config="mcfg")
        base_prepared = runner_data._prepare_data_base(setup_stub, model_stub)
        self.assertEqual(base_prepared.device, torch.device("cpu"))
        self.assertEqual(base_prepared.model_config, "mcfg")

        with (
            patch.object(TextTrainer, "_build_model_assets"),
            patch.object(TextTrainer, "_build_data_transform"),
            patch.object(TextTrainer, "_build_collate_fn"),
            patch.object(TextTrainer, "_build_get_batch"),
            patch.object(BaseTrainer, "_build_dataset"),
            patch.object(
                BaseTrainer,
                "_build_dataloader",
                autospec=True,
                side_effect=lambda self_base: setattr(self_base, "train_dataloader", [{"input_ids": torch.ones(1)}]),
            ),
            patch.object(runner_data, "_next_prepared_batch", return_value=batch_stub) as mock_next,
        ):
            prepared = runner_data._read_training_batch(setup_stub, model_stub)
        self.assertIs(prepared, batch_stub)
        mock_next.assert_called_once()
        with (
            patch.object(TextTrainer, "_build_model_assets"),
            patch.object(TextTrainer, "_build_data_transform"),
            patch.object(TextTrainer, "_build_collate_fn"),
            patch.object(TextTrainer, "_build_get_batch"),
            patch.object(BaseTrainer, "_build_dataset"),
            patch.object(BaseTrainer, "_build_dataloader"),
        ):
            with self.assertRaisesRegex(ValueError, "non-empty train dataloader"):
                runner_data._read_training_batch(setup_stub, model_stub)

    def test_cli_helpers_and_main_entry(self):
        """Exercise __main__ config/env helpers and main() under torchrun env."""
        self.assertIsInstance(dryrun_main._resolve_config({"enabled": True}), DryRunConfig)
        with self.assertRaisesRegex(TypeError, "dry_run mapping"):
            dryrun_main._resolve_config("bad")
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(dryrun_main._torchrun_identity())
        with self.assertRaisesRegex(ValueError, "WORLD_SIZE"):
            dryrun_main._configured_ranks(DryRunConfig(enabled=True), 0)

        cfg = _config()
        cfg.dry_run.enabled = False
        with patch.object(dryrun_main, "parse_training_args", return_value=cfg):
            dryrun_main.main()

        cfg = _config()
        cfg.dry_run.ranks = [0]
        with (
            patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}, clear=False),
            patch.object(dryrun_main, "parse_training_args", return_value=cfg),
            patch.object(dryrun_main, "HyperModelsDryRunRunner") as mock_runner_cls,
        ):
            mock_runner_cls.return_value.run.return_value = {"summary": {"peak_bytes": 1024}}
            dryrun_main.main()
            mock_runner_cls.return_value.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
