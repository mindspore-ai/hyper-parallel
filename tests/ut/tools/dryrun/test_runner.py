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

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import torch
from torch.nn import functional
from torch._subclasses.fake_tensor import FakeTensorMode

from hyper_parallel.models._transformers import model_builder
from hyper_parallel.tools.dryrun.config import DryRunConfig
from hyper_parallel.tools.dryrun.__main__ import _configured_ranks
from hyper_parallel.tools.dryrun.runtime import (
    DryRunBatchMocker,
    DryRunRuntime,
    ValueDependencyManager,
    _DryRunValueProfile,
    _MoERoutingValueDependencyRuntime,
    derive_tp_target_counts,
)
from hyper_parallel.trainer.config import (
    OptimizerConfig,
    Target,
    TrainerConfig,
    TrainingConfig,
)
from hyper_parallel.tools.dryrun.data import DryRunDataProbe
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


def _profile(*rules: dict[str, Any]) -> _DryRunValueProfile:
    """Build a value-dependency profile from rule dictionaries."""
    return _DryRunValueProfile(SimpleNamespace(value_dependencies={"rules": list(rules)}))


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

        config = _config()
        config.activation_swap = "attention"
        with self.assertRaisesRegex(NotImplementedError, "does not simulate activation swap"):
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
            "virtual_world_size": 1,
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

    def test_validate_rejects_rank_outside_virtual_world(self):
        """Configured ranks must exist in the logical topology."""
        runtime = DryRunRuntime(rank=0, world_size=2, local_rank=0)
        config = _config(DryRunConfig(ranks=[0, 2], virtual_world_size=2))
        with self.assertRaisesRegex(ValueError, "dry_run.ranks"):
            HyperModelsDryRunRunner(config, runtime)._validate_config()

    def test_cli_rank_defaults_and_validation(self):
        """CLI rank selection defaults to rank zero and rejects invalid entries."""
        self.assertEqual(_configured_ranks(DryRunConfig(), 8), [0])
        self.assertEqual(_configured_ranks(DryRunConfig(ranks=[0, 7]), 8), [0, 7])
        with self.assertRaisesRegex(ValueError, "dry_run.ranks"):
            _configured_ranks(DryRunConfig(ranks=[8]), 8)


class TestDryRunModelAndData(unittest.TestCase):
    """Tests for meta-model creation and normal-data reuse."""

    def test_dry_run_builds_the_configured_target_with_distributed_setup(self):
        """Use arbitrary configured factories instead of reparsing Hugging Face paths."""
        captured = {}

        def from_config_factory(distributed_setup: Any = None, **kwargs: Any) -> torch.nn.Module:
            """Capture build options and create a deferred meta model."""
            captured.update(kwargs)
            captured["distributed_setup"] = distributed_setup
            captured["deferred"] = model_builder.is_model_materialization_deferred()
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
        self.assertFalse(model_builder.is_model_materialization_deferred())

    def test_data_probe_reuses_trainer_build_and_reads_first_batch(self):
        """Read one batch through the normal Trainer data construction sequence."""
        labels = torch.tensor([[0, 1, -100, 3]])
        loss_mask = torch.tensor([[1, 1, 0, 1]])
        base = MagicMock()
        base.train_dataloader = [object()]
        base.get_batch.return_value = (
            {"input_ids": torch.tensor([[0, 1, 2, 3]])},
            {"labels": labels, "loss_mask": loss_mask},
        )
        probe = DryRunDataProbe(base)

        with (
            patch.object(type(probe._trainer), "_build_model_assets"),
            patch.object(type(probe._trainer), "_build_data_transform"),
            patch.object(type(probe._trainer), "_build_collate_fn"),
            patch.object(type(probe._trainer), "_build_get_batch"),
        ):
            probe.build()
            prepared = probe.read_first_batch()

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
        model = _named_model(("mlp", module))

        automatic = _profile()
        automatic.bind_model(model)
        automatic_plan = automatic.moe_routing_plan("mlp", module, 2, 0, 4, 2)
        self.assertEqual(automatic_plan.input_split_sizes, (4, 4))

        balanced = _profile({"match": "mlp", "handler": "moe_routing", "path": "balanced"})
        balanced.bind_model(model)
        balanced_plan = balanced.moe_routing_plan("mlp", module, 2, 0, 4, 2)
        self.assertEqual(balanced_plan.input_split_sizes, (4, 4))

        explicit = _profile({
            "match": "mlp",
            "handler": "moe_routing",
            "path": "explicit",
            "inputs": {"source_expert_loads": [[4, 4, 0, 0], [0, 0, 4, 4]]},
        })
        explicit.bind_model(model)
        explicit_plan = explicit.moe_routing_plan("mlp", module, 2, 1, 4, 2)
        self.assertEqual(explicit_plan.input_split_sizes, (0, 8))
        self.assertEqual(explicit_plan.output_split_sizes, (0, 8))

    def test_local_moe_replay_accepts_model_specific_keyword_arguments(self):
        """Ignore keyword-only model context that does not change routed shapes."""
        module = torch.nn.Module()
        module.experts = SimpleNamespace(w1=None, w2=None, w3=None)
        module.gate = SimpleNamespace()
        model = torch.nn.Module()
        model.mlp = module
        profile = _profile({"match": "mlp", "handler": "moe_routing", "path": "balanced"})
        profile.bind_model(model)
        base = SimpleNamespace(model=model, mesh=SimpleNamespace(ep_size=1, ep_rank=0))
        runtime = _MoERoutingValueDependencyRuntime(profile, base)
        expected = torch.empty(1, 2, 4)
        runtime._run_moe_forward = MagicMock(return_value=expected)

        self.assertEqual(runtime._ep_group_identity(), (1, 0))
        with runtime:
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

        actual = _MoERoutingValueDependencyRuntime._run_eager_experts(
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
        from hyper_parallel.trainer.base import BaseTrainer
        from hyper_parallel.trainer.runtime.model_integration import (
            DisabledModelIntegrationSession,
        )

        runner = HyperModelsDryRunRunner(_config(), DryRunRuntime(0, 1, 0))
        runner._simulation_device = "cpu"
        runner._target_device = "cuda"
        profile = _profile()

        with FakeTensorMode(allow_non_fake_inputs=True):
            model = _TinyModel()
            profile.bind_model(model)
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
            dependencies = ValueDependencyManager(profile, model, DryRunRuntime(0, 1, 0))
            report = runner._execute_step(base, batch, profile, dependencies)

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

if __name__ == "__main__":
    unittest.main()
