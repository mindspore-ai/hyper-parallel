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

import inspect
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.mem_tracker import MemTracker

from examples.training_demo import train_text
from hyper_parallel.models._transformers import model_builder
from hyper_parallel.platform.torch.dry_run import (
    DryRunBatchMocker,
    DryRunRuntime,
    UnconfiguredValueDependencyError,
    ValueDependencyManager,
    _DryRunValueProfile,
    _create_indexed_mem_tracker,
    _create_operator_trace_mode,
    _normalize_snapshot,
    build_memory_report,
    derive_tp_target_counts,
    value_dependency_decision,
)
from hyper_parallel.trainer.config import (
    DryRunConfig,
    OptimizerConfig,
    Target,
    TrainerConfig,
    TrainingConfig,
)
from hyper_parallel.trainer.dry_run import HyperModelsDryRunRunner, _DryRunTrainingBatch
from hyper_parallel.trainer.dry_run_data import DryRunDataProbe


def _target(**kwargs: Any) -> dict[str, Any]:
    """Provide a callable placeholder for validation-only configurations."""
    return kwargs


def _config(dry_run: Optional[DryRunConfig] = None) -> TrainerConfig:
    """Build a minimal Trainer configuration for dry-run tests."""
    return TrainerConfig(
        model=Target(_target, target_path="tests.model", path="unused"),
        optimizer=OptimizerConfig(Target(_target, target_path="tests.optimizer")),
        training=TrainingConfig(global_batch_size=1, micro_batch_size=1, backend="gloo"),
        dry_run=dry_run or DryRunConfig(sequence_length=8),
    )


def _named_model(*modules: tuple[str, Any]) -> Any:
    """Expose the named-module surface consumed by value-dependency rules."""
    return SimpleNamespace(named_modules=lambda: [("", SimpleNamespace()), *modules])


def _profile(*rules: dict[str, Any]) -> _DryRunValueProfile:
    """Build a value-dependency profile from rule dictionaries."""
    return _DryRunValueProfile(DryRunConfig(value_dependencies={"rules": list(rules)}))


class _ScalarBranch(torch.nn.Module):
    """Small module containing a FakeTensor data-dependent branch."""

    def forward(self, value: torch.Tensor) -> bool:
        """Return a Python value derived from a tensor scalar."""
        return bool((value.sum() > 0).item())


class _ConfiguredBranch(torch.nn.Module):
    """Small module using the public semantic branch API."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Select a branch without evaluating a FakeTensor scalar."""
        enabled = value_dependency_decision(
            "use_positive_path",
            lambda: bool((value.sum() > 0).item()),
        )
        return value + 1 if enabled else value - 1


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


class TestDryRunConfiguration(unittest.TestCase):
    """Tests for dry-run validation and Trainer entrypoint routing."""

    def test_validate_accepts_accelerator_targets_and_rejects_disabled_mode(self):
        """Accept supported report targets and require dry-run to be enabled."""
        runtime = DryRunRuntime(rank=0, world_size=1, local_rank=0)
        for target_device in ("npu", "cuda"):
            with self.subTest(target_device=target_device):
                runner = HyperModelsDryRunRunner(
                    _config(DryRunConfig(target_device=target_device, sequence_length=8)),
                    runtime,
                )
                self.assertEqual(runner._validate_config().target_device, target_device)

        runner = HyperModelsDryRunRunner(
            _config(DryRunConfig(enabled=False, sequence_length=8)),
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
        config.accelerator.pp_size = 2
        with self.assertRaisesRegex(
                NotImplementedError,
                "Pipeline parallel dry-run is not supported by the base implementation",
        ):
            HyperModelsDryRunRunner(config, runtime)._validate_config()

    def test_training_entrypoint_selects_dry_run_or_regular_trainer(self):
        """Route enabled dry-run configurations away from normal training."""
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                config = _config(DryRunConfig(enabled=enabled))
                with (
                    patch.object(train_text, "parse_training_args", return_value=config),
                    patch.object(train_text, "HyperModelsDryRunRunner") as dry_runner_type,
                    patch.object(train_text, "TextTrainer") as trainer_type,
                ):
                    train_text.main()

                if enabled:
                    dry_runner_type.assert_called_once_with(config)
                    dry_runner_type.return_value.run.assert_called_once_with()
                    trainer_type.assert_not_called()
                else:
                    trainer_type.assert_called_once_with(config)
                    trainer_type.return_value.train.assert_called_once_with()
                    dry_runner_type.assert_not_called()


class TestDryRunModelAndData(unittest.TestCase):
    """Tests for meta-model creation and normal-data reuse."""

    def test_model_build_context_is_nested_and_exception_safe(self):
        """Restore normal materialization state after every scoped build."""
        self.assertFalse(model_builder.is_model_materialization_deferred())
        with model_builder.model_build_context():
            self.assertTrue(model_builder.is_model_materialization_deferred())
            with model_builder.model_build_context():
                self.assertTrue(model_builder.is_model_materialization_deferred())
            self.assertTrue(model_builder.is_model_materialization_deferred())
        self.assertFalse(model_builder.is_model_materialization_deferred())

        with self.assertRaisesRegex(RuntimeError, "build failed"):
            with model_builder.model_build_context():
                raise RuntimeError("build failed")
        self.assertFalse(model_builder.is_model_materialization_deferred())

    def test_model_build_context_skips_materialization_and_initialization(self):
        """Leave a meta model untouched while normal builds keep atomic initialization."""
        model = torch.nn.Linear(4, 4, device="meta")
        with (
            patch.object(model_builder, "_move_model_to_device", return_value=model) as move_model,
            patch.object(model_builder, "_initialize_model_weights") as initialize_weights,
        ):
            with model_builder.model_build_context():
                deferred = model_builder._materialize_and_load_model(
                    model,
                    is_meta_device=True,
                    device=torch.device("cpu"),
                    load_base_model=False,
                    pretrained_path=None,
                    weights_mapping=None,
                )
            self.assertIs(deferred, model)
            move_model.assert_not_called()
            initialize_weights.assert_not_called()

            materialized = model_builder._materialize_and_load_model(
                model,
                is_meta_device=True,
                device=torch.device("cpu"),
                load_base_model=False,
                pretrained_path=None,
                weights_mapping=None,
            )
            self.assertIs(materialized, model)
            move_model.assert_called_once()
            initialize_weights.assert_called_once_with(model)

    def test_dry_run_builds_the_configured_target_with_distributed_setup(self):
        """Use arbitrary configured factories instead of reparsing Hugging Face paths."""
        captured = {}

        def from_config_factory(distributed_setup=None, **kwargs):
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
        loss_mask = torch.tensor([[1, 1, 0]])
        base = MagicMock()
        base.train_dataloader = [object()]
        base.get_batch.return_value = (
            {"input_ids": torch.tensor([[0, 1, 2, 3]])},
            {"labels": labels, "loss_mask": loss_mask},
        )
        probe = DryRunDataProbe(base)

        with (
            patch.object(type(probe._trainer), "_build_model_assets") as build_assets,
            patch.object(type(probe._trainer), "_build_data_transform") as build_transform,
            patch.object(type(probe._trainer), "_build_collate_fn") as build_collate,
            patch.object(type(probe._trainer), "_build_get_batch") as build_get_batch,
        ):
            probe.build()
            prepared = probe.read_first_batch()

        build_assets.assert_called_once_with()
        build_transform.assert_called_once_with()
        build_collate.assert_called_once_with()
        build_get_batch.assert_called_once_with()
        base._build_dataset.assert_called_once_with()
        base._build_dataloader.assert_called_once_with()
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

    def test_branch_rule_avoids_fake_scalar_materialization(self):
        """Use a configured decision instead of evaluating a FakeTensor scalar."""
        model = torch.nn.Sequential(_ConfiguredBranch())
        profile = _profile({
            "match": "0",
            "handler": "branch",
            "path": "positive",
            "inputs": {"decisions": {"use_positive_path": True}},
        })
        profile.bind_model(model)
        manager = ValueDependencyManager(profile, model, DryRunRuntime(0, 1, 0))

        with FakeTensorMode():
            value = torch.empty(4)
            with manager.fake_step_context(SimpleNamespace(model=model)):
                result = model(value)

        self.assertEqual(result.shape, (4,))
        self.assertEqual(
            profile.metadata()["runtime"]["branch"]["consumed_decisions"],
            ["0:use_positive_path"],
        )

    def test_operator_debug_handles_configured_and_unconfigured_scalar(self):
        """Mock an anchored scalar and diagnose the same unconfigured dependency."""
        model = torch.nn.Sequential(_ScalarBranch())
        source_lines, start_line = inspect.getsourcelines(_ScalarBranch.forward)
        item_line = start_line + next(index for index, line in enumerate(source_lines) if ".item()" in line)
        rule = {
            "match": "0",
            "handler": "operator_debug",
            "path": "positive",
            "inputs": {"mocks": [{
                "source": {
                    "file": Path(__file__).name,
                    "function": "_ScalarBranch.forward",
                    "line": item_line,
                },
                "op": "aten._local_scalar_dense.default",
                "occurrence": 0,
                "return": {"scalar": 1},
            }]},
        }

        configured = _profile(rule)
        configured.bind_model(model)
        configured_manager = ValueDependencyManager(configured, model, DryRunRuntime(0, 1, 0))
        with FakeTensorMode():
            with configured_manager.fake_step_context(SimpleNamespace(model=model)):
                self.assertTrue(model(torch.empty(4)))

        unconfigured = _profile()
        unconfigured.bind_model(model)
        manager = ValueDependencyManager(unconfigured, model, DryRunRuntime(0, 1, 0))
        with FakeTensorMode():
            with self.assertRaises(UnconfiguredValueDependencyError) as error:
                with manager.fake_step_context(SimpleNamespace(model=model)):
                    model(torch.empty(4))
        self.assertIn("Target: 0", str(error.exception))
        self.assertIn("Operator: aten._local_scalar_dense.default", str(error.exception))


class TestDryRunExecution(unittest.TestCase):
    """Tests for FakeTensor execution, memory accounting, and reporting."""

    def test_fake_step_runs_forward_backward_optimizer_and_builds_report(self):
        """Execute one complete FakeTensor training step on CPU."""
        runner = HyperModelsDryRunRunner(_config(), DryRunRuntime(0, 1, 0))
        runner._simulation_device = "cpu"
        runner._target_device = "cuda"
        profile = _profile()

        with FakeTensorMode(allow_non_fake_inputs=True):
            model = _TinyModel()
            profile.bind_model(model)
            batch = _DryRunTrainingBatch(
                model_inputs={"inputs": torch.empty(1, 4)},
                loss_inputs={"labels": None},
                token_counts={"foundation_tokens": 4},
                valid_token_count=0,
                target_tokens_per_rank=(0,),
            )
            base = SimpleNamespace(
                model=model,
                loss_fn=lambda **kwargs: kwargs["model_output"].loss,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1, foreach=False),
                mesh=SimpleNamespace(
                    dp_size=1,
                    dp_replicate_size=1,
                    dp_shard_size=1,
                    tp_size=1,
                    cp_size=1,
                    ep_size=1,
                    pp_size=1,
                    loss_parallel=False,
                ),
                hsdp_model_parts=[],
                model_fwd_context=nullcontext(),
                model_bwd_context=nullcontext(),
            )
            dependencies = ValueDependencyManager(profile, model, DryRunRuntime(0, 1, 0))
            with patch("hyper_parallel.trainer.dry_run.hsdp_sync_stream"):
                report = runner._execute_step(base, batch, profile, dependencies)

        self.assertEqual(report["metadata"]["num_micro_batches"], 1)
        self.assertEqual(report["metadata"]["parallel"]["dp"], 1)
        self.assertGreater(report["summary"]["peak_bytes"], 0)
        self.assertTrue(any(device.startswith("cuda") for device in report["devices"]))
        self.assertTrue(report["memory_blocks"])

    def test_fake_dtensor_gradient_is_tracked_and_released(self):
        """Expose a wrapped FakeTensor gradient to MemTracker and then release it."""
        with FakeTensorMode():
            module = torch.nn.Module()
            parameter = torch.nn.Parameter(torch.empty(8, dtype=torch.float32))
            parameter._is_fake_wrapper = True
            parameter._local_tensor = parameter
            module.register_parameter("weight", parameter)
            tracker = _create_indexed_mem_tracker(MemTracker)
            tracker.track_external(module)
            operator_trace = _create_operator_trace_mode(tracker, "cpu")
            with tracker, operator_trace:
                gradients = tracker.refresh_parameter_gradients(module)
                operator_trace.refresh_tensor_roles(gradients)
                current = _normalize_snapshot(tracker.get_tracker_snapshot())
                self.assertEqual(sum(item.get("Gradient", 0) for item in current.values()), 32)
                del gradients
                tracker.clear_fake_dtensor_grad_bridges()
            operator_trace.finalize()

        current = _normalize_snapshot(tracker.get_tracker_snapshot("current"))
        self.assertEqual(sum(item.get("Gradient", 0) for item in current.values()), 0)

    def test_memory_report_relabels_cpu_fallback_as_target(self):
        """Keep logical accelerator reporting when simulation falls back to CPU."""
        tracker = MagicMock()
        tracker.memory_tracking = {}
        tracker.get_tracker_snapshot.side_effect = lambda kind="current": {
            "cpu:0": {"Total": 128 if kind == "peak" else 96, "Parameter": 64}
        }
        report = build_memory_report(
            tracker,
            {"target_device": "npu", "simulation_device": "cpu"},
            "cpu",
            memory_blocks=[],
            report_device_type="npu",
        )

        self.assertEqual(report["summary"]["peak_bytes"], 128)
        self.assertIn("npu:0", report["devices"])


if __name__ == "__main__":
    unittest.main()
