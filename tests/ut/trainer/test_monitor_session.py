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
"""CPU contract tests for monitor ownership, arithmetic, lifecycle and output."""

from concurrent.futures import Future
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.monitor import report_statistics, statistics_scope
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.distributed.expert_parallel import experts as ep_experts
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.callbacks.environ_meter_callback import EnvironMeterCallback
from hyper_parallel.trainer.config.monitor import MonitorConfig, SamplingConfig, GradientMonitorConfig
from hyper_parallel.trainer.runtime.monitor.datasets import DatasetStatsRecorder
from hyper_parallel.trainer.runtime.monitor.flops import BlockFlopsEstimator, DevicePeak, FlopsBlock, transformer_blocks
from hyper_parallel.trainer.runtime.monitor.output import OutputWriter, json_value
from hyper_parallel.trainer.runtime.monitor.session import MonitorSession
from hyper_parallel.trainer.runtime.monitor.transfer import SummaryTransfer
from hyper_parallel.trainer.state import TrainerState


class TestMonitorSession(unittest.TestCase):
    """Exercise sampling without modifying the numerical training path."""

    def setUp(self) -> None:
        """Allocate isolated output and a CPU-only trainer fixture."""
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.config = MonitorConfig(sinks=["jsonl"], output_dir=directory)
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
        self.trainer = SimpleNamespace(
            config=SimpleNamespace(monitor=self.config, training=SimpleNamespace(log_tflops=True)),
            model=self.model,
            model_config=None,
            global_rank=0,
            world_size=1,
            mesh=None,
        )
        self.session = MonitorSession(self.trainer)
        self.trainer.monitor = self.session
        self.session.initialize()
        self.session.bind_model()
        self.session.start_output()
        self.addCleanup(self.session.close)

    def records(self) -> list:
        """Read parseable durable records before shutdown."""
        path = self.session.writer.directory / "rank_0.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_loss_weighting_cadence_and_gradient_invariance(self) -> None:
        """Feature: Loss sampling. Description: Unequal micro lengths. Expectation: Token-weighted mean."""
        self.config.local_loss = SamplingConfig(True, 2)
        for step in (1, 2):
            self.session.begin_step(step)
            for value, count in ((2.0, 2), (5.0, 1)):
                loss = torch.tensor(value, requires_grad=True)
                self.session.record_loss(loss, {"foundation_tokens": torch.tensor(count)})
                loss.backward()
                self.assertEqual(loss.grad.item(), 1.0)
            self.session.end_step()
        records = self.records()
        micros = [r for r in records if r["event"] == "local_micro_loss"]
        steps = [r for r in records if r["event"] == "local_step_loss"]
        self.assertEqual([r["step"] for r in micros], [2, 2])
        self.assertEqual(steps[0]["loss"], 3.0)
        self.assertEqual(steps[0]["tokens"], 3.0)

    def test_gradient_main_grad_and_subtree(self) -> None:
        """Feature: Local norm. Description: Prefer main_grad. Expectation: No mutation or collective."""
        self.config.local_grad_norm = GradientMonitorConfig(True, 1, ["0"])
        self.session.bind_model()
        parameter = self.model[0].weight
        parameter.grad = torch.ones_like(parameter)
        parameter.main_grad = torch.full_like(parameter, 2.0)
        self.session.begin_step(1)
        with patch("torch.distributed.all_reduce") as collective:
            self.session.record_gradients()
        collective.assert_not_called()
        record = next(r for r in self.records() if r["event"] == "local_grad_norm")
        self.assertAlmostEqual(record["values"][0], 24**0.5, places=5)
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
        torch.testing.assert_close(parameter.main_grad, torch.full_like(parameter, 2.0))

    def test_original_forward_only_with_checkpoint_recomputation(self) -> None:
        """Feature: EP recorder. Description: Reentrant recompute. Expectation: One logical count."""
        self.config.moe = replace(self.config.moe, enabled=True, sampler_steps=1, record_distribution=True)
        self.session.expert_recorder.register("0", 4, "cpu")
        self.session.begin_step(1)
        owner = self.model[0]

        def forward(value: torch.Tensor) -> torch.Tensor:
            """Replay a routed observation during checkpoint recomputation."""
            report_statistics("expert_load", owner, counts=torch.tensor([3, 1, 0, 0]))
            return value.square()

        source = torch.ones(2, requires_grad=True)
        with statistics_scope(self.session):
            result = checkpoint(forward, source, use_reentrant=True)
        result.sum().backward()
        self.session.end_step()
        records = [r for r in self.records() if r["event"] == "expert_load"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["counts"], [3, 1, 0, 0])
        self.assertEqual(records[0]["summary"]["max_vio"], 2.0)
        self.assertEqual(records[0]["summary"]["empty_expert_fraction"], 0.5)
        torch.testing.assert_close(source.grad, torch.full_like(source, 2.0))

    def test_ns_norm_is_pre_scale_and_does_not_retain_update(self) -> None:
        """Feature: Muon. Description: Batched NS matrices. Expectation: Frobenius norm per matrix."""
        self.config.optimizer.ns_norm = SamplingConfig(True, 1)
        self.session.begin_step(1)
        update = torch.tensor([[[3.0, 4.0]], [[0.0, 2.0]]])
        self.session.record_component("ns_norm", object(), {"values": update})
        update.zero_()
        record = next(r for r in self.records() if r["event"] == "ns_norm")
        self.assertEqual(record["values"], [5.0, 2.0])
        self.assertEqual(record["stage"], "post_ns_pre_scale")

    def test_native_ep_dispatch_preserves_counts_and_missing_observations(self) -> None:
        """Feature: Native EP. Description: Two dispatch paths. Expectation: Accepted and drop counts."""
        self.config.moe = replace(self.config.moe, enabled=True, sampler_steps=2, record_distribution=True)
        owner = self.model[0]
        owner.monitor_expert_count = 2
        owner.experts = SimpleNamespace(local_expert_count=2)
        self.session.bind_model()
        self.session.expert_recorder.register("not_executed", 2, "cpu")
        hidden = torch.ones(1, 2, 3)
        dispatch = SimpleNamespace(
            expert_indices=torch.tensor([[0], [0], [1]]),
            source_indices=torch.tensor([0, 0, 1]),
            expert_weights=torch.ones(3),
            dispatch_order=torch.arange(3),
            states=torch.ones(3, 3),
            send_counts=[3],
            receive_counts=[3],
            unpermute_indices=None,
            routing_weights=None,
        )
        router = Mock(return_value=(torch.tensor([[0, 1], [0, 1]]), torch.ones(2, 2)))
        group = Mock()
        group.size.return_value = 1
        with (
            patch.object(ep_experts.dist, "get_rank", return_value=0),
            patch.object(ep_experts, "_resolve_dispatch_chunks", return_value=1),
            patch.object(ep_experts, "_resolve_capacity_factor", return_value=None),
            patch.object(ep_experts, "_prepare_ep_dispatch", return_value=dispatch),
            patch.object(ep_experts, "_run_ep_local_experts", return_value=hidden),
            patch.object(ep_experts, "_aggregate_ep_outputs", return_value=hidden),
            patch.object(ep_experts, "_FUSED_DISPATCH_ENABLED", False),
            patch.object(ep_experts, "ep_all_to_all_async", side_effect=lambda tensor, *_: tensor),
        ):
            for step in (1, 2):
                self.session.begin_step(step)
                with statistics_scope(self.session):
                    ep_experts.ep_routed_forward(owner, hidden, router_fn=router, ep_group=group)
                    ep_experts.ep_routed_dispatch(owner, hidden, router_fn=router, ep_group=group)
                self.session.end_step()
        loads = [r for r in self.records() if r["event"] == "expert_load"]
        self.assertEqual(len(loads), 1)
        self.assertEqual(loads[0]["step"], 2)
        self.assertEqual(loads[0]["counts"], [4, 2])
        self.assertEqual(loads[0]["summary"]["attempted_assignments"], 8)
        self.assertEqual(loads[0]["summary"]["dropped_assignments"], 2)
        self.assertTrue(any(r.get("module") == "not_executed" and r["event"] == "unavailable" for r in self.records()))

    def test_muon_native_and_transformed_batches_retain_parameter_identity(self) -> None:
        """Feature: NS ownership. Description: Real optimizer paths. Expectation: Pre-scale norm and name."""
        self.config.optimizer.ns_norm = SamplingConfig(True, 1)
        parameter = self.model[0].weight
        optimizer = Muon.__new__(Muon)
        optimizer.reshape_fn = None
        optimizer.zeropower_fn = lambda value, **_: torch.ones_like(value)
        group = {
            "ns_steps": 1,
            "ns_variant": "asym5",
            "ns_coefficients": None,
            "ns_epsilon": 1e-10,
            "matched_adamw_rms": 0.2,
            "zero_rms_scale_mode": "zero",
            "apply_lr_in_update": True,
            "lr": 0.01,
        }
        self.session.begin_step(1)
        for transform in (None, lambda *_: None):
            optimizer.ns_transform_fn = transform
            with statistics_scope(self.session):
                optimizer._compute_batched_ns_updates(  # pylint: disable=protected-access
                    [parameter], {parameter: torch.ones_like(parameter)}, group, no_shard=True
                )
        records = [r for r in self.records() if r["event"] == "ns_norm"]
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertAlmostEqual(record["values"][0], 6**0.5, places=5)
            self.assertEqual(
                record["matrix_owners"],
                [
                    {"parameter": "0.weight", "transform_index": 0, "start": 0, "count": 1},
                ],
            )
        self.assertEqual(self.session.ns_tensor_names, {})

    def test_checkpoint_restores_monitor_identity_and_cumulative_counts(self) -> None:
        """Feature: Resume. Description: Real extra-state serialization. Expectation: Counts survive."""
        self.trainer.state = TrainerState(global_step=12)
        self.trainer.train_dataloader = None
        self.trainer.lr_scheduler = None
        meter = EnvironMeterCallback(self.trainer)
        self.trainer.environ_meter_callback = meter
        meter.load_state_dict({"consumed_tokens": 42, "consumed_samples": 6})
        callback = CheckpointerCallback.__new__(CheckpointerCallback)
        callback.trainer = self.trainer
        with (
            patch("hyper_parallel.trainer.callbacks.checkpoint_callback.get_device_rng_state", return_value=None),
            patch("hyper_parallel.trainer.callbacks.checkpoint_callback.set_device_rng_state"),
        ):
            state = callback._collect_extra_state(self.trainer.state)  # pylint: disable=protected-access
            expected_meter = meter.state_dict()
            expected_id = self.session.run_id
            meter.load_state_dict({"consumed_tokens": 0, "consumed_samples": 0})
            self.session.load_state_dict({"run_id": "new-run"})
            callback._apply_extra_state(state)  # pylint: disable=protected-access
        self.assertEqual(meter.state_dict(), expected_meter)
        self.assertEqual(self.session.run_id, expected_id)
        self.assertEqual(self.session.step, 12)

    def test_mfu_has_explicit_peak_and_time_weighted_window(self) -> None:
        """Feature: MFU. Description: Unequal duration steps. Expectation: Sum work / sum time."""
        self.config.mfu = True
        self.session.estimator = BlockFlopsEstimator([FlopsBlock("linear", "linear", 4, 8)])
        self.session.peak = DevicePeak("test-device", "float32", 1.0, "test-fixture")
        first = self.session.flops_metrics(10, 100, 1.0)
        second = self.session.flops_metrics(10, 100, 3.0)
        self.assertAlmostEqual(first["performance/mfu_percent"], 100 * 1920 / 1e12)
        self.assertAlmostEqual(second["performance/mfu_window_percent"], 100 * 3840 / 4 / 1e12)
        self.session.peak = None
        missing = self.session.flops_metrics(10, 100, 1.0)
        self.assertNotIn("performance/mfu_percent", missing)
        self.assertEqual(missing["performance/mfu_status"], "unavailable_peak")
        with self.assertRaises(ValueError):
            DevicePeak("test-device", "float32", 1.0, "test-fixture", sparse=True)

    def test_async_checkpoint_timing_keeps_original_step(self) -> None:
        """Feature: Checkpoint timing. Description: Future finishes later. Expectation: Original step."""
        self.session.step = 3
        with patch("hyper_parallel.trainer.runtime.monitor.session.time.perf_counter", return_value=10.0):
            callback = self.session.completion("checkpoint_persist")
        future = Future()
        future.add_done_callback(callback)
        self.session.step = 8
        with patch("hyper_parallel.trainer.runtime.monitor.session.time.perf_counter", return_value=12.0):
            future.set_result(None)
        self.session.begin_step(9)
        event = next(record for record in self.records() if record.get("phase") == "checkpoint_persist")
        self.assertEqual(event["step"], 3)
        self.assertEqual(event["duration_seconds"], 2.0)

    def test_disabled_monitor_has_no_files_or_sampling(self) -> None:
        """Feature: Disabled monitor. Description: Execute lifecycle. Expectation: No output or work."""
        self.config.enabled = False
        session = MonitorSession(self.trainer)
        session.begin_step(1)
        session.record_loss(torch.tensor(2.0), {})
        session.close()
        self.assertIsNone(session.writer)
        self.assertFalse(session.in_step)
        self.assertEqual(session.end_step(), {})

    def test_resume_identity_is_validated_and_reused(self) -> None:
        """Feature: Resume. Description: Restore before sinks start. Expectation: Append same run."""
        session = MonitorSession(self.trainer)
        session.load_state_dict(self.session.state_dict())
        self.assertEqual(session.run_id, self.session.run_id)
        with self.assertRaises(ValueError):
            session.load_state_dict({"run_id": "../escape"})


class TestMonitorDataAndOutput(unittest.TestCase):
    """Check data contracts, asynchronous readiness, and sink serialization."""

    def test_batch_statistics_and_packing_failures(self) -> None:
        """Feature: Data. Description: Masked targets and packed resets. Expectation: Exact counts."""
        config = replace(MonitorConfig().datasets, top_n_tokens=2)
        recorder = DatasetStatsRecorder(config)
        batch = {
            "input_ids": torch.tensor([[1, 2, 1, 3]]),
            "attention_mask": torch.ones(1, 4),
            "shift_labels": torch.tensor([[2, 9, 3, -100]]),
            "loss_mask": torch.tensor([[1, 1, 0, 1]]),
            "cu_seq_lens": torch.tensor([0, 2, 4]),
            "position_ids": torch.tensor([[0, 1, 0, 1]]),
            "packing_policy": "isolated",
        }
        recorder.record(batch)
        snapshot = recorder.snapshot()
        self.assertEqual(snapshot["counts"]["valid_tokens"], 2)
        self.assertEqual(snapshot["top_tokens"][0], {"token_id": 1, "count": 2, "fraction": 0.5})
        self.assertEqual(snapshot["segment_lengths"]["mean"], 2.0)
        self.assertIn("cross_segment_target", snapshot["packing_failures"])
        self.assertNotIn("position_reset", snapshot["packing_failures"])

    def test_step_top_tokens_merge_before_selecting_topn(self) -> None:
        """Feature: Top tokens. Description: Micro winners differ. Expectation: Exact step winner."""
        recorder = DatasetStatsRecorder(replace(MonitorConfig().datasets, top_n_tokens=1))
        for ids in ([1, 1, 1, 9, 9], [2, 2, 2, 9, 9]):
            recorder.record({"input_ids": torch.tensor([ids])})
        self.assertEqual(recorder.snapshot()["top_tokens"][0]["token_id"], 9)
        self.assertEqual(recorder.snapshot()["top_tokens"][0]["count"], 4)

    def test_transfer_waits_for_event_and_preserves_original_metadata(self) -> None:
        """Feature: Async copies. Description: Incomplete event. Expectation: Never read early."""
        transfer = SummaryTransfer(capacity=1)
        consume = Mock()
        event = Mock()
        event.query.return_value = False
        transfer.pending.append((event, torch.tensor(1), torch.tensor([2.0]), consume))
        transfer.poll()
        consume.assert_not_called()
        self.assertFalse(transfer.submit(torch.tensor([3.0]), consume))
        self.assertEqual(transfer.dropped, 1)
        event.query.return_value = True
        transfer.poll()
        consume.assert_called_once_with([2.0])
        self.assertFalse(transfer.pending)

    def test_jsonl_rank_and_nonfinite_output(self) -> None:
        """Feature: Outputs. Description: Nonzero rank plus NaN. Expectation: JSONL only, strict JSON."""
        with tempfile.TemporaryDirectory() as directory:
            config = MonitorConfig(output_dir=directory)
            writer = OutputWriter(config, 2, "test")
            with patch("hyper_parallel.trainer.runtime.monitor.output.logger.info") as log:
                writer.emit({"event": "train_step", "step": 1, "metrics": {"loss": float("nan")}})
            writer.close()
            log.assert_not_called()
            record = json.loads((Path(directory) / "test/rank_2.jsonl").read_text())
            self.assertEqual(record["metrics"]["loss"], "nan")
        with self.assertRaises(TypeError):
            json_value(torch.tensor(1.0))

    def test_invalid_sampling_configuration_is_rejected(self) -> None:
        """Feature: YAML boundary. Description: Invalid cadence. Expectation: Clear rejection."""
        for cadence in (0, -1, True, 1.5):
            with self.subTest(cadence=cadence), self.assertRaises(ValueError):
                SamplingConfig(sampler_steps=cadence)

    def test_block_flops_composition_and_missing_coverage(self) -> None:
        """Feature: MFU. Description: Linear and attention blocks. Expectation: Recomputable formula."""
        estimator = BlockFlopsEstimator(
            [FlopsBlock("q", "linear", 4, 8), FlopsBlock("attn", "attention", heads=2, head_dim=4)]
        )
        self.assertEqual(estimator.estimate(10, 100), {"q": 1920, "attn": 4800})
        with self.assertRaises(ValueError):
            BlockFlopsEstimator([FlopsBlock("same", "linear"), FlopsBlock("same", "linear")])
        incomplete = transformer_blocks(
            SimpleNamespace(
                hidden_size=4,
                num_hidden_layers=1,
                num_attention_heads=1,
                vocab_size=8,
                layer_types=["linear_attention"],
            )
        )
        self.assertEqual(incomplete.estimate(10, 100), {})
        self.assertTrue(incomplete.missing)

    def test_optional_wandb_preserves_native_scalars_and_tables(self) -> None:
        """Feature: W&B adapter. Description: Mock SDK. Expectation: One run, native data, finish."""
        run = Mock()
        sdk = SimpleNamespace(init=Mock(return_value=run), Table=Mock(side_effect=lambda **values: values))
        with tempfile.TemporaryDirectory() as directory:
            config = MonitorConfig(sinks=["jsonl", "wandb"], output_dir=directory)
            with patch("hyper_parallel.trainer.runtime.monitor.output.importlib.import_module", return_value=sdk):
                writer = OutputWriter(config, 0, "resume-id")
                writer.emit({"event": "train_step", "step": 7, "metrics": {"training/total_loss": 2.5}})
                writer.emit(
                    {"event": "datasets", "step": 7, "top_tokens": [{"token_id": 3, "count": 4, "fraction": 0.5}]}
                )
                writer.close()
            sdk.init.assert_called_once()
            self.assertEqual(sdk.init.call_args.kwargs["id"], "resume-id")
            self.assertEqual(sdk.init.call_args.kwargs["resume"], "allow")
            run.log.assert_any_call({"global_step": 7, "training/total_loss": 2.5})
            table_call = run.log.call_args_list[-1].args[0]
            self.assertEqual(table_call["data/top_tokens_rank_0"]["data"], [[7, 3, 4, 0.5]])
            run.finish.assert_called_once()

    def test_packing_missing_and_empty_batches_are_not_success(self) -> None:
        """Feature: Data coverage. Description: Empty input and no metadata. Expectation: Unavailable."""
        recorder = DatasetStatsRecorder(MonitorConfig().datasets)
        recorder.record({"input_ids": torch.empty(1, 0, dtype=torch.int64)})
        snapshot = recorder.snapshot()
        self.assertIn("packing_metadata", snapshot["unavailable"])
        self.assertIsNone(snapshot["padding_fraction"])
        recorder.reset()
        recorder.record(
            {"input_ids": torch.empty(0, 4, dtype=torch.int64), "labels": torch.empty(0, 4, dtype=torch.int64)}
        )
        self.assertEqual(recorder.snapshot()["counts"]["valid_tokens"], 0)
        recorder.reset()
        recorder.record({"input_ids": torch.ones(1, 4, dtype=torch.int64), "cu_seq_lens": torch.tensor([0.0, 4.0])})
        self.assertIn("offset_dtype", recorder.snapshot()["packing_failures"])
        self.assertEqual(snapshot["top_tokens"], [])
