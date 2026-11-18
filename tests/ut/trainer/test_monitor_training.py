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
"""Twenty-step numerical and logging acceptance through the real TextTrainer loop."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch
from torch.nn import functional
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from hyper_parallel.components.modules.moe import MoE
from hyper_parallel.components.monitor import report_statistics
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.callbacks.environ_meter_callback import EnvironMeterCallback
from hyper_parallel.trainer.callbacks.logging_callback import LoggingCallback
from hyper_parallel.trainer.config.monitor import MonitorConfig, SamplingConfig
from hyper_parallel.trainer.config.training import TrainingConfig
from hyper_parallel.trainer.runtime.monitor.flops import DevicePeak
from hyper_parallel.trainer.runtime.monitor.session import MonitorSession
from hyper_parallel.trainer.state import TrainerState
from hyper_parallel.trainer.text_trainer import TextTrainer
from tests.common.mark_utils import arg_mark


class TinyTextMoE(torch.nn.Module):
    """Small real dropless MoE, suitable for CPU gradient-invariance tests."""

    def __init__(self) -> None:
        """Build a small deterministic MoE for trainer acceptance."""
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.moe = MoE(dim=8, hidden_dim=12, num_experts=4, top_k=2)
        self.head = torch.nn.Linear(8, 16)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """Return logits with actual routing, expert and projection computation."""
        del use_cache
        return self.head(self.moe(self.embedding(input_ids)))


def tiny_loss(model_output: Any, labels: torch.Tensor) -> torch.Tensor:
    """Compute the same causal target alignment as the generic token counter."""
    model_output = getattr(model_output, "logits", model_output)
    return functional.cross_entropy(model_output[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1))


def tiny_hybrid_moe() -> torch.nn.Module:
    """Build the same GDN/full-attention/shared-expert structure as the Qwen3.6 text recipe."""
    config = Qwen3_5MoeTextConfig.from_dict({
        "vocab_size": 16,
        "hidden_size": 16,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 3,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "attn_output_gate": True,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
    })
    return Qwen3_5MoeForCausalLM(config)


def cpu_batch(iterator: object) -> tuple[dict, dict]:
    """Provide canonical CPU data and explicitly instrument the custom data target."""
    batch = next(iterator)
    report_statistics("dataset", None, batch=batch)
    return {"input_ids": batch["input_ids"]}, {"labels": batch["labels"]}


class TestMonitorTraining(unittest.TestCase):
    """Verify actual loop ordering, metric counts and unchanged optimizer results."""

    @staticmethod
    def _run(
        directory: str, enabled: bool, logging_steps: int, model_factory: Callable = TinyTextMoE, mfu: bool = False,
        configure: Callable | None = None, step_metrics: list | None = None,
    ) -> tuple[dict, list]:
        torch.manual_seed(123)
        config = MonitorConfig(enabled=enabled, sinks=["jsonl"], output_dir=directory)
        config.mfu = mfu
        config.local_loss = True
        config.local_grad_norm = True
        config.moe = replace(config.moe, enabled=True, sampler_steps=1, record_distribution=True)
        config.datasets = replace(config.datasets, sampler_steps=1)
        base = BaseTrainer.__new__(BaseTrainer)
        base.config = SimpleNamespace(
            monitor=config,
            training=TrainingConfig(logging_steps=logging_steps, log_tflops=mfu),
            fsdp_config=FSDP2Config(),
        )
        base.model = model_factory()
        base.model_config = getattr(base.model, "config", None)
        base.loss_fn = tiny_loss
        base.optimizer = torch.optim.SGD(base.model.parameters(), lr=0.01)
        base.lr_scheduler = None
        base.model_integration = Mock(runtime_enabled=False)
        base.graph_compiler = None
        base.model_fwd_context = nullcontext()
        base.model_bwd_context = nullcontext()
        base.mesh = SimpleNamespace(dp_cp_mesh=None, dp_size=1, cp_size=1, dp_replicate_size=1, sequence_parallel=False)
        base.hsdp_model_parts = []
        base.fsdp_runtime_diagnostics = None
        base.device = torch.device("cpu")
        base.global_rank, base.world_size = 0, 1
        base.num_micro_batches = 2
        base.state = TrainerState()
        base.step_train_metrics = {}
        base.get_batch = cpu_batch
        if configure is not None:
            configure(base)
        base.monitor = MonitorSession(base)
        base.monitor.initialize()
        base.monitor.bind_model()
        if mfu:
            base.monitor.peak = DevicePeak("test-device", "float32", 1.0, "test-fixture")
        base.environ_meter_callback = EnvironMeterCallback(base)
        base.logging_callback = LoggingCallback(base)
        base._callbacks = [base.environ_meter_callback, base.logging_callback]
        trainer = TextTrainer.__new__(TextTrainer)
        trainer.base = base
        generator = torch.Generator().manual_seed(56)
        batches = [
            {
                "input_ids": torch.randint(0, 16, (1, 6), generator=generator),
                "labels": torch.randint(0, 16, (1, 6), generator=generator),
            }
            for _ in range(40)
        ]
        with (
            patch("hyper_parallel.trainer.runtime.metrics.all_reduce", side_effect=lambda data, **_: data),
            patch("hyper_parallel.trainer.text_trainer.synchronize"),
        ):
            base.on_train_begin()
            iterator = iter(batches)
            try:
                for _ in range(20):
                    trainer.train_step(iterator)
                    if step_metrics is not None:
                        step_metrics.append(dict(base.step_train_metrics))
            finally:
                base.on_train_end()
        records = []
        if enabled:
            path = base.monitor.writer.directory / "rank_0.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
        return {key: value.clone() for key, value in base.model.state_dict().items()}, records

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="dryrun", essential_mark="essential")
    def test_twenty_step_output_cadence_and_numerical_parity(self) -> None:
        """
        Feature: Full monitor lifecycle.
        Description: Train a real tiny MoE for 20 optimizer steps with two microbatches.
        Expectation: Identical weights with monitoring off/on, exactly 20 or four train_step events.
        """
        with tempfile.TemporaryDirectory() as directory:
            reference, _ = self._run(directory, False, 1)
            actual, records = self._run(directory, True, 1)
            _, sparse_records = self._run(directory, True, 5)
        for key in reference:
            torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
        steps = [record for record in records if record["event"] == "train_step"]
        self.assertEqual([record["step"] for record in steps], list(range(1, 21)))
        self.assertEqual(
            [record["step"] for record in sparse_records if record["event"] == "train_step"], [5, 10, 15, 20]
        )
        self.assertEqual(steps[-1]["metrics"]["data/consumed_tokens"], 200)
        self.assertEqual(steps[-1]["metrics"]["data/consumed_samples"], 40)
        self.assertEqual(sum(record["event"] == "local_micro_loss" for record in records), 40)
        self.assertEqual(sum(record["event"] == "local_step_loss" for record in records), 20)
        self.assertEqual(sum(record["event"] == "local_grad_norm" for record in records), 20)
        self.assertEqual(sum(record["event"] == "expert_load" for record in records), 20)
        self.assertEqual(sum(record["event"] == "datasets" for record in records), 20)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="dryrun", essential_mark="essential")
    def test_twenty_step_hybrid_moe_reports_complete_mfu_without_changing_training(self) -> None:
        """Feature: Hybrid MFU.
        Description: Run real GDN training with monitor enabled and disabled.
        Expectation: Complete coverage and exact metrics and weight parity.
        """
        reference_metrics, actual_metrics = [], []
        with tempfile.TemporaryDirectory() as directory:
            reference, _ = self._run(directory, False, 1, tiny_hybrid_moe, mfu=True, step_metrics=reference_metrics)
            actual, records = self._run(directory, True, 1, tiny_hybrid_moe, mfu=True, step_metrics=actual_metrics)
        self.assertEqual(len(reference_metrics), 20)
        self.assertEqual(actual_metrics, reference_metrics)
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)
        coverage = [record for record in records if record["event"] == "flops_coverage"]
        self.assertEqual(len(coverage), 20)
        self.assertTrue(all(record["complete"] for record in coverage))
        self.assertTrue(all(sum(name.endswith("delta_rule") for name in record["blocks"]) == 3 for record in coverage))
        for record in records:
            if record["event"] != "train_step":
                continue
            metrics = record["metrics"]
            self.assertGreater(metrics["performance/tflops_per_device"], 0)
            self.assertGreater(metrics["performance/mfu_percent"], 0)
            self.assertAlmostEqual(metrics["performance/mfu_percent"], 100 * metrics["performance/tflops_per_device"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="dryrun", essential_mark="essential")
    def test_hybrid_sampling_and_disabled_data_fields(self) -> None:
        """Feature: Monitor sampling.
        Description: Local loss/norm follow logging cadence while dataset sampling remains independent.
        Expectation: Exact sampled steps, one norm per rank and disabled-field absence.
        """
        def configure(base: BaseTrainer) -> None:
            """Select sampling cadences and disable optional dataset fields."""
            config = base.config.monitor
            config.timing = False
            config.datasets = replace(config.datasets, sampler_steps=5, token_distribution=False,
                                      top_n_tokens=0, packing_stats=False, packing_boundary_check=False)

        with tempfile.TemporaryDirectory() as directory:
            _, records = self._run(directory, True, 4, tiny_hybrid_moe, configure=configure)
        expected = {"train_step": range(4, 21, 4), "local_step_loss": range(4, 21, 4),
                    "local_grad_norm": range(4, 21, 4), "datasets": range(5, 21, 5)}
        for event, steps in expected.items():
            with self.subTest(event=event):
                self.assertEqual([r["step"] for r in records if r["event"] == event], list(steps))
        for record in records:
            if record["event"] == "local_grad_norm":
                self.assertEqual(record["scope"], "rank")
                self.assertNotIn("module", record)
                self.assertGreater(record["values"][0], 0)
            if record["event"] == "datasets":
                absent = {"input_lengths", "top_tokens", "padding_fraction", "packing_failures"}
                self.assertTrue(absent.isdisjoint(record))
            self.assertNotEqual(record["event"], "timing")
            self.assertFalse(any(key.startswith("timing/") for key in record.get("metrics", {})))
        self.assertTrue(any(r.get("metric") == "expert_load" and r["event"] == "unavailable" for r in records))
        self.assertFalse(any(r["event"] == "expert_load" for r in records))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="dryrun", essential_mark="essential")
    def test_hybrid_loss_data_and_unclipped_norm_match_independent_values(self) -> None:
        """Feature: Monitor numerical correctness.
        Description: Independently recompute loss, token counts and FP64 gradient norms.
        Expectation: Recorded summaries match independent values.
        """
        expected_batches, expected_norms = [], []

        def configure(base: BaseTrainer) -> None:
            """Capture source batches and disable clipping for independent norm checks."""
            base.config.training.max_grad_norm = 0

            def get_batch(iterator: object) -> tuple[dict, dict]:
                """Record each source batch before the trainer consumes it."""
                batch = next(iterator)
                expected_batches.append(batch["input_ids"].clone())
                report_statistics("dataset", None, batch=batch)
                return {"input_ids": batch["input_ids"]}, {"labels": batch["labels"]}

            base.get_batch = get_batch

        original = MonitorSession.record_gradients

        def capture_norm(session: MonitorSession, gradient_views: dict | None = None) -> None:
            """Compare the recorder against an independent concatenated FP64 norm."""
            values = [parameter.grad.detach().double().reshape(-1)
                      for parameter in session.trainer.model.parameters() if parameter.grad is not None]
            expected_norms.append(torch.cat(values).norm().item())
            original(session, gradient_views)

        with tempfile.TemporaryDirectory() as directory, patch.object(MonitorSession, "record_gradients", capture_norm):
            _, records = self._run(directory, True, 1, tiny_hybrid_moe, configure=configure)
        norms = [r["values"][0] for r in records if r["event"] == "local_grad_norm"]
        torch.testing.assert_close(torch.tensor(norms), torch.tensor(expected_norms), rtol=1e-5, atol=1e-7)
        self.assertEqual(len(norms), 20)
        self._assert_loss_and_tokens(records, expected_batches)

    def _assert_loss_and_tokens(self, records: list, expected_batches: list) -> None:
        """Check token-weighted losses and independently counted input tokens."""
        for step in range(1, 21):
            micros = [r["values"] for r in records if r["event"] == "local_micro_loss" and r["step"] == step]
            loss = next(r for r in records if r["event"] == "local_step_loss" and r["step"] == step)
            self.assertAlmostEqual(loss["loss"], sum(value * count for value, count in micros) /
                                   sum(count for _, count in micros), places=6)
            data = next(r for r in records if r["event"] == "datasets" and r["step"] == step)
            self._assert_token_histogram(data, expected_batches[2 * step - 2:2 * step])

    def _assert_token_histogram(self, data: dict, batches: list) -> None:
        """Compare dataset summaries against independent source-token counts."""
        histogram = torch.bincount(torch.cat(batches).reshape(-1), minlength=16)
        self.assertEqual({row["token_id"]: row["count"] for row in data["top_tokens"]},
                         {index: count for index, count in enumerate(histogram.tolist()) if count})
        self.assertEqual(data["counts"]["input_tokens"], 12)
        self.assertEqual(data["counts"]["valid_tokens"], 10)
        self.assertIn("packing_metadata", data["unavailable"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="dryrun", essential_mark="essential")
    def test_hybrid_mfu_and_tflops_switches_are_independent(self) -> None:
        """Feature: FLOPs switches.
        Description: Exercise all MFU and TFLOP/s combinations without a hardware peak.
        Expectation: Independent switches and explicit unavailable MFU percentage.
        """
        for mfu in (False, True):
            for tflops in (False, True):
                with self.subTest(mfu=mfu, tflops=tflops), tempfile.TemporaryDirectory() as directory:
                    def configure(base: BaseTrainer, use_mfu: bool = mfu, use_tflops: bool = tflops) -> None:
                        """Set each FLOPs switch independently."""
                        base.config.monitor.mfu = use_mfu
                        base.config.training.log_tflops = use_tflops

                    with patch.dict("os.environ", {"HYPER_PARALLEL_MONITOR_PEAK": ""}):
                        _, records = self._run(directory, True, 1, tiny_hybrid_moe, configure=configure)
                    coverage = [r for r in records if r["event"] == "flops_coverage"]
                    self.assertEqual(len(coverage), 20 if mfu or tflops else 0)
                    for record in records:
                        if record["event"] != "train_step":
                            continue
                        metrics = record["metrics"]
                        self.assertEqual("performance/tflops_per_device" in metrics, tflops)
                        self.assertEqual("performance/mfu_status" in metrics, mfu)
                        self.assertNotIn("performance/mfu_percent", metrics)
                        if mfu:
                            self.assertEqual(metrics["performance/mfu_status"], "unavailable_peak")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level1",
              card_mark="dryrun", essential_mark="essential")
    def test_hybrid_muon_ns_records_do_not_change_updates(self) -> None:
        """Feature: Muon NS monitoring.
        Description: Update Qwen matrices with Muon and other shapes with SGD.
        Expectation: Valid sampled NS owners and exact final weights with monitor on or off.
        """
        def configure(base: BaseTrainer) -> None:
            """Use Muon for matrices and sample its NS statistics every second step."""
            matrices = [parameter for parameter in base.model.parameters() if parameter.ndim == 2]
            others = [parameter for parameter in base.model.parameters() if parameter.ndim != 2]
            base.optimizer = [Muon(matrices, lr=0.001), torch.optim.SGD(others, lr=0.001)]
            base.config.monitor.optimizer.ns_norm = SamplingConfig(True, 2)

        with tempfile.TemporaryDirectory() as directory:
            reference, _ = self._run(directory, False, 1, tiny_hybrid_moe, configure=configure)
            actual, records = self._run(directory, True, 1, tiny_hybrid_moe, configure=configure)
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)
        ns = [record for record in records if record["event"] == "ns_norm"]
        self.assertEqual({record["step"] for record in ns}, set(range(2, 21, 2)))
        self.assertTrue(all(math.isfinite(value) for record in ns for value in record["values"]))
        self.assertTrue(all(owner["parameter"] in actual for record in ns for owner in record["matrix_owners"]))
