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

import json
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as functional
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from hyper_parallel.components.monitor import report_statistics
from hyper_parallel.components.modules.moe import MoE
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.callbacks.environ_meter_callback import EnvironMeterCallback
from hyper_parallel.trainer.callbacks.logging_callback import LoggingCallback
from hyper_parallel.trainer.config.monitor import MonitorConfig, SamplingConfig
from hyper_parallel.trainer.config.training import TrainingConfig
from hyper_parallel.trainer.runtime.monitor.session import MonitorSession
from hyper_parallel.trainer.runtime.monitor.flops import DevicePeak
from hyper_parallel.trainer.state import TrainerState
from hyper_parallel.trainer.text_trainer import TextTrainer


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
    config = Qwen3_5MoeTextConfig(
        vocab_size=16,
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        attn_output_gate=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
    )
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
        directory: str, enabled: bool, logging_steps: int, model_factory: Callable = TinyTextMoE, mfu: bool = False
    ) -> tuple[dict, list]:
        torch.manual_seed(123)
        config = MonitorConfig(enabled=enabled, sinks=["jsonl"], output_dir=directory)
        config.mfu = mfu
        config.local_loss = SamplingConfig(True, 1)
        config.local_grad_norm = replace(config.local_grad_norm, enabled=True, sampler_steps=1)
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
        base.get_batch = cpu_batch
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
            finally:
                base.on_train_end()
        records = []
        if enabled:
            path = base.monitor.writer.directory / "rank_0.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
        return {key: value.clone() for key, value in base.model.state_dict().items()}, records

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

    def test_twenty_step_hybrid_moe_reports_complete_mfu_without_changing_training(self) -> None:
        """Feature: Hybrid MFU. Description: Real GDN training. Expectation: Complete coverage and exact parity."""
        with tempfile.TemporaryDirectory() as directory:
            reference, _ = self._run(directory, False, 1, tiny_hybrid_moe, mfu=True)
            actual, records = self._run(directory, True, 1, tiny_hybrid_moe, mfu=True)
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
