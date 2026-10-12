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
"""Loss diagnostics and throughput must not change the optimization objective."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers.modeling_outputs import CausalLMOutput

from hyper_parallel.components.losses.model_output import ModelOutputLoss
from hyper_parallel.components.losses.mtp import calculate_mtp_loss
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.callbacks.environ_meter_callback import EnvironMeterCallback
from hyper_parallel.trainer.callbacks.logging_callback import LoggingCallback
from hyper_parallel.trainer.runtime.flops import TransformerFlopsEstimator
from hyper_parallel.data.batching.get_batch import TextParallelBatch
from hyper_parallel.trainer.state import TrainerState
from tests.ut.trainer.test_flops import decoder_config


METER = "hyper_parallel.trainer.callbacks.environ_meter_callback"


class _ScalarModel(torch.nn.Module):
    """Return an objective and deliberately different logging-only scalar."""

    def __init__(self) -> None:
        """Use one parameter so an accidental diagnostic gradient is visible."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False,
                labels: torch.Tensor | None = None) -> SimpleNamespace:
        """Return a differentiable diagnostic to exercise the detach boundary."""
        del use_cache, labels
        return SimpleNamespace(loss=self.weight.square() * input_ids.mean(),
                               loss_metrics={"mtp_1_loss": self.weight * 100})


def metric_trainer(**overrides: object) -> SimpleNamespace:
    """Build the real callbacks' small owner contract."""
    values = {
        "mesh": SimpleNamespace(dp_cp_mesh=None, dp_size=1, cp_size=1, tp_size=1, pp_size=1),
        "lr_scheduler": None, "optimizer": SimpleNamespace(param_groups=[{"lr": 0.001}]),
        "global_rank": 0, "config": SimpleNamespace(training=SimpleNamespace(logging_steps=2)),
    }
    return SimpleNamespace(**(values | overrides))


class TestLossLogging(unittest.TestCase):
    """Verify detached collection, timing, units and complete log lines."""

    def setUp(self) -> None:
        """Keep logic tests independent of accelerator and process-group state."""
        self.enterContext(patch(f"{METER}.get_device_type", return_value="cpu"))
        self.enterContext(patch(f"{METER}.get_world_size_safe", return_value=1))

    def test_diagnostics_mean_reset_and_logging_cadence(self) -> None:
        """Two micro-step means are logged once and never overwrite total loss."""
        trainer = metric_trainer(train_iters=100)
        meter = EnvironMeterCallback(trainer)
        logger = LoggingCallback(trainer)
        state = TrainerState(global_step=1)
        with patch(f"{METER}.time.perf_counter", side_effect=(1.0, 3.0)):
            meter.on_step_begin(state)
            for value in (2.0, 4.0):
                diagnostic = torch.tensor(value, requires_grad=True)
                meter.record_loss_metrics(SimpleNamespace(loss_metrics={"mtp_1_loss": diagnostic},
                                                          indexer_loss=diagnostic * 2))
            self.assertTrue(all(not value.requires_grad for value in meter._loss_metrics.values()))
            meter.on_step_end(state, loss=5.0, loss_dict={"foundation_loss": 5.0}, grad_norm=0.2)
        self.assertEqual(trainer.step_train_metrics["training/mtp_1_loss"], 3.0)
        self.assertEqual(trainer.step_train_metrics["training/indexer_loss"], 6.0)
        self.assertEqual(trainer.step_train_metrics["training/total_loss"], 5.0)
        self.assertFalse(meter._loss_metrics)
        with patch.object(logger, "_write") as write:
            logger.on_step_end(state, loss=5.0, loss_dict={}, grad_norm=0.2)
            self.assertFalse(write.called)
            state.global_step = 2
            for _ in range(2):
                logger.on_step_end(state, loss=5.0, loss_dict={}, grad_norm=0.2)
            write.assert_called_once()
            self.assertIn("training/mtp_1_loss=3", write.call_args.args[0])
            self.assertIn("step=2 total_steps=100 epoch=0", write.call_args.args[0])
        meter.on_step_begin(state)
        meter.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.0)
        self.assertNotIn("training/mtp_1_loss", trainer.step_train_metrics)

    def test_throughput_counts_input_work_and_waits_for_device(self) -> None:
        """Masked labels change valid token/s, not padded model FLOPs."""
        trainer = metric_trainer(model_config=decoder_config())
        meter = EnvironMeterCallback(trainer)
        device = Mock()
        with patch(f"{METER}.get_device_type", return_value="npu"), \
                patch(f"{METER}.get_torch_device", return_value=device), \
                patch.object(meter, "_memory_metrics", return_value={}), \
                patch(f"{METER}.time.perf_counter", side_effect=(10.0, 12.0)):
            state = TrainerState(global_step=1)
            meter.on_step_begin(state, micro_batches=[{"input_ids": torch.zeros(2, 8), "token_count": 3}])
            meter.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.0)
        metrics = trainer.step_env_metrics
        self.assertEqual(metrics["performance/tokens_per_second"], 1.5)
        self.assertEqual(metrics["performance/input_tokens_per_second"], 8.0)
        self.assertEqual(metrics["performance/samples_per_second"], 1.0)
        expected = TransformerFlopsEstimator(trainer.model_config)({"input_ids": torch.zeros(2, 8)}) / 2e12
        self.assertEqual(metrics["performance/throughput_tflops_per_device"], expected)
        self.assertEqual(device.synchronize.call_count, 2)

    def test_invalid_metrics_do_not_silently_replace_objectives(self) -> None:
        """Reject vectors, unstable per-step keys and reserved-name collisions."""
        meter = EnvironMeterCallback(metric_trainer())
        state = TrainerState(global_step=1)
        meter.on_step_begin(state)
        with self.assertRaisesRegex(ValueError, "scalar"):
            meter.record_loss_metrics({"loss_metrics": {"mtp_1_loss": torch.ones(2)}})
        meter.record_loss_metrics({"loss_metrics": {"mtp_1_loss": torch.tensor(1.0)}})
        with self.assertRaisesRegex(ValueError, "stable"):
            meter.record_loss_metrics({"loss_metrics": {}})
        meter.on_step_begin(state)
        meter.record_loss_metrics({"loss_metrics": {"total_loss": torch.tensor(99.0)}})
        with self.assertRaisesRegex(ValueError, "collides"):
            meter.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.0)

    def test_model_output_accepts_attached_diagnostics(self) -> None:
        """HF ModelOutput can carry a new attribute outside its mapping keys."""
        trainer = metric_trainer()
        meter = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1)
        output = CausalLMOutput(loss=torch.tensor(2.0))
        output.loss_metrics = {"mtp_1_loss": torch.tensor(3.0)}
        meter.on_step_begin(state)
        meter.record_loss_metrics(output)
        meter.on_step_end(state, loss=2.0, loss_dict=None, grad_norm=0.0)
        self.assertEqual(trainer.step_train_metrics["training/mtp_1_loss"], 3.0)

    def test_real_backward_ignores_diagnostics(self) -> None:
        """Run BaseTrainer forward/backward; an extra diagnostic must not add its gradient."""
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.mesh = SimpleNamespace(dp_cp_mesh=None, dp_size=1, cp_size=1, sequence_parallel=False)
        trainer.model = _ScalarModel()
        trainer.loss_fn = ModelOutputLoss()
        trainer.config = SimpleNamespace(training=SimpleNamespace(empty_cache_before_backward=False))
        trainer.model_fwd_context = nullcontext()
        trainer.model_bwd_context = nullcontext()
        trainer.model_integration = SimpleNamespace(record_batch=lambda *args: None)
        trainer.preforward = lambda batch: batch
        trainer.current_token_counts = {"foundation_tokens": torch.tensor(1.0)}
        trainer.step_token_counts = trainer.current_token_counts
        trainer.environ_meter_callback = EnvironMeterCallback(trainer)
        trainer.environ_meter_callback.on_step_begin(TrainerState(global_step=1))
        with patch("hyper_parallel.trainer.runtime.metrics.all_reduce", side_effect=lambda value, **kwargs: value):
            loss, _ = trainer.forward_backward_step({"input_ids": torch.ones(1, 2)})
        self.assertEqual(loss.item(), 4.0)
        self.assertEqual(trainer.model.weight.grad.item(), 4.0)
        self.assertEqual(trainer.environ_meter_callback._loss_metrics["mtp_1_loss"].item(), 200.0)

    def test_mtp_depth_metrics_preserve_loss_and_gradients(self) -> None:
        """Collect each existing MTP objective without changing its sum or gradient."""
        torch.manual_seed(19)
        logits = [torch.randn(1, 4, 8, requires_grad=True) for _ in range(2)]
        reference = [value.detach().clone().requires_grad_() for value in logits]
        labels = torch.tensor([[1, 2, 3, -100]])
        loss_fn = torch.nn.CrossEntropyLoss()
        metrics = {}
        actual = calculate_mtp_loss(logits, [], labels, loss_fn, loss_metrics=metrics)
        expected = calculate_mtp_loss(reference, [], labels, loss_fn)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(sum(metrics.values()), expected.detach(), rtol=0, atol=0)
        self.assertEqual(set(metrics), {"mtp_1_loss", "mtp_2_loss"})
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        actual.backward()
        expected.backward()
        torch.testing.assert_close([value.grad for value in logits], [value.grad for value in reference],
                                   rtol=0, atol=0)


    def test_unavailable_and_invalid_flops(self) -> None:
        """Unsupported model geometry omits FLOPs; invalid work is rejected."""
        trainer = metric_trainer(model_config=SimpleNamespace(text_config=decoder_config()))
        meter = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1)
        batch = {"input_ids": torch.zeros(1, 8)}
        meter.on_step_begin(state, micro_batches=[batch])
        meter.on_step_end(state, loss=2.0, loss_dict=None, grad_norm=0.0)
        self.assertNotIn("performance/throughput_tflops_per_device", trainer.step_env_metrics)
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                meter.on_step_begin(state)
                meter._flops_estimator = Mock(return_value=value)
                meter.on_micro_step_begin(state, batch)
                with self.assertRaisesRegex(ValueError, "finite nonnegative"):
                    meter.on_step_end(state, loss=2.0, loss_dict=None, grad_norm=0.0)

    def test_packed_batch_metadata_reaches_meter_without_changing_forward(self) -> None:
        """Run actual TextParallelBatch; only accounting sees cumulative boundaries."""
        trainer = metric_trainer(model_config=decoder_config())
        get_batch = TextParallelBatch(
            mesh_context=trainer.mesh, device=torch.device("cpu"), tokenizer=None,
            data_config={}, pp_shared_data=False, source_type="online", attention_mode="dense",
        )
        source = {"input_ids": torch.arange(8).reshape(1, 8), "labels": torch.arange(8).reshape(1, 8),
                  "cu_seq_lens": torch.tensor([0, 3, 8], dtype=torch.int32)}
        model_inputs, loss_inputs = get_batch(iter([source]))
        self.assertNotIn("cu_seq_lens", model_inputs)
        self.assertIn("cu_seq_lens", loss_inputs)
        meter = EnvironMeterCallback(trainer)
        meter.on_step_begin(TrainerState(global_step=1), micro_batches=[{**model_inputs, **loss_inputs}])
        expected = TransformerFlopsEstimator(trainer.model_config)(source)
        self.assertEqual(meter._local_flops.item(), expected.item())

    def test_compressed_without_packed_forward_uses_full_sequence_work(self) -> None:
        """Online source boundaries alone do not make ordinary causal attention packed."""
        trainer = metric_trainer(model_config=decoder_config())
        get_batch = TextParallelBatch(
            mesh_context=trainer.mesh, device=torch.device("cpu"), tokenizer=None,
            data_config={}, pp_shared_data=False, source_type="online", attention_mode="compressed",
        )
        source = {"input_ids": torch.arange(8).reshape(1, 8), "labels": torch.arange(8).reshape(1, 8),
                  "cu_seq_lens": torch.tensor([0, 3, 8], dtype=torch.int32)}
        model_inputs, loss_inputs = get_batch(iter([source]))
        self.assertIsNone(model_inputs["attention_mask"])
        self.assertNotIn("cu_seq_lens", loss_inputs)
        estimator = TransformerFlopsEstimator(trainer.model_config)
        self.assertEqual(estimator({**model_inputs, **loss_inputs}), estimator({"input_ids": source["input_ids"]}))

    def test_negative_microstep_cannot_be_hidden_by_a_positive_one(self) -> None:
        """Validate each estimate on device before reducing once at step end."""
        meter = EnvironMeterCallback(metric_trainer())
        meter._flops_estimator = Mock(side_effect=[torch.tensor(-1.), torch.tensor(10.)])
        state = TrainerState(global_step=1)
        meter.on_step_begin(state)
        for _ in range(2):
            meter.on_micro_step_begin(state, {"input_ids": torch.zeros(1, 8)})
        with self.assertRaisesRegex(ValueError, "finite nonnegative"):
            meter.on_step_end(state, loss=2.0, loss_dict=None, grad_norm=0.0)
