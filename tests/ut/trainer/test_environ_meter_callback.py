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
"""Unit tests for Trainer environment and throughput metrics."""

from fractions import Fraction
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

import hyper_parallel.trainer.callbacks.environ_meter_callback as environ_meter_module
from hyper_parallel.trainer.callbacks.environ_meter_callback import EnvironMeterCallback
from hyper_parallel.trainer.runtime.flops import estimate_step_flops
from hyper_parallel.trainer.config import MfuConfig
from hyper_parallel.trainer.runtime.model_flops import MatmulParamCount
from hyper_parallel.trainer.state import TrainerState
from tests.common.mark_utils import arg_mark


class _NoItemScalar:
    """Represent a device scalar whose host conversion is forbidden."""

    def detach(self) -> "_NoItemScalar":
        """Return a detached scalar view."""
        return self

    def clone(self) -> "_NoItemScalar":
        """Return an independently owned scalar."""
        return _NoItemScalar()

    def item(self) -> float:
        """Fail if metric collection synchronizes the scalar to the host."""
        raise AssertionError("micro-step metric collection must not call item()")


class TestEnvironMeterCallback(unittest.TestCase):
    """Verify token throughput uses streamed, prepared text batches."""

    @staticmethod
    def _build_trainer(mesh: object = None) -> SimpleNamespace:
        """Build the callback's minimal Trainer dependency surface."""
        return SimpleNamespace(
            mesh=mesh,
            lr_scheduler=None,
            optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]),
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    def test_registered_metrics_average_observed_microbatches(self, mock_world: Mock, mock_device: Mock) -> None:
        """Average registered observations while preserving counts and training losses."""
        del mock_world, mock_device
        trainer = self._build_trainer()
        callback = EnvironMeterCallback(trainer, microbatch_mean_metrics=("router_aux_loss", "quality"))
        state = TrainerState(global_step=1)
        callback.on_step_begin(state)
        first_aux = torch.tensor(1.0, requires_grad=True)
        callback.on_micro_step_end(state, {"router_aux_loss": first_aux, "quality": 2.0, "token_count": 2})
        with torch.no_grad():
            first_aux.fill_(9.0)
        callback.on_micro_step_end(state, {})
        callback.on_micro_step_end(state, {"router_aux_loss": torch.tensor(3.0), "quality": 6.0, "token_count": 8})
        totals = {"router_aux_loss": 4.0, "quality": 8.0, "token_count": 10, "foundation_loss": 2.5}
        callback.on_step_end(state, loss=2.5, loss_dict=totals, grad_norm=1.0)
        expected = {"router_aux_loss": 2.0, "quality": 4.0, "token_count": 10.0, "foundation_loss": 2.5}
        observed = {name: trainer.step_env_metrics[f"training/{name}"] for name in expected}
        self.assertEqual(observed, expected, f"expected aggregation={expected}, got={observed}")
        means = vars(callback)["_local_mean_metrics"]
        self.assertEqual(means, {}, f"expected step-end scalar release, got={means}")

        callback.on_step_begin(state)
        callback.on_micro_step_end(state, {"router_aux_loss": torch.tensor(0.0)})
        callback.on_step_end(state, loss=1.0, loss_dict={"router_aux_loss": 0.0}, grad_norm=1.0)
        observed = trainer.step_env_metrics["training/router_aux_loss"]
        self.assertEqual(observed, 0.0, f"expected zero AUX on the next step, got={observed}")

        callback.on_step_begin(state)
        callback.on_step_end(state, loss=1.0, loss_dict={"router_aux_loss": 9.0}, grad_norm=1.0)
        observed = trainer.step_env_metrics["training/router_aux_loss"]
        self.assertEqual(observed, 9.0, f"expected supplied step value without observations, got={observed}")

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    def test_micro_step_publishes_nonzero_token_throughput(
            self,
            mock_time: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Publish the accumulated micro-step token count as throughput.

        Feature: Trainer throughput metrics.
        Description: Stream a prepared text micro-batch into the environment meter.
        Expectation: Step tokens and tokens per second are positive and scalar state is released.
        """
        del mock_time, mock_world_size, mock_device_type
        trainer = self._build_trainer()
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {
                "input_ids": torch.ones(2, 4),
                "token_count": torch.tensor(6),
            },
        )
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        metrics = trainer.step_env_metrics
        self.assertEqual(
            metrics["data/step_tokens"],
            6.0,
            f"expected six accumulated tokens, got={metrics['data/step_tokens']}",
        )
        self.assertEqual(
            metrics["performance/tokens_per_second"],
            3.0,
            f"expected three tokens per second, got={metrics['performance/tokens_per_second']}",
        )
        step_tokens = vars(callback)["_local_step_tokens"]
        self.assertIsNone(
            step_tokens,
            f"step token scalar must be released after publishing, got={step_tokens!r}",
        )

    def test_micro_step_avoids_host_conversion_and_step_begin_resets(self) -> None:
        """Keep device token scalars asynchronous and clear them at the next step.

        Feature: Environment meter memory lifecycle.
        Description: Accumulate a detached scalar that rejects host conversion and begin a new step.
        Expectation: Accumulation avoids item calls and the next step clears prior scalar state.
        """
        callback = EnvironMeterCallback(self._build_trainer(), microbatch_mean_metrics=("quality",))
        state = TrainerState(global_step=0, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {"token_count": _NoItemScalar()},
        )
        callback.on_micro_step_end(state, {"quality": _NoItemScalar()})

        step_tokens = vars(callback)["_local_step_tokens"]
        self.assertIsInstance(
            step_tokens,
            _NoItemScalar,
            f"expected detached scalar state, got={step_tokens!r}",
        )
        callback.on_step_begin(state)
        step_tokens = vars(callback)["_local_step_tokens"]
        self.assertIsNone(
            step_tokens,
            f"new step must release prior scalar state, got={step_tokens!r}",
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=4)
    @patch.object(environ_meter_module, "all_reduce")
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    def test_context_parallel_reduction_does_not_duplicate_samples(
            self,
            mock_time: Mock,
            mock_all_reduce: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Sum CP-sharded tokens while counting each logical sample once.

        Feature: Context-parallel throughput accounting.
        Description: Reduce local token and sample counts over a mocked DP+CP group.
        Expectation: Tokens sum across ranks while CP-replicated samples are counted once.
        """
        del mock_time, mock_world_size, mock_device_type
        mock_all_reduce.side_effect = lambda value, op, group: float(value) * 4 if op == "sum" else float(value)
        mesh = SimpleNamespace(dp_cp_mesh=None, cp_size=2)
        trainer = self._build_trainer(mesh=mesh)
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {
                "input_ids": torch.ones(2, 4),
                "token_count": torch.tensor(6),
            },
        )
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        metrics = trainer.step_env_metrics
        self.assertEqual(
            metrics["data/step_tokens"],
            24.0,
            f"expected tokens summed across DP+CP, got={metrics['data/step_tokens']}",
        )
        self.assertEqual(
            metrics["data/step_samples"],
            4.0,
            f"expected CP replicas removed from samples, got={metrics['data/step_samples']}",
        )

    @staticmethod
    def _build_tflops_model_config() -> SimpleNamespace:
        """Build a miniature Qwen3-MoE config so the estimator returns a value."""
        return SimpleNamespace(
            model_type="qwen3_moe",
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            intermediate_size=16,
            moe_intermediate_size=4,
            num_experts=4,
            num_experts_per_tok=1,
            vocab_size=32,
            hidden_act="silu",
            mlp_only_layers=[],
            decoder_sparse_step=1,
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    def test_tflops_metric_stays_opt_in(
            self,
            mock_time: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Keep the throughput metric absent unless the switch is enabled.

        Feature: Optional per-device TFLOP/s metrics.
        Description: Complete one step on a supported model with ``log_tflops`` unset.
        Expectation: No ``performance/tflops_per_device`` key is published.
        """
        del mock_time, mock_world_size, mock_device_type
        trainer = self._build_trainer()
        trainer.config = SimpleNamespace(training=SimpleNamespace())
        trainer.model_config = self._build_tflops_model_config()
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {"input_ids": torch.ones(2, 4), "token_count": torch.tensor(6)},
        )
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        metrics = trainer.step_env_metrics
        self.assertNotIn(
            "performance/tflops_per_device",
            metrics,
            f"tflops reporting must be opt-in, got keys={sorted(metrics)}",
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=4)
    @patch.object(environ_meter_module, "all_reduce")
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    def test_tflops_metric_divides_global_work_by_devices(
            self,
            mock_time: Mock,
            mock_all_reduce: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Publish cluster work per device using reduced input tokens and step time.

        Feature: Optional per-device TFLOP/s metrics.
        Description: Complete one step of a two-layer Qwen3-MoE across four mocked devices.
        Expectation: The metric equals step FLOPs / (step_time * world_size) / 1e12.
        """
        del mock_time, mock_world_size, mock_device_type
        mock_all_reduce.side_effect = lambda value, op, group: float(value) * 4 if op == "sum" else float(value)
        model_config = self._build_tflops_model_config()
        trainer = self._build_trainer()
        trainer.config = SimpleNamespace(training=SimpleNamespace(log_tflops=True))
        trainer.model_config = model_config
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {"input_ids": torch.ones(2, 4), "token_count": torch.tensor(6)},
        )
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        # ``input_ids`` covers 2*4 tokens whose two sequences are four long, so
        # each of the four ranks contributes (8, 32) before the DP+CP reduction.
        expected_flops = estimate_step_flops(model_config, 4 * 8.0, 4 * 32.0)
        expected = expected_flops / (2.0 * 4) / 1e12
        metrics = trainer.step_env_metrics
        self.assertIn(
            "performance/tflops_per_device",
            metrics,
            f"enabled tflops reporting must publish a metric, got keys={sorted(metrics)}",
        )
        self.assertAlmostEqual(
            metrics["performance/tflops_per_device"],
            expected,
            delta=expected * 1e-9,
            msg=(
                "expected cluster work divided by step time and devices, "
                f"got={metrics['performance/tflops_per_device']}, expected={expected}"
            ),
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    @patch.object(environ_meter_module, "all_reduce")
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    def test_tflops_metric_counts_prompt_tokens_excluded_from_loss(
            self,
            mock_time: Mock,
            mock_all_reduce: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Charge FLOPs for masked prompt tokens that the loss token count drops.

        Feature: Optional per-device TFLOP/s metrics.
        Description: Feed one length-8 sequence whose loss token count is only 2.
        Expectation: FLOPs use all eight input tokens while throughput keeps the loss count.
        """
        del mock_time, mock_all_reduce, mock_world_size, mock_device_type
        model_config = self._build_tflops_model_config()
        trainer = self._build_trainer()
        trainer.config = SimpleNamespace(training=SimpleNamespace(log_tflops=True))
        trainer.model_config = model_config
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {"input_ids": torch.ones(1, 8), "token_count": torch.tensor(2)},
        )
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        expected = estimate_step_flops(model_config, 8.0, 64.0) / 2.0 / 1e12
        metrics = trainer.step_env_metrics
        self.assertAlmostEqual(
            metrics["performance/tflops_per_device"],
            expected,
            delta=expected * 1e-9,
            msg=(
                "prompt tokens masked out of the loss still consume model FLOPs, "
                f"got={metrics['performance/tflops_per_device']}, expected={expected}"
            ),
        )
        self.assertEqual(
            metrics["data/step_tokens"],
            2.0,
            f"loss token accounting must stay unchanged, got={metrics['data/step_tokens']}",
        )
        self.assertEqual(
            metrics["performance/tokens_per_second"],
            1.0,
            "tokens per second must keep using loss tokens, "
            f"got={metrics['performance/tokens_per_second']}",
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    @patch.object(environ_meter_module, "all_reduce")
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    def test_tflops_metric_accumulates_squared_sequence_lengths(
            self,
            mock_time: Mock,
            mock_all_reduce: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Accumulate every sequence's attention boundary instead of a mean length.

        Feature: Optional per-device TFLOP/s metrics.
        Description: Stream two micro-batches whose attention masks keep 4 and 2 tokens.
        Expectation: FLOPs use sum(S)=6 and sum(S**2)=20, above the mean-length value.
        """
        del mock_time, mock_all_reduce, mock_world_size, mock_device_type
        model_config = self._build_tflops_model_config()
        trainer = self._build_trainer()
        trainer.config = SimpleNamespace(training=SimpleNamespace(log_tflops=True))
        trainer.model_config = model_config
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {
                "input_ids": torch.ones(1, 4),
                "attention_mask": torch.ones(1, 4, dtype=torch.int32),
            },
        )
        callback.on_micro_step_begin(
            state,
            {
                "input_ids": torch.ones(1, 4),
                "attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.int32),
            },
        )
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        expected = estimate_step_flops(model_config, 6.0, 20.0) / 2.0 / 1e12
        mean_length_only = estimate_step_flops(model_config, 6.0, 6.0 * 3.0) / 2.0 / 1e12
        metrics = trainer.step_env_metrics
        self.assertAlmostEqual(
            metrics["performance/tflops_per_device"],
            expected,
            delta=expected * 1e-9,
            msg=(
                "expected sum(S**2) attention work, "
                f"got={metrics['performance/tflops_per_device']}, expected={expected}"
            ),
        )
        self.assertGreater(
            metrics["performance/tflops_per_device"],
            mean_length_only,
            "averaging the sequence lengths must underestimate attention work, "
            f"got={metrics['performance/tflops_per_device']} against {mean_length_only}",
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    def test_tflops_metric_uses_packed_sequence_boundaries(
            self,
            mock_time: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Use ``cu_seq_lens`` boundaries when a row packs several sequences.

        Feature: Optional per-device TFLOP/s metrics.
        Description: Pack two sequences of length 2 and 4 into one length-6 row.
        Expectation: FLOPs use sum(S)=6 and sum(S**2)=20 instead of 6**2.
        """
        del mock_time, mock_world_size, mock_device_type
        model_config = self._build_tflops_model_config()
        trainer = self._build_trainer()
        trainer.config = SimpleNamespace(training=SimpleNamespace(log_tflops=True))
        trainer.model_config = model_config
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {
                "input_ids": torch.ones(1, 6),
                "cu_seq_lens": torch.tensor([0, 2, 6], dtype=torch.int32),
            },
        )
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        expected = estimate_step_flops(model_config, 6.0, 20.0) / 2.0 / 1e12
        metrics = trainer.step_env_metrics
        self.assertAlmostEqual(
            metrics["performance/tflops_per_device"],
            expected,
            delta=expected * 1e-9,
            msg=(
                "packed rows must charge each segment, "
                f"got={metrics['performance/tflops_per_device']}, expected={expected}"
            ),
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=2)
    @patch.object(environ_meter_module, "all_reduce")
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    def test_tflops_metric_restores_context_parallel_sequence_length(
            self,
            mock_time: Mock,
            mock_all_reduce: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Restore ``S ** 2`` after summing sequence squares over DP+CP shards.

        Feature: Optional per-device TFLOP/s metrics.
        Description: Hold one half of a CP-split length-12 sequence on each rank.
        Expectation: cp_size=2 turns the per-rank 6**2 into the real 12**2.
        """
        del mock_time, mock_world_size, mock_device_type
        mock_all_reduce.side_effect = lambda value, op, group: float(value) * 2 if op == "sum" else float(value)
        model_config = self._build_tflops_model_config()
        mesh = SimpleNamespace(dp_cp_mesh=None, cp_size=2)
        trainer = self._build_trainer(mesh=mesh)
        trainer.config = SimpleNamespace(training=SimpleNamespace(log_tflops=True))
        trainer.model_config = model_config
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {
                "input_ids": torch.ones(1, 6),
                # Global boundaries that still span both CP shards: the helper
                # must ignore them and keep the local sequence geometry.
                "cu_seq_lens": torch.tensor([0, 4, 12], dtype=torch.int32),
            },
        )
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        expected = estimate_step_flops(model_config, 12.0, 144.0) / 2.0 / 2 / 1e12
        metrics = trainer.step_env_metrics
        self.assertAlmostEqual(
            metrics["performance/tflops_per_device"],
            expected,
            delta=expected * 1e-9,
            msg=(
                "context parallelism must restore the per-sequence square, "
                f"got={metrics['performance/tflops_per_device']}, expected={expected}"
            ),
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    def test_tflops_metric_skips_unsupported_models(
            self,
            mock_time: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Skip models the estimator does not support instead of guessing.

        Feature: Optional per-device TFLOP/s metrics.
        Description: Enable ``log_tflops`` for a configuration declaring ``model_type='llama'``.
        Expectation: The step publishes no FLOPs metric.
        """
        del mock_time, mock_world_size, mock_device_type
        trainer = self._build_trainer()
        trainer.config = SimpleNamespace(training=SimpleNamespace(log_tflops=True))
        trainer.model_config = SimpleNamespace(model_type="llama")
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {"input_ids": torch.ones(2, 4), "token_count": torch.tensor(6)},
        )
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        metrics = trainer.step_env_metrics
        self.assertNotIn(
            "performance/tflops_per_device",
            metrics,
            f"unsupported models must not report FLOPs, got keys={sorted(metrics)}",
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    def test_tflops_metric_ignores_empty_input_geometry(
            self,
            mock_time: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Treat a batch with no tokens as having no geometry to charge.

        Feature: Optional per-device TFLOP/s metrics.
        Description: Stream a micro-batch whose ``input_ids`` has an empty row.
        Expectation: Nothing is charged, so the step publishes no FLOPs metric.
        """
        del mock_time, mock_world_size, mock_device_type
        trainer = self._build_trainer()
        trainer.config = SimpleNamespace(training=SimpleNamespace(log_tflops=True))
        trainer.model_config = self._build_tflops_model_config()
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(state, {"input_ids": torch.ones(0, 4)})
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        self.assertIsNone(
            callback._batch_input_geometry({"input_ids": torch.ones(0, 4)}),
            "an empty input must not produce geometry",
        )
        metrics = trainer.step_env_metrics
        self.assertNotIn(
            "performance/tflops_per_device",
            metrics,
            f"a token-free step must not report FLOPs, got keys={sorted(metrics)}",
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    def test_tflops_metric_requires_tokens(
            self,
            mock_time: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Avoid publishing a throughput for a step that consumed no tokens.

        Feature: Optional per-device TFLOP/s metrics.
        Description: Enable ``log_tflops`` and complete a step without any micro-batch.
        Expectation: The step publishes no FLOPs metric instead of zero or NaN.
        """
        del mock_time, mock_world_size, mock_device_type
        trainer = self._build_trainer()
        trainer.config = SimpleNamespace(training=SimpleNamespace(log_tflops=True))
        trainer.model_config = self._build_tflops_model_config()
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        metrics = trainer.step_env_metrics
        self.assertNotIn(
            "performance/tflops_per_device",
            metrics,
            f"a token-free step must not report FLOPs, got keys={sorted(metrics)}",
        )


class _TinyCausalLM(nn.Module):
    """One attention projection between an embedding table and an output head."""

    def __init__(self) -> None:
        """Allocate an embedding, one projection and an untied head."""
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 16, bias=False)


_FLOPS_KEYS = (
    "performance/model_tflops_per_device",
    "performance/mfu_percent",
    "performance/hfu_percent",
)


@patch.object(environ_meter_module, "get_device_type", return_value="cpu")
@patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
class TestEnvironMeterMfu(unittest.TestCase):
    """Verify opt-in PaLM model-FLOPs utilization metrics."""

    # 8 tokens in 2 samples over 2 s: sequence length 4, 4 tokens/s. N = 64 + 128 and the
    # head holds 128; one layer with 2 heads of width 4 adds 2 * 2 * 8 * 4 forward FLOPs.
    _FLOPS_PER_TOKEN = 6 * 192 + 3 * 128
    _RECOMPUTE_PER_TOKEN = 2 * 64 + 128

    @staticmethod
    def _build_trainer(mfu: MfuConfig, model_config: object = None, checkpoint_mode: str = "off") -> SimpleNamespace:
        """Build a Trainer surface carrying the MFU configuration and a tiny model."""
        model = _TinyCausalLM()
        return SimpleNamespace(
            mesh=None,
            lr_scheduler=None,
            optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]),
            config=SimpleNamespace(
                training=SimpleNamespace(mfu=mfu),
                activation_checkpoint=SimpleNamespace(mode=checkpoint_mode),
            ),
            model=model,
            model_parts=[model],
            model_config=model_config or SimpleNamespace(num_hidden_layers=1, num_attention_heads=2, hidden_size=8),
            global_rank=0,
        )

    @staticmethod
    def _run_step(callback: EnvironMeterCallback) -> dict:
        """Run train begin and one step of 2 x 4 tokens lasting 2 seconds."""
        state = TrainerState(global_step=1, epoch=0)
        callback.on_train_begin(state)
        with patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0)):
            callback.on_step_begin(state, micro_batches=[{"input_ids": torch.ones(2, 4)}])
            callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)
        return callback.trainer.step_env_metrics

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_flops_metrics_are_off_by_default(self, *mocks: Mock) -> None:
        """Publish no FLOPs metrics unless training.mfu.enabled is set.

        Feature: PaLM MFU reporting.
        Description: Run a step with the default MFU configuration.
        Expectation: No model-TFLOPS, MFU or HFU metric is published.
        """
        del mocks
        metrics = self._run_step(EnvironMeterCallback(self._build_trainer(MfuConfig())))
        for key in _FLOPS_KEYS:
            self.assertNotIn(key, metrics, f"case: {key}_absent_by_default, got={sorted(metrics)}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_mfu_and_hfu_follow_palm_formula(self, *mocks: Mock) -> None:
        """Divide PaLM model FLOPs throughput by the configured peak.

        Feature: PaLM MFU reporting.
        Description: Enable MFU with a peak and full activation recomputation, then run one step.
        Expectation: Model TFLOPS, MFU and HFU equal the hand-computed PaLM values.
        """
        del mocks
        trainer = self._build_trainer(MfuConfig(enabled=True, peak_tflops_per_device=1e-8), checkpoint_mode="full")
        metrics = self._run_step(EnvironMeterCallback(trainer))
        model_flops = self._FLOPS_PER_TOKEN * 4.0
        hardware_flops = (self._FLOPS_PER_TOKEN + self._RECOMPUTE_PER_TOKEN) * 4.0
        expected = {
            "performance/model_tflops_per_device": model_flops / 1e12,
            "performance/mfu_percent": model_flops / 1e4 * 100.0,
            "performance/hfu_percent": hardware_flops / 1e4 * 100.0,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(metrics[key], value, places=12, msg=f"case: {key}, got={metrics.get(key)}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_without_peak_only_model_tflops_is_published(self, *mocks: Mock) -> None:
        """Report achieved model TFLOPS without assuming a device peak.

        Feature: PaLM MFU reporting.
        Description: Enable MFU without peak_tflops_per_device and run one step.
        Expectation: Model TFLOPS is published; MFU and HFU are not.
        """
        del mocks
        metrics = self._run_step(EnvironMeterCallback(self._build_trainer(MfuConfig(enabled=True))))
        self.assertIn(_FLOPS_KEYS[0], metrics, f"case: model_tflops_published, got={sorted(metrics)}")
        for key in _FLOPS_KEYS[1:]:
            self.assertNotIn(key, metrics, f"case: {key}_needs_peak, got={sorted(metrics)}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_uncovered_model_reports_reason_instead_of_numbers(self, *mocks: Mock) -> None:
        """Warn and publish nothing for structures the PaLM formula does not cover.

        Feature: PaLM MFU reporting.
        Description: Enable MFU for linear-attention layers whose geometry fields are missing.
        Expectation: A warning names the structure and no FLOPs metric is published.
        """
        del mocks
        model_config = SimpleNamespace(
            num_hidden_layers=1, num_attention_heads=2, hidden_size=8, layer_types=["linear_attention"])
        trainer = self._build_trainer(MfuConfig(enabled=True, peak_tflops_per_device=400.0), model_config)
        with self.assertLogs(environ_meter_module.logger, level="WARNING") as logs:
            metrics = self._run_step(EnvironMeterCallback(trainer))
        self.assertTrue(
            any("linear_attention" in line for line in logs.output), f"case: reason_logged, got={logs.output}")
        for key in _FLOPS_KEYS:
            self.assertNotIn(key, metrics, f"case: {key}_absent_when_uncovered, got={sorted(metrics)}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_mfu_counts_processed_tokens_not_loss_tokens(self, *mocks: Mock) -> None:
        """Count prompt tokens and drop padding for MFU while tokens/s keeps loss tokens.

        Feature: PaLM MFU reporting for supervised fine-tuning batches.
        Description: Run a padded batch whose labels mask the prompt, so loss tokens are fewer than inputs.
        Expectation: tokens/s follows the 3 loss tokens; MFU uses the 6 unpadded input tokens and their
            the sum of squared sequence lengths for the attention term.
        """
        del mocks
        trainer = self._build_trainer(MfuConfig(enabled=True, peak_tflops_per_device=1e-8))
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)
        callback.on_train_begin(state)
        batch = {
            "input_ids": torch.ones(2, 4),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
            "token_count": torch.tensor(3),
        }
        with patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0)):
            callback.on_step_begin(state)
            callback.on_micro_step_begin(state, batch)
            callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)
        metrics = trainer.step_env_metrics
        self.assertEqual(
            metrics["performance/tokens_per_second"], 1.5, f"case: loss_token_throughput, got={metrics}")
        flops_per_token = 6 * 192 + 3 * (2 * 2 * 8 * (4**2 + 2**2) / 6)
        self.assertAlmostEqual(
            metrics["performance/mfu_percent"], flops_per_token * 3.0 / 1e4 * 100.0, places=12,
            msg=f"case: processed_token_mfu, got={metrics.get('performance/mfu_percent')}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_pipeline_stage_counts_are_summed_once(self, *mocks: Mock) -> None:
        """Sum disjoint pipeline-stage parameters exactly once per stage.

        Feature: PaLM MFU reporting under pipeline parallelism.
        Description: Gather counts from two stages of two ranks each, one stage holding the head.
        Expectation: Stage counts are summed, rank duplicates within a stage are removed and
            reasons are merged.
        """
        del mocks
        first = MatmulParamCount(Fraction(100), Fraction(0), vision_params=Fraction(8))
        last = MatmulParamCount(
            Fraction(300), Fraction(40), unsupported=("multi-token prediction layers are present",))

        def fake_all_gather_object(output: list, obj: object) -> None:
            """Return what four ranks of two pipeline stages would contribute."""
            del obj
            output[:] = [first, first, last, last]

        trainer = self._build_trainer(MfuConfig(enabled=True))
        trainer.mesh = SimpleNamespace(pp_size=2, cp_size=1)
        callback = EnvironMeterCallback(trainer)
        gather = getattr(callback, "_gather_pipeline_counts")
        with patch.object(environ_meter_module.dist, "is_initialized", return_value=True), \
                patch.object(environ_meter_module.dist, "get_world_size", return_value=4), \
                patch.object(environ_meter_module.dist, "all_gather_object", side_effect=fake_all_gather_object):
            merged = gather(first)
        self.assertEqual(merged.matmul_params, 400, f"case: stage_sum, got={merged.matmul_params}")
        self.assertEqual(merged.output_head_params, 40, f"case: head_sum, got={merged.output_head_params}")
        self.assertEqual(merged.vision_params, 8, f"case: vision_sum, got={merged.vision_params}")
        self.assertEqual(merged.unsupported, last.unsupported, f"case: reasons_merged, got={merged.unsupported}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_non_positive_peak_is_rejected(self, *mocks: Mock) -> None:
        """Reject a peak throughput that cannot normalize utilization.

        Feature: PaLM MFU reporting.
        Description: Enable MFU with a zero peak.
        Expectation: Training begin raises ValueError.
        """
        del mocks
        callback = EnvironMeterCallback(self._build_trainer(MfuConfig(enabled=True, peak_tflops_per_device=0.0)))
        with self.assertRaises(ValueError):
            callback.on_train_begin(TrainerState(global_step=0, epoch=0))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_targets_and_extra_hfu_are_judged_and_summarized(self, *mocks: Mock) -> None:
        """Judge every configured target per logged step and summarize at train end.

        Feature: MFU acceptance targets.
        Description: Enable MFU with a peak, a declared extra executed-work term
            (no recompute), and four targets; run one step, then end training.
        Expectation: HFU adds the extra term over step time; throughput is
            published; the summary logs MET for satisfied targets and MISSED
            for the rest.
        """
        del mocks
        config = MfuConfig(
            enabled=True,
            peak_tflops_per_device=1e-8,
            hfu_extra_tflops_per_step=2048e-12,
            target_mfu_percent=self._FLOPS_PER_TOKEN * 4.0 / 1e4 * 100.0 - 1.0,
            target_hfu_percent=(self._FLOPS_PER_TOKEN * 4.0 + 1024.0) / 1e4 * 100.0 + 5.0,
            target_tokens_per_s_per_device=5.0,
            target_step_time_s=2.5,
        )
        callback = EnvironMeterCallback(self._build_trainer(config))
        metrics = self._run_step(callback)
        self.assertAlmostEqual(metrics["performance/tokens_per_s_per_device"], 4.0, places=12,
                               msg=f"case: throughput_published, got={metrics}")
        expected_hfu = (self._FLOPS_PER_TOKEN * 4.0 + 2048.0 / 2.0) / 1e4 * 100.0
        self.assertAlmostEqual(metrics["performance/hfu_percent"], expected_hfu, places=10,
                               msg=f"case: hfu_adds_extra_over_step_time, got={metrics}")
        with self.assertLogs(environ_meter_module.logger, level="INFO") as logs:
            callback.on_train_end(TrainerState(global_step=1, epoch=0))
        summary = "\n".join(logs.output)
        for fragment in ("MFU %: required", "HFU %: required", "tokens/s/device: required",
                         "step time s: required"):
            self.assertIn(fragment, summary, f"case: {fragment}, got={summary}")
        self.assertIn("MFU %: required >=", summary)
        self.assertEqual(summary.count("verdict MET"), 2, f"case: met_count, got={summary}")
        self.assertEqual(summary.count("verdict MISSED"), 2, f"case: missed_count, got={summary}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_impossible_target_configurations_are_rejected(self, *mocks: Mock) -> None:
        """Refuse targets that could never be measured or are not positive.

        Feature: MFU acceptance targets.
        Description: Configure a utilization target without a peak, an HFU
            target without recompute or an extra term, and a negative target.
        Expectation: Training begin raises ValueError for each.
        """
        del mocks
        cases = (
            MfuConfig(enabled=True, target_mfu_percent=10.0),
            MfuConfig(enabled=True, peak_tflops_per_device=400.0, target_hfu_percent=20.0),
            MfuConfig(enabled=True, peak_tflops_per_device=400.0, target_step_time_s=-1.0),
        )
        for config in cases:
            callback = EnvironMeterCallback(self._build_trainer(config))
            with self.assertRaises(ValueError, msg=f"case: {config}"):
                callback.on_train_begin(TrainerState(global_step=0, epoch=0))


if __name__ == "__main__":
    unittest.main()
