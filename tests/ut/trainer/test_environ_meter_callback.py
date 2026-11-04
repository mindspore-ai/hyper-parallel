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

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import hyper_parallel.trainer.callbacks.environ_meter_callback as environ_meter_module
from hyper_parallel.trainer.callbacks.environ_meter_callback import EnvironMeterCallback
from hyper_parallel.trainer.runtime.flops import estimate_flops_per_token
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
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
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

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_micro_step_avoids_host_conversion_and_step_begin_resets(self) -> None:
        """Keep device token scalars asynchronous and clear them at the next step.

        Feature: Environment meter memory lifecycle.
        Description: Accumulate a detached scalar that rejects host conversion and begin a new step.
        Expectation: Accumulation avoids item calls and the next step clears prior scalar state.
        """
        callback = EnvironMeterCallback(self._build_trainer())
        state = TrainerState(global_step=0, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {"token_count": _NoItemScalar()},
        )

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
    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
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
    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
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
    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_tflops_metric_divides_global_work_by_devices(
            self,
            mock_time: Mock,
            mock_all_reduce: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Publish cluster work per device using reduced tokens and step time.

        Feature: Optional per-device TFLOP/s metrics.
        Description: Complete one step of a two-layer Qwen3-MoE across four mocked devices.
        Expectation: The metric equals tokens * FLOPs/token / (step_time * world_size) / 1e12.
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

        expected_tokens = 24.0
        expected_samples = 8.0
        expected_flops = estimate_flops_per_token(
            model_config, expected_tokens / expected_samples
        )
        expected = expected_tokens * expected_flops / (2.0 * 4) / 1e12
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
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
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
    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
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


if __name__ == "__main__":
    unittest.main()
