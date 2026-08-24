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
"""Tests for structured Trainer environment metrics."""

from types import SimpleNamespace

import pytest
import torch

import hyper_models.trainer.callbacks.environ_meter_callback as callback_module
from hyper_models.trainer.callbacks import EnvironMeterCallback, TrainerState


def _build_trainer() -> SimpleNamespace:
    """Build the callback's minimal Trainer dependency surface."""
    scheduler = SimpleNamespace(get_last_lr=lambda: [2.0e-4, 1.0e-4])
    optimizer = SimpleNamespace(param_groups=[{"lr": 5.0e-5}])
    return SimpleNamespace(
        mesh=SimpleNamespace(dp_cp_mesh=None),
        optimizer=[optimizer],
        lr_scheduler=[scheduler],
    )


def test_environ_meter_callback_publishes_structured_metrics(monkeypatch) -> None:
    """Publish training, throughput, and cumulative data metrics."""
    trainer = _build_trainer()
    callback = EnvironMeterCallback(trainer)
    timer = iter([10.0, 12.0])
    monkeypatch.setattr(callback_module.time, "perf_counter", lambda: next(timer))
    monkeypatch.setattr(callback_module, "get_device_type", lambda: "cpu")
    micro_batches = [
        {
            "input_ids": torch.ones(2, 4, dtype=torch.int64),
            "labels": torch.tensor([[1, 1, 1, -100], [1, 1, -100, -100]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        }
    ]
    original_keys = set(micro_batches[0])

    callback.on_step_begin(TrainerState(global_step=1), micro_batches=micro_batches)
    callback.on_step_end(
        TrainerState(global_step=1),
        loss=2.5,
        loss_dict={"foundation_loss": 2.25},
        grad_norm=0.75,
    )

    assert set(micro_batches[0]) == original_keys
    assert trainer.step_train_metrics == {
        "training/total_loss": 2.5,
        "training/grad_norm": 0.75,
        "training/lr": 2.0e-4,
        "training/foundation_loss": 2.25,
    }
    assert trainer.step_env_metrics["performance/step_time"] == 2.0
    assert trainer.step_env_metrics["performance/tokens_per_second"] == 2.5
    assert trainer.step_env_metrics["data/step_tokens"] == 5.0
    assert trainer.step_env_metrics["data/consumed_tokens"] == 5.0
    assert trainer.step_env_metrics["data/step_samples"] == 2.0
    assert "memory/device_max_allocated_gb" not in trainer.step_env_metrics


def test_environ_meter_callback_reduces_distributed_metrics(monkeypatch) -> None:
    """Use sum, mean, and max reductions with the Trainer DP+CP group."""
    group = object()
    trainer = _build_trainer()
    trainer.mesh.dp_cp_mesh = SimpleNamespace(get_group=lambda: group)
    callback = EnvironMeterCallback(trainer)
    calls = []

    def _all_reduce(value, op, group):
        calls.append((float(value), op, group))
        return float(value) * 2 if op == "sum" else float(value)

    monkeypatch.setattr(callback_module, "get_world_size_safe", lambda: 2)
    monkeypatch.setattr(callback_module, "all_reduce", _all_reduce)
    monkeypatch.setattr(callback_module, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(callback_module.time, "perf_counter", lambda: 1.0)

    callback.on_step_begin(
        TrainerState(global_step=1),
        micro_batches=[{"input_ids": torch.ones(1, 3, dtype=torch.int64)}],
    )
    callback.on_step_end(
        TrainerState(global_step=1),
        loss=1.0,
        loss_dict={},
        grad_norm=0.5,
    )

    assert trainer.step_env_metrics["data/step_tokens"] == 6.0
    assert {op for _, op, _ in calls} == {"max", "mean", "sum"}
    assert all(reduced_group is group for _, _, reduced_group in calls)


def test_environ_meter_callback_state_round_trip() -> None:
    """Restore cumulative counters and reject invalid checkpoint state."""
    callback = EnvironMeterCallback(_build_trainer())

    callback.load_state_dict({"consumed_tokens": 17, "consumed_samples": 3})

    assert callback.state_dict() == {
        "consumed_tokens": 17,
        "consumed_samples": 3,
    }
    with pytest.raises(ValueError, match="must contain integer"):
        callback.load_state_dict({"consumed_tokens": 1})
