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
"""CPU unit test for rank-zero monitoring fan-out and secret redaction. metric logging wandb"""
# Local test doubles are not public APIs.
# pylint: disable=invalid-name,missing-public-docstring

from pathlib import Path
from typing import Any

from rl.utils.monitoring.tracker import TrainingTracker


def test_tracker_sanitizes_config_and_fans_out_only_on_rank_zero(tmp_path: Path) -> None:
    """Only rank zero logs to fake W&B with recursively sanitized configuration."""
    init_calls: list[dict[str, Any]] = []
    log_calls: list[tuple[dict[str, Any], int]] = []
    finish_calls: list[str] = []

    class FakeRun:
        """Record W&B run operations."""

        @staticmethod
        def log(payload: dict[str, Any], *, step: int) -> None:
            log_calls.append((payload, step))

        @staticmethod
        def finish() -> None:
            finish_calls.append("finish")

    class FakeWandb:
        """Expose the small W&B interface used by the backend."""

        @staticmethod
        def init(**kwargs: Any) -> FakeRun:
            init_calls.append(kwargs)
            return FakeRun()

        @staticmethod
        def Table(*, columns: list[str], data: list[list[Any]]) -> dict[str, Any]:
            return {"columns": columns, "data": data}

    resolved_config = {
        "model": {"name": "qwen3"},
        "api_key": "top-secret",
        "nested": {
            "password": "hidden",
            "accessToken": "hidden-too",
            "batch_size": 4,
        },
    }
    owner = TrainingTracker(
        rank=0,
        world_size=2,
        backends=("console", "wandb"),
        project_name="project",
        experiment_name="experiment",
        resolved_config=resolved_config,
        wandb_mode="offline",
        wandb_directory=str(tmp_path),
        wandb_module=FakeWandb(),
    )
    non_owner = TrainingTracker(
        rank=1,
        world_size=2,
        backends=("console", "wandb"),
        project_name="project",
        experiment_name="experiment",
        resolved_config=resolved_config,
        wandb_mode="offline",
        wandb_directory=str(tmp_path),
        wandb_module=FakeWandb(),
    )
    sample = {
        "step": 1,
        "rank": 0,
        "prompt": "question",
        "response": "answer",
        "ground_truth": "answer",
        "extracted_answer": "answer",
        "reward": 1.0,
    }

    owner.log({"reward/mean": 1.0}, step=1, samples=[sample])
    non_owner.log({"reward/mean": 0.0}, step=1, samples=[sample])
    owner.finish()
    owner.finish()
    non_owner.finish()

    assert len(init_calls) == 1
    sanitized = init_calls[0]["config"]
    assert sanitized["model"] == {"name": "qwen3"}
    assert sanitized["api_key"] == "***"
    assert sanitized["nested"] == {
        "password": "***",
        "accessToken": "***",
        "batch_size": 4,
    }
    assert len(log_calls) == 1
    payload, step = log_calls[0]
    assert step == 1
    assert payload["reward/mean"] == 1.0
    assert payload["rollout/samples"]["data"][0][-1] == 1.0
    assert finish_calls == ["finish"]
