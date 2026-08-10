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
"""CPU unit tests for rank-zero experiment monitoring."""

from pathlib import Path
from typing import Any

from rl.utils.monitoring import TrainingTracker, sanitize_config


class FakeRun:
    """Small W&B run stub recording log and finish calls."""

    def __init__(self) -> None:
        """Initialize empty log and finish-call records."""
        self.logged: list[tuple[dict[str, Any], int]] = []
        self.finish_calls = 0

    def log(self, payload: dict[str, Any], step: int) -> None:
        """Record one W&B log call."""
        self.logged.append((payload, step))

    def finish(self) -> None:
        """Record run finalization."""
        self.finish_calls += 1


class FakeWandb:
    """W&B module stub exposing ``init`` and ``Table``."""

    class Table:
        """W&B table stub."""

        def __init__(self, columns: list[str], data: list[list[Any]]) -> None:
            """Store table columns and rows."""
            self.columns = columns
            self.data = data

    def __init__(self) -> None:
        """Initialize fake module call records."""
        self.init_calls: list[dict[str, Any]] = []
        self.run = FakeRun()

    def init(self, **kwargs: Any) -> FakeRun:
        """Record initialization and return the fake run."""
        self.init_calls.append(kwargs)
        return self.run


def _build_tracker(rank: int, fake_wandb: FakeWandb, directory: Path) -> TrainingTracker:
    """Construct one tracker for a test rank."""
    return TrainingTracker(
        rank=rank,
        world_size=2,
        backends=("wandb",),
        project_name="hyper-rl",
        experiment_name="unit-test",
        resolved_config={"train": {"max_steps": 2}},
        wandb_mode="offline",
        wandb_directory=str(directory),
        wandb_module=fake_wandb,
    )


def test_only_rank_zero_initializes_and_logs(tmp_path: Path) -> None:
    """Verify non-zero ranks never call W&B."""
    fake_wandb = FakeWandb()
    tracker = _build_tracker(rank=1, fake_wandb=fake_wandb, directory=tmp_path)
    tracker.log({"train/total_loss": 1.0}, step=1)
    tracker.finish()
    expected_calls = 0
    actual_calls = len(fake_wandb.init_calls)
    assert actual_calls == expected_calls, (
        f"Non-zero rank initialized W&B: expected_calls={expected_calls}, got={actual_calls}"
    )


def test_rank_zero_logs_metrics_samples_and_finishes(tmp_path: Path) -> None:
    """Verify rank zero forwards steps, metrics, sample tables, and finish."""
    fake_wandb = FakeWandb()
    tracker = _build_tracker(rank=0, fake_wandb=fake_wandb, directory=tmp_path)
    samples = [
        {
            "step": 1,
            "rank": 0,
            "prompt": "p",
            "response": "r",
            "ground_truth": "1",
            "extracted_answer": "1",
            "reward": 1.0,
        }
    ]
    tracker.log(
        {"train/total_loss": 0.25, "validation/accuracy": 1.0},
        step=1,
        samples=samples,
        sample_tables={"validation/samples": samples},
    )
    tracker.finish()
    expected_init_calls = 1
    assert len(fake_wandb.init_calls) == expected_init_calls, (
        f"Unexpected W&B init count: expected={expected_init_calls}, got={len(fake_wandb.init_calls)}"
    )
    expected_step = 1
    actual_step = fake_wandb.run.logged[0][1]
    assert actual_step == expected_step, f"Unexpected logged step: expected={expected_step}, got={actual_step}"
    payload = fake_wandb.run.logged[0][0]
    assert "rollout/samples" in payload, f"Sample table missing from W&B payload: payload_keys={sorted(payload)}"
    assert "validation/samples" in payload, (
        f"Validation table missing from W&B payload: payload_keys={sorted(payload)}"
    )
    expected_finish_calls = 1
    assert fake_wandb.run.finish_calls == expected_finish_calls, (
        f"Unexpected finish count: expected={expected_finish_calls}, got={fake_wandb.run.finish_calls}"
    )


def test_sanitize_config_redacts_nested_secrets() -> None:
    """Verify W&B config upload redacts secrets without hiding normal token settings."""
    config = {
        "wandb": {"api_key": "private"},
        "auth": {"access_token": "also-private"},
        "model": {
            "weights_path": "/safe/path",
            "tokenizer_path": "/safe/tokenizer",
        },
        "rollout": {"max_new_tokens": 256},
    }
    sanitized = sanitize_config(config)
    assert sanitized["wandb"]["api_key"] == "***", (
        f"W&B API key was not redacted: got={sanitized['wandb']['api_key']}"
    )
    assert sanitized["auth"]["access_token"] == "***", (
        f"Access token was not redacted: got={sanitized['auth']['access_token']}"
    )
    expected_path = "/safe/path"
    actual_path = sanitized["model"]["weights_path"]
    assert actual_path == expected_path, f"Safe config changed: expected={expected_path}, got={actual_path}"
    assert sanitized["model"]["tokenizer_path"] == "/safe/tokenizer"
    assert sanitized["rollout"]["max_new_tokens"] == 256
