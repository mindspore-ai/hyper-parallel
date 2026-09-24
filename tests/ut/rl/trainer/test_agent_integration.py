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
"""Episode reporting and inference shutdown contracts for agent training."""

import signal
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from rl.algorithm import build_algorithm
from rl.config import validate_config
from rl.dataset.data_source import collate_prompt_samples
from rl.evaluation import Evaluator
from rl.roles.rollout import vllm as vllm_module
from rl.trainer import SyncTrainer
from rl.utils.monitoring import metrics as metrics_module



def _episode_rows() -> tuple:
    """One successful one-call episode and one failed three-call episode."""
    return tuple(SimpleNamespace(
        prompt_id=str(episode), group_id=str(episode), policy_version=0, worker_policy_version=0,
        reward=float(episode == 0), reward_components={"success": float(episode == 0)},
        truncated=False, terminal_reason="completed",
        metadata={"episode_id": str(episode), "call_index": call, "call_count": count},
    ) for episode, count in enumerate((1, 3)) for call in range(count))


def test_episode_reporting_excludes_extra_calls_and_dp_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accuracy counts episodes, response length sums calls, and padding contributes nothing."""
    rows = _episode_rows()
    rollout = SimpleNamespace(trajectories=rows, rewards=torch.tensor([1., 0., 0., 0.]),
                              action_mask=torch.tensor([[False, True, True]] * 4),
                              responses=("right", "call1", "call2", "wrong"), generation_seconds=2.)
    dataset = [{"sample_index": i, "source_prompt": str(i), "prompt": str(i),
                "input_ids": torch.tensor([10]), "attention_mask": torch.tensor([1]), "ground_truth": "1"}
               for i in range(2)]
    evaluator = Evaluator(dataset=dataset, collate_fn=partial(collate_prompt_samples, pad_token_id=0),
                          rollout_manager=SimpleNamespace(generate=lambda **_kwargs: rollout),
                          device=torch.device("cpu"), batch_size=2, max_samples=2, log_samples=2, progress_steps=0)
    record = evaluator._evaluate_batch(1, 0, [(0, True), (1, True)], 2)
    assert (record["correct"], record["total"], record["generated_tokens"]) == (1., 2, 8)
    assert record["samples"][1]["response"] == "call1\ncall2\nwrong"
    invalid = evaluator._evaluate_batch(1, 0, [(0, True), (1, False)], 2)
    assert (invalid["total"], invalid["generated_tokens"]) == (1, 2)
    rollout.trajectories += (SimpleNamespace(metadata={"dp_padding": True}),)
    rollout.rewards = torch.cat((rollout.rewards, torch.tensor([999.])))
    rollout.action_mask = torch.cat((rollout.action_mask, torch.zeros((1, 3), dtype=torch.bool)))
    rollout.responses += ("padding",)
    monkeypatch.setattr(metrics_module.dist, "get_rank", lambda: 0)
    local = metrics_module._local_rollout_record(
        rollout, collate_prompt_samples(dataset, pad_token_id=0), step=1, sample_limit=3,
    )
    metrics = metrics_module._rollout_metrics([local])
    assert metrics["rollout/sequence_count"] == 2
    assert metrics["reward/accuracy"] == metrics["reward/mean"] == 0.5
    assert metrics["rollout/response_length_mean"] == 4
    assert metrics["rollout/generated_tokens"] == 8
    assert len(local["samples"]) == 2


@pytest.mark.parametrize("alive, stubborn", [(True, False), (False, False), (False, True)])
def test_server_shutdown_drains_children_before_escalation(
    monkeypatch: pytest.MonkeyPatch, alive: bool, stubborn: bool,
) -> None:
    """An exited parent does not justify killing descendants still unlinking IPC resources."""
    process = SimpleNamespace(pid=8123, poll=lambda: None if alive else 0, wait=Mock(return_value=0))
    client = vllm_module._VLLMHTTPClient(process, "http://localhost:1", "model", 1)
    monkeypatch.setattr(client, "_close_async_runtime", Mock())
    wait = Mock(side_effect=[RuntimeError("busy"), None] if stubborn else None)
    monkeypatch.setattr(client, "_wait_process_group_exit", wait)
    kill = Mock()
    monkeypatch.setattr(vllm_module.os, "killpg", kill)
    client.close()
    assert [call.args[1] for call in kill.call_args_list] == (
        [signal.SIGTERM, signal.SIGKILL] if stubborn else [signal.SIGTERM])
    wait.assert_any_call(8123, timeout=5)
    assert client._process is None
    client.close()
    assert kill.call_count == (2 if stubborn else 1)


def test_final_checkpoint_shutdown_is_once_and_preserves_retry_on_failure() -> None:
    """The engine is released before training-state release, and failed close is not marked complete."""
    trainer = object.__new__(SyncTrainer)
    events = []
    trainer.rollout_engine = SimpleNamespace(close=lambda: events.append("close"))
    trainer._release_training_state_for_rollout = lambda: events.append("release")
    trainer._close_rollout_for_final_checkpoint()
    trainer._close_rollout_for_final_checkpoint()
    assert events == ["close", "release"]
    trainer._rollout_closed_for_final_checkpoint = False
    trainer.rollout_engine.close = Mock(side_effect=RuntimeError("close failed"))
    with pytest.raises(RuntimeError, match="close failed"):
        trainer._close_rollout_for_final_checkpoint()
    assert not trainer._rollout_closed_for_final_checkpoint


@pytest.mark.parametrize("runner", ["codex", "deepseek"])
def test_segmented_harness_ppo_rejected_before_model_initialization(runner: str) -> None:
    """Unsupported segmented value targets fail before device or checkpoint construction."""
    algorithm = build_algorithm({"name": "ppo", "loss_aggregation": "token-mean"})
    with pytest.raises(ValueError, match="Segmented .*GRPO"):
        validate_config({"model": {}, "data": {}, "rollout": {}, "agentic": {"runner": runner}}, algorithm)
