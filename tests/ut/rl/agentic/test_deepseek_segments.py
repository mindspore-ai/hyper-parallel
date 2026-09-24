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
"""DeepSeek runtime uses actual call contexts while retaining its status semantics."""

import asyncio
from pathlib import Path
from typing import Any, Optional

import pytest
import torch

from rl.agentic.core.types import RewardResult
from rl.agentic.ds_harness import harness
from rl.dataset.contracts import Message, PromptRecord



def _completion(ordinal: int, prompt: list[int], action: list[int]) -> dict:
    return {"ordinal": ordinal, "request": {"messages": []},
            "original_request": {"messages": [{"role": "tool", "content": "real tool output"}]},
            "response": {"prompt_token_ids": prompt, "choices": [{"token_ids": action, "finish_reason": "stop",
                "message": {"content": "answer"},
                "logprobs": {"content": [{"token_id": token, "logprob": -0.25} for token in action]}}]}}


@pytest.mark.parametrize("finish_reason,call_finish,terminal,model_failure", [
    ("completed", "stop", "completed", None),
    ("error", "stop", "harness_error", "call_budget_exhausted"),
    ("aborted", "stop", "harness_aborted", "call_budget_exhausted"),
    ("completed", "length", "max_tokens", None),
    ("error", "stop", "unknown", None), ("aborted", "stop", "unknown", None),
    ("error", "stop", "harness_error", "invalid_json"),
    ("error", "stop", "harness_error", "invalid_tool_schema"),
])
def test_deepseek_runtime_retains_rewritten_context_and_episode_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish_reason: str, call_finish: str,
    terminal: str, model_failure: Optional[str],
) -> None:
    """Real per-call contexts replace unsafe stitching, without changing DeepSeek reward/status rules."""
    prompt = PromptRecord("prompt", (Message("user", "question"),), "1", {"input_ids": torch.tensor([1, 2])})
    records = [_completion(0, [1, 2, 30, 31], [4, 99]), _completion(1, [7, 8], [5, 99])]
    records[1]["response"]["choices"][0]["finish_reason"] = call_finish
    requests = []

    def http_json(method: str, unused_url: str, unused_payload: Any, unused_timeout: float) -> dict:
        """Return recorded calls through the unchanged gateway session lifecycle."""
        del unused_url, unused_payload, unused_timeout
        requests.append(method)
        return {"policy_version": 1, "completions": records, "failure": (
            {"failure_origin": "model", "failure_reason": model_failure, "trainable": True}
            if model_failure else None
        )} if method == "GET" else {}

    monkeypatch.setattr(harness, "_http_json", http_json)
    monkeypatch.setattr(harness, "_load_reward_callable", lambda _value: lambda _answer, _prompt: RewardResult(1.0))
    config = {"session_root": str(tmp_path), "max_turns": 2, "max_new_tokens": 8, "temperature": 1.0,
              "top_p": 1.0, "top_k": 0, "max_episode_tokens": 64, "timeout_seconds": 60}
    program = harness.DeepSeekAgentProgram(prompt, 1, 0, "http://gateway/v1", "http://gateway", config, 99)
    monkeypatch.setattr(program, "_run_harness", lambda *_args: ("1", finish_reason, []))
    if terminal == "unknown":
        with pytest.raises(RuntimeError, match="unknown failure origin"):
            asyncio.run(program.run())
        assert requests == ["POST", "GET", "DELETE"]
        return
    rows = asyncio.run(program.run())
    assert len(rows) == 2 and requests == ["POST", "GET", "DELETE"]
    for index, row in enumerate(rows):
        expected_prompt = records[index]["response"]["prompt_token_ids"]
        expected_action = records[index]["response"]["choices"][0]["token_ids"]
        assert row.token_ids.tolist() == expected_prompt + expected_action
        assert row.action_mask.tolist() == [False] * len(expected_prompt) + [True] * len(expected_action)
        torch.testing.assert_close(row.rollout_log_probs[row.action_mask[1:]], torch.tensor([-0.25, -0.25]))
        assert row.metadata["call_index"] == index and row.metadata["call_count"] == 2
        assert row.metadata["harness_finish_reason"] == finish_reason
        assert row.metadata["finish_reason"] == records[index]["response"]["choices"][0]["finish_reason"]
        assert row.metadata["tool_history"] == records[index]["original_request"]["messages"]
        assert row.done and row.truncated == (terminal != "completed")
        assert row.terminal_reason == terminal
        assert row.reward == (0.0 if finish_reason in {"error", "aborted"} else 1.0)
    with pytest.raises(ValueError, match="exact sampled-action prefix"):
        harness.build_deepseek_trajectory(prompt=prompt, policy_version=1, sample_index=0,
            completion_records=records, reward=1.0, reward_components={}, end_of_turn_token_id=99)


@pytest.mark.parametrize("failure", [
    {"failure_origin": "unknown", "failure_reason": "sdk_error", "trainable": False},
    {"failure_origin": "infrastructure", "failure_reason": "backend_error", "trainable": False},
    {"failure_origin": "model", "failure_reason": "call_budget_exhausted", "trainable": False},
])
def test_deepseek_rejects_sdk_errors_without_trainable_model_evidence(failure: Any) -> None:
    """SDK errors cannot silently become zero-reward samples without known model evidence."""
    with pytest.raises(RuntimeError, match="not trainable|unknown failure origin"):
        harness._validated_failure({"failure": failure}, "error")  # pylint: disable=protected-access


def test_deepseek_rejects_gateway_infrastructure_failure_after_sdk_completion() -> None:
    """A successful SDK status cannot erase a gateway infrastructure failure."""
    failure = {"failure_origin": "infrastructure", "failure_reason": "protocol_error", "trainable": False}
    with pytest.raises(RuntimeError, match="not trainable"):
        harness._validated_failure({"failure": failure}, "completed")  # pylint: disable=protected-access
