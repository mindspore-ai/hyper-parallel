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
"""Repository configuration and distributed controller-credential contracts."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rl.agentic.core import program_runner
from rl.agentic.codex import harness
from rl.config import _validate_codex_agentic, _validate_external_agentic_rollout


_CONFIG = {"version": "0.152.1", "executable": "codex", "session_root": "/tmp/repository-contract",
           "gateway_port": 8200, "timeout_seconds": 30, "request_timeout": 10, "model_context_window": 40960,
           "reward_callable": "examples.swe_smith.reward:score_submission", "admin_auth": True}


@pytest.mark.parametrize("shell", [None, "/bin/zsh"])
def test_codex_child_has_explicit_shell_without_mutating_parent(
        monkeypatch: pytest.MonkeyPatch, shell: str | None) -> None:
    """Parallel PTYs must not fall back to libc user lookup in a minimal environment."""
    monkeypatch.delenv("SHELL", raising=False)
    if shell is not None:
        monkeypatch.setenv("SHELL", shell)
    environment = harness._codex_environment("http://localhost:8200", Path("/tmp/codex-test"), "session")
    assert environment["SHELL"] == (shell or "/bin/bash")
    assert os.environ.get("SHELL") == shell


def test_custom_program_and_existing_text_configs_share_the_reward_extension() -> None:
    """Program choice changes task orchestration without changing the base reward contract."""
    _validate_codex_agentic({"codex": deepcopy(_CONFIG)})
    text = deepcopy(_CONFIG)
    text.pop("admin_auth")
    text["reward_callable"] = "examples.gsm8k.agent:score_codex_gsm8k_answer"
    _validate_codex_agentic({"codex": text})
    for value in (None, "", "missing-colon", "a:b:c"):
        invalid = {**_CONFIG, "reward_callable": value}
        with pytest.raises(ValueError):
            _validate_codex_agentic({"codex": invalid})


@pytest.mark.parametrize("override", [
    {"admin_auth": "yes"}, {"request_timeout": float("inf")}, {"max_response_bytes": 0},
    {"model_instructions": ""}, {"model_instructions": "  \n"}, {"model_instructions": ["fix"]},
])
def test_codex_rejects_invalid_shared_runtime_settings(override: dict) -> None:
    """Task-owned workspace rules do not expand the shared configuration validator."""
    with pytest.raises(ValueError):
        _validate_codex_agentic({"codex": {**_CONFIG, **override}})


def _external_rollout(model_limit: int, output_limit: int) -> dict:
    """Supply the external gateway fields relevant to the context reserve."""
    return {"engine": "vllm", "max_new_tokens": output_limit,
            "vllm": {"max_model_len": model_limit, "port": 8201,
                     "logprobs_mode": "raw_logprobs", "enable_auto_tool_choice": True,
                     "tool_call_parser": "hermes"}}


@pytest.mark.parametrize("context_window,model_limit,output_limit", [
    (16384, 16384, 4096), (10000, 8192, 512),
])
def test_repository_rejects_context_without_response_reserve(
        context_window: int, model_limit: int, output_limit: int) -> None:
    """Reject known compaction-threshold requests that can exceed vLLM's hard limit."""
    codex = {**_CONFIG, "model_context_window": context_window}
    with pytest.raises(ValueError, match="Repository Codex context reserve"):
        _validate_external_agentic_rollout("codex", "vllm", _external_rollout(model_limit, output_limit),
                                           {"codex": codex})


@pytest.mark.parametrize("context_window,model_limit,output_limit", [
    (16384, 16384, 1024), (32768, 32768, 2048), (14336, 16384, 2048), (8192, 16384, 4096),
])
def test_repository_accepts_context_with_response_reserve(
        context_window: int, model_limit: int, output_limit: int) -> None:
    """Allow either a shorter response or an earlier CLI compaction threshold."""
    codex = {**_CONFIG, "model_context_window": context_window}
    _validate_external_agentic_rollout("codex", "vllm", _external_rollout(model_limit, output_limit),
                                       {"codex": codex})


@pytest.mark.parametrize("runner", ["codex", "deepseek"])
def test_repository_context_reserve_does_not_change_other_runners(runner: str) -> None:
    """A text Codex or DeepSeek request keeps its existing vLLM validation."""
    config = {"gateway_port": 8200}
    if runner == "codex":
        config["reward_callable"] = "examples.gsm8k.agent:score_codex_gsm8k_answer"
    _validate_external_agentic_rollout(runner, "vllm", _external_rollout(16384, 4096),
                                       {runner: config})


def test_non_owner_runtime_receives_controller_secret_without_public_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """All owners use rank zero's credential; public configuration never holds that secret."""
    engine = SimpleNamespace(inference_base_url="http://backend", inference_model_name="policy",
                             synchronize_error=Mock())
    factory = Mock()
    runtime = program_runner.HarnessRuntime(engine, {"gateway_public_host": "model", "gateway_admin_host": "admin"},
                                             factory, "Codex", 8200, gateway_options={"admin_token": None})
    monkeypatch.setattr(program_runner.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(program_runner.dist, "barrier", lambda: None)

    def broadcast(values: list, src: int) -> None:
        """Represent the distributed credential published by the sole Gateway owner."""
        assert src == 0 and values == [None]
        values[0] = "rank-zero-secret"

    monkeypatch.setattr(program_runner.dist, "broadcast_object_list", broadcast)
    runtime.ensure_started()
    factory.assert_not_called()
    assert runtime.admin_token == "rank-zero-secret"
    assert runtime.admin_url == "http://admin:8200" and runtime.gateway_url == "http://model:8200"
    assert "rank-zero-secret" not in repr(runtime.config)
