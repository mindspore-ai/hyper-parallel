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
"""CPU contracts for the isolated DeepSeek Harness Agentic RL integration."""

import http.client
import json
import os
import subprocess
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
import yaml
from examples.agents.gsm8k.agent import score_deepseek_gsm8k_answer
from rl.agentic.deepseek.gateway import DeepSeekGateway, _Handler
from rl.agentic.deepseek.harness import DeepSeekAgentProgram
from rl.agentic.deepseek.protocol import DeepSeekChatProtocol
from rl.agentic.deepseek.trajectory import build_deepseek_trajectory
from rl.config import _validate_agentic
from rl.dataset.contracts import Message, PromptRecord


def _prompt() -> PromptRecord:
    return PromptRecord(
        prompt_id="question-1",
        messages=(Message(role="user", content="What is the answer?"),),
        ground_truth="42",
        metadata={"input_ids": torch.tensor([10, 11], dtype=torch.long)},
    )


def _completion(
    ordinal: int,
    prompt_ids: list[int],
    response_ids: list[int],
    logprobs: list[float],
) -> dict:
    return {
        "ordinal": ordinal,
        "original_request": {"messages": [{"role": "user", "content": "question"}]},
        "request": {"messages": [{"role": "user", "content": "question"}]},
        "response": {
            "prompt_token_ids": prompt_ids,
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                    "token_ids": response_ids,
                    "logprobs": {
                        "content": [
                            {"token_id": token_id, "logprob": logprob}
                            for token_id, logprob in zip(response_ids, logprobs)
                        ]
                    },
                }
            ],
        },
    }


def _deepseek_agentic_config() -> dict:
    return {
        "runner": "deepseek",
        "module_path": "examples.agents.gsm8k.agent",
        "environment": "gsm8k_tools",
        "interaction_mode": "multi_turn",
        "max_turns": 2,
        "max_observation_tokens": 0,
        "apply_chat_template": True,
        "deepseek": {
            "version": "0.1.1rc1",
            "provider": "deepseek-official",
            "model": "qwen3",
            "reasoning_effort": "off",
            "required_tool_calls": 1,
            "required_tool_name": "bash",
            "session_root": "/tmp/deepseek",
            "reward_callable": (
                "examples.agents.gsm8k.agent:score_deepseek_gsm8k_answer"
            ),
            "gateway_port": 8300,
            "timeout_seconds": 60,
            "request_timeout": 30,
        },
    }


def test_deepseek_config_and_example_are_independent_from_codex() -> None:
    """The runner owns a separate config section and a separate peer package."""
    _validate_agentic(_deepseek_agentic_config())
    source_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (
            source_root
            / "examples"
            / "agents"
            / "gsm8k"
            / "configs"
            / "deepseek_multi_turn.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["agentic"]["runner"] == "deepseek"
    assert "deepseek" in config["agentic"]
    assert "codex" not in config["agentic"]
    assert config["agentic"]["deepseek"]["gateway_port"] == 8300
    assert config["agentic"]["deepseek"]["reasoning_effort"] == "off"
    assert config["agentic"]["deepseek"]["required_tool_calls"] == 1
    assert config["agentic"]["deepseek"]["required_tool_name"] == "bash"
    assert config["rollout"]["vllm"]["gpu_memory_utilization"] == 0.15
    assert config["rollout"]["vllm"]["kv_cache_memory_bytes"] == 1073741824
    assert config["rollout"]["vllm"]["max_model_len"] == 2048
    assert config["agentic"]["max_episode_tokens"] == 2048
    assert config["train"]["learning_gate"] == {
        "enabled": False,
        "min_gradient_norm": 0.0,
        "require_mixed_rewards": False,
        "require_fingerprint_change": False,
    }

    launcher = (
        source_root / "examples" / "scripts" / "run_qwen3_4b_deepseek_agentic_docker.sh"
    ).read_text(encoding="utf-8")
    assert 'HYPER_DEEPSEEK_REQUIRE_LEARNING_UPDATE:=true' in launcher
    assert 'HYPER_DEEPSEEK_GPU_MEMORY_UTILIZATION:=0.15' in launcher
    assert 'HYPER_DEEPSEEK_KV_CACHE_MEMORY_BYTES:=1073741824' in launcher
    assert 'HYPER_DEEPSEEK_MAX_MODEL_LEN:=2048' in launcher
    assert 'reward/max=1(\\.0+)?[, ].*reward/min=0(\\.0+)?' in launcher


def test_deepseek_image_inherits_the_common_hyper_rl_runtime() -> None:
    """The optional Harness layer must not rebuild or inherit the Codex image."""
    source_root = Path(__file__).resolve().parents[1]
    dockerfile = (source_root / "docker" / "Dockerfile.deepseek").read_text(
        encoding="utf-8"
    )
    builder = (source_root / "docker" / "build_deepseek_image.sh").read_text(
        encoding="utf-8"
    )
    assert "ARG BASE_IMAGE=hyper-parallel/hyper-rl:v0.22.1rc1" in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert 'org.opencontainers.image.base.name="${BASE_IMAGE}"' in dockerfile
    assert "FLASH_ATTN_NPU_WHEEL" not in dockerfile
    assert "hyper-rl-codex" not in dockerfile
    assert "HYPER_RL_FA3_WHEEL" not in builder
    assert '--build-arg "BASE_IMAGE=${base_image}"' in builder


def test_deepseek_config_rejects_unvalidated_sdk_version() -> None:
    """Wire behavior remains pinned to the SDK release used by the adapter."""
    config = _deepseek_agentic_config()
    config["deepseek"]["version"] = "0.1.2"
    with pytest.raises(ValueError, match="validated only.*0.1.1rc1"):
        _validate_agentic(config)


def test_deepseek_config_rejects_unknown_reasoning_effort() -> None:
    """Only values understood by the pinned provider may reach the gateway."""
    config = _deepseek_agentic_config()
    config["deepseek"]["reasoning_effort"] = "medium"
    with pytest.raises(ValueError, match="reasoning_effort must be"):
        _validate_agentic(config)


def test_deepseek_episode_pins_isolated_non_thinking_settings(tmp_path) -> None:
    """Ambient Harness settings cannot silently restore high-effort thinking."""
    program = object.__new__(DeepSeekAgentProgram)
    program.config = {
        "provider": "deepseek-official",
        "model": "qwen3",
        "reasoning_effort": "off",
    }

    dsh_home = program._write_dsh_settings(tmp_path)

    settings = json.loads((dsh_home / "settings.yaml").read_text(encoding="utf-8"))
    assert dsh_home == tmp_path / "dsh-home"
    assert settings == {
        "agent-default-model": {
            "provider": "deepseek-official",
            "model": "qwen3",
            "reasoningEffort": "off",
        }
    }


def test_deepseek_episode_enforces_tool_call_contract() -> None:
    """A text-only answer cannot masquerade as a tool-assisted trajectory."""
    program = object.__new__(DeepSeekAgentProgram)
    program.policy_version = 7
    program.policy_fingerprint = "digest-7"
    program.config = {"required_tool_calls": 1, "required_tool_name": "bash"}
    captured = {
        "policy_version": 7,
        "policy_fingerprint": "digest-7",
        "completions": [_completion(0, [10, 11], [20, 99], [-0.1, -0.2])],
    }

    with pytest.raises(RuntimeError, match="tool-call contract"):
        program._validate_capture(captured)

    captured["completions"][0]["response"]["choices"][0]["message"][
        "tool_calls"
    ] = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command":"true"}'},
        }
    ]
    program._validate_capture(captured)


def test_deepseek_tool_mistake_becomes_rewardable_episode_evidence() -> None:
    """A stochastic tool mistake must not abort every other sample in the batch."""
    program = object.__new__(DeepSeekAgentProgram)
    program.policy_version = 7
    program.policy_fingerprint = "digest-7"
    program.config = {"required_tool_calls": 1, "required_tool_name": "bash"}
    captured = {
        "policy_version": 7,
        "policy_fingerprint": "digest-7",
        "completions": [_completion(0, [10, 11], [20, 99], [-0.1, -0.2])],
    }

    error = program._capture_contract_error(captured)

    assert error is not None
    assert "expected=1, actual=0" in error


def test_codex_import_does_not_require_or_load_deepseek_sdk() -> None:
    """The existing Codex runner stays usable without the optional DeepSeek SDK."""
    source_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(source_root),
            str(source_root.parents[1]),
            environment.get("PYTHONPATH"),
        )
        if value
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import rl.agentic.codex; "
                "assert 'deepseek_harness' not in sys.modules"
            ),
        ],
        cwd=source_root,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_deepseek_reward_uses_strict_gsm8k_numeric_matching() -> None:
    prompt = _prompt()
    assert score_deepseek_gsm8k_answer("#### 42", prompt) == 1.0
    assert score_deepseek_gsm8k_answer("#### 41", prompt) == 0.0


def test_deepseek_protocol_requests_evidence_and_emits_chat_sse() -> None:
    """The independent adapter captures raw evidence and preserves tool calls."""
    protocol = DeepSeekChatProtocol()
    request = protocol.transform_request(
        {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "calculate"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "reasoning_effort": "off",
        },
        "served-policy",
    )
    assert request["model"] == "served-policy"
    assert request["stream"] is False
    assert request["logprobs"] is True
    assert request["return_token_ids"] is True
    assert request["chat_template_kwargs"]["enable_thinking"] is False
    assert request["tools"][0]["function"]["name"] == "bash"

    events = list(
        protocol.stream_events(
            {
                "id": "chat-1",
                "model": "served-policy",
                "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": '{"command":"python -V"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        )
    )
    assert events[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call-1"
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[-1]["usage"]["completion_tokens"] == 2


def test_deepseek_trajectory_merges_actions_and_tool_interstitials() -> None:
    """Only sampled model tokens are trainable across multiple Harness calls."""
    trajectory = build_deepseek_trajectory(
        prompt=_prompt(),
        policy_version=3,
        policy_fingerprint="digest-3",
        sample_index=0,
        completion_records=[
            _completion(0, [10, 11], [20, 99], [-0.1, -0.2]),
            _completion(1, [10, 11, 20, 99, 30], [21, 99], [-0.3, -0.4]),
        ],
        reward=1.0,
        reward_components={"outcome": 1.0},
        end_of_turn_token_id=99,
    )
    assert trajectory.token_ids.tolist() == [10, 11, 20, 99, 30, 21, 99]
    assert trajectory.action_mask.tolist() == [
        False,
        False,
        True,
        True,
        False,
        True,
        True,
    ]
    assert trajectory.rollout_log_probs.tolist() == pytest.approx(
        [0.0, -0.1, -0.2, 0.0, -0.3, -0.4]
    )
    assert trajectory.metadata["runner"] == "deepseek"
    assert trajectory.worker_policy_fingerprint == "digest-3"


def test_deepseek_trajectory_fails_closed_on_lossy_evidence() -> None:
    missing = _completion(0, [10, 11], [20, 99], [-0.1])
    with pytest.raises(ValueError, match="logprobs must align"):
        build_deepseek_trajectory(
            prompt=_prompt(),
            policy_version=0,
            policy_fingerprint="digest",
            sample_index=0,
            completion_records=[missing],
            reward=0.0,
            reward_components={},
            end_of_turn_token_id=99,
        )


def test_deepseek_trajectory_allows_bounded_generation_prefill_rewrite() -> None:
    """Qwen may canonicalize the short non-thinking prefill after generation."""
    trajectory = build_deepseek_trajectory(
        prompt=_prompt(),
        policy_version=0,
        policy_fingerprint="digest",
        sample_index=0,
        completion_records=[
            _completion(0, [10, 11, 12, 13], [20, 99], [-0.1, -0.2]),
            _completion(
                1,
                [10, 11, 12, 14, 20, 99, 30],
                [21, 99],
                [-0.3, -0.4],
            ),
        ],
        reward=1.0,
        reward_components={"outcome": 1.0},
        end_of_turn_token_id=99,
    )

    assert trajectory.token_ids.tolist() == [10, 11, 12, 13, 20, 99, 30, 21, 99]
    assert trajectory.action_mask.tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        True,
        True,
    ]


def test_deepseek_trajectory_rejects_prompt_body_rewrite() -> None:
    """Only a short template-owned generation suffix may change between calls."""
    first_prompt = list(range(30))
    second_prompt = [0, 100, *range(2, 30), 20, 99, 30]
    with pytest.raises(ValueError, match="rewrote its canonical prompt body"):
        build_deepseek_trajectory(
            prompt=_prompt(),
            policy_version=0,
            policy_fingerprint="digest",
            sample_index=0,
            completion_records=[
                _completion(0, first_prompt, [20, 99], [-0.1, -0.2]),
                _completion(1, second_prompt, [21, 99], [-0.3, -0.4]),
            ],
            reward=0.0,
            reward_components={},
            end_of_turn_token_id=99,
        )


def test_deepseek_trajectory_reports_backend_token_limit() -> None:
    """A length-limited model response must remain visible in rollout metrics."""
    completion = _completion(0, [10, 11], [20, 99], [-0.1, -0.2])
    completion["response"]["choices"][0]["finish_reason"] = "length"

    trajectory = build_deepseek_trajectory(
        prompt=_prompt(),
        policy_version=0,
        policy_fingerprint="digest",
        sample_index=0,
        completion_records=[completion],
        reward=0.0,
        reward_components={},
        end_of_turn_token_id=99,
    )

    assert trajectory.done
    assert trajectory.truncated
    assert trajectory.terminal_reason == "max_tokens"


def test_deepseek_trajectory_reports_harness_failure_as_truncated() -> None:
    """Captured policy actions survive a Harness-level terminal failure."""
    trajectory = build_deepseek_trajectory(
        prompt=_prompt(),
        policy_version=0,
        policy_fingerprint="digest",
        sample_index=0,
        completion_records=[
            _completion(0, [10, 11], [20, 99], [-0.1, -0.2])
        ],
        reward=0.0,
        reward_components={"tool_contract": 0.0},
        end_of_turn_token_id=99,
        metadata={"finish_reason": "error"},
    )

    assert trajectory.done
    assert trajectory.truncated
    assert trajectory.terminal_reason == "harness_error"


class _BackendHandler(BaseHTTPRequestHandler):
    payloads: ClassVar[list[dict]] = []

    def do_POST(self) -> None:  # pylint: disable=C0103
        length = int(self.headers["Content-Length"])
        self.payloads.append(json.loads(self.rfile.read(length)))
        body = json.dumps(
            {
                "id": "chat-1",
                "model": "served-policy",
                "prompt_token_ids": [10, 11],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "42"},
                        "finish_reason": "stop",
                        "token_ids": [20, 99],
                        "logprobs": {
                            "content": [
                                {"token_id": 20, "logprob": -0.1},
                                {"token_id": 99, "logprob": -0.2},
                            ]
                        },
                    }
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _request_json(
    url: str, payload: dict, headers: dict[str, str] | None = None
) -> dict:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=request_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def test_deepseek_gateway_records_policy_bound_chat_completion(tmp_path) -> None:
    """The Harness session header selects an inspectable exact completion trace."""
    _BackendHandler.payloads = []
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _BackendHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    gateway = DeepSeekGateway(
        "127.0.0.1",
        0,
        f"http://127.0.0.1:{backend.server_port}",
        "served-policy",
        5.0,
    )
    gateway.start()
    gateway_url = f"http://127.0.0.1:{gateway.address[1]}"
    try:
        _request_json(
            f"{gateway_url}/internal/sessions",
            {
                "session_id": "session-1",
                "policy_version": 7,
                "policy_fingerprint": "digest-7",
                "artifact_dir": str(tmp_path),
                "max_completions": 2,
                "generation": {
                    "temperature": 0.7,
                    "max_tokens": 32,
                    "reasoning_effort": "off",
                },
            },
        )
        request = urllib.request.Request(
            f"{gateway_url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "deepseek",
                    "messages": [{"role": "user", "content": "question"}],
                    "stream": True,
                    "reasoning_effort": "high",
                }
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "x-deepseek-harness-session-id": "session-1",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            sse = response.read().decode()
        with urllib.request.urlopen(
            f"{gateway_url}/internal/sessions/session-1", timeout=5
        ) as response:
            captured = json.loads(response.read())
    finally:
        gateway.close()
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=5)
    assert '"content": "42"' in sse
    assert "data: [DONE]" in sse
    assert captured["policy_version"] == 7
    assert captured["policy_fingerprint"] == "digest-7"
    assert len(captured["completions"]) == 1
    assert _BackendHandler.payloads[0]["return_token_ids"] is True
    assert _BackendHandler.payloads[0]["stream"] is False
    assert _BackendHandler.payloads[0]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert "reasoning_effort" not in _BackendHandler.payloads[0]
    assert (tmp_path / "gateway-events.jsonl").is_file()


def test_deepseek_gateway_wraps_backend_disconnect(monkeypatch) -> None:
    """A disappearing vLLM backend surfaces a stable DeepSeek gateway error."""
    handler = object.__new__(_Handler)
    handler.server = SimpleNamespace(
        state=SimpleNamespace(
            backend_url="http://127.0.0.1:1",
            request_timeout=1.0,
        )
    )

    def disconnect(*_args, **_kwargs):
        raise http.client.RemoteDisconnected("backend closed")

    monkeypatch.setattr(urllib.request, "urlopen", disconnect)
    with pytest.raises(
        RuntimeError,
        match="vLLM chat completion connection closed before a response",
    ):
        handler._backend_request({"model": "policy"})
