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
"""CPU contracts for the Codex black-box Agentic RL integration."""

import http.client
import json
import os
import subprocess
import sys
import threading
import tomllib
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from examples.agents.gsm8k.agent import score_codex_gsm8k_answer
from rl.agentic.codex.gateway import CodexGateway, _Handler
from rl.agentic.codex.harness import CodexAgentProgram, _classify_codex_events
from rl.agentic.codex.protocol import CodexResponsesProtocol
from rl.agentic.codex.trajectory import build_codex_trajectory
from rl.agentic.core.program_runner import ProgramAgentRunner
from rl.dataset.contracts import Message, PromptRecord
from rl.roles.rollout.base import GenerationSettings


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
        "original_request": {"input": [{"type": "message", "role": "user"}]},
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


def test_codex_gsm8k_reward_reuses_strict_numeric_matching() -> None:
    """Codex final answers use the same GSM8K correctness contract."""
    prompt = _prompt()

    assert score_codex_gsm8k_answer("#### 42", prompt) == 1.0
    assert score_codex_gsm8k_answer("42", prompt) == 1.0
    assert score_codex_gsm8k_answer("#### 41", prompt) == 0.0


def test_gsm8k_codex_mvp_uses_local_shell_without_mcp() -> None:
    """The two-card MVP selects Codex and its built-in local shell only."""
    source_root = Path(__file__).resolve().parents[1]
    config_path = (
        source_root / "examples" / "agents" / "gsm8k" / "configs" / "codex_multi_turn.yaml"
    )
    launcher_path = source_root / "examples" / "scripts" / "run_qwen3_4b_agentic_docker.sh"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    launcher = launcher_path.read_text(encoding="utf-8")

    assert config["agentic"]["runner"] == "codex"
    assert config["agentic"]["max_turns"] == 2
    assert "mcp_servers" not in config["agentic"]["codex"]
    codex_config = config["agentic"]["codex"]
    instruction = codex_config["instruction_template"]
    assert config["rollout"]["max_new_tokens"] == 256
    assert codex_config["reasoning_effort"] == "low"
    assert instruction.startswith("/no_think\n")
    assert "local shell" in instruction
    assert "run Python" in instruction
    assert config["agentic"]["codex"]["reward_callable"].endswith(
        ":score_codex_gsm8k_answer"
    )
    assert config["rollout"]["vllm"]["data_parallel_size"] == 2
    assert config["train"]["accelerator"]["dp_shard"] == 2
    assert config["train"]["accelerator"]["activation_checkpoint"] == "full"
    assert (
        'HYPER_CODEX_IMAGE:=hyper-parallel/hyper-rl-codex:v0.22.1rc1'
        in launcher
    )
    assert "gsm8k_codex)" in launcher
    assert '[[ "${HYPER_RUN_TASK}" == *_codex ]]' in launcher
    assert '"${HYPER_RUN_TASK}" != *_codex' in launcher
    assert "learning_update_pattern=" in launcher
    assert "policy/fingerprint_changed=1" in launcher
    assert "reward/max=1" in launcher
    assert "reward/min=0" in launcher
    assert "train/gradient_norm=" in launcher
    assert "train/optimizer_steps=[1-9]" in launcher
    assert "Codex learning-update gate passed." in launcher


def test_codex_event_classifier_keeps_metadata_fallback_as_diagnostic() -> None:
    """Unknown local model metadata must not discard a completed episode."""
    warning = {
        "type": "item.completed",
        "item": {
            "type": "error",
            "message": (
                "Model metadata for `qwen3` not found. Defaulting to fallback "
                "metadata; this can degrade performance and cause issues."
            ),
        },
    }
    answer = {
        "type": "item.completed",
        "item": {"type": "agent_message", "text": "42"},
    }

    final_answer, diagnostics, failures = _classify_codex_events(
        [json.dumps(warning), json.dumps(answer)]
    )

    assert final_answer == "42"
    assert diagnostics == [warning]
    assert failures == []


@pytest.mark.parametrize(
    "event",
    [
        {"type": "turn.failed", "error": {"message": "backend failed"}},
        {"type": "error", "message": "protocol failed"},
        {
            "type": "item.completed",
            "item": {"type": "error", "message": "MCP server failed"},
        },
    ],
)
def test_codex_event_classifier_preserves_terminal_failures(event: dict) -> None:
    """Only the known metadata fallback is recoverable."""
    final_answer, diagnostics, failures = _classify_codex_events(
        [json.dumps(event)]
    )

    assert final_answer == ""
    assert diagnostics == []
    assert failures == [event]


def test_gateway_tolerates_codex_sse_disconnect() -> None:
    """A cancelled Codex request must not leak a BrokenPipeError from the gateway."""

    class DisconnectedStream:
        @staticmethod
        def write(_payload: bytes) -> None:
            raise BrokenPipeError("client closed")

        @staticmethod
        def flush() -> None:
            return

    protocol = SimpleNamespace(
        stream_events=lambda _result: iter(({"type": "response.completed"},))
    )
    handler = object.__new__(_Handler)
    handler.server = SimpleNamespace(state=SimpleNamespace(protocol=protocol))
    handler.wfile = DisconnectedStream()

    handler._write_sse_events({"id": "response-1"})


def test_gateway_wraps_backend_disconnect(monkeypatch) -> None:
    """A disappearing vLLM backend must surface a stable gateway error."""
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


def test_search_mcp_handshake_stays_independent_from_training_runtime(tmp_path) -> None:
    """The Codex MCP child starts without importing HyperParallel or an NPU backend."""
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps({"id": "doc-1", "title": "Answer", "text": "The answer is 42."}) + "\n",
        encoding="utf-8",
    )
    settings = json.dumps(
        {
            "search_corpus_path": str(corpus_path),
            "search_top_k": 2,
            "search_max_query_chars": 256,
            "search_max_document_chars": 300,
        }
    )
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "answer"}},
        },
    ]
    source_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source_root), existing_pythonpath) if value
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rl.agentic.mcp_server",
            "--factory",
            "examples.agents.search_R1.tools:build_codex_search_registry",
            "--settings-json",
            settings,
        ],
        input="".join(json.dumps(request) + "\n" for request in requests),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert responses[0]["id"] == 1
    assert responses[0]["result"]["protocolVersion"] == "2025-06-18"
    assert responses[1]["id"] == 2
    assert [tool["name"] for tool in responses[1]["result"]["tools"]] == ["search"]
    assert responses[2]["id"] == 3
    assert responses[2]["result"]["isError"] is False
    assert "The answer is 42." in responses[2]["result"]["content"][0]["text"]
    assert "torch_npu" not in completed.stderr
    assert "hyper_parallel" not in completed.stderr


def test_codex_config_explicitly_passes_pythonpath_to_mcp(
    tmp_path, monkeypatch
) -> None:
    """Codex must not choose an image-installed rl package for its MCP child."""
    mounted_pythonpath = (
        "/workspace/hyper-parallel/hyper_parallel/rl:/workspace/hyper-parallel"
    )
    monkeypatch.setenv("PYTHONPATH", mounted_pythonpath)
    program = object.__new__(CodexAgentProgram)
    program.gateway_url = "http://127.0.0.1:8200"
    program.config = {
        "mcp_servers": [
            {
                "name": "search",
                "command": "python",
                "args": ["-m", "rl.agentic.mcp_server"],
                "required": True,
                "inherit_env": ["PYTHONPATH"],
            }
        ]
    }

    program._write_codex_config(tmp_path, "session-id")

    config = tomllib.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
    assert config["mcp_servers"]["search"]["env"]["PYTHONPATH"] == mounted_pythonpath


def test_codex_config_rejects_missing_inherited_mcp_environment(
    tmp_path, monkeypatch
) -> None:
    """A missing source-path contract fails before Codex starts an opaque child."""
    monkeypatch.delenv("HYPER_TEST_MISSING_ENV", raising=False)
    program = object.__new__(CodexAgentProgram)
    program.gateway_url = "http://127.0.0.1:8200"
    program.config = {
        "mcp_servers": [
            {
                "name": "search",
                "command": "python",
                "inherit_env": ["HYPER_TEST_MISSING_ENV"],
            }
        ]
    }

    with pytest.raises(ValueError, match="requires an unset inherited environment"):
        program._write_codex_config(tmp_path, "session-id")


def test_protocol_preserves_tools_and_requests_exact_token_evidence() -> None:
    """The gateway asks vLLM for raw IDs/logprobs and round-trips shell calls."""
    protocol = CodexResponsesProtocol()
    request = protocol.transform_request(
        {
            "model": "ignored",
            "stream": True,
            "instructions": "Use tools when needed.",
            "input": "inspect the repository",
            "tools": [{"type": "local_shell", "description": "run a command"}],
        },
        "served-policy",
    )

    assert request["model"] == "served-policy"
    assert request["stream"] is False
    assert request["logprobs"] is True
    assert request["top_logprobs"] == 0
    assert request["return_token_ids"] is True
    assert request["tools"][0]["function"]["name"] == "shell"

    response = protocol.transform_response(
        {
            "id": "chat-1",
            "model": "served-policy",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
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
                                    "name": "shell",
                                    "arguments": '{"cmd":"pwd"}',
                                },
                            }
                        ],
                    }
                }
            ],
        },
        {"model": "codex-policy"},
    )
    assert response["output"][0]["type"] == "local_shell_call"
    assert response["output"][0]["action"]["commands"] == ["pwd"]
    assert list(protocol.stream_events(response))[-1]["type"] == "response.completed"


def test_protocol_round_trips_namespaced_mcp_tools() -> None:
    """Codex 0.152.1 namespace tools retain identity across vLLM turns."""
    protocol = CodexResponsesProtocol()
    namespace_tool = {
        "type": "namespace",
        "name": "search",
        "description": "Local document search",
        "tools": [
            {
                "type": "function",
                "name": "search",
                "description": "Search the corpus",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
    }
    original = {
        "model": "policy",
        "input": "find the answer",
        "tools": [namespace_tool],
        "tool_choice": "auto",
    }

    request = protocol.transform_request(original, "served-policy")
    alias = request["tools"][0]["function"]["name"]
    assert alias.startswith("search__search__")
    assert len(alias) <= 64

    response = protocol.transform_response(
        {
            "id": "chat-namespace",
            "model": "served-policy",
            "usage": {"prompt_tokens": 4, "completion_tokens": 3},
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-search",
                                "type": "function",
                                "function": {
                                    "name": alias,
                                    "arguments": '{"query":"answer"}',
                                },
                            }
                        ],
                    }
                }
            ],
        },
        original,
    )
    call = response["output"][0]
    assert call["type"] == "function_call"
    assert call["namespace"] == "search"
    assert call["name"] == "search"

    followup = protocol.transform_request(
        {
            "model": "policy",
            "tools": [namespace_tool],
            "input": [
                call,
                {
                    "type": "function_call_output",
                    "call_id": "call-search",
                    "output": "The answer is 42.",
                },
            ],
        },
        "served-policy",
    )
    followup_call = followup["messages"][0]["tool_calls"][0]
    assert followup_call["function"]["name"] == alias
    assert followup["messages"][1]["tool_call_id"] == "call-search"
    done_events = [
        event
        for event in protocol.stream_events(response)
        if event["type"] == "response.function_call_arguments.done"
    ]
    assert done_events[0]["namespace"] == "search"
    assert done_events[0]["name"] == "search"


def test_trajectory_merges_raw_actions_and_canonical_tool_interstitials() -> None:
    """Only model-sampled tokens train, and every sampled logprob stays exact."""
    records = [
        _completion(0, [10, 11], [20, 99], [-0.1, -0.2]),
        _completion(1, [10, 11, 20, 99, 30], [21, 99], [-0.3, -0.4]),
    ]

    trajectory = build_codex_trajectory(
        prompt=_prompt(),
        policy_version=3,
        policy_fingerprint="digest-3",
        sample_index=0,
        completion_records=records,
        reward=1.0,
        reward_components={"outcome": 1.0},
        end_of_turn_token_id=99,
    )

    assert trajectory.token_ids.tolist() == [10, 11, 20, 99, 30, 21, 99]
    assert trajectory.action_mask.tolist() == [False, False, True, True, False, True, True]
    assert trajectory.rollout_log_probs.tolist() == pytest.approx(
        [0.0, -0.1, -0.2, 0.0, -0.3, -0.4]
    )
    assert trajectory.worker_policy_fingerprint == "digest-3"
    assert len(trajectory.metadata["tool_history"]) == 2


def test_trajectory_fails_closed_on_missing_logprobs_or_prefix_rewrite() -> None:
    """Lossy backend evidence and rewritten history never reach GRPO."""
    missing = _completion(0, [10, 11], [20, 99], [-0.1])
    with pytest.raises(ValueError, match="logprobs must align"):
        build_codex_trajectory(
            prompt=_prompt(),
            policy_version=0,
            policy_fingerprint="digest",
            sample_index=0,
            completion_records=[missing],
            reward=0.0,
            reward_components={},
            end_of_turn_token_id=99,
        )

    rewritten = [
        _completion(0, [10, 11], [20, 99], [-0.1, -0.2]),
        _completion(1, [10, 12, 20, 99, 30], [21, 99], [-0.3, -0.4]),
    ]
    with pytest.raises(ValueError, match="rewrote its canonical prompt prefix"):
        build_codex_trajectory(
            prompt=_prompt(),
            policy_version=0,
            policy_fingerprint="digest",
            sample_index=0,
            completion_records=rewritten,
            reward=0.0,
            reward_components={},
            end_of_turn_token_id=99,
        )


class _BackendHandler(BaseHTTPRequestHandler):
    payloads: list[dict] = []

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


def _request_json(url: str, payload: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def test_gateway_records_every_policy_bound_completion(tmp_path) -> None:
    """One gateway request produces one inspectable exact completion record."""
    _BackendHandler.payloads = []
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _BackendHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    gateway = CodexGateway(
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
                "generation": {"temperature": 0.7, "max_tokens": 32},
            },
        )
        result = _request_json(
            f"{gateway_url}/responses",
            {"model": "codex", "input": "question", "stream": False},
            "session-1",
        )
        with urllib.request.urlopen(
            f"{gateway_url}/internal/sessions/session-1",
            timeout=5,
        ) as response:
            captured = json.loads(response.read())
    finally:
        gateway.close()
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=5)

    assert result["output"][0]["content"][0]["text"] == "42"
    assert captured["policy_version"] == 7
    assert captured["policy_fingerprint"] == "digest-7"
    assert len(captured["completions"]) == 1
    assert _BackendHandler.payloads[0]["return_token_ids"] is True
    assert _BackendHandler.payloads[0]["temperature"] == 0.7
    assert _BackendHandler.payloads[0]["max_tokens"] == 32
    assert (tmp_path / "gateway-events.jsonl").is_file()


def test_program_runner_rehydrates_owner_trajectory_on_tp_sibling() -> None:
    """A non-owner rank never launches a second harness and receives exact data."""
    trajectory = build_codex_trajectory(
        prompt=_prompt(),
        policy_version=0,
        policy_fingerprint="digest",
        sample_index=0,
        completion_records=[_completion(0, [10, 11], [20, 99], [-0.1, -0.2])],
        reward=1.0,
        reward_components={"outcome": 1.0},
        end_of_turn_token_id=99,
    )
    payload = ProgramAgentRunner._serialize_trajectories((trajectory,))

    class SiblingEngine:
        is_request_owner = False

        @staticmethod
        def synchronize_error(error, _operation):
            assert error is None

        @staticmethod
        def synchronize_agent_payload(local_payload):
            assert local_payload is None
            return payload

    runner = ProgramAgentRunner(
        program_factory=lambda *_args: pytest.fail("TP sibling launched an agent"),
        num_samples=1,
        settings=GenerationSettings(
            max_new_tokens=16,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            do_sample=True,
            pad_token_id=0,
            eos_token_id=99,
            collect_log_probs=True,
        ),
        engine=SiblingEngine(),
    )

    batch = runner.rollout([_prompt()], policy_version=0)
    assert batch.sequences.tolist() == [[10, 11, 20, 99]]
    assert batch.old_log_probs[0].tolist() == pytest.approx([0.0, -0.1, -0.2])
