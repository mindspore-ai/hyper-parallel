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
"""CPU unit tests for the primary shared-vLLM runtime contracts."""
# Tests intentionally exercise stable internal HTTP and ownership seams.
# Local test doubles are not public APIs; the suite intentionally uses Torch CPU tensors.
# pylint: disable=forbidden-backend-import,missing-public-docstring,protected-access,no-member,unnecessary-lambda

import asyncio
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import rl.roles.rollout.vllm as vllm_module
from rl.roles.model import ModelRegistration, resolve_vllm_model
from rl.roles.rollout.base import GenerationRequest, GenerationSettings
from rl.roles.rollout.vllm import VLLMGenerationEngine


def _model() -> ModelRegistration:
    """Return one tied Qwen3 model registration."""
    return ModelRegistration(
        "qwen",
        "qwen3",
        "/model",
        "/tokenizer",
        "Qwen3ForCausalLM",
        "qwen3",
        "qwen3",
        True,
    )


def _settings(collect_log_probs: bool = True) -> GenerationSettings:
    """Return deterministic completion settings."""
    return GenerationSettings(
        max_new_tokens=3,
        temperature=0.7,
        top_p=0.9,
        top_k=5,
        do_sample=True,
        pad_token_id=0,
        eos_token_id=2,
        collect_log_probs=collect_log_probs,
        seed=17,
        eos_token_ids=(3,),
    )


def test_vllm_completion_preserves_payload_and_choice_order() -> None:
    """Completion payload settings and out-of-order choices retain token logprobs."""
    client = vllm_module._VLLMHTTPClient(None, "http://127.0.0.1:8100", "qwen", 1)
    settings = _settings()
    payload = client._completion_payload([[1], [2]], 17, settings)
    records = client._completion_records_from_response(
        [[1], [2]],
        1,
        settings,
        {
            "choices": [
                {
                    "index": 1,
                    "token_ids": [20],
                    "logprobs": {"token_logprobs": [-0.2]},
                },
                {
                    "index": 0,
                    "token_ids": [10, 2],
                    "logprobs": {"token_logprobs": [-0.1, -0.3]},
                },
            ]
        },
    )
    assert payload["prompt"] == [[1], [2]]
    assert payload["seed"] == 17
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert payload["top_k"] == 5
    assert payload["stop_token_ids"] == [2, 3]
    assert payload["logprobs"] == 1
    assert records == [([10, 2], [-0.1, -0.3]), ([20], [-0.2])]


def test_vllm_router_respects_capacity_and_restores_row_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completion admission never exceeds capacity and restores original row order."""
    client = vllm_module._VLLMHTTPClient(None, "http://127.0.0.1:8100", "qwen", 1)
    events: list[tuple[str, int]] = []
    active = 0
    peak = 0

    async def request(_method: str, _route: str, payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal active, peak
        seed = payload["seed"]
        events.append(("start", seed))
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.03 if seed == 10 else 0.001)
            return {"choices": [{"index": 0, "token_ids": [seed]}]}
        finally:
            active -= 1
            events.append(("finish", seed))

    monkeypatch.setattr(client, "_request_async", request)
    settings = SimpleNamespace(
        max_new_tokens=1,
        temperature=1.0,
        do_sample=True,
        top_p=1.0,
        top_k=0,
        collect_log_probs=False,
        ignore_eos=False,
        eos_token_ids=(2,),
        seed=10,
    )

    requests = client._completion_requests(  # pylint: disable=protected-access
        [[1], [2], [3], [4]],
        settings,
        child_capacity=2,
        batch_invariant=False,
    )
    records = asyncio.run(
        client._dispatch_completion_requests(  # pylint: disable=protected-access
            requests,
            row_count=4,
            settings=settings,
            child_capacity=2,
        )
    )

    assert records == [([10], None), ([11], None), ([12], None), ([13], None)]
    assert peak == 2
    assert events.index(("start", 12)) < events.index(("finish", 10))


@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
def test_shared_vllm_endpoint_has_one_owner_and_normal_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    deployment: str,
) -> None:
    """Coordinator launches the shared endpoint, peers connect, and owner closes it."""
    events: list[Any] = []
    config: dict[str, Any] = {
        "deployment": deployment,
        "data_parallel_size": 1,
        "tensor_parallel_size": 2,
        "port": 8100,
    }
    if deployment == "disjoint":
        config["visible_devices"] = "2,3"
    transfer = SimpleNamespace(close=lambda: events.append("transfer-close"))
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(
        vllm_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unused")),
    )
    monkeypatch.setattr(
        vllm_module.subprocess,
        "Popen",
        lambda command, **kwargs: events.append(("launch", command, kwargs))
        or SimpleNamespace(pid=99),
    )
    monkeypatch.setattr(
        vllm_module._VLLMHTTPClient,
        "wait_ready",
        lambda client, timeout: events.append(
            ("ready", client.base_url, timeout)
        ),
    )
    monkeypatch.setattr(vllm_module, "synchronize_error", lambda _error, _operation: None)
    monkeypatch.setattr(
        vllm_module,
        "synchronized_call",
        lambda _operation, callback: callback(),
    )
    owner = VLLMGenerationEngine(_model(), {"vllm": config}, refitter=transfer)

    assert not owner.client_initialized
    owner_client = owner._ensure_client()
    assert owner.client_initialized
    assert owner_client.is_server_owner
    assert owner_client.base_url == "http://127.0.0.1:8100"
    monkeypatch.setattr(owner_client, "close", lambda: events.append("server-close"))
    owner.close()

    monkeypatch.setenv("LOCAL_RANK", "1")
    peer = VLLMGenerationEngine(_model(), {"vllm": config})
    peer_client = peer._ensure_client()

    assert not peer_client.is_server_owner
    assert peer_client.base_url == owner_client.base_url
    assert len([event for event in events if isinstance(event, tuple) and event[0] == "launch"]) == 1
    assert events[-2:] == ["server-close", "transfer-close"]


def test_sync_generate_executes_complete_http_generation_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production synchronous generate executes request extraction, padding, and identity checks."""
    client = vllm_module._VLLMHTTPClient(None, "http://127.0.0.1:8100", "qwen", 1)
    captured: dict[str, Any] = {}

    def generate_tokens(
        prompts: list[list[int]],
        settings: Any,
        *,
        child_capacity: int,
        batch_invariant: bool,
        row_seeds: tuple[int, ...],
    ) -> list[tuple[list[int], list[float]]]:
        captured.update(
            prompts=prompts,
            settings=settings,
            child_capacity=child_capacity,
            batch_invariant=batch_invariant,
            row_seeds=row_seeds,
        )
        return [([9, 2], [-0.1, -0.2]), ([8], [-0.3])]

    monkeypatch.setattr(client, "generate_tokens", generate_tokens)
    engine = VLLMGenerationEngine(
        _model(),
        {
            "vllm": {
                "data_parallel_size": 1,
                "max_num_seqs": 2,
                "batch_invariant": True,
            }
        },
        client=client,
    )
    identity_calls: list[str] = []
    engine._weight_sync = SimpleNamespace(
        generation_identity=lambda used_client: identity_calls.append("identity")
        or (4, "digest-v4")
    )
    boundaries = iter((10.0, 12.5))
    monkeypatch.setattr(vllm_module.time, "perf_counter", lambda: next(boundaries))
    request = GenerationRequest(
        input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]),
        attention_mask=torch.tensor([[False, True, True], [True, True, True]]),
        settings=_settings(),
        row_seeds=(20, 21),
    )

    result = engine.generate(request)

    assert captured["prompts"] == [[1, 2], [3, 4, 5]]
    assert captured["child_capacity"] == 4
    assert captured["batch_invariant"]
    assert captured["row_seeds"] == (20, 21)
    assert result.sequences.tolist() == [[0, 1, 2, 9, 2, 0], [3, 4, 5, 8, 0, 0]]
    assert result.response_mask.tolist() == [[True, True, False], [True, False, False]]
    torch.testing.assert_close(
        result.rollout_log_probs,
        torch.tensor([[-0.1, -0.2, 0.0], [-0.3, 0.0, 0.0]]),
    )
    assert result.generation_seconds == 2.5
    assert result.worker_policy_version == 4
    assert result.worker_policy_fingerprint == "digest-v4"
    assert identity_calls == ["identity", "identity"]


def test_tp_owner_generates_once_and_broadcasts_complete_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the TP owner generates and production broadcast gives every sibling its result."""
    request = GenerationRequest(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones((1, 2), dtype=torch.bool),
        settings=_settings(),
    )
    owner_result = VLLMGenerationEngine._build_generation_result(
        request,
        [([9, 2], [-0.1, -0.2])],
        1.5,
        3,
        "digest-v3",
    )
    payloads: list[torch.Tensor] = []
    broadcast_index = [0]

    def broadcast(tensor: torch.Tensor, *, group: Any, group_src: int) -> None:
        assert group == "tp"
        assert group_src == 0
        index = broadcast_index[0]
        if index < 3:
            payloads.append(tensor.clone())
        else:
            tensor.copy_(payloads[index - 3])
        broadcast_index[0] += 1

    def gather(output: list[Any], value: Any, group: Any) -> None:
        assert group == "tp"
        output[:] = [value, value] if isinstance(value, tuple) else [1.5, None]

    monkeypatch.setattr(vllm_module.platform, "broadcast", broadcast)
    monkeypatch.setattr(vllm_module.platform, "all_gather_object", gather)
    generation_calls: list[str] = []
    owner = VLLMGenerationEngine(_model(), {"vllm": {}}, client=object())
    owner.configure_trainer_tensor_parallel(
        group="tp", tp_rank=0, tp_size=2, request_rank=0, request_size=1
    )
    owner._weight_sync = SimpleNamespace(
        generation_identity=lambda _client: (3, "digest-v3")
    )
    owner._generate_request = (
        lambda *_args: generation_calls.append("owner") or owner_result
    )
    owner.synchronize_error = lambda _error, _operation: None

    owner_copy = owner._generate_tp_owned(request)

    non_owner = VLLMGenerationEngine(_model(), {"vllm": {}}, client=object())
    non_owner.configure_trainer_tensor_parallel(
        group="tp", tp_rank=1, tp_size=2, request_rank=0, request_size=1
    )
    non_owner._weight_sync = SimpleNamespace(
        generation_identity=lambda _client: (3, "digest-v3")
    )
    non_owner._generate_request = lambda *_args: pytest.fail(
        "non-owner submitted a generation request"
    )
    non_owner.synchronize_error = lambda _error, _operation: None

    replica = non_owner._generate_tp_owned(request)

    assert generation_calls == ["owner"]
    assert owner.request_owner_generate_count == 1
    assert non_owner.request_owner_generate_count == 0
    torch.testing.assert_close(replica.sequences, owner_copy.sequences)
    assert torch.equal(replica.response_mask, owner_copy.response_mask)
    torch.testing.assert_close(replica.rollout_log_probs, owner_copy.rollout_log_probs)
    assert replica.generation_seconds == 1.5
    assert replica.worker_policy_version == 3
    assert replica.worker_policy_fingerprint == "digest-v3"


def test_http_client_executes_sync_and_async_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production HTTP helpers serialize requests and parse successful JSON responses."""
    sync_requests: list[Any] = []

    class SyncResponse:
        def __enter__(self) -> "SyncResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return json.dumps({"mode": "sync"}).encode("utf-8")

    def urlopen(request: Any, timeout: float) -> SyncResponse:
        sync_requests.append((request, timeout))
        return SyncResponse()

    class AsyncResponse:
        status = 200

        async def __aenter__(self) -> "AsyncResponse":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        @staticmethod
        async def read() -> bytes:
            return json.dumps({"mode": "async"}).encode("utf-8")

    class AsyncSession:
        closed = False

        @staticmethod
        def request(method: str, url: str, **kwargs: Any) -> AsyncResponse:
            assert (method, url, kwargs["json"]) == (
                "POST",
                "http://127.0.0.1:8100/v1/completions",
                {"prompt": [[1]]},
            )
            return AsyncResponse()

    monkeypatch.setattr(vllm_module.urllib_request, "urlopen", urlopen)
    monkeypatch.setattr(
        vllm_module,
        "_load_aiohttp",
        lambda: SimpleNamespace(ClientError=RuntimeError),
    )
    client = vllm_module._VLLMHTTPClient(None, "http://127.0.0.1:8100", "qwen", 3)
    client._async_session = AsyncSession()

    sync_result = client._request("POST", "control", {"value": 1})
    async_result = asyncio.run(
        client._request_async("POST", "v1/completions", {"prompt": [[1]]})
    )

    assert sync_result == {"mode": "sync"}
    assert async_result == {"mode": "async"}
    request, timeout = sync_requests[0]
    assert request.full_url == "http://127.0.0.1:8100/control"
    assert json.loads(request.data) == {"value": 1}
    assert timeout == 3

    class EmptyResponse(SyncResponse):
        @staticmethod
        def read() -> bytes:
            return b""

    monkeypatch.setattr(
        vllm_module.urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: EmptyResponse(),
    )
    assert client._request("POST", "empty") == {}


def test_shared_server_builds_command_and_isolated_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registered engine command and environment retain all shared-server settings."""
    vllm_config = {
        "deployment": "colocated",
        "data_parallel_size": 2,
        "tensor_parallel_size": 2,
        "model_implementation": "hyper",
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "enable_prompt_tokens_details": True,
        "skip_mm_profiling": True,
        "profiler_config": {"profiler": "none"},
        "batch_invariant": True,
        "server_hccl_if_base_port": 65000,
        "server_hccl_npu_socket_port_range": "65000-65010",
        "consistency_profile": "profile",
        "max_num_seqs": 8,
        "weight_sync": {
            "bucket_size_mb": 1,
            "strategy": "direct_reshard",
            "fallback_strategy": "full_gather",
        },
    }
    transfer = object()
    monkeypatch.setattr(
        vllm_module,
        "build_weight_transfer",
        lambda *args, **kwargs: transfer,
    )
    engine = vllm_module.build_vllm_engine({"vllm": vllm_config}, _model())
    monkeypatch.setenv("RANK", "9")
    command = engine._server_command("127.0.0.1", 8100)
    environment = engine._server_environment("0,1,2,3")

    assert engine._weight_sync._weight_transfer is transfer
    assert command[command.index("--data-parallel-size") + 1] == "2"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert "--no-enable-prefix-caching" in command
    assert "--enable-chunked-prefill" in command
    assert "--enable-prompt-tokens-details" in command
    assert "--skip-mm-profiling" in command
    assert command[command.index("--profiler-config") + 1] == '{"profiler": "none"}'
    assert "HyperQwen3ForCausalLM" in command[command.index("--hf-overrides") + 1]
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3"
    assert environment["VLLM_BATCH_INVARIANT"] == "1"
    assert environment["HCCL_IF_BASE_PORT"] == "65000"
    assert environment["HYPER_RL_CONSISTENCY_PROFILE"] == "profile"
    assert "RANK" not in environment


def test_engine_computes_child_capacity_and_inprocess_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production capacity and injected-vLLM paths retain row seeds and raw logprobs."""

    class SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class LocalClient:
        def __init__(self) -> None:
            self.sampling: Any = None

        def generate(self, prompts: Any, *, sampling_params: Any, use_tqdm: bool) -> list[Any]:
            assert not use_tqdm
            assert [prompt["prompt_token_ids"] for prompt in prompts] == [[1], [2]]
            self.sampling = sampling_params
            outputs = []
            for token_id in (7, 8):
                candidate = SimpleNamespace(logprob=-0.5)
                completion = SimpleNamespace(
                    token_ids=[token_id],
                    logprobs=[{token_id: candidate}],
                )
                outputs.append(SimpleNamespace(outputs=[completion]))
            return outputs

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            SamplingParams=SamplingParams,
            TokensPrompt=lambda *, prompt_token_ids: {
                "prompt_token_ids": prompt_token_ids
            },
        ),
    )
    monkeypatch.setattr(vllm_module.platform, "get_world_size", lambda: 3)
    monkeypatch.setattr(vllm_module.platform, "get_rank", lambda: 1)
    engine = VLLMGenerationEngine(
        _model(),
        {"vllm": {"data_parallel_size": 2, "max_num_seqs": 3}},
        client=object(),
    )
    local_client = LocalClient()

    capacity = engine._local_child_capacity()
    records = engine._inprocess_completions(
        local_client,
        [[1], [2]],
        _settings(),
        row_seeds=(30, 31),
    )

    assert capacity == 4
    assert records == [([7], [-0.5]), ([8], [-0.5])]
    assert [sampling.seed for sampling in local_client.sampling] == [30, 31]
    assert all(sampling.stop_token_ids == [2, 3] for sampling in local_client.sampling)


def test_http_client_runs_and_closes_persistent_async_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normal generation transport owns one reusable loop, connector, and session."""
    events = []

    class Session:
        def __init__(self, *, connector: Any, timeout: Any) -> None:
            self.connector = connector
            self.timeout = timeout
            self.closed = False
            events.append(("session", connector.limit, timeout.total))

        async def close(self) -> None:
            self.closed = True
            events.append("close")

    monkeypatch.setattr(
        vllm_module,
        "_load_aiohttp",
        lambda: SimpleNamespace(
            TCPConnector=lambda **kwargs: SimpleNamespace(**kwargs),
            ClientTimeout=lambda **kwargs: SimpleNamespace(**kwargs),
            ClientSession=Session,
        ),
    )
    client = vllm_module._VLLMHTTPClient(
        None, "http://127.0.0.1:8100", "qwen", 9
    )
    asyncio.run(client._create_async_session(4))
    assert events == [("session", 4, 9)]
    asyncio.run(client._async_session.close())
    client._async_session = None
    events.clear()

    class Loop:
        def __init__(self) -> None:
            self.running = True

        def is_running(self) -> bool:
            return self.running

        def call_soon_threadsafe(self, callback: Any) -> None:
            callback()

        def stop(self) -> None:
            self.running = False

    class Thread:
        def __init__(self, *, target: Any, args: Any, **_kwargs: Any) -> None:
            self.target = target
            self.args = args
            self.owner = target.__self__

        def start(self) -> None:
            self.owner._async_loop = Loop()
            self.owner._async_session = Session(
                connector=SimpleNamespace(limit=self.args[0]),
                timeout=SimpleNamespace(total=9),
            )
            self.args[1].set()

        def is_alive(self) -> bool:
            return self.owner._async_loop is not None and self.owner._async_loop.is_running()

        @staticmethod
        def join(timeout: Any = None) -> None:
            assert timeout in (None, 10)

    class Future:
        def __init__(self, coroutine: Any) -> None:
            self.coroutine = coroutine

        def result(self, timeout: Any) -> None:
            assert timeout == 10
            asyncio.run(self.coroutine)

    monkeypatch.setattr(vllm_module.threading, "Thread", Thread)
    monkeypatch.setattr(
        vllm_module.asyncio,
        "run_coroutine_threadsafe",
        lambda coroutine, _loop: Future(coroutine),
    )

    loop = client._ensure_async_runtime(4)
    reused = client._ensure_async_runtime(2)
    assert reused is loop
    assert loop.is_running()
    assert events == [("session", 4, 9)]

    client._close_async_runtime()

    assert events == [("session", 4, 9), "close"]
    assert client._async_loop is None
    assert client._async_thread is None
    assert client._async_session is None
    assert client._async_connection_limit == 0


def test_http_client_retries_readiness_and_completes_bounded_process_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient health and process delays recover before the client returns."""
    attempts = []
    process = SimpleNamespace(poll=lambda: None)
    client = vllm_module._VLLMHTTPClient(
        process, "http://127.0.0.1:8100", "qwen", 3
    )

    def request(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        attempts.append("health")
        if len(attempts) == 1:
            raise RuntimeError("starting")
        return {}

    times = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr(vllm_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(vllm_module.time, "sleep", lambda _seconds: None)
    client.wait_ready(5)
    assert attempts == ["health", "health"]

    events = []

    class DelayedProcess:
        pid = 101

        def __init__(self) -> None:
            self.waits = 0

        @staticmethod
        def poll() -> None:
            return None

        def wait(self, timeout: int) -> int:
            self.waits += 1
            events.append(("wait", timeout))
            if self.waits == 1:
                raise vllm_module.subprocess.TimeoutExpired("vllm", timeout)
            return 0

    delayed = vllm_module._VLLMHTTPClient(
        DelayedProcess(), "http://127.0.0.1:8100", "qwen", 3
    )
    monkeypatch.setattr(delayed, "_close_async_runtime", lambda: None)
    monkeypatch.setattr(
        delayed,
        "_wait_process_group_exit",
        lambda process_group: events.append(("exit", process_group)),
    )
    monkeypatch.setattr(
        vllm_module.os,
        "killpg",
        lambda process_group, signal_value: events.append(
            ("signal", process_group, signal_value)
        ),
    )
    delayed.close()

    exited_process = SimpleNamespace(pid=102, poll=lambda: 0)
    exited = vllm_module._VLLMHTTPClient(
        exited_process, "http://127.0.0.1:8100", "qwen", 3
    )
    monkeypatch.setattr(exited, "_close_async_runtime", lambda: None)
    monkeypatch.setattr(
        exited,
        "_wait_process_group_exit",
        lambda process_group: events.append(("exit", process_group)),
    )
    exited.close()

    assert events == [
        ("signal", 101, vllm_module.signal.SIGTERM),
        ("wait", 20),
        ("signal", 101, vllm_module.signal.SIGKILL),
        ("wait", 10),
        ("exit", 101),
        ("signal", 102, vllm_module.signal.SIGTERM),
        ("signal", 102, vllm_module.signal.SIGKILL),
        ("exit", 102),
    ]


def test_http_client_exercises_current_weight_control_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every successful rollout-control operation maps to its stable HTTP contract."""
    calls = []

    def request(
        method: str,
        route: str,
        payload: Any = None,
        timeout: Any = None,
        base_url: Any = None,
    ) -> dict[str, Any]:
        calls.append((method, route, payload, timeout, base_url))
        if route == "get_world_size":
            return {"world_size": 2}
        if route.startswith("pause"):
            return {"status": "paused"}
        if route == "is_paused":
            return {"is_paused": True}
        if route == "is_sleeping":
            return {"is_sleeping": True}
        if route == "resume":
            return {"status": "resumed"}
        if route == "collective_rpc":
            return {
                "results": [
                    {
                        "version": 3,
                        "algorithm": "test",
                        "digest": "digest-v3",
                        "tensor_count": 1,
                        "value_count": 2,
                        "tensors": {},
                    }
                ]
            }
        return {}

    client = vllm_module._VLLMHTTPClient(
        None, "http://127.0.0.1:8100/", "qwen", 9
    )
    monkeypatch.setattr(client, "_request", request)
    fingerprint = {
        "algorithm": "test",
        "digest": "digest-v3",
        "tensor_count": 1,
        "value_count": 2,
        "tensors": {},
    }

    assert client.base_url == "http://127.0.0.1:8100"
    assert client.get_world_size(base_url="http://replica") == 2
    client.pause()
    assert client.is_paused()
    client.sleep(level=1, mode="wait")
    client.wake_up(("weights",))
    assert client.is_sleeping()
    client.start_weight_update()
    client.finish_weight_update()
    assert client.collective_rpc("get_policy_version") == [
        {
            "version": 3,
            "algorithm": "test",
            "digest": "digest-v3",
            "tensor_count": 1,
            "value_count": 2,
            "tensors": {},
        }
    ]
    assert client.get_policy_weight_fingerprints()[0]["digest"] == "digest-v3"
    assert client.get_policy_weight_fingerprints("http://replica")[0]["version"] == 3
    client.verify_policy_weight_identity(3, fingerprint)
    client.verify_direct_content_identity(3, {0: fingerprint})
    client.resume()

    routes = [call[1] for call in calls]
    assert routes == [
        "get_world_size",
        "pause?mode=abort&clear_cache=true",
        "is_paused",
        "sleep?level=1&mode=wait",
        "wake_up?tags=weights&tags=_hyper_keep_scheduler_paused",
        "is_sleeping",
        "start_weight_update",
        "finish_weight_update",
        "collective_rpc",
        "collective_rpc",
        "collective_rpc",
        "collective_rpc",
        "collective_rpc",
        "resume",
    ]


def test_http_process_group_helpers_detect_live_and_completed_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux process scanning distinguishes a live member and an exited group."""

    class Stat:
        def __init__(self, text: str) -> None:
            self.text = text

        def read_text(self, **_kwargs: Any) -> str:
            return self.text

    class Entry:
        def __init__(self, name: str, stat: str) -> None:
            self.name = name
            self.stat = stat

        def __truediv__(self, _name: str) -> Stat:
            return Stat(self.stat)

    entries = (
        Entry("self", ""),
        Entry("10", "10 (worker) S 1 123 0"),
        Entry("11", "11 (zombie) Z 1 123 0"),
    )

    class Proc:
        def __init__(self, _path: str) -> None:
            pass

        @staticmethod
        def iterdir() -> Any:
            return entries

    monkeypatch.setattr(vllm_module, "Path", Proc)
    assert vllm_module._VLLMHTTPClient._has_live_process_group_members(123)
    assert not vllm_module._VLLMHTTPClient._has_live_process_group_members(999)

    monkeypatch.setattr(
        vllm_module.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    vllm_module._VLLMHTTPClient._wait_process_group_exit(999)


def test_http_generate_tokens_routes_unseeded_and_batch_invariant_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public HTTP generation call builds parent requests and restores row order."""
    client = vllm_module._VLLMHTTPClient(
        None, "http://127.0.0.1:8100", "qwen", 3
    )
    settings = SimpleNamespace(
        max_new_tokens=1,
        temperature=1.0,
        do_sample=True,
        top_p=1.0,
        top_k=0,
        collect_log_probs=False,
        ignore_eos=False,
        eos_token_ids=(2,),
        seed=None,
    )

    async def generate_parent(request: Any, _settings_value: Any) -> Any:
        return [([prompt[0] + 10], None) for prompt in request.prompts]

    class Future:
        def __init__(self, coroutine: Any) -> None:
            self.coroutine = coroutine

        def result(self) -> Any:
            return asyncio.run(self.coroutine)

    monkeypatch.setattr(client, "_generate_completion_request", generate_parent)
    monkeypatch.setattr(client, "_ensure_async_runtime", lambda _capacity: "loop")
    monkeypatch.setattr(
        vllm_module.asyncio,
        "run_coroutine_threadsafe",
        lambda coroutine, _loop: Future(coroutine),
    )

    records = client.generate_tokens(
        [[1], [2], [3]],
        settings,
        child_capacity=2,
    )
    invariant = client._completion_requests(
        [[5], [5], [5]],
        SimpleNamespace(**{**settings.__dict__, "seed": 20}),
        child_capacity=3,
        batch_invariant=True,
        row_seeds=(20, 21, 22),
    )

    assert records == [([11], None), ([12], None), ([13], None)]
    assert [(request.start_row, request.child_count) for request in invariant] == [(0, 3)]


def test_engine_exposes_weight_sync_observability_and_ep_command() -> None:
    """Rollout forwards publication status and enables static EP explicitly."""
    engine = object.__new__(VLLMGenerationEngine)
    engine._weight_sync = SimpleNamespace(
        configured_strategy="direct_reshard",
        last_strategy="full_gather",
        fallback_count=1,
        direct_success_count=2,
        attempted_strategies=("direct_reshard", "full_gather"),
        completed_strategy="full_gather",
        fallback_reason="direct unavailable",
        streaming_stats={"bucket_count": 3},
    )
    engine._model = _model()
    engine._rollout_model = resolve_vllm_model(engine._model, "hyper")
    engine._deployment = "colocated"
    engine._config = {
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "dtype": "bfloat16",
        "enable_expert_parallel": True,
    }

    command = engine._server_command("127.0.0.1", 8100)

    assert engine.weight_sync_configured_strategy == "direct_reshard"
    assert engine.weight_sync_last_strategy == "full_gather"
    assert engine.weight_sync_fallback_count == 1
    assert engine.weight_sync_direct_success_count == 2
    assert engine.weight_sync_attempted_strategies == (
        "direct_reshard",
        "full_gather",
    )
    assert engine.weight_sync_completed_strategy == "full_gather"
    assert engine.weight_sync_fallback_reason == "direct unavailable"
    assert engine.weight_sync_streaming_stats == {"bucket_count": 3}
    assert "--enable-expert-parallel" in command
