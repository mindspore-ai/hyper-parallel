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
"""CPU contracts for model-scoped vLLM implementation selection."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch
import rl.roles.rollout.vllm as vllm_module
from rl.roles.model import (
    ModelRegistration,
    architecture_for_implementation,
    normalize_model_implementation,
    resolve_vllm_model,
)
from rl.roles.rollout.vllm import VLLMGenerationEngine
from rl.roles.weight_sync.transfer import (
    ColocatedFullGatherWeightTransfer,
    map_actor_state_dict,
)
from rl.roles.weight_sync.sync import (
    ActorRolloutWeightSync,
    PolicySnapshot,
    VLLMWeightSyncClientMixin,
    verify_policy_fingerprints,
)

import hyper_parallel_vllm_plugin


def test_installed_vllm_plugin_entry_point_forwards_qwen3_runtime() -> None:
    """The fixed image's legacy entry point resolves to the migrated Qwen3 plugin."""
    assert hyper_parallel_vllm_plugin.HYPER_QWEN3_ARCHITECTURE == "HyperQwen3ForCausalLM"
    assert callable(hyper_parallel_vllm_plugin.register_hyper_models)


def _model(
    hyper_model_name: str = "qwen3",
    *,
    tie_word_embeddings: bool = True,
) -> ModelRegistration:
    if hyper_model_name != "qwen3":
        raise ValueError(f"Unsupported test model: {hyper_model_name}")
    return ModelRegistration(
        "qwen",
        hyper_model_name,
        "/model",
        "/tokenizer",
        "Qwen3ForCausalLM",
        "qwen3",
        "qwen3",
        tie_word_embeddings,
    )


def test_model_implementation_defaults_to_native() -> None:
    """An omitted rollout implementation must select upstream vLLM."""
    assert normalize_model_implementation(None) == "native"


def test_native_architecture_and_refit_names_follow_qwen3_contract() -> None:
    """Native Qwen3 keeps causal-model parameter names."""
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {"vllm": {"model_implementation": "native"}},
        client=object(),
    )

    server_command = engine._server_command(  # pylint: disable=protected-access
        "127.0.0.1", 8100
    )

    assert "--hf-overrides" not in server_command
    assert architecture_for_implementation("native", "qwen3") == (
        "Qwen3ForCausalLM"
    )
    qwen3_state = {
        "model.layers.0.self_attn.q_proj.weight": object(),
        "lm_head.weight": object(),
    }
    qwen3_rollout_model = resolve_vllm_model(_model("qwen3"), "native")
    assert list(map_actor_state_dict(qwen3_state, qwen3_rollout_model)) == [
        "model.layers.0.self_attn.q_proj.weight"
    ]
    untied_rollout_model = resolve_vllm_model(
        _model(tie_word_embeddings=False),
        "native",
    )
    assert list(map_actor_state_dict(qwen3_state, untied_rollout_model)) == [
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    ]


def test_native_qwen3_can_launch_lazy_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native Qwen3 may launch its client lazily."""
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {"vllm": {"model_implementation": "native"}},
    )
    client = object()
    monkeypatch.setattr(
        engine,
        "_server_endpoint",
        lambda: ("0", "127.0.0.1", 8100),
    )
    engine._resolved_topology = SimpleNamespace(server_owner=True)  # pylint: disable=protected-access
    monkeypatch.setattr(
        engine,
        "_launch_client",
        lambda _visible_devices, _host, _port: client,
    )

    assert engine._ensure_client() is client  # pylint: disable=protected-access


def test_sync_request_preserves_explicit_data_parallel_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance probes can target one DP engine when validating its prefix cache."""
    captured: dict[str, Any] = {}

    class _Response:
        def __enter__(self) -> "_Response":
            """Return the fake response context."""
            return self

        def __exit__(self, *_args: Any) -> None:
            """Close the fake response context."""
            return None

        @staticmethod
        def read() -> bytes:
            """Return one successful JSON payload."""
            return json.dumps({"ok": True}).encode("utf-8")

    def _urlopen(request: Any, timeout: float) -> _Response:  # pylint: disable=unused-argument
        captured["request"] = request
        return _Response()

    monkeypatch.setattr(vllm_module.urllib_request, "urlopen", _urlopen)
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        None,
        "http://127.0.0.1:8000",
        "model",
        10.0,
    )

    result = client._request(  # pylint: disable=protected-access
        "POST",
        "v1/completions",
        {"prompt": [[1]]},
        request_headers={"X-data-parallel-rank": "0"},
    )

    assert result == {"ok": True}
    assert captured["request"].get_header("X-data-parallel-rank") == "0"


@pytest.mark.parametrize(
    "local_rank",
    [0, 1],
)
def test_disjoint_endpoint_is_shared_by_all_trainer_ranks(
    monkeypatch: pytest.MonkeyPatch,
    local_rank: int,
) -> None:
    """Every Trainer rank resolves the complete external deployment and endpoint."""
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {
            "vllm": {
                "deployment": "disjoint",
                "visible_devices": "2,3",
                "data_parallel_size": 2,
                "tensor_parallel_size": 1,
                "port": 8200,
            }
        },
        client=object(),
    )

    endpoint = engine._server_endpoint()  # pylint: disable=protected-access

    assert endpoint == ("2,3", "127.0.0.1", 8200)


def test_endpoint_defaults_to_actual_platform_trainer_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Topology resolution uses the running Trainer world when env metadata is absent."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setattr(vllm_module.platform, "get_rank", lambda: 3)
    monkeypatch.setattr(vllm_module.platform, "get_world_size", lambda: 4)
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {
            "vllm": {
                "deployment": "disjoint",
                "visible_devices": "6,7",
                "data_parallel_size": 1,
                "tensor_parallel_size": 2,
                "port": 8200,
            }
        },
        client=object(),
    )

    assert engine._server_endpoint() == ("6,7", "127.0.0.1", 8200)  # pylint: disable=protected-access
    assert engine._resolved_topology.trainer_rank == 3  # pylint: disable=protected-access
    assert engine._resolved_topology.trainer_world_size == 4  # pylint: disable=protected-access


def test_hyper_qwen3_uses_registered_architecture() -> None:
    """Qwen3 Hyper uses the same server interface with only an HF override."""
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {"vllm": {"model_implementation": "hyper"}},
        client=object(),
    )

    command = engine._server_command("127.0.0.1", 8100)  # pylint: disable=protected-access
    override = command[command.index("--hf-overrides") + 1]

    assert override == '{"architectures": ["HyperQwen3ForCausalLM"]}'


@pytest.mark.parametrize("implementation", ["hyper", "native"])
def test_internal_dp_command_is_shared_by_model_implementations(
    implementation: str,
) -> None:
    """Qwen3 uses one vLLM-owned topology for Hyper and native implementations."""
    engine = VLLMGenerationEngine(
        _model(),
        {
            "vllm": {
                "deployment": "colocated",
                "data_parallel_size": 2,
                "model_implementation": implementation,
                "tensor_parallel_size": 1,
            }
        },
        client=object(),
    )

    command = engine._server_command("127.0.0.1", 8100)  # pylint: disable=protected-access

    assert command[command.index("--data-parallel-size") + 1] == "2"
    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    assert "--api-server-count" not in command
    if implementation == "hyper":
        assert "HyperQwen3ForCausalLM" in command[command.index("--hf-overrides") + 1]
    else:
        assert "--hf-overrides" not in command


def test_dp4_internal_dp_command_uses_upstream_frontend_auto() -> None:
    """DP4 launches four engines without pinning the upstream API process count."""
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {
            "vllm": {
                "deployment": "colocated",
                "data_parallel_size": 4,
                "model_implementation": "hyper",
            }
        },
        client=object(),
    )

    command = engine._server_command("127.0.0.1", 8100)  # pylint: disable=protected-access

    assert command[command.index("--data-parallel-size") + 1] == "4"
    assert "--api-server-count" not in command


def test_internal_dp1_tp2_command_uses_shared_vllm_runtime() -> None:
    """A single TP2 engine still uses the explicit shared-deployment command."""
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {
            "vllm": {
                "deployment": "colocated",
                "data_parallel_size": 1,
                "model_implementation": "hyper",
                "tensor_parallel_size": 2,
            }
        },
        client=object(),
    )

    command = engine._server_command("127.0.0.1", 8100)  # pylint: disable=protected-access

    assert command[command.index("--data-parallel-size") + 1] == "1"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert "--api-server-count" not in command


def test_completion_payload_uses_selected_transport_contract() -> None:
    """The selected transport requests raw logprobs and authoritative token IDs."""
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        None,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )

    payload = client._completion_payload(  # pylint: disable=protected-access
        [[1]],
        10,
        SimpleNamespace(
            max_new_tokens=1,
            temperature=1.0,
            do_sample=True,
            top_p=1.0,
            top_k=0,
            collect_log_probs=True,
            ignore_eos=False,
            eos_token_ids=(2,),
        ),
    )

    assert payload["logprobs"] == 1
    assert "return_tokens_as_token_ids" not in payload
    assert payload["return_token_ids"] is True


def test_completion_parser_uses_token_ids_and_sampled_logprobs_only() -> None:
    """Token-ID logprob labels cannot alter authoritative IDs or raw sampled values."""
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        None,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )
    result = {
        "choices": [
            {
                "index": 0,
                "token_ids": [7, 8],
                "logprobs": {
                    "tokens": ["token_id:999", "token_id:998"],
                    "token_logprobs": [-0.25, -0.5],
                    "top_logprobs": [{}, {}],
                },
            }
        ]
    }

    records = client._completion_records_from_response(  # pylint: disable=protected-access
        [[1]],
        1,
        SimpleNamespace(collect_log_probs=True),
        result,
    )

    assert records == [([7, 8], [-0.25, -0.5])]


def test_internal_dp_endpoint_uses_all_devices_and_one_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared deployment must not derive its endpoint from LOCAL_RANK."""
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {
            "vllm": {
                "deployment": "colocated",
                "data_parallel_size": 2,
                "tensor_parallel_size": 1,
                "port": 8100,
            }
        },
        client=object(),
    )

    assert engine._server_endpoint() == ("4,5", "127.0.0.1", 8100)  # pylint: disable=protected-access


def test_seeded_identical_prompts_share_one_completion_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling samples preserve row seeds while avoiding duplicate HTTP requests."""
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        None,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )
    payloads = []

    async def request(
        _method: str,
        _route: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Return one token carrying each child request's derived seed."""
        payloads.append(payload)
        return {
            "choices": [
                {"index": index, "token_ids": [payload["seed"] + index]}
                for index in range(payload["n"])
            ]
        }

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

    records = client.generate_tokens(
        [[1], [1], [2], [2], [2], [3]],
        settings,
        child_capacity=2,
        batch_invariant=True,
    )

    request_shapes = [
        (payload["prompt"], payload["seed"], payload["n"])
        for payload in payloads
    ]
    assert request_shapes == [
        ([[1]], 10, 2),
        ([[2]], 12, 2),
        ([[2]], 14, 1),
        ([[3]], 15, 1),
    ]
    assert records == [
        ([10], None),
        ([11], None),
        ([12], None),
        ([13], None),
        ([14], None),
        ([15], None),
    ]
    client.close()


def test_explicit_row_seeds_preserve_grouping_across_dp_partitions() -> None:
    """Grouped HTTP parents use canonical seeds instead of local row offsets."""
    settings = SimpleNamespace(
        seed=10,
        do_sample=True,
        temperature=1.0,
    )

    requests = vllm_module._VLLMHTTPClient._completion_requests(  # pylint: disable=protected-access
        [[1], [1], [2], [2], [2], [3]],
        settings,
        child_capacity=4,
        batch_invariant=True,
        row_seeds=(100, 101, 200, 201, 205, 300),
    )

    assert [
        (request.start_row, request.seed, request.child_count)
        for request in requests
    ] == [
        (0, 100, 2),
        (2, 200, 2),
        (4, 205, 1),
        (5, 300, 1),
    ]


def test_http_client_reuses_persistent_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation reuses its session and closes the persistent runtime."""
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        None,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )

    async def request(
        _method: str,
        _route: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Return one successful choice for each independent parent."""
        return {"choices": [{"index": 0, "token_ids": [payload["seed"]]}]}

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

    first_records = client.generate_tokens([[1, 2]] * 8, settings, child_capacity=4)
    session = client._async_session  # pylint: disable=protected-access
    second_records = client.generate_tokens([[3]], settings, child_capacity=1)

    assert first_records == [([seed], None) for seed in range(10, 18)]
    assert second_records == [([10], None)]
    assert client._async_session is session  # pylint: disable=protected-access
    client.close()
    assert client._async_thread is None  # pylint: disable=protected-access


def test_generation_result_measures_completion_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Engine results retain the completion call's elapsed time."""
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        None,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )

    def generate_tokens(*_args: Any, **_kwargs: Any) -> list[Any]:
        """Return one completion through the HTTP-client seam."""
        return [([2], None)]

    monkeypatch.setattr(client, "generate_tokens", generate_tokens)
    elapsed_boundaries = iter((10.0, 12.5))
    monkeypatch.setattr(vllm_module.time, "perf_counter", lambda: next(elapsed_boundaries))
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {
            "vllm": {
                "deployment": "disjoint",
                "data_parallel_size": 1,
                "tensor_parallel_size": 1,
                "max_num_seqs": 1,
            }
        },
        client=client,
    )
    engine._weight_sync = SimpleNamespace(  # pylint: disable=protected-access
        generation_identity=lambda _client: (0, "digest-v0")
    )
    request = SimpleNamespace(
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones((1, 1), dtype=torch.bool),
        settings=SimpleNamespace(
            max_new_tokens=1,
            pad_token_id=0,
            collect_log_probs=False,
        ),
        row_seeds=None,
    )

    result = engine._generate(request)  # pylint: disable=protected-access

    assert result.generation_seconds == 2.5


def test_child_admission_refills_on_first_completion_and_restores_row_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A freed child slot admits the next row without waiting for the slow sibling."""
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        None,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )
    events = []
    active = 0
    active_peak = 0

    async def request(
        _method: str,
        _route: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Delay the first row until a completion-driven refill starts row 12."""
        nonlocal active, active_peak
        seed = payload["seed"]
        events.append(("start", seed))
        active += 1
        active_peak = max(active_peak, active)
        try:
            if seed == 10:
                while ("start", 12) not in events:
                    await asyncio.sleep(0.001)
            elif seed == 11:
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0)
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

    records = client.generate_tokens(
        [[1], [2], [3], [4]],
        settings,
        child_capacity=2,
    )

    assert records == [([10], None), ([11], None), ([12], None), ([13], None)]
    assert events.index(("start", 12)) < events.index(("finish", 10))
    assert active_peak == 2
    client.close()


def test_child_admission_fills_capacity_around_larger_queued_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A smaller later parent uses quota that the next FIFO parent cannot fit."""
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        None,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )
    requests = [
        vllm_module._CompletionRequest(0, 0, [[1]], 10, 3),  # pylint: disable=protected-access
        vllm_module._CompletionRequest(1, 3, [[2]], 20, 2),  # pylint: disable=protected-access
        vllm_module._CompletionRequest(2, 5, [[3]], 30, 1),  # pylint: disable=protected-access
    ]
    events = []

    async def generate(
        request: Any,
        _settings: Any,
    ) -> list[tuple[list[int], None]]:
        """Record admission order while the first parent remains active briefly."""
        events.append(("start", request.ordinal))
        if request.ordinal == 0:
            await asyncio.sleep(0.01)
        events.append(("finish", request.ordinal))
        return [([request.ordinal], None)] * request.child_count

    monkeypatch.setattr(client, "_completion_requests", lambda *_args, **_kwargs: requests)
    monkeypatch.setattr(client, "_generate_completion_request", generate)
    settings = SimpleNamespace(seed=10)

    records = client.generate_tokens([[1], [1], [1], [2], [2], [3]], settings, child_capacity=4)

    assert records == [([0], None)] * 3 + [([1], None)] * 2 + [([2], None)]
    assert events.index(("start", 2)) < events.index(("finish", 0))
    client.close()


def test_child_admission_cancels_inflight_and_stops_refill_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first failed parent cancels active work and leaves queued rows unsubmitted."""
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        None,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )
    started = []
    cancelled = []

    async def request(
        _method: str,
        _route: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Fail the first row while a second row remains cancellable in flight."""
        seed = payload["seed"]
        started.append(seed)
        if seed == 10:
            await asyncio.sleep(0.01)
            raise RuntimeError("request failed")
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(seed)
            raise
        return {"choices": [{"index": 0, "token_ids": [seed]}]}

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

    with pytest.raises(RuntimeError, match="request failed"):
        client.generate_tokens(
            [[1], [2], [3], [4]],
            settings,
            child_capacity=2,
        )

    assert started == [10, 11]
    assert cancelled == [11]
    client.close()


def test_child_admission_retrieves_other_completed_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent failed tasks are gathered after the first propagated failure."""
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        None,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )
    started = 0
    release = asyncio.Event()
    gathered_task_counts = []
    original_gather = asyncio.gather

    async def generate(_request: Any, _settings: Any) -> list[Any]:
        """Release two admitted parents together so both fail before dispatch resumes."""
        nonlocal started
        started += 1
        if started == 2:
            release.set()
        await release.wait()
        raise RuntimeError("request failed")

    async def gather(*tasks: Any, **kwargs: Any) -> Any:
        """Record that dispatch retrieves every other active task result."""
        gathered_task_counts.append(len(tasks))
        return await original_gather(*tasks, **kwargs)

    requests = [
        vllm_module._CompletionRequest(0, 0, [[1]], 10, 1),  # pylint: disable=protected-access
        vllm_module._CompletionRequest(1, 1, [[2]], 11, 1),  # pylint: disable=protected-access
    ]
    monkeypatch.setattr(client, "_generate_completion_request", generate)
    monkeypatch.setattr(asyncio, "gather", gather)

    with pytest.raises(RuntimeError, match="request failed"):
        asyncio.run(
            client._dispatch_completion_requests(  # pylint: disable=protected-access
                requests,
                row_count=2,
                settings=SimpleNamespace(),
                child_capacity=2,
            )
        )

    assert gathered_task_counts == [1]


def test_async_runtime_propagates_event_loop_creation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Event-loop startup errors signal the caller instead of blocking forever."""
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        None,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )
    monkeypatch.setattr(
        asyncio,
        "new_event_loop",
        lambda: (_ for _ in ()).throw(RuntimeError("loop unavailable")),
    )

    with pytest.raises(RuntimeError, match="loop unavailable"):
        client._ensure_async_runtime(1)  # pylint: disable=protected-access


@pytest.mark.parametrize("indices", [[0, 0], [-1, 1], [0, 2], [False, 1]])
def test_completion_response_rejects_invalid_choice_indices(indices: list[int]) -> None:
    """Choice indices must form the exact child ordering expected by row restoration."""
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        None,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )
    result = {
        "choices": [
            {"index": index, "token_ids": [index]}
            for index in indices
        ]
    }

    with pytest.raises(RuntimeError, match="choice ind"):
        client._completion_records_from_response(  # pylint: disable=protected-access
            [[1]],
            num_choices=2,
            settings=SimpleNamespace(collect_log_probs=False),
            result=result,
        )


@pytest.mark.parametrize(
    ("rank", "expected_capacity"),
    [(0, 5), (1, 5), (2, 5), (3, 5), (4, 4)],
)
def test_internal_dp_child_capacity_partitions_global_engine_capacity(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    expected_capacity: int,
) -> None:
    """Internal DP assigns two bounded capacity waves once across Trainer ranks."""
    monkeypatch.setattr(vllm_module.platform, "get_rank", lambda: rank)
    monkeypatch.setattr(vllm_module.platform, "get_world_size", lambda: 5)
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {
            "vllm": {
                "data_parallel_size": 3,
                "max_num_seqs": 4,
            }
        },
        client=object(),
    )

    assert engine._local_child_capacity() == expected_capacity  # pylint: disable=protected-access


@pytest.mark.parametrize(
    ("tp_rank", "is_owner"),
    [(0, True), (1, False)],
)
def test_trainer_tp_request_ownership_uses_logical_dp_capacity(
    tp_rank: int,
    is_owner: bool,
) -> None:
    """TP siblings share one logical request quota and only rank zero owns HTTP."""
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {
            "vllm": {
                "data_parallel_size": 2,
                "max_num_seqs": 4,
            }
        },
        client=object(),
    )

    engine.configure_trainer_tensor_parallel(
        group=object(),
        tp_rank=tp_rank,
        tp_size=2,
        request_rank=1,
        request_size=2,
    )

    assert engine.is_request_owner is is_owner
    assert engine._local_child_capacity() == 8  # pylint: disable=protected-access


@pytest.mark.parametrize(
    ("tp_rank", "expected_calls"),
    [(0, 1), (1, 0)],
)
def test_trainer_tp_generation_calls_only_the_request_owner(
    tp_rank: int,
    expected_calls: int,
) -> None:
    """Every TP rank participates while only group rank zero invokes generation."""
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {"vllm": {"data_parallel_size": 1, "max_num_seqs": 2}},
        client=object(),
    )
    engine.configure_trainer_tensor_parallel(
        group=object(),
        tp_rank=tp_rank,
        tp_size=2,
        request_rank=0,
        request_size=1,
    )
    request = SimpleNamespace()
    expected = object()
    calls = []
    # pylint: disable=protected-access
    engine._validate_tp_request = lambda _request: None  # type: ignore[method-assign]
    engine._weight_sync = SimpleNamespace(  # pylint: disable=protected-access
        generation_identity=lambda _client: (0, "digest-v0")
    )
    engine._generate_request = (  # type: ignore[method-assign]  # pylint: disable=protected-access
        lambda _request, _client, _identity: calls.append("generate") or expected
    )
    engine.synchronize_error = lambda error, _operation: (  # type: ignore[method-assign]
        None if error is None else (_ for _ in ()).throw(error)
    )
    engine._broadcast_tp_result = (  # type: ignore[method-assign]  # pylint: disable=protected-access
        lambda _request, result, _identity: expected if result is None else result
    )

    assert engine._generate_tp_owned(request) is expected  # pylint: disable=protected-access
    assert calls == ["generate"] * expected_calls
    assert engine.request_owner_generate_count == expected_calls


@pytest.mark.parametrize(
    ("do_sample", "temperature", "batch_invariant"),
    [
        (False, 1.0, True),
        (True, 1.0, False),
        (True, 0.0, True),
        (True, 1e-6, True),
    ],
)
def test_seeded_prompt_grouping_requires_sampled_batch_invariance(
    monkeypatch: pytest.MonkeyPatch,
    do_sample: bool,
    temperature: float,
    batch_invariant: bool,
) -> None:
    """Greedy or non-invariant generation retains one HTTP request per row."""
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        None,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )
    payloads = []

    async def request(
        _method: str,
        _route: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Capture independent completion requests."""
        payloads.append(payload)
        return {"choices": [{"index": 0, "token_ids": [payload["seed"]]}]}

    monkeypatch.setattr(client, "_request_async", request)
    settings = SimpleNamespace(
        max_new_tokens=1,
        temperature=temperature,
        do_sample=do_sample,
        top_p=1.0,
        top_k=0,
        collect_log_probs=False,
        ignore_eos=False,
        eos_token_ids=(2,),
        seed=10,
    )

    client.generate_tokens(
        [[1], [1]],
        settings,
        child_capacity=2,
        batch_invariant=batch_invariant,
    )

    assert [(payload["seed"], payload["n"]) for payload in payloads] == [
        (10, 1),
        (11, 1),
    ]
    client.close()


@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
def test_shared_coordinator_launches_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
    deployment: str,
) -> None:
    """Rank zero owns the only process for either shared deployment."""
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    events = []
    monkeypatch.setattr(
        vllm_module,
        "synchronize_error",
        lambda _error, operation: events.append(operation),
    )
    vllm_config = {
        "deployment": deployment,
        "data_parallel_size": 1,
        "tensor_parallel_size": 2,
        "port": 8100,
    }
    if deployment == "disjoint":
        vllm_config["visible_devices"] = "2,3"
    engine = VLLMGenerationEngine(_model("qwen3"), {"vllm": vllm_config})
    client = object()
    monkeypatch.setattr(
        engine,
        "_launch_client",
        lambda *_args: events.append("launch") or client,
    )
    monkeypatch.setattr(
        engine,
        "_connect_client",
        lambda *_args: pytest.fail("The coordinator connected instead of launching"),
    )

    assert engine._ensure_client() is client  # pylint: disable=protected-access
    assert events == [
        "launch",
        "shared vLLM server startup",
    ]


def test_owned_server_rejects_an_endpoint_that_is_already_in_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale service must not satisfy health checks for a new owned server."""
    connection = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        vllm_module.socket,
        "create_connection",
        lambda *_args, **_kwargs: connection,
    )
    monkeypatch.setattr(
        vllm_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("vLLM launched on an occupied endpoint"),
    )
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {"vllm": {"deployment": "colocated", "data_parallel_size": 1}},
    )

    with pytest.raises(RuntimeError, match="endpoint is already in use"):
        engine._launch_client("0", "127.0.0.1", 8100)  # pylint: disable=protected-access


def test_owned_health_response_does_not_hide_an_exited_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Health from another service cannot mask an owned process startup failure."""
    return_codes = iter((None, 17))
    process = SimpleNamespace(poll=lambda: next(return_codes))
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        process,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        vllm_module.time,
        "sleep",
        lambda _seconds: pytest.fail("An exited owned server was retried"),
    )

    with pytest.raises(RuntimeError, match="exited during startup with code 17"):
        client.wait_ready(1)


@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
def test_shared_non_coordinator_connects_without_launching(
    monkeypatch: pytest.MonkeyPatch,
    deployment: str,
) -> None:
    """Only rank zero may own either shared vLLM deployment."""
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(vllm_module.platform, "get_rank", lambda: 1)
    monkeypatch.setattr(vllm_module.platform, "get_world_size", lambda: 2)
    events = []
    monkeypatch.setattr(
        vllm_module,
        "synchronize_error",
        lambda _error, operation: events.append(operation),
    )
    vllm_config = {
        "deployment": deployment,
        "data_parallel_size": 1,
        "tensor_parallel_size": 2,
        "port": 8100,
    }
    if deployment == "disjoint":
        vllm_config["visible_devices"] = "2,3"
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {"vllm": vllm_config},
    )
    monkeypatch.setattr(
        engine,
        "_launch_client",
        lambda *_args: pytest.fail("A non-coordinator rank launched vLLM"),
    )
    client = SimpleNamespace(
        wait_ready=lambda _timeout: pytest.fail(
            "A non-owner duplicated the coordinator startup health check"
        ),
    )

    def connect(host: str, port: int) -> Any:
        """Record that every rank waits on the same endpoint before synchronizing."""
        events.append("connect")
        assert (host, port) == ("127.0.0.1", 8100)
        return client

    monkeypatch.setattr(engine, "_connect_client", connect)

    result = engine._ensure_client()  # pylint: disable=protected-access

    assert result is client
    assert events == [
        "connect",
        "shared vLLM server startup",
    ]


@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
def test_shared_close_waits_for_server_before_releasing_transfer(
    monkeypatch: pytest.MonkeyPatch,
    deployment: str,
) -> None:
    """IPC producer storage remains alive until either shared server has stopped."""
    events = []
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        None,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )
    monkeypatch.setattr(client, "close", lambda: events.append("server stopped"))
    transfer = SimpleNamespace(close=lambda: events.append("transfer released"))
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {
            "vllm": {
                "deployment": deployment,
                "data_parallel_size": 1,
                "tensor_parallel_size": 2,
            }
        },
        client=client,
        refitter=transfer,
    )

    def synchronized_call(_operation: str, callback: Any) -> Any:
        """Record the shutdown barrier around the owner operation."""
        result = callback()
        events.append("shutdown synchronized")
        return result

    monkeypatch.setattr(vllm_module, "synchronized_call", synchronized_call)

    engine.close()

    assert events == ["server stopped", "shutdown synchronized", "transfer released"]


def test_http_client_close_waits_for_process_group_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The owner does not complete shutdown while an EngineCore descendant remains."""
    process = SimpleNamespace(pid=123, poll=lambda: None, wait=lambda timeout: 0)
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        process,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )
    signals = []
    probes = 0

    def kill_process_group(process_group_id: int, sent_signal: int) -> None:
        """Keep the process group visible for one post-kill probe."""
        nonlocal probes
        signals.append((process_group_id, sent_signal))
        if sent_signal == 0:
            probes += 1
            if probes > 1:
                raise ProcessLookupError

    monkeypatch.setattr(vllm_module.os, "killpg", kill_process_group)
    monkeypatch.setattr(
        vllm_module._VLLMHTTPClient,  # pylint: disable=protected-access
        "_has_live_process_group_members",
        lambda _group: True,
    )
    monkeypatch.setattr(vllm_module.time, "sleep", lambda _seconds: None)

    client.close()

    assert signals == [
        (123, vllm_module.signal.SIGTERM),
        (123, vllm_module.signal.SIGKILL),
        (123, 0),
        (123, 0),
    ]


def test_http_client_close_ignores_reaped_resource_zombies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zombie-only process groups no longer hold NPU or IPC resources."""
    process = SimpleNamespace(pid=123, poll=lambda: None, wait=lambda timeout: 0)
    client = vllm_module._VLLMHTTPClient(  # pylint: disable=protected-access
        process,
        "http://127.0.0.1:8100",
        "qwen",
        request_timeout=1,
    )
    signals = []
    monkeypatch.setattr(
        vllm_module.os,
        "killpg",
        lambda process_group_id, sent_signal: signals.append(
            (process_group_id, sent_signal)
        ),
    )
    monkeypatch.setattr(
        vllm_module._VLLMHTTPClient,  # pylint: disable=protected-access
        "_has_live_process_group_members",
        lambda _group: False,
    )

    client.close()

    assert signals == [
        (123, vllm_module.signal.SIGTERM),
        (123, vllm_module.signal.SIGKILL),
        (123, 0),
    ]


@pytest.mark.parametrize("implementation", ["hyper", "native"])
def test_qwen3_internal_dp_weight_transfer_uses_one_shared_endpoint(
    implementation: str,
) -> None:
    """Both Qwen3 implementations submit one shared full-gather transaction."""
    transfer = ColocatedFullGatherWeightTransfer(
        resolve_vllm_model(_model(), implementation),
    )
    client = SimpleNamespace(base_url="http://127.0.0.1:8100")

    assert transfer._transfer_endpoints(client) == (  # pylint: disable=protected-access
        "http://127.0.0.1:8100",
    )


def test_server_command_preserves_explicit_prefill_and_logprob_semantics() -> None:
    """False cache flags and raw logprobs must reach the owned vLLM server."""
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {
            "vllm": {
                "model_implementation": "hyper",
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
                "enable_prompt_tokens_details": True,
                "attention_backend": "FLASH_ATTN",
                "logprobs_mode": "raw_logprobs",
            }
        },
        client=object(),
    )

    command = engine._server_command("127.0.0.1", 8100)  # pylint: disable=protected-access

    assert "--no-enable-prefix-caching" in command
    assert "--no-enable-chunked-prefill" in command
    assert "--enable-prompt-tokens-details" in command
    assert command[command.index("--attention-backend") + 1] == "FLASH_ATTN"
    assert command[command.index("--logprobs-mode") + 1] == "raw_logprobs"


def test_server_environment_carries_consistency_profile() -> None:
    """The isolated rollout process receives the paired profile identity."""
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {
            "vllm": {
                "batch_invariant": True,
                "consistency_profile": "qwen3_ascend_consistency_v1",
                "model_implementation": "hyper",
            }
        },
        client=object(),
    )

    environment = engine._server_environment("0")  # pylint: disable=protected-access

    assert environment["VLLM_BATCH_INVARIANT"] == "1"
    assert environment["HYPER_RL_CONSISTENCY_PROFILE"] == "qwen3_ascend_consistency_v1"


def test_server_environment_can_isolate_hccl_ports_from_trainer() -> None:
    """A colocated server can use a distinct HCCL socket range."""
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {
            "vllm": {
                "server_hccl_if_base_port": 64400,
                "server_hccl_npu_socket_port_range": "64400-64500",
            }
        },
        client=object(),
    )

    environment = engine._server_environment("0,1")  # pylint: disable=protected-access

    assert environment["HCCL_IF_BASE_PORT"] == "64400"
    assert environment["HCCL_NPU_SOCKET_PORT_RANGE"] == "64400-64500"


def test_server_environment_removes_inherited_consistency_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An off-profile rollout must not inherit a parent process patch selection."""
    monkeypatch.setenv(
        "HYPER_RL_CONSISTENCY_PROFILE",
        "qwen3_ascend_consistency_v1",
    )
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {"vllm": {"model_implementation": "hyper"}},
        client=object(),
    )

    environment = engine._server_environment("0")  # pylint: disable=protected-access

    assert "HYPER_RL_CONSISTENCY_PROFILE" not in environment


def test_generation_rejects_identity_change_while_request_is_served() -> None:
    """Worker identity must bracket the completion request rather than label it optimistically."""
    engine = object.__new__(VLLMGenerationEngine)
    engine._config = {"consistency_profile": "profile"}
    engine._ensure_client = object
    identities = iter(((0, "digest-v0"), (1, "digest-v1")))

    def next_identity(_client: object) -> tuple[int, str]:
        """Return the next fake worker-owned identity."""
        return next(identities)

    engine._weight_sync = SimpleNamespace(generation_identity=next_identity)
    engine._completion_records = lambda _client, _prompts, _settings, _row_seeds: [([2], None)]
    request = SimpleNamespace(
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones((1, 1), dtype=torch.bool),
        settings=SimpleNamespace(
            max_new_tokens=1,
            pad_token_id=0,
            collect_log_probs=False,
        ),
        row_seeds=None,
    )

    with pytest.raises(RuntimeError, match="changed while serving"):
        engine._generate(request)  # pylint: disable=protected-access


def test_generation_request_failure_synchronizes_before_post_identity() -> None:
    """A failed rank must synchronize before successful ranks enter the post-request collective."""
    engine = object.__new__(VLLMGenerationEngine)
    engine._ensure_client = object
    events = []

    def generation_identity(_client: object) -> tuple[int, str]:
        """Record each collective policy-identity boundary."""
        events.append("identity")
        return 0, "digest-v0"

    def fail_completion(*_args: Any) -> None:
        """Fail one rank's local HTTP generation request."""
        events.append("request")
        raise RuntimeError("request failed")

    def synchronize_error(error: Optional[Exception], operation: str) -> None:
        """Model the rank-synchronized request failure."""
        events.append(operation)
        if error is not None:
            raise error

    engine._weight_sync = SimpleNamespace(generation_identity=generation_identity)
    engine._completion_records = fail_completion
    engine.synchronize_error = synchronize_error
    request = SimpleNamespace(
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones((1, 1), dtype=torch.bool),
        settings=SimpleNamespace(max_new_tokens=1),
        row_seeds=None,
    )

    with pytest.raises(RuntimeError, match="request failed"):
        engine._generate(request)  # pylint: disable=protected-access

    assert events == ["identity", "request", "generation request"]


@pytest.mark.parametrize("failure_stage", ["prompt", "request", "no_result"])
def test_generation_request_failure_keeps_two_rank_collectives_aligned(
    failure_stage: str,
) -> None:
    """Both ranks stop before the post-request identity collective after one local failure."""
    barriers = {
        "generation request": Barrier(2),
        "generation": Barrier(2),
    }
    local_errors = {
        operation: [None, None]
        for operation in barriers
    }
    operations: list[list[str]] = [[], []]
    identity_calls = [0, 0]

    class FailingRows:
        """Raise while one rank materializes its prompt rows."""

        def __iter__(self) -> Any:
            """Fail before the local HTTP request starts."""
            raise RuntimeError("prompt failed")

    def run_rank(rank: int) -> None:
        """Execute one simulated Trainer rank against a shared error collective."""
        engine = object.__new__(VLLMGenerationEngine)
        engine._ensure_client = object

        def generation_identity(_client: object) -> tuple[int, str]:
            """Count pre- and post-request identity collectives."""
            identity_calls[rank] += 1
            return 0, "digest-v0"

        def completion_records(*_args: Any) -> Optional[list[tuple[list[int], None]]]:
            """Fail only rank zero's local completion request when requested."""
            if rank == 0 and failure_stage == "request":
                raise RuntimeError("request failed")
            if rank == 0 and failure_stage == "no_result":
                return None
            return [([2], None)]

        def synchronize_error(error: Optional[Exception], operation: str) -> None:
            """Model one indexed all-gather of local error strings."""
            operations[rank].append(operation)
            local_errors[operation][rank] = error
            barriers[operation].wait(timeout=5)
            if any(local_error is not None for local_error in local_errors[operation]):
                raise RuntimeError("synchronized generation failure")

        engine._weight_sync = SimpleNamespace(generation_identity=generation_identity)
        engine._completion_records = completion_records
        engine.synchronize_error = synchronize_error
        input_ids = FailingRows() if rank == 0 and failure_stage == "prompt" else torch.tensor([[1]])
        request = SimpleNamespace(
            input_ids=input_ids,
            attention_mask=torch.ones((1, 1), dtype=torch.bool),
            settings=SimpleNamespace(max_new_tokens=1),
            row_seeds=None,
        )
        engine.generate(request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_rank, rank) for rank in range(2)]
        for future in futures:
            with pytest.raises(RuntimeError, match="synchronized generation failure"):
                future.result(timeout=5)

    assert identity_calls == [1, 1]
    assert operations == [
        ["generation request", "generation"],
        ["generation request", "generation"],
    ]


@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
def test_generation_failure_synchronizes_both_shared_deployments(
    deployment: str,
) -> None:
    """A local request failure reaches the common rank-synchronized boundary."""
    engine = object.__new__(VLLMGenerationEngine)
    engine._deployment = deployment
    engine._generate = lambda _request: (_ for _ in ()).throw(RuntimeError("request failed"))
    synchronized = []

    def synchronize_error(error: Optional[Exception], operation: str) -> None:
        """Record and rethrow the synchronized local failure."""
        synchronized.append((str(error), operation))
        raise error

    engine.synchronize_error = synchronize_error

    with pytest.raises(RuntimeError, match="request failed"):
        engine.generate(object())

    assert synchronized == [("request failed", "generation")]


def test_one_worker_fingerprint_mismatch_rejects_publication() -> None:
    """A mismatch from any DP x TP worker invalidates the shared publication."""
    expected = {
        "algorithm": "qwen_norms_f32_v3",
        "tensor_count": 1,
        "value_count": 1,
        "digest": "expected",
        "tensors": {"model.norm.weight": "expected"},
    }
    workers = [
        {**expected, "version": 1, "rank": 0},
        {
            **expected,
            "digest": "changed",
            "tensors": {"model.norm.weight": "changed"},
            "version": 1,
            "rank": 1,
        },
    ]

    with pytest.raises(RuntimeError, match="rank.*1.*changed"):
        verify_policy_fingerprints(expected, workers, expected_version=1)


class _LifecycleClient(VLLMWeightSyncClientMixin):
    """In-memory control client for the colocated publication transaction."""

    def __init__(
        self,
        fail_post_refit_pause: bool = False,
        fail_resume: bool = False,
        fail_residency: bool = False,
        fail_compensating_pause: bool = False,
    ) -> None:
        """Initialize fake worker identity, residency, and call history."""
        self.version = 0
        self.digest = "digest-v0"
        self.paused = False
        self.sleeping = False
        self.fail_post_refit_pause = fail_post_refit_pause
        self.fail_resume = fail_resume
        self.fail_residency = fail_residency
        self.fail_compensating_pause = fail_compensating_pause
        self.pause_count = 0
        self.calls: list[str] = []

    def get_policy_weight_fingerprints(
        self,
        base_url: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return one worker-owned identity."""
        del base_url
        self.calls.append("identity")
        return [
            {
                "algorithm": "qwen_norms_f32_v3",
                "tensor_count": 1,
                "value_count": 1,
                "digest": self.digest,
                "tensors": {"model.norm.weight": self.digest},
                "version": self.version,
                "rank": 0,
            }
        ]

    def verify_policy_weight_identity(
        self,
        expected_version: int,
        expected_fingerprint: dict[str, Any],
    ) -> None:
        """Validate the expected identity against the fake worker."""
        self.calls.append("verify_identity")
        assert expected_version == self.version
        assert expected_fingerprint["digest"] == self.digest

    def sleep(self, level: int = 1, mode: str = "wait") -> None:
        """Enter training residency with generation blocked."""
        assert level == 1 and mode == "wait"
        self.calls.append("sleep")
        self.sleeping = True
        self.paused = True

    def is_sleeping(self) -> bool:
        """Return fake device residency."""
        self.calls.append("is_sleeping")
        return self.sleeping

    def pause(self) -> None:
        """Close fake generation admission."""
        self.pause_count += 1
        self.calls.append("pause")
        if self.fail_post_refit_pause and self.pause_count == 2:
            raise RuntimeError("post-refit cache reset failed")
        if self.fail_compensating_pause and self.pause_count == 3:
            raise RuntimeError("compensating pause failed")
        self.paused = True

    def is_paused(self) -> bool:
        """Return fake scheduler admission state."""
        self.calls.append("is_paused")
        return self.paused

    def wake_up(self, tags: tuple[str, ...]) -> None:
        """Restore selected fake memory tags without resuming generation."""
        self.calls.append(f"wake:{','.join(tags)}")
        self.sleeping = False

    def resume(self) -> None:
        """Open fake generation admission."""
        self.calls.append("resume")
        self.paused = False
        if self.fail_residency:
            self.sleeping = True
        if self.fail_resume:
            raise RuntimeError("planned resume failure")


class _LifecycleTransfer:
    """Simulate one successful transfer and expose its verified fingerprint."""

    last_policy_fingerprint = None

    def publish(
        self,
        client: _LifecycleClient,
        snapshot: PolicySnapshot,
    ) -> None:
        """Pause and publish one fake worker-owned snapshot."""
        client.pause()
        client.version = snapshot.version
        client.digest = f"digest-v{snapshot.version}"
        self.last_policy_fingerprint = {"digest": client.digest}
        client.calls.append("transfer")


class _LegacyLifecycleTransfer:
    """Retain the injected pre-publication transfer contract."""

    last_policy_fingerprint = None

    def transfer(
        self,
        client: _LifecycleClient,
        snapshot: PolicySnapshot,
    ) -> None:
        """Publish through the historical transfer verb."""
        client.pause()
        client.version = snapshot.version
        client.digest = f"digest-v{snapshot.version}"
        self.last_policy_fingerprint = {"digest": client.digest}
        client.calls.append("legacy_transfer")


def test_memory_wake_marks_scheduler_to_remain_paused() -> None:
    """Tagged memory restore must tell the patched EngineCore to keep admission closed."""
    requests = []

    class _Client(VLLMWeightSyncClientMixin):
        def _request(self, method, route, payload=None, timeout=None, base_url=None):
            requests.append((method, route, payload, timeout, base_url))
            return {}

    _Client().wake_up(("weights", "kv_cache"))

    assert requests == [
        (
            "POST",
            "wake_up?tags=weights&tags=kv_cache&tags=_hyper_keep_scheduler_paused",
            None,
            None,
            None,
        )
    ]


def test_disjoint_refit_resumes_before_controller_publication() -> None:
    """External admission opens while the controller still blocks generation in refit."""
    client = _LifecycleClient()
    sync = ActorRolloutWeightSync(
        "qwen",
        "disjoint",
        lambda: client,
        _LifecycleTransfer(),
    )

    sync.update_weights(PolicySnapshot(1, "qwen", object()))

    assert sync.policy_version == 1
    assert sync.phase == "rollout"
    assert client.calls.index("transfer") < client.calls.index("resume")


def test_injected_legacy_transfer_remains_compatible() -> None:
    """Canonical publication accepts existing externally injected refitters."""
    client = _LifecycleClient()
    sync = ActorRolloutWeightSync(
        "qwen",
        "disjoint",
        lambda: client,
        _LegacyLifecycleTransfer(),
    )

    sync.update_weights(PolicySnapshot(1, "qwen", object()))

    assert sync.policy_version == 1
    assert client.calls.index("legacy_transfer") < client.calls.index("resume")


def test_colocated_refit_resets_cache_before_version_publication() -> None:
    """A worker version becomes visible only after reset, identity, and resume succeed."""
    client = _LifecycleClient()
    transfer = _LifecycleTransfer()
    sync = ActorRolloutWeightSync("qwen", "colocated", lambda: client, transfer)

    sync.prepare_for_training()
    sync.update_weights(PolicySnapshot(1, "qwen", object()))
    assert sync.policy_version == 0
    assert sync.phase == "refit"

    sync.prepare_for_rollout()

    assert sync.policy_version == 1
    assert sync.phase == "rollout"
    wake_index = client.calls.index("wake:kv_cache")
    reset_index = client.calls.index("pause", client.calls.index("transfer"))
    pending_identity_index = client.calls.index("verify_identity", reset_index)
    resume_index = client.calls.index("resume")
    assert wake_index < reset_index < pending_identity_index < resume_index


def test_colocated_refit_reset_failure_does_not_publish_version() -> None:
    """A failed post-refit cache reset leaves the pending policy unpublished."""
    client = _LifecycleClient(fail_post_refit_pause=True)
    sync = ActorRolloutWeightSync(
        "qwen",
        "colocated",
        lambda: client,
        _LifecycleTransfer(),
    )
    sync.prepare_for_training()
    sync.update_weights(PolicySnapshot(1, "qwen", object()))

    with pytest.raises(RuntimeError, match="post-refit cache reset failed"):
        sync.prepare_for_rollout()

    assert sync.policy_version == 0
    assert sync.phase == "refit"
    assert client.paused is True


def test_colocated_resume_failure_repauses_without_publishing_version() -> None:
    """A failed resume closes admission again and leaves controller identity at V0."""
    client = _LifecycleClient(fail_resume=True)
    sync = ActorRolloutWeightSync(
        "qwen",
        "colocated",
        lambda: client,
        _LifecycleTransfer(),
    )
    sync.prepare_for_training()
    sync.update_weights(PolicySnapshot(1, "qwen", object()))

    with pytest.raises(RuntimeError, match="planned resume failure"):
        sync.prepare_for_rollout()

    assert sync.policy_version == 0
    assert sync.phase == "refit"
    assert client.paused is True
    assert client.calls[-2:] == ["resume", "pause"]


def test_colocated_residency_failure_repauses_without_publishing_version() -> None:
    """A post-resume residency mismatch closes admission and preserves V0."""
    client = _LifecycleClient(fail_residency=True)
    sync = ActorRolloutWeightSync(
        "qwen",
        "colocated",
        lambda: client,
        _LifecycleTransfer(),
    )
    sync.prepare_for_training()
    sync.update_weights(PolicySnapshot(1, "qwen", object()))

    with pytest.raises(RuntimeError, match="remained paused or sleeping"):
        sync.prepare_for_rollout()

    assert sync.policy_version == 0
    assert sync.phase == "refit"
    assert client.paused is True
    assert client.calls[-1] == "pause"


def test_resume_and_compensating_pause_failures_preserve_both_causes() -> None:
    """A compensation error reports both failures without publishing V1."""
    client = _LifecycleClient(
        fail_resume=True,
        fail_compensating_pause=True,
    )
    sync = ActorRolloutWeightSync(
        "qwen",
        "colocated",
        lambda: client,
        _LifecycleTransfer(),
    )
    sync.prepare_for_training()
    sync.update_weights(PolicySnapshot(1, "qwen", object()))

    with pytest.raises(RuntimeError) as error:
        sync.prepare_for_rollout()

    assert "planned resume failure" in str(error.value)
    assert "compensating pause failed" in str(error.value)
    assert sync.policy_version == 0
    assert sync.phase == "refit"


def test_shared_deployment_caches_one_verified_generation_identity() -> None:
    """Generation must bracket requests with current identity verification."""
    client = _LifecycleClient()
    sync = ActorRolloutWeightSync(
        "qwen",
        "colocated",
        lambda: client,
        _LifecycleTransfer(),
    )

    first = sync.generation_identity(client)
    second = sync.generation_identity(client)

    assert first == second == (0, "digest-v0")
    assert client.calls == ["identity", "verify_identity", "verify_identity"]
