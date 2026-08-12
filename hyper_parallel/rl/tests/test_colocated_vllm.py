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
"""Contract tests for strong-sync colocated vLLM rollout."""

from dataclasses import dataclass
import json
import os
import sys
import threading
import types
from typing import Any

import pytest
import torch

from rl import trainer as trainer_backend
from rl.roles.model import ModelRegistration
from rl.roles.rollout import GenerationRequest, GenerationSettings
from rl.roles.rollout.vllm import NPUIPCWeightRefitter, VLLMGenerationEngine
from rl.roles.rollout import vllm as vllm_backend
from rl.roles.rollout.vllm_policy import is_policy_fingerprint_weight, map_policy_state_dict
from rl.roles.weight_sync import PolicySnapshot
from rl.trainer import SyncTrainer, _validate_rollout_and_agentic


def _colocated_rollout_config() -> dict[str, Any]:
    """Return the minimal valid colocated rollout configuration."""
    return {
        "engine": "vllm",
        "max_new_tokens": 4,
        "vllm": {
            "deployment": "colocated",
            "port": 8100,
            "tensor_parallel_size": 1,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.5,
        },
    }


def _colocated_accelerator_config() -> dict[str, Any]:
    """Return the required multi-rank FSDP residency configuration."""
    return {
        "dp_shard": 2,
        "cpu_offload": True,
        "reshard_after_forward": True,
    }


def test_colocated_config_requires_fsdp_cpu_offload_tp1_and_npu(monkeypatch) -> None:
    """Reject topologies that cannot release training state or map local IPC."""
    agentic = {"environment": "gsm8k", "max_turns": 1, "max_observation_tokens": 0}
    rollout = _colocated_rollout_config()
    accelerator = _colocated_accelerator_config()
    _validate_rollout_and_agentic(rollout, agentic, accelerator)

    rollout["vllm"]["tensor_parallel_size"] = 2
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        _validate_rollout_and_agentic(rollout, agentic, accelerator)
    rollout["vllm"]["tensor_parallel_size"] = 1

    accelerator["cpu_offload"] = False
    with pytest.raises(ValueError, match="cpu_offload=true"):
        _validate_rollout_and_agentic(rollout, agentic, accelerator)
    accelerator["cpu_offload"] = True

    monkeypatch.setattr(trainer_backend.platform, "device_type", lambda: "cuda")
    with pytest.raises(ValueError, match="Torch NPU platform"):
        _validate_rollout_and_agentic(rollout, agentic, accelerator)


def test_vllm_model_implementation_and_native_state_names() -> None:
    """Native refit reuses HF names while Hyper keeps Actor names unchanged."""
    state_dict = {
        "model.embed_tokens.weight": object(),
        "model.layers.0.input_layernorm.weight": object(),
        "lm_head.weight": object(),
    }
    native = map_policy_state_dict(state_dict, "native")
    assert list(native) == [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "lm_head.weight",
    ]
    assert list(native.values()) == list(state_dict.values())
    assert map_policy_state_dict(state_dict, "hyper") == state_dict

    rollout = _colocated_rollout_config()
    rollout["vllm"]["model_implementation"] = "unsupported"
    with pytest.raises(ValueError, match="model_implementation"):
        _validate_rollout_and_agentic(
            rollout,
            {"environment": "gsm8k", "max_turns": 1, "max_observation_tokens": 0},
            _colocated_accelerator_config(),
        )

    rollout = _colocated_rollout_config()
    rollout["vllm"]["request_concurrency"] = 0
    with pytest.raises(ValueError, match="request_concurrency"):
        _validate_rollout_and_agentic(
            rollout,
            {"environment": "gsm8k", "max_turns": 1, "max_observation_tokens": 0},
            _colocated_accelerator_config(),
        )


def test_policy_fingerprint_is_canonical_and_reports_changed_tensors(monkeypatch) -> None:
    """Actor fingerprints should ignore native namespaces and diagnose mismatches."""
    monkeypatch.setattr(vllm_backend.platform, "tensor_type_cast", lambda tensor, _dtype: tensor.float())
    monkeypatch.setattr(vllm_backend.platform, "tensor_to_numpy", lambda tensor: tensor.numpy())
    actor = vllm_backend._policy_weight_fingerprint(  # pylint: disable=protected-access
        {"model.language_model.norm.weight": torch.tensor([1.0, 2.0])}
    )
    hyper = vllm_backend._policy_weight_fingerprint(  # pylint: disable=protected-access
        {"model.norm.weight": torch.tensor([1.0, 2.0])}
    )
    changed = vllm_backend._policy_weight_fingerprint(  # pylint: disable=protected-access
        {"model.norm.weight": torch.tensor([1.0, 3.0])}
    )

    assert actor == hyper
    assert list(actor["tensors"]) == ["model.norm.weight"]
    with pytest.raises(RuntimeError, match=r"changed.*model.norm.weight"):
        vllm_backend._verify_policy_fingerprints(actor, [changed])  # pylint: disable=protected-access

    assert not is_policy_fingerprint_weight("model.layers.0.linear_attn.norm.weight")


def test_seeded_http_generation_uses_one_stable_seed_per_response() -> None:
    """M3 response groups must not share one ambiguous batched RNG seed."""
    payloads = []

    class Client(vllm_backend._VLLMHTTPClient):  # pylint: disable=protected-access
        def __init__(self) -> None:
            """Construct without a real server process."""
            self._model_name = "qwen"

        def _request(self, method: str, route: str, payload: Any = None, **_kwargs: Any) -> dict[str, Any]:
            """Record each seeded completion request."""
            assert method == "POST"
            assert route == "v1/completions"
            payloads.append(payload)
            return {
                "choices": [
                    {
                        "index": 0,
                        "token_ids": [int(payload["seed"])],
                        "logprobs": None,
                    }
                ]
            }

    settings = GenerationSettings(1, 1.0, 1.0, 0, True, 0, 2, seed=20260811)
    records = Client().generate_tokens([[1, 2], [1, 2]], settings)

    assert [payload["seed"] for payload in payloads] == [20260811, 20260812]
    assert [payload["prompt"] for payload in payloads] == [[[1, 2]], [[1, 2]]]
    assert [record[0] for record in records] == [[20260811], [20260812]]


def test_seeded_http_generation_preserves_order_with_concurrent_requests() -> None:
    """Concurrent independent seeds must retain prompt-major response order."""
    second_request_started = threading.Event()
    request_lock = threading.Lock()
    active_requests = 0
    max_active_requests = 0

    class Client(vllm_backend._VLLMHTTPClient):  # pylint: disable=protected-access
        def __init__(self) -> None:
            """Construct without a real server process."""
            self._model_name = "qwen"

        def _request(self, method: str, route: str, payload: Any = None, **_kwargs: Any) -> dict[str, Any]:
            """Force the first request to finish after the second one."""
            nonlocal active_requests, max_active_requests
            assert method == "POST"
            assert route == "v1/completions"
            with request_lock:
                active_requests += 1
                max_active_requests = max(max_active_requests, active_requests)
            try:
                if payload["seed"] == 20260811:
                    assert second_request_started.wait(timeout=1)
                else:
                    second_request_started.set()
                return {
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": [int(payload["seed"])],
                            "logprobs": {
                                "token_logprobs": [-float(payload["seed"])],
                            },
                        }
                    ]
                }
            finally:
                with request_lock:
                    active_requests -= 1

    settings = GenerationSettings(
        1,
        1.0,
        1.0,
        0,
        True,
        0,
        2,
        collect_log_probs=True,
        seed=20260811,
    )
    records = Client().generate_tokens([[1], [2]], settings, request_concurrency=2)

    assert max_active_requests == 2
    assert [record[0] for record in records] == [[20260811], [20260812]]
    assert [record[1] for record in records] == [[-20260811.0], [-20260812.0]]


def test_rl_config_propagates_cpu_offload_to_hyper_trainer() -> None:
    """The colocated YAML flag must reach Qwen's FSDP policy construction."""
    config = {
        "model": {
            "name": "qwen3_5",
            "weights_path": "/model",
            "tokenizer_path": "/model",
            "config_overrides": None,
        },
        "data": {
            "train_path": "/data/train.parquet",
            "max_prompt_length": 16,
        },
        "rollout": _colocated_rollout_config(),
        "agentic": {"max_turns": 1, "max_observation_tokens": 0},
        "train": {
            "max_steps": 2,
            "prompt_batch_size": 2,
            "accelerator": _colocated_accelerator_config(),
            "optimizer": {},
            "mixed_precision": {},
            "checkpoint": {"output_dir": "/tmp/hyper-rl-test", "save_final": False},
        },
    }

    base_config = SyncTrainer._build_base_config(config)  # pylint: disable=protected-access

    assert base_config.train.accelerator.cpu_offload is True
    assert base_config.train.accelerator.dp_shard == 2
    assert base_config.train.comm_backend == "cpu:gloo,npu:hccl"
    assert base_config.train.micro_batch_size == 2
    assert base_config.train.global_batch_size == 4


def test_colocated_server_uses_rank_local_npu_ipc_and_sleep(monkeypatch) -> None:
    """Each trainer rank must launch a TP1 IPC server on its own physical NPU."""
    process_calls = []

    class Process:
        """Stand-in process handle."""

    class HTTPClient:
        """Capture readiness without starting vLLM."""

        def __init__(self, process: Process, base_url: str, model_name: str, request_timeout: float) -> None:
            """Store the server construction arguments."""
            self.args = (process, base_url, model_name, request_timeout)

        @staticmethod
        def wait_ready(startup_timeout: float) -> None:
            """Accept the configured startup timeout."""
            assert startup_timeout == 300

        @staticmethod
        def close() -> None:
            """Reject cleanup on a successful startup."""
            raise AssertionError("Successful startup must not close the client")

    def popen(command: list[str], **kwargs: Any) -> Process:
        """Capture the rank-local server command and environment."""
        process_calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(vllm_backend.subprocess, "Popen", popen)
    monkeypatch.setattr(vllm_backend, "_VLLMHTTPClient", HTTPClient)
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5")
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    model = ModelRegistration("qwen", "qwen3_5", "/model", "/tokenizer")
    engine = VLLMGenerationEngine(
        model,
        {
            "vllm": {
                "deployment": "colocated",
                "tensor_parallel_size": 1,
                "port": 8100,
            }
        },
    )

    client = engine._ensure_client()  # pylint: disable=protected-access

    command, process_kwargs = process_calls[0]
    assert command[command.index("--port") + 1] == "8101"
    transfer_config = json.loads(command[command.index("--weight-transfer-config") + 1])
    assert transfer_config == {"backend": "ipc"}
    assert "--enable-sleep-mode" in command
    assert json.loads(command[command.index("--additional-config") + 1]) == {"weight_nz_mode": 0}
    assert process_kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "5"
    assert process_kwargs["env"]["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"
    assert "PYTORCH_NPU_ALLOC_CONF" not in process_kwargs["env"]
    assert client.args[1] == "http://127.0.0.1:8101"
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"


def test_colocated_version_commits_only_after_refit_and_wake(monkeypatch) -> None:
    """A refitted policy remains pending until KV cache is awake on every rank."""
    calls = []

    class Client:
        """Record strong-sync residency calls."""

        sleeping = False

        def sleep(self, level: int, mode: str) -> None:
            """Record entry into level-1 sleep."""
            calls.append(("sleep", level, mode))
            self.sleeping = True

        def is_sleeping(self) -> bool:
            """Return the recorded residency state."""
            calls.append("is_sleeping")
            return self.sleeping

        def wake_up(self, tags: tuple[str, ...]) -> None:
            """Record restored allocator tags."""
            calls.append(("wake_up", tags))
            self.sleeping = False

        @staticmethod
        def resume() -> None:
            """Record generation admission reopening."""
            calls.append("resume")

    class Refitter:
        """Record the completed weight transaction."""

        @staticmethod
        def refit(client: Client, snapshot: PolicySnapshot) -> None:
            """Record the pending policy publication."""
            calls.append(("refit", client, snapshot.version))

    monkeypatch.setattr(vllm_backend.platform, "get_world_size", lambda: 1)
    client = Client()
    model = ModelRegistration("qwen", "qwen3_5", "/model", "/model")
    engine = VLLMGenerationEngine(
        model,
        {"vllm": {"deployment": "colocated"}},
        client=client,
        refitter=Refitter(),
    )
    engine.prepare_for_training()
    engine.update_weights(PolicySnapshot(1, "qwen", payload={"weight": 1}))
    assert engine.policy_version == 0

    engine.prepare_for_rollout()

    assert engine.policy_version == 1
    assert ("wake_up", ("kv_cache",)) in calls
    assert calls[-2:] == ["resume", "is_sleeping"]


def test_colocated_training_records_verified_initial_fingerprint(monkeypatch) -> None:
    """The first learning gate must compare refit weights with the loaded policy."""
    fingerprint = {
        "algorithm": "qwen3_5_norms_f32_v2",
        "tensor_count": 1,
        "value_count": 2,
        "digest": "initial",
        "tensors": {"model.norm.weight": "tensor-digest"},
    }

    class Client(vllm_backend._VLLMHTTPClient):  # pylint: disable=protected-access
        """Expose one valid initial worker fingerprint and sleep state."""

        def __init__(self) -> None:
            self.sleeping = False

        @staticmethod
        def get_policy_weight_fingerprints(version: int) -> list[dict[str, Any]]:
            assert version == 0
            return [{**fingerprint, "rank": 0}]

        def sleep(self, level: int, mode: str) -> None:
            assert (level, mode) == (1, "wait")
            self.sleeping = True

        def is_sleeping(self) -> bool:
            return self.sleeping

    monkeypatch.setattr(vllm_backend.platform, "get_world_size", lambda: 1)
    engine = VLLMGenerationEngine(
        ModelRegistration("qwen", "qwen3_5", "/model", "/model"),
        {"vllm": {"deployment": "colocated"}},
        client=Client(),
        refitter=object(),
    )

    engine.prepare_for_training()

    assert engine.policy_fingerprint == "initial"
    assert engine.policy_fingerprint_changed is False


def test_colocated_wake_failure_keeps_pending_version_uncommitted(monkeypatch) -> None:
    """A replica that cannot restore KV cache must poison the pending policy."""
    class Client:
        """Enter training sleep but reject the post-refit wake."""

        sleeping = False

        def sleep(self, level: int, mode: str) -> None:
            """Enter the mocked training residency state."""
            del level, mode
            self.sleeping = True

        def is_sleeping(self) -> bool:
            """Return the mocked residency state."""
            return self.sleeping

        @staticmethod
        def wake_up(tags: tuple[str, ...]) -> None:
            """Reject KV restoration."""
            del tags
            raise RuntimeError("KV allocation failed")

    class Refitter:
        """Accept the weight transfer before wake fails."""

        @staticmethod
        def refit(client: Client, snapshot: PolicySnapshot) -> None:
            """Accept one mocked weight update."""
            del client, snapshot

    monkeypatch.setattr(vllm_backend.platform, "get_world_size", lambda: 1)
    model = ModelRegistration("qwen", "qwen3_5", "/model", "/model")
    engine = VLLMGenerationEngine(
        model,
        {"vllm": {"deployment": "colocated"}},
        client=Client(),
        refitter=Refitter(),
    )
    engine.prepare_for_training()
    engine.update_weights(PolicySnapshot(1, "qwen", payload={"weight": 1}))

    with pytest.raises(RuntimeError, match="KV allocation failed"):
        engine.prepare_for_rollout()

    assert engine.policy_version == 0


def test_colocated_checkpoint_resume_wakes_weights_and_kv(monkeypatch) -> None:
    """A checkpoint-only sleep restores both inference residency tags."""
    calls = []

    class Client:
        """Record a sleep/wake cycle without a policy refit."""

        sleeping = False

        def sleep(self, level: int, mode: str) -> None:
            """Enter checkpoint-time sleep."""
            del level, mode
            self.sleeping = True

        def is_sleeping(self) -> bool:
            """Return checkpoint-time residency state."""
            return self.sleeping

        def wake_up(self, tags: tuple[str, ...]) -> None:
            """Record tags restored after checkpointing."""
            calls.append(tags)
            self.sleeping = False

        @staticmethod
        def resume() -> None:
            """Accept generation admission reopening."""
            return None

    monkeypatch.setattr(vllm_backend.platform, "get_world_size", lambda: 1)
    model = ModelRegistration("qwen", "qwen3_5", "/model", "/model")
    engine = VLLMGenerationEngine(
        model,
        {"vllm": {"deployment": "colocated"}},
        client=Client(),
        refitter=object(),
    )

    engine.prepare_for_training()
    engine.prepare_for_rollout()

    assert calls == [("weights", "kv_cache")]
    assert engine.policy_version == 0


def test_colocated_generation_failure_is_synchronized(monkeypatch) -> None:
    """A rank-local HTTP failure must reach every replica before the next phase."""
    class Client(vllm_backend._VLLMHTTPClient):  # pylint: disable=protected-access
        def __init__(self) -> None:
            """Construct without a real process."""

        @staticmethod
        def generate_tokens(
            prompt_token_ids: list[list[int]],
            settings: Any,
            request_concurrency: int = 1,
        ) -> Any:
            """Reject one local generation request."""
            del prompt_token_ids, settings, request_concurrency
            raise RuntimeError("local server exited")

    def all_gather_object(outputs: list[Any], local_error: Any) -> None:
        """Expose the local error to every mocked rank."""
        outputs[:] = [local_error, None]

    monkeypatch.setattr(vllm_backend.platform, "get_world_size", lambda: 2)
    monkeypatch.setattr(vllm_backend.platform, "all_gather_object", all_gather_object)
    model = ModelRegistration("qwen", "qwen3_5", "/model", "/model")
    engine = VLLMGenerationEngine(
        model,
        {"vllm": {"deployment": "colocated"}},
        client=Client(),
        refitter=object(),
    )
    request = GenerationRequest(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.tensor([[1, 1]]),
        settings=GenerationSettings(1, 0.0, 1.0, 0, False, 0, 2),
    )

    with pytest.raises(RuntimeError, match="local server exited"):
        engine.generate(request)


def test_colocated_server_startup_failure_is_synchronized(monkeypatch) -> None:
    """Server startup failure must enter the rank-aligned error exchange."""
    def all_gather_object(outputs: list[Any], local_error: Any) -> None:
        """Expose the local startup error to every mocked rank."""
        outputs[:] = [local_error, None]

    def fail_startup() -> None:
        """Reject local server construction."""
        raise RuntimeError("server startup failed")

    monkeypatch.setattr(vllm_backend.platform, "get_world_size", lambda: 2)
    monkeypatch.setattr(vllm_backend.platform, "all_gather_object", all_gather_object)
    model = ModelRegistration("qwen", "qwen3_5", "/model", "/model")
    engine = VLLMGenerationEngine(
        model,
        {"vllm": {"deployment": "colocated"}},
        refitter=object(),
    )
    monkeypatch.setattr(engine, "_ensure_client", fail_startup)

    with pytest.raises(RuntimeError, match="server startup failed"):
        engine.prepare_for_training()


def test_vllm_cleanup_kills_descendants_after_server_leader_exits(monkeypatch) -> None:
    """A clean API-server exit must not leave EngineCore descendants alive."""
    calls = []

    class Process:
        """Exit immediately when waited."""

        pid = 123

        @staticmethod
        def poll() -> int:
            """Report that the process-group leader already exited."""
            return 0

        @staticmethod
        def wait(timeout: int) -> int:
            """Accept the graceful leader wait."""
            calls.append(("wait", timeout))
            return 0

    monkeypatch.setattr(
        vllm_backend.os,
        "killpg",
        lambda process_id, sig: calls.append(("killpg", process_id, sig)),
    )
    monkeypatch.setattr(vllm_backend.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))
    client = vllm_backend._VLLMHTTPClient(  # pylint: disable=protected-access
        Process(),
        "http://127.0.0.1:8000",
        "qwen",
        60,
    )

    client.close()

    assert calls[0] == ("killpg", 123, vllm_backend.signal.SIGTERM)
    assert calls[-1] == ("killpg", 123, vllm_backend.signal.SIGKILL)


def test_npu_ipc_staging_failure_is_synchronized(monkeypatch) -> None:
    """A rank-local full-state staging failure must stop every refit rank."""
    class Client(vllm_backend._VLLMHTTPClient):  # pylint: disable=protected-access
        def __init__(self) -> None:
            """Construct without a real process."""

    def all_gather_object(outputs: list[Any], local_error: Any) -> None:
        """Expose one rank's staging failure to all mocked ranks."""
        outputs[:] = [local_error, None]

    def fail_staging(_payload: Any) -> dict[str, Any]:
        """Reject one rank's device-state allocation."""
        raise RuntimeError("NPU staging OOM")

    monkeypatch.setattr(vllm_backend.platform, "get_world_size", lambda: 2)
    monkeypatch.setattr(vllm_backend.platform, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        NPUIPCWeightRefitter,
        "_local_state_dict",
        staticmethod(fail_staging),
    )

    with pytest.raises(RuntimeError, match="NPU staging OOM"):
        NPUIPCWeightRefitter().refit(
            Client(),
            PolicySnapshot(1, "qwen", payload=object()),
        )


def test_training_state_release_failure_is_synchronized(monkeypatch) -> None:
    """One rank's reshard failure must be exchanged before allocator cleanup."""
    trainer = object.__new__(SyncTrainer)
    trainer.resolved_config = {"rollout": _colocated_rollout_config()}
    trainer.model = object()
    trainer.reference_model = None
    trainer.critic_model = None
    trainer.optimizer = None
    trainer.critic_optimizer = None

    def all_gather_object(outputs: list[Any], local_error: Any) -> None:
        """Expose one rank's release failure to all mocked ranks."""
        outputs[:] = [local_error, None]

    def fail_reshard(_model: Any) -> None:
        """Reject one rank's reshard operation."""
        raise RuntimeError("reshard failed")

    monkeypatch.setattr(vllm_backend.platform, "get_world_size", lambda: 2)
    monkeypatch.setattr(vllm_backend.platform, "all_gather_object", all_gather_object)
    monkeypatch.setattr(trainer_backend, "hsdp_sync_stream", lambda: None)
    monkeypatch.setattr(trainer, "_reshard_model", fail_reshard)

    with pytest.raises(RuntimeError, match="reshard failed"):
        trainer._release_training_state_for_rollout()  # pylint: disable=protected-access


def test_learning_gate_requires_reward_variance_gradient_and_fingerprint(monkeypatch) -> None:
    """M3 acceptance rejects a systems-only zero-learning smoke result."""
    trainer = object.__new__(SyncTrainer)
    trainer.resolved_config = {
        "train": {
            "learning_gate": {
                "enabled": True,
                "min_gradient_norm": 1.0e-8,
                "require_mixed_rewards": True,
                "require_fingerprint_change": True,
            }
        }
    }
    passing = {
        "train/gradient_norm": 0.25,
        "reward/min": 0.0,
        "reward/max": 1.0,
        "reward/zero_std_groups": 1.0,
        "policy/fingerprint_changed": 1.0,
        "policy/version": 1.0,
    }
    monkeypatch.setattr(trainer_backend.platform, "get_world_size", lambda: 1)
    monkeypatch.setattr(trainer_backend.platform, "get_rank", lambda: 0)

    trainer._enforce_learning_gate(passing, 1)  # pylint: disable=protected-access
    failing = dict(passing, **{"train/gradient_norm": 0.0})
    with pytest.raises(RuntimeError, match="gradient_norm"):
        trainer._enforce_learning_gate(failing, 1)  # pylint: disable=protected-access
    constant_rewards = dict(passing, **{"reward/min": 0.0, "reward/max": 0.0})
    with pytest.raises(RuntimeError, match="global reward variance"):
        trainer._enforce_learning_gate(constant_rewards, 1)  # pylint: disable=protected-access


def test_checkpoint_resume_preflight_rejects_missing_rank_artifact(monkeypatch, tmp_path) -> None:
    """A completion manifest cannot hide a missing rank-local resume artifact."""
    trainer = object.__new__(SyncTrainer)
    trainer.checkpoint_callback = types.SimpleNamespace(load_path=str(tmp_path))
    trainer.optimizer = object()
    trainer.lr_scheduler = None
    trainer.train_dataloader = types.SimpleNamespace(state_dict=lambda: {})
    (tmp_path / "checkpoint_complete.json").write_text(
        json.dumps({"step": 1, "world_size": 1}),
        encoding="utf-8",
    )
    for name in ("extra_state.json", "rng_rank0.pt", "optimizer_rank0.pt", "dataloader_rank0.pt"):
        (tmp_path / name).touch()

    monkeypatch.setattr(trainer_backend.platform, "get_world_size", lambda: 1)
    monkeypatch.setattr(trainer_backend.platform, "get_rank", lambda: 0)
    trainer._validate_checkpoint_for_resume()  # pylint: disable=protected-access

    (tmp_path / "rng_rank0.pt").unlink()
    with pytest.raises(RuntimeError, match="rng_rank0.pt"):
        trainer._validate_checkpoint_for_resume()  # pylint: disable=protected-access


def test_tracking_failure_does_not_skip_rollout_cleanup() -> None:
    """Monitoring shutdown errors must not leave vLLM descendants alive."""
    calls = []
    trainer = object.__new__(SyncTrainer)
    trainer._tracker = types.SimpleNamespace(
        finish=lambda: (_ for _ in ()).throw(RuntimeError("tracker failed"))
    )
    trainer.rollout_engine = types.SimpleNamespace(close=lambda: calls.append("close"))
    trainer._runtime_started = False

    trainer._cleanup_distributed()  # pylint: disable=protected-access

    assert calls == ["close"]
    assert trainer._tracker is None


def test_npu_ipc_refitter_updates_all_replicas_before_finish(monkeypatch) -> None:
    """The IPC transaction keeps KV asleep and finishes after the fan-out send."""
    calls = []

    class Client(vllm_backend._VLLMHTTPClient):  # pylint: disable=protected-access
        def __init__(self) -> None:
            """Bind the mocked rank-local endpoint."""
            self._base_url = "http://127.0.0.1:8100"

        @staticmethod
        def wake_up(tags: tuple[str, ...]) -> None:
            """Record weights-only wake."""
            calls.append(("wake", tags))

        @staticmethod
        def pause() -> None:
            """Record scheduler pause."""
            calls.append("pause")

        @staticmethod
        def start_weight_update() -> None:
            """Record transaction start."""
            calls.append("start")

        @staticmethod
        def receive_ipc_weights(base_url: str, update_info: Any) -> None:
            """Record fan-out destination and names."""
            calls.append(("send", base_url, tuple(update_info.names)))

        @staticmethod
        def finish_weight_update() -> None:
            """Record transaction finish."""
            calls.append("finish")

        @staticmethod
        def get_policy_weight_fingerprints(version: int) -> list[dict[str, Any]]:
            """Return the verified rank-local policy probe."""
            calls.append(("fingerprint", version))
            return [
                {
                    "algorithm": "qwen3_5_norms_f32_v1",
                    "tensor_count": 1,
                    "value_count": 1,
                    "digest": "digest",
                }
            ]

    @dataclass
    class UpdateInfo:
        """Minimal NPU IPC update payload."""

        names: list[str]
        dtype_names: list[str]
        shapes: list[list[int]]
        ipc_handles: list[dict[str, tuple]]
        tensor_sizes: Any = None
        packed: bool = False

    transfer_module = types.ModuleType(
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine"
    )
    transfer_module.NPUIPCWeightTransferUpdateInfo = UpdateInfo
    transfer_module.npu_generate_uuid = lambda: "host-npu0"
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        transfer_module,
    )
    monkeypatch.setattr(vllm_backend.platform, "get_world_size", lambda: 1)
    monkeypatch.setattr(vllm_backend.platform, "get_rank", lambda: 0)
    monkeypatch.setattr(
        vllm_backend.platform,
        "all_gather_object",
        lambda outputs, value: outputs.__setitem__(0, value),
    )
    monkeypatch.setattr(
        vllm_backend.platform,
        "get_current_stream",
        lambda: types.SimpleNamespace(synchronize=lambda: calls.append("synchronize")),
    )
    monkeypatch.setattr(
        vllm_backend.platform,
        "get_tensor_ipc_rebuild_args",
        lambda tensor: (tensor.shape,),
    )
    monkeypatch.setattr(
        NPUIPCWeightRefitter,
        "_device_state_dict",
        staticmethod(lambda state_dict: dict(state_dict)),
    )
    monkeypatch.setattr(
        NPUIPCWeightRefitter,
        "_local_state_dict",
        staticmethod(lambda _payload: {"weight": torch.ones(2, dtype=torch.bfloat16)}),
    )
    monkeypatch.setattr(
        vllm_backend.platform,
        "gather_state_dict",
        lambda state_dict, **_kwargs: dict(state_dict),
    )
    monkeypatch.setattr(
        vllm_backend,
        "_policy_weight_fingerprint",
        lambda _state_dict: {
            "algorithm": "qwen3_5_norms_f32_v1",
            "tensor_count": 1,
            "value_count": 1,
            "digest": "digest",
        },
    )

    NPUIPCWeightRefitter().refit(
        Client(),
        PolicySnapshot(1, "qwen", payload=object()),
    )

    assert calls[:3] == [("wake", ("weights",)), "pause", "start"]
    assert calls.index("synchronize") < calls.index(("send", "http://127.0.0.1:8100", ("weight",)))
    assert calls[-2:] == ["finish", ("fingerprint", 1)]
