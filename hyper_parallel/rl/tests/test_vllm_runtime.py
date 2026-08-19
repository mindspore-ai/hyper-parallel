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

from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch
from rl.roles.model import (
    ModelRegistration,
    architecture_for_implementation,
    normalize_model_implementation,
    resolve_vllm_model,
)
from rl.roles.rollout.vllm import VLLMGenerationEngine
from rl.roles.weight_sync.transfer import (
    CPUStateDictRefitter,
    map_actor_state_dict,
    map_policy_state_dict,
)
from rl.roles.weight_sync.sync import (
    ActorRolloutWeightSync,
    PolicySnapshot,
    VLLMWeightSyncClientMixin,
)


def _model(
    hyper_model_name: str = "qwen3_5",
    *,
    tie_word_embeddings: bool = True,
) -> ModelRegistration:
    if hyper_model_name == "qwen3":
        architecture = "Qwen3ForCausalLM"
        model_type = "qwen3"
        text_model_type = "qwen3"
    else:
        architecture = "Qwen3_5ForConditionalGeneration"
        model_type = "qwen3_5"
        text_model_type = "qwen3_5_text"
    return ModelRegistration(
        "qwen",
        hyper_model_name,
        "/model",
        "/tokenizer",
        architecture,
        model_type,
        text_model_type,
        tie_word_embeddings,
    )


def test_model_implementation_defaults_to_native() -> None:
    """An omitted rollout implementation must select upstream vLLM."""
    assert normalize_model_implementation(None) == "native"


def test_legacy_refitter_and_policy_mapping_keep_their_signatures() -> None:
    """Retained refitter names translate old arguments to the unified model contract."""
    refitter = CPUStateDictRefitter("native")
    state_dict = {
        "model.norm.weight": object(),
        "lm_head.weight": object(),
    }

    assert refitter._model.implementation == "native"  # pylint: disable=protected-access
    assert list(map_policy_state_dict(state_dict, "native")) == [
        "model.language_model.norm.weight"
    ]


def test_native_architecture_and_refit_names_follow_model_family() -> None:
    """Qwen3 uses causal names directly while Qwen3.5 targets its vLLM wrapper."""
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
    assert list(
        map_actor_state_dict(qwen3_state, resolve_vllm_model(_model(), "native"))
    ) == ["model.language_model.layers.0.self_attn.q_proj.weight"]
    untied_rollout_model = resolve_vllm_model(
        _model(tie_word_embeddings=False),
        "native",
    )
    assert list(map_actor_state_dict(qwen3_state, untied_rollout_model)) == [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    ]
    text_model = ModelRegistration(
        "qwen",
        "qwen3_5",
        "/model",
        "/tokenizer",
        "Qwen3_5ForCausalLM",
        "qwen3_5_text",
        "qwen3_5_text",
        False,
    )
    text_rollout_model = resolve_vllm_model(text_model, "native")
    assert list(map_actor_state_dict(qwen3_state, text_rollout_model)) == [
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    ]


def test_native_qwen3_can_launch_lazy_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The historical Qwen3.5-only guard must not reject native Qwen3."""
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
    monkeypatch.setattr(
        engine,
        "_launch_client",
        lambda _visible_devices, _host, _port: client,
    )

    assert engine._ensure_client() is client  # pylint: disable=protected-access


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


def test_server_command_preserves_explicit_prefill_and_logprob_semantics() -> None:
    """False cache flags and raw logprobs must reach the owned vLLM server."""
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {
            "vllm": {
                "model_implementation": "hyper",
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
                "attention_backend": "FLASH_ATTN",
                "logprobs_mode": "raw_logprobs",
            }
        },
        client=object(),
    )

    command = engine._server_command("127.0.0.1", 8100)  # pylint: disable=protected-access

    assert "--no-enable-prefix-caching" in command
    assert "--no-enable-chunked-prefill" in command
    assert command[command.index("--attention-backend") + 1] == "FLASH_ATTN"
    assert command[command.index("--logprobs-mode") + 1] == "raw_logprobs"


def test_server_environment_carries_consistency_profile() -> None:
    """The isolated rollout process receives the paired profile identity."""
    engine = VLLMGenerationEngine(
        _model("qwen3"),
        {
            "vllm": {
                "batch_invariant": True,
                "consistency_profile": "qwen3_ascend_fa3_batch_invariant_v1",
                "model_implementation": "hyper",
            }
        },
        client=object(),
    )

    environment = engine._server_environment("0")  # pylint: disable=protected-access

    assert environment["VLLM_BATCH_INVARIANT"] == "1"
    assert environment["HYPER_RL_CONSISTENCY_PROFILE"] == "qwen3_ascend_fa3_batch_invariant_v1"


def test_server_environment_removes_inherited_consistency_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An off-profile rollout must not inherit a parent process patch selection."""
    monkeypatch.setenv(
        "HYPER_RL_CONSISTENCY_PROFILE",
        "qwen3_ascend_fa3_batch_invariant_v1",
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
    engine._completion_records = lambda _client, _prompts, _settings: [([2], None)]
    request = SimpleNamespace(
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones((1, 1), dtype=torch.bool),
        settings=SimpleNamespace(
            max_new_tokens=1,
            pad_token_id=0,
            collect_log_probs=False,
        ),
    )

    with pytest.raises(RuntimeError, match="changed while serving"):
        engine._generate(request)  # pylint: disable=protected-access


class _LifecycleClient(VLLMWeightSyncClientMixin):
    """In-memory control client for the colocated publication transaction."""

    def __init__(self, fail_post_refit_pause: bool = False) -> None:
        """Initialize fake worker identity, residency, and call history."""
        self.version = 0
        self.digest = "digest-v0"
        self.paused = False
        self.sleeping = False
        self.fail_post_refit_pause = fail_post_refit_pause
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


class _LifecycleTransfer:
    """Simulate one successful transfer and expose its verified fingerprint."""

    last_policy_fingerprint = None

    def transfer(
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
    pending_identity_index = client.calls.index("identity", reset_index)
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
