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
"""Contract tests for the minimal extensible architecture."""

import asyncio
import json
import os
from pathlib import Path
import sys
import types
from typing import Any

import pytest
import torch

from rl.agentic import (
    ENVIRONMENTS,
    Action,
    AgentRunner,
    Observation,
    ProgramAgentRunner,
    Transition,
)
from rl.algorithm import build_algorithm
from rl.async_trainer import AsyncTrainer
from rl.contracts import Message, PromptRecord, Trajectory, Turn
from rl.roles.model import ModelRegistration
from rl.roles.rollout import (
    ROLLOUT_ENGINES,
    GenerationRequest,
    GenerationResult,
    GenerationSettings,
    PolicySnapshot,
    build_rollout_engine,
)
from rl.roles.rollout.vllm import (
    CPUStateDictRefitter,
    HCCLWeightRefitter,
    VLLMGenerationEngine,
)
from rl.roles.rollout import vllm as vllm_backend


def test_async_trainer_fails_fast_until_ray_runtime_exists() -> None:
    try:
        AsyncTrainer()
    except NotImplementedError as error:
        assert "Ray" in str(error)
        assert "SyncTrainer" in str(error)
    else:
        raise AssertionError("AsyncTrainer must not expose an empty success path")


def test_grpo_requirements_create_reference_but_not_critic() -> None:
    algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
    assert algorithm.requirements.roles.reference is True
    assert algorithm.requirements.roles.critic is False
    assert algorithm.requirements.data.grouped_responses is True


def test_hyper_and_vllm_engines_are_registered_and_vllm_is_lazy() -> None:
    assert ROLLOUT_ENGINES.names == ("hyper", "vllm")
    model = ModelRegistration("qwen", "qwen3_5", "/model", "/model")
    engine = build_rollout_engine({"engine": "vllm", "vllm": {}}, model)
    assert engine.name == "vllm"
    assert engine.client_initialized is False


@pytest.mark.parametrize(
    ("model_implementation", "expected_architecture"),
    (
        ("hyper", "HyperQwen3_5ForCausalLM"),
        ("native", "Qwen3_5ForConditionalGeneration"),
    ),
)
def test_vllm_server_selects_qwen_architecture(
    monkeypatch,
    model_implementation: str,
    expected_architecture: str,
) -> None:
    """The unified server path must select the requested validated model."""
    process_calls = []
    client_calls = []

    class Process:
        """Capture process construction without launching vLLM."""

    def popen(command: list[str], **kwargs: Any) -> Process:
        process_calls.append((command, kwargs))
        return Process()

    class HTTPClient:
        """Capture readiness configuration for the external server."""

        def __init__(self, process: Process, base_url: str, model_name: str, request_timeout: float) -> None:
            client_calls.append((process, base_url, model_name, request_timeout))

        @staticmethod
        def wait_ready(startup_timeout: float) -> None:
            client_calls.append(("wait_ready", startup_timeout))

        @staticmethod
        def close() -> None:
            raise AssertionError("A successful server startup must not close the client")

    monkeypatch.setattr(vllm_backend.subprocess, "Popen", popen)
    monkeypatch.setattr(vllm_backend, "_VLLMHTTPClient", HTTPClient)
    monkeypatch.setattr(vllm_backend, "_open_port", lambda: 8123)
    model = ModelRegistration("qwen", "qwen3_5", "/model", "/tokenizer")
    engine = VLLMGenerationEngine(
        model,
        {
            "vllm": {
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.4,
                "visible_devices": "4,5",
                "model_implementation": model_implementation,
                "batch_invariant": True,
                "skip_mm_profiling": True,
            }
        },
    )

    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("RANK", "0")
    engine._ensure_client()  # pylint: disable=protected-access

    command, process_kwargs = process_calls[0]
    assert command[:5] == [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", "/model"]
    assert command[command.index("--tokenizer") + 1] == "/tokenizer"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.4"
    assert "--skip-mm-profiling" in command
    assert command[command.index("--hf-overrides") + 1] == json.dumps(
        {"architectures": [expected_architecture]}
    )
    assert process_kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"
    assert process_kwargs["env"]["VLLM_BATCH_INVARIANT"] == "1"
    assert "RANK" not in process_kwargs["env"]
    assert process_kwargs["shell"] is False
    assert process_kwargs["start_new_session"] is True
    assert client_calls[0][1:] == ("http://127.0.0.1:8123", "qwen", 600.0)
    assert client_calls[1] == ("wait_ready", 300.0)
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "0,1"
    assert os.environ["RANK"] == "0"


def test_vllm_adapter_returns_explicit_variable_length_mask(monkeypatch) -> None:
    """Verify that vLLM completions preserve their explicit valid-token masks."""
    class SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            """Capture vLLM sampling keyword arguments."""
            self.kwargs = kwargs

    class TokensPrompt:
        def __init__(self, prompt_token_ids: list[int]) -> None:
            """Capture one pre-tokenized vLLM prompt."""
            self.prompt_token_ids = prompt_token_ids

    fake_module = types.ModuleType("vllm")
    fake_module.SamplingParams = SamplingParams
    fake_module.TokensPrompt = TokensPrompt
    monkeypatch.setitem(sys.modules, "vllm", fake_module)

    def completion(tokens: list[int]) -> types.SimpleNamespace:
        """Build one fake vLLM completion record."""
        return types.SimpleNamespace(token_ids=tokens, logprobs=None)

    class Client:
        def generate(
            self,
            prompts: list[TokensPrompt],
            sampling_params: SamplingParams,
            use_tqdm: bool,
        ) -> list[Any]:
            """Return deterministic variable-length completions."""
            assert [prompt.prompt_token_ids for prompt in prompts] == [[1, 2], [3]]
            assert sampling_params.kwargs["max_tokens"] == 4
            assert use_tqdm is False
            return [
                types.SimpleNamespace(outputs=[completion([7, 8])]),
                types.SimpleNamespace(outputs=[completion([9])]),
            ]

    model = ModelRegistration("qwen", "qwen3_5", "/model", "/model")
    engine = VLLMGenerationEngine(model, {"vllm": {}}, client=Client())
    settings = GenerationSettings(4, 1.0, 1.0, 0, True, 0, 2)
    result = engine.generate(
        GenerationRequest(
            torch.tensor([[1, 2], [0, 3]]),
            torch.tensor([[1, 1], [0, 1]]),
            settings,
        )
    )
    assert result.response_mask.tolist() == [
        [True, True, False, False],
        [True, False, False, False],
    ]


def test_vllm_advances_version_only_after_a_real_refit() -> None:
    """Advance the acknowledged policy version only after a successful refit."""
    calls = []

    class Refitter:
        def refit(self, client: Any, snapshot: PolicySnapshot) -> None:
            """Record the concrete client and loaded policy version."""
            calls.append((client, snapshot.version))

    client = object()
    model = ModelRegistration("qwen", "qwen3_5", "/model", "/model")
    engine = VLLMGenerationEngine(
        model, {"vllm": {}}, client=client, refitter=Refitter()
    )
    engine.update_weights(PolicySnapshot(1, "qwen", payload={"weight": 1}))
    assert calls == [(client, 1)]
    assert engine.policy_version == 1


def test_vllm_cpu_refitter_reloads_workers_and_resets_cache() -> None:
    """The initial online path must publish complete CPU tensors before acknowledgement."""
    calls = []

    class Client:
        """Record the synchronous vLLM weight reload contract."""

        @staticmethod
        def collective_rpc(method: str, kwargs: dict[str, Any]) -> list[None]:
            checkpoint = Path(kwargs["weights_path"]) / "model.safetensors"
            calls.append((method, kwargs, checkpoint.is_file()))
            return [None]

        @staticmethod
        def reset_prefix_cache(**kwargs: bool) -> bool:
            calls.append(("reset_prefix_cache", kwargs))
            return True

    source = torch.tensor([1.0], requires_grad=True)
    snapshot = PolicySnapshot(1, "qwen", payload={"weight": source})

    CPUStateDictRefitter().refit(Client(), snapshot)

    method, kwargs, checkpoint_existed = calls[0]
    assert method == "reload_weights"
    assert kwargs["is_checkpoint_format"] is True
    assert checkpoint_existed is True
    assert "weights_iterator" not in kwargs
    assert calls[1] == (
        "reset_prefix_cache",
        {"reset_running_requests": True, "reset_connector": False},
    )


def test_vllm_cpu_refitter_gathers_the_unwrapped_actor(monkeypatch) -> None:
    """FSDP publication must gather the native module rather than the generation facade."""
    module = object()
    actor = types.SimpleNamespace(module=module)
    calls = []

    def get_model_state_dict(model: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((model, kwargs))
        return {"weight": torch.tensor([2.0])}

    monkeypatch.setattr(vllm_backend.platform, "get_model_state_dict", get_model_state_dict)

    state_dict = CPUStateDictRefitter._cpu_state_dict(actor)  # pylint: disable=protected-access

    assert state_dict["weight"].device.type == "cpu"
    assert calls[0][0] is module
    assert calls[0][1] == {"full_state_dict": True, "cpu_offload": False}


def test_vllm_hccl_refitter_synchronizes_optimizer_stream(monkeypatch) -> None:
    """Packed HCCL streams must not read parameters before optimizer writes finish."""
    calls = []

    class Tensor:
        dtype = torch.bfloat16
        shape = (1,)

        @staticmethod
        def numel() -> int:
            return 1

        @staticmethod
        def element_size() -> int:
            return 2

    class Client(vllm_backend._VLLMHTTPClient):  # pylint: disable=protected-access
        def __init__(self) -> None:
            pass

        @staticmethod
        def pause() -> None:
            calls.append("pause")

        @staticmethod
        def start_weight_update() -> None:
            calls.append("start")

        @staticmethod
        def receive_weights(_update_info: dict[str, Any]) -> None:
            calls.append("receive")

        @staticmethod
        def finish_weight_update() -> None:
            calls.append("finish")

        @staticmethod
        def get_policy_weight_fingerprints(version: int) -> list[dict[str, Any]]:
            calls.append(("fingerprint", version))
            return [
                {
                    "algorithm": "qwen3_5_norms_f32_v1",
                    "tensor_count": 1,
                    "value_count": 1,
                    "digest": "digest",
                }
            ]

        @staticmethod
        def resume() -> None:
            calls.append("resume")

    class TrainerArgs:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class TransferEngine:
        @staticmethod
        def trainer_send_weights(iterator: Any, trainer_args: TrainerArgs) -> None:
            list(iterator)
            assert trainer_args.kwargs["packed"] is True
            calls.append("send")

    transfer_module = types.ModuleType(
        "vllm_ascend.distributed.weight_transfer.hccl_engine"
    )
    transfer_module.HCCLTrainerSendWeightsArgs = TrainerArgs
    transfer_module.HCCLWeightTransferEngine = TransferEngine
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        transfer_module,
    )
    monkeypatch.setattr(
        vllm_backend.platform,
        "get_current_stream",
        lambda: types.SimpleNamespace(synchronize=lambda: calls.append("synchronize")),
    )
    monkeypatch.setattr(
        HCCLWeightRefitter,
        "_device_state_dict",
        staticmethod(lambda _payload: {"weight": Tensor()}),
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
    refitter = HCCLWeightRefitter()
    refitter._group = object()  # pylint: disable=protected-access

    refitter.refit(Client(), PolicySnapshot(1, "qwen", payload=object()))

    assert calls.index("synchronize") < calls.index("send")
    assert calls[-3:] == ["finish", ("fingerprint", 1), "resume"]


def test_vllm_does_not_acknowledge_refit_when_cache_reset_fails() -> None:
    """A stale prefix cache must prevent committing the next policy version."""
    class Client:
        """Accept weights but reject cache invalidation."""

        @staticmethod
        def collective_rpc(method: str, kwargs: dict[str, Any]) -> list[None]:
            del method, kwargs
            return [None]

        @staticmethod
        def reset_prefix_cache(**kwargs: bool) -> bool:
            del kwargs
            return False

    model = ModelRegistration("qwen", "qwen3_5", "/model", "/model")
    engine = VLLMGenerationEngine(
        model,
        {"vllm": {}},
        client=Client(),
        refitter=CPUStateDictRefitter(),
    )

    try:
        engine.update_weights(
            PolicySnapshot(1, "qwen", payload={"weight": torch.tensor([1.0])})
        )
    except RuntimeError as error:
        assert "prefix cache" in str(error)
    else:
        raise AssertionError("A failed cache reset must reject the policy update")
    assert engine.policy_version == 0


def test_vllm_refit_stops_before_cache_reset_when_reload_fails() -> None:
    """A worker reload error must keep the old version and skip cache mutation."""
    calls = []

    class Client:
        """Reject weight loading before cache reset."""

        @staticmethod
        def collective_rpc(method: str, kwargs: dict[str, Any]) -> list[None]:
            del method, kwargs
            calls.append("reload")
            raise RuntimeError("worker load failed")

        @staticmethod
        def reset_prefix_cache(**kwargs: bool) -> bool:
            del kwargs
            calls.append("reset")
            return True

    model = ModelRegistration("qwen", "qwen3_5", "/model", "/model")
    engine = VLLMGenerationEngine(
        model,
        {"vllm": {}},
        client=Client(),
        refitter=CPUStateDictRefitter(),
    )

    try:
        engine.update_weights(
            PolicySnapshot(1, "qwen", payload={"weight": torch.tensor([1.0])})
        )
    except RuntimeError as error:
        assert "worker load failed" in str(error)
    else:
        raise AssertionError("A failed worker reload must reject the policy update")
    assert calls == ["reload"]
    assert engine.policy_version == 0


def test_vllm_refit_propagates_remote_rank_failure(monkeypatch) -> None:
    """A successful local reload must not commit when another training rank failed."""
    monkeypatch.setattr(vllm_backend.platform, "get_world_size", lambda: 2)

    def all_gather_object(outputs: list[Any], local_error: Any) -> None:
        outputs[:] = [local_error, "rank 1 failed"]

    monkeypatch.setattr(vllm_backend.platform, "all_gather_object", all_gather_object)

    try:
        CPUStateDictRefitter._synchronize_error(  # pylint: disable=protected-access
            None,
            "weight reload",
        )
    except RuntimeError as error:
        assert "rank 1 failed" in str(error)
    else:
        raise AssertionError("A remote refit error must fail the local rank")


def test_vllm_refuses_to_acknowledge_unloaded_policy_version() -> None:
    model = ModelRegistration("qwen", "qwen3_5", "/model", "/model")
    engine = VLLMGenerationEngine(model, {"vllm": {}}, client=object())
    try:
        engine.update_weights(PolicySnapshot(1, "qwen", payload={}))
    except NotImplementedError as error:
        assert "Refitter" in str(error)
    else:
        raise AssertionError("vLLM must not acknowledge weights it did not load")


def test_vllm_rejects_snapshot_from_another_model_registration() -> None:
    model = ModelRegistration("qwen", "qwen3_5", "/model", "/model")
    engine = VLLMGenerationEngine(model, {"vllm": {}}, client=object())
    try:
        engine.update_weights(PolicySnapshot(1, "other-model", payload={}))
    except ValueError as error:
        assert "model mismatch" in str(error)
    else:
        raise AssertionError("vLLM must reject another model's snapshot")


_CREATED_ENVIRONMENTS = []


class _TwoTurnEnvironment:
    """Deterministic two-step environment used by agent runner tests."""

    def __init__(self, prompt: PromptRecord) -> None:
        """Initialize deterministic episode state."""
        self.prompt = prompt
        self.closed = False
        self.step_index = 0
        self.event_loop = None
        _CREATED_ENVIRONMENTS.append(self)

    async def reset(self, prompt: PromptRecord) -> Observation:
        """Return the initial question observation."""
        assert prompt is self.prompt
        self.event_loop = asyncio.get_running_loop()
        return Observation(
            "question",
            torch.tensor([10, 11]),
            metadata={"role": "user"},
        )

    async def step(self, action: Action) -> Transition:
        """Return a tool observation followed by a terminal observation."""
        assert asyncio.get_running_loop() is self.event_loop
        self.step_index += 1
        assert action.content == f"answer-{self.step_index}"
        if self.step_index == 1:
            return Transition(
                Observation("tool result", torch.tensor([30]), metadata={"role": "tool"}),
                0.25,
                done=False,
                info={"reward_components": {"progress": 0.25}},
            )
        return Transition(
            Observation("final", torch.tensor([40]), metadata={"role": "environment"}),
            0.75,
            done=True,
            info={"reward_components": {"success": 0.75}},
        )

    async def close(self) -> None:
        """Record closure on the same environment event loop."""
        assert asyncio.get_running_loop() is self.event_loop
        self.closed = True


@ENVIRONMENTS.register("two_turn_test")
def _build_two_turn_environment(prompt: PromptRecord) -> _TwoTurnEnvironment:
    return _TwoTurnEnvironment(prompt)


class _TwoTurnEngine:
    """Generate deterministic action tokens for two turns."""

    name = "test"

    def __init__(self) -> None:
        """Initialize the generation call counter."""
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Append deterministic answer and EOS tokens."""
        self.calls += 1
        token = 19 + self.calls
        batch_size = request.input_ids.shape[0]
        responses = request.input_ids.new_tensor([[token, 2]]).repeat(batch_size, 1)
        log_probs = torch.tensor([[-0.1 * self.calls, -0.9]]).repeat(batch_size, 1)
        return GenerationResult(
            sequences=torch.cat((request.input_ids, responses), dim=-1),
            rollout_log_probs=log_probs,
            generation_seconds=0.01,
        )

    def update_weights(self, weight_handle: Any) -> None:
        """Accept the no-op test weight handle."""
        del weight_handle


class _TwoTurnTokenizer:
    @staticmethod
    def decode(token_ids: list[int], skip_special_tokens: bool) -> str:
        """Decode deterministic answer tokens into turn labels."""
        assert skip_special_tokens is True
        return f"answer-{token_ids[0] - 19}"


def test_agent_runner_emits_token_aligned_multi_turn_experience() -> None:
    """Keep multi-turn observations and actions token-aligned in experience."""
    _CREATED_ENVIRONMENTS.clear()
    prompt = PromptRecord("p0", (Message("user", "question"),), ground_truth="answer")
    engine = _TwoTurnEngine()
    runner = AgentRunner(
        engine=engine,
        tokenizer=_TwoTurnTokenizer(),
        environment_name="two_turn_test",
        num_samples=1,
        max_turns=2,
        max_observation_tokens=1,
        settings=GenerationSettings(2, 1.0, 1.0, 0, True, 0, 2, True),
    )

    experience = runner.rollout((prompt,), policy_version=7)
    trajectory = experience.trajectories[0]
    assert trajectory.token_ids.tolist() == [10, 11, 20, 30, 21, 40]
    assert trajectory.action_mask.tolist() == [False, False, True, False, True, False]
    assert torch.allclose(
        trajectory.rollout_log_probs,
        torch.tensor([0.0, -0.1, 0.0, -0.2, 0.0]),
    )
    assert trajectory.reward == 1.0
    assert trajectory.reward_components == {"progress": 0.25, "success": 0.75}
    assert trajectory.done is True
    assert trajectory.policy_version == 7
    assert [turn.role for turn in trajectory.turns] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "environment",
    ]
    assert experience.loss_action_mask.tolist() == [[False, True, False, True, False]]
    assert torch.allclose(
        experience.old_log_probs,
        torch.tensor([[0.0, -0.1, 0.0, -0.2, 0.0]]),
    )
    assert engine.calls == 2
    assert _CREATED_ENVIRONMENTS[0].closed is True


def test_trajectory_rejects_misaligned_action_mask() -> None:
    """Reject a trajectory whose action mask does not align with its tokens."""
    try:
        Trajectory(
            trajectory_id="t",
            prompt_id="p",
            group_id=None,
            policy_version=0,
            turns=(),
            token_ids=torch.tensor([1, 2]),
            attention_mask=torch.tensor([1, 1]),
            action_mask=torch.tensor([1]),
            rollout_log_probs=None,
            reward=0.0,
            reward_components={},
            done=True,
            truncated=False,
            terminal_reason="test",
        )
    except ValueError as error:
        assert "action_mask" in str(error)
    else:
        raise AssertionError("Expected token alignment validation to fail")


def test_program_agent_runner_leaves_episode_control_flow_to_user_code() -> None:
    """Let user programs own episode control while enforcing output contracts."""
    prompt = PromptRecord("p0", (Message("user", "question"),))

    class Program:
        """Return a complete user-owned trajectory."""

        def __init__(self, policy_version: int, sample_index: int) -> None:
            """Store the requested policy version and sample index."""
            self.policy_version = policy_version
            self.sample_index = sample_index

        async def run(self) -> Trajectory:
            """Build one deterministic program-owned trajectory."""
            await asyncio.sleep(0)
            tokens = torch.tensor([10, 20])
            return Trajectory(
                trajectory_id=f"p0:{self.policy_version}:{self.sample_index}",
                prompt_id="p0",
                group_id="p0",
                policy_version=self.policy_version,
                turns=(
                    Turn("user", "question", 0, 1, False),
                    Turn("assistant", "answer", 1, 2, True),
                ),
                token_ids=tokens,
                attention_mask=torch.ones_like(tokens, dtype=torch.bool),
                action_mask=torch.tensor([False, True]),
                rollout_log_probs=torch.tensor([-0.25]),
                reward=1.0,
                reward_components={"user": 1.0},
                done=True,
                truncated=False,
                terminal_reason="done",
            )

    runner = ProgramAgentRunner(
        program_factory=lambda _prompt, version, sample: Program(version, sample),
        num_samples=2,
        settings=GenerationSettings(1, 1.0, 1.0, 0, True, 0, 2, True),
    )
    experience = runner.rollout((prompt,), policy_version=3)
    assert len(experience.trajectories) == 2
    assert experience.metadata["runner"] == "program"
    assert experience.old_log_probs.tolist() == [[-0.25], [-0.25]]
