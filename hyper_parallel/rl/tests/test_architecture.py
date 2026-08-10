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
import sys
import types
from typing import Any

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
from rl.roles.rollout.vllm import VLLMGenerationEngine


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


def test_vllm_adapter_returns_explicit_variable_length_mask(monkeypatch) -> None:
    """Verify that vLLM completions preserve their explicit valid-token masks."""
    class SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            """Capture vLLM sampling keyword arguments."""
            self.kwargs = kwargs

    fake_module = types.ModuleType("vllm")
    fake_module.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_module)

    def completion(tokens: list[int]) -> types.SimpleNamespace:
        """Build one fake vLLM completion record."""
        return types.SimpleNamespace(token_ids=tokens, logprobs=None)

    class Client:
        def generate(
            self,
            prompt_token_ids: list[list[int]],
            sampling_params: SamplingParams,
        ) -> list[Any]:
            """Return deterministic variable-length completions."""
            assert prompt_token_ids == [[1, 2], [3]]
            assert sampling_params.kwargs["max_tokens"] == 4
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
