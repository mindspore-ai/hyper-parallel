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
"""Regression tests for token-first agent rollout masking."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

import rl.trainer as trainer_backend
from rl.agentic.base import Observation, Transition
from rl.agentic.registry import ENVIRONMENTS
from rl.agentic.runner import AgentRunner, _canonical_row_seed
from rl.dataset.contracts import Message, PromptRecord
from rl.roles.rollout.base import GenerationResult, GenerationSettings
from rl.roles.rollout.vllm import _VLLMHTTPClient, VLLMGenerationEngine
from rl.roles.rollout.worker import RolloutManager


def _generation_settings() -> GenerationSettings:
    """Build Qwen3 settings with both official terminal token IDs."""
    return GenerationSettings(
        max_new_tokens=3,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        do_sample=False,
        pad_token_id=151643,
        eos_token_id=151645,
        eos_token_ids=(151643, 151645),
    )


def _install_fake_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the two optional vLLM value types used by in-process contracts."""
    module = ModuleType("vllm")

    class SamplingParams:
        """Record sampling keyword arguments like the real vLLM value object."""

        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class TokensPrompt:
        """Record prompt token IDs like the real vLLM value object."""

        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    module.SamplingParams = SamplingParams
    module.TokensPrompt = TokensPrompt
    monkeypatch.setitem(sys.modules, "vllm", module)


def test_generation_settings_normalize_all_qwen3_eos_ids() -> None:
    """The primary and additional Qwen3 EOS IDs must form one unique contract."""
    settings = _generation_settings()

    assert settings.eos_token_ids == (151645, 151643), (
        f"Qwen3 EOS IDs were not normalized: actual={settings.eos_token_ids}, "
        f"expected={(151645, 151643)}"
    )


def test_response_mask_includes_either_qwen3_eos_and_excludes_following_tokens() -> None:
    """Both Qwen3 terminal tokens remain trainable while later tokens are masked."""
    settings = _generation_settings()
    runner = AgentRunner(
        engine=object(),
        tokenizer=object(),
        environment_name="gsm8k",
        num_samples=1,
        max_turns=1,
        max_observation_tokens=0,
        settings=settings,
    )
    response_ids = torch.tensor(
        [
            [42, 151645, 99],
            [42, 151643, 99],
            [151645, 99, 100],
        ]
    )
    explicit_mask = torch.ones_like(response_ids, dtype=torch.bool)
    expected = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [True, False, False],
        ]
    )

    actual = runner._response_mask(  # pylint: disable=protected-access
        response_ids,
        explicit_mask,
    )

    assert torch.equal(actual, expected), (
        f"Qwen3 EOS response mask mismatch: actual={actual.tolist()}, "
        f"expected={expected.tolist()}"
    )


def test_response_mask_keeps_engine_valid_tokens_when_eos_is_ignored() -> None:
    """Fixed-length generation does not truncate valid tokens after an observed EOS."""
    settings = _generation_settings()
    object.__setattr__(settings, "ignore_eos", True)
    runner = AgentRunner(
        engine=object(),
        tokenizer=object(),
        environment_name="gsm8k",
        num_samples=1,
        max_turns=1,
        max_observation_tokens=0,
        settings=settings,
    )
    response_ids = torch.tensor([[42, 151645, 99]])
    explicit_mask = torch.tensor([[True, True, True]])

    assert torch.equal(
        runner._response_mask(response_ids, explicit_mask),  # pylint: disable=protected-access
        explicit_mask,
    )


def test_trainer_uses_generation_config_eos_ids_with_tokenizer_fallback() -> None:
    """Rollout setup must preserve every EOS declared by the Qwen3 model."""
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[151645, 151643])
    )
    tokenizer = SimpleNamespace(eos_token_id=151645)

    actual = trainer_backend._resolve_eos_token_ids(model, tokenizer)  # pylint: disable=protected-access

    assert actual == (151645, 151643), (
        f"Trainer resolved the wrong Qwen3 EOS IDs: actual={actual}, "
        f"expected={(151645, 151643)}"
    )


def test_vllm_completion_receives_all_qwen3_stop_token_ids() -> None:
    """The inference request must stop on either Qwen3 EOS ID used by masking."""
    client = _VLLMHTTPClient(
        process=object(),
        base_url="http://127.0.0.1:8000",
        model_name="qwen3",
        request_timeout=1.0,
    )
    captured_payload = client._completion_payload(  # pylint: disable=protected-access
        prompts=[[42]],
        seed=None,
        settings=SimpleNamespace(
            max_new_tokens=3,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            do_sample=False,
            collect_log_probs=False,
            ignore_eos=False,
            eos_token_ids=(151645, 151643),
        ),
    )

    assert captured_payload.get("stop_token_ids") == [151645, 151643], (
        f"vLLM received the wrong stop token IDs: actual={captured_payload}, "
        f"expected={[151645, 151643]}"
    )


def test_vllm_completion_removes_explicit_stops_when_eos_is_ignored() -> None:
    """Fixed-length generation cannot retain EOS as an explicit stop token."""
    client = _VLLMHTTPClient(
        process=None,
        base_url="http://127.0.0.1:8000",
        model_name="qwen3",
        request_timeout=1.0,
    )
    payload = client._completion_payload(  # pylint: disable=protected-access
        prompts=[[42]],
        seed=7,
        settings=SimpleNamespace(
            max_new_tokens=3,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            do_sample=True,
            collect_log_probs=True,
            ignore_eos=True,
            eos_token_ids=(151645, 151643),
        ),
    )

    assert payload["ignore_eos"] is True
    assert payload["stop_token_ids"] == []


def test_inprocess_vllm_receives_all_qwen3_stop_token_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The injected vLLM client must stop on every Qwen3 EOS ID as well."""
    _install_fake_vllm(monkeypatch)
    captured_sampling = {}

    class _Client:
        @staticmethod
        def generate(
            _prompts: Any,
            sampling_params: Any,
            use_tqdm: bool,
        ) -> list[Any]:
            """Return one synthetic completion and capture its sampling settings."""
            captured_sampling["params"] = sampling_params
            captured_sampling["use_tqdm"] = use_tqdm
            completion = SimpleNamespace(token_ids=[151643], logprobs=None)
            return [SimpleNamespace(outputs=[completion])]

    engine = object.__new__(VLLMGenerationEngine)
    records = engine._inprocess_completions(  # pylint: disable=protected-access
        _Client(),
        [[42]],
        SimpleNamespace(
            max_new_tokens=3,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            do_sample=False,
            collect_log_probs=False,
            seed=None,
            ignore_eos=False,
            eos_token_ids=(151645, 151643),
        ),
    )

    actual_stop_ids = captured_sampling["params"].stop_token_ids
    assert actual_stop_ids == [151645, 151643], (
        f"In-process vLLM received the wrong stop IDs: actual={actual_stop_ids}, "
        f"expected={[151645, 151643]}"
    )
    assert records == [([151643], None)], (
        f"In-process vLLM returned unexpected records: actual={records}, "
        f"expected={[([151643], None)]}"
    )


def test_inprocess_vllm_receives_explicit_seed_per_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-process diagnostic path preserves canonical row seeds as well."""
    _install_fake_vllm(monkeypatch)
    captured_sampling = {}

    class _Client:
        @staticmethod
        def generate(
            _prompts: Any,
            sampling_params: Any,
            use_tqdm: bool,
        ) -> list[Any]:
            """Capture per-prompt settings and return one completion per prompt."""
            captured_sampling["params"] = sampling_params
            captured_sampling["use_tqdm"] = use_tqdm
            return [
                SimpleNamespace(outputs=[SimpleNamespace(token_ids=[seed], logprobs=None)])
                for seed in (100, 200)
            ]

    engine = object.__new__(VLLMGenerationEngine)
    records = engine._inprocess_completions(  # pylint: disable=protected-access
        _Client(),
        [[1], [2]],
        SimpleNamespace(
            max_new_tokens=1,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            do_sample=True,
            collect_log_probs=False,
            seed=10,
            ignore_eos=False,
            eos_token_ids=(2,),
        ),
        row_seeds=(100, 200),
    )

    assert [params.seed for params in captured_sampling["params"]] == [100, 200]
    assert records == [([100], None), ([200], None)]


def test_rollout_manager_preserves_existing_positional_arguments() -> None:
    """Adding multiple EOS IDs must not shift the existing optional arguments."""
    manager = RolloutManager(
        object(),
        object(),
        "gsm8k",
        1,
        1,
        0,
        3,
        0.0,
        1.0,
        0,
        151643,
        151645,
        False,
        True,
        7,
    )
    settings = manager.agent_runner.settings

    assert settings.do_sample is False, (
        f"RolloutManager shifted do_sample: actual={settings.do_sample}, expected={False}"
    )
    assert settings.collect_log_probs is True, (
        "RolloutManager shifted collect_old_log_probs: "
        f"actual={settings.collect_log_probs}, expected={True}"
    )
    assert settings.seed == 7, (
        f"RolloutManager shifted seed: actual={settings.seed}, expected={7}"
    )
    assert settings.ignore_eos is False


def test_canonical_row_seeds_are_independent_of_dp_partitioning() -> None:
    """The same global prompt and response identities keep their seeds at DP2 and DP4."""
    base_seed = 20260814
    samples_per_prompt = 4

    def partition_seeds(partitions: list[list[str]]) -> dict[tuple[str, int], int]:
        return {
            (prompt_id, sample_index): _canonical_row_seed(
                base_seed,
                prompt_id,
                sample_index,
                samples_per_prompt,
            )
            for partition in partitions
            for prompt_id in partition
            for sample_index in range(samples_per_prompt)
        }

    dp2 = partition_seeds([["0", "1"], ["2", "3"]])
    dp4 = partition_seeds([["0"], ["1"], ["2"], ["3"]])

    assert dp2 == dp4
    assert [dp4[("2", index)] for index in range(samples_per_prompt)] == [
        20260822,
        20260823,
        20260824,
        20260825,
    ]


def test_agent_runner_preserves_two_turn_eos_mask_and_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two real turns retain each generated EOS and mask environment observations."""
    environments = []

    class TwoTurnEnvironment:
        """Return one intermediate observation before terminating on turn two."""

        def __init__(self) -> None:
            self.turn = 0
            self.closed = False

        async def reset(self, _prompt: PromptRecord) -> Observation:
            """Return one initial user token."""
            return Observation("prompt", torch.tensor([1]), {"role": "user"})

        async def step(self, _action: Any) -> Transition:
            """Advance once and terminate after the second generated action."""
            self.turn += 1
            return Transition(
                Observation(
                    f"observation-{self.turn}",
                    torch.tensor([30 + self.turn]),
                    {"role": "environment"},
                ),
                reward=float(self.turn == 2),
                done=self.turn == 2,
            )

        async def close(self) -> None:
            """Record lifecycle cleanup."""
            self.closed = True

    def build_environment(_prompt: PromptRecord) -> TwoTurnEnvironment:
        environment = TwoTurnEnvironment()
        environments.append(environment)
        return environment

    monkeypatch.setitem(ENVIRONMENTS._items, "phase2-two-turn", build_environment)  # pylint: disable=protected-access

    class Engine:
        """Return one EOS-terminated action per synchronous turn."""

        policy_version = 0

        def __init__(self) -> None:
            self.calls = 0
            self.row_seeds = []

        def generate(self, request: Any) -> GenerationResult:
            """Append a two-token response and preserve raw response logprobs."""
            self.calls += 1
            self.row_seeds.append(request.row_seeds)
            response = torch.tensor([[9 + self.calls, 2, 0]])
            return GenerationResult(
                sequences=torch.cat((request.input_ids, response), dim=-1),
                rollout_log_probs=torch.tensor(
                    [[-0.1 * self.calls, -0.2 * self.calls, 0.0]],
                    dtype=torch.float32,
                ),
                generation_seconds=1.0,
                response_mask=torch.tensor([[True, True, False]]),
                worker_policy_version=0,
                worker_policy_fingerprint="fingerprint",
            )

        @staticmethod
        def synchronize_error(error: Any, _operation: str) -> None:
            """Re-raise local failures in this single-rank contract test."""
            if error is not None:
                raise error

    engine = Engine()
    runner = AgentRunner(
        engine=engine,
        tokenizer=SimpleNamespace(decode=lambda tokens, **_kwargs: str(tokens)),
        environment_name="phase2-two-turn",
        num_samples=1,
        max_turns=2,
        max_observation_tokens=1,
        settings=GenerationSettings(
            max_new_tokens=3,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            do_sample=True,
            pad_token_id=0,
            eos_token_id=2,
            collect_log_probs=True,
            seed=10,
        ),
    )
    prompt = PromptRecord("0", (Message("user", "prompt"),))

    batch = runner.rollout([prompt], policy_version=0)
    trajectory = batch.trajectories[0]

    assert engine.calls == 2
    assert engine.row_seeds == [(10,), (10,)]
    assert trajectory.token_ids.tolist() == [1, 10, 2, 31, 11, 2, 32]
    assert trajectory.action_mask.tolist() == [False, True, True, False, True, True, False]
    assert torch.equal(
        trajectory.rollout_log_probs,
        torch.tensor([-0.1, -0.2, 0.0, -0.2, -0.4, 0.0], dtype=torch.float32),
    )
    assert trajectory.metadata["num_actions"] == 2
    assert trajectory.worker_policy_version == 0
    assert trajectory.worker_policy_fingerprint == "fingerprint"
    assert trajectory.done and not trajectory.truncated
    assert environments[0].closed
