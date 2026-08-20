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

from types import SimpleNamespace
from typing import Any

import torch

import rl.trainer as trainer_backend
from rl.agentic.runner import AgentRunner
from rl.roles.rollout.base import GenerationSettings
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


def test_generation_settings_normalize_all_qwen3_eos_ids() -> None:
    """The primary and additional Qwen3 EOS IDs must form one unique contract."""
    settings = _generation_settings()

    assert settings.eos_token_ids == (151645, 151643), (
        f"Qwen3 EOS IDs were not normalized: actual={settings.eos_token_ids}, "
        f"expected={(151645, 151643)}"
    )


def test_response_mask_excludes_either_qwen3_eos_and_following_tokens() -> None:
    """Both Qwen3 terminal tokens and all later tokens must be masked consistently."""
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
            [True, False, False],
            [True, False, False],
            [False, False, False],
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
    captured_payload = {}

    def _request(_method, _route, payload, **_kwargs):
        captured_payload.update(payload)
        return {
            "choices": [
                {
                    "index": 0,
                    "token_ids": [151643],
                    "logprobs": None,
                }
            ]
        }

    client._request = _request  # type: ignore[method-assign]  # pylint: disable=protected-access

    client._generate_completion_batch(  # pylint: disable=protected-access
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


def test_inprocess_vllm_receives_all_qwen3_stop_token_ids() -> None:
    """The injected vLLM client must stop on every Qwen3 EOS ID as well."""
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
