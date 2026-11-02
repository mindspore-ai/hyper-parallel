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
"""CPU unit test for distributed Hyper-RL evaluation aggregation."""
# Local test doubles are not public APIs; the suite intentionally uses Torch CPU tensors.
# pylint: disable=forbidden-backend-import,missing-public-docstring

from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import rl.evaluation as evaluation_module
from rl.evaluation import Evaluator
from rl.dataset.data_source import build_padded_evaluation_batches, collate_prompt_samples
from rl.dataset.episodes import generation_finish_reason


@pytest.mark.parametrize("sample_count,batch_size", [(500, 64), (3, 2), (8, 1)])
def test_evaluator_padding_keeps_harness_episode_ids_unique(sample_count: int, batch_size: int) -> None:
    """Repeated tail questions form distinct multi-call episodes but never add validation credit."""
    dataset = [
        {"sample_index": index, "source_prompt": str(index), "prompt": str(index), "ground_truth": "1",
         "input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1])}
        for index in range(sample_count)
    ]

    def generate(prompt_records: Any, policy_version: int) -> Any:
        """Produce two calls per unique episode so padding cannot gain validation credit."""
        identities = [prompt.prompt_id for prompt in prompt_records]
        assert len(set(identities)) == len(identities), f"Expected unique episode IDs, got {identities}"
        trajectories = tuple(
            SimpleNamespace(
                prompt_id=prompt.prompt_id, group_id=prompt.prompt_id, policy_version=policy_version,
                worker_policy_version=policy_version, reward=1.0, reward_components={"success": 1.0},
                metadata={"episode_id": prompt.prompt_id, "call_index": call, "call_count": 2},
            )
            for prompt in prompt_records for call in range(2)
        )
        return SimpleNamespace(
            trajectories=trajectories, rewards=torch.ones(len(trajectories)),
            action_mask=torch.tensor([[False, True]] * len(trajectories)),
            responses=("answer",) * len(trajectories), generation_seconds=1.0,
        )

    evaluator = Evaluator(
        dataset=dataset, collate_fn=partial(collate_prompt_samples, pad_token_id=0),
        rollout_manager=SimpleNamespace(generate=generate), device=torch.device("cpu"),
        batch_size=batch_size, max_samples=sample_count, log_samples=sample_count, progress_steps=0,
    )
    records = []
    for rank in range(8):
        batches = build_padded_evaluation_batches(sample_count, 8, rank, batch_size, sample_count)
        for entries in batches:
            record = evaluator._evaluate_batch(0, rank, entries, sample_count)
            records.append(record)
    actual_total = sum(record["total"] for record in records)
    actual_correct = sum(record["correct"] for record in records)
    actual_tokens = sum(record["generated_tokens"] for record in records)
    assert actual_total == sample_count, f"Expected {sample_count} scored questions, got {actual_total}"
    assert actual_correct == sample_count, f"Expected {sample_count} successes, got {actual_correct}"
    assert actual_tokens == sample_count * 2, f"Expected {sample_count * 2} action tokens, got {actual_tokens}"


def test_finish_reason_uses_last_model_action_and_preserves_unknown() -> None:
    """An environment completion must not hide a truncated last generation or invent a stop."""
    trajectory = SimpleNamespace(metadata={"finish_reason": "stop"}, turns=(
        SimpleNamespace(role="assistant", trainable=True, metadata={"finish_reason": "stop"}),
        SimpleNamespace(role="assistant", trainable=True, metadata={"finish_reason": "length"}),
        SimpleNamespace(role="environment", trainable=False, metadata={"finish_reason": "stop"}),
    ))
    assert generation_finish_reason(trajectory) == "length"
    trajectory.turns[1].metadata.clear()
    assert generation_finish_reason(trajectory) is None
    assert generation_finish_reason(SimpleNamespace(metadata={"finish_reason": "stop"})) == "stop"


def test_evaluator_excludes_padding_and_aggregates_rank_zero_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Padded rows are ignored while rank-zero metrics merge true global samples."""
    dataset = [
        {
            "sample_index": 0,
            "source_prompt": "p0",
            "prompt": "p0",
            "ground_truth": "1",
            "input_ids": torch.tensor([1]),
            "attention_mask": torch.tensor([1]),
        },
        {
            "sample_index": 1,
            "source_prompt": "padding",
            "prompt": "padding",
            "ground_truth": "0",
            "input_ids": torch.tensor([2]),
            "attention_mask": torch.tensor([1]),
        },
    ]

    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "input_ids": torch.stack([sample["input_ids"] for sample in samples]),
            "attention_mask": torch.stack([sample["attention_mask"] for sample in samples]),
            "sample_indices": [sample["sample_index"] for sample in samples],
            "prompts": [sample["prompt"] for sample in samples],
            "ground_truths": [sample["ground_truth"] for sample in samples],
        }

    class RolloutManager:
        """Return one valid row and one deliberately extreme padded row."""

        @staticmethod
        def generate(prompt_records: Any, policy_version: int) -> Any:
            assert len(prompt_records) == 2
            assert all(prompt.metadata["phase"] == "evaluation" for prompt in prompt_records)
            assert policy_version in (7, 8)
            return SimpleNamespace(
                rewards=torch.tensor([1.0, 999.0]),
                action_mask=torch.tensor([[False, True, True], [False, True, True]]),
                responses=("correct", "padding"),
                trajectories=(
                    SimpleNamespace(metadata={"extracted_answer": "1"}, reward_components={"success": 1.0}),
                    SimpleNamespace(metadata={"extracted_answer": "999"}, reward_components={"success": 0.0}),
                ),
                generation_seconds=2.0,
            )

    evaluator = Evaluator(
        dataset=dataset,
        collate_fn=collate,
        rollout_manager=RolloutManager(),
        device=torch.device("cpu"),
        batch_size=2,
        max_samples=None,
        log_samples=2,
        progress_steps=0,
    )
    monkeypatch.setattr(
        evaluation_module,
        "build_padded_evaluation_batches",
        lambda **_kwargs: [[(0, True), (1, False)]],
    )
    current_rank = [0]
    monkeypatch.setattr(evaluation_module.dist, "get_rank", lambda: current_rank[0])
    monkeypatch.setattr(evaluation_module.dist, "get_world_size", lambda: 2)

    def all_gather(output: list[Any], local: dict[str, Any]) -> None:
        output[0] = local
        output[1] = {
            "correct": 0.0,
            "total": 1,
            "generated_tokens": 3,
            "response_length": 3,
            "generation_seconds": 4.0,
            "samples": [
                {
                    "step": 7,
                    "rank": 1,
                    "prompt": "p1",
                    "response": "wrong",
                    "ground_truth": "1",
                    "extracted_answer": "0",
                    "reward": 0.0,
                }
            ],
        }

    monkeypatch.setattr(evaluation_module.dist, "all_gather_object", all_gather)

    metrics, samples = evaluator.run(7)
    current_rank[0] = 1
    non_owner_metrics, non_owner_samples = evaluator.run(8)

    assert metrics == {
        "validation/accuracy": 0.5,
        "validation/correct": 1.0,
        "validation/total": 2.0,
        "validation/response_length_mean": 2.5,
        "validation/generated_tokens": 5.0,
        "validation/generation_seconds": 4.0,
        "validation/tokens_per_second": 1.25,
    }
    assert [sample["rank"] for sample in samples] == [0, 1]
    assert all(sample["response"] != "padding" for sample in samples)
    assert not non_owner_metrics
    assert not non_owner_samples
    assert evaluator.last_step == 8
