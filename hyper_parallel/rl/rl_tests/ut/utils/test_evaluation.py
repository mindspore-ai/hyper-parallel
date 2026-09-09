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

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import rl.evaluation as evaluation_module
from rl.evaluation import Evaluator


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
            assert policy_version in (7, 8)
            return SimpleNamespace(
                rewards=torch.tensor([1.0, 999.0]),
                action_mask=torch.tensor([[False, True, True], [False, True, True]]),
                responses=("correct", "padding"),
                trajectories=(
                    SimpleNamespace(metadata={"extracted_answer": "1"}),
                    SimpleNamespace(metadata={"extracted_answer": "999"}),
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
    monkeypatch.setattr(evaluation_module.platform, "get_rank", lambda: current_rank[0])
    monkeypatch.setattr(evaluation_module.platform, "get_world_size", lambda: 2)

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

    monkeypatch.setattr(evaluation_module.platform, "all_gather_object", all_gather)

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
