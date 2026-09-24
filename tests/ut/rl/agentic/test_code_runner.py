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
"""CPU TP simulations for single-owner environment effects and failure ordering."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch

from rl.agentic.core import runner as runner_module
from rl.agentic.core.types import Observation, Transition
from rl.dataset.contracts import ExperienceBatch, Message, PromptRecord, Trajectory
from rl.dataset.data_source import collate_prompt_samples
import rl.evaluation as evaluation_module
from rl.roles.rollout.base import GenerationResult, GenerationSettings
import rl.utils.monitoring.metrics as metrics_module




@pytest.mark.parametrize("failure", [None, "reset", "step", "close", "payload_count", "payload_identity"])
def test_tp_environment_ownership_and_failure_order(monkeypatch: pytest.MonkeyPatch, failure: Optional[str]) -> None:
    """TP peers share effects, exact actions, and errors without mismatched collectives."""
    barrier = threading.Barrier(2, timeout=5)
    slots = {}
    events = []
    generated = [0, 0]

    def exchange(rank: int, value: Any) -> list:
        """Exchange each rank's value before allowing the next simulated collective."""
        slots[rank] = value
        barrier.wait()
        result = [slots[0], slots[1]]
        barrier.wait()
        return result

    class Engine:
        """Simulate ordered TP communication without initializing a process group."""

        policy_version = 2

        def __init__(self, rank: int) -> None:
            """Select the simulated request owner."""
            self.rank = rank
            self.is_request_owner = rank == 0

        def synchronize_error(self, error: Any, operation: str) -> None:
            """Fail explicitly if peers enter different communication stages."""
            values = exchange(self.rank, (operation, None if error is None else str(error)))
            assert values[0][0] == values[1][0], f"Collective order differs: {values}"
            errors = [value[1] for value in values if value[1] is not None]
            if errors:
                raise RuntimeError(str(errors))

        def synchronize_agent_payload(self, payload: Any) -> Any:
            """Mirror owner payload, optionally corrupting only the receiving peer."""
            values = exchange(self.rank, payload)
            assert values[0] is not None and values[1] is None
            result = values[0]
            if self.rank == 1 and failure == "payload_count":
                return result[:-1]
            if self.rank == 1 and failure == "payload_identity":
                result = [("wrong-prompt", item[1], item[2]) for item in result]
            return result

        def generate(self, request: Any) -> GenerationResult:
            """Check matching inputs and return distinct raw actions with EOS."""
            values = exchange(self.rank, request.input_ids.tolist())
            assert values[0] == values[1]
            generated[self.rank] += 1
            tokens = torch.tensor([[5, 2], [6, 2]])
            return GenerationResult(torch.cat((request.input_ids, tokens), dim=1),
                                    torch.tensor([[-0.1, -0.2], [-0.3, -0.4]]), 0.1,
                                    worker_policy_version=2, finish_reasons=("stop", "length"))

    class Environment:
        """Record real side effects; delay one candidate to test settled failure handling."""

        def __init__(self, context: Any) -> None:
            """Record construction of an owned candidate environment."""
            self.index = context.sample_index
            events.append(("build", self.index))

        async def reset(self, context: Any) -> Observation:
            """Return initial policy tokens without exposing hidden tests."""
            events.append(("reset", self.index))
            if failure == "reset" and self.index == 1:
                raise RuntimeError("reset failure")
            await asyncio.sleep(0.01 if self.index == 0 else 0)
            return Observation("question", context.prompt.metadata["input_ids"], {"role": "user"})

        async def step(self, action: Any, context: Any) -> Transition:
            """Check raw tokens/logprobs and complete candidates on different turns."""
            events.append(("step", self.index))
            try:
                if failure == "step" and self.index == 1:
                    raise RuntimeError("step failure")
                await asyncio.sleep(0.01 if self.index == 0 else 0)
                assert action.metadata["finish_reason"] == ("stop" if self.index == 0 else "length")
                assert action.token_ids.tolist() == [5 + self.index, 2]
                torch.testing.assert_close(action.rollout_log_probs, torch.tensor(
                    [-0.1, -0.2] if self.index == 0 else [-0.3, -0.4]))
                done = self.index == 0 or context.turn_index == 1
                return Transition(Observation("feedback", torch.tensor([20 + self.index])), float(done), done)
            finally:
                events.append(("settled", self.index))

        async def close(self) -> None:
            """Close exactly the owned environments after in-flight work settles."""
            events.append(("close", self.index))
            if failure == "close" and self.index == 1:
                raise RuntimeError("close failure")

    monkeypatch.setattr(runner_module.ENVIRONMENTS, "build", lambda _name, context: Environment(context))
    monkeypatch.setattr(runner_module, "build_experience_batch", SimpleNamespace)
    tokenizer = SimpleNamespace(batch_decode=lambda rows, **_kwargs: [str(row) for row in rows])
    settings = GenerationSettings(2, 0, 1, 0, False, 0, 2, collect_log_probs=True)
    prompt = PromptRecord("task-id", (Message("user", "question"),),
                          ground_truth={"inputs": ["private"], "outputs": ["private"]},
                          metadata={"input_ids": torch.tensor([10, 11])})
    runners = [runner_module.AgentRunner(Engine(rank), tokenizer, "mock", 2, 2, 4, settings) for rank in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(runner.rollout, [prompt], 2) for runner in runners]
        if failure:
            message = "payload" if failure.startswith("payload") else f"{failure} failure"
            for future in futures:
                with pytest.raises(RuntimeError, match=message):
                    future.result(timeout=15)
        else:
            batches = [future.result(timeout=15) for future in futures]
            assert generated == [2, 2]
            for owner, replica in zip(batches[0].trajectories, batches[1].trajectories):
                torch.testing.assert_close(owner.token_ids, replica.token_ids)
                torch.testing.assert_close(owner.action_mask, replica.action_mask)
                torch.testing.assert_close(owner.rollout_log_probs, replica.rollout_log_probs)
                assert owner.reward == replica.reward == 1
                assert owner.turns == replica.turns
    for event in ("build", "reset", "close"):
        assert events.count((event, 0)) == events.count((event, 1)) == 1
    if failure == "step":
        first_close = events.index(("close", 0))
        assert all(events.index(("settled", index)) < first_close for index in range(2))
    if failure is None:
        assert events.count(("step", 0)) == 1 and events.count(("step", 1)) == 2


def test_evaluation_tp_shares_prompts_and_counts_only_owners(monkeypatch: pytest.MonkeyPatch) -> None:
    """DP2/TP2 evaluates each task once and hides structured judging tests in logs."""
    dataset = [{"sample_index": index, "source_prompt": f"task-{index}", "prompt": f"task-{index}",
                "prompt_id": f"stable-{index}", "input_ids": torch.tensor([index + 10]),
                "attention_mask": torch.tensor([1]), "ground_truth": {"inputs": ["secret"], "outputs": ["secret"]}}
               for index in range(3)]
    current_rank = [0]
    prompts_by_rank = {}
    contributions = {}
    monkeypatch.setattr(evaluation_module.dist, "get_rank", lambda: current_rank[0])
    monkeypatch.setattr(evaluation_module.dist, "get_world_size", lambda: 4)

    def gather(output: list, record: Any) -> None:
        """Collect simulated ranks before the last rank-zero summary."""
        contributions[current_rank[0]] = record
        output[:] = [contributions.get(rank) for rank in range(4)]

    def generate(prompt_records: Any, policy_version: int) -> Any:
        """Produce success components distinct from training rewards."""
        assert policy_version == 3
        prompts_by_rank.setdefault(current_rank[0], []).extend(record.prompt_id for record in prompt_records)
        count = len(prompt_records)
        return SimpleNamespace(rewards=torch.full((count,), 0.25),
                               action_mask=torch.tensor([[False, True, True]] * count),
                               responses=tuple("answer" for _ in range(count)), generation_seconds=1.0,
                               trajectories=tuple(SimpleNamespace(reward_components={"success": 1.0}, metadata={})
                                                  for _ in range(count)))

    monkeypatch.setattr(evaluation_module.dist, "all_gather_object", gather)
    metrics, samples = {}, []
    for rank in (1, 2, 3, 0):
        current_rank[0] = rank
        evaluator = evaluation_module.Evaluator(
            dataset=dataset, collate_fn=partial(collate_prompt_samples, pad_token_id=0),
            rollout_manager=SimpleNamespace(generate=generate), device=torch.device("cpu"),
            batch_size=2, max_samples=None, log_samples=4, progress_steps=0,
            data_parallel_rank=rank // 2, data_parallel_size=2, is_request_owner=rank % 2 == 0,
        )
        metrics, samples = evaluator.run(3)
    assert prompts_by_rank[0] == prompts_by_rank[1]
    assert prompts_by_rank[2] == prompts_by_rank[3]
    assert contributions[1] is contributions[3] is None
    assert metrics["validation/total"] == 3
    assert metrics["validation/correct"] == 3
    assert metrics["validation/accuracy"] == 1
    assert metrics["validation/generated_tokens"] == 6
    assert len(samples) == 3 and all("ground_truth" not in sample for sample in samples)


def test_rollout_metrics_count_only_tp_owners(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real DP2/TP2 metric aggregation counts sixteen unique candidates, not replicas."""
    current_rank = [0]
    contributions = {}
    monkeypatch.setattr(metrics_module.dist, "get_rank", lambda: current_rank[0])
    monkeypatch.setattr(metrics_module.dist, "get_world_size", lambda: 4)

    def gather(output: list, record: Any) -> None:
        """Store each rank contribution before the final rank-zero summary."""
        contributions[current_rank[0]] = record
        output[:] = [contributions.get(rank) for rank in range(4)]

    monkeypatch.setattr(metrics_module.dist, "all_gather_object", gather)
    metrics, samples = {}, []
    for rank in (1, 2, 3, 0):
        current_rank[0] = rank
        prompt_id = f"task-{rank // 2}"
        tokens = torch.tensor([10, 20, 2])
        mask = torch.tensor([False, True, True])
        attention = torch.ones(3, dtype=torch.bool)
        log_probs = torch.tensor([-0.1, -0.2])
        trajectories = tuple(Trajectory(
            trajectory_id=f"{prompt_id}-{index}", prompt_id=prompt_id, group_id=prompt_id, policy_version=1,
            turns=(), token_ids=tokens, attention_mask=attention, action_mask=mask,
            rollout_log_probs=log_probs, reward=float(index < 4), reward_components={"success": float(index < 4)},
            done=True, truncated=False, terminal_reason="completed",
            metadata={"status": "passed" if index < 4 else "wrong_answer"}, worker_policy_version=1,
        ) for index in range(8))
        rollout = ExperienceBatch(
            trajectories=trajectories, sequences=tokens.repeat(8, 1), attention_mask=attention.repeat(8, 1),
            action_mask=mask.repeat(8, 1), rewards=torch.tensor([item.reward for item in trajectories]),
            old_log_probs=log_probs.repeat(8, 1), responses=tuple("answer" for _ in range(8)),
            generation_seconds=2.0, worker_policy_version=1,
        )
        batch = {"prompt_ids": [prompt_id], "sample_indices": [rank // 2], "prompts": ["question"],
                 "ground_truths": [{"inputs": ["secret"], "outputs": ["secret"]}]}
        metrics, samples = metrics_module.summarize_rollout(
            rollout, batch, step=2, sample_limit=8, is_request_owner=rank % 2 == 0,
        )
    assert contributions[1] is contributions[3] is None
    assert metrics["rollout/sequence_count"] == 16
    assert metrics["rollout/status/passed"] == metrics["rollout/status/wrong_answer"] == 8
    assert metrics["rollout/generated_tokens"] == 32
    assert metrics["rollout/tokens_per_second"] == 16
    assert metrics["reward/accuracy"] == metrics["reward/mean"] == metrics["reward/component/success"] == 0.5
    assert {sample["rank"] for sample in samples} == {0, 2}
    assert all("ground_truth" not in sample for sample in samples)
