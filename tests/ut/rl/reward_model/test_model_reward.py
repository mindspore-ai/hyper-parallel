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
"""CPU contracts for task-owned scoring and a colocated frozen RM service."""

import asyncio
import json
from dataclasses import replace
import os
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

import torch

from rl.agentic.core.types import Action, EpisodeContext, InteractionMode, RewardResult, TurnContext
from rl.dataset.batch_builder import ExperiencePreparer, build_experience_batch
from rl.dataset.contracts import Message, PromptRecord, Trajectory, Turn
from rl.reward_model.client import RewardModelClient
from rl.reward_model.scoring import load_reward_function, score_model_batch, scorer_fingerprint
from rl.roles.rollout.base import GenerationSettings
from rl.trainer import SyncTrainer
from rl.utils.monitoring.metrics import _local_rollout_record, _rollout_metrics

from examples.gsm8k.agent import (
    GSM8KMultiTurnEnvironment, GSM8KSingleTurnEnvironment, build_gsm8k_environment,
    score_gsm8k_environment_reward,
)


def _batch() -> tuple[tuple[PromptRecord, ...], object]:
    """Make one prompt group with two unscored, token-aligned trajectories."""
    prompt = PromptRecord("p", (Message("user", "What is 6 times 7?"),), "42")
    trajectories = tuple(Trajectory(
        trajectory_id=f"p-{index}", prompt_id="p", group_id="p", policy_version=0,
        turns=(Turn("assistant", answer, 1, 3, True),), token_ids=torch.tensor([1, 2, 3]),
        attention_mask=torch.tensor([True, True, True]), action_mask=torch.tensor([False, True, True]),
        rollout_log_probs=torch.tensor([-1.0, -1.0]), reward=0.0, reward_components={},
        done=True, truncated=False, terminal_reason="completed", worker_policy_version=0,
    ) for index, answer in enumerate(("####42", "####41")))
    settings = GenerationSettings(max_new_tokens=2, temperature=1.0, top_p=1.0, top_k=0,
                                  do_sample=True, pad_token_id=0, eos_token_id=3, collect_log_probs=True)
    metadata = {"reward_status": "pending", "interaction_mode": "single_turn"}
    return (prompt,), build_experience_batch(trajectories, 0.1, settings, metadata)


class _FakeClient:
    """Emulate the RM response without starting a model or HTTP server."""

    def __init__(self, response: Optional[dict] = None) -> None:
        """Store a controllable model response."""
        self.config = {"scoring": "generative", "model_path": "qwen3"}
        self.model_name = "qwen3"
        self.state = "awake"
        self.request_count = 0
        self.retry_count = 0
        self.closed_connections = 0
        self.response = response

    async def request(self, endpoint: str, payload: dict) -> dict:
        """Return a complete score for one candidate or an injected response."""
        assert endpoint == "v1/chat/completions"
        self.request_count += 1
        if self.response is not None:
            return self.response
        candidate = json.loads(payload["messages"][-1]["content"])["candidate"]
        score = 1 if "####42" in candidate else 0
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"score": score})}}]}

    async def close_connection(self) -> None:
        """Track closing the session on the scoring event loop."""
        self.closed_connections += 1


class TestModelReward(unittest.TestCase):
    """Validate the actual task scorer and immutable training boundary."""

    def test_gsm8k_environment_defers_only_in_model_mode(self) -> None:
        """The same completed answer remains available while rule scoring stays unchanged."""
        prompt = PromptRecord("p", (Message("user", "What is 6 times 7?"),), "42")
        action = Action("####42", torch.tensor([2]))
        for deferred, expected in ((False, 1.0), (True, 0.0)):
            with self.subTest(deferred=deferred):
                context = EpisodeContext(prompt, 0, 0, 1, settings={"defer_reward_model": deferred})
                transition = asyncio.run(GSM8KSingleTurnEnvironment(context).step(
                    action, TurnContext(context, 0, 0.0),
                ))
                self.assertEqual(transition.reward, expected)
                self.assertEqual(transition.info["extracted_answer"], "42")
                self.assertEqual(transition.info["reward_components"], {} if deferred else {"correctness": 1.0})

    def test_rule_and_model_use_gsm8k_environment_scoring(self) -> None:
        """Single and multi-turn environments own rule and deferred RM selection."""
        prompts, batch = _batch()
        prompt = prompts[0]
        context = EpisodeContext(prompt, 0, 0, 2, settings={}, interaction_mode=InteractionMode.MULTI_TURN)
        environment = build_gsm8k_environment(context)
        self.assertIsInstance(environment, GSM8KMultiTurnEnvironment)
        try:
            with patch.object(GSM8KMultiTurnEnvironment, "score_reward",
                              wraps=GSM8KMultiTurnEnvironment.score_reward) as rule_scorer:
                self.assertEqual(environment.reward_function("####42", prompt), 1.0)
                rule_scorer.assert_called_once_with(prompt, answer="####42")
        finally:
            asyncio.run(environment.close())

        for mode, environment_type in (
            ("single_turn", GSM8KSingleTurnEnvironment),
            ("multi_turn", GSM8KMultiTurnEnvironment),
        ):
            with self.subTest(mode=mode):
                pending = replace(batch, metadata={**batch.metadata, "interaction_mode": mode})
                client = _FakeClient()
                with patch.object(environment_type, "score_reward", wraps=environment_type.score_reward) as scorer:
                    scored = score_model_batch(
                        prompts, pending, scorer=score_gsm8k_environment_reward, client=client,
                        scorer_id="judge-v1", tp_rank=0, tp_size=1, tp_group=None, max_concurrency=2,
                    )
                self.assertEqual(scored.rewards.tolist(), [1.0, 0.0])
                self.assertEqual(scorer.call_count, 2)
                self.assertEqual(client.request_count, 2)

    def test_environment_scorer_rejects_missing_mode_before_rm_request(self) -> None:
        """A missing environment mode must fail before requesting a model score."""
        prompts, batch = _batch()
        client = _FakeClient()
        pending = replace(batch, metadata={"reward_status": "pending"})
        with self.assertRaisesRegex(ValueError, "interaction_mode"):
            score_model_batch(
                prompts, pending, scorer=score_gsm8k_environment_reward, client=client,
                scorer_id="judge-v1", tp_rank=0, tp_size=1, tp_group=None, max_concurrency=2,
            )
        self.assertEqual(client.request_count, 0)
        self.assertEqual(pending.metadata["reward_status"], "pending")

    def test_scoring_commits_only_completed_batch(self) -> None:
        """Model scores replace placeholders without changing the original rollout."""
        prompts, batch = _batch()
        client = _FakeClient()
        with self.assertRaisesRegex(ValueError, "Pending"):
            ExperiencePreparer(SimpleNamespace(name="grpo")).prepare(batch)
        scored = score_model_batch(prompts, batch, scorer=score_gsm8k_environment_reward, client=client,
                                   scorer_id="judge-v1", tp_rank=0, tp_size=1, tp_group=None, max_concurrency=2)
        self.assertEqual(scored.rewards.tolist(), [1.0, 0.0])
        self.assertEqual(batch.rewards.tolist(), [0.0, 0.0])
        self.assertEqual(batch.metadata["reward_status"], "pending")
        self.assertEqual(scored.metadata["reward_status"], "scored")
        self.assertEqual(client.request_count, 2)
        self.assertEqual(client.closed_connections, 1)
        with patch("rl.utils.monitoring.metrics.dist.get_rank", return_value=0):
            record = _local_rollout_record(scored, {"sample_indices": [0]}, step=1, sample_limit=0)
        self.assertNotIn("reward/accuracy", _rollout_metrics([record]))
        self.assertEqual(_rollout_metrics([record])["reward/request_count"], 2)

    def test_task_owned_tool_penalty_is_applied_after_model_score(self) -> None:
        """Deferred GSM8K scoring preserves the original per-error penalty."""
        prompts, batch = _batch()
        first = replace(batch.trajectories[0], metadata={
            "turn_infos": ({"interaction_error": "invalid calculator call"},),
        })
        batch = replace(batch, trajectories=(first, batch.trajectories[1]))
        scored = score_model_batch(prompts, batch, scorer=score_gsm8k_environment_reward, client=_FakeClient(),
                                   scorer_id="judge-v1", tp_rank=0, tp_size=1, tp_group=None, max_concurrency=2)
        self.assertAlmostEqual(scored.rewards[0].item(), 0.95)
        self.assertEqual(scored.trajectories[0].reward_components["interaction_penalty"], -0.05)

    def test_bad_model_response_never_commits(self) -> None:
        """Truncated, malformed, and out-of-range scores fail the whole batch."""
        prompts, batch = _batch()
        for response in (
            {"choices": [{"finish_reason": "length", "message": {"content": '{"score": 1}'}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": "not json"}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": '{"score": 5}'}}]},
        ):
            with self.subTest(response=response):
                client = _FakeClient(response)
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    score_model_batch(prompts, batch, scorer=score_gsm8k_environment_reward, client=client,
                                      scorer_id="judge-v1", tp_rank=0, tp_size=1, tp_group=None, max_concurrency=2)
                self.assertEqual(batch.metadata["reward_status"], "pending")
                self.assertEqual(client.closed_connections, 1)

    def test_tp_sibling_uses_owner_scores_without_requesting_rm(self) -> None:
        """One logical TP candidate is scored once even with two trainer ranks."""
        prompts, batch = _batch()
        client = _FakeClient()
        groups = []

        def gather(output: list, value: object, *, group: str) -> None:
            """Give the sibling matching inputs and the owner's completed scores."""
            groups.append(group)
            if value is None:
                output[:] = [[RewardResult(1.0), RewardResult(0.0)], None]
            else:
                output[:] = [value, value]

        with patch("rl.reward_model.scoring.dist.all_gather_object", side_effect=gather):
            scored = score_model_batch(
                prompts, batch, scorer=score_gsm8k_environment_reward, client=client, scorer_id="judge-v1",
                tp_rank=1, tp_size=2, tp_group="tp", max_concurrency=2,
            )
        self.assertEqual(scored.rewards.tolist(), [1.0, 0.0])
        self.assertEqual(client.request_count, 0)
        self.assertEqual(groups, ["tp", "tp"])

    def test_nonfinite_and_float32_overflow_fail_closed(self) -> None:
        """A Python score must fit the training reward tensor dtype."""
        prompts, batch = _batch()

        async def oversized(*_args: object) -> RewardResult:
            """Return a score outside the training tensor range."""
            return RewardResult(1e100)

        with self.assertRaisesRegex(ValueError, "finite"):
            score_model_batch(prompts, batch, scorer=oversized, client=_FakeClient(),
                              scorer_id="judge-v1", tp_rank=0, tp_size=1, tp_group=None, max_concurrency=2)

    def test_scorer_loader_and_identity(self) -> None:
        """Task callback must be async; changing ports does not change score identity."""
        self.assertIs(
            load_reward_function("examples.gsm8k.agent:score_gsm8k_environment_reward"),
            score_gsm8k_environment_reward,
        )
        with self.assertRaisesRegex(ValueError, "async"):
            load_reward_function("examples.gsm8k.agent:compute_gsm8k_reward")
        first = {"model_path": "qwen3", "scorer": "task:fn", "port": 8200}
        second = {**first, "port": 8300}
        self.assertEqual(scorer_fingerprint(first), scorer_fingerprint(second))
        self.assertNotEqual(scorer_fingerprint(first), scorer_fingerprint({**first, "model_path": "other"}))

    def test_client_isolates_rank_environment_and_requires_sleep(self) -> None:
        """RM command has no Actor weight transfer and only owner may mutate residency."""
        config = {"port": 8200, "model_path": "qwen3", "scoring": "generative",
                  "server_hccl_if_base_port": 64000,
                  "server_hccl_npu_socket_port_range": "64000-64099"}
        client = RewardModelClient(config, ("4", "5"), owner=True)
        self.assertIn("--enable-sleep-mode", client.server_command())
        self.assertNotIn("--weight-transfer-config", client.server_command())
        with patch.dict(os.environ, {"RANK": "1", "TORCHELASTIC_ERROR_FILE": "trainer-error"}):
            environment = client.server_environment()
        self.assertEqual(environment["ASCEND_RT_VISIBLE_DEVICES"], "4,5")
        self.assertNotIn("RANK", environment)
        self.assertNotIn("TORCHELASTIC_ERROR_FILE", environment)
        with patch.object(client, "_start"), patch.object(client, "_require_sleep_state"):
            self.assertEqual(client.prepare(), "awake")
            with patch.object(client, "control"):
                self.assertEqual(client.sleep(), "sleeping")
        client.close()
        sibling = RewardModelClient(config, ("4", "5"), owner=False)
        with self.assertRaisesRegex(RuntimeError, "owner"):
            sibling.prepare()
        sibling.close()


class TestColocatedScoringOrder(unittest.TestCase):
    """Keep shared-device ownership in Trainer without a reward manager."""

    def test_rollout_sleeps_before_rm_and_eval_resumes_afterward(self) -> None:
        """The task callback runs only between RM wake and verified sleep."""
        events = []
        trainer = object.__new__(SyncTrainer)
        trainer.rollout_engine = SimpleNamespace(
            prepare_for_training=lambda: events.append("rollout_sleep"),
            prepare_for_rollout=lambda: events.append("rollout_resume"),
        )
        trainer._release_training_state_for_rollout = lambda: events.append("release")
        trainer.parallel_dims = SimpleNamespace(tp_rank=0, tp_size=1)
        trainer.reward_model_config = {"max_concurrency": 2}
        trainer.reward_model_scorer = score_gsm8k_environment_reward
        trainer.reward_model_scorer_id = "judge-v1"

        class Client:
            """Record the only operations that may change RM residency."""
            state = "sleeping"

            @staticmethod
            def prepare() -> str:
                """Wake the fake reward service."""
                events.append("rm_wake")
                return "awake"

            @staticmethod
            def sleep() -> str:
                """Sleep the fake reward service."""
                events.append("rm_sleep")
                return "sleeping"

        trainer.reward_model_client = Client()
        scored = SimpleNamespace(metadata={"reward_status": "scored"})
        with patch("rl.trainer.score_model_batch", side_effect=lambda *_args, **_kwargs: (
                events.append("score") or scored)):
            result = trainer._score_model_rollout((), object(), evaluation=True)
        self.assertIs(result, scored)
        self.assertEqual(events, ["rollout_sleep", "release", "rm_wake", "score", "rm_sleep", "rollout_resume"])
        self.assertEqual(trainer.reward_model_client.state, "sleeping")

    def test_score_failure_still_sleeps_rm(self) -> None:
        """An exception cannot leave the shared reward service active."""
        events = []
        trainer = object.__new__(SyncTrainer)
        trainer.rollout_engine = SimpleNamespace(prepare_for_training=lambda: events.append("rollout_sleep"))
        trainer._release_training_state_for_rollout = lambda: events.append("release")
        trainer.parallel_dims = SimpleNamespace(tp_rank=0, tp_size=1)
        trainer.reward_model_config = {}
        trainer.reward_model_scorer = score_gsm8k_environment_reward
        trainer.reward_model_scorer_id = "judge-v1"
        client = SimpleNamespace(state="sleeping")

        def prepare() -> str:
            """Wake the fake reward service."""
            events.append("rm_wake")
            return "awake"

        def sleep() -> str:
            """Sleep the fake reward service."""
            events.append("rm_sleep")
            return "sleeping"

        client.prepare, client.sleep = prepare, sleep
        trainer.reward_model_client = client
        with patch("rl.trainer.score_model_batch", side_effect=RuntimeError("score failed")):
            with self.assertRaisesRegex(RuntimeError, "score failed"):
                trainer._score_model_rollout((), object())
        self.assertEqual(events, ["rollout_sleep", "release", "rm_wake", "rm_sleep"])
        self.assertEqual(client.state, "sleeping")
