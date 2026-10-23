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
"""Candidate execution adapters; no task admission, publication or checkpoint ownership."""
import threading
from dataclasses import replace
from typing import Any, Mapping

from rl.agentic.codex import CodexRuntime
from rl.roles.model_setup import load_tokenizer
from rl.roles.rollout.client import HTTPGenerationEngine
from rl.roles.rollout.worker import CodexRolloutManager, RolloutManager


class CodexGenerationEngine(HTTPGenerationEngine):
    """CPU gateway context; actual model calls use the guarded native chat route."""

    asynchronous = True

    @property
    def closed(self) -> bool:
        """Expose producer cancellation to running program event loops."""
        return self.client.closed

    def __init__(self, client: Any, version: int) -> None:
        super().__init__(client, version)
        self.inference_base_url, self.inference_model_name = client.base_url, client.model_name

    @staticmethod
    def synchronize_error(error: Exception | None, operation: str) -> None:
        """Keep invalid task contracts distinct from retryable execution failures."""
        if isinstance(error, (ValueError, TypeError)):
            raise error
        HTTPGenerationEngine.synchronize_error(error, operation)


class CandidateCollector:
    """Adapt internal single-turn or existing Codex multi-call executors."""

    def __init__(self, config: Mapping[str, Any], client: Any) -> None:
        self.config, self.client = config, client
        self.task_local = threading.local()
        self.runtime = None
        agent = config["agentic"]
        if agent.get("runner", "internal") == "codex":
            self.runtime = CodexRuntime(CodexGenerationEngine(client, 1), {
                **agent["codex"], "max_turns": agent["max_turns"],
                "max_episode_tokens": agent.get("max_episode_tokens"),
            })
            self.runtime.ensure_started()

    def generate(self, task: Any, prompt: Any, candidate: int, tokenizer: Any = None) -> tuple:
        """Return every call belonging to exactly one stable candidate identity."""
        if tokenizer is None:
            if not hasattr(self.task_local, "tokenizer"):
                self.task_local.tokenizer = load_tokenizer(self.config["model"])
            tokenizer = self.task_local.tokenizer
        rollout, agent = self.config["rollout"], self.config["agentic"]
        common = {
            "num_return_sequences": 1, "max_new_tokens": rollout["max_new_tokens"],
            "temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
            "collect_old_log_probs": True,
            "seed": (int(rollout.get("seed") or self.config["train"]["seed"])
                     + task.identity * rollout["num_return_sequences"] + candidate),
            "ignore_eos": rollout.get("ignore_eos", False),
        }
        if self.runtime is not None:
            manager = CodexRolloutManager(runtime=self.runtime, sample_index_offset=candidate, **common)
            rows = manager.generate([prompt], policy_version=task.policy_version).trajectories
            return tuple(replace(row, metadata={
                **row.metadata, "source_prompt_id": prompt.metadata["source_prompt_id"],
            }) for row in rows)
        engine = HTTPGenerationEngine(self.client, task.policy_version)
        manager = RolloutManager(
            engine=engine, tokenizer=tokenizer, environment_name=agent["environment"], max_turns=1,
            max_observation_tokens=agent.get("max_observation_tokens", 0),
            max_episode_tokens=agent.get("max_episode_tokens"), environment_settings=dict(agent),
            interaction_mode="single_turn", **common,
        )
        result = manager.generate([prompt], policy_version=task.policy_version)
        if len(result.trajectories) != 1 or not engine.segments:
            raise ValueError("Async internal candidate must contain sampled actions")
        row = result.trajectories[0]
        version = engine.segments[-1]["policy_version"]
        return (replace(
            row, trajectory_id=f"{task.identity}:{candidate}",
            policy_version=version, worker_policy_version=version,
            metadata={**row.metadata, "source_prompt_id": prompt.metadata["source_prompt_id"],
                      "behavior_segments": tuple(engine.segments), "partial_interruptions": engine.interruptions},
        ),)

    def close(self) -> None:
        """Release the sole gateway after candidate executors finish."""
        if self.runtime is not None:
            self.runtime.close()
