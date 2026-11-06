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
"""Colocated reward service execution."""
import time
from typing import Any
from rl.data.contracts import ExperienceBatch
from rl.reward_model import score_model_batch
from rl.weight_sync.sync import coordinator_call

import logging

logger = logging.getLogger(__name__)

class RewardScorer:
    """Own reward service wake, scoring and sleep ordering."""

    def __init__(self, *, reward_model_client: Any = None, reward_model_scorer: Any = None, reward_model_scorer_id: Any = None, reward_model_config: Any = None, parallel_dims: Any = None, rollout_engine: Any = None, residency: Any = None) -> None:
        """Bind explicit dependencies for this lifecycle boundary."""
        self.reward_model_client = reward_model_client
        self.reward_model_scorer = reward_model_scorer
        self.reward_model_scorer_id = reward_model_scorer_id
        self.reward_model_config = reward_model_config
        self.parallel_dims = parallel_dims
        self.rollout_engine = rollout_engine
        self.residency = residency

    def score(
        self, prompts: Any, rollout: ExperienceBatch, *, evaluation: bool = False,
    ) -> ExperienceBatch:
        """Switch shared NPU residency, invoke the task scorer, and restore rollout for evaluation."""
        client = self.reward_model_client
        if client is None:
            raise RuntimeError("Model reward scoring requires a configured RM")
        started = time.perf_counter()
        self.rollout_engine.prepare_for_training()
        self.residency.release()
        wake_started = time.perf_counter()
        state = coordinator_call("reward model wake", client.prepare)
        client.state = state
        wake_seconds = time.perf_counter() - wake_started
        primary_error = None
        try:
            scored = score_model_batch(
                prompts, rollout, scorer=self.reward_model_scorer, client=client,
                scorer_id=self.reward_model_scorer_id,
                tp_rank=int(self.parallel_dims.tp_rank), tp_size=int(self.parallel_dims.tp_size),
                tp_group=(self.parallel_dims.device_mesh["tp"].get_group()
                          if self.parallel_dims.tp_size > 1 else None),
                max_concurrency=int(self.reward_model_config.get("max_concurrency", 16)),
            )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                sleep_started = time.perf_counter()
                state = coordinator_call("reward model sleep", client.sleep)
                client.state = state
                sleep_seconds = time.perf_counter() - sleep_started
            except Exception:  # pylint: disable=broad-exception-caught
                # Keep the original scoring failure while logging any cleanup failure.
                if primary_error is None:
                    raise
                logger.exception("RM sleep failed while handling the original scoring error")
        if client.state != "sleeping":
            raise RuntimeError("Reward model must sleep before training continues")
        scored.metadata.update({
            "reward_seconds": time.perf_counter() - started,
            "reward_wake_seconds": wake_seconds,
            "reward_sleep_seconds": sleep_seconds,
        })
        if evaluation:
            self.rollout_engine.prepare_for_rollout()
        return scored
