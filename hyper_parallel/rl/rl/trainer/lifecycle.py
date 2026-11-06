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
"""Run resource ownership and failure cleanup."""
from typing import Any
from rl.utils.process_cleanup import cleanup_processes
from rl.utils.failure_reporting import report_failure

import logging

logger = logging.getLogger(__name__)

class RunLifecycle:
    """Close owned services and distributed resources, preserving primary failures."""

    def __init__(self, *, _tracker: Any = None, rollout_manager: Any = None, rollout_engine: Any = None, reward_model_client: Any = None, _router_replay: Any = None, _runtime_started: Any = None, state: Any = None) -> None:
        """Bind explicit dependencies for this lifecycle boundary."""
        self._tracker = _tracker
        self.rollout_manager = rollout_manager
        self.rollout_engine = rollout_engine
        self.reward_model_client = reward_model_client
        self._router_replay = _router_replay
        self._runtime_started = _runtime_started
        self.state = state
        self._rollout_closed_for_final_checkpoint = False

    def fail(self, error: BaseException, stage: str) -> None:
        """Expose the original failure before entering service or collective teardown."""
        report_failure(error, stage=stage, step=self.state.global_step)
        try:
            self.close()
        # Preserve the original training failure even if teardown is interrupted.
        except BaseException:  # pylint: disable=broad-exception-caught
            logger.exception("Secondary error during RL failure cleanup")

    def close(self) -> None:
        """Close owned services and reset lifecycle state even when cleanup raises."""
        try:
            reward_model = getattr(self, "reward_model_client", None)
            if reward_model is not None:
                try:
                    reward_model.close()
                except Exception as error:  # pylint: disable=broad-exception-caught
                    logger.warning("Reward model cleanup failed: %s", error)
            cleanup_processes(
                self._tracker,
                getattr(self, "rollout_manager", None),
                (None if getattr(self, "_rollout_closed_for_final_checkpoint", False)
                 else getattr(self, "rollout_engine", None)),
                self._runtime_started,
            )
        finally:
            replay = getattr(self, "_router_replay", None)
            if replay is not None:
                replay.remove()
                self._router_replay = None
            self._tracker = None
            self._runtime_started = False
