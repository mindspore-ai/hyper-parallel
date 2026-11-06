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
"""Required role inference and immutable RL target preparation."""
import functools
import time
from dataclasses import replace
from typing import Any, Callable, Optional
import torch
from rl.data.contracts import ExperienceBatch
from rl.data.experience import get_bootstrap_values
from rl.utils.monitoring.metrics import summarize_training_diagnostics


class TargetPreparer:
    """Own only role dependencies needed to construct an update's fixed targets."""

    def __init__(self, *, actor: Any, reference_actor: Any, critic: Any,
                 algorithm: Any, experience_preparer: Any,
                 run_synchronized: Callable, balance_dp_tokens: bool = False,
                 old_log_prob_source: str = "rollout", has_teacher: bool = False) -> None:
        """Bind inference roles and explicit target preparation choices."""
        self.actor = actor
        self.reference_actor = reference_actor
        self.critic = critic
        self.algorithm = algorithm
        self.experience_preparer = experience_preparer
        self.run_synchronized = run_synchronized
        self.balance_dp_tokens = balance_dp_tokens
        self.old_log_prob_source = old_log_prob_source
        self.has_teacher = has_teacher

    def prepare(
        self,
        rollout: ExperienceBatch,
        collect_diagnostics: bool,
        timings: dict[str, float],
    ) -> tuple[ExperienceBatch, dict[str, float]]:
        """Run required role inference and build immutable training targets."""
        balance = self.balance_dp_tokens
        if balance:
            stage_started = time.perf_counter()
            rollout = self.run_synchronized(
                "GRPO targets before DP balance", lambda: self.experience_preparer.prepare_targets(rollout),
            )
            timings["adv"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            rollout = self.actor.balance_rollout(rollout)
            timings["balance"] = time.perf_counter() - stage_started
        recompute_anchor = self.old_log_prob_source == "actor"
        has_teacher = self.has_teacher
        actor_log_probs = self._compute_actor_log_probs(
            rollout, collect_diagnostics or has_teacher or recompute_anchor, timings,
        )
        reference_log_probs, values, bootstrap = self._compute_role_targets(rollout, timings)
        if has_teacher or recompute_anchor:
            if has_teacher and rollout.rollout_log_probs is None:
                raise ValueError("OPD requires independently retained rollout_log_probs")
            rollout = replace(rollout, old_log_probs=actor_log_probs.detach())
        stage_started = time.perf_counter()
        prepare = functools.partial(
            self.experience_preparer.attach_role_outputs if balance else self.experience_preparer.prepare, rollout,
            reference_log_probs=reference_log_probs, values=values, bootstrap_values=bootstrap,
        )
        experience = (self.run_synchronized("GRPO target preparation", prepare)
                      if self.algorithm.name == "grpo" else prepare())
        timings["adv"] = timings.get("adv", 0.0) + time.perf_counter() - stage_started
        diagnostic_metrics = (
            summarize_training_diagnostics(
                experience,
                None if recompute_anchor and rollout.rollout_log_probs is None else actor_log_probs,
            )
            if collect_diagnostics or has_teacher else {}
        )
        return experience, diagnostic_metrics


    def _compute_actor_log_probs(self, rollout: ExperienceBatch, required: bool,
                                 timings: dict[str, float]) -> Optional[torch.Tensor]:
        """Compute the pre-update Actor anchor only when a configured consumer needs it."""
        if not required:
            return None
        stage_started = time.perf_counter()
        actor_log_probs = self.run_synchronized(
            "pre-update Actor log-probabilities",
            lambda: self.actor.compute_log_probs(rollout),
        )
        if actor_log_probs is None:
            raise RuntimeError(
                "Pre-update Actor log-probability computation failed without a synchronized error"
            )
        timings["old_log_prob"] = time.perf_counter() - stage_started
        return actor_log_probs


    def _compute_role_targets(self, rollout: ExperienceBatch, timings: dict[str, float]) -> tuple[Any, Any, Any]:
        """Compute required Reference probabilities and Critic bootstrap values."""
        requirements = self.algorithm.requirements.data
        reference_log_probs = None
        if requirements.reference_log_probs:
            if self.reference_actor is None:
                raise RuntimeError(
                    f"Algorithm '{self.algorithm.name}' requires a reference model"
                )
            stage_started = time.perf_counter()
            reference_log_probs = self.reference_actor.compute_log_probs(rollout)
            timings["ref"] = time.perf_counter() - stage_started
        values = None
        bootstrap = None
        if requirements.values:
            if self.critic is None:
                raise RuntimeError(f"Algorithm '{self.algorithm.name}' requires a Critic")
            stage_started = time.perf_counter()
            full_values = self.critic.compute_values(rollout, include_last=True)
            bootstrap = self.run_synchronized(
                "PPO bootstrap context", lambda: get_bootstrap_values(rollout, full_values),
            )
            values = full_values[:, :-1]
            timings["values"] = time.perf_counter() - stage_started
        return reference_log_probs, values, bootstrap
