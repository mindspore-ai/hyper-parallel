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
"""Hyper-Parallel native co-located generation adapter."""

import time
from typing import Any, Mapping

from rl.roles.model import ModelRegistration
from rl.roles.rollout.base import GenerationRequest, GenerationResult
from rl.roles.rollout.registry import ROLLOUT_ENGINES
from rl.roles.weight_sync import PolicySnapshot
from hyper_parallel import get_platform
from hyper_parallel.infer.utils import GenerationConfig

platform = get_platform()


class HyperGenerationEngine:
    """Run co-located generation through Hyper-Parallel native inference."""

    name = "hyper"

    def __init__(self, actor: Any) -> None:
        """Initialize generation around the shared training actor."""
        if actor is None:
            raise ValueError("The Hyper rollout engine requires the co-located actor")
        self.actor = actor
        self.policy_version = 0

    @staticmethod
    def _synchronize_device() -> None:
        handle = platform.get_device_handle(platform.device_type())
        synchronize = getattr(handle, "synchronize", None)
        if synchronize is not None:
            synchronize()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate fixed-iteration responses and optional old log-probabilities."""
        settings = request.settings
        generation_config = GenerationConfig(
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            do_sample=settings.do_sample,
            # Hyper's fixed-iteration decode keeps every FSDP rank collective-aligned.
            eos_token_id=None,
            pad_token_id=settings.pad_token_id,
            use_cache=False,
        )
        prompt_length = int(request.input_ids.shape[1])
        was_training = self.actor.training
        self.actor.eval()
        rollout_log_probs = None
        try:
            self._synchronize_device()
            started = time.perf_counter()
            with platform.no_grad():
                sequences = self.actor.generate(
                    input_ids=request.input_ids,
                    attention_mask=request.attention_mask,
                    generation_config=generation_config,
                )
                if settings.collect_log_probs:
                    generated_tokens = sequences.shape[1] - prompt_length
                    full_attention_mask = platform.cat(
                        (
                            request.attention_mask,
                            request.attention_mask.new_ones(
                                (sequences.shape[0], generated_tokens)
                            ),
                        ),
                        dim=-1,
                    )
                    rollout_log_probs = self.actor.response_log_probs(
                        sequences,
                        prompt_length,
                        full_attention_mask,
                    ).detach()
            self._synchronize_device()
            elapsed = time.perf_counter() - started
        finally:
            self.actor.train(was_training)
        return GenerationResult(sequences, rollout_log_probs, elapsed)

    def update_weights(self, snapshot: PolicySnapshot) -> None:
        """Publish a newer version of the already shared actor weights."""
        if snapshot.payload is not self.actor:
            raise ValueError("Co-located Hyper rollout shares the actor and needs no weight copy")
        if snapshot.version <= self.policy_version:
            raise ValueError(
                f"Policy snapshot version must increase: current={self.policy_version}, "
                f"received={snapshot.version}"
            )
        self.policy_version = snapshot.version


@ROLLOUT_ENGINES.register("hyper")
def build_hyper_engine(
    config: Mapping[str, Any],
    model: ModelRegistration,
    actor: Any,
) -> HyperGenerationEngine:
    """Build the registered co-located Hyper generation engine."""
    del config, model
    return HyperGenerationEngine(actor)
