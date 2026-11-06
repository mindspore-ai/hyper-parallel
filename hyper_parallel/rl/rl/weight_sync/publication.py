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
"""Training-to-rollout publication boundary."""
from typing import Any
from rl.weight_sync.sync import PolicySnapshot
from hyper_parallel import hsdp_sync_stream

class PolicyPublication:
    """Publish Actor parameters while enforcing training and rollout residency."""

    def __init__(self, *, actor: Any = None, model_registration: Any = None, rollout_engine: Any = None, residency: Any = None, reward_model_client: Any = None) -> None:
        """Bind explicit dependencies for this lifecycle boundary."""
        self.actor = actor
        self.model_registration = model_registration
        self.rollout_engine = rollout_engine
        self.residency = residency
        self.reward_model_client = reward_model_client

    def publish(self, next_step: int) -> None:
        """Transfer the updated Actor and restore rollout residency."""
        if getattr(self, "reward_model_client", None) is not None and self.reward_model_client.state != "sleeping":
            raise RuntimeError("Reward model must be sleeping before Actor publication")
        hsdp_sync_stream()
        self.residency.reshard_model(self.actor.actor_model)
        self.residency.release()
        self.rollout_engine.update_weights(
            PolicySnapshot(
                version=next_step,
                model_name=self.model_registration.name,
                payload=self.actor.actor_model,
            )
        )
        # Publication can refill the allocator with gathered weights and IPC buffers.
        # Receivers acknowledge synchronized copies before we release cache and wake KV memory.
        self.residency.release()
        self.residency.move(offload=True)
        self.rollout_engine.prepare_for_rollout()
