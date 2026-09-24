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
"""Actor-to-rollout policy publication and synchronization lifecycle."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import torch.distributed as dist

from rl.roles.weight_sync.vllm_client import (
    VLLMWeightSyncClientMixin,
    committed_policy_version,
)


@dataclass(frozen=True)
class PolicySnapshot:
    """One immutable publication of policy Actor weights."""
    version: int
    model_name: str
    payload: Any

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("PolicySnapshot version must be non-negative")
        if not self.model_name:
            raise ValueError("PolicySnapshot model_name must be non-empty")


def synchronize_error(local_error: Optional[Exception], operation: str) -> None:
    """Make every training rank observe one local synchronization failure."""
    try:
        world_size = dist.get_world_size()
    except (RuntimeError, ValueError):
        world_size = 1
    if world_size <= 1:
        if local_error is not None:
            raise local_error
        return
    errors: list[Optional[str]] = [None] * world_size
    dist.all_gather_object(
        errors,
        None if local_error is None else str(local_error),
    )
    if any(error is not None for error in errors):
        raise RuntimeError(f"vLLM {operation} failed on at least one rank: {errors}")


def synchronized_call(operation: str, callback: Callable[[], Any]) -> Any:
    """Run one local operation and propagate its failure to every training rank."""
    result = None
    local_error = None
    try:
        result = callback()
    except Exception as error:  # pylint: disable=W0718
        local_error = error
    synchronize_error(local_error, operation)
    return result


def coordinator_call(operation: str, callback: Callable[[], Any]) -> Any:
    """Run one coordinator operation and return its result on every rank."""
    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    except (RuntimeError, ValueError):
        world_size = 1
        rank = 0
    if world_size <= 1:
        return callback()
    result = None
    local_error = None
    if rank == 0:
        try:
            result = callback()
        except Exception as error:  # pylint: disable=W0718
            local_error = error
    synchronize_error(local_error, operation)
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, result if rank == 0 else None)
    return gathered[0]


class ActorRolloutWeightSync:
    """Move policy Actor weights into rollout and publish versions atomically."""

    def __init__(
        self,
        model_name: str,
        deployment: str,
        client_provider: Callable[[], Any],
        weight_transfer: Optional[Any],
    ) -> None:
        """Initialize one controller-owned policy publication transaction."""
        self._model_name = model_name
        self._deployment = deployment
        self._control_call = coordinator_call
        self._client_provider = client_provider
        self._weight_transfer = weight_transfer
        self._policy_version = 0
        self._pending_policy_version: Optional[int] = None
        self._phase = "rollout"

    @property
    def policy_version(self) -> int:
        """Return the policy version admitted for generation."""
        return self._policy_version

    @property
    def configured_strategy(self) -> Optional[str]:
        """Return the effective configured weight-transfer strategy."""
        return None if self._weight_transfer is None else self._weight_transfer.configured_strategy

    @property
    def last_strategy(self) -> Optional[str]:
        """Return the strategy that completed the latest publication."""
        return None if self._weight_transfer is None else self._weight_transfer.last_strategy

    @property
    def streaming_stats(self) -> Optional[Mapping[str, Any]]:
        """Return bounded streaming counters from the latest publication."""
        stats = None if self._weight_transfer is None else self._weight_transfer.last_streaming_stats
        return None if stats is None else dict(stats)

    @property
    def phase(self) -> str:
        """Return the current residency and publication phase."""
        return self._phase

    def generation_version(self, client: Any) -> int:
        """Verify and return the worker version serving the next request."""
        if self._phase != "rollout" or self._pending_policy_version is not None:
            raise RuntimeError(
                "Cannot generate from an unpublished rollout policy: "
                f"phase={self._phase!r}, pending={self._pending_policy_version}"
            )
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise RuntimeError("Generation requires the owned vLLM HTTP client")
        worker_version = self._control_call(
            "generation rollout version",
            lambda: committed_policy_version(client),
        )
        if worker_version != self._policy_version:
            raise RuntimeError(
                "vLLM worker policy version differs from the published version: "
                f"expected={self._policy_version}, actual={worker_version}"
            )
        return worker_version

    @staticmethod
    def _server_owner_call(
        operation: str,
        callback: Callable[[], Any],
    ) -> Any:
        """Run one mutating server operation exactly once on the coordinator."""
        return coordinator_call(operation, callback)

    def prepare_for_training(self) -> None:
        """Sleep colocated rollout before policy Actor training starts."""
        if self._deployment != "colocated":
            return
        if self._phase != "rollout":
            raise RuntimeError(f"Cannot prepare vLLM for training from phase {self._phase!r}")
        client = synchronized_call("server startup", self._client_provider)
        if client is None:
            raise RuntimeError("vLLM server startup failed without a synchronized error")
        worker_version = self._control_call(
            "initial rollout version",
            lambda: committed_policy_version(client),
        )
        if worker_version != self._policy_version:
            raise RuntimeError(
                "Initial rollout worker version differs from controller: "
                f"expected={self._policy_version}, actual={worker_version}"
            )
        self._server_owner_call(
            "sleep before training",
            lambda: client.sleep(level=1, mode="wait"),
        )
        def verify_sleeping() -> None:
            """Require every rank's connected replica to be sleeping."""
            if not client.is_sleeping():
                raise RuntimeError("vLLM did not enter sleep mode before training")
        synchronized_call("sleep residency check", verify_sleeping)
        self._phase = "training"

    def update_weights(self, snapshot: PolicySnapshot) -> None:
        """Transfer a newer Actor snapshot and stage its rollout version."""
        if snapshot.model_name != self._model_name:
            raise ValueError(
                f"Policy snapshot model mismatch: expected={self._model_name}, "
                f"received={snapshot.model_name}"
            )
        if snapshot.version <= self._policy_version:
            raise ValueError(
                "Policy snapshot version must increase: "
                f"current={self._policy_version}, received={snapshot.version}"
            )
        if self._weight_transfer is None:
            raise NotImplementedError(
                "vLLM iterative training requires a concrete WeightPublisher; "
                "the adapter will not acknowledge a new policy version without loading it"
            )
        if self._deployment == "colocated" and self._phase != "training":
            raise RuntimeError(f"Cannot publish to colocated vLLM from phase {self._phase!r}")
        client = synchronized_call("server startup", self._client_provider)
        if client is None:
            raise RuntimeError("vLLM server startup failed without a synchronized error")
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise RuntimeError("Weight publication requires the owned vLLM HTTP client")
        self._pending_policy_version = snapshot.version
        self._phase = "publishing"
        if self._deployment == "colocated":
            self._server_owner_call("wake weights for publication", lambda: client.wake_up(("weights",)))
        self._weight_transfer.publish(client, snapshot)
        if self._deployment == "colocated":
            return
        self._resume_rollout(client, check_residency=False)
        self._policy_version = snapshot.version
        self._pending_policy_version = None
        self._phase = "rollout"

    def prepare_for_rollout(self) -> None:
        """Wake rollout memory and atomically expose a transferred policy."""
        if self._deployment != "colocated":
            return
        if self._phase not in ("training", "publishing"):
            raise RuntimeError(f"Cannot prepare vLLM for rollout from phase {self._phase!r}")
        if self._phase == "publishing" and self._pending_policy_version is None:
            raise RuntimeError("Colocated publication completed without a pending policy version")
        client = synchronized_call("server startup", self._client_provider)
        if client is None:
            raise RuntimeError("vLLM server startup failed without a synchronized error")
        is_publishing = self._phase == "publishing"
        tags = ("kv_cache",) if is_publishing else ("weights", "kv_cache")
        self._server_owner_call(
            "wake before rollout",
            lambda: client.wake_up(tags),
        )
        if is_publishing:
            self._server_owner_call(
                "post-publication cache reset",
                client.pause,
            )
            def verify_post_publication_pause() -> None:
                """Require admission to remain closed after publication."""
                if not client.is_paused():
                    raise RuntimeError(
                        "vLLM did not remain paused after the publication cache reset"
                    )
            _ = synchronized_call(
                "post-publication pause check",
                verify_post_publication_pause,
            )
            worker_version = self._control_call(
                "pending rollout version",
                lambda: committed_policy_version(client),
            )
            if worker_version != self._pending_policy_version:
                raise RuntimeError(
                    "vLLM worker version differs from the pending policy: "
                    f"expected={self._pending_policy_version}, actual={worker_version}"
                )
        self._resume_rollout(client, check_residency=True)
        if self._pending_policy_version is not None:
            self._policy_version = self._pending_policy_version
        self._pending_policy_version = None
        self._phase = "rollout"

    def _resume_rollout(self, client: VLLMWeightSyncClientMixin, *, check_residency: bool) -> None:
        """Open admission and let any failure terminate the current run."""
        self._server_owner_call("resume rollout admission", client.resume)
        if check_residency:
            def verify_ready() -> None:
                """Require scheduler and memory readiness before advancing the visible version."""
                if client.is_paused() or client.is_sleeping():
                    raise RuntimeError("vLLM remained paused or sleeping after resume")
            self._control_call("rollout residency check", verify_ready)

    def close(self) -> None:
        """Release resources owned by the selected transfer implementation."""
        if self._weight_transfer is not None:
            self._weight_transfer.close()
__all__ = [
    "ActorRolloutWeightSync",
    "PolicySnapshot",
    "coordinator_call",
    "synchronized_call",
    "synchronize_error",
]
