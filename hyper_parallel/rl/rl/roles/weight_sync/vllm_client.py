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
"""HTTP control operations for the shared vLLM rollout endpoint."""
from typing import Any, Mapping, Optional
from urllib import parse as urllib_parse

import torch.distributed as dist

KEEP_SCHEDULER_PAUSED_TAG = "_hyper_keep_scheduler_paused"


class VLLMWeightSyncClientMixin:
    """Weight-control requests shared by the rollout-side vLLM HTTP client."""
    _base_url: str

    @property
    def is_server_owner(self) -> bool:
        """Return whether this trainer rank owns the connected server process."""
        return True

    def _request(
        self,
        method: str,
        route: str,
        payload: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        base_url: Optional[str] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def base_url(self) -> str:
        """Return this rank-local rollout endpoint."""
        return self._base_url

    def get_world_size(self, base_url: Optional[str] = None) -> int:
        """Return the number of vLLM inference workers."""
        return int(
            self._request("GET", "get_world_size", base_url=base_url)["world_size"]
        )

    def pause(self) -> None:
        """Pause generation and invalidate request caches before transfer."""
        status = self._request("POST", "pause?mode=abort&clear_cache=true").get("status")
        if status != "paused":
            raise RuntimeError(f"vLLM /pause returned invalid status {status!r}")

    def is_paused(self) -> bool:
        """Return whether generation admission is closed."""
        value = self._request("GET", "is_paused").get("is_paused")
        if not isinstance(value, bool):
            raise RuntimeError("vLLM /is_paused did not return a boolean state")
        return value

    def sleep(self, level: int = 1, mode: str = "wait") -> None:
        """Drain generation and release tagged vLLM device memory."""
        self._request("POST", f"sleep?level={level}&mode={mode}")

    def wake_up(self, tags: tuple[str, ...]) -> None:
        """Restore executor memory without letting EngineCore resume scheduling."""
        query = urllib_parse.urlencode(
            [("tags", tag) for tag in (*tags, KEEP_SCHEDULER_PAUSED_TAG)]
        )
        self._request("POST", f"wake_up?{query}")

    def is_sleeping(self) -> bool:
        """Return the server's combined scheduler/device sleep state."""
        value = self._request("GET", "is_sleeping").get("is_sleeping")
        if not isinstance(value, bool):
            raise RuntimeError("vLLM /is_sleeping did not return a boolean state")
        return value

    def start_weight_update(self) -> None:
        """Start loading checkpoint-format Actor weights."""
        self._request("POST", "start_weight_update", {"is_checkpoint_format": True})

    def finish_weight_update(self) -> None:
        """Commit one completed Actor-to-rollout weight transfer."""
        self._request("POST", "finish_weight_update")

    def collective_rpc(
        self,
        method: str,
        kwargs: Optional[Mapping[str, Any]] = None,
        base_url: Optional[str] = None,
    ) -> list[Any]:
        """Invoke one registered worker method on every inference rank."""
        response = self._request(
            "POST",
            "collective_rpc",
            {"method": method, "kwargs": dict(kwargs or {})},
            base_url=base_url,
        )
        results = response.get("results")
        if not isinstance(results, list):
            raise RuntimeError("vLLM collective RPC returned invalid worker results")
        return results

    def resume(self) -> None:
        """Resume rollout admission after a completed transfer."""
        status = self._request("POST", "resume").get("status")
        if status != "resumed":
            raise RuntimeError(f"vLLM /resume returned invalid status {status!r}")


def shared_endpoint(client: VLLMWeightSyncClientMixin) -> str:
    """Require every Trainer rank to use the same nonempty endpoint."""
    endpoints = [None] * dist.get_world_size()
    dist.all_gather_object(endpoints, client.base_url)
    if not endpoints or not endpoints[0] or any(value != endpoints[0] for value in endpoints):
        raise RuntimeError(f"Weight synchronization requires one shared rollout endpoint: {endpoints}")
    return str(endpoints[0])


def direct_reshard_workers(
    client: VLLMWeightSyncClientMixin,
    *,
    data_parallel_size: int,
    tensor_parallel_size: int,
) -> list[Mapping[str, Any]]:
    """Query and reduce physical worker layouts to one representative DP replica."""
    expected_world_size = data_parallel_size * tensor_parallel_size
    actual_world_size = client.get_world_size()
    if actual_world_size != expected_world_size:
        raise RuntimeError(
            "Direct reshard rollout world size differs from configured DP x TP: "
            f"expected={expected_world_size}, actual={actual_world_size}"
        )
    workers = client.collective_rpc("get_direct_reshard_layout")
    if not workers or not all(isinstance(worker, Mapping) for worker in workers):
        raise RuntimeError(
            f"Direct reshard rollout returned invalid layouts: {workers}"
        )
    by_coordinate = {}
    for worker in workers:
        coordinate = (int(worker["dp_rank"]), int(worker["tp_rank"]))
        if (
            coordinate in by_coordinate
            or int(worker["dp_size"]) not in (1, data_parallel_size)
            or int(worker["tp_size"]) != tensor_parallel_size
            or not 0 <= coordinate[0] < data_parallel_size
            or not 0 <= coordinate[1] < tensor_parallel_size
        ):
            raise RuntimeError(f"Direct reshard worker has invalid topology: {worker}")
        by_coordinate[coordinate] = worker
    dp_ranks = sorted({coordinate[0] for coordinate in by_coordinate})
    expected_tp_ranks = set(range(tensor_parallel_size))
    for dp_rank in dp_ranks:
        actual_tp_ranks = {
            tp_rank
            for worker_dp_rank, tp_rank in by_coordinate
            if worker_dp_rank == dp_rank
        }
        if actual_tp_ranks != expected_tp_ranks:
            raise RuntimeError(
                f"Direct reshard DP{dp_rank} returned incomplete TP layouts"
            )
    representative_dp_rank = dp_ranks[0]
    representatives = []
    for tp_rank in range(tensor_parallel_size):
        representative = by_coordinate.get((representative_dp_rank, tp_rank))
        tensors = representative.get("tensors")
        if any(
            by_coordinate.get((dp_rank, tp_rank)).get("tensors") != tensors
            for dp_rank in dp_ranks[1:]
        ):
            raise RuntimeError(
                f"Direct reshard layouts differ across DP replicas for TP{tp_rank}"
            )
        representatives.append(representative)
    return representatives


def committed_policy_version(client: VLLMWeightSyncClientMixin) -> int:
    """Read a single committed worker version without inspecting model tensors."""
    results = client.collective_rpc("get_policy_version")
    if not results or not all(isinstance(result, Mapping) and "version" in result for result in results):
        raise RuntimeError(f"Invalid committed policy versions: {results}")
    versions = {int(result["version"]) for result in results}
    if len(versions) != 1:
        raise RuntimeError(f"Rollout worker versions differ: {sorted(versions)}")
    return versions.pop()
