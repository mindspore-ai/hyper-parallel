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
"""Actor-to-rollout weight preparation, transport, and verification."""
from concurrent.futures import ThreadPoolExecutor
import base64
import json
import logging
import os
from pathlib import Path
import pickle
import resource
from typing import Any, Mapping, Optional, Protocol

from torch.distributed.checkpoint.state_dict import StateDictOptions  # pylint: disable=forbidden-backend-import
from torch.multiprocessing.reductions import reduce_tensor  # pylint: disable=forbidden-backend-import

from rl.roles.model import VLLMModelRegistration
from rl.roles.weight_sync.config import validate_weight_sync_support
from rl.roles.weight_sync.hccl import BroadcastDirectReshardHCCLTransport
from rl.roles.weight_sync.layout import (
    DirectReshardPlan,
    build_direct_reshard_plan,
    resolve_destination_layouts,
    resolve_physical_worker_topology,
    resolve_source_layouts,
)
from rl.roles.weight_sync.model_adapter import (
    aggregate_direct_content_identity,
    build_model_weight_adapter,
    direct_fragment_record,
)
from rl.roles.weight_sync.sync import (
    PolicySnapshot,
    VLLMWeightSyncClientMixin,
    coordinator_call,
    is_policy_fingerprint_weight,
    policy_weight_fingerprint,
    synchronized_call,
    synchronize_error,
)
from rl.roles.weight_sync.streaming_full_gather import (
    MaterializedStreamingContribution,
    StreamingBucketAck,
    StreamingContentIdentityAccumulator,
    StreamingFullGatherExecutor,
    StreamingFullGatherPlan,
    StreamingGatherBucket,
    StreamingMaterializedBucket,
    assemble_streaming_fragment,
    build_streaming_full_gather_plan,
    extract_streaming_contributions,
    pack_streaming_bucket,
)
from rl.roles.weight_sync.tensor_ops import local_tensor, pack_direct_bucket

from hyper_parallel import get_platform
platform = get_platform()
logger = logging.getLogger(__name__)
_TEST_WEIGHT_SYNC_FAULT_ENV = "HYPER_RL_TEST_WEIGHT_SYNC_FAULT"
_SUPPORTED_TEST_WEIGHT_SYNC_FAULTS = frozenset(
    ("direct_receive_once", "direct_bucket_once", "streaming_bucket_once")
)


def _current_process_rss_bytes() -> int:
    """Return current Linux resident memory, falling back to the process peak."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _current_trainer_memory_stats() -> dict[str, int]:
    """Sample current Trainer device allocation and host residency."""
    handle = platform.get_device_handle(platform.device_type())
    allocated_fn = getattr(handle, "memory_allocated", None)
    reserved_fn = getattr(handle, "memory_reserved", None)
    return {
        "allocated_bytes": int(allocated_fn()) if allocated_fn else 0,
        "reserved_bytes": int(reserved_fn()) if reserved_fn else 0,
        "host_rss_bytes": _current_process_rss_bytes(),
    }


def _memory_acceptance_fields(
    samples: list[Mapping[str, int]],
    *,
    prefix: str,
) -> dict[str, int]:
    """Summarize baseline, observed peak, and post-release current memory."""
    if not samples:
        return {}
    result = {}
    for name in ("allocated_bytes", "reserved_bytes", "host_rss_bytes"):
        result[f"{prefix}_baseline_{name}"] = int(samples[0][name])
        result[f"{prefix}_peak_current_{name}"] = max(
            int(sample[name]) for sample in samples
        )
        result[f"{prefix}_post_release_{name}"] = int(samples[-1][name])
    return result


def _worker_memory_acceptance_fields(
    baseline: list[Mapping[str, Any]],
    post_release: list[Mapping[str, Any]],
) -> dict[str, int]:
    """Summarize rollout-worker current memory before and after publication."""
    if not baseline or not post_release:
        return {}
    result = {}
    for name in ("current_memory_allocated_bytes", "current_memory_reserved_bytes", "current_host_rss_bytes"):
        result[f"rollout_baseline_{name.removeprefix('current_')}"] = max(
            int(item.get(name, 0)) for item in baseline
        )
        result[f"rollout_post_release_{name.removeprefix('current_')}"] = max(
            int(item.get(name, 0)) for item in post_release
        )
    return result


def _streaming_memory_summary(
    trainer_samples: list[Mapping[str, int]],
    worker_before: list[Mapping[str, Any]],
    worker_after: list[Mapping[str, Any]],
) -> dict[str, int]:
    """Keep lifetime peaks and per-publication baselines in one metrics mapping."""
    handle = platform.get_device_handle(platform.device_type())
    return {
        **_memory_acceptance_fields(trainer_samples, prefix="trainer"),
        **_worker_memory_acceptance_fields(worker_before, worker_after),
        "trainer_max_memory_allocated_bytes": int(handle.max_memory_allocated()),
        "trainer_max_memory_reserved_bytes": int(handle.max_memory_reserved()),
        "trainer_host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        **{
            f"rollout_{name}": max(int(item[name]) for item in worker_after)
            for name in ("max_memory_allocated_bytes", "max_memory_reserved_bytes", "host_max_rss_bytes")
        },
    }


class _WeightSyncTestFaultInjector:
    """Inject explicit one-shot acceptance faults without changing normal runs."""

    def __init__(self) -> None:
        """Initialize WeightSyncTestFaultInjector state."""
        self._triggered: set[str] = set()

    @staticmethod
    def configured() -> frozenset[str]:
        """Parse the opt-in P7 acceptance fault list."""
        raw = os.environ.get(_TEST_WEIGHT_SYNC_FAULT_ENV, "").strip()
        if not raw or raw == "none":
            return frozenset()
        configured = frozenset(item.strip() for item in raw.split(",") if item.strip())
        unsupported = sorted(configured - _SUPPORTED_TEST_WEIGHT_SYNC_FAULTS)
        if unsupported:
            raise ValueError(
                f"{_TEST_WEIGHT_SYNC_FAULT_ENV} has unsupported faults: {unsupported}"
            )
        return configured

    def trigger(self, fault: str, *, context: str = "") -> None:
        """Raise once on rank zero so synchronized callers fail together."""
        if fault not in self.configured() or fault in self._triggered:
            return
        if platform.get_rank() != 0:
            return
        self._triggered.add(fault)
        raise RuntimeError(f"Injected P7.3 weight-sync fault: {fault} {context}".rstrip())


def _tensor_ipc_rebuild_args(tensor: Any) -> tuple[Any, ...]:
    """Share tensor storage and return Torch multiprocessing rebuild arguments."""
    _, rebuild_args = reduce_tensor(tensor)
    return rebuild_args


def _write_rollout_parameter_manifest(
    client: VLLMWeightSyncClientMixin,
    *,
    strategy: str,
    policy_version: int,
    data_parallel_size: int = 1,
) -> None:
    """Persist exact rank-local rollout hashes for an explicit verification run."""
    output_dir = os.environ.get("HYPER_RL_WEIGHT_MANIFEST_DIR")
    if not output_dir:
        return
    if not bool(getattr(client, "is_server_owner", True)):
        return
    oracle_run_id = os.environ.get("HYPER_RL_WEIGHT_ORACLE_RUN_ID")
    if not oracle_run_id:
        raise RuntimeError("HYPER_RL_WEIGHT_ORACLE_RUN_ID must identify the verification run")
    oracle_dir = (
        os.environ.get("HYPER_RL_WEIGHT_MANIFEST_ORACLE_DIR")
        if strategy != "full_gather"
        else None
    )
    expected_dir = os.environ.get("HYPER_RL_WEIGHT_EXPECTED_MANIFEST_DIR")
    results = client.collective_rpc(
        "write_parameter_manifest",
        {
            "output_dir": output_dir,
            "strategy": strategy,
            "policy_version": int(policy_version),
            "rollout_replica_rank": int(platform.get_rank()),
            "expected_data_parallel_size": int(data_parallel_size),
            "oracle_run_id": oracle_run_id,
            "oracle_dir": oracle_dir,
            "oracle_strategy": "full_gather" if oracle_dir else None,
            "expected_dir": expected_dir,
        },
    )
    if not results or not all(
        isinstance(result, Mapping) and bool(result.get("written"))
        for result in results
    ):
        raise RuntimeError(f"Invalid rollout parameter manifest results: {results}")


def _local_state_dict(payload: Any, *, operation: str) -> dict[str, Any]:
    """Extract FSDP-local model shards and validate their tensor-only contract."""
    state_dict = (
        dict(payload)
        if isinstance(payload, Mapping)
        else platform.get_model_state_dict(
            payload,
            options=StateDictOptions(
                full_state_dict=False,
                cpu_offload=False,
            ),
        )
    )
    invalid = next(
        ((name, value) for name, value in state_dict.items() if not platform.is_tensor(value)),
        None,
    )
    if invalid is not None:
        name, value = invalid
        state_dict.clear()
        raise ValueError(
            f"{operation} state entry {name!r} must be a tensor, got {type(value)!r}"
        )
    return state_dict


def _gather_selected_state_dict(
    state_dict: Mapping[str, Any],
    *,
    cpu_offload: bool,
) -> dict[str, Any]:
    """Materialize only the selected DTensor values using master semantics."""
    is_rank_zero = platform.get_rank() == 0
    gathered = {}
    for name, value in state_dict.items():
        full_tensor = getattr(value, "full_tensor", None)
        if callable(full_tensor):
            value = full_tensor()
        if cpu_offload:
            if not is_rank_zero:
                continue
            value = value.to("cpu")
        gathered[name] = value
    return gathered


def _alias_tied_embeddings(
    state_dict: dict[str, Any],
    model: VLLMModelRegistration,
) -> dict[str, Any]:
    """Expose both tied checkpoint names without allocating another tensor."""
    if not model.model.tie_word_embeddings:
        return state_dict
    embedding_name = "model.embed_tokens.weight"
    lm_head_name = "lm_head.weight"
    if embedding_name in state_dict and lm_head_name not in state_dict:
        state_dict[lm_head_name] = state_dict[embedding_name]
    elif lm_head_name in state_dict and embedding_name not in state_dict:
        state_dict[embedding_name] = state_dict[lm_head_name]
    return state_dict


class WeightTransfer(Protocol):
    """Publish one policy Actor snapshot into an existing rollout model."""

    def publish(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Atomically publish one policy snapshot to rollout."""


class _RefitCompatibleTransfer:
    """Expose historical verbs as adapters to the canonical publication operation."""

    def publish(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Atomically publish one policy snapshot to rollout."""
        raise NotImplementedError

    def transfer(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Route the historical transfer verb to publication."""
        self.publish(client, snapshot)

    def refit(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Route the historical refit verb to publication."""
        self.publish(client, snapshot)


class DirectReshardHCCLWeightTransfer(_RefitCompatibleTransfer):
    """Broadcast FSDP-local fragments directly into rollout-owned shards."""

    def __init__(
        self,
        model: VLLMModelRegistration,
        *,
        bucket_size_bytes: int = 128 * 2**20,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
    ) -> None:
        """Store the fixed model contract and defer layout-plan construction."""
        if model.family not in ("qwen3", "qwen3_moe", "deepseek_v3"):
            raise ValueError(
                "Direct reshard supports Qwen3, Qwen3-MoE, and DeepSeek-V3 rollout models"
            )
        if bucket_size_bytes <= 0:
            raise ValueError("Direct reshard bucket_size_bytes must be positive")
        if data_parallel_size <= 0:
            raise ValueError("Direct reshard data_parallel_size must be positive")
        if tensor_parallel_size <= 0:
            raise ValueError("Direct reshard tensor_parallel_size must be positive")
        self._model = model
        self._adapter = build_model_weight_adapter(model)
        self._bucket_size_bytes = int(bucket_size_bytes)
        self._data_parallel_size = int(data_parallel_size)
        self._tensor_parallel_size = int(tensor_parallel_size)
        self._transport = BroadcastDirectReshardHCCLTransport(
            data_parallel_size=self._data_parallel_size,
            tensor_parallel_size=self._tensor_parallel_size,
        )
        self._plan: Optional[DirectReshardPlan] = None
        self._flatten_native_moe_dp_destinations = False
        self._parameter_names: Optional[frozenset[str]] = None
        self.last_policy_fingerprint: Optional[dict[str, Any]] = None
        self.configured_strategy = "direct_reshard"
        self.last_strategy: Optional[str] = None
        self.last_attempted_strategies: tuple[str, ...] = ()
        self.last_completed_strategy: Optional[str] = None
        self.fallback_count = 0
        self.direct_success_count = 0
        self._test_fault_injector = _WeightSyncTestFaultInjector()

    @staticmethod
    def _local_state_dict(payload: Any) -> dict[str, Any]:
        """Return DTensor local shards without any FSDP all-gather."""
        return _local_state_dict(
            payload,
            operation="vLLM direct reshard",
        )

    def _mapped_local_state_dict(self, payload: Any) -> dict[str, Any]:
        """Map Actor names and alias tied embeddings without copying their shard."""
        return _alias_tied_embeddings(
            self._adapter.map_local_state_dict(
                self._local_state_dict(payload),
            ),
            self._model,
        )

    def _query_destination_workers(
        self,
        client: VLLMWeightSyncClientMixin,
    ) -> list[Mapping[str, Any]]:
        """Query and collapse the DP-engine layouts returned by vLLM."""
        expected_world_size = self._data_parallel_size * self._tensor_parallel_size

        def query_workers() -> list[Any]:
            """Query every physical rollout worker after validating world size."""
            actual_world_size = client.get_world_size()
            if actual_world_size != expected_world_size:
                raise RuntimeError(
                    "Direct reshard rollout world size differs from configured DP x TP: "
                    f"expected={expected_world_size}, actual={actual_world_size}"
                )
            if self._flatten_native_moe_dp_destinations:
                prepared = client.collective_rpc("prepare_direct_reshard_layout")
                if not prepared or not all(
                    isinstance(result, Mapping) and bool(result.get("prepared"))
                    for result in prepared
                ):
                    raise RuntimeError(
                        "Native MoE workers did not restore executable "
                        f"FusedMoE layouts before direct planning: {prepared}"
                    )
            return client.collective_rpc("get_direct_reshard_layout")

        workers = coordinator_call("direct reshard rollout layout query", query_workers)
        if not isinstance(workers, list) or not workers or not all(
            isinstance(worker, Mapping) for worker in workers
        ):
            raise RuntimeError(
                f"Direct reshard rollout returned invalid layouts: {workers}"
            )
        by_identity: dict[tuple[int, int], Mapping[str, Any]] = {}
        for worker in workers:
            identity = (int(worker["dp_rank"]), int(worker["tp_rank"]))
            worker_dp_size = int(worker["dp_size"])
            # Non-MoE vLLM engines may expose engine-local DP size 1 while
            # retaining the deployment-global data_parallel_index.
            if worker_dp_size not in (1, self._data_parallel_size):
                raise RuntimeError(
                    "Direct reshard worker DP size differs from configured topology: "
                    f"worker={identity}, expected={self._data_parallel_size}, "
                    f"actual={worker_dp_size}"
                )
            if int(worker["tp_size"]) != self._tensor_parallel_size:
                raise RuntimeError(
                    "Direct reshard worker TP size differs from configured topology: "
                    f"worker={identity}, expected={self._tensor_parallel_size}, "
                    f"actual={worker['tp_size']}"
                )
            if (
                not 0 <= identity[0] < self._data_parallel_size
                or not 0 <= identity[1] < self._tensor_parallel_size
            ):
                raise RuntimeError(f"Direct reshard worker identity is out of range: {identity}")
            if identity in by_identity:
                raise RuntimeError(f"Direct reshard returned duplicate worker identity {identity}")
            by_identity[identity] = worker
        returned_dp_ranks = sorted({dp_rank for dp_rank, _ in by_identity})
        expected_tp_ranks = set(range(self._tensor_parallel_size))
        for dp_rank in returned_dp_ranks:
            actual_tp_ranks = {
                tp_rank
                for worker_dp_rank, tp_rank in by_identity
                if worker_dp_rank == dp_rank
            }
            if actual_tp_ranks != expected_tp_ranks:
                raise RuntimeError(
                    "Direct reshard layout query returned an incomplete TP engine: "
                    f"dp_rank={dp_rank}, expected={sorted(expected_tp_ranks)}, "
                    f"actual={sorted(actual_tp_ranks)}"
                )
        ep_size = int(workers[0].get("ep_size", 1))
        if ep_size > 1:
            if (
                not isinstance(self, ColocatedDirectReshardWeightTransfer)
                or isinstance(self, FullGatherHCCLWeightTransfer)
            ):
                raise ValueError("EP-local publication currently requires colocated IPC")
            if ep_size != expected_world_size:
                raise ValueError("EP-local publication requires EP == DP x TP")
            representatives = {
                tp_rank: by_identity[(returned_dp_ranks[0], tp_rank)]
                for tp_rank in range(self._tensor_parallel_size)
            }
            for (dp_rank, tp_rank), worker in by_identity.items():
                if (
                    int(worker.get("ep_rank", -1)) != dp_rank * self._tensor_parallel_size + tp_rank
                    or int(worker.get("ep_size", 1)) != ep_size
                    or worker["tensors"] != representatives[tp_rank]["tensors"]
                ):
                    raise RuntimeError("EP workers disagree on contiguous equal expert ownership")
            # The planner's historical tp_rank field is a destination coordinate.
            # IPC retains physical DP/TP identity and delivers each EP bucket once.
            destinations = []
            for rank in range(ep_size):
                tp_rank = rank % self._tensor_parallel_size
                representative = representatives[tp_rank]
                tensors = [dict(tensor) for tensor in representative["tensors"]]
                if self._tensor_parallel_size > 1:
                    for tensor in tensors:
                        if tensor["placement"] == "shard":
                            expert = ".experts." in tensor["name"]
                            tensor.update(
                                shard_rank=rank if expert else tp_rank,
                                shard_group_size=ep_size if expert else self._tensor_parallel_size,
                            )
                destinations.append({
                    **representative,
                    "dp_rank": rank // self._tensor_parallel_size,
                    "ep_rank": rank,
                    "tp_rank": rank,
                    "tp_size": ep_size,
                    "tensors": tensors,
                })
            return destinations
        if self._flatten_native_moe_dp_destinations:
            if self._tensor_parallel_size != 1:
                raise ValueError(
                    "Native MoE DP-flattened direct reshard currently requires TP1"
                )
            representative = by_identity[(returned_dp_ranks[0], 0)]
            if int(representative["dp_size"]) != self._data_parallel_size:
                raise RuntimeError(
                    "Native MoE DP-flattened layout requires the worker to "
                    f"report DP{self._data_parallel_size}, got {representative['dp_size']}"
                )
            return [
                {
                    **representative,
                    "dp_rank": dp_rank,
                    "tp_rank": dp_rank,
                    "tp_size": self._data_parallel_size,
                    "tensors": [
                        dict(tensor) for tensor in representative["tensors"]
                    ],
                }
                for dp_rank in range(self._data_parallel_size)
            ]
        # vLLM internal-DP utilities fan out to every engine but may expose only
        # one engine's return value. Destination layouts are identical across DP,
        # while each worker validates its own global identity before receiving.
        representative_dp_rank = returned_dp_ranks[0]
        representatives = []
        for tp_rank in range(self._tensor_parallel_size):
            representative = by_identity[(representative_dp_rank, tp_rank)]
            expected_tensors = representative.get("tensors")
            for dp_rank in returned_dp_ranks[1:]:
                replica_tensors = by_identity[(dp_rank, tp_rank)].get("tensors")
                if replica_tensors != expected_tensors:
                    raise RuntimeError(
                        "Direct reshard layouts differ across same-TP DP replicas: "
                        f"tp_rank={tp_rank}, dp_rank={dp_rank}"
                    )
            representatives.append(representative)
        return representatives

    def _build_plan(
        self,
        client: VLLMWeightSyncClientMixin,
        state_dict: Mapping[str, Any],
    ) -> DirectReshardPlan:
        """Compile metadata-only source and destination layouts once."""
        destination_workers = self._query_destination_workers(client)
        source_rank = platform.get_rank()
        local_descriptions = self._adapter.direct_source_descriptions(
            state_dict,
            source_rank,
        )
        source_world_size = platform.get_world_size()
        rank_descriptions: list[Any] = [None] * source_world_size
        platform.all_gather_object(rank_descriptions, local_descriptions)
        source_layouts = resolve_source_layouts(rank_descriptions)
        global_shapes = {
            source.name: source.global_shape for source in source_layouts
        }
        destination_layouts = resolve_destination_layouts(
            destination_workers,
            global_shapes,
        )
        trace_dir = os.environ.get("HYPER_RL_DIRECT_PLAN_TRACE_DIR")
        if trace_dir and source_rank == 0:
            trace = {
                "rank_descriptions": rank_descriptions,
                "source_layouts": [
                    {
                        "name": source.name,
                        "source_rank": source.source_rank,
                        "global_shape": list(source.global_shape),
                        "starts": list(source.region.starts),
                        "lengths": list(source.region.lengths),
                    }
                    for source in source_layouts
                ],
                "destination_layouts": [
                    {
                        "name": destination.name,
                        "tp_rank": destination.tp_rank,
                        "placement": destination.placement,
                        "shard_dim": destination.shard_dim,
                        "global_shape": list(destination.global_shape),
                        "starts": list(destination.region.starts),
                        "lengths": list(destination.region.lengths),
                    }
                    for destination in destination_layouts
                ],
            }
            directory = Path(trace_dir)
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "layout.json"
            temporary = directory / f".{target.name}.{os.getpid()}.tmp"
            temporary.write_text(
                json.dumps(trace, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
        self._parameter_names = frozenset(
            source.source_key for source in source_layouts
        )
        plan = build_direct_reshard_plan(
            source_layouts,
            destination_layouts,
            source_world_size=source_world_size,
            bucket_size_bytes=self._bucket_size_bytes,
        )
        if source_rank == 0:
            logger.info(
                "direct-reshard plan: family=%s source_world_size=%s destination_tp=%s "
                "destination_workers=%s routes=%s fragments=%s max_bucket_bytes=%s",
                self._model.family,
                source_world_size,
                plan.destination_tp_size,
                self._data_parallel_size * self._tensor_parallel_size,
                plan.route_count,
                plan.fragment_count,
                max(
                    bucket.total_bytes
                    for route_buckets in plan.buckets.values()
                    for bucket in route_buckets
                ),
            )
        return plan

    def _ensure_plan(
        self,
        client: VLLMWeightSyncClientMixin,
        state_dict: Mapping[str, Any],
    ) -> DirectReshardPlan:
        if self._plan is None:
            self._plan = self._build_plan(client, state_dict)
        if self._parameter_names is None:
            raise RuntimeError("Direct reshard plan has no parameter contract")
        missing = sorted(self._parameter_names - set(state_dict))
        if missing:
            raise ValueError(
                "Direct reshard Actor state changed after planning; missing="
                + ", ".join(missing)
            )
        return self._plan

    @staticmethod
    def _distributed_policy_fingerprint(
        state_dict: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Gather only small norm tensors needed by publication verification."""
        fingerprint_shards = {
            name: tensor
            for name, tensor in state_dict.items()
            if is_policy_fingerprint_weight(name)
        }
        if not fingerprint_shards:
            raise RuntimeError("Direct reshard found no Actor norm tensors to verify")
        full_norms = _gather_selected_state_dict(
            fingerprint_shards,
            cpu_offload=True,
        )
        local_fingerprint = (
            policy_weight_fingerprint(full_norms)
            if platform.get_rank() == 0
            else None
        )
        fingerprints: list[Any] = [None] * platform.get_world_size()
        platform.all_gather_object(fingerprints, local_fingerprint)
        expected = fingerprints[0]
        if not isinstance(expected, Mapping) or any(
            fingerprint is not None for fingerprint in fingerprints[1:]
        ):
            raise RuntimeError(
                f"Direct reshard Actor fingerprint publication is invalid: {fingerprints}"
            )
        return dict(expected)

    @staticmethod
    def _distributed_source_content_identities(
        state_dict: Mapping[str, Any],
        plan: DirectReshardPlan,
    ) -> dict[int, dict[str, Any]]:
        """Hash every bounded canonical fragment without gathering model tensors."""
        # Torch is optional outside the Torch-NPU RL runtime.
        import torch  # pylint: disable=C0415,forbidden-backend-import

        source_rank = platform.get_rank()
        local_records: dict[int, dict[str, dict[str, Any]]] = {}
        for tp_rank in range(plan.destination_tp_size):
            fragments = {}
            for bucket in plan.for_route(source_rank, tp_rank):
                for entry in bucket.entries:
                    value = state_dict.get(entry.source_key)
                    if value is None:
                        raise ValueError(
                            f"Direct content source tensor {entry.source_key!r} is missing"
                        )
                    source_tensor = local_tensor(value)
                    source_slice = tuple(
                        slice(start, start + length)
                        for start, length in zip(entry.source_starts, entry.lengths)
                    )
                    raw = (
                        source_tensor[source_slice]
                        .detach()
                        .contiguous()
                        .view(torch.uint8)
                        .view(-1)
                        .to(device="cpu")
                    )
                    payload = platform.tensor_to_numpy(raw).tobytes()
                    key, record = direct_fragment_record(
                        entry.name,
                        entry.logical_starts,
                        entry.lengths,
                        entry.dtype_name,
                        payload,
                    )
                    if key in fragments:
                        raise RuntimeError(
                            f"Direct content source has duplicate fragment {key!r}"
                        )
                    fragments[key] = record
                    del raw, payload
            if fragments:
                local_records[tp_rank] = fragments
        gathered: list[Any] = [None] * platform.get_world_size()
        platform.all_gather_object(gathered, local_records)
        merged = {tp_rank: {} for tp_rank in range(plan.destination_tp_size)}
        for rank_records in gathered:
            if not isinstance(rank_records, Mapping):
                raise RuntimeError(
                    f"Direct content source rank returned invalid records: {rank_records!r}"
                )
            for tp_rank, fragments in rank_records.items():
                target = merged[int(tp_rank)]
                for key, record in fragments.items():
                    if key in target:
                        raise RuntimeError(
                            f"Direct content source ranks overlap fragment {key!r}"
                        )
                    target[key] = dict(record)
        identities = {
            tp_rank: aggregate_direct_content_identity(fragments)
            for tp_rank, fragments in merged.items()
        }
        if source_rank == 0:
            logger.info(
                "direct-reshard source identity: tp=%s total_bytes=%s fragments=%s digests=%s",
                plan.destination_tp_size,
                {rank: value["total_bytes"] for rank, value in identities.items()},
                {rank: value["fragment_count"] for rank, value in identities.items()},
                {rank: value["digest"] for rank, value in identities.items()},
            )
        return identities

    @staticmethod
    def _verify_worker_content_identities(
        client: VLLMWeightSyncClientMixin,
        policy_version: int,
        expected: Mapping[int, Mapping[str, Any]],
    ) -> None:
        """Require every rollout worker to match its source-derived TP identity."""
        client.verify_direct_content_identity(policy_version, expected)

    def publish(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Plan, broadcast local shards, commit, and verify one policy version."""
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise ValueError(
                "DirectReshardHCCLWeightTransfer requires an external vLLM HTTP client"
            )
        self.last_attempted_strategies = ("direct_reshard",)
        self.last_completed_strategy = None
        local_state_dict = synchronized_call(
            "direct reshard local-state extraction",
            lambda: self._mapped_local_state_dict(snapshot.payload),
        )
        try:
            plan = synchronized_call(
                "direct reshard layout planning",
                lambda: self._ensure_plan(client, local_state_dict),
            )
            expected_content = synchronized_call(
                "direct reshard source content identity",
                lambda: self._distributed_source_content_identities(
                    local_state_dict,
                    plan,
                ),
            )
            coordinator_call("direct reshard pause", client.pause)
            coordinator_call("direct reshard start", client.start_weight_update)
            synchronized_call(
                "direct reshard producer synchronization",
                platform.get_current_stream().synchronize,
            )
            self._transport.transfer(
                client,
                local_state_dict,
                plan,
                snapshot.version,
            )
            synchronized_call(
                "direct reshard HCCL injected receive failure",
                lambda: self._test_fault_injector.trigger("direct_receive_once"),
            )
            coordinator_call("direct reshard finish", client.finish_weight_update)
            coordinator_call(
                "direct reshard source content verification",
                lambda: self._verify_worker_content_identities(
                    client,
                    snapshot.version,
                    expected_content,
                ),
            )
            coordinator_call(
                "direct reshard rollout parameter manifest",
                lambda: _write_rollout_parameter_manifest(
                    client,
                    strategy="direct_reshard",
                    policy_version=snapshot.version,
                    data_parallel_size=self._data_parallel_size,
                ),
            )
            expected_fingerprint = synchronized_call(
                "direct reshard Actor policy fingerprint",
                lambda: self._distributed_policy_fingerprint(local_state_dict),
            )

            def verify_worker_fingerprints() -> None:
                """Verify every worker loaded the published policy version."""
                client.verify_policy_weight_identity(
                    snapshot.version,
                    expected_fingerprint,
                )

            coordinator_call("direct reshard policy fingerprint", verify_worker_fingerprints)
            self.last_policy_fingerprint = expected_fingerprint
            self.last_strategy = "direct_reshard"
            self.last_completed_strategy = "direct_reshard"
            self.direct_success_count += 1
        finally:
            local_state_dict.clear()

    def close(self) -> None:
        """Release cached stateless HCCL route references."""
        self._transport.close()


class ColocatedDirectReshardWeightTransfer(DirectReshardHCCLWeightTransfer):
    """Redistribute FSDP fragments among trainers, then use same-NPU IPC."""

    def __init__(
        self,
        model: VLLMModelRegistration,
        *,
        bucket_size_bytes: int = 128 * 2**20,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
    ) -> None:
        """Initialize direct resharding and retain failed IPC producers safely."""
        super().__init__(
            model,
            bucket_size_bytes=bucket_size_bytes,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
        self._flatten_native_moe_dp_destinations = (
            model.family in ("deepseek_v3", "qwen3_moe") and not model.is_hyper
        )
        if data_parallel_size <= 0:
            raise ValueError("Direct reshard data_parallel_size must be positive")
        self._data_parallel_size = int(data_parallel_size)
        self._failed_buffers: list[Any] = []

    @property
    def weights_awake(self) -> bool:
        """Return whether this transfer successfully restored vLLM weights."""
        return bool(getattr(self, "_weights_awake", False))

    @staticmethod
    def _run_control(
        operation: str,
        client: VLLMWeightSyncClientMixin,
        callback: Any,
    ) -> None:
        """Run a mutating endpoint call once per colocated rollout replica."""
        synchronized_call(
            operation,
            lambda: callback()
            if bool(getattr(client, "is_server_owner", True))
            else None,
        )

    @staticmethod
    def _gather_endpoints(client: VLLMWeightSyncClientMixin) -> tuple[str, ...]:
        """Return every unique colocated rollout endpoint."""
        endpoints = [""] * platform.get_world_size()
        platform.all_gather_object(endpoints, client.base_url)
        unique = tuple(sorted(set(endpoints)))
        if not unique or any(not endpoint for endpoint in unique):
            raise RuntimeError(f"Colocated direct endpoints are invalid: {endpoints}")
        return unique

    @staticmethod
    def _rollout_device_order(physical_device_ids: list[Any]) -> tuple[Any, ...]:
        """Order gathered UUIDs by the rollout's explicit visible-device contract."""
        visible_devices = tuple(
            device.strip()
            for device in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
            if device.strip()
        )
        by_physical_id = {
            str(device_id).rsplit("-", maxsplit=1)[-1]: device_id
            for device_id in physical_device_ids
        }
        if not visible_devices or set(visible_devices) != set(by_physical_id):
            raise RuntimeError(
                "Colocated direct physical devices do not match ASCEND_RT_VISIBLE_DEVICES: "
                f"visible={visible_devices}, workers={sorted(by_physical_id)}"
            )
        return tuple(by_physical_id[device] for device in visible_devices)

    @staticmethod
    def _ipc_delivery(
        workers: tuple[Any, ...],
        local_worker: Any,
        *,
        destination_size: int,
        tensor_parallel_size: int,
        target_rank: int,
    ) -> tuple[str, int, set[Any], list[dict[str, Any]]]:
        """Resolve one logical destination to its colocated physical producers."""
        replicated = destination_size == tensor_parallel_size
        flattened = destination_size == len(workers)
        if not replicated and not flattened:
            raise RuntimeError(
                "Colocated destination size must equal rollout TP or DP x TP: "
                f"destination={destination_size}, tp={tensor_parallel_size}, "
                f"workers={len(workers)}"
            )
        local_target = (
            local_worker.tp_rank
            if replicated
            else local_worker.dp_rank * tensor_parallel_size + local_worker.tp_rank
        )
        expected_devices = (
            {
                worker.physical_device_id
                for worker in workers
                if worker.tp_rank == target_rank
            }
            if replicated
            else {workers[target_rank].physical_device_id}
        )
        worker_topology = [
            {
                "dp_rank": worker.dp_rank,
                "tp_rank": worker.tp_rank,
                "physical_device_id": worker.physical_device_id,
            }
            for worker in workers
        ]
        return (
            "replicated_tp" if replicated else "flattened_dp_tp",
            local_target,
            expected_devices,
            worker_topology,
        )

    @classmethod
    def _resolve_ipc_topology(
        cls,
        data_parallel_size: int,
        tensor_parallel_size: int,
    ) -> tuple[Any, Any, tuple[Any, ...]]:
        """Map every Trainer NPU to its colocated rollout DP x TP worker."""
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (  # pylint: disable=C0415
            npu_generate_uuid,
        )

        npu_uuid = npu_generate_uuid()
        physical_device_ids: list[Any] = [None] * platform.get_world_size()
        platform.all_gather_object(physical_device_ids, npu_uuid)
        ordered_device_ids = cls._rollout_device_order(physical_device_ids)
        workers = resolve_physical_worker_topology(
            ordered_device_ids,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
        local_worker = next(
            worker for worker in workers if worker.physical_device_id == npu_uuid
        )
        return npu_uuid, local_worker, workers

    def _stream_redistribute_and_send(
        self,
        client: VLLMWeightSyncClientMixin,
        endpoints: tuple[str, ...],
        state_dict: Mapping[str, Any],
        plan: DirectReshardPlan,
        policy_version: int,
    ) -> None:
        """Publish one direct IPC bucket at a time and release it after ACK."""
        import torch  # pylint: disable=C0415,forbidden-backend-import

        device_handle = platform.get_device_handle(platform.device_type())
        device = torch.device(platform.device_type(), device_handle.current_device())
        rank = platform.get_rank()
        npu_uuid, local_worker, workers = self._resolve_ipc_topology(
            self._data_parallel_size,
            self._tensor_parallel_size,
        )
        for source_rank in range(plan.source_world_size):
            for target_rank in range(plan.destination_tp_size):
                delivery_mode, local_target, expected_devices, worker_topology = (
                    self._ipc_delivery(
                        workers,
                        local_worker,
                        destination_size=plan.destination_tp_size,
                        tensor_parallel_size=self._tensor_parallel_size,
                        target_rank=target_rank,
                    )
                )
                for bucket_index, bucket in enumerate(
                    plan.for_route(source_rank, target_rank)
                ):
                    packed = (
                        pack_direct_bucket(state_dict, bucket, device)
                        if rank == source_rank
                        else torch.empty(
                            bucket.total_bytes,
                            dtype=torch.uint8,
                            device=device,
                        )
                    )
                    platform.get_current_stream().synchronize()
                    platform.broadcast(packed, src=source_rank)
                    platform.get_current_stream().synchronize()
                    local_handles = (
                        {npu_uuid: _tensor_ipc_rebuild_args(packed)}
                        if local_target == target_rank
                        else {}
                    )
                    gathered_handles: list[Any] = [None] * platform.get_world_size()
                    platform.all_gather_object(gathered_handles, local_handles)
                    merged_handles = {
                        device_id: rebuild_args
                        for handles in gathered_handles
                        for device_id, rebuild_args in handles.items()
                    }
                    if set(merged_handles) != expected_devices:
                        raise RuntimeError(
                            "Colocated direct IPC handles do not cover target workers: "
                            f"target={target_rank}, expected={expected_devices}, "
                            f"actual={set(merged_handles)}"
                        )
                    payload = {
                        "buckets_by_target": {
                            target_rank: [
                                {
                                    "target_rank": target_rank,
                                    "bucket_index": bucket_index,
                                    "metadata": bucket.worker_metadata(),
                                    "ipc_handles": merged_handles,
                                }
                            ]
                        },
                        "delivery_mode": delivery_mode,
                        "tensor_parallel_size": self._tensor_parallel_size,
                        "worker_topology": worker_topology,
                    }
                    try:
                        self._send_payload(
                            client,
                            endpoints,
                            payload,
                            policy_version,
                            self._tensor_parallel_size,
                        )
                        platform.get_current_stream().synchronize()
                        if "direct_bucket_once" in self._test_fault_injector.configured():
                            synchronized_call(
                                "colocated direct acknowledged-bucket failure",
                                lambda: self._test_fault_injector.trigger(
                                    "direct_bucket_once",
                                    context=(f"source_rank={source_rank} target_rank={target_rank} "
                                             f"bucket_index={bucket_index} acknowledged_bytes={bucket.total_bytes} "
                                             f"first_tensor={bucket.entries[0].name}"),
                                ),
                            )
                    except Exception:
                        if local_handles:
                            self._failed_buffers.append(packed)
                        raise
                    finally:
                        # Failed buffers have an explicit owner; tracebacks must not retain a second reference.
                        del packed

    @staticmethod
    def _send_payload(
        client: VLLMWeightSyncClientMixin,
        endpoints: tuple[str, ...],
        payload: Mapping[str, Any],
        policy_version: int,
        destination_tp_size: int,
    ) -> None:
        """Send IPC handles once and validate every colocated TP worker."""
        payload_pickled = base64.b64encode(pickle.dumps(payload)).decode("ascii")
        send_error = None
        if platform.get_rank() == 0:
            try:
                with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
                    requests = [
                        executor.submit(
                            client.collective_rpc,
                            "receive_ipc_direct_reshard",
                            {
                                "payload_pickled": payload_pickled,
                                "policy_version": policy_version,
                            },
                            endpoint,
                        )
                        for endpoint in endpoints
                    ]
                    for endpoint, request in zip(endpoints, requests):
                        results = request.result(timeout=600)
                        expected_workers = sorted(
                            (
                                int(worker["dp_rank"]),
                                int(worker["tp_rank"]),
                                str(worker["physical_device_id"]),
                            )
                            for worker in payload["worker_topology"]
                        )
                        expected_by_identity = {
                            (dp_rank, tp_rank): physical_device_id
                            for dp_rank, tp_rank, physical_device_id in expected_workers
                        }
                        expected_dp_ranks = sorted({worker[0] for worker in expected_workers})
                        expected_tp_ranks = set(range(destination_tp_size))
                        if not expected_dp_ranks or any(
                            {
                                tp_rank
                                for worker_dp_rank, tp_rank in expected_by_identity
                                if worker_dp_rank == dp_rank
                            }
                            != expected_tp_ranks
                            for dp_rank in expected_dp_ranks
                        ):
                            raise RuntimeError(
                                f"Colocated direct payload has invalid TP workers: {expected_workers}"
                            )
                        if not isinstance(results, list) or not all(
                            isinstance(result, Mapping) and result.get("received") is True
                            for result in results
                        ):
                            raise RuntimeError(
                                f"Colocated direct endpoint {endpoint} returned {results}"
                            )
                        received_workers = sorted(
                            (
                                int(result["dp_rank"]),
                                int(result["tp_rank"]),
                                str(result["physical_device_id"]),
                            )
                            for result in results
                        )
                        received_identities = {
                            (dp_rank, tp_rank): physical_device_id
                            for dp_rank, tp_rank, physical_device_id in received_workers
                        }
                        returned_dp_ranks = sorted(
                            {dp_rank for dp_rank, _tp_rank in received_identities}
                        )
                        if (
                            len(received_identities) != len(received_workers)
                            or not returned_dp_ranks
                            or any(
                                {
                                    tp_rank
                                    for worker_dp_rank, tp_rank in received_identities
                                    if worker_dp_rank == dp_rank
                                }
                                != expected_tp_ranks
                                for dp_rank in returned_dp_ranks
                            )
                            or any(
                                expected_by_identity.get(identity) != physical_device_id
                                for identity, physical_device_id in received_identities.items()
                            )
                        ):
                            raise RuntimeError(
                                f"Colocated direct endpoint {endpoint} returned {results}"
                            )
            except Exception as error:  # pylint: disable=W0718
                send_error = error
        synchronize_error(send_error, "colocated direct IPC transfer")

    def publish(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Publish one FSDP policy without materializing full Actor tensors."""
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise ValueError("Colocated direct reshard requires a vLLM HTTP client")
        self.last_attempted_strategies = ("direct_reshard",)
        self.last_completed_strategy = None
        self._weights_awake = False
        local_state_dict = synchronized_call(
            "colocated direct local-state extraction",
            lambda: self._mapped_local_state_dict(snapshot.payload),
        )
        try:
            self._run_control(
                "colocated direct weight wake",
                client,
                lambda: client.wake_up(("weights",)),
            )
            self._weights_awake = True
            plan = synchronized_call(
                "colocated direct layout planning",
                lambda: self._ensure_plan(client, local_state_dict),
            )
            expected_content = synchronized_call(
                "colocated direct source content identity",
                lambda: self._distributed_source_content_identities(
                    local_state_dict,
                    plan,
                ),
            )
            self._run_control("colocated direct pause", client, client.pause)
            self._run_control(
                "colocated direct start",
                client,
                client.start_weight_update,
            )
            endpoints = synchronized_call(
                "colocated direct endpoints",
                lambda: self._gather_endpoints(client),
            )
            synchronized_call(
                "colocated direct streaming redistribution",
                lambda: self._stream_redistribute_and_send(
                    client,
                    endpoints,
                    local_state_dict,
                    plan,
                    snapshot.version,
                ),
            )
            synchronized_call(
                "colocated direct injected receive failure",
                lambda: self._test_fault_injector.trigger("direct_receive_once"),
            )
            synchronized_call(
                "colocated direct producer synchronization",
                platform.get_current_stream().synchronize,
            )
            self._run_control(
                "colocated direct finish",
                client,
                client.finish_weight_update,
            )
            synchronized_call(
                "colocated direct source content verification",
                lambda: self._verify_worker_content_identities(
                    client,
                    snapshot.version,
                    expected_content,
                ),
            )
            synchronized_call(
                "colocated direct rollout parameter manifest",
                lambda: _write_rollout_parameter_manifest(
                    client,
                    strategy="direct_reshard",
                    policy_version=snapshot.version,
                    data_parallel_size=self._data_parallel_size,
                ),
            )
            expected_fingerprint = synchronized_call(
                "colocated direct Actor fingerprint",
                lambda: self._distributed_policy_fingerprint(local_state_dict),
            )

            def verify_local_fingerprint() -> None:
                """Compare every rollout worker against the transferred policy."""
                client.verify_policy_weight_identity(
                    snapshot.version,
                    expected_fingerprint,
                )

            synchronized_call(
                "colocated direct policy fingerprint",
                verify_local_fingerprint,
            )
            self.last_policy_fingerprint = expected_fingerprint
            self.last_strategy = "direct_reshard"
            self.last_completed_strategy = "direct_reshard"
            self.direct_success_count += 1
        finally:
            local_state_dict.clear()

    def release_failed_buffers(self) -> None:
        """Release failed IPC producers after fallback fully overwrote the transaction."""
        self._failed_buffers.clear()

    def close(self) -> None:
        """Release route metadata and any failed IPC producer buffers."""
        super().close()
        self._failed_buffers.clear()


class ColocatedFullGatherWeightTransfer(ColocatedDirectReshardWeightTransfer):
    """Gather one bounded canonical bucket and publish it through same-NPU IPC."""

    def __init__(
        self,
        model: VLLMModelRegistration,
        *,
        bucket_size_bytes: int = 128 * 2**20,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
    ) -> None:
        """Initialize the bounded gather-first plan and IPC transport."""
        super().__init__(
            model,
            bucket_size_bytes=bucket_size_bytes,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
        self._streaming_plan: Optional[StreamingFullGatherPlan] = None
        self._streaming_parameter_names: Optional[frozenset[str]] = None
        self._streaming_executor = StreamingFullGatherExecutor()
        self.configured_strategy = "full_gather"
        self.last_attempted_strategies = ()
        self.last_completed_strategy = None
        self.last_streaming_stats: Optional[dict[str, Any]] = None

    def _build_streaming_plan(
        self,
        client: VLLMWeightSyncClientMixin,
        state_dict: Mapping[str, Any],
    ) -> StreamingFullGatherPlan:
        """Build gather-first metadata independently of direct route planning."""
        destination_workers = self._query_destination_workers(client)
        source_rank = platform.get_rank()
        local_descriptions = self._adapter.direct_source_descriptions(
            state_dict,
            source_rank,
        )
        source_world_size = platform.get_world_size()
        rank_descriptions: list[Any] = [None] * source_world_size
        platform.all_gather_object(rank_descriptions, local_descriptions)
        source_layouts = resolve_source_layouts(rank_descriptions)
        global_shapes = {
            source.name: source.global_shape for source in source_layouts
        }
        destination_layouts = resolve_destination_layouts(
            destination_workers,
            global_shapes,
        )
        aliases = {}
        if self._model.model.tie_word_embeddings:
            source_names = set(global_shapes)
            destination_names = {layout.name for layout in destination_layouts}
            embedding_name = "model.embed_tokens.weight"
            lm_head_name = "lm_head.weight"
            if {embedding_name, lm_head_name} <= source_names & destination_names:
                aliases[lm_head_name] = embedding_name
        plan = build_streaming_full_gather_plan(
            source_layouts,
            destination_layouts,
            source_world_size=source_world_size,
            bucket_size_bytes=self._bucket_size_bytes,
            aliases=aliases,
        )
        self._streaming_parameter_names = frozenset(
            contribution.source_name
            for buckets in plan.buckets.values()
            for bucket in buckets
            for entry in bucket.entries
            for contribution in entry.contributions
        )
        if source_rank == 0:
            logger.info(
                "streaming full-gather plan: family=%s source_world_size=%s "
                "destination_tp=%s buckets=%s fragments=%s bytes=%s "
                "max_bucket_bytes=%s aliases=%s",
                self._model.family,
                source_world_size,
                plan.destination_tp_size,
                plan.bucket_count,
                plan.fragment_count,
                plan.total_bytes,
                max(
                    bucket.total_bytes
                    for target_buckets in plan.buckets.values()
                    for bucket in target_buckets
                ),
                [(alias.alias_name, alias.target_name) for alias in plan.aliases],
            )
        return plan

    def _ensure_streaming_plan(
        self,
        client: VLLMWeightSyncClientMixin,
        state_dict: Mapping[str, Any],
    ) -> StreamingFullGatherPlan:
        """Build once and reject a changed Trainer source contract."""
        if self._streaming_plan is None:
            self._streaming_plan = self._build_streaming_plan(client, state_dict)
        if self._streaming_parameter_names is None:
            raise RuntimeError("Streaming full-gather plan has no source contract")
        missing = sorted(self._streaming_parameter_names - set(state_dict))
        if missing:
            raise ValueError(
                "Streaming full-gather Actor state changed after planning; missing="
                + ", ".join(missing)
            )
        return self._streaming_plan

    @classmethod
    def _materialize_streaming_bucket(
        cls,
        state_dict: Mapping[str, Any],
        bucket: StreamingGatherBucket,
        device: Any,
        fragment_records: dict[str, dict[str, Any]],
        content_accumulator: StreamingContentIdentityAccumulator,
    ) -> StreamingMaterializedBucket:
        """Gather, assemble, hash, and pack exactly one canonical bucket."""
        # Torch is optional outside the Torch-NPU RL runtime.
        import torch  # pylint: disable=C0415,forbidden-backend-import

        rank = platform.get_rank()
        fragment_tensors = []
        for fragment in bucket.entries:
            gathered = []
            for contribution in fragment.contributions:
                if rank == contribution.source_rank:
                    local = extract_streaming_contributions(
                        fragment,
                        rank,
                        state_dict,
                    )
                    matching = [item for item in local if item.layout == contribution]
                    if len(matching) != 1:
                        raise RuntimeError(
                            "Streaming full-gather source contribution is not unique: "
                            f"rank={rank}, contribution={contribution}, matches={len(matching)}"
                        )
                    tensor = matching[0].tensor
                    if str(tensor.device) != str(device):
                        tensor = tensor.to(device)
                else:
                    tensor = torch.empty(
                        contribution.lengths,
                        dtype=getattr(torch, fragment.dtype_name),
                        device=device,
                    )
                platform.get_current_stream().synchronize()
                platform.broadcast(tensor, src=contribution.source_rank)
                platform.get_current_stream().synchronize()
                gathered.append(
                    MaterializedStreamingContribution(contribution, tensor)
                )
            canonical = assemble_streaming_fragment(fragment, gathered)
            raw = canonical.detach().contiguous().view(torch.uint8).view(-1).to("cpu")
            values = platform.tensor_to_numpy(raw).tobytes()
            content_accumulator.add(fragment, values)
            key, record = direct_fragment_record(
                fragment.name,
                fragment.canonical_starts,
                fragment.lengths,
                fragment.dtype_name,
                values,
            )
            if key in fragment_records:
                raise RuntimeError(
                    f"Streaming full-gather source duplicated fragment {key!r}"
                )
            fragment_records[key] = record
            fragment_tensors.append(canonical)
            del gathered, raw, values
        return pack_streaming_bucket(bucket, fragment_tensors)

    def _send_streaming_bucket(
        self,
        client: VLLMWeightSyncClientMixin,
        endpoints: tuple[str, ...],
        workers: tuple[Any, ...],
        npu_uuid: Any,
        local_worker: Any,
        target_tp_rank: int,
        bucket_index: int,
        bucket: StreamingGatherBucket,
        materialized: StreamingMaterializedBucket,
        policy_version: int,
        destination_size: int,
    ) -> StreamingBucketAck:
        """Export one same-NPU buffer per target worker and wait for its ACK."""
        delivery_mode, local_target_rank, expected_devices, worker_topology = (
            self._ipc_delivery(
                workers,
                local_worker,
                destination_size=destination_size,
                tensor_parallel_size=self._tensor_parallel_size,
                target_rank=target_tp_rank,
            )
        )
        local_handles = (
            {npu_uuid: _tensor_ipc_rebuild_args(materialized.value)}
            if local_target_rank == target_tp_rank
            else {}
        )
        gathered_handles: list[Any] = [None] * platform.get_world_size()
        platform.all_gather_object(gathered_handles, local_handles)
        merged_handles = {}
        for handles in gathered_handles:
            merged_handles.update(handles)
        if set(merged_handles) != expected_devices:
            raise RuntimeError(
                "Streaming full-gather IPC handles do not cover target workers: "
                f"target_tp={target_tp_rank}, expected={expected_devices}, "
                f"actual={set(merged_handles)}"
            )
        payload = {
            "buckets_by_target": {
                target_tp_rank: [
                    {
                        "target_rank": target_tp_rank,
                        "bucket_index": bucket_index,
                        "metadata": bucket.worker_metadata(),
                        "ipc_handles": merged_handles,
                    }
                ]
            },
            "delivery_mode": delivery_mode,
            "tensor_parallel_size": self._tensor_parallel_size,
            "worker_topology": worker_topology,
            "streaming_full_gather": True,
        }
        self._send_payload(
            client,
            endpoints,
            payload,
            policy_version,
            self._tensor_parallel_size,
        )
        return StreamingBucketAck(
            target_tp_rank=target_tp_rank,
            bucket_index=bucket_index,
            total_bytes=bucket.total_bytes,
            worker_count=len(expected_devices),
        )

    @property
    def _streaming_transport_name(self) -> str:
        """Return the executor transport used for allocation metrics."""
        return "ipc"

    def _streaming_control_call(
        self,
        operation: str,
        client: VLLMWeightSyncClientMixin,
        callback: Any,
    ) -> Any:
        """Run one colocated mutation on each server owner."""
        return self._run_control(operation, client, callback)

    @staticmethod
    def _streaming_query_call(operation: str, callback: Any) -> Any:
        """Run one colocated query and synchronize failures across trainers."""
        return synchronized_call(operation, callback)

    def _wake_streaming_weights(
        self,
        client: VLLMWeightSyncClientMixin,
        weights_already_awake: bool,
    ) -> None:
        """Restore colocated rollout weights once before publication."""
        if not weights_already_awake:
            self._streaming_control_call(
                "streaming full-gather weight wake",
                client,
                lambda: client.wake_up(("weights",)),
            )
        self._weights_awake = True

    def _prepare_streaming_transport(
        self,
        client: VLLMWeightSyncClientMixin,
        _plan: StreamingFullGatherPlan,
    ) -> tuple[Any, ...]:
        """Resolve colocated endpoints, devices, and worker ownership."""
        endpoints = self._gather_endpoints(client)
        return (
            endpoints,
            *self._resolve_ipc_topology(
                self._data_parallel_size,
                self._tensor_parallel_size,
            ),
        )

    def _send_streaming_transport(
        self,
        client: VLLMWeightSyncClientMixin,
        context: tuple[Any, ...],
        target_tp_rank: int,
        bucket_index: int,
        bucket: StreamingGatherBucket,
        materialized: StreamingMaterializedBucket,
        policy_version: int,
        destination_size: int,
    ) -> StreamingBucketAck:
        """Send one bounded bucket through colocated NPU IPC."""
        endpoints, npu_uuid, local_worker, workers = context
        return self._send_streaming_bucket(
            client,
            endpoints,
            workers,
            npu_uuid,
            local_worker,
            target_tp_rank,
            bucket_index,
            bucket,
            materialized,
            policy_version,
            destination_size,
        )

    @staticmethod
    def _current_policy_version(client: VLLMWeightSyncClientMixin) -> int:
        """Return one committed version shared by all rollout workers."""
        versions = {
            int(result["version"])
            for result in client.collective_rpc("get_policy_version")
        }
        if len(versions) != 1:
            raise RuntimeError(
                f"Streaming full-gather worker versions differ: {sorted(versions)}"
            )
        return versions.pop()

    @staticmethod
    def _abort_update(
        client: VLLMWeightSyncClientMixin,
        restore_policy_version: int,
    ) -> None:
        """Abort pending identity without claiming failed IPC buffers are reusable."""
        def abort_and_validate() -> None:
            """Abort pending worker identities and require successful replies."""
            results = client.collective_rpc(
                "abort_weight_update",
                kwargs={"restore_policy_version": restore_policy_version},
            )
            if results and all(
                isinstance(result, Mapping) and bool(result.get("aborted"))
                for result in results
            ):
                return
            raise RuntimeError(
                f"vLLM rejected streaming full-gather abort: {results}"
            )

        coordinator_call("streaming full-gather abort", abort_and_validate)

    def publish(
        self,
        client: Any,
        snapshot: PolicySnapshot,
        *,
        weights_already_awake: bool = False,
        manifest_strategy: str = "full_gather",
    ) -> None:
        """Publish one policy through an ACK-gated, bounded transport."""
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise ValueError("Colocated streaming full-gather requires a vLLM HTTP client")
        self.last_attempted_strategies = ("full_gather",)
        self.last_completed_strategy = None
        local_state_dict = synchronized_call(
            "streaming full-gather local-state extraction",
            lambda: self._mapped_local_state_dict(snapshot.payload),
        )
        trainer_memory_samples = [_current_trainer_memory_stats()]
        worker_memory_before: list[Mapping[str, Any]] = []
        fragment_records: dict[int, dict[str, dict[str, Any]]] = {}
        try:
            previous_version = coordinator_call(
                "streaming full-gather baseline version",
                lambda: self._current_policy_version(client),
            )
            self._wake_streaming_weights(client, weights_already_awake)
            plan = synchronized_call(
                "streaming full-gather layout planning",
                lambda: self._ensure_streaming_plan(client, local_state_dict),
            )
            transport_context = synchronized_call(
                "streaming full-gather transport preparation",
                lambda: self._prepare_streaming_transport(client, plan),
            )
            device_handle = platform.get_device_handle(platform.device_type())
            device = f"{platform.device_type()}:{device_handle.current_device()}"
            accumulators = {
                tp_rank: StreamingContentIdentityAccumulator(plan, tp_rank)
                for tp_rank in range(plan.destination_tp_size)
            }
            fragment_records = {
                tp_rank: {} for tp_rank in range(plan.destination_tp_size)
            }
            worker_memory_before = self._streaming_query_call(
                "streaming full-gather rollout baseline memory stats",
                lambda: client.collective_rpc("get_weight_sync_memory_stats"),
            )
            self._streaming_control_call(
                "streaming full-gather pause",
                client,
                client.pause,
            )

            def materialize(bucket: StreamingGatherBucket) -> StreamingMaterializedBucket:
                """Materialize one bounded bucket for the transport callback."""
                materialized = synchronized_call(
                    "streaming full-gather bucket materialization",
                    lambda: self._materialize_streaming_bucket(
                        local_state_dict,
                        bucket,
                        device,
                        fragment_records[bucket.target_tp_rank],
                        accumulators[bucket.target_tp_rank],
                    ),
                )
                trainer_memory_samples.append(_current_trainer_memory_stats())
                return materialized

            def send(
                target_tp_rank: int,
                bucket_index: int,
                bucket: StreamingGatherBucket,
                materialized: StreamingMaterializedBucket,
            ) -> StreamingBucketAck:
                """Send one bucket and return its acknowledgement."""
                ack = synchronized_call(
                    "streaming full-gather bucket transfer",
                    lambda: self._send_streaming_transport(
                        client,
                        transport_context,
                        target_tp_rank,
                        bucket_index,
                        bucket,
                        materialized,
                        snapshot.version,
                        plan.destination_tp_size,
                    ),
                )
                synchronized_call(
                    "streaming full-gather injected bucket failure",
                    lambda: self._test_fault_injector.trigger(
                        "streaming_bucket_once"
                    ),
                )
                return ack

            stats = self._streaming_executor.execute(
                plan,
                start=lambda: self._streaming_control_call(
                    "streaming full-gather start",
                    client,
                    client.start_weight_update,
                ),
                materialize_bucket=materialize,
                send_bucket=send,
                release_payload=lambda _payload: synchronized_call(
                    "streaming full-gather acknowledged-buffer release",
                    platform.get_current_stream().synchronize,
                ),
                finish=lambda: self._streaming_control_call(
                    "streaming full-gather finish",
                    client,
                    client.finish_weight_update,
                ),
                abort=lambda _error: self._abort_update(client, previous_version),
                transport=self._streaming_transport_name,
            )
            synchronized_call(
                "streaming full-gather post-release synchronization",
                platform.get_current_stream().synchronize,
            )
            trainer_memory_samples.append(_current_trainer_memory_stats())
            expected_content = {
                tp_rank: aggregate_direct_content_identity(records)
                for tp_rank, records in fragment_records.items()
            }
            self._streaming_query_call(
                "streaming full-gather source content verification",
                lambda: self._verify_worker_content_identities(
                    client,
                    snapshot.version,
                    expected_content,
                ),
            )
            self._streaming_query_call(
                "streaming full-gather rollout parameter manifest",
                lambda: _write_rollout_parameter_manifest(
                    client,
                    strategy=manifest_strategy,
                    policy_version=snapshot.version,
                    data_parallel_size=self._data_parallel_size,
                ),
            )
            worker_memory = self._streaming_query_call(
                "streaming full-gather rollout memory stats",
                lambda: client.collective_rpc("get_weight_sync_memory_stats"),
            )
            expected_fingerprint = synchronized_call(
                "streaming full-gather Actor fingerprint",
                lambda: self._distributed_policy_fingerprint(local_state_dict),
            )
            self._streaming_control_call(
                "streaming full-gather policy fingerprint",
                client,
                lambda: client.verify_policy_weight_identity(
                    snapshot.version,
                    expected_fingerprint,
                ),
            )
            streaming_content = {
                tp_rank: accumulator.finalize()
                for tp_rank, accumulator in accumulators.items()
            }
            self.last_policy_fingerprint = expected_fingerprint
            self.last_streaming_stats = {
                **{name: value for name, value in vars(stats).items() if name != "buckets"},
                **_streaming_memory_summary(trainer_memory_samples, worker_memory_before, worker_memory),
            }
            self.last_strategy = "full_gather"
            self.last_completed_strategy = "full_gather"
            if platform.get_rank() == 0:
                logger.info(
                    "streaming full-gather %s complete: version=%s stats=%s content=%s",
                    self._streaming_transport_name,
                    snapshot.version,
                    self.last_streaming_stats,
                    {
                        rank: identity["digest"]
                        for rank, identity in streaming_content.items()
                    },
                )
                for bucket_stats in stats.buckets:
                    logger.debug(
                        "streaming full-gather bucket complete: version=%s target_tp=%s "
                        "bucket=%s fragments=%s gathered_bytes=%s packed_bytes=%s "
                        "workers=%s sent_bytes=%s acked_bytes=%s released_bytes=%s",
                        snapshot.version,
                        bucket_stats.target_tp_rank,
                        bucket_stats.bucket_index,
                        bucket_stats.fragment_count,
                        bucket_stats.gathered_bytes,
                        bucket_stats.packed_bytes,
                        bucket_stats.worker_count,
                        bucket_stats.sent_bytes,
                        bucket_stats.acked_bytes,
                        bucket_stats.released_bytes,
                    )
        finally:
            local_state_dict.clear()

    def release_failed_buffers(self) -> None:
        """Release failed IPC storage after fallback completely overwrote it."""
        self._streaming_executor.release_failed_payloads(lambda _payload: None)
        super().release_failed_buffers()

    def close(self) -> None:
        """Release failed streaming storage after rollout shutdown."""
        self._streaming_executor.release_failed_payloads(lambda _payload: None)
        super().close()


class FullGatherHCCLWeightTransfer(ColocatedFullGatherWeightTransfer):
    """Gather and broadcast one bounded destination bucket over disjoint HCCL."""

    def __init__(
        self,
        model: VLLMModelRegistration,
        *,
        bucket_size_bytes: int = 128 * 2**20,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
    ) -> None:
        """Reuse the gather-first planner with one HCCL route per TP rank."""
        super().__init__(
            model,
            bucket_size_bytes=bucket_size_bytes,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
        self._flatten_native_moe_dp_destinations = False

    @property
    def _streaming_transport_name(self) -> str:
        """Return the executor transport used for allocation metrics."""
        return "hccl"

    def _streaming_control_call(
        self,
        operation: str,
        client: VLLMWeightSyncClientMixin,
        callback: Any,
    ) -> Any:
        """Run one shared disjoint mutation on the Trainer coordinator."""
        del client
        return coordinator_call(operation, callback)

    @staticmethod
    def _streaming_query_call(operation: str, callback: Any) -> Any:
        """Run one shared disjoint query on the Trainer coordinator."""
        return coordinator_call(operation, callback)

    def _wake_streaming_weights(
        self,
        client: VLLMWeightSyncClientMixin,
        weights_already_awake: bool,
    ) -> None:
        """Leave resident disjoint rollout weights unchanged."""
        del client, weights_already_awake

    def _prepare_streaming_transport(
        self,
        client: VLLMWeightSyncClientMixin,
        plan: StreamingFullGatherPlan,
    ) -> tuple[Any, ...]:
        """Initialize one HCCL producer route per destination TP rank."""
        endpoint = self._transport.ensure_streaming_groups(
            client,
            plan.destination_tp_size,
        )
        return (endpoint,)

    def _send_streaming_transport(
        self,
        client: VLLMWeightSyncClientMixin,
        context: tuple[Any, ...],
        target_tp_rank: int,
        bucket_index: int,
        bucket: StreamingGatherBucket,
        materialized: StreamingMaterializedBucket,
        policy_version: int,
        destination_size: int,
    ) -> StreamingBucketAck:
        """Broadcast one bounded bucket through disjoint HCCL."""
        del destination_size
        worker_count = self._transport.broadcast_streaming_bucket(
            client,
            context[0],
            materialized.value,
            bucket.worker_metadata(),
            target_tp_rank=target_tp_rank,
            bucket_index=bucket_index,
            policy_version=policy_version,
        )
        return StreamingBucketAck(
            target_tp_rank=target_tp_rank,
            bucket_index=bucket_index,
            total_bytes=bucket.total_bytes,
            worker_count=worker_count,
        )


class FallbackWeightTransfer(_RefitCompatibleTransfer):
    """Try direct reshard first and recover with bounded full-gather."""

    def __init__(self, primary: WeightTransfer, fallback: WeightTransfer) -> None:
        """Store both strategies and expose the successful publication identity."""
        self._primary = primary
        self._fallback = fallback
        self.last_policy_fingerprint: Optional[dict[str, Any]] = None
        self.configured_strategy = "direct_reshard"
        self.last_strategy: Optional[str] = None
        self.last_fallback_reason: Optional[str] = None
        self.last_attempted_strategies: tuple[str, ...] = ()
        self.last_completed_strategy: Optional[str] = None
        self.last_streaming_stats: Optional[dict[str, Any]] = None
        self.fallback_count = 0
        self.direct_success_count = 0

    @staticmethod
    def _current_policy_version(client: VLLMWeightSyncClientMixin) -> int:
        """Read committed versions without touching potentially sleeping weights."""
        versions = {
            int(result["version"])
            for result in client.collective_rpc("get_policy_version")
        }
        if len(versions) != 1:
            raise RuntimeError(
                "Rollout versions differ before direct-reshard fallback: "
                f"{sorted(versions)}"
            )
        return versions.pop()

    @staticmethod
    def _abort_direct_update(
        client: VLLMWeightSyncClientMixin,
        restore_policy_version: int,
    ) -> None:
        """Discard pending identity so a full update can overwrite partial weights."""
        def abort_and_validate() -> None:
            """Require every worker to acknowledge transaction recovery."""
            results = client.collective_rpc(
                "abort_weight_update",
                kwargs={"restore_policy_version": restore_policy_version},
            )
            if results and all(
                isinstance(result, Mapping) and bool(result.get("aborted"))
                for result in results
            ):
                return
            raise RuntimeError(f"vLLM rejected direct-update abort: {results}")

        coordinator_call("direct-reshard fallback abort", abort_and_validate)

    def publish(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Use full gather only when the direct transaction raises an error."""
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise ValueError("Fallback weight transfer requires a vLLM HTTP client")
        previous_version = coordinator_call(
            "direct-reshard fallback baseline",
            lambda: self._current_policy_version(client),
        )
        fallback_strategy = "full_gather"
        self.last_attempted_strategies = ("direct_reshard",)
        self.last_completed_strategy = None
        self.last_fallback_reason = None
        try:
            self._primary.publish(client, snapshot)
        except Exception as direct_error:  # pylint: disable=W0718
            self.last_fallback_reason = repr(direct_error)
            self.last_attempted_strategies = (
                "direct_reshard",
                fallback_strategy,
            )
            if platform.get_rank() == 0:
                logger.warning(
                    "weight-sync fallback starting: configured=direct_reshard "
                    "attempted=%s completed=none fallback=%s reason=%r",
                    self.last_attempted_strategies,
                    fallback_strategy,
                    self.last_fallback_reason,
                )
            try:
                self._abort_direct_update(client, previous_version)
            except Exception as abort_error:
                raise RuntimeError(
                    "Direct reshard failed and its transaction could not be aborted: "
                    f"direct={direct_error!r}, abort={abort_error!r}"
                ) from abort_error
            try:
                self._fallback.publish(
                    client,
                    snapshot,
                    weights_already_awake=bool(
                        getattr(self._primary, "weights_awake", False)
                    ),
                    manifest_strategy="full_gather_fallback",
                )
            except Exception as fallback_error:
                try:
                    self._abort_direct_update(client, previous_version)
                except Exception as abort_error:
                    raise RuntimeError(
                        "Direct reshard and full-gather fallback failed, then recovery "
                        "also failed: "
                        f"direct={direct_error!r}, fallback={fallback_error!r}, "
                        f"abort={abort_error!r}"
                    ) from abort_error
                recovered_version = coordinator_call(
                    "failed fallback committed-version check",
                    lambda: self._current_policy_version(client),
                )
                admission_paused = coordinator_call(
                    "failed fallback admission check",
                    client.is_paused,
                )
                if recovered_version != previous_version or not admission_paused:
                    raise RuntimeError(
                        "Failed fallback did not preserve the previous closed policy: "
                        f"expected_version={previous_version}, actual_version={recovered_version}, "
                        f"admission_paused={admission_paused}"
                    ) from fallback_error
                if platform.get_rank() == 0:
                    logger.error(
                        "weight-sync publication failed closed: configured=direct_reshard "
                        "attempted=%s completed=none committed_version=%s "
                        "admission_paused=%s direct_reason=%r fallback_reason=%r",
                        self.last_attempted_strategies,
                        recovered_version,
                        admission_paused,
                        self.last_fallback_reason,
                        repr(fallback_error),
                    )
                raise RuntimeError(
                    "Both direct reshard and full-gather fallback failed: "
                    f"direct={direct_error!r}, fallback={fallback_error!r}"
                ) from fallback_error
            release_failed_buffers = getattr(self._primary, "release_failed_buffers", None)
            if callable(release_failed_buffers):
                release_failed_buffers()
            self.last_strategy = "full_gather"
            self.last_completed_strategy = fallback_strategy
            self.fallback_count += 1
            self.last_policy_fingerprint = getattr(
                self._fallback,
                "last_policy_fingerprint",
                None,
            )
            self.last_streaming_stats = getattr(
                self._fallback,
                "last_streaming_stats",
                None,
            )
            if platform.get_rank() == 0:
                logger.info(
                    "weight-sync publication complete: configured=direct_reshard "
                    "attempted=%s completed=%s fallback_count=%s reason=%r",
                    self.last_attempted_strategies,
                    self.last_completed_strategy,
                    self.fallback_count,
                    self.last_fallback_reason,
                )
            return
        self.last_strategy = "direct_reshard"
        self.last_completed_strategy = "direct_reshard"
        self.direct_success_count += 1
        self.last_fallback_reason = None
        self.last_policy_fingerprint = getattr(
            self._primary,
            "last_policy_fingerprint",
            None,
        )
        self.last_streaming_stats = None

    def close(self) -> None:
        """Release resources owned by both transfer strategies."""
        for transfer in (self._primary, self._fallback):
            close = getattr(transfer, "close", None)
            if callable(close):
                close()


def build_weight_transfer(
    deployment: str,
    model: VLLMModelRegistration,
    *,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    bucket_size_bytes: int = 128 * 2**20,
    strategy: str = "full_gather",
    fallback_strategy: str = "none",
) -> WeightTransfer:
    """Build bounded full-gather or TP-aware direct reshard with recovery."""
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    if data_parallel_size <= 0:
        raise ValueError("data_parallel_size must be positive")
    validate_weight_sync_support(
        deployment=deployment,
        model_family=model.family,
        rollout_tp=tensor_parallel_size,
        strategy=strategy,
        fallback_strategy=fallback_strategy,
    )
    needs_full_gather = strategy == "full_gather" or fallback_strategy == "full_gather"
    full: Optional[WeightTransfer] = None
    if needs_full_gather and deployment == "colocated":
        full = ColocatedFullGatherWeightTransfer(
            model,
            bucket_size_bytes=bucket_size_bytes,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
    elif needs_full_gather:
        full = FullGatherHCCLWeightTransfer(
            model,
            bucket_size_bytes=bucket_size_bytes,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
    if strategy == "full_gather":
        if full is None:
            raise RuntimeError("Full-gather transfer was not constructed")
        return full
    if deployment == "colocated":
        direct = ColocatedDirectReshardWeightTransfer(
            model,
            bucket_size_bytes=bucket_size_bytes,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
    else:
        direct = DirectReshardHCCLWeightTransfer(
            model,
            bucket_size_bytes=bucket_size_bytes,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
    if fallback_strategy == "full_gather":
        if full is None:
            raise RuntimeError("Full-gather fallback was not constructed")
        return FallbackWeightTransfer(direct, full)
    return direct
__all__ = [
    "ColocatedFullGatherWeightTransfer",
    "ColocatedDirectReshardWeightTransfer",
    "DirectReshardHCCLWeightTransfer",
    "FallbackWeightTransfer",
    "FullGatherHCCLWeightTransfer",
    "WeightTransfer",
    "build_weight_transfer",
]
