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
"""Development tests for bounded streaming full-gather primitives."""

from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch

import rl.roles.weight_sync.vllm_worker as worker_module
from rl.config import _validate_vllm_weight_sync
from rl.roles.weight_sync.layout import (
    DestinationTensorLayout,
    SourceTensorLayout,
    TensorRegion,
    build_direct_reshard_plan,
    resolve_destination_layouts,
    resolve_source_layouts,
)
from rl.roles.weight_sync.model_adapter import (
    aggregate_direct_content_identity,
    build_model_weight_adapter,
    direct_fragment_record,
)
from rl.roles.weight_sync.streaming_full_gather import (
    StreamingBucketAck,
    StreamingContentIdentityAccumulator,
    StreamingFullGatherExecutor,
    StreamingMaterializedBucket,
    assemble_streaming_fragment,
    build_streaming_full_gather_plan,
    extract_streaming_contributions,
    pack_streaming_bucket,
)
from rl.roles.weight_sync.transfer import (
    ColocatedDirectReshardWeightTransfer,
    ColocatedFullGatherWeightTransfer,
    DirectReshardHCCLWeightTransfer,
    FullGatherHCCLWeightTransfer,
    _memory_acceptance_fields,
    _worker_memory_acceptance_fields,
    build_weight_transfer,
)


_DTYPE_NAME = "float32"
_ELEMENT_SIZE = 4


def test_memory_acceptance_fields_track_baseline_peak_and_release() -> None:
    """Summarize current Trainer and rollout memory without using lifetime peaks."""
    trainer = _memory_acceptance_fields(
        [
            {"allocated_bytes": 10, "reserved_bytes": 20, "host_rss_bytes": 30},
            {"allocated_bytes": 50, "reserved_bytes": 60, "host_rss_bytes": 70},
            {"allocated_bytes": 11, "reserved_bytes": 60, "host_rss_bytes": 31},
        ],
        prefix="trainer",
    )
    rollout = _worker_memory_acceptance_fields(
        [
            {
                "current_memory_allocated_bytes": 100,
                "current_memory_reserved_bytes": 200,
                "current_host_rss_bytes": 300,
            }
        ],
        [
            {
                "current_memory_allocated_bytes": 101,
                "current_memory_reserved_bytes": 200,
                "current_host_rss_bytes": 301,
            }
        ],
    )

    assert trainer["trainer_baseline_allocated_bytes"] == 10
    assert trainer["trainer_peak_current_allocated_bytes"] == 50
    assert trainer["trainer_post_release_allocated_bytes"] == 11
    assert trainer["trainer_post_release_reserved_bytes"] == 60
    assert rollout["rollout_baseline_memory_allocated_bytes"] == 100
    assert rollout["rollout_post_release_memory_allocated_bytes"] == 101


def _source(
    name: str,
    rank: int,
    global_shape: tuple[int, ...],
    starts: tuple[int, ...],
    lengths: tuple[int, ...],
) -> SourceTensorLayout:
    """Build one explicit FSDP source region."""
    return SourceTensorLayout(
        name=name,
        dtype_name=_DTYPE_NAME,
        element_size=_ELEMENT_SIZE,
        global_shape=global_shape,
        source_rank=rank,
        region=TensorRegion(starts, lengths),
    )


def _destination(
    name: str,
    global_shape: tuple[int, ...],
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    starts: Optional[tuple[int, ...]] = None,
    lengths: Optional[tuple[int, ...]] = None,
) -> DestinationTensorLayout:
    """Build one canonical rollout destination region."""
    starts = starts or (0,) * len(global_shape)
    lengths = lengths or global_shape
    return DestinationTensorLayout(
        name=name,
        dtype_name=_DTYPE_NAME,
        element_size=_ELEMENT_SIZE,
        global_shape=global_shape,
        tp_rank=tp_rank,
        tp_size=tp_size,
        placement="replicate" if tp_size == 1 else "shard",
        shard_dim=None if tp_size == 1 else 1,
        region=TensorRegion(starts, lengths),
    )


def _rank_states(global_tensor: torch.Tensor) -> dict[int, dict[str, torch.Tensor]]:
    """Shard one matrix unevenly across its column dimension."""
    return {
        0: {"weight": global_tensor[:, :3].clone()},
        1: {"weight": global_tensor[:, 3:].clone()},
    }


def _materialize_fragment(
    fragment: Any,
    rank_states: dict[int, dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Extract every rank contribution and assemble one canonical fragment."""
    contributions = []
    for rank, state_dict in rank_states.items():
        contributions.extend(
            extract_streaming_contributions(fragment, rank, state_dict)
        )
    return assemble_streaming_fragment(fragment, contributions)


def _content_identity(
    plan: Any,
    rank_states: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Hash one TP target in its deterministic canonical order."""
    accumulator = StreamingContentIdentityAccumulator(plan, 0)
    for bucket in plan.for_target(0):
        for fragment in bucket.entries:
            tensor = _materialize_fragment(fragment, rank_states)
            values = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            accumulator.add(fragment, values)
    return accumulator.finalize()


def _replicated_plan(bucket_size_bytes: int) -> tuple[Any, torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
    """Build a plan whose fragments cross uneven column shards."""
    global_tensor = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    sources = (
        _source("weight", 0, (2, 5), (0, 0), (2, 3)),
        _source("weight", 1, (2, 5), (0, 3), (2, 2)),
    )
    plan = build_streaming_full_gather_plan(
        sources,
        (_destination("weight", (2, 5)),),
        source_world_size=2,
        bucket_size_bytes=bucket_size_bytes,
    )
    return plan, global_tensor, _rank_states(global_tensor)


def test_streaming_plan_splits_oversized_tensor_and_covers_uneven_shards() -> None:
    """Bound every fragment while covering each destination byte exactly once."""
    plan, _global_tensor, _states = _replicated_plan(24)

    fragments = [
        fragment
        for bucket in plan.for_target(0)
        for fragment in bucket.entries
    ]
    covered = sum(fragment.numel for fragment in fragments)

    assert covered == 10, f"Unexpected coverage: actual={covered}, expected=10"
    assert plan.total_bytes == 40, (
        f"Unexpected transfer bytes: actual={plan.total_bytes}, expected=40"
    )
    assert all(fragment.num_bytes <= 24 for fragment in fragments), (
        f"Fragment exceeds bucket: sizes={[fragment.num_bytes for fragment in fragments]}"
    )
    assert all(bucket.total_bytes <= 24 for bucket in plan.for_target(0)), (
        f"Packed bucket exceeds limit: sizes={[bucket.total_bytes for bucket in plan.for_target(0)]}"
    )
    assert any(len(fragment.contributions) == 2 for fragment in fragments), (
        f"No fragment crosses source shards: contributions={[len(item.contributions) for item in fragments]}"
    )


def test_streaming_assemble_and_pack_reconstructs_destination() -> None:
    """Assemble source slices and replay packed metadata into a full destination."""
    plan, global_tensor, rank_states = _replicated_plan(16)
    destination = torch.empty_like(global_tensor)

    for bucket in plan.for_target(0):
        fragments = [
            _materialize_fragment(fragment, rank_states)
            for fragment in bucket.entries
        ]
        payload = pack_streaming_bucket(bucket, fragments)
        assert payload.packed_bytes <= 16, (
            f"Packed bytes exceed limit: actual={payload.packed_bytes}, expected<=16"
        )
        for fragment in bucket.entries:
            raw = payload.value.narrow(
                0,
                fragment.buffer_offset,
                fragment.num_bytes,
            )
            tensor = raw.view(torch.float32).view(fragment.lengths)
            target_slice = tuple(
                slice(start, start + length)
                for start, length in zip(
                    fragment.destination_starts,
                    fragment.destination_lengths,
                )
            )
            destination[target_slice].copy_(tensor)

    assert torch.equal(destination, global_tensor), (
        f"Destination differs: actual={destination}, expected={global_tensor}"
    )


def test_streaming_identity_does_not_depend_on_bucket_boundaries() -> None:
    """Hash identical canonical tensor bytes under two fragment granularities."""
    small_plan, _tensor, rank_states = _replicated_plan(16)
    large_plan, _tensor, large_rank_states = _replicated_plan(32)

    small_identity = _content_identity(small_plan, rank_states)
    large_identity = _content_identity(large_plan, large_rank_states)

    assert small_identity["digest"] == large_identity["digest"], (
        "Streaming digest changed with bucket size: "
        f"small={small_identity['digest']}, large={large_identity['digest']}"
    )
    assert small_identity["total_bytes"] == large_identity["total_bytes"] == 40, (
        f"Unexpected identity bytes: small={small_identity}, large={large_identity}"
    )
    assert small_identity["fragment_count"] != large_identity["fragment_count"], (
        f"Test did not vary fragmentation: small={small_identity}, large={large_identity}"
    )


def test_qwen3_moe_direct_and_full_gather_share_canonical_identity() -> None:
    """Both P8.2 strategies hash the same bounded packed-expert fragments."""
    class LocalExperts:
        """Mark the physical expert storage as the common EP1 leaf."""

        hyper_local_expert_leaf = True

    class Model:
        """Expose only the physical tensors required by the synthetic plan."""

        def __init__(self) -> None:
            """Initialize Model state."""
            self.parameters = {
                "model.layers.0.mlp.experts.w13_weight": torch.empty(2, 3, 4),
                "model.layers.0.mlp.experts.w2_weight": torch.empty(2, 2, 3),
            }

        def named_parameters(self) -> Any:
            """Expose fixture parameters through the model interface."""
            return self.parameters.items()

        @staticmethod
        def named_buffers() -> tuple[object, ...]:
            """Expose fixture buffers through the model interface."""
            return ()

        @staticmethod
        def named_modules() -> tuple[tuple[str, object], ...]:
            """Expose fixture modules through the model interface."""
            return (("model.layers.0.mlp.experts", LocalExperts()),)

    registration = SimpleNamespace(
        family="qwen3_moe",
        is_hyper=True,
        model=SimpleNamespace(tie_word_embeddings=False),
        actor_weight_name=lambda name: name,
    )
    state_dict = {
        "model.layers.0.mlp.experts.gate_up_proj": (
            torch.arange(24, dtype=torch.float32).view(2, 4, 3)
        ),
        "model.layers.0.mlp.experts.down_proj": (
            torch.arange(12, dtype=torch.float32).view(2, 3, 2)
        ),
    }
    adapter = build_model_weight_adapter(registration)
    sources = resolve_source_layouts(
        [adapter.direct_source_descriptions(state_dict, 0)]
    )
    destination_descriptions = [
        {
            "tp_rank": 0,
            "tp_size": 1,
            "tensors": worker_module._hyper_qwen3_moe_direct_tensors(  # pylint: disable=protected-access
                Model(),
                SimpleNamespace(num_experts=2, moe_intermediate_size=2, hidden_size=3),
                1,
            ),
        }
    ]
    shapes = {source.name: source.global_shape for source in sources}
    destinations = resolve_destination_layouts(destination_descriptions, shapes)
    direct = build_direct_reshard_plan(
        sources,
        destinations,
        source_world_size=1,
        bucket_size_bytes=16,
    )
    full = build_streaming_full_gather_plan(
        sources,
        destinations,
        source_world_size=1,
        bucket_size_bytes=16,
    )
    direct_records = {}
    for bucket in direct.for_route(0, 0):
        assert bucket.total_bytes <= 16
        for entry in bucket.entries:
            source_slice = tuple(
                slice(start, start + length)
                for start, length in zip(entry.source_starts, entry.lengths)
            )
            values = (
                state_dict[entry.source_key][source_slice]
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            key, record = direct_fragment_record(
                entry.name,
                entry.logical_starts,
                entry.lengths,
                entry.dtype_name,
                values,
            )
            direct_records[key] = record
    full_records = {}
    for bucket in full.for_target(0):
        assert bucket.total_bytes <= 16
        for fragment in bucket.entries:
            tensor = _materialize_fragment(fragment, {0: state_dict})
            values = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            key, record = direct_fragment_record(
                fragment.name,
                fragment.canonical_starts,
                fragment.lengths,
                fragment.dtype_name,
                values,
            )
            full_records[key] = record

    assert aggregate_direct_content_identity(direct_records) == (
        aggregate_direct_content_identity(full_records)
    )


def test_streaming_plan_transfers_tied_storage_once() -> None:
    """Keep a tied alias in the contract without planning duplicate payload bytes."""
    global_shape = (2, 3)
    sources = (
        _source("model.embed_tokens.weight", 0, global_shape, (0, 0), global_shape),
        _source("lm_head.weight", 0, global_shape, (0, 0), global_shape),
    )
    destinations = (
        _destination("model.embed_tokens.weight", global_shape),
        _destination("lm_head.weight", global_shape),
    )

    plan = build_streaming_full_gather_plan(
        sources,
        destinations,
        source_world_size=1,
        bucket_size_bytes=64,
        aliases={"lm_head.weight": "model.embed_tokens.weight"},
    )
    names = {
        fragment.name
        for bucket in plan.for_target(0)
        for fragment in bucket.entries
    }

    assert names == {"model.embed_tokens.weight"}, (
        f"Tied payload was not deduplicated: actual={names}"
    )
    assert plan.total_bytes == 24, (
        f"Tied storage bytes were duplicated: actual={plan.total_bytes}, expected=24"
    )
    assert [(alias.alias_name, alias.target_name) for alias in plan.aliases] == [
        ("lm_head.weight", "model.embed_tokens.weight")
    ], f"Unexpected alias contract: {plan.aliases}"


def test_streaming_executor_releases_each_acknowledged_bucket() -> None:
    """Keep at most one payload live and finish only after every exact ACK."""
    plan, _tensor, _rank_states = _replicated_plan(16)
    executor = StreamingFullGatherExecutor()
    events = []

    def materialize(bucket: Any) -> StreamingMaterializedBucket:
        """Materialize one bounded bucket for the transport callback."""
        gathered_bytes = sum(entry.num_bytes for entry in bucket.entries)
        return StreamingMaterializedBucket(
            torch.empty(bucket.total_bytes, dtype=torch.uint8),
            gathered_bytes,
            bucket.total_bytes,
        )

    def send(
        target_tp_rank: int,
        bucket_index: int,
        bucket: Any,
        _payload: StreamingMaterializedBucket,
    ) -> StreamingBucketAck:
        """Send one bucket and return its acknowledgement."""
        events.append(("send", bucket_index))
        return StreamingBucketAck(
            target_tp_rank,
            bucket_index,
            bucket.total_bytes,
            worker_count=2,
        )

    stats = executor.execute(
        plan,
        start=lambda: events.append(("start", None)),
        materialize_bucket=materialize,
        send_bucket=send,
        release_payload=lambda _payload: events.append(("release", None)),
        finish=lambda: events.append(("finish", None)),
        abort=lambda error: events.append(("abort", type(error).__name__)),
    )

    assert stats.bucket_count == plan.bucket_count, (
        f"Unexpected bucket count: actual={stats.bucket_count}, expected={plan.bucket_count}"
    )
    assert stats.acked_buckets == stats.released_buckets == plan.bucket_count, (
        f"ACK/release mismatch: stats={stats}, expected={plan.bucket_count}"
    )
    assert stats.max_packed_bytes <= plan.bucket_size_bytes, (
        f"Executor exceeded bucket: stats={stats}, limit={plan.bucket_size_bytes}"
    )
    assert stats.max_gathered_bytes <= plan.bucket_size_bytes
    assert stats.max_ipc_shared_bytes <= plan.bucket_size_bytes
    assert stats.max_transport_bytes <= 2 * plan.bucket_size_bytes
    assert stats.transport_buffer_count == 2
    assert stats.max_inflight_buckets == 1
    assert len(stats.buckets) == plan.bucket_count, (
        f"Missing per-bucket stats: actual={len(stats.buckets)}, expected={plan.bucket_count}"
    )
    for bucket_stats in stats.buckets:
        assert bucket_stats.gathered_bytes == bucket_stats.packed_bytes, (
            f"Gathered/packed bytes differ: {bucket_stats}"
        )
        assert bucket_stats.sent_bytes == bucket_stats.acked_bytes == 2 * bucket_stats.packed_bytes, (
            f"Logical delivery byte accounting differs: {bucket_stats}"
        )
        assert bucket_stats.released_bytes == bucket_stats.packed_bytes, (
            f"Released payload bytes differ: {bucket_stats}"
        )
    assert events[0] == ("start", None) and events[-1] == ("finish", None), (
        f"Unexpected transaction order: events={events}"
    )
    assert not any(event[0] == "abort" for event in events), (
        f"Successful transaction aborted: events={events}"
    )


def test_streaming_executor_aborts_and_retains_failed_payload() -> None:
    """Do not finish or release an unacknowledged payload until recovery is safe."""
    plan, _tensor, _rank_states = _replicated_plan(16)
    executor = StreamingFullGatherExecutor()
    events = []

    def materialize(bucket: Any) -> StreamingMaterializedBucket:
        """Materialize one bounded bucket for the transport callback."""
        return StreamingMaterializedBucket(
            torch.empty(bucket.total_bytes, dtype=torch.uint8),
            sum(entry.num_bytes for entry in bucket.entries),
            bucket.total_bytes,
        )

    def fail_send(
        _target_tp_rank: int,
        _bucket_index: int,
        _bucket: Any,
        _payload: StreamingMaterializedBucket,
    ) -> StreamingBucketAck:
        """Inject a send failure to check retained payload ownership."""
        raise RuntimeError("injected receive failure")

    with pytest.raises(RuntimeError, match="injected receive failure"):
        executor.execute(
            plan,
            start=lambda: events.append("start"),
            materialize_bucket=materialize,
            send_bucket=fail_send,
            release_payload=lambda _payload: events.append("release"),
            finish=lambda: events.append("finish"),
            abort=lambda _error: events.append("abort"),
        )

    assert events == ["start", "abort"], f"Unexpected failure order: events={events}"
    assert executor.failed_payload_count == 1, (
        f"Failed payload was not retained: count={executor.failed_payload_count}"
    )
    executor.release_failed_payloads(lambda _payload: events.append("recovery_release"))
    assert executor.failed_payload_count == 0, (
        f"Failed payload was not released: count={executor.failed_payload_count}"
    )
    assert events[-1] == "recovery_release", f"Unexpected recovery events: {events}"


def test_bounded_full_gather_factory_covers_qwen3_colocated_and_disjoint() -> None:
    """Use the same gather-first contract with IPC and HCCL transports."""
    rollout_model = SimpleNamespace(
        family="qwen3",
        is_hyper=True,
        model=SimpleNamespace(tie_word_embeddings=False),
    )
    config = {
        "tensor_parallel_size": 2,
        "weight_sync": {
            "strategy": "full_gather",
            "fallback_strategy": "none",
            "bucket_size_mb": 16,
        },
    }

    _validate_vllm_weight_sync(config, "colocated", rollout_model, {"tp": 2})
    transfer = build_weight_transfer(
        "colocated",
        rollout_model,
        tensor_parallel_size=2,
        data_parallel_size=2,
        bucket_size_bytes=16 * 2**20,
        strategy="full_gather",
        fallback_strategy="none",
    )

    assert isinstance(transfer, ColocatedFullGatherWeightTransfer), (
        f"Unexpected full-gather transfer: actual={type(transfer)}"
    )
    _validate_vllm_weight_sync(config, "disjoint", rollout_model, {"tp": 2})
    disjoint = build_weight_transfer(
        "disjoint",
        rollout_model,
        tensor_parallel_size=2,
        data_parallel_size=2,
        strategy="full_gather",
    )
    assert isinstance(disjoint, FullGatherHCCLWeightTransfer)


def test_removed_full_gather_implementation_selector_is_rejected() -> None:
    """The public config exposes strategies, not historical implementations."""
    rollout_model = SimpleNamespace(family="qwen3", is_hyper=True)
    with pytest.raises(ValueError, match="full_gather_implementation was removed"):
        _validate_vllm_weight_sync(
            {
                "weight_sync": {
                    "strategy": "full_gather",
                    "full_gather_implementation": "legacy",
                }
            },
            "colocated",
            rollout_model,
            {},
        )


@pytest.mark.parametrize("family", ["qwen3", "deepseek_v3"])
@pytest.mark.parametrize("is_hyper", [False, True])
def test_all_colocated_models_use_bounded_weight_transfers(
    family: str,
    is_hyper: bool,
) -> None:
    """Native and Hyper model families share the bounded full/direct paths."""
    model = SimpleNamespace(
        family=family,
        is_hyper=is_hyper,
        model=SimpleNamespace(tie_word_embeddings=False),
    )

    full = build_weight_transfer(
        "colocated",
        model,
        strategy="full_gather",
        fallback_strategy="none",
    )
    direct = build_weight_transfer(
        "colocated",
        model,
        strategy="direct_reshard",
        fallback_strategy="none",
    )

    assert full.__class__ is ColocatedFullGatherWeightTransfer
    assert direct.__class__ is ColocatedDirectReshardWeightTransfer


@pytest.mark.parametrize("is_hyper", [False, True])
def test_all_qwen3_disjoint_models_use_bounded_weight_transfers(
    is_hyper: bool,
) -> None:
    """Qwen3 Native and Hyper disjoint paths share bounded HCCL transfers."""
    model = SimpleNamespace(
        family="qwen3",
        is_hyper=is_hyper,
        model=SimpleNamespace(tie_word_embeddings=False),
    )

    full = build_weight_transfer(
        "disjoint",
        model,
        strategy="full_gather",
        fallback_strategy="none",
    )
    direct = build_weight_transfer(
        "disjoint",
        model,
        strategy="direct_reshard",
        fallback_strategy="none",
    )

    assert full.__class__ is FullGatherHCCLWeightTransfer
    assert direct.__class__ is DirectReshardHCCLWeightTransfer
