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
"""CPU unit tests for bounded streaming full-gather planning and execution."""
# pylint: disable=forbidden-backend-import,missing-public-docstring,missing-public-type-hints

from typing import Iterable

import torch

from rl.roles.weight_sync.layout import (
    DestinationTensorLayout,
    SourceTensorLayout,
    TensorRegion,
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


def _sources(name: str = "weight") -> tuple[SourceTensorLayout, ...]:
    """Return two row-sharded sources covering one 4x2 tensor."""
    return tuple(
        SourceTensorLayout(
            name=name,
            dtype_name="float32",
            element_size=4,
            global_shape=(4, 2),
            source_rank=rank,
            region=TensorRegion((rank * 2, 0), (2, 2)),
        )
        for rank in range(2)
    )


def _destinations(
    *,
    name: str = "weight",
    tp_size: int = 2,
) -> tuple[DestinationTensorLayout, ...]:
    """Return replicated TP1 or row-sharded TP2 destination layouts."""
    return tuple(
        DestinationTensorLayout(
            name=name,
            dtype_name="float32",
            element_size=4,
            global_shape=(4, 2),
            tp_rank=rank,
            tp_size=tp_size,
            placement="replicate" if tp_size == 1 else "shard",
            shard_dim=None if tp_size == 1 else 0,
            region=(
                TensorRegion((0, 0), (4, 2))
                if tp_size == 1
                else TensorRegion((rank * 2, 0), (2, 2))
            ),
        )
        for rank in range(tp_size)
    )


def _fragment_values(
    tensor: torch.Tensor,
    fragments: Iterable[object],
) -> Iterable[tuple[object, bytes]]:
    """Yield canonical bytes for each planned fragment."""
    for fragment in fragments:
        slices = tuple(
            slice(start, start + length)
            for start, length in zip(fragment.canonical_starts, fragment.lengths)
        )
        yield fragment, tensor[slices].contiguous().numpy().tobytes()


def test_streaming_plan_builds_deterministic_bounded_tp_buckets() -> None:
    """The plan covers both TP ranks once, removes aliases, and respects the byte bound."""
    sources = (*_sources(), *_sources("lm_head.weight"))
    destinations = (*_destinations(), *_destinations(name="lm_head.weight"))

    plan = build_streaming_full_gather_plan(
        sources,
        destinations,
        source_world_size=2,
        bucket_size_bytes=16,
        aliases={"lm_head.weight": "weight"},
    )

    assert plan.destination_tp_size == 2, (
        f"Unexpected destination TP size: expected=2, actual={plan.destination_tp_size}"
    )
    assert plan.bucket_count == 2, (
        f"Unexpected bucket count: expected=2, actual={plan.bucket_count}"
    )
    assert plan.fragment_count == 2, (
        f"Unexpected fragment count: expected=2, actual={plan.fragment_count}"
    )
    assert plan.total_bytes == 32, (
        f"Unexpected transfer bytes: expected=32, actual={plan.total_bytes}"
    )
    assert [(alias.alias_name, alias.target_name) for alias in plan.aliases] == [
        ("lm_head.weight", "weight")
    ]
    assert all(
        bucket.total_bytes <= plan.bucket_size_bytes
        for tp_rank in range(2)
        for bucket in plan.for_target(tp_rank)
    )


def test_streaming_fragments_reconstruct_and_pack_expected_tp_values() -> None:
    """Source contributions reconstruct and pack the canonical tensor without reordering."""
    plan = build_streaming_full_gather_plan(
        _sources(),
        _destinations(tp_size=1),
        source_world_size=2,
        bucket_size_bytes=32,
    )
    bucket = plan.for_target(0)[0]
    fragment = bucket.entries[0]
    shards = (
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
        torch.arange(4, 8, dtype=torch.float32).reshape(2, 2),
    )
    contributions = tuple(
        contribution
        for rank, shard in enumerate(shards)
        for contribution in extract_streaming_contributions(
            fragment,
            rank,
            {"weight": shard},
        )
    )

    assembled = assemble_streaming_fragment(fragment, contributions)
    packed = pack_streaming_bucket(bucket, (assembled,))

    torch.testing.assert_close(
        assembled,
        torch.arange(8, dtype=torch.float32).reshape(4, 2),
    )
    torch.testing.assert_close(
        packed.value.view(torch.float32),
        torch.arange(8, dtype=torch.float32),
    )
    assert packed.gathered_bytes == 32, (
        f"Unexpected gathered bytes: expected=32, actual={packed.gathered_bytes}"
    )
    assert packed.packed_bytes == 32, (
        f"Unexpected packed bytes: expected=32, actual={packed.packed_bytes}"
    )


def test_streaming_executor_acknowledges_releases_and_reports_bounds() -> None:
    """The executor starts once, releases only acknowledged buckets, and reports bounded storage."""
    plan = build_streaming_full_gather_plan(
        _sources(),
        _destinations(),
        source_world_size=2,
        bucket_size_bytes=16,
    )
    events: list[str] = []

    def materialize(bucket):
        events.append(f"materialize:{bucket.target_tp_rank}")
        return StreamingMaterializedBucket(
            value=torch.zeros(bucket.total_bytes, dtype=torch.uint8),
            gathered_bytes=sum(entry.num_bytes for entry in bucket.entries),
            packed_bytes=bucket.total_bytes,
        )

    def send(tp_rank, bucket_index, bucket, _payload):
        events.append(f"send:{tp_rank}:{bucket_index}")
        return StreamingBucketAck(tp_rank, bucket_index, bucket.total_bytes, 2)

    executor = StreamingFullGatherExecutor()
    stats = executor.execute(
        plan,
        start=lambda: events.append("start"),
        materialize_bucket=materialize,
        send_bucket=send,
        release_payload=lambda value: events.append(f"release:{value.numel()}"),
        finish=lambda: events.append("finish"),
        abort=lambda error: events.append(f"abort:{error!r}"),
        transport="ipc",
    )

    assert events == [
        "start",
        "materialize:0",
        "send:0:0",
        "release:16",
        "materialize:1",
        "send:1:0",
        "release:16",
        "finish",
    ]
    assert stats.bucket_count == stats.acked_buckets == stats.released_buckets == 2
    assert stats.fragment_count == 2
    assert stats.max_gathered_bytes == stats.max_packed_bytes == 16
    assert stats.max_ipc_shared_bytes == stats.max_transport_bytes // 2 == 16
    assert stats.max_hccl_bytes == 0
    assert stats.max_inflight_buckets == 1
    assert executor.failed_payload_count == 0


def test_streaming_content_identity_is_independent_of_bucket_partition() -> None:
    """The same canonical bytes produce one digest across different fragment sizes."""
    tensor = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    identities = []
    for bucket_size in (16, 32):
        plan = build_streaming_full_gather_plan(
            _sources(),
            _destinations(tp_size=1),
            source_world_size=2,
            bucket_size_bytes=bucket_size,
        )
        accumulator = StreamingContentIdentityAccumulator(plan, 0)
        fragments = (
            fragment
            for bucket in plan.for_target(0)
            for fragment in bucket.entries
        )
        for fragment, values in _fragment_values(tensor, fragments):
            accumulator.add(fragment, values)
        identities.append(accumulator.finalize())

    assert identities[0]["digest"] == identities[1]["digest"], (
        f"Canonical digest changed with bucket size: identities={identities}"
    )
    assert identities[0]["total_bytes"] == identities[1]["total_bytes"] == 32
    assert identities[0]["tensor_count"] == identities[1]["tensor_count"] == 1
