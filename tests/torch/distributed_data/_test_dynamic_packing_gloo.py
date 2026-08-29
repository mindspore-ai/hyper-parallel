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
"""Four-process CPU/Gloo workers for distributed dynamic packing."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    DistributedPackingPlan,
    SampleMetadata,
    WorkloadCost,
    build_distributed_dataloader,
    default_pack_fn,
)

_WORLD_SIZE = 4
_DATASET_SIZE = 8
_CONSTRUCTOR_RANKS = (0, 2)


class _RawDataset:
    """Return individual raw samples so redistribution precedes packing."""

    def __init__(self, size: int = _DATASET_SIZE) -> None:
        """Store the rank-local Dataset length."""
        self._size = size

    def __len__(self) -> int:
        """Return the complete deterministic epoch size."""
        return self._size

    def __getitem__(self, index: int) -> dict[str, int]:
        """Return one sample carrying its stable identifier and token cost."""
        return {"sample_id": index, "pack_tokens": 1}


def _metadata_fn(sample: dict[str, int]) -> SampleMetadata:
    return SampleMetadata(
        pack_tokens=sample["pack_tokens"],
        cost=WorkloadCost(llm=1.0),
        sample_id=sample["sample_id"],
    )


def _failing_metadata_fn(sample: dict[str, int]) -> SampleMetadata:
    if sample["sample_id"] == 1:
        raise ValueError("injected metadata failure")
    return _metadata_fn(sample)


def _pack_fn(samples: Sequence[dict[str, int]], seq_len: int) -> dict[str, Any]:
    token_count = sum(sample["pack_tokens"] for sample in samples)
    if token_count > seq_len:
        raise ValueError(f"Test pack received {token_count} tokens for seq_len={seq_len}.")
    return {
        "sample_ids": tuple(sample["sample_id"] for sample in samples),
        "token_count": token_count,
    }


def _collate_fn(packed_sequences: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(packed_sequences)


def _all_gather_object(value: Any) -> tuple[Any, ...]:
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return tuple(gathered)


def _assert_same_model_parallel_batches(outputs: tuple[Any, ...]) -> None:
    for first_rank, second_rank in ((0, 1), (2, 3)):
        assert outputs[first_rank] == outputs[second_rank], (
            f"MP peers must receive the same constructed batch: "
            f"rank{first_rank}={outputs[first_rank]!r}, rank{second_rank}={outputs[second_rank]!r}."
        )


def _assert_exactly_once(outputs: tuple[Any, ...]) -> None:
    global_sample_ids = []
    for constructor_rank in _CONSTRUCTOR_RANKS:
        local_batch = outputs[constructor_rank]
        assert len(local_batch) == 1, (
            f"Each constructor must produce one packed sequence: "
            f"rank={constructor_rank}, batch={local_batch!r}."
        )
        packed_sequence = local_batch[0]
        assert packed_sequence["token_count"] == 4, (
            f"Each DP constructor should dynamically pack four one-token samples: "
            f"rank={constructor_rank}, packed={packed_sequence!r}."
        )
        global_sample_ids.extend(packed_sequence["sample_ids"])

    expected_sample_ids = list(range(_DATASET_SIZE))
    assert sorted(global_sample_ids) == expected_sample_ids, (
        f"The two DP constructors must emit the complete epoch exactly once: "
        f"expected={expected_sample_ids!r}, got={global_sample_ids!r}."
    )
    assert len(global_sample_ids) == len(set(global_sample_ids)), (
        f"A raw sample may be constructed only once globally: sample_ids={global_sample_ids!r}."
    )


def _assert_plan_matches_outputs(plan: DistributedPackingPlan, outputs: tuple[Any, ...]) -> None:
    routes = []
    for data_rank, constructor_rank in enumerate(_CONSTRUCTOR_RANKS):
        constructor = plan.constructor_for(data_rank)
        planned_sample_ids = tuple(
            sample.key.source_index
            for packing_bin in constructor.bins
            for sample in packing_bin.samples
        )
        output_sample_ids = tuple(
            sample_id
            for packed_sequence in outputs[constructor_rank]
            for sample_id in packed_sequence["sample_ids"]
        )
        assert output_sample_ids == planned_sample_ids, (
            f"Constructor output must preserve its sample-level packing plan: "
            f"rank={constructor_rank}, expected={planned_sample_ids!r}, got={output_sample_ids!r}."
        )
        routes.extend(
            (sample.key.source_rank, constructor_rank)
            for packing_bin in constructor.bins
            for sample in packing_bin.samples
        )

    assert any(source_rank != target_rank for source_rank, target_rank in routes), (
        f"The test epoch must exercise cross-rank sample redistribution: routes={routes!r}."
    )


def _assert_collective_stop(loader: Any) -> None:
    for attempt in range(2):
        stopped = False
        try:
            next(loader)
        except StopIteration:
            stopped = True
        assert stopped, (
            f"Every rank must observe StopIteration after the complete epoch: attempt={attempt}, stopped={stopped}."
        )
    dist.monitored_barrier(timeout=timedelta(seconds=30))


def _assert_collective_build_error(
        mesh: Any,
        dataset: _RawDataset,
        config: DistributedDatasetConfig,
        expected_message: str,
) -> None:
    error_type = None
    error_message = None
    try:
        build_distributed_dataloader(
            dataset,
            mesh,
            config,
            metadata_fn=_metadata_fn,
            pack_fn=_pack_fn,
            collate_fn=_collate_fn,
        )
    except Exception as exc:  # The assertion below verifies the public error type.
        error_type = type(exc).__name__
        error_message = str(exc)

    statuses = _all_gather_object((error_type, error_message))
    _assert_build_error_statuses(statuses, expected_message)


def _assert_build_error_statuses(statuses: tuple[Any, ...], expected_message: str) -> None:
    expected_types = tuple("ValueError" for _ in range(_WORLD_SIZE))
    actual_types = tuple(status[0] for status in statuses)
    assert actual_types == expected_types, (
        f"Every WORLD rank must receive ValueError from build preflight: "
        f"expected={expected_types!r}, got={actual_types!r}."
    )
    messages = tuple(status[1] for status in statuses)
    assert all(message == messages[0] for message in messages), (
        f"Build preflight must report the identical error on every WORLD rank: messages={messages!r}."
    )
    assert expected_message in messages[0].lower(), (
        f"Build preflight must identify the mismatched field: "
        f"expected_fragment={expected_message!r}, got={messages[0]!r}."
    )
    dist.monitored_barrier(timeout=timedelta(seconds=30))


def _assert_callback_mode_build_error(mesh: Any) -> None:
    callback_options: dict[str, Any] = {"metadata_fn": _metadata_fn}
    if dist.get_rank() == _WORLD_SIZE - 1:
        callback_options["pack_fn"] = _pack_fn
    error_type = None
    error_message = None
    try:
        build_distributed_dataloader(
            _RawDataset(),
            mesh,
            DistributedDatasetConfig(
                seq_len=4,
                local_batch_size=1,
                dp_dim_names=("dp",),
                source_loader_ranks=(0, 1, 2, 3),
                buffer_size_multiplier=1.0,
                max_buffered_samples=8,
                cpu_backend="gloo",
            ),
            **callback_options,
        )
    except Exception as exc:  # The assertion below verifies the public error type.
        error_type = type(exc).__name__
        error_message = str(exc)
    statuses = _all_gather_object((error_type, error_message))
    _assert_build_error_statuses(statuses, "build configuration mismatch")


def _assert_build_preflight_errors(mesh: Any) -> None:
    rank = dist.get_rank()
    mismatched_seq_len = 5 if rank == _WORLD_SIZE - 1 else 4
    _assert_collective_build_error(
        mesh,
        _RawDataset(),
        DistributedDatasetConfig(
            seq_len=mismatched_seq_len,
            local_batch_size=1,
            dp_dim_names=("dp",),
            source_loader_ranks=(0, 1, 2, 3),
            buffer_size_multiplier=1.0,
            max_buffered_samples=8,
            cpu_backend="gloo",
        ),
        "build configuration mismatch",
    )

    mismatched_dataset_size = _DATASET_SIZE - 1 if rank == _WORLD_SIZE - 1 else _DATASET_SIZE
    _assert_collective_build_error(
        mesh,
        _RawDataset(mismatched_dataset_size),
        DistributedDatasetConfig(
            seq_len=4,
            local_batch_size=1,
            dp_dim_names=("dp",),
            source_loader_ranks=(0, 1, 2, 3),
            buffer_size_multiplier=1.0,
            max_buffered_samples=8,
            cpu_backend="gloo",
        ),
        "source dataset length mismatch",
    )
    _assert_callback_mode_build_error(mesh)


def _assert_explicit_default_pack_is_equivalent(mesh: Any) -> None:
    rank = dist.get_rank()
    callback_options: dict[str, Any] = {"metadata_fn": _metadata_fn}
    if rank == _WORLD_SIZE - 1:
        callback_options["pack_fn"] = default_pack_fn
    loader = build_distributed_dataloader(
        _RawDataset(0) if rank in _CONSTRUCTOR_RANKS else None,
        mesh,
        DistributedDatasetConfig(
            seq_len=4,
            local_batch_size=1,
            dp_dim_names=("dp",),
            source_loader_ranks=None,
            buffer_size_multiplier=1.0,
            max_buffered_samples=8,
            cpu_backend="gloo",
        ),
        **callback_options,
    )
    _assert_collective_stop(loader)


def _assert_checkpoint_step_error_is_collective(mesh: Any) -> None:
    loader = build_distributed_dataloader(
        _RawDataset(),
        mesh,
        DistributedDatasetConfig(
            seq_len=4,
            local_batch_size=1,
            dp_dim_names=("dp",),
            source_loader_ranks=(0, 1, 2, 3),
            buffer_size_multiplier=1.0,
            max_buffered_samples=8,
            cpu_backend="gloo",
        ),
        metadata_fn=_metadata_fn,
        pack_fn=_pack_fn,
        collate_fn=_collate_fn,
    )
    state = loader.state_dict()
    if dist.get_rank() == _WORLD_SIZE - 1:
        state["step"] = 1
    loader.load_state_dict(state)

    error_type = None
    error_message = None
    try:
        next(loader)
    except Exception as exc:  # The collective contract is asserted after every rank exits.
        error_type = type(exc).__name__
        error_message = str(exc)
    statuses = _all_gather_object((error_type, error_message))
    error_types = tuple(status[0] for status in statuses)
    assert all(error_type == error_types[0] and error_type is not None for error_type in error_types), (
        f"Every rank must receive the same checkpoint-step error type: error_types={error_types!r}."
    )
    messages = tuple(status[1] for status in statuses)
    assert all(message == messages[0] for message in messages), (
        f"Every rank must receive the same checkpoint-step error: messages={messages!r}."
    )
    assert "step" in messages[0].lower(), (
        f"The synchronized checkpoint error must identify the step mismatch: error={messages[0]!r}."
    )
    dist.monitored_barrier(timeout=timedelta(seconds=30))


def _assert_model_peer_checkpoint_step_error_is_collective(mesh: Any) -> None:
    rank = dist.get_rank()
    loader = build_distributed_dataloader(
        _RawDataset() if rank in _CONSTRUCTOR_RANKS else None,
        mesh,
        DistributedDatasetConfig(
            seq_len=4,
            local_batch_size=1,
            dp_dim_names=("dp",),
            source_loader_ranks=None,
            buffer_size_multiplier=1.0,
            max_buffered_samples=8,
            cpu_backend="gloo",
        ),
        metadata_fn=_metadata_fn,
        pack_fn=_pack_fn,
        collate_fn=_collate_fn,
    )
    state = loader.state_dict()
    if rank == 1:
        state["step"] = 1
    loader.load_state_dict(state)

    error_type = None
    error_message = None
    try:
        next(loader)
    except Exception as exc:  # The assertions below verify collective failure semantics.
        error_type = type(exc).__name__
        error_message = str(exc)
    statuses = _all_gather_object((error_type, error_message))
    error_types = tuple(status[0] for status in statuses)
    assert all(error_type == error_types[0] and error_type is not None for error_type in error_types), (
        f"Every rank must receive the same MP-state error type: error_types={error_types!r}."
    )
    messages = tuple(status[1] for status in statuses)
    assert all(message == messages[0] for message in messages), (
        f"Every rank must receive the same MP-state error: messages={messages!r}."
    )
    normalized_message = messages[0].lower()
    assert "step" in normalized_message or "state" in normalized_message, (
        f"The synchronized MP error must identify the step/state mismatch: error={messages[0]!r}."
    )

    post_state = loader.state_dict()
    expected_step = 1 if rank == 1 else 0
    assert post_state["step"] == expected_step, (
        f"MP-state preflight must not advance the rank-local iterator: "
        f"rank={rank}, expected_step={expected_step}, got={post_state['step']}."
    )
    assert post_state["last_plan_id"] is None and loader.last_plan is None, (
        f"MP-state preflight must fail before planning: "
        f"rank={rank}, last_plan_id={post_state['last_plan_id']!r}, last_plan={loader.last_plan!r}."
    )
    if rank in _CONSTRUCTOR_RANKS:
        source_state = post_state["source_loader"]
        assert source_state["next_ordinal"] == 0 and not source_state["buffer"], (
            f"MP-state preflight must fail before Source reads or commits samples: "
            f"rank={rank}, next_ordinal={source_state['next_ordinal']}, buffer={source_state['buffer']!r}."
        )
    dist.monitored_barrier(timeout=timedelta(seconds=30))


def _run_epoch(mesh: Any, source_loader_ranks: tuple[int, ...] | None) -> None:
    rank = dist.get_rank()
    effective_sources = source_loader_ranks or _CONSTRUCTOR_RANKS
    dataset = _RawDataset() if rank in effective_sources else None
    loader = build_distributed_dataloader(
        dataset,
        mesh,
        DistributedDatasetConfig(
            seq_len=4,
            local_batch_size=1,
            dp_dim_names=("dp",),
            source_loader_ranks=source_loader_ranks,
            buffer_size_multiplier=1.0,
            max_buffered_samples=8,
            cpu_backend="gloo",
        ),
        metadata_fn=_metadata_fn,
        pack_fn=_pack_fn,
        collate_fn=_collate_fn,
    )

    outputs = _all_gather_object(next(loader))
    _assert_same_model_parallel_batches(outputs)
    _assert_exactly_once(outputs)

    gathered_plans = _all_gather_object(loader.last_plan)
    for source_rank in effective_sources:
        assert gathered_plans[source_rank] == gathered_plans[effective_sources[0]], (
            f"All data-plane ranks must receive the same plan: "
            f"rank{effective_sources[0]}={gathered_plans[effective_sources[0]]!r}, "
            f"rank{source_rank}={gathered_plans[source_rank]!r}."
        )
    for model_only_rank in set(range(_WORLD_SIZE)) - set(effective_sources):
        assert gathered_plans[model_only_rank] is None, (
            f"A model-only rank must wait for MP delivery without joining planning: "
            f"rank={model_only_rank}, plan={gathered_plans[model_only_rank]!r}."
        )
    reference_plan = gathered_plans[effective_sources[0]]
    assert isinstance(reference_plan, DistributedPackingPlan), (
        f"A data-plane rank must retain the last distributed plan: plan={reference_plan!r}."
    )
    _assert_plan_matches_outputs(reference_plan, outputs)
    _assert_collective_stop(loader)


def _run_default_constructor_epoch(mesh: Any) -> None:
    rank = dist.get_rank()
    loader = build_distributed_dataloader(
        _RawDataset() if rank in _CONSTRUCTOR_RANKS else None,
        mesh,
        DistributedDatasetConfig(
            seq_len=2,
            local_batch_size=2,
            dp_dim_names=("dp",),
            source_loader_ranks=None,
            buffer_size_multiplier=1.0,
            max_buffered_samples=8,
            cpu_backend="gloo",
        ),
        metadata_fn=_metadata_fn,
    )

    outputs = _all_gather_object(next(loader))
    _assert_same_model_parallel_batches(outputs)
    global_sample_ids = []
    for constructor_rank in _CONSTRUCTOR_RANKS:
        local_batch = outputs[constructor_rank]
        assert isinstance(local_batch, tuple) and len(local_batch) == 2, (
            f"The default constructor must return a tuple of packing bins: "
            f"rank={constructor_rank}, batch={local_batch!r}."
        )
        assert all(isinstance(packing_bin, tuple) and packing_bin for packing_bin in local_batch), (
            f"Every default packing bin must be a non-empty tuple of raw samples: "
            f"rank={constructor_rank}, batch={local_batch!r}."
        )
        raw_samples = tuple(sample for packing_bin in local_batch for sample in packing_bin)
        assert all(isinstance(sample, dict) for sample in raw_samples), (
            f"The default constructor must preserve raw sample payloads: "
            f"rank={constructor_rank}, raw_samples={raw_samples!r}."
        )
        global_sample_ids.extend(sample["sample_id"] for sample in raw_samples)

    expected_sample_ids = list(range(_DATASET_SIZE))
    assert sorted(global_sample_ids) == expected_sample_ids, (
        f"Default construction must emit every raw sample exactly once globally: "
        f"expected={expected_sample_ids!r}, got={global_sample_ids!r}."
    )
    assert len(global_sample_ids) == len(set(global_sample_ids)), (
        f"Default construction must not duplicate raw samples: sample_ids={global_sample_ids!r}."
    )
    _assert_collective_stop(loader)


def _assert_metadata_error_is_collective(mesh: Any) -> None:
    loader = build_distributed_dataloader(
        _RawDataset(),
        mesh,
        DistributedDatasetConfig(
            seq_len=4,
            local_batch_size=1,
            dp_dim_names=("dp",),
            source_loader_ranks=(0, 1, 2, 3),
            buffer_size_multiplier=1.0,
            max_buffered_samples=8,
            cpu_backend="gloo",
        ),
        metadata_fn=_failing_metadata_fn,
        pack_fn=_pack_fn,
        collate_fn=_collate_fn,
    )
    error = None
    try:
        next(loader)
    except RuntimeError as exc:
        error = str(exc)
    errors = _all_gather_object(error)
    assert all(message == errors[0] for message in errors), (
        f"Every model rank must receive the same Source Loader error: errors={errors!r}."
    )
    assert "injected metadata failure" in errors[0], (
        f"The collective failure must retain its root cause: error={errors[0]!r}."
    )
    dist.monitored_barrier(timeout=timedelta(seconds=30))


def test_dynamic_packing_dp2_mp2_gloo() -> None:
    """Verify Source Loader, Planner, and Data Constructor delivery on DP=2/MP=2."""
    dist.init_process_group(backend="gloo")
    try:
        world_size = dist.get_world_size()
        assert world_size == _WORLD_SIZE, (
            f"This system test requires four Gloo processes: expected={_WORLD_SIZE}, got={world_size}."
        )
        mesh = init_device_mesh(
            "cpu",
            mesh_shape=(2, 2),
            mesh_dim_names=("dp", "mp"),
        )

        # The WORLD preflight must reject rank-local build disagreement before
        # any rank enters a differently shaped service-group creation path.
        _assert_build_preflight_errors(mesh)

        # Explicitly passing the public identity packer is semantically the
        # same callback mode as omitting pack_fn on the other WORLD ranks.
        _assert_explicit_default_pack_is_equivalent(mesh)

        # Rank 1 is a pure MP peer in the default Source topology. Its divergent
        # checkpoint state must fail on WORLD before Sources read any sample.
        _assert_model_peer_checkpoint_step_error_is_collective(mesh)

        # A fresh default loader must remain usable immediately after the
        # synchronized MP-state failure, proving collective order was preserved.
        _run_epoch(mesh, source_loader_ranks=None)

        # Without construction callbacks, the built-in constructor returns
        # local-batch and packing-bin structure as nested immutable tuples.
        _run_default_constructor_epoch(mesh)

        # All ranks act as Sources, so payloads from MP peers must route to a
        # Data Constructor before their constructed batches return over MP.
        _run_epoch(mesh, source_loader_ranks=(0, 1, 2, 3))

        # A single Source callback failure must reach every model rank before
        # any participant enters payload or MP collectives in a different order.
        _assert_metadata_error_is_collective(mesh)

        # Rank-local checkpoints may be loaded independently, but iteration
        # must reject a divergent step collectively before planning or routing.
        _assert_checkpoint_step_error_is_collective(mesh)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
