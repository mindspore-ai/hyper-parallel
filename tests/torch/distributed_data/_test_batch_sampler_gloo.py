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
"""DP2/TP2 native BatchSampler ownership, metadata, and checkpoint coverage."""

import logging
import random
from collections import Counter
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import default_collate

from hyper_parallel.data.batching.build_collate_fn import OmniCollator, TextPackingCollator
from hyper_parallel.data.batching.build_dataloader import build_dataloader as build_trainer_dataloader
from hyper_parallel.data.batching.get_batch import OmniParallelBatch, TextParallelBatch
from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter
from hyper_parallel.data.parallel import build_dataset_batch_sampler, create_dataloader_parallel_context
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig, SampleMetadata, TensorFieldSpec, WorkloadCost, build_distributed_dataloader,
    collate_indexed_text_sequences, pack_indexed_text_samples, split_packed_text_sample,
)
from hyper_parallel.distributed_data.balance_logging import logger as balance_logger
from hyper_parallel.trainer.config.target import Target


class _Dataset:
    """Track native index reads independently from metadata lookup."""

    def __init__(self) -> None:
        """Initialize the payload read log."""
        self.reads = []

    def __len__(self) -> int:
        """Return a size with an incomplete distributed tail."""
        return 18

    def __getitem__(self, index: int) -> dict:
        """Record a read and return original model fields."""
        self.reads.append(index)
        return {"id": index, "labels": [index + 1], "position_ids": [7], "loss_mask": [1]}


def _metadata(sample: dict) -> SampleMetadata:
    return SampleMetadata(1, cost=WorkloadCost(llm=9 if sample["id"] % 4 < 2 else 1), sample_id=sample["id"])


def _sampler(sampler_type: str, data_sharding: bool) -> object:
    return build_dataset_batch_sampler(
        total_samples=18, micro_batch_size=2, global_batch_size=8,
        dp_rank=dist.get_rank() // 2, dp_world_size=2,
        sampler_type=sampler_type, data_sharding=data_sharding, seed=17,
        index_mapping=[index // 2 for index in range(18)] if sampler_type == "single" else None,
    )


def _plain(value: object) -> object:
    """Convert test outputs for cross-rank equality and occurrence counting."""
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    return value


def _native_collate(samples: list) -> tuple:
    """Use fixed tensor fields for model-parallel native sample tests."""
    return tuple({key: torch.as_tensor(value) for key, value in sample.items()} for sample in samples)


def _native_spec(size: int) -> tuple:
    """Declare the fixture's tensor layout on every peer."""
    return tuple({"id": TensorFieldSpec((), torch.int64),
                  "labels": TensorFieldSpec((1,), torch.int64),
                  "position_ids": TensorFieldSpec((1,), torch.int64),
                  "loss_mask": TensorFieldSpec((1,), torch.int64)} for _ in range(size))


def _gather(value: object) -> list:
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, _plain(value))
    return gathered


def _check_round_payloads(batch: tuple[dict, ...], expected_local_indices: list[int]) -> None:
    """Check selected occurrences and whole-sample fields across DP and TP peers."""
    all_batches = _gather(batch)
    expected_indices = _gather(expected_local_indices)
    expected = Counter(expected_indices[0] + expected_indices[2])
    actual = Counter(sample["id"] for target in (0, 2) for sample in all_batches[target])
    assert actual == expected, f"Native round membership changed: actual={actual}, expected={expected}"
    assert all_batches[0] == all_batches[1] and all_batches[2] == all_batches[3], (
        f"TP peers received different batches: batches={all_batches}"
    )
    for sample in _plain(batch):
        assert sample["labels"] == [sample["id"] + 1] and sample["position_ids"] == [7], (
            f"Original native fields changed: sample={sample}"
        )


def _check_dataset_reads(dataset: _Dataset, reference: list[list[int]]) -> None:
    """Compare physical reads across ranks with all selected native occurrences."""
    all_reads = _gather(dataset.reads)
    actual_reads = Counter(index for rank_reads in all_reads for index in rank_reads)
    expected_reads = Counter(index for refs in _gather(reference)[::2] for row in refs for index in row)
    assert actual_reads == expected_reads, (
        f"Read coverage differs: actual={actual_reads}, expected={expected_reads}"
    )


def _run_case(
        mesh: object, *, metadata_mode: bool, sampler_type: str, data_sharding: bool, broadcast_plan: bool,
) -> list:
    """Verify native sampler membership, model-parallel delivery and checkpoint replay."""
    rank = dist.get_rank()
    dataset = _Dataset()
    reference = list(_sampler(sampler_type, data_sharding))
    source_sampler = _sampler(sampler_type, data_sharding)
    config = DistributedDatasetConfig(
        seq_len=8, local_batch_size=2, communication_backend="gloo", metadata_mode=metadata_mode,
        broadcast_plan=broadcast_plan,
    )
    metadata_options = (
        {"metadata": [_metadata({"id": index}) for index in range(len(dataset))]}
        if metadata_mode else {"metadata_fn": _metadata}
    )
    no_a2a = patch.object(dist, "all_to_all_single", side_effect=AssertionError("metadata used A2A"))
    with (no_a2a if metadata_mode else nullcontext()), ExitStack() as resources:
        loader = resources.enter_context(build_distributed_dataloader(
            dataset if rank % 2 == 0 else None, mesh, config, batch_sampler=source_sampler, **metadata_options,
            cost_model=lambda metadata: metadata.cost, collate_fn=_native_collate, model_batch_spec=_native_spec(2),
        ))
        saved_state = None
        delivered = []
        for round_idx, expected_local_indices in enumerate(reference):
            batch = _plain(next(loader))
            delivered.append(batch)
            _check_round_payloads(batch, expected_local_indices)
            if rank % 2:
                assert source_sampler.consumed_samples == 0 and not dataset.reads, (
                    f"TP peer advanced sampler or read payloads: cursor={source_sampler.consumed_samples}, "
                    f"reads={dataset.reads}"
                )
            if round_idx == 1:
                saved_state = loader.state_dict()
                if rank % 2 == 0:
                    saved_cursor = saved_state["_index_sampler_state"]["consumed_samples"]
                    assert saved_cursor == 8, f"Saved speculative sampler cursor: actual={saved_cursor}, expected=8"

        assert not list(loader), f"Native sampler should exhaust after {len(reference)} rounds"
        _check_dataset_reads(dataset, reference)
        resumed = resources.enter_context(build_distributed_dataloader(
            dataset if rank % 2 == 0 else None, mesh, config,
            batch_sampler=_sampler(sampler_type, data_sharding), **metadata_options,
            cost_model=lambda metadata: metadata.cost, collate_fn=_native_collate, model_batch_spec=_native_spec(2),
        ))
        resumed.load_state_dict(saved_state)
        remaining = [_plain(batch) for batch in resumed]
        assert remaining == delivered[2:], f"Native replay differs: actual={remaining}, expected={delivered[2:]}"
    return delivered


def _run_cost_balance(mesh: object, broadcast_plan: bool) -> None:
    """Redistribute unequal native samples while keeping the selected four outputs."""
    sampler = build_dataset_batch_sampler(
        total_samples=18, micro_batch_size=2, global_batch_size=8,
        dp_rank=dist.get_rank() // 2, dp_world_size=2,
    )
    loader = build_distributed_dataloader(
        _Dataset(), mesh, DistributedDatasetConfig(
            seq_len=8, local_batch_size=2, communication_backend="gloo", broadcast_plan=broadcast_plan,
        ),
        batch_sampler=sampler, metadata_fn=_metadata,
        cost_model=lambda metadata: metadata.cost, collate_fn=_native_collate, model_batch_spec=_native_spec(2),
    )
    batches = _gather(next(loader))
    actual_costs = [sum(_metadata(sample).cost.llm for sample in batches[rank]) for rank in (0, 2)]
    assert actual_costs == [10, 10], f"Expected costs=[10, 10] instead of native [18, 2], got costs={actual_costs}"
    actual_indices = sorted(sample["id"] for rank in (0, 2) for sample in batches[rank])
    assert actual_indices == [0, 1, 2, 3], f"Cost balancing changed the native round: indices={actual_indices}"
    loader.close()


def test_native_batch_sampler_dp2_tp2_gloo() -> None:
    """Check native DP slicing and collective replay under both data paths."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        for metadata_mode in (False, True):
            for sampler_type, data_sharding in (("single", False), ("cyclic", False), ("cyclic", True)):
                options = {
                    "metadata_mode": metadata_mode, "sampler_type": sampler_type, "data_sharding": data_sharding,
                }
                baseline = _run_case(mesh, **options, broadcast_plan=True)
                independent = _run_case(mesh, **options, broadcast_plan=False)
                assert independent == baseline, (
                    f"Broadcast mode changed batches: actual={independent}, expected={baseline}"
                )
        for broadcast_plan in (True, False):
            _run_cost_balance(mesh, broadcast_plan)
            _run_sequence_repacking(mesh, broadcast_plan)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_changed_gbs_dp2_tp2_gloo() -> None:
    """Repartition an unaligned global cursor and replay it on Constructors and TP peers."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        for metadata_mode, sampler_type in product((False, True), ("single", "cyclic")):
            _run_changed_gbs(mesh, metadata_mode, sampler_type)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_changed_gbs(mesh: object, metadata_mode: bool, sampler_type: str) -> None:
    """Check GBS ten to eight, DP one to two, and a second restore from cursor thirty-eight."""
    rank = dist.get_rank()
    source = _Dataset()
    dataset = [source[index] for index in range(65)]
    order = list(range(65))
    if sampler_type == "cyclic":
        random.Random(44).shuffle(order)
    metadata_options = {"metadata": [_metadata(sample) for sample in dataset]} if metadata_mode else {
        "metadata_fn": _metadata,
    }

    def build_loader() -> object:
        """Build the target layout on every TP peer."""
        sampler = build_dataset_batch_sampler(
            total_samples=65, micro_batch_size=2, global_batch_size=8,
            dp_rank=rank // 2, dp_world_size=2, sampler_type=sampler_type, seed=42,
        )
        return build_distributed_dataloader(
            dataset if rank % 2 == 0 else None, mesh,
            DistributedDatasetConfig(seq_len=8, local_batch_size=2, balance_window_size=3,
                                     metadata_mode=metadata_mode, communication_backend="gloo"),
            batch_sampler=sampler, collate_fn=_native_collate, model_batch_spec=_native_spec(2),
            cost_model=lambda item: item.cost, **metadata_options,
        )

    with build_loader() as loader:
        loader.load_state_dict({"_index_sampler_state": {
            "consumed_samples": 30, "epoch": 2, "global_batch_size": 10, "dp_world_size": 1,
        }})
        delivered = [_plain(next(loader))]
        assert loader.state_dict()["_index_sampler_state"]["consumed_samples"] == 30, "Saved a partial step"
        delivered.append(_plain(next(loader)))
        saved = loader.state_dict()
        cursor = saved["_index_sampler_state"]
        assert cursor == {"consumed_samples": 38, "epoch": 2, "global_batch_size": 8,
                          "dp_world_size": 2, "optimizer_step_offset": 6}, f"Wrong cursor on rank {rank}: {cursor}"
        delivered.extend(_plain(batch) for batch in loader)
    gathered = _gather(delivered)
    assert all(len(batches) == 8 for batches in gathered), f"Unequal DP lengths: {gathered}"
    assert gathered[0] == gathered[1] and gathered[2] == gathered[3], f"TP mismatch: {gathered}"
    actual = sorted(sample["id"] for batches in gathered[::2] for batch in batches for sample in batch)
    assert actual == sorted(order[30:62]), f"Changed suffix: {actual}"
    with build_loader() as resumed:
        resumed.load_state_dict(saved)
        remaining = [_plain(batch) for batch in resumed]
    assert remaining == delivered[2:], f"Replayed offset differs on rank {rank}: {remaining}"


def _window_loader(
        mesh: object, *, metadata_mode: bool, repack: bool, broadcast_plan: bool, accumulation: int,
        device_prefetch_depth: int = 1,
) -> object:
    """Build a DP2-by-TP2 loader with uneven costs and optional document repacking."""
    records = [
        {"id": index, "parts": [{"id": index, "part": part} for part in range(2)]}
        for index in range(44)
    ]
    sampler = build_dataset_batch_sampler(
        total_samples=len(records), micro_batch_size=1, global_batch_size=2 * accumulation,
        dp_rank=dist.get_rank() // 2, dp_world_size=2,
    )

    def metadata(sample: dict) -> SampleMetadata:
        """Assign uneven native-record costs to exercise cross-round placement.

        Args:
            sample: A native record or one independently repacked sequence.
        """
        tokens = 1 if "part" in sample else 2
        return SampleMetadata(tokens, cost=WorkloadCost(llm=tokens * (9 if sample["id"] % 2 == 0 else 1)))

    options = {"metadata": [metadata(sample) for sample in records]} if metadata_mode else {"metadata_fn": metadata}
    if repack:
        options.update(split_fn=lambda sample: sample["parts"], pack_fn=lambda samples, _: tuple(samples))
    def collate(samples: tuple) -> dict:
        """Represent variable document membership as a dynamic tensor dimension.

        Args:
            samples: Local native records or repacked groups of document fragments.
        """
        flattened = [item for packed in samples for item in packed] if repack else samples
        return {"pairs": torch.tensor([(item["id"], item.get("part", -1)) for item in flattened])}

    return build_distributed_dataloader(
        records, mesh,
        DistributedDatasetConfig(
            seq_len=4, local_batch_size=1, communication_backend="gloo", balance_window_size=3,
            device_prefetch_depth=device_prefetch_depth,
            metadata_mode=metadata_mode, broadcast_plan=broadcast_plan,
        ),
        batch_sampler=sampler, cost_model=lambda item: item.cost, device="cpu", collate_fn=collate,
        model_batch_spec={"pairs": TensorFieldSpec((None, 2), torch.int64)}, **options,
    )


def _window_output(batch: tuple, repack: bool) -> list[tuple[int, int]]:
    del repack
    return [tuple(pair) for pair in batch["pairs"].tolist()]


def _check_window_membership(outputs: list, repack: bool, accumulation: int) -> None:
    """Check TP equality and exact sample occurrence conservation in each planning window."""
    assert outputs[0] == outputs[1] and outputs[2] == outputs[3], f"TP batches differ: outputs={outputs}"
    start = 0
    while start < len(outputs[0]):
        rounds = min(3, accumulation - start % accumulation, len(outputs[0]) - start)
        stop = start + rounds
        actual = Counter(item for rank in (0, 2) for batch in outputs[rank][start:stop] for item in batch)
        expected = Counter(
            (index, part) for index in range(start * 2, stop * 2) for part in (range(2) if repack else (-1,))
        )
        assert actual == expected, f"Window membership differs: actual={actual}, expected={expected}"
        start = stop


def test_accumulation_window_dp2_tp2_gloo() -> None:
    """Verify real payload redistribution, TP broadcast, and incomplete-step replay."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        for accumulation in (3, 10):
            for metadata_mode, repack in ((False, False), (True, False), (False, True)):
                for broadcast_plan in (False, True):
                    options = {
                        "metadata_mode": metadata_mode, "repack": repack,
                        "broadcast_plan": broadcast_plan, "accumulation": accumulation,
                    }
                    with _window_loader(mesh, **options) as loader:
                        first = _window_output(next(loader), repack)
                        state = loader.state_dict()
                        remaining = [_window_output(batch, repack) for batch in loader]
                    outputs = _gather([first] + remaining)
                    _check_window_membership(outputs, repack, accumulation)
                    with _window_loader(mesh, **options) as resumed:
                        resumed.load_state_dict(state)
                        actual = [_window_output(batch, repack) for batch in resumed]
                        expected = [first, *remaining]
                        assert actual == expected, f"Step replay differs: actual={actual}, expected={expected}"
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_host_window_device_queue_dp2_tp2_gloo() -> None:
    """Exercise Host preparation and caller device streams with real collectives."""
    # Inspect internal queues and ownership to verify prefetch lifetime invariants.
    # pylint: disable=protected-access
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    accelerator = Mock()
    accelerator.stream.side_effect = lambda _: nullcontext()
    accelerator.Event.return_value.query.return_value = True
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        _check_early_close_with_uneven_lookahead(mesh, accelerator)
        for metadata_mode, depth in product((False, True), (0, 1, 2, 24)):
            options = {
                "metadata_mode": metadata_mode, "repack": False, "broadcast_plan": True, "accumulation": 5,
                "device_prefetch_depth": depth,
            }
            with patch.object(torch, "cuda", accelerator), _window_loader(mesh, **options) as loader:
                # Keep real payload/model traffic on Gloo while exercising device
                # scheduling on every rank, including consumer-only TP peers.
                loader._device = torch.device("cuda:0")
                outputs = []
                state = None
                for batch in loader:
                    if dist.get_rank() % 2 == 0:
                        # Constructor-only H2D hooks must not enter MP collectives.
                        loader.prefetch()
                        loader.prefetch()
                    outputs.append(_window_output(batch, False))
                    if len(outputs) == 5:
                        state = loader.state_dict()
                assert state["_index_sampler_state"]["consumed_samples"] == 10, (
                    f"Delivered checkpoint differs: state={state}"
                )
                _check_window_membership(_gather(outputs), False, 5)
                worker = loader._prefetch_thread
                loader.set_epoch(1)
                replay = [_window_output(batch, False) for batch in loader]
                assert replay == outputs, f"Epoch replay differs: actual={replay}, expected={outputs}"
                assert loader._prefetch_thread is worker, f"Host worker replaced: {worker}"
                accelerator.Event.return_value.record.assert_called_with(accelerator.current_stream.return_value)
            with patch.object(torch, "cuda", accelerator), _window_loader(mesh, **options) as resumed:
                resumed._device = torch.device("cuda:0")
                resumed.load_state_dict(state)
                actual = [_window_output(batch, False) for batch in resumed]
                assert actual == outputs[5:], f"Checkpoint replay differs: actual={actual}, expected={outputs[5:]}"
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _check_early_close_with_uneven_lookahead(mesh: object, accelerator: object) -> None:
    """Keep one Constructor behind while its peer enters the next Host collective."""
    # Inspect internal queues and ownership to verify prefetch lifetime invariants.
    # pylint: disable=protected-access
    for metadata_mode in (False, True):
        with patch.object(torch, "cuda", accelerator), _window_loader(
                mesh, metadata_mode=metadata_mode, repack=False, broadcast_plan=True,
                accumulation=1, device_prefetch_depth=1,
        ) as loader:
            loader._device = torch.device("cuda:0")
            # Delay all optional staging until ranks can deliberately diverge.
            with patch.object(loader, "_host_batch_ready", return_value=False):
                next(loader)
            if dist.get_rank() == 0:
                loader._prefetch_device_batches(1, blocking=True)
            queued = _gather(len(loader._device_batches))
            assert queued == [1, 0, 0, 0], f"Fixture did not create uneven lookahead: {queued}"
            worker = loader._prefetch_thread
        assert not worker.is_alive(), "Early close left a Host producer running"
        dist.barrier()


def test_trainer_batch_layout_dp2_tp2_gloo() -> None:
    """Run native Trainer collation with inferred text and configured multimodal layouts."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        context = SimpleNamespace(device_mesh=mesh, dp_rank=dist.get_rank() // 2, dp_size=2, pp_size=1)
        for kind in ("gpt", "online", "omni"):
            _check_trainer_batch_layout(context, kind)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _check_trainer_batch_layout(context: object, kind: str) -> None:
    """Compare the real Trainer outputs on both model peers and DP partitions."""
    records = [{"input_ids": torch.full((4,), index), "labels": torch.full((4,), index + 1)} for index in range(8)]
    collator = default_collate if kind == "gpt" else TextPackingCollator()
    target = SimpleNamespace(load_balance={})
    if kind == "omni":
        collator = OmniCollator()
        for index, sample in enumerate(records):
            sample["pixel_values"] = torch.full((index % 2 + 1, 3), float(index))
        spec = {
            "input_ids": TensorFieldSpec((1, 4), torch.int64),
            "labels": TensorFieldSpec((1, 4), torch.int64),
            "pixel_values": TensorFieldSpec((None, 3), torch.float32),
        }
        target.model_batch_spec = Target(lambda: spec, target_path="test.omni_model_batch_spec")
    loaders, _ = build_trainer_dataloader(
        target, datasets=(records, None, None), collate_fn=collator, mesh_context=context,
        training_config=SimpleNamespace(micro_batch_size=1, global_batch_size=4, seed=7),
        data_config={"seq_length": 4},
        cost_model=lambda item: WorkloadCost(llm=float(item.pack_tokens)),
    )
    with loaders[0] as loader:
        outputs = _gather([_plain(batch) for batch in loader])
    assert outputs[0] == outputs[1] and outputs[2] == outputs[3], f"Trainer model peers differ: {kind}"
    ids = sorted(batch["input_ids"][0][0] for rank in (0, 2) for batch in outputs[rank])
    assert ids == list(range(8)), f"Trainer lost native samples: {kind}: {ids}"


def _check_repacked_tokens(records: list[dict], gathered: list[dict]) -> None:
    """Check token and target conservation across the two DP representatives."""
    expected = Counter(
        (token, label) for record in records
        for token, label in zip(record["input_ids"].tolist(), record["labels"].tolist())
    )
    actual = Counter(
        (token, label) for rank in (0, 2)
        for token, label in zip(gathered[rank]["input_ids"][0], gathered[rank]["labels"][0])
        if token != 0
    )
    assert actual == expected, f"Repacking changed selected tokens/labels: actual={actual}, expected={expected}"


def _run_sequence_repacking(mesh: object, broadcast_plan: bool) -> None:
    """Balance independent sequences with TP replication and per-rank cursor replay."""
    records = []
    for index in range(6):
        lengths = [6, 6] if index % 2 == 0 else [2, 2, 2, 2]
        ids = torch.arange(index * 20 + 1, index * 20 + sum(lengths) + 1)
        labels = ids + 1
        labels[::3] = -100
        records.append({
            "input_ids": ids, "labels": labels,
            "cu_seq_lens": torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()]),
        })

    def build_loader() -> object:
        """Build a fresh reader and sampler for checkpoint replay."""
        return build_distributed_dataloader(
            records if dist.get_rank() % 2 == 0 else None, mesh,
            DistributedDatasetConfig(
                seq_len=12, local_batch_size=1, communication_backend="gloo", broadcast_plan=broadcast_plan,
            ),
            batch_sampler=build_dataset_batch_sampler(
                total_samples=6, micro_batch_size=1, global_batch_size=2,
                dp_rank=dist.get_rank() // 2, dp_world_size=2,
            ),
            split_fn=split_packed_text_sample, pack_fn=pack_indexed_text_samples,
            collate_fn=collate_indexed_text_sequences,
            metadata_fn=lambda sample: SampleMetadata(sample["input_ids"].numel()),
            cost_model=lambda item: WorkloadCost(llm=item.pack_tokens ** 2),
        )

    delivered = []
    with build_loader() as loader:
        for step, batch in enumerate(loader):
            serializable = {name: value.tolist() for name, value in batch.items()}
            delivered.append(serializable)
            gathered = _gather(serializable)
            assert gathered[0] == gathered[1] and gathered[2] == gathered[3], (
                f"TP batch mismatch after sequence repacking: batches={gathered}"
            )
            _check_repacked_tokens(records[step * 2:step * 2 + 2], gathered)
            costs = [
                sum((end - start) ** 2
                    for start, end in zip(gathered[rank]["cu_seq_lens"], gathered[rank]["cu_seq_lens"][1:])
                    if gathered[rank]["input_ids"][0][start] != 0)
                for rank in (0, 2)
            ]
            assert costs == [44, 44], f"Expected repacked costs [44, 44], got {costs}"
            if step == 0:
                saved_state = loader.state_dict()
                if dist.get_rank() % 2 == 0:
                    cursor = saved_state["_index_sampler_state"]["consumed_samples"]
                    assert cursor == 2, f"Prefetched record progress was saved: cursor={cursor}"
    with build_loader() as resumed:
        resumed.load_state_dict(saved_state)
        remaining = [{name: value.tolist() for name, value in batch.items()} for batch in resumed]
    assert remaining == delivered[1:], f"Repacked checkpoint replay differs: actual={remaining}"


def _check_full_pack_documents(records: list[dict], gathered: list[dict]) -> None:
    """Compare each received attention segment with its complete original document."""
    for rank in (0, 2):
        value = gathered[rank]
        offsets = value["cu_seq_lens"]
        assert offsets[0] == 0 and offsets[-1] == 10, f"Invalid repaired boundaries: offsets={offsets}"
        for start, end in zip(offsets, offsets[1:]):
            document = value["input_ids"][0][start:end]
            original = records[(document[0] - 1) // 10]
            original_start = (document[0] - 1) % 10
            original_offsets = original["cu_seq_lens"].tolist()
            source_index = original_offsets.index(original_start)
            expected_end = original_offsets[source_index + 1]
            expected = original["input_ids"][original_start:expected_end].tolist()
            assert document == expected, f"Document boundary changed: actual={document}, expected={expected}"


def _full_pack_records() -> tuple[list[dict], dict[int, int]]:
    """Create full records whose token lengths lead greedy LPT into a dead end."""
    records, costs = [], {}
    for index in range(14):
        lengths = (6, 2, 2) if index % 2 == 0 else (5, 3, 2)
        weights = (90, 70, 50) if index % 2 == 0 else (100, 80, 60)
        ids = torch.arange(index * 10 + 1, index * 10 + 11)
        labels = ids + 1
        labels[::3] = -100
        offsets = [0, *torch.tensor(lengths).cumsum(0).tolist()]
        records.append({"input_ids": ids, "labels": labels, "cu_seq_lens": torch.tensor(offsets)})
        costs.update({int(ids[start]): weight for start, weight in zip(offsets, weights)})
    return records, costs


def _full_pack_statistics(gathered: list[dict], costs: dict[int, int]) -> tuple[Counter, Counter, int]:
    """Count delivered token/document occurrences and the slowest DP pack's cost."""
    tokens, documents = Counter(), Counter()
    rank_costs = []
    for rank in (0, 2):
        value = gathered[rank]
        tokens.update(zip(value["input_ids"][0], value["labels"][0]))
        starts = [value["input_ids"][0][start] for start in value["cu_seq_lens"][:-1]]
        documents.update(starts)
        rank_costs.append(sum(costs[token] for token in starts))
    return tokens, documents, max(rank_costs)


def _run_full_pack_repair(mesh: object, accumulation: int, broadcast_plan: bool) -> None:
    """Exercise repaired full packs, repeated records, window tails, and checkpoint replay."""
    records, costs = _full_pack_records()
    mapping = [0, 1] * 7

    def build_loader() -> object:
        """Create a fresh sampler and source reader for the same occurrence stream."""
        return build_distributed_dataloader(
            records, mesh, DistributedDatasetConfig(
                seq_len=10, local_batch_size=1, communication_backend="gloo", balance_window_size=3,
                broadcast_plan=broadcast_plan,
            ),
            batch_sampler=build_dataset_batch_sampler(
                total_samples=len(records), micro_batch_size=1, global_batch_size=2 * accumulation,
                dp_rank=dist.get_rank() // 2, dp_world_size=2, index_mapping=mapping,
            ),
            device="cpu", split_fn=split_packed_text_sample, pack_fn=pack_indexed_text_samples,
            collate_fn=collate_indexed_text_sequences,
            metadata_fn=lambda sample: SampleMetadata(
                sample["input_ids"].numel(), cost=WorkloadCost(llm=costs[int(sample["input_ids"][0])]),
            ),
            cost_model=lambda item: item.cost,
        )

    delivered, window_tokens, window_documents = [], Counter(), Counter()
    window_objective = 0
    saved = None
    with build_loader() as loader:
        for step, batch in enumerate(loader):
            output = {name: value.tolist() for name, value in batch.items()}
            delivered.append(output)
            gathered = _gather(output)
            assert gathered[0] == gathered[1] and gathered[2] == gathered[3], (
                f"Repaired TP batches differ: batches={gathered}"
            )
            _check_full_pack_documents(records, gathered)
            tokens, documents, objective = _full_pack_statistics(gathered, costs)
            window_tokens.update(tokens)
            window_documents.update(documents)
            window_objective += objective
            if step == accumulation - 1:
                saved = loader.state_dict()
            if (step + 1) % accumulation == 0 or step == 6:
                start = step // accumulation * accumulation
                assert window_objective < (step + 1 - start) * 240, (
                    f"Full-pack repair did not improve the delivered window: cost={window_objective}"
                )
                window_objective = 0
                expected_documents = Counter(
                    int(records[mapping[position]]["input_ids"][offset])
                    for position in range(start * 2, (step + 1) * 2)
                    for offset in records[mapping[position]]["cu_seq_lens"][:-1].tolist()
                )
                expected_tokens = Counter(
                    (token, label) for position in range(start * 2, (step + 1) * 2)
                    for token, label in zip(records[mapping[position]]["input_ids"].tolist(),
                                            records[mapping[position]]["labels"].tolist())
                )
                assert window_documents == expected_documents, (
                    f"Document occurrences changed: actual={window_documents}, expected={expected_documents}"
                )
                assert window_tokens == expected_tokens, (
                    f"Tokens or labels changed: actual={window_tokens}, expected={expected_tokens}"
                )
                window_tokens.clear()
                window_documents.clear()
    assert saved is not None, f"No checkpoint was captured: delivered={len(delivered)}"
    with build_loader() as restored:
        restored.load_state_dict(saved)
        actual = [{name: value.tolist() for name, value in batch.items()} for batch in restored]
    assert actual == delivered[accumulation:], f"Repaired replay differs: {actual} vs {delivered[accumulation:]}"


def test_full_pack_repair_dp2_tp2_gloo() -> None:
    """Verify repairs with and without gradient accumulation under both plan modes."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        for accumulation in (1, 3):
            for broadcast_plan in (False, True):
                _run_full_pack_repair(mesh, accumulation, broadcast_plan)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _gather_balance_logs(caplog: pytest.LogCaptureFixture, planner_rank: int, expected_count: int) -> list[str]:
    """Check that only the planner emits the expected number of window reports."""
    messages = _gather([record.getMessage() for record in caplog.records if record.name == balance_logger.name])
    counts = [len(records) for records in messages]
    expected_counts = [expected_count if rank == planner_rank else 0 for rank in range(4)]
    assert counts == expected_counts, f"Duplicate or missing logs: actual={counts}, expected={expected_counts}"
    return messages[planner_rank]


def _run_balance_logging_case(
        mesh: object, caplog: pytest.LogCaptureFixture, *, metadata_mode: bool, broadcast_plan: bool,
        planner_rank: int, min_gain: float,
) -> None:
    """Verify one writer, modeled window gains and deferred speculative output."""
    dataset = _Dataset()

    def metadata(sample: dict) -> SampleMetadata:
        """Assign alternating costs to the two native DP sources.

        Args:
            sample: Original indexed source sample.
        """
        return SampleMetadata(1, cost=WorkloadCost(llm=9 if sample["id"] % 2 == 0 else 1))

    options = {"metadata": [metadata(dataset[index]) for index in range(len(dataset))]} if metadata_mode else {
        "metadata_fn": metadata,
    }
    caplog.clear()
    with build_distributed_dataloader(
            dataset, mesh, DistributedDatasetConfig(
                seq_len=8, local_batch_size=1, communication_backend="gloo", balance_window_size=3,
                metadata_mode=metadata_mode, broadcast_plan=broadcast_plan,
                planner_rank=planner_rank, min_balance_gain=min_gain, log_balance=True,
            ),
            batch_sampler=build_dataset_batch_sampler(
                total_samples=len(dataset), micro_batch_size=1, global_batch_size=10,
                dp_rank=dist.get_rank() // 2, dp_world_size=2,
            ),
            cost_model=lambda item: item.cost, collate_fn=_native_collate,
            model_batch_spec=_native_spec(1), **options,
    ) as loader:
        loader.prefetch()
        loader.wait_for_prefetch()
        assert not caplog.records, f"Unconsumed step emitted logs: {caplog.records}"
        first = next(loader)
        loader.wait_for_prefetch()
        first_window = _gather_balance_logs(caplog, planner_rank, expected_count=1)[0]
        assert "iteration 1 :" in first_window and "micro_batches=1-3" in first_window, (
            f"Incorrect first window identity: {first_window}"
        )
        assert f"planner_rank={planner_rank}" in first_window, f"Incorrect planner rank: {first_window}"
        assert "constructor_ranks=(0, 2)" in first_window, f"Incorrect DP rank mapping: {first_window}"
        assert "original=27, candidate=19," in first_window, f"Incorrect window costs: {first_window}"
        accepted = min_gain == 0
        assert f"accepted={accepted}" in first_window, f"Incorrect acceptance decision: {first_window}"
        assert f"accepted_objective={19 if accepted else 27}" in first_window, (
            f"Incorrect accepted objective: {first_window}"
        )
        batches = _gather(first)
        costs = [metadata(batches[rank][0]).cost.llm for rank in (0, 2)]
        expected_costs = [9, 9] if accepted else [9, 1]
        assert costs == expected_costs, f"Logged assignment differs: actual={costs}, expected={expected_costs}"
        for _ in range(3):
            next(loader)
        second_window = _gather_balance_logs(caplog, planner_rank, expected_count=2)[1]
        assert "iteration 1 :" in second_window and "micro_batches=4-5" in second_window, (
            f"Incorrect second window identity: {second_window}"
        )


def test_native_balance_logging_dp2_tp2_gloo(caplog: pytest.LogCaptureFixture) -> None:
    """Check INFO reports on the chosen planner under both metadata and planning paths.

    Args:
        caplog: Per-process log capture, without replacing the logging callbacks.
    """
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        with caplog.at_level(logging.INFO, logger=balance_logger.name):
            for metadata_mode, broadcast_plan, planner_rank, min_gain in (
                    (False, True, 0, 0.0), (False, False, 2, 0.0),
                    (True, True, 2, 0.9), (True, False, 0, 0.0),
            ):
                _run_balance_logging_case(
                    mesh, caplog, metadata_mode=metadata_mode, broadcast_plan=broadcast_plan,
                    planner_rank=planner_rank, min_gain=min_gain,
                )
        dist.barrier()
    finally:
        dist.destroy_process_group()


class _BoundaryRuntimeAdapter(RuntimeInputAdapter):
    """Expose global boundaries after CP slicing for integration assertions."""

    def build_runtime_inputs(self, *, batch: object, context: object) -> dict:
        """Preserve boundaries without replacing canonical model inputs."""
        del context
        return {"global_boundaries": batch["cu_seq_lens"]}


def test_single_distribution_dp2_cp2_tp2_gloo() -> None:
    """Compare native and balanced runtimes with all three parallel dimensions."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("dp", "cp", "tp"))
        dp_rank, cp_rank, tp_rank = mesh.get_coordinate()
        context = SimpleNamespace(
            device_mesh=mesh, dp_rank=dp_rank, dp_size=2, cp_rank=cp_rank, cp_size=2,
            tp_rank=tp_rank, tp_size=2, pp_size=1, sequence_parallel=False,
        )
        for kind, balanced in product(("text", "omni"), (False, True)):
            _check_single_distribution(context, kind, balanced)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _check_single_distribution(context: object, kind: str, balanced: bool) -> None:
    """Verify local fields, TP equality, DP conservation and broadcast ownership."""
    records = []
    for index in range(8):
        ids = torch.arange(8) + index * 20
        labels = ids + 1
        labels[-1] = -100
        records.append({"input_ids": ids, "labels": labels, "cu_seq_lens": torch.tensor([0, 3, 8])})
    collator = _collate_runtime_records
    parallel_context = create_dataloader_parallel_context(context, tp_source_rank=None if balanced else 0)
    options = {"mesh_context": context, "device": "cpu", "dataloader_context": parallel_context,
               "runtime_input_adapter": _BoundaryRuntimeAdapter()}
    runtime = (TextParallelBatch(**options, tokenizer=None, data_config={}, pp_shared_data=False,
                                source_type="packed_text", reset_position_ids=True)
               if kind == "text" else OmniParallelBatch(**options))
    with ExitStack() as stack:
        if balanced:
            spec = {name: TensorFieldSpec((1, 8), torch.int64) for name in ("input_ids", "labels")}
            spec["cu_seq_lens"] = TensorFieldSpec((3,), torch.int64)
            loaders, _ = build_trainer_dataloader(
                SimpleNamespace(load_balance={}, model_batch_spec=spec),
                datasets=(records, None, None), collate_fn=collator, mesh_context=context,
                training_config=SimpleNamespace(micro_batch_size=1, global_batch_size=4, seed=7),
                data_config={"seq_length": 8}, cost_model=lambda item: WorkloadCost(llm=float(item.pack_tokens)),
            )
            loader = stack.enter_context(loaders[0])
            # Observe only get_batch transport, leaving the Host producer collectives untouched.
            iterator = iter(loader)
        else:
            iterator = iter(collator([record]) for record in records[context.dp_rank::2])
        outputs = []
        for _ in range(4):
            if balanced:
                with patch("hyper_parallel.data.parallel.batch_parallel.dist", wraps=dist) as runtime_dist:
                    model, loss = runtime(iterator)
                    assert not runtime_dist.broadcast.called, "Duplicate TP tensor broadcast"
                    assert not runtime_dist.broadcast_object_list.called, "Duplicate TP structure broadcast"
            else:
                model, loss = runtime(iterator)
            _check_cp_runtime_batch(model, loss, context.cp_rank, kind)
            outputs.append(_plain(model["input_ids"]))
        if balanced:
            with pytest.raises(StopIteration):
                runtime(iterator)
    gathered = _gather(outputs)
    assert all(gathered[rank] == gathered[rank + 1] for rank in range(0, 8, 2)), "TP peers differ"
    ids = sorted(batch[0][0] // 20 for rank in (0, 4) for batch in gathered[rank])
    assert ids == list(range(8)), f"DP sample occurrences changed: {ids}"


def _check_cp_runtime_batch(model: dict, loss: dict, cp_rank: int, kind: str) -> None:
    """Check a document crossing the CP boundary and a masked final label."""
    tokens = model["input_ids"]
    index = int(tokens[0, 0]) // 20
    expected = torch.arange(cp_rank * 4, (cp_rank + 1) * 4).reshape(1, 4) + index * 20
    torch.testing.assert_close(tokens, expected)
    labels = expected + 1
    if cp_rank == 1:
        labels[0, -1] = -100
    torch.testing.assert_close(loss["labels"], labels)
    assert loss["loss_mask"].tolist() == [[1, 1, 1, 1 if cp_rank == 0 else 0]], "Loss mask changed"
    assert model["global_boundaries"].reshape(-1).tolist() == [0, 3, 8], "CP sliced document boundaries"
    if kind == "text":
        assert model["position_ids"].tolist() == ([[0, 1, 2, 0]] if cp_rank == 0 else [[1, 2, 3, 4]])


def _collate_runtime_records(records: list[dict]) -> dict:
    """Flatten one record's global boundaries for both runtime contracts."""
    batch = default_collate(records)
    batch["cu_seq_lens"] = batch["cu_seq_lens"].reshape(-1)
    return batch
