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
"""Concurrent window data communication and training on DP2/TP2 NPUs."""

import os
from collections import Counter
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig, DistributedDataLoader, SampleMetadata, WorkloadCost, build_distributed_dataloader,
    TensorFieldSpec,
)


class _Dataset:
    """Provide six optimizer steps with deterministic CPU input tensors."""

    def __len__(self) -> int:
        """Return eighteen synchronized micro-batches across DP2."""
        return 36

    def __getitem__(self, index: int) -> dict:
        """Keep identity and exact tensor contents observable after routing."""
        return {"id": index, "input_ids": torch.tensor([index, index + 1]), "offsets": torch.tensor([0, 2])}


def _metadata(sample: dict) -> SampleMetadata:
    return SampleMetadata(2, cost=WorkloadCost(llm=9 if sample["id"] % 2 == 0 else 1))


def _loader(
        device: torch.device, backend: str, metadata_mode: bool, broadcast_plan: bool, consumed_samples: int = 0,
        accumulation: int = 3, device_depth: int = 1,
) -> object:
    dataset = _Dataset()
    mesh = SimpleNamespace(mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"), rank_list=(0, 1, 2, 3))
    options = {"metadata": [_metadata(dataset[i]) for i in range(len(dataset))]} if metadata_mode else {
        "metadata_fn": _metadata,
    }
    return build_distributed_dataloader(
        dataset, mesh, DistributedDatasetConfig(
            seq_len=8, local_batch_size=1, communication_backend=backend,
            metadata_mode=metadata_mode, balance_window_size=3, broadcast_plan=broadcast_plan,
            device_prefetch_depth=device_depth,
        ),
        batch_sampler=build_dataset_batch_sampler(
            total_samples=len(dataset), micro_batch_size=1, global_batch_size=2 * accumulation,
            dp_world_size=2, dp_rank=dist.get_rank() // 2, consumed_samples=consumed_samples,
        ),
        collate_fn=lambda samples: tuple({**sample, "id": torch.tensor(sample["id"])} for sample in samples),
        model_batch_spec=({"id": TensorFieldSpec((), torch.int64),
                           "input_ids": TensorFieldSpec((2,), torch.int64),
                           "offsets": TensorFieldSpec((2,), torch.int64)},),
        cost_model=lambda metadata: metadata.cost, device=device, **options,
    )


def _check_batch(batch: tuple, device: torch.device) -> list[int]:
    """Check device payload contents and return the shared sample IDs from each DP rank."""
    sample = batch[0]
    assert sample["input_ids"].device == device, f"Input device differs: {sample}"
    assert sample["offsets"].device == device, f"Offsets device differs: {sample}"
    assert sample["input_ids"].tolist() == [sample["id"], sample["id"] + 1], f"Input corrupted: {sample}"
    assert sample["offsets"].tolist() == [0, 2], f"Offsets corrupted: {sample}"
    ids = torch.tensor([sample["id"]], device=device)
    gathered = [torch.empty_like(ids) for _ in range(4)]
    dist.all_gather(gathered, ids)
    values = [value.item() for value in gathered]
    assert values[0] == values[1] and values[2] == values[3], f"TP input mismatch: {values}"
    return [values[0], values[2]]


def _run_case(
        device: torch.device, backend: str, metadata_mode: bool, broadcast_plan: bool, accumulation: int,
        device_depth: int,
) -> None:
    """Verify device delivery and checkpoint boundaries alongside training collectives."""
    # Inspect internal queues and ownership to verify prefetch lifetime invariants.
    # pylint: disable=protected-access
    weights = torch.nn.Parameter(torch.full((128, 128), 0.001, device=device))
    inputs = torch.ones((128, 128), device=device)
    delivered, positions = [], Counter()
    saved = None
    options = {"accumulation": accumulation, "device_depth": device_depth}
    with _loader(device, backend, metadata_mode, broadcast_plan, **options) as loader:
        assert isinstance(loader, DistributedDataLoader), f"Missing window buffering: {type(loader)}"
        for micro, batch in enumerate(loader):
            assert loader._consumer_stream == torch.npu.current_stream(device), "Delivery changed the training stream"
            # WORLD represents model communication; data uses separate groups
            # concurrently on the Host thread and caller's device streams.
            consumed_input = batch[0]["input_ids"].square()
            loss = (inputs @ weights).square().mean()
            loss.backward()
            dist.all_reduce(weights.grad)
            assert torch.isfinite(weights.grad).all(), f"Non-finite concurrent gradient at micro={micro}"
            weights.grad = None
            delivered.append(batch[0]["id"].item())
            positions.update(_check_batch(batch, device))
            expected_input = [batch[0]["id"] ** 2, (batch[0]["id"] + 1) ** 2]
            actual_input = consumed_input.tolist()
            assert actual_input == expected_input, (
                f"Training stream read unfinished input: actual={actual_input}, expected={expected_input}"
            )
            boundary = (micro + 1) // accumulation * accumulation
            assert loader.state_dict()["_index_sampler_state"]["consumed_samples"] == 2 * boundary, (
                f"Incomplete step committed at micro={micro}"
            )
            if micro == 0:
                loader.wait_for_prefetch()
                ready_batches = loader._future.result().batches
                expected_size = min(3, accumulation - min(3, accumulation) or accumulation)
                assert len(ready_batches) == expected_size, (
                    f"Next window size differs: actual={len(ready_batches)}, expected={expected_size}"
                )
            if micro % accumulation == accumulation - 1:
                if saved is None:
                    saved = loader.state_dict()
                start = (micro // accumulation) * 2 * accumulation
                expected = Counter(range(start, start + 2 * accumulation))
                assert positions == expected, f"Optimizer membership changed: {positions} vs {expected}"
                positions.clear()
            del batch
        _check_next_epoch(loader, delivered, device)
    assert saved is not None, "No checkpoint captured"
    with _loader(device, backend, metadata_mode, broadcast_plan, **options) as resumed:
        resumed.load_state_dict(saved)
        actual = [batch[0]["id"].item() for batch in resumed]
    assert actual == delivered[accumulation:], f"Device replay differs: {actual} vs {delivered[accumulation:]}"


def _check_next_epoch(loader: DistributedDataLoader, expected: list[int], device: torch.device) -> None:
    """Verify sample replay and producer/stream reuse after an epoch reset."""
    # Inspect internal queues and ownership to verify prefetch lifetime invariants.
    # pylint: disable=protected-access
    worker = loader._prefetch_thread
    copy_stream = loader._device_prefetch._copy_stream if loader._device_prefetch is not None else None
    assert worker.is_alive(), "Producer exited at epoch exhaustion"
    loader.set_epoch(1)
    actual = []
    for batch in loader:
        _check_batch(batch, device)
        actual.append(batch[0]["id"].item())
    assert actual == expected, f"Epoch reset changed sample order: {actual} vs {expected}"
    assert loader._prefetch_thread is worker, "Epoch reset replaced the producer"
    if loader._device_prefetch is not None:
        assert loader._device_prefetch._copy_stream is copy_stream, "Epoch reset replaced the copy stream"
    assert worker.is_alive(), "Producer did not survive the next epoch"


def _check_legacy_complete_step(device: torch.device, metadata_mode: bool) -> None:
    """Verify legacy checkpoint recovery preserves the remaining window and device depth."""
    # Inspect internal queues and ownership to verify prefetch lifetime invariants.
    # pylint: disable=protected-access
    with _loader(device, "hccl", metadata_mode, True, consumed_samples=6) as loader:
        for _ in range(3):
            next(loader)
        saved = loader.state_dict()
        saved["optimizer_step_remaining"] = 3
        expected = [batch[0]["id"] for batch in loader]
    with _loader(device, "hccl", metadata_mode, True) as resumed:
        resumed.load_state_dict(saved)
        first = next(resumed)
        resumed.wait_for_prefetch()
        queued = len(resumed._device_batches)
        assert queued == (1 if dist.get_rank() % 2 == 0 else 0), "Only Constructors may stage future device inputs"
        assert len(resumed._current.batches) + queued == 2, "Legacy resume lost remaining window inputs"
        assert "optimizer_step_remaining" not in resumed.state_dict(), "Legacy micro-batch count was retained"
        actual = [first[0]["id"], *[batch[0]["id"] for batch in resumed]]
    assert actual == expected, f"Legacy complete-step replay differs: {actual} vs {expected}"


@pytest.mark.parametrize("accumulation,device_depth", ((1, 1), (3, 0), (3, 1), (3, 2), (5, 2)))
def test_optimizer_step_loading_hccl(accumulation: int, device_depth: int) -> None:
    """Overlap window data preparation with compute and independent HCCL collectives.

    Args:
        accumulation: Micro-batches prepared per optimizer step.
        device_depth: Future micro-batches staged on Device.
    """
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=90))
    try:
        for backend, metadata_mode, broadcast_plan in (
                ("hccl", False, True), ("hccl", False, False),
                ("hccl", True, True), ("gloo", False, True),
        ):
            with torch.npu.stream(torch.npu.Stream(device=device)):
                _run_case(device, backend, metadata_mode, broadcast_plan, accumulation, device_depth)
        if accumulation > 1:
            for metadata_mode in (False, True):
                _check_legacy_complete_step(device, metadata_mode)
        dist.barrier()
    finally:
        torch.npu.synchronize()
        dist.destroy_process_group()
