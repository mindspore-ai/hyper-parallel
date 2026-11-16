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
"""Accelerator integration for automatic H2D, caller collectives and MP broadcast."""

import os
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleKey,
    SampleMetadata,
    WorkloadCost,
    build_distributed_dataloader,
)
from hyper_parallel.distributed_data.transport import all_gather_control_object


def _control_value(rank: int, step: int) -> object:
    if (rank + step) % 5 == 0:
        return None
    size = (0, 31, 513, 4097)[(rank + step) % 4]
    return {
        "sample": (SampleKey(rank, step, rank + step), SampleMetadata(
            rank + 1, features={"P": rank + 1, "D": 0}, sample_id=f"sample-{rank}-{step}",
        )),
        "bytes": bytes([rank]) * size,
        "nested": ("多模态", [None, True, 2 ** 65]),
    }


def test_control_gather_hccl() -> None:
    """Check variable-length objects and bounded host copies on a non-default stream."""
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=90))
    stream = torch.npu.Stream(device=device)
    tensor_cpu = torch.Tensor.cpu
    transfers = []

    def copy_to_cpu(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Count actual device-to-host calls while preserving their synchronization."""
        transfers.append((tensor.device, tensor.dtype, tensor.numel()))
        return tensor_cpu(tensor, *args, **kwargs)

    try:
        rank = dist.get_rank()
        for ranks in ((0, 1, 2, 3), (0, 2), (1, 3)):
            group = dist.new_group(ranks=list(ranks), backend="hccl")
            if rank not in ranks:
                continue
            try:
                for step in range(3):
                    transfers.clear()
                    with torch.npu.stream(stream), patch.object(torch.Tensor, "cpu", new=copy_to_cpu):
                        received = all_gather_control_object(_control_value(rank, step), group=group, device=device)
                    expected = tuple(_control_value(member, step) for member in ranks)
                    assert received == expected, f"Control values differ: received={received}, expected={expected}"
                    assert len(transfers) == 2, f"Host transfer count: actual={len(transfers)}, expected=2"
                    actual = [(source, dtype) for source, dtype, _ in transfers]
                    expected_transfers = [(device, torch.int64), (device, torch.uint8)]
                    assert actual == expected_transfers, (
                        f"Host transfer sources: actual={actual}, expected={expected_transfers}"
                    )
                    assert transfers[0][2] == len(ranks), (
                        f"Length result size: actual={transfers[0][2]}, expected={len(ranks)}"
                    )
            finally:
                stream.synchronize()
                dist.destroy_process_group(group)
        dist.barrier()
    finally:
        torch.npu.synchronize()
        dist.destroy_process_group()


class _Dataset:
    """Return CPU tensors whose complete batch must move to the training device."""

    def __len__(self) -> int:
        """Return three complete global steps."""
        return 12

    def __getitem__(self, index: int) -> dict:
        """Build one small CPU sample with known tensor values."""
        return {"id": index, "input_ids": torch.tensor([index, index + 1]), "offsets": torch.tensor([0, 2])}


def _metadata(sample: dict) -> SampleMetadata:
    return SampleMetadata(2, cost=WorkloadCost(llm=9 if sample["id"] % 4 < 2 else 1))


def _native(device: torch.device, backend: str, metadata_mode: bool) -> None:
    rank = dist.get_rank()
    mesh = SimpleNamespace(mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"), rank_list=(0, 1, 2, 3))
    dataset = _Dataset()
    sampler = build_dataset_batch_sampler(
        total_samples=12, micro_batch_size=2, global_batch_size=4, dp_world_size=2, dp_rank=rank // 2,
    )
    options = {"metadata": [_metadata(dataset[index]) for index in range(12)]} if metadata_mode else {
        "metadata_fn": _metadata,
    }
    with build_distributed_dataloader(
            dataset, mesh, DistributedDatasetConfig(
                seq_len=8, local_batch_size=2, communication_backend=backend, metadata_mode=metadata_mode,
            ),
            batch_sampler=sampler, device=device,
            cost_model=lambda metadata: metadata.cost, **options,
    ) as loader:
        for step in range(3):
            batch = next(loader)
            for sample in batch:
                assert sample["input_ids"].device == device, (
                    f"Wrong device: got={sample['input_ids'].device}, expected={device}"
                )
                assert sample["offsets"].device == device, f"Offsets did not move: {sample['offsets'].device}"
                assert sample["offsets"].tolist() == [0, 2], f"Offsets changed: {sample['offsets']}"
                assert sample["input_ids"].tolist() == [sample["id"], sample["id"] + 1], (
                    f"Incomplete H2D or MP broadcast: sample={sample}"
                )
            # Independent compute is enqueued before routing the next step.
            compute = torch.ones((64, 64), device=device)
            compute = compute @ compute
            loader.prefetch_plan()
            loader.prefetch()
            ids = torch.tensor([sample["id"] for sample in batch], device=device)
            gathered = [torch.empty_like(ids) for _ in range(4)]
            dist.all_gather(gathered, ids)
            values = [value.tolist() for value in gathered]
            assert values[0] == values[1] and values[2] == values[3], f"MP mismatch: {values}"
            actual = sorted(values[0] + values[2])
            expected = list(range(step * 4, step * 4 + 4))
            assert actual == expected, f"Selection changed: actual={actual}, expected={expected}"
            if step == 0:
                loader.wait_for_prefetch()
                assert loader.state_dict()["step"] == 1, f"Speculative progress committed: step={step}"
            assert compute[0, 0].item() == 64, f"Concurrent compute failed at step={step}"
        assert not list(loader), "Loader should be exhausted after 3 steps"


def _local(device: torch.device) -> None:
    rank = dist.get_rank()
    mesh = SimpleNamespace(mesh_shape=(4,), mesh_dim_names=("dp",), rank_list=(0, 1, 2, 3))
    source = [
        [[_Dataset()[step * 8 + rank * 2 + index] for index in range(2)]]
        for step in range(3)
    ]
    loader = build_distributed_dataloader(
        None, mesh, DistributedDatasetConfig(seq_len=8, local_batch_size=1),
        external_step_source=source, metadata_fn=_metadata,
        pack_fn=lambda samples, _: tuple(samples), collate_fn=list,
        cost_model=lambda metadata: metadata.cost, device=device,
    )
    try:
        for step in range(3):
            batch = next(loader)[0]
            assert batch[0]["input_ids"].device == device, f"Local source did not return device data: {batch}"
            assert batch[0]["offsets"].device == device, f"Local offsets did not move: {batch}"
            assert loader.last_host_batch[0][0]["input_ids"].device.type == "cpu", "Host metering view moved"
            assert loader.last_host_batch[0][0]["offsets"].device.type == "cpu", "Host offsets moved"
            loader.prefetch_plan()
            loader.prefetch()
            ids = torch.tensor([sample["id"] for sample in batch], device=device)
            gathered = [torch.empty_like(ids) for _ in range(4)]
            dist.all_gather(gathered, ids)
            actual = sorted(value for peer in gathered for value in peer.cpu().tolist())
            assert actual == list(range(step * 8, step * 8 + 8)), f"Local sample union changed: {actual}"
            assert loader.last_balance_stats["moved_samples"] > 0, "Expected a real payload redistribution"
        assert next(loader, None) is None, "Local source should be exhausted"
    finally:
        loader.close()


def test_buffered_device_loading_hccl() -> None:
    """Verify native online, metadata, mixed-backend MP and raw-step loaders on NPU."""
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=90))
    try:
        for backend in ("hccl", "gloo"):
            for metadata_mode in (False, True):
                _native(device, backend, metadata_mode)
        _local(device)
        dist.barrier()
    finally:
        torch.npu.synchronize()
        dist.destroy_process_group()
