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

import torch
import torch.distributed as dist

from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig, SampleMetadata, TensorFieldSpec, WorkloadCost, build_distributed_dataloader,
)


class _Dataset:
    """Return CPU tensors whose complete batch must move to the training device."""

    def __len__(self) -> int:
        """Return two three-round accumulation windows."""
        return 24

    def __getitem__(self, index: int) -> dict:
        """Build one small CPU sample with known tensor values."""
        return {"id": index, "input_ids": torch.tensor([index, index + 1]), "offsets": torch.tensor([0, 2])}


def _metadata(sample: dict) -> SampleMetadata:
    return SampleMetadata(2, cost=WorkloadCost(llm=9 if sample["id"] % 4 < 2 else 1))


def _native(device: torch.device, backend: str, metadata_mode: bool) -> None:
    """Verify native device batches, Host buffering and metadata-mode sample conservation."""
    # Inspect internal queues and ownership to verify prefetch lifetime invariants.
    # pylint: disable=protected-access
    rank = dist.get_rank()
    mesh = SimpleNamespace(mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"), rank_list=(0, 1, 2, 3))
    dataset = _Dataset()
    sampler = build_dataset_batch_sampler(
        total_samples=len(dataset), micro_batch_size=2, global_batch_size=12, dp_world_size=2, dp_rank=rank // 2,
    )
    options = {"metadata": [_metadata(dataset[index]) for index in range(len(dataset))]} if metadata_mode else {
        "metadata_fn": _metadata,
    }
    sample_spec = {
        "id": TensorFieldSpec((), torch.int64),
        "input_ids": TensorFieldSpec((2,), torch.int64),
        "offsets": TensorFieldSpec((2,), torch.int64),
    }
    with build_distributed_dataloader(
            dataset, mesh, DistributedDatasetConfig(
                seq_len=8, local_batch_size=2, communication_backend=backend, metadata_mode=metadata_mode,
                balance_window_size=3,
            ),
            batch_sampler=sampler, device=device,
            collate_fn=lambda samples: tuple({**sample, "id": torch.tensor(sample["id"])} for sample in samples),
            model_batch_spec=(sample_spec, sample_spec),
            cost_model=lambda metadata: metadata.cost, **options,
    ) as loader:
        window_ids = []
        for step in range(6):
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
                del sample
            # Independent compute is enqueued before routing the next step.
            compute = torch.ones((64, 64), device=device)
            compute = compute @ compute
            loader.prefetch()
            ids = torch.tensor([sample["id"] for sample in batch], device=device)
            gathered = [torch.empty_like(ids) for _ in range(4)]
            dist.all_gather(gathered, ids)
            values = [value.tolist() for value in gathered]
            assert values[0] == values[1] and values[2] == values[3], f"MP mismatch: {values}"
            window_ids.extend(values[0] + values[2])
            if step % 3 == 2:
                actual = sorted(window_ids)
                expected = list(range((step - 2) * 4, (step + 1) * 4))
                assert actual == expected, f"Window selection changed: actual={actual}, expected={expected}"
                window_ids.clear()
            if step == 0:
                loader.wait_for_prefetch()
                assert loader.state_dict()["_index_sampler_state"]["consumed_samples"] == 0, (
                    f"Incomplete step committed: step={step}"
                )
                if loader._topology.is_constructor:
                    for host_batch in loader._current.batches:
                        assert all(item["input_ids"].device.type == "cpu" for item in host_batch), (
                            f"Host window contains device inputs: rank={rank}"
                        )
            assert compute[0, 0].item() == 64, f"Concurrent compute failed at step={step}"
            del batch
        assert not list(loader), "Loader should be exhausted after 6 micro-batches"


def _local(device: torch.device) -> None:
    """Verify external-step device delivery while preserving the original Host view."""
    rank = dist.get_rank()
    mesh = SimpleNamespace(mesh_shape=(4,), mesh_dim_names=("dp",), rank_list=(0, 1, 2, 3))
    source = [
        [[_Dataset()[step * 4 + rank]]]
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
            assert batch[0]["input_ids"][0].item() == step * 4 + rank, f"Local step changed: {batch}"
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
