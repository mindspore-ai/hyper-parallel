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
"""Native and external process balancing with nested DataLoader workers."""

import os
from contextlib import closing
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader

from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import (
    CausalAttentionCostModel, DistributedDatasetConfig, SampleMetadata, TensorFieldSpec, build_distributed_dataloader,
)


def _bins(samples: list[dict]) -> list[list[dict]]:
    return [[sample] for sample in samples]


def _pack(samples: list[dict], seq_len: int) -> dict:
    del seq_len
    return {"ids": torch.tensor([item["id"] for item in samples]),
            "tokens": torch.cat([item["tokens"] for item in samples]),
            "producer": os.getpid(), "backend": dist.get_backend()}


class _NativeDataset:
    """Expose read ownership independently of packing ownership."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict:
        return {"id": index, "reader": os.getpid()}


def _native_collate(samples: list[dict]) -> dict:
    return {"ids": torch.tensor([sample["id"] for sample in samples]),
            "reader": torch.tensor([sample["reader"] for sample in samples]),
            "producer": torch.tensor(os.getpid())}


def _native_loader(mesh: object, device: torch.device) -> object:
    rank = dist.get_rank()
    sampler = build_dataset_batch_sampler(total_samples=8, micro_batch_size=1, global_batch_size=2,
                                          dp_rank=rank // 2, dp_world_size=2)
    return build_distributed_dataloader(
        _NativeDataset() if rank % 2 == 0 else None, mesh,
        DistributedDatasetConfig(seq_len=8, local_batch_size=1, communication_backend="hccl", planner_rank=2),
        batch_sampler=sampler, metadata_fn=lambda _: SampleMetadata(1), collate_fn=_native_collate,
        cost_model=CausalAttentionCostModel(), device=device,
        dataloader_kwargs={"num_workers": 1, "persistent_workers": True},
        model_batch_spec={"ids": TensorFieldSpec((1,), torch.int64),
                          "reader": TensorFieldSpec((1,), torch.int64),
                          "producer": TensorFieldSpec((), torch.int64)},
    )


def test_native_process_hccl() -> None:
    """Restore DP2/CP2 data with nested workers and a nonzero child planner."""
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    try:
        mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "cp"))
        with _native_loader(mesh, device) as loader:
            first = next(loader)
            producer = first["producer"].item()
            assert producer != os.getpid(), "Packing ran in the training process"
            assert first["reader"].item() != producer, "Source read bypassed the nested DataLoader worker"
            assert first["ids"].device == device, "Native delivery bypassed parent H2D"
            state = loader.state_dict()
            expected = [batch["ids"].tolist() for batch in loader]
        with _native_loader(mesh, device) as resumed:
            resumed.load_state_dict(state)
            actual = [batch["ids"].tolist() for batch in resumed]
            assert actual == expected, f"Delivered-cursor resume changed inputs: {actual}, {expected}"
        dist.barrier()
    finally:
        torch.npu.synchronize()
        dist.destroy_process_group()


def test_external_process_hccl() -> None:
    """Preserve exact global samples while child HCCL balances a worker-backed source."""
    rank = int(os.environ["RANK"])
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    try:
        mesh = init_device_mesh("npu", (4,), mesh_dim_names=("dp",))
        samples = [{"id": rank * 100 + index,
                    "tokens": torch.full(((9, 1, 4, 4)[rank],), rank * 100 + index, dtype=torch.int64)}
                   for index in range(8)]
        source = DataLoader(samples, batch_size=2, collate_fn=_bins, num_workers=1, persistent_workers=True)
        with closing(build_distributed_dataloader(
                None, mesh, DistributedDatasetConfig(seq_len=16, local_batch_size=2, communication_backend="hccl"),
                external_step_source=source, metadata_fn=lambda sample: SampleMetadata(len(sample["tokens"])),
                pack_fn=_pack, collate_fn=list, cost_model=CausalAttentionCostModel(), device=device,
        )) as loader:
            outputs = []
            accepted = []
            for step, batch in enumerate(loader):
                for item in batch:
                    assert item["producer"] != os.getpid(), "Collation ran in the training process"
                    assert item["backend"] == "hccl", "Child metadata used the wrong backend"
                    assert item["tokens"].device == device, "Parent H2D did not deliver device input"
                local_ids = [value for item in batch for value in item["ids"].tolist()]
                all_ids = [[] for _ in range(4)]
                dist.all_gather_object(all_ids, local_ids)
                expected = sorted(peer * 100 + index for peer in range(4) for index in (step * 2, step * 2 + 1))
                assert sorted(value for ids in all_ids for value in ids) == expected, "Sample membership changed"
                outputs.append(local_ids)
                accepted.append(loader.last_balance_stats["accepted"])
            assert len(outputs) == 4, "External source produced an unexpected step count"
            assert any(accepted), "Fixture failed to exercise accepted redistribution"
            loader.set_epoch(1)
            replay = [[value for item in batch for value in item["ids"].tolist()] for batch in loader]
            assert replay == outputs, "Epoch reset changed balanced input order"
        dist.barrier()
    finally:
        torch.npu.synchronize()
        dist.destroy_process_group()
