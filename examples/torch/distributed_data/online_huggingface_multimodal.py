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
"""Run sidecar-planned DP/MP data loading over downloaded Hugging Face JPEG samples."""
# This file is intentionally a PyTorch-only runnable example.
# pylint: disable=C0413,forbidden-backend-import

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

# HyperParallel selects its framework before importing platform modules.
os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from PIL import Image
import torch
import torch.distributed as dist

try:
    import torch_npu  # pylint: disable=W0611
except ImportError:
    torch_npu = None

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.distributed_data import DistributedDatasetConfig, SampleMeta, build_distributed_dataset


class HuggingFaceManifestDataset:
    """Expose local image-caption records and their lightweight metadata sidecar."""

    _REQUIRED_FIELDS = ("sample_id", "image", "text", "width", "height", "jpeg_bytes", "sha256")

    def __init__(self, manifest_path: Path, sample_count: int) -> None:
        """Load and validate a finite manifest window.

        Args:
            manifest_path: JSON manifest generated from the downloaded Hugging Face dataset.
            sample_count: Number of records used by one global optimizer step.
        """
        try:
            rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read JSON manifest {manifest_path}.") from exc
        if not isinstance(rows, list):
            raise ValueError(f"Manifest {manifest_path} must contain a JSON list.")
        if len(rows) < sample_count:
            raise ValueError(
                f"One distributed step requires {sample_count} samples, but manifest contains {len(rows)}."
            )

        self._manifest_dir = manifest_path.parent
        self._rows = rows[:sample_count]
        sidecar_metadata = []
        for index, row in enumerate(self._rows):
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row {index} must be a JSON object.")
            missing = [field for field in self._REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(f"Manifest row {index} is missing fields {missing}.")
            sidecar_metadata.append(
                SampleMeta(
                    sample_id=index,
                    text_tokens=max(1, len(row["text"].split())),
                    vision_tokens=math.ceil(row["width"] / 14) * math.ceil(row["height"] / 14),
                    io_bytes=row["jpeg_bytes"],
                )
            )
        self._sidecar_metadata = tuple(sidecar_metadata)

    def __len__(self) -> int:
        """Return the records in the selected global-step window."""
        return len(self._rows)

    @property
    def sidecar_metadata(self) -> tuple[SampleMeta, ...]:
        """Return precomputed sample costs without reading JPEG contents."""
        return self._sidecar_metadata

    def __getitem__(self, data_index: int) -> dict[str, Any]:
        """Read one caption and its original JPEG bytes in a DataLoader worker."""
        row = self._rows[data_index]
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = self._manifest_dir / image_path
        return {
            "record_id": row["sample_id"],
            "data_index": data_index,
            "image_bytes": image_path.read_bytes(),
            "caption": row["text"],
            "width": row["width"],
            "height": row["height"],
            "sha256": row["sha256"],
            "worker_pid": os.getpid(),
        }


def decode_and_collate(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Decode sidecar-planned JPEGs and return a validated training batch."""
    records = []
    for sample in samples:
        digest = hashlib.sha256(sample["image_bytes"]).hexdigest()
        if digest != sample["sha256"]:
            raise ValueError(f"JPEG checksum mismatch for {sample['record_id']}.")
        with Image.open(io.BytesIO(sample["image_bytes"])) as image:
            image.load()
            if image.size != (sample["width"], sample["height"]):
                raise ValueError(f"JPEG dimensions mismatch for {sample['record_id']}.")
        records.append(
            {
                "record_id": sample["record_id"],
                "data_index": sample["data_index"],
                "caption": sample["caption"],
                "worker_pid": sample["worker_pid"],
                "sha256": digest,
            }
        )
    return tuple(records)


def _all_gather_object(value: Any, world_size: int) -> list[Any]:
    """Gather one small validation object from every training rank."""
    gathered = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


def _parse_args() -> argparse.Namespace:
    """Parse example arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=("hccl", "gloo"), default="hccl")
    parser.add_argument("--micro-batch-num", type=int, default=4)
    parser.add_argument("--model-parallel-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--double-buffer", action="store_true")
    return parser.parse_args()


def _initialize_distributed(backend: str) -> tuple[int, int, Any, str]:
    """Initialize a pure-DP process group and return its local communication device."""
    local_rank = int(os.environ["LOCAL_RANK"])
    if backend == "hccl":
        if torch_npu is None:
            raise ValueError("The HCCL example requires torch_npu.")
        torch.npu.set_device(local_rank)
        communication_device = torch.device(f"npu:{local_rank}")
        device_type = "npu"
    else:
        communication_device = None
        device_type = "cpu"
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), communication_device, device_type


def _validate_global_result(
    local_records: list[dict[str, Any]],
    local_plan_ids: list[str],
    step_plan_id: str,
    sample_count: int,
    num_workers: int,
    mesh_shape: tuple[int, int],
) -> int:
    """Validate sidecar planning, sample reassignment, and MP replication."""
    data_parallel_size, model_parallel_size = mesh_shape
    world_size = data_parallel_size * model_parallel_size
    gathered_plan_ids = _all_gather_object(local_plan_ids, world_size)
    if any(plan_ids != local_plan_ids for plan_ids in gathered_plan_ids):
        raise ValueError(f"Ranks produced inconsistent sidecar plans: {gathered_plan_ids}.")
    gathered_step_ids = _all_gather_object(step_plan_id, world_size)
    if len(set(gathered_step_ids)) != 1:
        raise ValueError(f"Ranks produced inconsistent step plan IDs: {gathered_step_ids}.")

    gathered_records = _all_gather_object(local_records, world_size)
    representative_records = []
    for data_rank in range(data_parallel_size):
        group_start = data_rank * model_parallel_size
        group_records = gathered_records[group_start: group_start + model_parallel_size]
        if any(records != group_records[0] for records in group_records[1:]):
            raise ValueError(f"Model-parallel ranks for data rank {data_rank} received different microbatches.")
        representative_records.append(group_records[0])

    flat_records = [record for rank_records in representative_records for record in rank_records]
    received_indices = [record["data_index"] for record in flat_records]
    if len(received_indices) != len(set(received_indices)) or set(received_indices) != set(range(sample_count)):
        raise ValueError("Distributed step did not preserve every manifest sample exactly once.")

    main_pids = set(_all_gather_object(os.getpid(), world_size))
    worker_pids = {record["worker_pid"] for record in flat_records}
    if num_workers > 0 and worker_pids & main_pids:
        raise ValueError("Expected sidecar-planned reads to run in DataLoader worker processes.")
    if num_workers == 0 and not worker_pids <= main_pids:
        raise ValueError("num_workers=0 unexpectedly returned samples from worker processes.")

    reassigned_samples = sum(
        record["data_index"] % data_parallel_size != target_data_rank
        for target_data_rank, rank_records in enumerate(representative_records)
        for record in rank_records
    )
    if data_parallel_size > 1 and reassigned_samples == 0:
        raise ValueError("The sidecar plan did not reassign any samples across data ranks.")
    expected_local_samples = sample_count // data_parallel_size
    if any(len(records) != expected_local_samples for records in gathered_records):
        raise ValueError(f"Each rank must receive {expected_local_samples} samples, got {gathered_records}.")
    return reassigned_samples


def main() -> None:
    """Run and validate one sidecar-planned optimizer step."""
    args = _parse_args()
    rank, world_size, communication_device, device_type = _initialize_distributed(args.backend)
    if args.model_parallel_size < 1 or world_size % args.model_parallel_size != 0:
        raise ValueError(
            f"model_parallel_size must be positive and divide world_size={world_size}, "
            f"but got {args.model_parallel_size}."
        )
    data_parallel_size = world_size // args.model_parallel_size
    mesh_shape = (data_parallel_size, args.model_parallel_size)
    config = DistributedDatasetConfig(
        micro_batch_num=args.micro_batch_num,
        prefetch_steps=1,
        dp_dim_names=("dp",),
        double_buffer=args.double_buffer,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    sample_count = data_parallel_size * config.raw_sample_size * config.micro_batch_num
    dataset = HuggingFaceManifestDataset(args.manifest, sample_count)
    mesh_ranks = [
        list(range(data_rank * args.model_parallel_size, (data_rank + 1) * args.model_parallel_size))
        for data_rank in range(data_parallel_size)
    ]
    mesh = DeviceMesh(
        device_type,
        mesh_ranks,
        mesh_dim_names=("dp", "mp"),
        _init_backend=False,
    )
    loader = build_distributed_dataset(
        dataset,
        mesh,
        config,
        metadata=dataset.sidecar_metadata,
        collate_fn=decode_and_collate,
        communication_device=communication_device,
    )

    local_records = []
    local_plan_ids = []
    try:
        step = next(loader)
        for micro_batch in step:
            data_indices = tuple(record["data_index"] for record in micro_batch.data)
            if data_indices != micro_batch.sample_ids:
                raise ValueError(f"Rank {rank} received data in an order different from its plan.")
            local_records.extend(micro_batch.data)
            local_plan_ids.append(micro_batch.plan_id)
        reassigned_samples = _validate_global_result(
            local_records,
            local_plan_ids,
            step.plan_id,
            sample_count,
            args.num_workers,
            mesh_shape,
        )
        loader.commit(step.plan_id)
        expected_offset = config.raw_sample_size * config.micro_batch_num
        if loader.consumed_offset != expected_offset:
            raise ValueError(f"Expected consumed_offset={expected_offset}, got {loader.consumed_offset}.")
    finally:
        loader.close()

    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "dataset": "diffusers/pokemon-gpt4-captions",
                    "backend": args.backend,
                    "world_size": world_size,
                    "data_parallel_size": data_parallel_size,
                    "model_parallel_size": args.model_parallel_size,
                    "data_owner_ranks": list(range(0, world_size, args.model_parallel_size)),
                    "model_parallel_groups": mesh_ranks,
                    "samples": sample_count,
                    "dataset_units_per_micro_batch": config.raw_sample_size,
                    "micro_batch_num": args.micro_batch_num,
                    "num_workers": args.num_workers,
                    "double_buffer": args.double_buffer,
                    "metadata_mode": "sidecar",
                    "sample_loading": "shared_storage_index_fetch",
                    "reassigned_samples": reassigned_samples,
                    "consumed_offset": loader.consumed_offset,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
