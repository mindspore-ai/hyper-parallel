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
"""Run online multimodal planning over downloaded Hugging Face JPEG samples."""
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
    """Read locally downloaded image-caption records through map-style indexing."""

    _REQUIRED_FIELDS = ("sample_id", "image", "text", "width", "height", "sha256")

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
        for index, row in enumerate(self._rows):
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row {index} must be a JSON object.")
            missing = [field for field in self._REQUIRED_FIELDS if field not in row]
            if missing:
                raise ValueError(f"Manifest row {index} is missing fields {missing}.")

    def __len__(self) -> int:
        """Return the records in the selected global-step window."""
        return len(self._rows)

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


def derive_online_metadata(sample: dict[str, Any], sample_id: int) -> SampleMeta:
    """Estimate text, vision, and I/O cost after the owner reads a raw sample."""
    vision_tokens = math.ceil(sample["width"] / 14) * math.ceil(sample["height"] / 14)
    return SampleMeta(
        sample_id=sample_id,
        text_tokens=max(1, len(sample["caption"].split())),
        vision_tokens=vision_tokens,
        io_bytes=len(sample["image_bytes"]),
    )


def decode_and_collate(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Decode target-rank JPEGs after A2A and return a validated training batch."""
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
    parser.add_argument("--raw-sample-size", type=int, default=2)
    parser.add_argument("--micro-batch-num", type=int, default=4)
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
    step_replay_id: str,
    sample_count: int,
    num_workers: int,
    rank: int,
    world_size: int,
) -> int:
    """Validate plan agreement, worker execution, A2A movement, and sample coverage."""
    gathered_plan_ids = _all_gather_object(local_plan_ids, world_size)
    if any(plan_ids != local_plan_ids for plan_ids in gathered_plan_ids):
        raise ValueError(f"Ranks produced inconsistent online plans: {gathered_plan_ids}.")
    gathered_step_ids = _all_gather_object(step_replay_id, world_size)
    if len(set(gathered_step_ids)) != 1:
        raise ValueError(f"Ranks produced inconsistent step replay IDs: {gathered_step_ids}.")

    gathered_records = _all_gather_object(local_records, world_size)
    flat_records = [record for rank_records in gathered_records for record in rank_records]
    received_indices = [record["data_index"] for record in flat_records]
    if len(received_indices) != len(set(received_indices)) or set(received_indices) != set(range(sample_count)):
        raise ValueError("Distributed step did not preserve every manifest sample exactly once.")

    main_pids = set(_all_gather_object(os.getpid(), world_size))
    worker_pids = {record["worker_pid"] for record in flat_records}
    if num_workers > 0 and worker_pids & main_pids:
        raise ValueError("Expected online reads to run in DataLoader worker processes.")
    if num_workers == 0 and not worker_pids <= main_pids:
        raise ValueError("num_workers=0 unexpectedly returned samples from worker processes.")

    cross_owner_samples = sum(
        record["data_index"] % world_size != target_rank
        for target_rank, rank_records in enumerate(gathered_records)
        for record in rank_records
    )
    if world_size > 1 and cross_owner_samples == 0:
        raise ValueError("The balanced plan did not exercise cross-owner A2A movement.")
    if len(local_records) * world_size != sample_count:
        raise ValueError(f"Rank {rank} received an unexpected local sample count {len(local_records)}.")
    return cross_owner_samples


def main() -> None:
    """Run and validate one online-planned optimizer step."""
    args = _parse_args()
    rank, world_size, communication_device, device_type = _initialize_distributed(args.backend)
    sample_count = world_size * args.raw_sample_size * args.micro_batch_num
    dataset = HuggingFaceManifestDataset(args.manifest, sample_count)
    mesh = DeviceMesh(
        device_type,
        list(range(world_size)),
        mesh_dim_names=("dp",),
        _init_backend=False,
    )
    loader = build_distributed_dataset(
        dataset,
        mesh,
        DistributedDatasetConfig(
            raw_sample_size=args.raw_sample_size,
            micro_batch_num=args.micro_batch_num,
            prefetch_steps=1,
            sample_transport="packed_bytes_a2a",
            double_buffer=args.double_buffer,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        ),
        metadata_fn=derive_online_metadata,
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
            local_plan_ids.append(micro_batch.replay_id)
        cross_owner_samples = _validate_global_result(
            local_records,
            local_plan_ids,
            step.replay_id,
            sample_count,
            args.num_workers,
            rank,
            world_size,
        )
        loader.commit(step.replay_id)
        expected_offset = args.raw_sample_size * args.micro_batch_num
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
                    "samples": sample_count,
                    "raw_sample_size": args.raw_sample_size,
                    "micro_batch_num": args.micro_batch_num,
                    "num_workers": args.num_workers,
                    "double_buffer": args.double_buffer,
                    "transport": "packed_bytes_a2a",
                    "cross_owner_samples": cross_owner_samples,
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
