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
"""Four-process CPU example of sampler-defined, metadata-first sample balancing.

Run from the repository root:

``python -m torch.distributed.run --standalone --nproc-per-node=4 \
examples/torch/distributed_data/dynamic_packing.py``
"""
# This example intentionally exercises the PyTorch-only distributed-data API.

from __future__ import annotations

from typing import Any, Sequence

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.auto_models.components.datasets.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    WorkloadCost,
    build_distributed_dataloader,
)


class TokenDataset:
    """Small mapping Dataset with variable sample lengths."""

    def __init__(self) -> None:
        """Build enough samples for two distributed batches."""
        lengths = (11, 5, 9, 7, 12, 4, 8, 8, 10, 6, 13, 3, 9, 7, 12, 4)
        self._samples = [
            {
                "sample_id": index,
                "input_ids": torch.full((length,), index, dtype=torch.int64),
            }
            for index, length in enumerate(lengths)
        ]

    def __len__(self) -> int:
        """Return sample count."""
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one raw variable-length sample."""
        return self._samples[index]


class _Mesh:
    """Minimal named layout accepted by the distributed-data topology."""

    mesh_shape = (2, 2)
    mesh_dim_names = ("dp", "tp")
    rank_list = (0, 1, 2, 3)


def metadata_fn(sample: dict[str, Any]) -> SampleMetadata:
    """Describe packing length without coupling metadata to a cost formula.

    Args:
        sample: A complete variable-length source sample.
    """
    return SampleMetadata(
        pack_tokens=int(sample["input_ids"].numel()),
        sample_id=sample["sample_id"],
    )


def collate_fn(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad complete samples to equal widths without cross-sample packing.

    Args:
        samples: Whole Dataset outputs assigned to this Constructor.
    """
    padded = torch.full((len(samples), 16), -1, dtype=torch.int64)
    for row, sample in enumerate(samples):
        tokens = sample["input_ids"]
        padded[row, :tokens.numel()] = tokens
    return {
        "input_ids": padded,
        "sample_ids": tuple(sample["sample_id"] for sample in samples),
        "valid_tokens": tuple(sample["input_ids"].numel() for sample in samples),
    }


def main() -> None:
    """Initialize Gloo and print every rank's constructed batches."""
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    dataset = TokenDataset()
    metadata = [metadata_fn(dataset[index]) for index in range(len(dataset))]
    batch_sampler = build_dataset_batch_sampler(
        total_samples=len(dataset), micro_batch_size=2, global_batch_size=4,
        dp_world_size=2, dp_rank=rank // 2,
    )
    loader = build_distributed_dataloader(
        dataset,
        _Mesh(),
        DistributedDatasetConfig(
            seq_len=16,
            local_batch_size=2,
        ),
        metadata=metadata,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        cost_model=lambda sample: WorkloadCost(llm=float(sample.pack_tokens)),
    )
    for step, batch in enumerate(loader):
        print(
            f"rank={rank} step={step} plan={loader.last_plan_id} "
            f"sample_ids={batch['sample_ids']} valid_tokens={batch['valid_tokens']}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
