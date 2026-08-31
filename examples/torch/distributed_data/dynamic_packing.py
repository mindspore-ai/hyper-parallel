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
"""Four-process CPU example of sample-level balancing and dynamic packing.

Run from the repository root:

``python -m torch.distributed.run --standalone --nproc-per-node=4 \
examples/torch/distributed_data/dynamic_packing.py``
"""
# This example intentionally exercises the PyTorch-only distributed-data API.

from __future__ import annotations

from typing import Any, Sequence

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    WorkloadCost,
    build_distributed_dataloader,
)


class TokenDataset:
    """Small mapping Dataset with variable sample lengths and costs."""

    def __init__(self) -> None:
        """Build enough samples for two distributed batches."""
        lengths = (11, 5, 9, 7, 12, 4, 8, 8, 10, 6, 13, 3, 9, 7, 12, 4)
        self._samples = [
            {
                "sample_id": index,
                "input_ids": torch.full((length,), index, dtype=torch.int64),
                "vision_cost": float((index % 4) + 1),
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
    """Describe packing length and a synthetic multimodal encoder cost."""
    return SampleMetadata(
        pack_tokens=int(sample["input_ids"].numel()),
        cost=WorkloadCost(encoder=sample["vision_cost"]),
        sample_id=sample["sample_id"],
    )


def pack_fn(samples: Sequence[dict[str, Any]], seq_len: int) -> dict[str, Any]:
    """Concatenate one planned bin and pad it to the configured capacity."""
    tokens = torch.cat([sample["input_ids"] for sample in samples])
    padded = torch.full((seq_len,), -1, dtype=tokens.dtype)
    padded[:tokens.numel()] = tokens
    return {
        "input_ids": padded,
        "sample_ids": tuple(sample["sample_id"] for sample in samples),
        "valid_tokens": int(tokens.numel()),
    }


def collate_fn(packed_sequences: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack the constructor's planned sequence bins into one local batch."""
    return {
        "input_ids": torch.stack([sequence["input_ids"] for sequence in packed_sequences]),
        "sample_ids": tuple(sequence["sample_ids"] for sequence in packed_sequences),
        "valid_tokens": tuple(sequence["valid_tokens"] for sequence in packed_sequences),
    }


def main() -> None:
    """Initialize Gloo and print every rank's constructed batches."""
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    loader = build_distributed_dataloader(
        TokenDataset(),
        _Mesh(),
        DistributedDatasetConfig(
            seq_len=16,
            local_batch_size=2,
            dataset_reader_ranks=(0, 1, 2, 3),
        ),
        metadata_fn=metadata_fn,
        pack_fn=pack_fn,
        collate_fn=collate_fn,
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
