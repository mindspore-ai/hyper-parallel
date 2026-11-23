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
"""Finite evaluation data planning, isolated from training sampler semantics."""

from __future__ import annotations

from itertools import islice
from typing import Any, Iterator

from torch.utils.data import IterableDataset


class EvaluationBatchSampler:
    """Shard a fixed global range exactly once, keeping every incomplete tail."""

    def __init__(self, total: int, batch_size: int, rank: int, world_size: int) -> None:
        """Describe a repeatable lane, with a dummy source item for an empty lane."""
        for name, value in (("total", total), ("batch_size", batch_size), ("world_size", world_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"evaluation sampler {name} must be a positive integer")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
            raise ValueError("evaluation sampler rank must be in [0, world_size)")
        self.batch_size = batch_size
        self.dp_world_size = world_size
        self.indices = range(rank, total, world_size)
        self.has_samples = bool(self.indices)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield deterministic batches without updating a training resume cursor."""
        indices = self.indices or range(1)
        for start in range(0, len(indices), self.batch_size):
            yield list(indices[start:start + self.batch_size])

    def __len__(self) -> int:
        """Return the number of source batches, including an empty lane's dummy."""
        return (max(len(self.indices), 1) + self.batch_size - 1) // self.batch_size


def build_evaluation_dataloader(trainer: Any, dataloader_config: Any) -> None:
    """Build only the selected split using the caller's configured loader and collator.

    The evaluation range is applied before DP sharding, independently of the
    dataset builder. No training drop_last, shuffle or resume policy is used.
    """
    evaluation = trainer.config.evaluation
    split = "valid" if evaluation.split == "dataset" else evaluation.split
    dataset = getattr(trainer, f"{split}_dataset", None)
    if dataset is None:
        raise ValueError(f"evaluation split {evaluation.split!r} is empty")
    if isinstance(dataset, IterableDataset) or not hasattr(dataset, "__len__"):
        if evaluation.max_samples is None:
            raise ValueError("an iterable evaluation source requires evaluation.max_samples")
        dataset = list(islice(dataset, evaluation.max_samples))
    total = len(dataset)
    if evaluation.max_samples is not None:
        total = min(total, evaluation.max_samples)
    if total == 0:
        raise ValueError("evaluation dataset contains no samples")
    if getattr(dataloader_config, "load_balance", None) is not None:
        raise NotImplementedError("standalone evaluation does not support dataloader.load_balance")
    sampler = EvaluationBatchSampler(total, evaluation.batch_size, trainer.mesh.dp_rank, trainer.mesh.dp_size)
    loader = dataloader_config.build(
        dataset=dataset, batch_sampler=sampler, collate_fn=trainer.collate_fn,
        batch_size=evaluation.batch_size, sampler_type="single", drop_last=False,
        dp_world_size=trainer.mesh.dp_size, max_seq_len=getattr(trainer.data_transform, "max_seq_len", None),
        seed=trainer.config.training.seed,
    )
    setattr(trainer, f"{split}_dataloader", loader)
    setattr(trainer, f"{split}_batch_sampler", sampler)
