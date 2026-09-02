"""Unit tests for AutoModels Dataset batch samplers."""

from __future__ import annotations

from collections.abc import Sequence

from torch.utils.data.distributed import DistributedSampler

from hyper_parallel.auto_models.components.datasets.parallel.batch_sampler import build_dataset_batch_sampler


class _SizedDataset:
    """Minimal Dataset-sized object accepted by PyTorch DistributedSampler."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def _flatten(batches: Sequence[Sequence[int]]) -> list[int]:
    return [index for batch in batches for index in batch]


def test_distributed_sampler_matches_pytorch_distributed_sampler() -> None:
    total_samples = 29
    dp_world_size = 4
    seed = 123
    epoch = 2

    for dp_rank in range(dp_world_size):
        hyper_sampler = build_dataset_batch_sampler(
            total_samples=total_samples,
            micro_batch_size=1,
            global_batch_size=dp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            sampler_type="distributed",
            sampler_shuffle=True,
            sampler_drop_last=False,
            seed=seed,
        )
        hyper_sampler.set_epoch(epoch)

        torch_sampler = DistributedSampler(
            _SizedDataset(total_samples),
            num_replicas=dp_world_size,
            rank=dp_rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        torch_sampler.set_epoch(epoch)

        assert _flatten(list(hyper_sampler)) == list(torch_sampler)


def test_distributed_sampler_batches_rank_local_indices() -> None:
    total_samples = 30
    dp_world_size = 4
    micro_batch_size = 2
    seed = 7
    dp_rank = 1

    torch_sampler = DistributedSampler(
        _SizedDataset(total_samples),
        num_replicas=dp_world_size,
        rank=dp_rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    expected_indices = list(torch_sampler)
    expected_batches = [
        expected_indices[start:start + micro_batch_size]
        for start in range(0, len(expected_indices), micro_batch_size)
        if len(expected_indices[start:start + micro_batch_size]) == micro_batch_size
    ]

    hyper_sampler = build_dataset_batch_sampler(
        total_samples=total_samples,
        micro_batch_size=micro_batch_size,
        global_batch_size=micro_batch_size * dp_world_size,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        drop_last=True,
        sampler_type="distributed",
        sampler_shuffle=True,
        sampler_drop_last=False,
        seed=seed,
    )

    assert list(hyper_sampler) == expected_batches


def test_distributed_sampler_marks_dropped_tail_consumed() -> None:
    hyper_sampler = build_dataset_batch_sampler(
        total_samples=17,
        micro_batch_size=2,
        global_batch_size=8,
        dp_rank=0,
        dp_world_size=4,
        drop_last=True,
        sampler_type="distributed",
        sampler_shuffle=False,
        sampler_drop_last=False,
        seed=0,
    )

    assert list(hyper_sampler) == [[0, 4], [8, 12]]
    assert hyper_sampler.consumed_samples == hyper_sampler.total_samples
