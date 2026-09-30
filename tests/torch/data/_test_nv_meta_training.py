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
"""Read selected nv-meta parts and agree on complete distributed training steps."""

import os
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from hyper_parallel.data.batching.get_batch import SynchronizedBatchReader
from hyper_parallel.data.nv_meta.reader import NvMetaDataset
from hyper_parallel.data.online.source_views import IndexedMappingView
from tests.fixtures.data.nv_meta import make_nv_meta


class _BatchRuntime:
    """Represent an owner runtime or a TP-style broadcast peer."""

    def __init__(self, *, owner_only: bool = False) -> None:
        """Select independent readers or one owner with a broadcast peer."""
        self.owner_only = owner_only
        self.calls = 0
        self.parallel_context = SimpleNamespace(build_on_rank=lambda: not owner_only or dist.get_rank() == 0)

    def __call__(self, iterator: Any) -> torch.Tensor:
        """Read locally or receive the owner's next prepared batch."""
        self.calls += 1
        if not self.owner_only:
            return next(iterator)
        value = next(iterator) if dist.get_rank() == 0 else torch.zeros(1, 1)
        dist.broadcast(value, src=0)
        return value


def _check_uneven_training_steps(dataset: NvMetaDataset) -> None:
    """Train on indexed text parts without reading the unselected media parts."""
    model = DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
    initial_weight = model.module.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    indices = range(5) if dist.get_rank() == 0 else range(5, 8)
    source = iter(DataLoader(
        Subset(IndexedMappingView(dataset), indices), batch_size=1,
        collate_fn=lambda samples: torch.tensor([[float(samples[0]["txt"])]]),
    ))
    runtime = _BatchRuntime()
    reader = SynchronizedBatchReader(runtime, num_micro_batches=2, device="cpu")
    steps = 0
    while True:
        try:
            reader.prepare_step(source)
        except StopIteration:
            break
        optimizer.zero_grad()
        for _ in range(2):
            model(reader(source)).sum().backward()
        optimizer.step()
        steps += 1
    counts = [None] * dist.get_world_size()
    dist.all_gather_object(counts, (steps, runtime.calls))
    expected = [(1, 2), (1, 2)]
    assert counts == expected, f"Expected complete-step counts {expected}, got {counts}"
    # DDP averages gradients from inputs (1, 6), then (2, 7): total gradient 8.
    torch.testing.assert_close(model.module.weight, initial_weight - 0.08)


def _check_non_owner_eof() -> None:
    """Stop broadcast peers together before the owner's partial step."""
    runtime = _BatchRuntime(owner_only=True)
    reader = SynchronizedBatchReader(runtime, num_micro_batches=2, device="cpu")
    source = iter([torch.ones(1, 1)] * 3) if dist.get_rank() == 0 else None
    received = 0
    try:
        while True:
            reader(source)
            received += 1
    except StopIteration:
        pass
    assert received == 2, f"Expected two broadcast batches on every rank, got {received}"


def _check_reader_error(dataset: NvMetaDataset) -> None:
    """Propagate an owner read failure before either rank runs the model."""
    def source_items() -> Iterator[torch.Tensor]:
        """Truncate one rank's shard midway through preparing a step."""
        yield torch.tensor([[float(dataset[0]["txt"])]])
        if dist.get_rank() == 0:
            dataset.close()
            (dataset.dataset_root / "samples.tar").write_bytes(b"")
        yield torch.tensor([[float(dataset[1]["txt"])]])

    runtime = _BatchRuntime()
    reader = SynchronizedBatchReader(runtime, num_micro_batches=2, device="cpu")
    failed = False
    try:
        reader(source_items())
    except RuntimeError as error:
        failed = True
        if dist.get_rank() == 0:
            assert isinstance(error.__cause__, OSError), f"Expected shard I/O failure, got {error.__cause__!r}"
    assert failed and runtime.calls == 0, f"Expected failure before model work, got {failed=}, {runtime.calls=}"
    dist.barrier()


def test_nv_meta_training(tmp_path: Path) -> None:
    """Train from selected nv-meta parts and stop peers together on EOF/error."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", timeout=timedelta(seconds=45), init_method=os.environ.get("HP_TEST_INIT_METHOD", "env://"),
        rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]),
    )
    dataset = None
    try:
        records = [{"txt": str(index + 1).encode(), "jpg": b"unused image", "wav": b"unused audio"}
                   for index in range(8)]
        make_nv_meta(tmp_path, records)
        dataset = NvMetaDataset(tmp_path, required_parts=["txt"], cache_dir=tmp_path / "cache")
        assert dataset[0]["parts"] == {"txt": b"1"}, f"Unexpected selected parts: {dataset[0]['parts']}"
        _check_uneven_training_steps(dataset)
        _check_non_owner_eof()
        _check_reader_error(dataset)
    finally:
        if dataset is not None:
            dataset.close()
        dist.destroy_process_group()
