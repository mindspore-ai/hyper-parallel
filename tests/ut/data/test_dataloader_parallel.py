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
"""Shared source ownership and single-distribution batch runtime tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.data.batching.build_dataloader import FixedBatchDataLoader
from hyper_parallel.high_availability.recovery_state import TrainState
from hyper_parallel.data.batching.get_batch import OmniParallelBatch, TextParallelBatch
from hyper_parallel.data.parallel import (
    DataLoaderParallelContext, TPBatchBroadcaster, build_dataset_for_dataloader,
    create_dataloader_parallel_context,
)


class _RandomRecoveryDataset(torch.utils.data.Dataset):
    """Keep data-side random state inside the native worker snapshot."""

    def __init__(self) -> None:
        """Start a private random stream for the Dataset."""
        self.generator = torch.Generator().manual_seed(19)

    def __len__(self) -> int:
        """Return enough samples to leave prefetched batches pending."""
        return 48

    def __getitem__(self, index: int) -> torch.Tensor:
        """Pair a sample index with a stateful random value."""
        return torch.tensor([index, torch.rand((), generator=self.generator).item()])

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Capture the worker-local random stream."""
        return {"rng": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore the worker-local random stream."""
        self.generator.set_state(state["rng"])


class TestDataLoaderWorkerRecovery(unittest.TestCase):
    """Replay consumed data while old training frames still retain their iterators."""

    def test_next_epoch_worker_seed_matches_after_restore(self):
        """Keep the next epoch's worker seed unchanged by mid-epoch reconstruction."""
        loader = FixedBatchDataLoader(torch.arange(12), batch_size=2, num_workers=1)
        restored = FixedBatchDataLoader(torch.arange(12), batch_size=2, num_workers=1)
        source_state = TrainState(loader, {}, generators={"data_loader": loader.generator})
        target_state = TrainState(restored, {}, generators={"data_loader": restored.generator})
        try:
            iterator = iter(loader)
            next(iterator)
            snapshot = source_state.capture()
            list(iterator)
            expected = iter(loader)
            target_state.restore(snapshot["rank_state"], snapshot["replica_state"])
            iterator = iter(restored)
            target_state.restore_pending_rng()
            list(iterator)
            actual = iter(restored)
            self.assertEqual(actual._base_seed, expected._base_seed)
            torch.testing.assert_close(restored.generator.get_state(), loader.generator.get_state(), rtol=0, atol=0)
        finally:
            for active in (loader, restored):
                if active._iterator is not None:
                    active._iterator._shutdown_workers()

    def test_worker_snapshots_replay_on_survivor_and_replacement(self):
        """Restore prefetched data across repeated rollback and fresh-loader startup."""
        for workers in (1, 4):
            with self.subTest(workers=workers):
                options = {
                    "batch_size": 2,
                    "num_workers": workers,
                    "prefetch_factor": 2,
                }
                options["dataset"] = _RandomRecoveryDataset()
                loader = FixedBatchDataLoader(**options)
                iterator = iter(loader)
                options["dataset"] = _RandomRecoveryDataset()
                replacement = FixedBatchDataLoader(**options)
                state_codec = TrainState(loader, {}, generators={"data_loader": loader.generator})
                replacement_codec = TrainState(replacement, {}, generators={"data_loader": replacement.generator})
                try:
                    for _ in range(3):
                        next(iterator)
                    state = state_codec.capture()
                    generator_state = loader.generator.get_state()
                    expected = [next(iterator) for _ in range(3)]
                    for _ in range(2):
                        old_iterator = iterator
                        state_codec.restore(state["rank_state"], state["replica_state"])
                        self.assertTrue(all(not worker.is_alive() for worker in old_iterator._workers))
                        iterator = iter(loader)
                        state_codec.restore_pending_rng()
                        torch.testing.assert_close(loader.generator.get_state(), generator_state, rtol=0, atol=0)
                        for batch in expected:
                            torch.testing.assert_close(next(iterator), batch)
                    replacement_codec.restore(state["rank_state"], state["replica_state"])
                    restored = iter(replacement)
                    replacement_codec.restore_pending_rng()
                    torch.testing.assert_close(replacement.generator.get_state(), generator_state, rtol=0, atol=0)
                    for batch in expected:
                        torch.testing.assert_close(next(restored), batch)
                finally:
                    for active in (loader, replacement):
                        if active._iterator is not None:
                            active._iterator._shutdown_workers()


class TestDataLoaderParallelContext(unittest.TestCase):
    """Dataset construction and consumption follow the same source policy."""

    def test_source_ownership_and_dataset_cache(self):
        """Every balanced peer owns an iterator; native peers only read on TP0."""
        mesh = SimpleNamespace(mesh_dim_names=())
        for source_rank in (None, 0):
            for tp_rank in (0, 1):
                with self.subTest(source_rank=source_rank, tp_rank=tp_rank), patch(
                    "hyper_parallel.data.parallel.dataloader_parallel.dist.get_world_size", return_value=4,
                ):
                    runtime = SimpleNamespace(device_mesh=mesh, tp_size=2, tp_rank=tp_rank, cp_size=2, cp_rank=1)
                    context = create_dataloader_parallel_context(
                        runtime, tp_source_rank=source_rank, shared_storage=False,
                    )
                    owns_source = source_rank is None or tp_rank == 0
                    self.assertEqual(context.build_on_rank(), owns_source)
                    self.assertEqual(context.cp_world_size, 2)
                    factory = Mock(return_value=object())
                    dataset = build_dataset_for_dataloader(factory, context, barrier_needed=False)
                    self.assertEqual(factory.call_count, int(owns_source))
                    self.assertIs(dataset, factory.return_value if owns_source else None)

    def test_balanced_text_and_omni_skip_tp_communication(self):
        """Both runtimes preserve CP slices, dtypes and EOF without another broadcast."""
        batch = {
            "input_ids": torch.arange(8, dtype=torch.int32).reshape(1, 8),
            "labels": torch.tensor([[1, 2, -100, 4, 5, 6, 7, -100]], dtype=torch.int32),
            "cu_seq_lens": torch.tensor([0, 3, 8]),
        }
        for kind in ("text", "omni"):
            for cp_rank in (0, 1):
                with self.subTest(kind=kind, cp_rank=cp_rank):
                    context = DataLoaderParallelContext(
                        tp_source_rank=None, tp_rank=1, tp_world_size=2,
                        cp_rank=cp_rank, cp_world_size=2, distributed_enabled=True,
                    )
                    options = {"mesh_context": SimpleNamespace(sequence_parallel=False),
                               "device": "cpu", "dataloader_context": context}
                    if kind == "text":
                        runtime = TextParallelBatch(
                            **options, tokenizer=None, data_config={}, pp_shared_data=False,
                            source_type="packed_text", reset_position_ids=True,
                        )
                    else:
                        runtime = OmniParallelBatch(**options)
                        # Model-owned transports still execute, using the same local-only broadcaster.
                        runtime.batch_transport = Mock(side_effect=lambda value, broadcaster: (
                            broadcaster.broadcast(value, broadcast_all_fields=True), {"custom_input": 7},
                        ))
                    iterator = iter([batch])
                    with patch("hyper_parallel.data.parallel.batch_parallel.dist.broadcast") as broadcast, patch(
                        "hyper_parallel.data.parallel.batch_parallel.dist.broadcast_object_list",
                    ) as broadcast_objects:
                        model, loss = runtime(iterator)
                        with self.assertRaises(StopIteration):
                            runtime(iterator)
                        broadcast.assert_not_called()
                        broadcast_objects.assert_not_called()
                    local_slice = slice(cp_rank * 4, (cp_rank + 1) * 4)
                    self.assertEqual(model["input_ids"].tolist(), batch["input_ids"][:, local_slice].tolist())
                    self.assertEqual(loss["loss_mask"].tolist(), (batch["labels"][:, local_slice] >= 0).int().tolist())
                    if kind == "text":
                        self.assertEqual(model["input_ids"].dtype, torch.int64)
                        self.assertEqual(model["position_ids"].tolist(), [[0, 1, 2, 0]] if cp_rank == 0 else [[1, 2, 3, 4]])
                    else:
                        self.assertEqual(model["input_ids"].dtype, torch.int32)
                        self.assertEqual(runtime.batch_transport.call_args.args[0]["cu_seq_lens"].tolist(), [0, 3, 8])
                        self.assertEqual(model["custom_input"], 7)
                        runtime.batch_transport.assert_called_once()

    def test_native_source_still_broadcasts(self):
        """The native source sends shape, tokens, labels and global boundaries."""
        context = DataLoaderParallelContext(tp_world_size=2, tp_group=object())
        broadcaster = TPBatchBroadcaster(context, "cpu")
        batch = {"input_ids": torch.arange(4).reshape(1, 4), "labels": torch.ones((1, 4))}
        with patch("hyper_parallel.data.parallel.batch_parallel.dist.broadcast") as broadcast:
            result = broadcaster.broadcast(batch, torch.tensor([0, 4]))
        self.assertEqual(broadcast.call_count, 4)
        self.assertTrue(all(call.kwargs == {"group": context.tp_group, "group_src": 0}
                            for call in broadcast.call_args_list))
        self.assertEqual(result["labels"].dtype, torch.int64)
        self.assertEqual(result["cu_seq_lens"].dtype, torch.int32)
