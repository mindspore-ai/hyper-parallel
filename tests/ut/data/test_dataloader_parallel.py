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

from hyper_parallel.data.batching.get_batch import OmniParallelBatch, TextParallelBatch
from hyper_parallel.data.parallel import (
    DataLoaderParallelContext, TPBatchBroadcaster, build_dataset_for_dataloader,
    create_dataloader_parallel_context,
)


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
