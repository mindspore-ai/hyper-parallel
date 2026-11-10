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
"""Packed-data balancing through the DeepSeek-V4.1 runtime input adapter."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.data.batching import TextParallelBatch
from hyper_parallel.data.parallel import DataLoaderParallelContext, build_dataset_batch_sampler
from hyper_parallel.distributed_data import DistributedDatasetConfig, SampleMetadata, build_distributed_dataloader
from hyper_parallel.models.deepseek_v41.adapter.data.runtime import DeepseekV41Runtime
from tests.common.mark_utils import arg_mark


class TestDeepseekV41Runtime(unittest.TestCase):
    """Retain document semantics between the repacker, CP and compressed attention."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_balanced_packed_text_preserves_labels_and_document_geometry(self) -> None:
        """Feature: DS4.1 balanced packed-text runtime.
        Description: Repack real documents and PAD, then build each CP slice.
        Expectation: Labels, positions and attention/Engram boundaries remain aligned.
        """
        source = {
            "input_ids": torch.tensor([4, 5, 6, 7, 8, 9, 10, 11, 2, 2, 2, 2]),
            "labels": torch.tensor([-100, 6, 1, -100, 9, 10, 11, 1, -100, -100, -100, -100]),
            "cu_seq_lens": torch.tensor([0, 3, 12]),
        }
        sampler = build_dataset_batch_sampler(
            total_samples=1, micro_batch_size=1, global_batch_size=1, dp_rank=0, dp_world_size=1,
        )
        mesh = SimpleNamespace(mesh_shape=(1,), mesh_dim_names=("dp",), rank_list=(0,))
        with build_distributed_dataloader(
                [source], mesh,
                DistributedDatasetConfig(seq_len=12, local_batch_size=1, repack_sequences=True, pad_token_id=2),
                batch_sampler=sampler, device="cpu",
                metadata_fn=lambda sample: SampleMetadata(len(sample["input_ids"])),
        ) as loader:
            batch = next(loader)
        self.assertEqual(batch["cu_seq_lens"].tolist(), [0, 3, 8, 12])
        for rank in range(2):
            with self.subTest(rank=rank):
                runtime = TextParallelBatch(
                    mesh_context=SimpleNamespace(sequence_parallel=False), device="cpu", tokenizer=None,
                    data_config={"labels_are_shifted": True, "create_attention_mask_in_dataloader": True},
                    pp_shared_data=False, source_type="packed_text", attention_mode="compressed",
                    dataloader_context=DataLoaderParallelContext(
                        tp_source_rank=None, cp_rank=rank, cp_world_size=2,
                    ),
                    runtime_input_adapter=DeepseekV41Runtime(
                        include_position_ids=False, include_image_sequence_start=False,
                        pad_token_id=2,
                    ),
                )
                with patch("hyper_parallel.data.parallel.batch_parallel.dist.broadcast") as broadcast:
                    model, loss = runtime(iter([batch]))
                    broadcast.assert_not_called()
                local = slice(rank * 6, (rank + 1) * 6)
                torch.testing.assert_close(model["input_ids"][0], source["input_ids"][local])
                torch.testing.assert_close(model["shift_labels"][0], source["labels"][local])
                torch.testing.assert_close(model["router_token_mask"][0], source["input_ids"][local] != 2)
                torch.testing.assert_close(loss["loss_mask"][0], (source["labels"][local] >= 0).long())
                self.assertIsNone(model.get("attention_mask"))
                self.assertNotIn("image_sequence_start", model)
                self.assertEqual(model["position_ids"].tolist(), [[0, 1, 2, 0, 1, 2]] if rank == 0
                                 else [[3, 4, 0, 1, 2, 3]])
                packed = model["packed_seq_params"].prepare(torch.device("cpu"), (0, 2))
                self.assertEqual(packed.document_boundaries(), (0, 3, 8, 12))
                self.assertEqual((packed.local_query_start, packed.local_query_length,
                                  packed.global_sequence_length), (rank * 6, 6, 12))
                self.assertEqual(packed.segment_start_mask(torch.device("cpu")).reshape(-1).tolist(),
                                 [True, False, False, True, False, False] if rank == 0
                                 else [False, False, True, False, False, False])
