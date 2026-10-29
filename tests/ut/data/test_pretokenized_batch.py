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
"""End-to-end prepared text batching, attention and TP/CP metadata tests."""

import unittest
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from hyper_parallel.data.batching import TextParallelBatch, build_online_text_collate_fn
from hyper_parallel.data.batching.build_collate_fn import TextPackingCollator
from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter, RuntimeInputContext
from hyper_parallel.data.online.mapping import MappingTransformDataset
from hyper_parallel.data.parallel import DataLoaderParallelContext, CPBatchSharder, TPBatchBroadcaster
from hyper_parallel.data.text import PreTokenizedTransform


class _BoundaryAdapter(RuntimeInputAdapter):
    """Model-side stand-in receiving JT-compatible global boundary metadata."""

    def build_runtime_inputs(
        self, *, batch: Mapping[str, Any], context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Expose canonical boundaries without changing supervision."""
        del context
        return {"model_boundaries": batch["cu_seq_lens"]}


def _records():
    return [
        {"input_ids": [11, 12, 13, 14, 21, 22, 23, 24],
         "labels": [-100, -100, 14, 2, -100, -100, 24, 2], "boundaries": [0, 4, 8]},
        {"input_ids": [31, 32, 33, 34], "labels": [-100, -100, 34, 2], "boundaries": [0, 4]},
    ]


def _runtime(context=None, **kwargs):
    context = context or DataLoaderParallelContext()
    with patch("hyper_parallel.data.batching.get_batch.create_dataloader_parallel_context", return_value=context):
        return TextParallelBatch(
            mesh_context=SimpleNamespace(), device="cpu", tokenizer=None,
            data_config=kwargs.pop("data_config", {}), pp_shared_data=False,
            source_type=kwargs.pop("source_type", "pretokenized"), **kwargs,
        )


class TestPreTokenizedBatch(unittest.TestCase):
    """Exercise the owner's worked example through the actual batch runtime."""

    def test_owner_example_dataloader_masks_positions_targets(self):
        """Three documents plus padding have six targets and isolated attention."""
        dataset = MappingTransformDataset(_records(), PreTokenizedTransform())
        loader = DataLoader(dataset, batch_size=2, collate_fn=TextPackingCollator(16))
        batch = next(iter(loader))
        self.assertEqual(batch["cu_seq_lens"].tolist(), [0, 4, 8, 12, 16])
        model, loss = _runtime()(iter([batch]))
        self.assertEqual(model["position_ids"].tolist(), [[0, 1, 2, 3] * 4])
        self.assertEqual(int(loss["loss_mask"].sum()), 6)
        torch.testing.assert_close(loss["labels"], batch["labels"])
        torch.testing.assert_close(model["shift_labels"], batch["labels"])
        mask = model["attention_mask"][0, 0]
        self.assertTrue(mask[6, 4])
        self.assertFalse(mask[6, 3])
        self.assertFalse(mask[6, 7])
        self.assertFalse(mask[:12, 12:].any())

    def test_online_builder_preserves_internal_boundaries(self):
        """The existing online builder retains internal boundaries and zero padding."""
        samples = [PreTokenizedTransform()(record)[0] for record in _records()]
        batch = build_online_text_collate_fn(SimpleNamespace(cp_size=16))(samples)
        self.assertEqual(tuple(batch["input_ids"].shape), (1, 16))
        self.assertEqual(batch["cu_seq_lens"].tolist(), [0, 4, 8, 12, 16])
        self.assertEqual(batch["input_ids"][0, 12:].tolist(), [0] * 4)
        model, loss = _runtime()(iter([batch]))
        self.assertEqual(model["position_ids"].tolist(), [[0, 1, 2, 3] * 4])
        self.assertEqual(int(loss["loss_mask"].sum()), 6)
        self.assertEqual(loss["labels"][0, 12:].tolist(), [-100] * 4)
        self.assertFalse(model["attention_mask"][0, 0, :12, 12:].any())

    def test_single_document_records_keep_original_packing(self):
        """Ordinary records keep the original tensors with or without metadata."""
        for with_boundaries in (False, True):
            for alignment in (1, 8):
                with self.subTest(with_boundaries=with_boundaries, alignment=alignment):
                    samples = [
                        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 3, 4])},
                        {"input_ids": torch.tensor([5, 6]), "labels": torch.tensor([6, 7])},
                    ]
                    if with_boundaries:
                        for sample in samples:
                            sample["cu_seq_lens"] = torch.tensor([0, sample["input_ids"].numel()])
                    batch = TextPackingCollator(alignment)(samples)
                    tail = 3 if alignment == 8 else 0
                    self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3, 5, 6] + [0] * tail])
                    self.assertEqual(batch["labels"].tolist(), [[-100, 3, 4, 6, 7] + [-100] * tail])
                    self.assertEqual(batch["cu_seq_lens"].tolist(), [0, 3, 5] + ([8] if tail else []))

    def test_cp_shards_preserve_document_positions_and_masked_targets(self):
        """A CP cut inside a document must not restart its positions."""
        record = {"input_ids": list(range(1, 13)), "labels": list(range(2, 14)), "boundaries": [0, 5, 12]}
        batch = TextPackingCollator(16)(PreTokenizedTransform()(record))
        context = DataLoaderParallelContext(cp_rank=1, cp_world_size=2)
        model, loss = _runtime(context)(iter([batch]))
        self.assertEqual(model["position_ids"].tolist(), [[3, 4, 5, 6, 0, 1, 2, 3]])
        self.assertEqual(loss["loss_mask"].tolist(), [[1, 1, 1, 1, 0, 0, 0, 0]])
        sharded = CPBatchSharder(context).shard(batch)
        self.assertEqual(set(sharded), {"input_ids", "labels"})
        self.assertEqual(sharded["labels"].tolist(), [[10, 11, 12, 13, -100, -100, -100, -100]])

    def test_tp_receiver_gets_global_boundaries(self):
        """Replay tensor broadcasts to check both source and receiver protocols."""
        batch = TextPackingCollator(16)([PreTokenizedTransform()(record)[0] for record in _records()])
        context = DataLoaderParallelContext(tp_world_size=2, cp_world_size=2, cp_rank=1)
        local = CPBatchSharder(context).shard(batch)
        payloads = []
        with patch("hyper_parallel.data.parallel.batch_parallel.dist.broadcast",
                   side_effect=lambda tensor, **_: payloads.append(tensor.clone())):
            source = TPBatchBroadcaster(context, "cpu").broadcast(local, batch["cu_seq_lens"])
        self.assertEqual(payloads[0].numel(), 3)
        received = iter(payloads)
        with patch("hyper_parallel.data.parallel.batch_parallel.dist.broadcast",
                   side_effect=lambda tensor, **_: tensor.copy_(next(received))):
            peer = TPBatchBroadcaster(replace(context, tp_rank=1), "cpu").broadcast(None)
        self.assertEqual(source.keys(), peer.keys())
        for key in source:
            torch.testing.assert_close(source[key], peer[key])
        self.assertEqual(peer["cu_seq_lens"].tolist(), [0, 4, 8, 12, 16])

    def test_legacy_batch_without_boundaries_still_broadcasts(self):
        """Existing indexed/online callers retain their metadata contract."""
        batch = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[2, 3]])}
        context = DataLoaderParallelContext(tp_world_size=2)
        payloads = []
        with patch("hyper_parallel.data.parallel.batch_parallel.dist.broadcast",
                   side_effect=lambda tensor, **_: payloads.append(tensor.clone())):
            source = TPBatchBroadcaster(context, "cpu").broadcast(batch)
        self.assertEqual(payloads[0].numel(), 3)
        received = iter(payloads)
        with patch("hyper_parallel.data.parallel.batch_parallel.dist.broadcast",
                   side_effect=lambda tensor, **_: tensor.copy_(next(received))):
            peer = TPBatchBroadcaster(replace(context, tp_rank=1), "cpu").broadcast(None)
        self.assertIsNone(peer["cu_seq_lens"])
        torch.testing.assert_close(source["labels"], peer["labels"])

    def test_jt_compressed_adapter_receives_metadata(self):
        """Boundary-aware model adapters keep global metadata on a CP-local batch."""
        batch = TextPackingCollator(16)([PreTokenizedTransform()(record)[0] for record in _records()])
        context = DataLoaderParallelContext(cp_world_size=2, cp_rank=1)
        runtime = _runtime(context, attention_mode="compressed", runtime_input_adapter=_BoundaryAdapter())
        model, loss = runtime(iter([batch]))
        self.assertEqual(model["model_boundaries"].tolist(), [0, 4, 8, 12, 16])
        self.assertIsNone(model["attention_mask"])
        self.assertEqual(int(loss["loss_mask"].sum()), 2)

    def test_refuse_semantic_changes_or_missing_attention(self):
        """Reject double shifting, EOD loss filtering and unconsumed boundaries."""
        for options in ({"eod_mask_loss": True}, {"data_config": {"labels_are_shifted": False}},
                        {"attention_mode": "compressed"},
                        {"data_config": {"create_attention_mask_in_dataloader": False}},
                        {"causal": False}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                _runtime(**options)

    def test_ignored_prompt_keeps_positions_and_labels_control_supervision(self):
        """Ignored prompt targets do not make their input tokens into padding."""
        record = {"input_ids": [0, 2, 3], "labels": [-100, 3, 2]}
        batch = TextPackingCollator(8)(PreTokenizedTransform()(record))
        self.assertEqual(set(batch), {"input_ids", "labels", "cu_seq_lens"})
        model, loss = _runtime()(iter([batch]))
        self.assertEqual(model["position_ids"].tolist(), [[0, 1, 2, 0, 1, 2, 3, 4]])
        self.assertEqual(loss["loss_mask"].tolist(), [[0, 1, 1, 0, 0, 0, 0, 0]])
        self.assertFalse(model["attention_mask"][0, 0, :3, 3:].any())

    def test_malformed_batch_boundaries_or_labels_rejected(self):
        """Invalid cumulative boundaries or mismatched labels fail before distribution."""
        sample = PreTokenizedTransform()({"input_ids": [1, 2, 3], "labels": [2, 3, 4]})
        for key, value in (("cu_seq_lens", torch.tensor([0, 3], dtype=torch.int32)),
                           ("cu_seq_lens", torch.tensor([0, 3, 3, 4], dtype=torch.int32)),
                           ("cu_seq_lens", torch.tensor([1, 3, 4], dtype=torch.int32)),
                           ("labels", torch.tensor([[2, 3]]))):
            batch = TextPackingCollator(4)(sample)
            batch[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                _runtime()(iter([batch]))

    def test_packed_and_separate_documents_have_same_loss_and_gradients(self):
        """Packed attention matches independent documents, including gradients."""
        samples = [PreTokenizedTransform()(record)[0] for record in _records()]
        documents = []
        for sample in samples:
            boundaries = sample["cu_seq_lens"].tolist()
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                documents.append({field: sample[field][start:end] for field in ("input_ids", "labels")})
        target_count = sum(int((sample["labels"] != -100).sum()) for sample in samples)
        torch.manual_seed(42)
        initial = torch.randn(40, 8, dtype=torch.float64)
        outcomes = []
        for groups in ([samples], [[document] for document in documents]):
            weight = initial.clone().requires_grad_()
            total_loss = 0
            for group in groups:
                batch = TextPackingCollator(8)(group)
                model, loss_inputs = _runtime()(iter([batch]))
                hidden = functional.embedding(model["input_ids"], weight).unsqueeze(1)
                scores = (hidden @ hidden.transpose(-1, -2)) / hidden.shape[-1] ** 0.5
                scores = scores.masked_fill(~model["attention_mask"], float("-inf"))
                attended = (scores.softmax(dim=-1) @ hidden).squeeze(1)
                logits = attended @ weight.T
                total_loss = total_loss + functional.cross_entropy(
                    logits.flatten(0, 1), loss_inputs["labels"].flatten(), reduction="sum",
                ) / target_count
            total_loss.backward()
            outcomes.append((total_loss.detach(), weight.grad))
        torch.testing.assert_close(outcomes[0][0], outcomes[1][0], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(outcomes[0][1], outcomes[1][1], atol=1e-12, rtol=1e-12)
