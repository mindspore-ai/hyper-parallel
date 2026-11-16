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
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.data.batching.get_batch import OmniParallelBatch, TextParallelBatch
from hyper_parallel.data.parallel import (
    DataLoaderParallelContext, TPBatchBroadcaster, build_dataset_for_dataloader,
    create_dataloader_parallel_context,
)
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token


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
                        expected_positions = [[0, 1, 2, 0]] if cp_rank == 0 else [[1, 2, 3, 4]]
                        self.assertEqual(model["position_ids"].tolist(), expected_positions)
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

    def test_eod_mask_requires_eod_and_accepts_zero(self) -> None:
        """Feature: EOD masking configuration validation.
        Description: Enable the option via either entry point with absent or zero EOD IDs.
        Expectation: Missing IDs fail at construction; zero remains a valid EOD token.
        """
        for from_config in (False, True):
            for tokenizer in (None, SimpleNamespace(), SimpleNamespace(eod=None), SimpleNamespace(eod=0)):
                with self.subTest(from_config=from_config, tokenizer=tokenizer):
                    options = {
                        "mesh_context": SimpleNamespace(sequence_parallel=False), "device": "cpu",
                        "tokenizer": tokenizer, "data_config": {"eod_mask_loss": from_config},
                        "pp_shared_data": False, "source_type": "indexed", "eod_mask_loss": not from_config,
                        "dataloader_context": DataLoaderParallelContext(),
                    }
                    if getattr(tokenizer, "eod", None) is None:
                        with self.assertRaisesRegex(ValueError, "eod_mask_loss=True requires tokenizer.eod"):
                            TextParallelBatch(**options)
                        continue
                    runtime = TextParallelBatch(**options)
                    model, loss = runtime(iter([{
                        "input_ids": torch.tensor([[3, 0, 2, 1]]), "labels": torch.tensor([[0, 2, 1, -100]]),
                    }]))
                    torch.testing.assert_close(model["shift_labels"], torch.tensor([[0, -100, 1, -100]]))
                    self.assertEqual(count_loss_token(loss)["foundation_tokens"].item(), 2)

    def test_text_eod_mask_prepares_targets_without_default_copy(self) -> None:
        """Feature: Text supervision at the batch boundary.
        Description: Exercise EOD off, partial and empty masks with both label alignments.
        Expectation: CE labels and counts agree, default labels are reused and input is unchanged.
        """
        cases = (
            (False, [3, 2, 2, 1], [1, -100, 2, 3], [1, -100, 2, 3]),
            (True, [3, 2, 2, 1], [1, -100, -100, 3], [1, -100, -100, -100]),
            (True, [2] * 4, [-100] * 4, [1, -100, -100, -100]),
        )
        for aligned in (False, True):
            for eod_enabled, token_values, aligned_targets, raw_targets in cases:
                with self.subTest(aligned=aligned, eod=eod_enabled, tokens=token_values):
                    labels = torch.tensor([[1, -100, 2, 3]])
                    batch = {"input_ids": torch.tensor([token_values]), "labels": labels}
                    runtime = TextParallelBatch(
                        mesh_context=SimpleNamespace(sequence_parallel=False), device="cpu",
                        tokenizer=SimpleNamespace(eod=2), data_config={"labels_are_shifted": aligned},
                        pp_shared_data=False, source_type="indexed", eod_mask_loss=eod_enabled,
                        dataloader_context=DataLoaderParallelContext(),
                    )
                    model, loss = runtime(iter([batch]))
                    expected = torch.tensor([aligned_targets if aligned else raw_targets])
                    if not eod_enabled:
                        self.assertIs(model["labels"], labels)
                    torch.testing.assert_close(model["labels"], expected)
                    self.assertIs(loss["labels"], model["labels"])
                    if aligned:
                        self.assertIs(model["shift_labels"], model["labels"])
                        self.assertIs(loss["shift_labels"], model["labels"])
                    effective = expected if aligned else expected[..., 1:]
                    self.assertEqual(
                        count_loss_token(loss)["foundation_tokens"].item(), effective.ne(-100).sum().item(),
                    )
                    torch.testing.assert_close(labels, torch.tensor([[1, -100, 2, 3]]))

    def test_eod_mask_preserves_each_label_convention(self) -> None:
        """Feature: EOD exclusions use prediction positions.
        Description: Vary EOD location, ignored targets and CP-local boundaries.
        Expectation: Each convention masks EOD predictions while preserving its CP-boundary targets.
        """
        cases = (
            ([2, 10, 20, 30], 1, False),
            ([10, 2, 20, 30], 1, False),
            ([10, 20, 30, 2], 1, False),
            ([10, 2, 2, 30], 1, False),
            ([2, 2, 2, 2], 1, False),
            ([2], 1, False),
            ([10, 2, 20, 30], 1, True),
            # EOD at the first shard's end also excludes its cross-shard prediction.
            ([10, 2, 20, 2, 2, 30, 2, 40], 2, False),
            # Only aligned labels retain this non-EOD cross-shard prediction.
            ([10, 2, 20, 30, 40, 50, 2, 60], 2, False),
        )
        for token_values, cp_size, ignore_last in cases:
            input_ids = torch.tensor([token_values])
            raw_labels = input_ids.clone()
            if ignore_last:
                raw_labels[..., -1] = -100
            for cp_rank in range(cp_size):
                for aligned in (False, True):
                    with self.subTest(
                        tokens=token_values, cp_rank=cp_rank, aligned=aligned, ignore_last=ignore_last,
                    ):
                        labels = raw_labels.clone()
                        if aligned:
                            labels = torch.nn.functional.pad(labels[..., 1:], (0, 1), value=-100)
                        original_labels = labels.clone()
                        runtime = TextParallelBatch(
                            mesh_context=SimpleNamespace(sequence_parallel=False), device="cpu",
                            tokenizer=SimpleNamespace(eod=2), data_config={"labels_are_shifted": aligned},
                            pp_shared_data=False, source_type="indexed", eod_mask_loss=True,
                            dataloader_context=DataLoaderParallelContext(cp_rank=cp_rank, cp_world_size=cp_size),
                        )
                        model, loss = runtime(iter([{"input_ids": input_ids, "labels": labels}]))
                        chunk = input_ids.shape[-1] // cp_size
                        local_slice = slice(cp_rank * chunk, (cp_rank + 1) * chunk)
                        local_labels = original_labels[:, local_slice]
                        # Build the reference in prediction coordinates, independently of the returned mask.
                        targets = local_labels if aligned else torch.nn.functional.pad(
                            local_labels[..., 1:], (0, 1), value=-100,
                        )
                        valid = input_ids[:, local_slice].ne(2) & targets.ne(-100)
                        expected_targets = targets.masked_fill(~valid, -100)
                        actual_targets = model.get("shift_labels")
                        if actual_targets is None:
                            actual_targets = torch.nn.functional.pad(
                                model["labels"][..., 1:], (0, 1), value=-100,
                            )
                        torch.testing.assert_close(actual_targets, expected_targets)
                        counts = count_loss_token(loss)["foundation_tokens"]
                        torch.testing.assert_close(counts, valid.sum())
                        torch.testing.assert_close(labels, original_labels)
                        torch.testing.assert_close(input_ids, torch.tensor([token_values]))

    def test_text_source_mask_survives_cp_and_causal_alignment(self) -> None:
        """Feature: Text source supervision before parallel transport.
        Description: Combine source masks, both label alignments, EOD masking and CP slices.
        Expectation: Targets and counts match the source exclusions without mutating the source.
        """
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 2, 8]])
        mask = torch.tensor([[1, 1, 0, 1, 0, 1, -0.5, 0]])
        cases = product((False, True), (False, True), (0, 1), (torch.int64, torch.int32, torch.uint8))
        for aligned, eod_enabled, cp_rank, dtype in cases:
            with self.subTest(aligned=aligned, eod=eod_enabled, cp_rank=cp_rank, dtype=dtype):
                labels = (torch.tensor([[2, 3, 4, 5, 6, 2, 8, 9]]) if aligned else input_ids).to(dtype)
                source = {"input_ids": input_ids, "labels": labels, "loss_mask": mask}
                originals = {key: value.clone() for key, value in source.items()}
                runtime = TextParallelBatch(
                    mesh_context=SimpleNamespace(sequence_parallel=False), device="cpu",
                    tokenizer=SimpleNamespace(eod=2), data_config={"labels_are_shifted": aligned},
                    pp_shared_data=False, source_type="indexed", eod_mask_loss=eod_enabled,
                    dataloader_context=DataLoaderParallelContext(cp_rank=cp_rank, cp_world_size=2),
                )
                model, loss = runtime(iter([source]))
                local_slice = slice(cp_rank * 4, (cp_rank + 1) * 4)
                targets = labels[:, local_slice].long()
                valid = mask[:, local_slice].ne(0)
                if not aligned:
                    targets = torch.nn.functional.pad(targets[..., 1:], (0, 1), value=-100)
                    valid = torch.nn.functional.pad(valid[..., 1:], (0, 1), value=False)
                if eod_enabled:
                    valid = valid & input_ids[:, local_slice].ne(2)
                expected = targets.masked_fill(~valid, -100)
                actual = model.get("shift_labels")
                if actual is None:
                    actual = torch.nn.functional.pad(model["labels"][..., 1:], (0, 1), value=-100)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(count_loss_token(loss)["foundation_tokens"], valid.sum())
                self.assertIs(loss["labels"], model["labels"])
                for key, value in source.items():
                    torch.testing.assert_close(value, originals[key])

    def test_text_invalid_source_mask_is_rejected(self) -> None:
        """Feature: Text supervision contract validation.
        Description: Provide a mask with a mismatched shape or a non-tensor value.
        Expectation: Invalid masks fail at the source boundary before parallel transport.
        """
        runtime = TextParallelBatch(
            mesh_context=SimpleNamespace(sequence_parallel=False), device="cpu", tokenizer=None,
            data_config={}, pp_shared_data=False, source_type="indexed",
            dataloader_context=DataLoaderParallelContext(),
        )
        for mask in (torch.ones(3), [1, 0, 1]):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "same shape"):
                runtime(iter([{"input_ids": torch.tensor([[1, 2, 3]]),
                               "labels": torch.tensor([[2, 3, 4]]), "loss_mask": mask}]))

    def test_omni_explicit_mask_survives_cp_without_mutating_source(self) -> None:
        """Feature: Omni supervision before parallel transport.
        Description: Distribute a source with explicit partial masking to two CP slices.
        Expectation: Model targets exclude masked positions and source labels stay intact.
        """
        labels = torch.tensor([[1, 2, -100, 4, 5, 6, 7, -100]])
        mask = torch.tensor([[1, 0, 1, 1, 1, 1, 0, 0]], dtype=torch.float32)
        source = {"input_ids": torch.arange(8).reshape(1, 8), "labels": labels, "loss_mask": mask}
        for cp_rank in (0, 1):
            with self.subTest(cp_rank=cp_rank):
                context = DataLoaderParallelContext(
                    tp_source_rank=None, tp_rank=1, tp_world_size=2,
                    cp_rank=cp_rank, cp_world_size=2, distributed_enabled=True,
                )
                runtime = OmniParallelBatch(
                    mesh_context=SimpleNamespace(sequence_parallel=False), device="cpu", dataloader_context=context,
                )
                model, loss = runtime(iter([source]))
                expected = torch.tensor([[1, -100, -100, 4, 5, 6, -100, -100]])
                expected = expected[:, cp_rank * 4:(cp_rank + 1) * 4]
                torch.testing.assert_close(model["labels"], expected)
                self.assertIs(loss["labels"], model["labels"])
                self.assertEqual(
                    count_loss_token(loss)["foundation_tokens"].item(), expected[..., 1:].ne(-100).sum().item(),
                )
        torch.testing.assert_close(labels, torch.tensor([[1, 2, -100, 4, 5, 6, 7, -100]]))
        self.assertIs(source["loss_mask"], mask)

    def test_omni_default_mask_reuses_labels_and_invalid_mask_fails(self) -> None:
        """Feature: Optional Omni loss masks.
        Description: Omit a mask, exclude all targets, or provide an invalid mask.
        Expectation: Derived masks reuse labels; explicit masks validate shape before transport.
        """
        labels = torch.tensor([[1, 2, -100]])
        source = {"input_ids": torch.tensor([[0, 1, 2]]), "labels": labels}
        prepared = OmniParallelBatch._normalize_source_batch(source)
        self.assertIs(prepared["labels"], labels)
        self.assertNotIn("loss_mask", source)
        prepared = OmniParallelBatch._normalize_source_batch({**source, "loss_mask": torch.zeros_like(labels)})
        torch.testing.assert_close(prepared["labels"], torch.full_like(labels, -100))
        for invalid_mask in (torch.ones(3), [1, 0, 1]):
            with self.subTest(mask=invalid_mask), self.assertRaisesRegex(ValueError, "same shape"):
                OmniParallelBatch._normalize_source_batch({**source, "loss_mask": invalid_mask})
