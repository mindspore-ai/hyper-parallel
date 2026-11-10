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
"""Native packed-record decomposition, TND semantics and committed-cursor replay."""

import copy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from datasets import Dataset, load_from_disk
import torch
from torch.nn import functional

from hyper_parallel.data.batching import TextParallelBatch
from hyper_parallel.data.batching.build_dataloader import build_dataloader
from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig, SampleKey, SampleMetadata, WorkloadCost, build_distributed_dataloader,
    build_local_balancing_dataloader, collate_indexed_text_sequences,
    pack_indexed_text_samples, split_packed_text_sample,
)
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner


_MESH = SimpleNamespace(mesh_shape=(1,), mesh_dim_names=("dp",), rank_list=(0,))


def _record(lengths: list[int], offset: int = 1) -> dict:
    count = sum(lengths)
    labels = torch.arange(offset + 1, offset + count + 1)
    labels[::3] = -100
    return {
        "input_ids": torch.arange(offset, offset + count),
        "labels": labels,
        "cu_seq_lens": torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()]),
    }


def _metadata(sample: dict) -> SampleMetadata:
    length = sample["input_ids"].numel()
    return SampleMetadata(length, cost=WorkloadCost(llm=length ** 2), features={"P": length, "D": 0})


def _loader(records: list[dict], **kwargs: object) -> object:
    return build_distributed_dataloader(
        records, _MESH, DistributedDatasetConfig(seq_len=12, local_batch_size=1),
        batch_sampler=build_dataset_batch_sampler(
            total_samples=len(records), micro_batch_size=1, global_batch_size=1, dp_rank=0, dp_world_size=1,
            index_mapping=[0] * len(records),
        ),
        metadata_fn=_metadata, split_fn=split_packed_text_sample, pack_fn=pack_indexed_text_samples,
        collate_fn=collate_indexed_text_sequences, device="cpu", cost_model=lambda metadata: metadata.cost,
        **kwargs,
    )


def _attention_loss(records: list[dict], weights: torch.Tensor) -> torch.Tensor:
    """Compute a document-isolated causal attention loss over non-ignored targets."""
    losses = []
    for record in records:
        ids = record["input_ids"].reshape(-1)
        labels = record["labels"].reshape(-1)
        offsets = record["cu_seq_lens"].tolist()
        positions = torch.arange(ids.numel())
        documents = torch.bucketize(positions, torch.tensor(offsets[1:]), right=True)
        allowed = (documents[:, None] == documents[None, :]) & (positions[:, None] >= positions[None, :])
        hidden = weights[ids]
        scores = (hidden @ hidden.T) / hidden.shape[-1] ** 0.5
        output = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1) @ hidden
        token_losses = functional.cross_entropy(output @ weights.T, labels, ignore_index=-100, reduction="none")
        losses.append(token_losses[labels >= 0])
    return torch.cat(losses).mean()


class TestSequenceRepacking(unittest.TestCase):
    """Exercise the real reader, planner, constructors and checkpoint entry points."""

    def test_config_repacking_applies_padding_cost_and_logging(self) -> None:
        """Feature: Configuration-only text repacking.
        Description: Read padded and short records without supplying packing callbacks or a cost model.
        Expectation: Config controls trimming, new PAD, boundaries, square cost and emitted reports.
        """
        padded = _record([4, 8])
        padded["input_ids"][-3:] = 9
        padded["labels"][-3:] = -100
        cases = (
            (padded, None, [0, 4, 12], 80, 9),
            (padded, 9, [0, 4, 9, 12], 50, 9),
            (_record([4, 5]), None, [0, 4, 9, 12], 50, 0),
        )
        logger = "hyper_parallel.distributed_data.balance_logging"
        for record, pad_id, boundaries, cost, fill_id in cases:
            for log_balance in (False, True):
                with self.subTest(pad_id=pad_id, boundaries=boundaries, log_balance=log_balance):
                    config = DistributedDatasetConfig(
                        seq_len=12, local_batch_size=1, repack_sequences=True,
                        pad_token_id=pad_id, log_balance=log_balance,
                    )
                    sampler = build_dataset_batch_sampler(
                        total_samples=1, micro_batch_size=1, global_batch_size=1, dp_rank=0, dp_world_size=1,
                    )
                    with build_distributed_dataloader(
                        [record], _MESH, config, batch_sampler=sampler, metadata_fn=_metadata, device="cpu",
                    ) as loader:
                        capture = self.assertLogs(logger, level="INFO") if log_balance else self.assertNoLogs(logger)
                        with capture as logs:
                            batch = next(loader)
                        if log_balance:
                            self.assertIn(f"original={cost}, candidate={cost}", logs.output[0])
                    self.assertEqual(batch["cu_seq_lens"].tolist(), boundaries)
                    self.assertEqual(batch["input_ids"][0, -3:].tolist(), [fill_id] * 3)
                    torch.testing.assert_close(batch["input_ids"][0, :9], record["input_ids"][:9])
                    torch.testing.assert_close(batch["labels"][0, :9], record["labels"][:9])
                    self.assertEqual(batch["labels"][0, -3:].tolist(), [-100] * 3)

    def test_pad_id_alone_preserves_complete_records(self) -> None:
        """Feature: Disabled built-in repacking.
        Description: Set a PAD ID without enabling repacking or logging.
        Expectation: The complete record and its boundaries remain unchanged and no report is emitted.
        """
        record = _record([4, 8])
        record["input_ids"][-3:] = 9
        record["labels"][-3:] = -100
        sampler = build_dataset_batch_sampler(
            total_samples=1, micro_batch_size=1, global_batch_size=1, dp_rank=0, dp_world_size=1,
        )
        with build_distributed_dataloader(
            [record], _MESH, DistributedDatasetConfig(seq_len=12, local_batch_size=1, pad_token_id=9),
            batch_sampler=sampler, metadata_fn=_metadata, device="cpu", cost_model=lambda item: item.cost,
        ) as loader, self.assertNoLogs("hyper_parallel.distributed_data.balance_logging"):
            batch = next(loader)
        for field, expected in record.items():
            torch.testing.assert_close(batch[0][field], expected, rtol=0, atol=0)

    def test_config_repacking_preserves_custom_cost_and_collator(self) -> None:
        """Feature: Explicit policy precedence with built-in text packing.
        Description: Supply a custom cost model and collator alongside the repacking switch.
        Expectation: The requested policy and collation are used while built-in packing rebuilds boundaries.
        """
        sampler = build_dataset_batch_sampler(
            total_samples=1, micro_batch_size=1, global_batch_size=1, dp_rank=0, dp_world_size=1,
        )
        with build_distributed_dataloader(
            [_record([4, 8])], _MESH,
            DistributedDatasetConfig(seq_len=12, local_batch_size=1, repack_sequences=True, log_balance=True),
            batch_sampler=sampler, metadata_fn=_metadata, device="cpu", collate_fn=tuple,
            cost_model=lambda _: WorkloadCost(llm=7),
        ) as loader, self.assertLogs("hyper_parallel.distributed_data.balance_logging", level="INFO") as logs:
            batch = next(loader)
        self.assertIsInstance(batch, tuple)
        self.assertEqual(batch[0]["cu_seq_lens"].tolist(), [0, 4, 12])
        self.assertIn("original=14, candidate=14", logs.output[0])

    def test_config_repacking_rejects_custom_pack_and_external_sources(self) -> None:
        """Feature: Unambiguous repacking contract.
        Description: Combine the switch with custom packing callbacks or unsupported sources.
        Expectation: Builders reject the combinations before loading or creating distributed groups.
        """
        config = DistributedDatasetConfig(seq_len=12, local_batch_size=1, repack_sequences=True)
        sampler = build_dataset_batch_sampler(
            total_samples=1, micro_batch_size=1, global_batch_size=1, dp_rank=0, dp_world_size=1,
        )
        for callback in ({"split_fn": split_packed_text_sample}, {"pack_fn": pack_indexed_text_samples}):
            with self.subTest(callback=tuple(callback)), self.assertRaisesRegex(ValueError, "cannot be combined"):
                build_distributed_dataloader(
                    [_record([4, 8])], _MESH, config, batch_sampler=sampler, metadata_fn=_metadata,
                    device="cpu", **callback,
                )
        for source in ({}, {"external_step_source": []}):
            with self.subTest(source=tuple(source)), self.assertRaisesRegex(
                ValueError, "requires a native batch_sampler",
            ):
                build_distributed_dataloader(None, _MESH, config, metadata_fn=_metadata, device="cpu", **source)
        with self.assertRaisesRegex(ValueError, "requires a native batch_sampler"):
            build_local_balancing_dataloader(
                [], _MESH, config, metadata_fn=_metadata, pack_fn=pack_indexed_text_samples, device="cpu",
            )

    def test_trim_only_configured_ignored_tail_padding(self) -> None:
        """Feature: Explicit source PAD trimming.
        Description: Split records containing ignored prompts, internal PAD IDs and trailing PAD.
        Expectation: Only the configured ignored tail is removed; invalid inputs are rejected.
        """
        record = {
            "input_ids": [1, 9, 2, 3, 9, 9],
            "labels": [-100, -100, 3, -100, -100, -100],
            "cu_seq_lens": [0, 3, 6],
        }
        documents = split_packed_text_sample(record, pad_token_id=9)
        self.assertEqual([doc["input_ids"].tolist() for doc in documents], [[1, 9, 2], [3]])
        self.assertEqual([doc["labels"].tolist() for doc in documents], [[-100, -100, 3], [-100]])
        packed = pack_indexed_text_samples(documents, seq_len=6, pad_token_id=9)
        self.assertEqual(packed["cu_seq_lens"].tolist(), [0, 3, 4, 6])
        self.assertEqual(packed["input_ids"].tolist(), record["input_ids"])
        self.assertEqual(packed["labels"].tolist(), record["labels"])
        self.assertEqual([doc["input_ids"].numel() for doc in split_packed_text_sample(record)], [3, 3])

        supervised_tail = {**record, "labels": [-100, -100, 3, -100, -100, 4]}
        self.assertEqual([doc["input_ids"].numel() for doc in
                          split_packed_text_sample(supervised_tail, pad_token_id=9)], [3, 3])
        separate_padding = {**record, "cu_seq_lens": [0, 3, 4, 5, 6]}
        self.assertEqual([doc["input_ids"].numel() for doc in
                          split_packed_text_sample(separate_padding, pad_token_id=9)], [3, 1])
        with self.assertRaisesRegex(ValueError, "non-padding"):
            split_packed_text_sample(
                {"input_ids": [9, 9], "labels": [-100, -100], "cu_seq_lens": [0, 2]}, pad_token_id=9,
            )
        for pad_token_id in (-1, True, 1.5):
            with self.subTest(pad_token_id=pad_token_id), self.assertRaisesRegex(ValueError, "pad_token_id"):
                split_packed_text_sample(record, pad_token_id=pad_token_id)

    def test_trim_and_repack_preserve_causal_loss_and_gradients(self) -> None:
        """Feature: Independent PAD sequence boundaries.
        Description: Move ignored trailing PAD into its own sequence in a causal model.
        Expectation: Real-token loss and gradients match the source layout.
        """
        record = _record([4, 8])
        record["input_ids"][-3:] = 0
        record["labels"][-3:] = -100
        repacked = pack_indexed_text_samples(split_packed_text_sample(record, pad_token_id=0), 12)
        self.assertEqual(repacked["cu_seq_lens"].tolist(), [0, 4, 9, 12])
        torch.manual_seed(7)
        reference_weights = torch.randn(32, 5, dtype=torch.float64, requires_grad=True)
        balanced_weights = reference_weights.detach().clone().requires_grad_()
        reference_loss = _attention_loss([record], reference_weights)
        balanced_loss = _attention_loss([repacked], balanced_weights)
        reference_loss.backward()
        balanced_loss.backward()
        torch.testing.assert_close(balanced_loss, reference_loss, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(balanced_weights.grad, reference_weights.grad, rtol=1e-12, atol=1e-12)

    def test_repeated_record_indices_keep_distinct_sequence_occurrences(self) -> None:
        """Feature: Repeated record sampling.
        Description: Select the same packed record twice in one native round.
        Expectation: Both occurrences retain their own sequence keys and payloads.
        """
        record = _record([4, 8])
        sampler = build_dataset_batch_sampler(
            total_samples=2, micro_batch_size=2, global_batch_size=2, dp_rank=0, dp_world_size=1,
            index_mapping=[0, 0],
        )
        with build_distributed_dataloader(
            [record], _MESH, DistributedDatasetConfig(seq_len=12, local_batch_size=2),
            batch_sampler=sampler, metadata_fn=_metadata, split_fn=split_packed_text_sample,
            pack_fn=pack_indexed_text_samples, collate_fn=collate_indexed_text_sequences,
            device="cpu", cost_model=lambda item: item.cost,
        ) as loader:
            reader = loader._planning_reader()
            reader.prepare_next_step(1)
            keys = [item.key for item in reader.metadata()]
            self.assertEqual({(key.dataset_index, key.global_sample_position, key.sequence_index) for key in keys},
                             {(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)})
            batch = next(loader)
            torch.testing.assert_close(batch["input_ids"], record["input_ids"].repeat(2, 1))
            torch.testing.assert_close(batch["labels"], record["labels"].repeat(2, 1))
            self.assertEqual(batch["cu_seq_lens"].tolist(), [0, 4, 12, 16, 24])
            self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 2)

    def test_repacking_preserves_masked_attention_loss_and_gradients(self) -> None:
        """Feature: Independent sequence repacking.
        Description: Redistribute unequal records and compare a causal attention model's gradients.
        Expectation: Rank costs improve while token targets, loss and gradients are conserved.
        """
        records = [_record([6, 6]), _record([2, 2, 2, 2], offset=13)]
        payloads, metadata, reference = {}, [], []
        for index, record in enumerate(records):
            keys = []
            for sequence_index, sample in enumerate(split_packed_text_sample(record)):
                key = SampleKey(index, index, index, sequence_index)
                item = _metadata(sample)
                metadata.append(SampleMetadata(item.pack_tokens, cost=item.cost, key=key))
                payloads[key] = sample
                keys.append(key)
            reference.append(keys)
        planner = DynamicPackingPlanner(
            data_parallel_size=2, local_batch_size=1, seq_len=12, cost_model=lambda item: item.cost,
        )
        plan = planner.plan(metadata, reference_bins=reference, step=0)
        self.assertEqual([cost.llm for cost in plan.rank_costs], [44, 44])
        repacked = [pack_indexed_text_samples([payloads[key] for key in plan.local_sample_keys(rank)], 12)
                    for rank in range(2)]
        torch.manual_seed(7)
        reference_weights = torch.randn(32, 5, dtype=torch.float64, requires_grad=True)
        balanced_weights = reference_weights.detach().clone().requires_grad_()
        reference_loss = _attention_loss(records, reference_weights)
        balanced_loss = _attention_loss(repacked, balanced_weights)
        reference_loss.backward()
        balanced_loss.backward()
        torch.testing.assert_close(balanced_loss, reference_loss, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(balanced_weights.grad, reference_weights.grad, rtol=1e-12, atol=1e-12)

    def test_no_gain_replays_original_grouping_and_committed_cursor(self) -> None:
        """Feature: Native-round checkpoint replay.
        Description: Prefetch repeated records and resume after one delivered batch.
        Expectation: Original grouping and targets survive; saved progress excludes the next round.
        """
        records = [_record([4, 8])] * 3
        with _loader(records) as loader:
            first = next(loader)
            loader.wait_for_prefetch()
            state = copy.deepcopy(loader.state_dict())
            self.assertEqual(state["_index_sampler_state"]["consumed_samples"], 1)
            self.assertEqual(first["cu_seq_lens"].tolist(), [0, 4, 12])
            torch.testing.assert_close(first["input_ids"][0], records[0]["input_ids"])
            torch.testing.assert_close(first["labels"][0], records[0]["labels"])
            expected = list(loader)
            loader.set_epoch(1)
            self.assertEqual(next(loader)["cu_seq_lens"].tolist(), [0, 4, 12])
        with _loader(records) as resumed:
            resumed.load_state_dict(state)
            actual = list(resumed)
        self.assertEqual(len(actual), 2)
        for left, right in zip(actual, expected):
            for field in left:
                torch.testing.assert_close(left[field], right[field], rtol=0, atol=0)

    def test_explicit_boundaries_and_online_mode_are_required(self) -> None:
        """Feature: Repacking input contract.
        Description: Supply invalid boundaries and incompatible callback combinations.
        Expectation: Invalid inputs fail directly before producing a training batch.
        """
        record = _record([4, 8])
        for boundaries in ([1, 12], [0, 4, 4, 12], [0, 13], [0., 12.]):
            with self.subTest(boundaries=boundaries), self.assertRaisesRegex(ValueError, "cu_seq_lens"):
                split_packed_text_sample({**record, "cu_seq_lens": boundaries})
        with self.assertRaises(KeyError):
            split_packed_text_sample({key: value for key, value in record.items() if key != "cu_seq_lens"})
        sampler = build_dataset_batch_sampler(
            total_samples=1, micro_batch_size=1, global_batch_size=1, dp_rank=0, dp_world_size=1,
        )
        with self.assertRaisesRegex(ValueError, "online metadata"):
            build_distributed_dataloader(
                [record], _MESH, DistributedDatasetConfig(seq_len=12, local_batch_size=1, metadata_mode=True),
                batch_sampler=sampler, metadata=[SampleMetadata(12)], split_fn=split_packed_text_sample,
                pack_fn=pack_indexed_text_samples, device="cpu",
            )
        with self.assertRaisesRegex(ValueError, "requires pack_fn"):
            build_distributed_dataloader(
                [record], _MESH, DistributedDatasetConfig(seq_len=12, local_batch_size=1),
                batch_sampler=sampler, metadata_fn=_metadata, split_fn=split_packed_text_sample, device="cpu",
            )
        with build_distributed_dataloader(
            [record], _MESH, DistributedDatasetConfig(seq_len=12, local_batch_size=1),
            batch_sampler=sampler, metadata_fn=_metadata, split_fn=lambda _: (),
            pack_fn=pack_indexed_text_samples, device="cpu", cost_model=lambda item: item.cost,
        ) as loader:
            with self.assertRaisesRegex(ValueError, "at least one independent sequence"):
                next(loader)

    def test_saved_arrow_through_trainer_builder_preserves_document_boundaries(self) -> None:
        """Feature: Offline packed Arrow integration.
        Description: Build the native loader and shared text adapter from stored packed records.
        Expectation: Dense attention isolates documents and preserves reset positions and pre-shifted labels.
        """
        record = _record([4, 8])
        record["input_ids"][-3:] = 0
        record["labels"][-3:] = -100
        with TemporaryDirectory() as directory:
            Dataset.from_dict({
                "input_ids": [record["input_ids"].tolist()], "labels": [record["labels"].tolist()],
                "cu_seqlens": [record["cu_seq_lens"].tolist()],
            }).save_to_disk(directory)
            config = {
                "repack_sequences": True, "pad_token_id": 0, "log_balance": True,
            }
            dataset = load_from_disk(directory).rename_column("cu_seqlens", "cu_seq_lens").with_format("torch")
            context = SimpleNamespace(
                dp_rank=0, dp_size=1, tp_size=1, cp_size=1, pp_size=1, device_mesh=_MESH,
            )
            target = SimpleNamespace(load_balance=config, num_workers=0)
            loaders, _ = build_dataloader(
                target, datasets=(dataset, None, None), collate_fn=tuple, mesh_context=context,
                training_config=SimpleNamespace(micro_batch_size=1, global_batch_size=1, seed=7),
                data_config={"seq_length": 12},
            )
            batch_context = SimpleNamespace(**{**vars(context), "device_mesh": None})
            batch_adapter = TextParallelBatch(
                batch_context, torch.device("cpu"), None, {"create_attention_mask_in_dataloader": True}, False,
                source_type="packed_text",
                attention_mode="dense", reset_position_ids=True,
            )
            with loaders[0] as loader:
                with self.assertLogs("hyper_parallel.distributed_data.balance_logging", level="INFO") as logs:
                    model, loss = batch_adapter(loader)
            self.assertIn("original=50, candidate=50", logs.output[0])
            self.assertIn("samples=2, cost=50, padding=3", logs.output[0])
            self.assertEqual(model["attention_mask"].shape[-2:], (12, 12))
            self.assertFalse(bool(model["attention_mask"][0, 0, 4, 0]))
            self.assertTrue(bool(model["attention_mask"][0, 0, 4, 4]))
            self.assertEqual(model["position_ids"].tolist(), [[0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2]])
            torch.testing.assert_close(loss["shift_labels"][0], record["labels"])

    def test_repack_respects_supplied_and_configured_cost_models(self) -> None:
        """Feature: Explicit cost-model precedence.
        Description: Build repacking loaders with configured and supplied cost models.
        Expectation: Explicit policies override the causal-square default in the documented order.
        """
        def configured_cost(sample: SampleMetadata) -> WorkloadCost:
            """Use the factory's document cost."""
            del sample
            return WorkloadCost(llm=7)

        def supplied_cost(sample: SampleMetadata) -> WorkloadCost:
            """Use the Python argument's document cost."""
            del sample
            return WorkloadCost(llm=3)

        context = SimpleNamespace(dp_rank=0, dp_size=1, tp_size=1, cp_size=1, pp_size=1, device_mesh=_MESH)
        target = SimpleNamespace(
            dataloader_type="single", num_workers=0,
            cost_model=SimpleNamespace(build=lambda **_: configured_cost),
        )
        target.load_balance = {"repack_sequences": True, "log_balance": True}
        for supplied, expected in ((None, 14), (supplied_cost, 6)):
            with self.subTest(supplied=supplied is not None):
                loaders, _ = build_dataloader(
                    target, datasets=([_record([4, 8])], None, None), collate_fn=tuple,
                    mesh_context=context,
                    training_config=SimpleNamespace(micro_batch_size=1, global_batch_size=1, seed=7),
                    data_config={"seq_length": 12}, cost_model=supplied,
                )
                with loaders[0] as loader:
                    with self.assertLogs("hyper_parallel.distributed_data.balance_logging", level="INFO") as logs:
                        next(loader)
                self.assertIn(f"original={expected}, candidate={expected}", logs.output[0])
