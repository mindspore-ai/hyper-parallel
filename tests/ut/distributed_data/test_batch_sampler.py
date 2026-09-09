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
"""Native HP sampling, occurrence conservation, and transactional replay."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hyper_parallel.data.batching.build_dataloader import DynamicBatchDataLoader, build_dataloader
from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import DistributedDatasetConfig, SampleMetadata, build_distributed_dataloader
from hyper_parallel.distributed_data.batch_sampler import BatchSamplerReader, native_sampler_fingerprint
from hyper_parallel.distributed_data.step_sample_selection import StepSampleSelector
from hyper_parallel.platform import get_platform

platform = get_platform()


class _StandaloneMesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


class _TrackedDataset:
    """Return simple whole samples and record all physical reads."""

    def __init__(self, size: int = 10) -> None:
        """Initialize the read log and logical Dataset size."""
        self.size = size
        self.reads = []

    def __len__(self) -> int:
        """Return the logical Dataset size."""
        return self.size

    def __getitem__(self, index: int) -> dict:
        """Record and return one native sample."""
        self.reads.append(index)
        return {"id": index, "tokens": 1, "label": index + 1, "position": 7}


def _metadata(sample: dict) -> SampleMetadata:
    return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])


def _sampler(**overrides: object) -> object:
    options = {"total_samples": 10, "micro_batch_size": 2, "global_batch_size": 4, "dp_world_size": 1, "dp_rank": 0}
    options.update(overrides)
    return build_dataset_batch_sampler(**options)


def _loader(dataset: object, *, sampler: object = None, sidecar: bool = False, **options: object) -> object:
    kwargs = {"metadata": [SampleMetadata(pack_tokens=1, sample_id=index) for index in range(len(dataset))]}
    if not sidecar:
        kwargs = {"metadata_fn": _metadata}
    return build_distributed_dataloader(
        dataset, _StandaloneMesh(), DistributedDatasetConfig(seq_len=16, local_batch_size=2, **options),
        batch_sampler=sampler if sampler is not None else _sampler(), **kwargs,
    )


def _batch_ids(batch: tuple) -> list[int]:
    return sorted(sample["id"] for sample in batch)


class TestNativeBatchSampler(unittest.TestCase):
    """Test without process groups or accelerator hardware."""

    def test_fixed_membership_without_dynamic_selection(self) -> None:
        """Short samples still occupy native batches instead of filling token budgets."""
        for sidecar in (False, True):
            with self.subTest(sidecar=sidecar):
                dataset = _TrackedDataset()
                loader = _loader(dataset, sidecar=sidecar, buffer_size_multiplier=100)
                with patch.object(StepSampleSelector, "select", side_effect=AssertionError("must not select")):
                    self.assertEqual(_batch_ids(next(loader)), [0, 1])
                    self.assertEqual(sorted(dataset.reads), [0, 1])
                    self.assertEqual(_batch_ids(next(loader)), [2, 3])
                self.assertEqual(sorted(dataset.reads), [0, 1, 2, 3])

    def test_cyclic_order_and_epoch_remain_native(self) -> None:
        """Both cyclic sharding policies retain the original round membership."""
        for data_sharding in (False, True):
            with self.subTest(data_sharding=data_sharding):
                options = {"sampler_type": "cyclic", "data_sharding": data_sharding, "seed": 13}
                reference = _sampler(**options)
                loader = _loader(_TrackedDataset(), sampler=_sampler(**options))
                for epoch in (0, 1):
                    reference.set_epoch(epoch)
                    loader.set_epoch(epoch)
                    expected = [sorted(indices) for indices in reference]
                    self.assertEqual([_batch_ids(batch) for batch in loader], expected)

    def test_duplicate_indices_are_distinct_occurrences(self) -> None:
        """A repeated physical index is read and delivered once per sampling occurrence."""
        for sidecar in (False, True):
            with self.subTest(sidecar=sidecar):
                dataset = _TrackedDataset()
                loader = _loader(dataset, sampler=_sampler(index_mapping=[3] * 10), sidecar=sidecar)
                self.assertEqual(_batch_ids(next(loader)), [3, 3])
                self.assertEqual(dataset.reads, [3, 3])
                self.assertEqual(len(set(loader.last_plan.selected_keys)), 2)

    def test_checkpoint_at_accumulation_round_excludes_prefetch(self) -> None:
        """A delivered FB round can replay even before an optimizer batch is complete."""
        for sidecar in (False, True):
            with self.subTest(sidecar=sidecar):
                sampler = _sampler()
                loader = _loader(_TrackedDataset(), sampler=sampler, sidecar=sidecar, double_buffer=True)
                self.assertEqual(_batch_ids(next(loader)), [0, 1])
                loader.wait_for_prefetch()
                self.assertEqual(sampler.consumed_samples, 4)
                state = loader.state_dict()
                owner = "sidecar_reader" if sidecar else "dataset_reader"
                self.assertEqual(state[owner]["sampler"]["consumed_samples"], 2)
                original = [_batch_ids(batch) for batch in loader]
                resumed = _loader(_TrackedDataset(), sidecar=sidecar, double_buffer=True)
                resumed.load_state_dict(state)
                self.assertEqual([_batch_ids(batch) for batch in resumed], original)
                self.assertEqual(original[0], [2, 3])

    def test_restore_rejects_changed_sampling_policy(self) -> None:
        """Changing shuffle seeds is not silently accepted by sampler replay."""
        loader = _loader(_TrackedDataset(), sampler=_sampler(sampler_type="cyclic", seed=13))
        next(loader)
        resumed = _loader(_TrackedDataset(), sampler=_sampler(sampler_type="cyclic", seed=14))
        with self.assertRaisesRegex(ValueError, "config_fingerprint"):
            resumed.load_state_dict(loader.state_dict())

    def test_tensor_index_mapping_has_stable_value_fingerprint(self) -> None:
        """Separately loaded tensor mappings do not fingerprint temporary storage IDs."""
        first = _loader(_TrackedDataset(), sampler=_sampler(index_mapping=platform.tensor([3] * 10)))
        self.assertEqual(_batch_ids(next(first)), [3, 3])
        second = _loader(_TrackedDataset(), sampler=_sampler(index_mapping=platform.tensor([3] * 10)))
        second.load_state_dict(first.state_dict())
        self.assertEqual(_batch_ids(next(second)), [3, 3])

    def test_existing_native_cursor_is_respected(self) -> None:
        """An already-restored sampler determines the first distributed read."""
        sampler = _sampler(consumed_samples=4)
        sampler.epoch = 2
        loader = _loader(_TrackedDataset(), sampler=sampler)
        self.assertEqual(_batch_ids(next(loader)), [4, 5])
        self.assertEqual(loader.state_dict()["epoch"], 2)

    def test_metadata_failure_does_not_look_like_exhaustion(self) -> None:
        """StopIteration from a metadata callback is a synchronized error, not EOF."""
        def broken_metadata(sample: object) -> SampleMetadata:
            """Raise an accidental StopIteration outside sampler advancement."""
            del sample
            raise StopIteration("invalid metadata callback")

        loader = build_distributed_dataloader(
            _TrackedDataset(), _StandaloneMesh(), DistributedDatasetConfig(seq_len=16, local_batch_size=2),
            batch_sampler=_sampler(), metadata_fn=broken_metadata,
        )
        with self.assertRaisesRegex(RuntimeError, "invalid metadata callback"):
            next(loader)

    def test_invalid_native_sampler_options_fail_at_build(self) -> None:
        """Reject incompatible slicing, shuffle, sizing, and custom packing early."""
        for options in ({"shuffle": True}, {"dataset_already_sharded": True}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "owns DP slicing"):
                _loader(_TrackedDataset(), **options)
        with self.assertRaisesRegex(ValueError, "micro_batch_size"):
            _loader(_TrackedDataset(), sampler=_sampler(micro_batch_size=1))
        with self.assertRaisesRegex(ValueError, "drop_last"):
            _loader(_TrackedDataset(), sampler=_sampler(drop_last=False))
        with self.assertRaisesRegex(ValueError, "pack_fn must be omitted"):
            build_distributed_dataloader(
                _TrackedDataset(), _StandaloneMesh(), DistributedDatasetConfig(seq_len=16, local_batch_size=2),
                batch_sampler=_sampler(), metadata_fn=_metadata, pack_fn=lambda samples, _: samples,
            )

    def test_sidecar_reader_advances_only_on_complete_commit(self) -> None:
        """Metadata-only reads retain the committed sampler cursor until delivery."""
        sampler = _sampler()
        reader = BatchSamplerReader(
            sampler, reader_rank=0,
            policy_fingerprint=native_sampler_fingerprint(sampler, data_rank=0, dp_size=1, local_batch_size=2),
            metadata=[SampleMetadata(1)] * 10, metadata_fn=None, sample_loader=None,
        )
        self.assertIsNone(reader.fill(min_samples=99, min_tokens=999, max_samples=100))
        saved = copy.deepcopy(reader.state_dict())
        self.assertEqual(saved["sampler"]["consumed_samples"], 0)
        with self.assertRaisesRegex(ValueError, "complete pending"):
            reader.commit({reader.metadata()[0].key})
        reader.commit({item.key for item in reader.metadata()})
        self.assertEqual(reader.state_dict()["sampler"]["consumed_samples"], 2)

    def test_trainer_builder_reuses_sampler_and_collator(self) -> None:
        """Opt-in uses the native sampler and delegates progress to the collective loader."""
        context = SimpleNamespace(dp_rank=0, dp_size=1, device_mesh=_StandaloneMesh(), pp_size=1)
        target = SimpleNamespace(dataloader_type="single", data_rearrange_map=[3] * 10)
        collator = object()
        with patch("hyper_parallel.data.batching.build_dataloader.build_distributed_dataloader") as build:
            loaders, samplers = build_dataloader(
                target, datasets=(_TrackedDataset(), None, None), collate_fn=collator,
                mesh_context=context,
                training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=7),
                data_config={"seq_length": 16, "load_balance": "native_batch_sampler"},
            )
        self.assertIs(loaders[0], build.return_value)
        self.assertEqual(samplers, (None, None, None))
        self.assertIs(build.call_args.kwargs["collate_fn"], collator)
        self.assertEqual(next(iter(build.call_args.kwargs["batch_sampler"])), [3, 3])

    def test_trainer_rejects_conflicting_selection_modes(self) -> None:
        """The new opt-in must not silently replace an independent packing/selection stage."""
        context = SimpleNamespace(dp_rank=0, dp_size=1, device_mesh=_StandaloneMesh(), pp_size=1)
        for target, config, message in (
                (SimpleNamespace(), {"packing_stage": "distributed_dataloader"}, "packing_stage"),
                (SimpleNamespace(_target_=DynamicBatchDataLoader), {}, "dynamic batching"),
        ):
            data_config = {"seq_length": 16, "load_balance": "native_batch_sampler", **config}
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, message):
                build_dataloader(
                    target, datasets=(_TrackedDataset(), None, None), collate_fn=tuple,
                    mesh_context=context,
                    training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=7),
                    data_config=data_config,
                )

    def test_trainer_uses_rank_local_payload_device(self) -> None:
        """Resolve the device in the build thread before background A2A starts."""
        mesh = _StandaloneMesh()
        mesh.device_type = "cuda"
        context = SimpleNamespace(dp_rank=0, dp_size=1, device_mesh=mesh, pp_size=1)
        with patch("hyper_parallel.data.batching.build_dataloader.torch.cuda.current_device", return_value=3), \
                patch("hyper_parallel.data.batching.build_dataloader.build_distributed_dataloader") as build:
            build_dataloader(
                SimpleNamespace(), datasets=(_TrackedDataset(), None, None), collate_fn=tuple,
                mesh_context=context,
                training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=7),
                data_config={"seq_length": 16, "load_balance": "native_batch_sampler"},
            )
        self.assertEqual(str(build.call_args.kwargs["communication_device"]), "cuda:3")
