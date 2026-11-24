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
"""Tests for user-supplied PyTorch DataLoader execution options."""

from __future__ import annotations

import inspect
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.batching.build_dataloader import build_dataloader as build_trainer_dataloader
from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleKey,
    SampleMetadata,
    build_distributed_dataloader,
)
from hyper_parallel.distributed_data.api import _normalize_dataloader_kwargs
from hyper_parallel.distributed_data.metadata import PlannedSampleLoader
from hyper_parallel.trainer.config import DataLoaderConfig
from hyper_parallel.trainer.config.resolver import ConfigResolutionError, resolve_component
from tests.fixtures.data.workers import initialize_worker


class _StandaloneMesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


def _metadata_fn(sample: dict[str, int]) -> SampleMetadata:
    """Derive deterministic packing metadata for one sample."""
    return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])


class TestDataLoaderKwargs(unittest.TestCase):
    """Verify DataLoader execution options without ceding sample ownership."""

    def test_legacy_sampler_option_reports_renamed_field(self) -> None:
        """Feature: DataLoader configuration compatibility.
        Description: Resolve a removed dataloader_type option.
        Expectation: Report the new sampler_type name through the configuration error API.
        """
        with self.assertRaisesRegex(ConfigResolutionError, "dataloader_type.*renamed to sampler_type"):
            resolve_component({"dataloader_type": "single"}, annotation=DataLoaderConfig, path="dataloader")

    @staticmethod
    def _config(**kwargs: object) -> DistributedDatasetConfig:
        """Build the smallest valid distributed-data configuration."""
        return DistributedDatasetConfig(seq_len=8, local_batch_size=1, **kwargs)

    @staticmethod
    def _samples() -> list[dict[str, int]]:
        """Return a mapping-style, metadata-compatible test Dataset."""
        return [{"id": 0, "tokens": 4}, {"id": 1, "tokens": 4}]

    @staticmethod
    def _execution_options() -> dict[str, object]:
        """Return valid non-default worker options for forwarding tests."""
        return {
            "num_workers": 2,
            "pin_memory": True,
            "prefetch_factor": 3,
            "persistent_workers": True,
            "timeout": 7,
            "worker_init_fn": initialize_worker,
            "multiprocessing_context": "spawn",
            "pin_memory_device": "npu",
            "in_order": True,
        }

    def test_public_builder_exposes_keyword_only_dataloader_kwargs(self) -> None:
        """Feature: DataLoader execution options API.
        Description: Inspect the public builder's dataloader_kwargs parameter.
        Expectation: The parameter is optional and keyword-only.
        """
        parameter = inspect.signature(build_distributed_dataloader).parameters["dataloader_kwargs"]

        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(parameter.default)

    def test_dataloader_kwargs_preserve_supplied_options(self) -> None:
        """Feature: DataLoader execution options.
        Description: Normalize explicit options over independent execution defaults.
        Expectation: Supplied options are retained without mutating the caller's mapping.
        """
        supplied = self._execution_options()
        original = supplied.copy()

        normalized = _normalize_dataloader_kwargs(supplied)

        for name, expected in original.items():
            self.assertIs(normalized[name], expected)
        self.assertEqual(supplied, original)

    def test_omitted_execution_options_keep_native_defaults(self) -> None:
        """Feature: Default DataLoader execution.
        Description: Build online and metadata loaders without explicit worker settings.
        Expectation: Defaults reach PyTorch and leave its prefetch factor unspecified.
        """
        for metadata_mode in (False, True):
            for supplied in (None, {}):
                with self.subTest(metadata_mode=metadata_mode, supplied=supplied), patch(
                        "hyper_parallel.distributed_data.metadata.DataLoader",
                ) as dataloader_type:
                    with self._checkpoint_loader(self._config(metadata_mode=metadata_mode), dataloader_kwargs=supplied):
                        forwarded = dataloader_type.call_args.kwargs
                        self.assertEqual(forwarded["num_workers"], 0, "Default reads must remain synchronous.")
                        self.assertFalse(forwarded["pin_memory"], "Pinning must remain opt-in.")
                        self.assertFalse(forwarded["persistent_workers"], "Workers must not persist by default.")
                        self.assertNotIn("prefetch_factor", forwarded, "PyTorch must retain its prefetch default.")

    @staticmethod
    def _trainer_loader(worker_options: dict, distributed_options: dict | None = None):
        """Build through the Trainer entry with a CPU-only, single-rank mesh."""
        loaders, _ = build_trainer_dataloader(
            SimpleNamespace(load_balance=distributed_options or {}, **worker_options),
            datasets=(TestDataLoaderKwargs._samples(), None, None),
            collate_fn=tuple,
            mesh_context=SimpleNamespace(dp_rank=0, dp_size=1, pp_size=1, device_mesh=_StandaloneMesh()),
            training_config=SimpleNamespace(micro_batch_size=1, global_batch_size=1, seed=17),
            data_config={"seq_length": 8},
            metadata_fn=_metadata_fn,
            cost_model=lambda metadata: metadata.cost,
        )
        return loaders[0]

    def test_trainer_forwards_execution_options_to_native_dataloader(self) -> None:
        """Feature: Trainer worker configuration.
        Description: Build with default and explicit DataLoader target options.
        Expectation: All configured execution options reach the underlying DataLoader.
        """
        for supplied in ({}, self._execution_options()):
            with self.subTest(supplied=tuple(supplied)), patch(
                    "hyper_parallel.distributed_data.metadata.DataLoader",
            ) as dataloader_type:
                with self._trainer_loader(supplied):
                    forwarded = dataloader_type.call_args.kwargs
                    for name, expected in supplied.items():
                        self.assertIs(forwarded[name], expected, f"Trainer must forward {name} unchanged.")
                    self.assertEqual(forwarded["num_workers"], supplied.get("num_workers", 0))
                    self.assertIsNone(forwarded["batch_size"], "Worker options must not change plan-aware batching.")

    def test_trainer_rejects_worker_options_in_packing_config(self) -> None:
        """Feature: Trainer configuration ownership.
        Description: Put DataLoader execution options in the distributed packing section.
        Expectation: The error directs callers to the dataloader section before workers start.
        """
        for name in ("num_workers", "pin_memory", "prefetch_factor", "persistent_workers"):
            with self.subTest(name=name), patch(
                    "hyper_parallel.distributed_data.metadata.DataLoader",
            ) as dataloader_type:
                with self.assertRaisesRegex(ValueError, "configure DataLoader execution options on dataloader"):
                    self._trainer_loader({}, {name: self._execution_options()[name]})
                dataloader_type.assert_not_called()

    def test_online_source_forwards_execution_options_to_native_dataloader(self) -> None:
        """Feature: Online sample loading options.
        Description: Build an online loader with custom worker execution options.
        Expectation: Normalized options reach PyTorch without changing sample routing.
        """
        supplied = self._execution_options()
        with patch("hyper_parallel.distributed_data.metadata.DataLoader") as dataloader_type:
            build_distributed_dataloader(
                self._samples(),
                _StandaloneMesh(),
                self._config(),
                metadata_fn=_metadata_fn,
                device="cpu", cost_model=lambda metadata: metadata.cost,
                dataloader_kwargs=supplied,
                batch_sampler=build_dataset_batch_sampler(
                    total_samples=2, micro_batch_size=1, global_batch_size=1, dp_world_size=1, dp_rank=0,
                ),
            )

        forwarded = dataloader_type.call_args.kwargs
        for name, expected in supplied.items():
            self.assertIs(forwarded[name], expected)
        self.assertIsNone(forwarded["batch_size"])
        self.assertIn("sampler", forwarded)
        self.assertIn("collate_fn", forwarded)
        self.assertIn("generator", forwarded)

    def test_metadata_reader_forwards_execution_options_to_native_dataloader(self) -> None:
        """Feature: Metadata-mode sample loading options.
        Description: Build a metadata-mode loader with custom worker execution options.
        Expectation: Plan-aware sample reads receive the supplied PyTorch options.
        """
        samples = self._samples()
        metadata = [_metadata_fn(sample) for sample in samples]
        supplied = self._execution_options()
        with patch("hyper_parallel.distributed_data.metadata.DataLoader") as dataloader_type:
            build_distributed_dataloader(
                samples,
                _StandaloneMesh(),
                self._config(metadata_mode=True),
                metadata=metadata,
                device="cpu", cost_model=lambda metadata: metadata.cost,
                dataloader_kwargs=supplied,
                batch_sampler=build_dataset_batch_sampler(
                    total_samples=2, micro_batch_size=1, global_batch_size=1, dp_world_size=1, dp_rank=0,
                ),
            )

        forwarded = dataloader_type.call_args.kwargs
        for name, expected in supplied.items():
            self.assertIs(forwarded[name], expected)
        self.assertIsNone(forwarded["batch_size"])
        self.assertIn("sampler", forwarded)
        self.assertIn("collate_fn", forwarded)
        self.assertIn("generator", forwarded)

    def test_rejects_framework_managed_and_unknown_options(self) -> None:
        """Feature: DataLoader option ownership.
        Description: Supply framework-managed or unsupported DataLoader options.
        Expectation: Normalization rejects options that could replace sample routing.
        """
        cases = (
            ({"batch_size": 2}, "cannot override distributed sampling"),
            ({"sampler": object()}, "cannot override distributed sampling"),
            ({"collate_fn": list}, "cannot override distributed sampling"),
            ({"generator": torch.Generator()}, "cannot override distributed sampling"),
            ({"unknown_option": True}, "contains unsupported options"),
        )
        for supplied, expected_error in cases:
            with self.subTest(supplied=tuple(supplied)):
                with self.assertRaisesRegex(ValueError, expected_error):
                    _normalize_dataloader_kwargs(supplied)

    def test_rejects_invalid_worker_option_combinations(self) -> None:
        """Feature: DataLoader worker option validation.
        Description: Supply incompatible worker settings or invalid callback types.
        Expectation: Invalid options fail before native DataLoader iteration.
        """
        cases = (
            ({"num_workers": 0, "prefetch_factor": 2}, "prefetch_factor requires num_workers"),
            ({"num_workers": 0, "persistent_workers": True}, "persistent_workers=True requires"),
            ({"num_workers": 0, "timeout": 1}, "timeout must be zero"),
            ({"num_workers": 0, "multiprocessing_context": "spawn"}, "multiprocessing_context requires"),
            ({"num_workers": 1, "in_order": False}, "in_order must remain True"),
            ({"num_workers": 1, "worker_init_fn": object()}, "worker_init_fn must be callable"),
            ({"pin_memory_device": object()}, "pin_memory_device must be a string"),
        )
        for supplied, expected_error in cases:
            with self.subTest(supplied=tuple(supplied)):
                with self.assertRaisesRegex(ValueError, expected_error):
                    _normalize_dataloader_kwargs(supplied)

    def test_worker_validation_is_shared_by_kwargs_and_direct_readers(self) -> None:
        """Feature: Shared worker validation.
        Description: Supply invalid settings through execution kwargs and planned reads.
        Expectation: Every entry point rejects invalid settings before creating workers.
        """
        cases = (
            {"num_workers": True}, {"num_workers": -1}, {"pin_memory": 1}, {"persistent_workers": 1},
            {"prefetch_factor": True}, {"prefetch_factor": 0}, {"prefetch_factor": 2},
            {"persistent_workers": True},
        )
        for invalid in cases:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _normalize_dataloader_kwargs(invalid)
                options = {"num_workers": 0, "pin_memory": False, "prefetch_factor": None, "persistent_workers": False}
                options.update(invalid)
                with self.assertRaises(ValueError):
                    PlannedSampleLoader(self._samples(), seed=17, **options)

    def test_accepts_string_and_object_multiprocessing_contexts(self) -> None:
        """Feature: Native DataLoader execution options.
        Description: Validate both supported multiprocessing context representations.
        Expectation: The context and worker initializer reach the loader unchanged.
        """
        for context in ("spawn", torch.multiprocessing.get_context("spawn")):
            with self.subTest(context=context):
                options = _normalize_dataloader_kwargs(
                    {"num_workers": 1, "multiprocessing_context": context, "worker_init_fn": initialize_worker},
                )
                self.assertIs(options["multiprocessing_context"], context)
                self.assertIs(options["worker_init_fn"], initialize_worker)

    def test_direct_planned_reads_reject_unordered_results(self) -> None:
        """Feature: Ordered plan-aware reads.
        Description: Disable ordered results when constructing a planned reader directly.
        Expectation: Construction fails before starting workers that could misroute payloads.
        """
        with self.assertRaisesRegex(ValueError, "in_order must remain True"):
            PlannedSampleLoader(
                self._samples(), seed=17, num_workers=2, pin_memory=False,
                prefetch_factor=2, persistent_workers=True, dataloader_kwargs={"in_order": False},
            )

    def test_planned_reads_replace_requests_with_persistent_workers(self) -> None:
        """Feature: Reusable plan-aware sample loading.
        Description: Fetch changing requests, repeated indices and an empty request with and without workers.
        Expectation: Each request returns exactly its ordered occurrences without reusing stale indices.
        """
        for num_workers in (0, 2):
            with self.subTest(num_workers=num_workers):
                loader = PlannedSampleLoader(
                    self._samples(), seed=17, num_workers=num_workers, pin_memory=False,
                    prefetch_factor=2 if num_workers else None, persistent_workers=bool(num_workers),
                    dataloader_kwargs={"multiprocessing_context": "spawn", "timeout": 30} if num_workers else None,
                )
                for step, indices in enumerate(((1, 0, 1), (), (0,), (1, 0))):
                    keys = [SampleKey(0, index, step * 3 + offset) for offset, index in enumerate(indices)]
                    payloads = loader.fetch_samples(keys)
                    self.assertEqual(list(payloads), keys, "Each sample occurrence must keep its requested order.")
                    self.assertEqual(
                        [sample["id"] for sample in payloads.values()], list(indices),
                        "Reused workers must read the current request, including duplicate indices.",
                    )
                del loader

    @staticmethod
    def _checkpoint_loader(config: DistributedDatasetConfig, **kwargs: object):
        """Build a deterministic loader with enough samples to resume multiple steps."""
        samples = [{"id": index, "tokens": 4} for index in range(6)]
        metadata_options = (
            {"metadata": [_metadata_fn(sample) for sample in samples]}
            if config.metadata_mode else {"metadata_fn": _metadata_fn}
        )
        return build_distributed_dataloader(
            samples, _StandaloneMesh(), config, **metadata_options,
            batch_sampler=build_dataset_batch_sampler(
                total_samples=len(samples), micro_batch_size=1, global_batch_size=1, dp_world_size=1, dp_rank=0,
            ),
            device="cpu", cost_model=lambda metadata: metadata.cost, **kwargs,
        )

    def test_checkpoint_restore_allows_different_worker_options(self) -> None:
        """Feature: DataLoader checkpoint replay.
        Description: Resume online and metadata loaders with new worker settings in dataloader_kwargs.
        Expectation: Changing worker count and prefetch settings preserves remaining samples.
        """
        worker_options = {"num_workers": 1, "prefetch_factor": 2, "persistent_workers": True}
        extra_options = {"multiprocessing_context": "spawn", "worker_init_fn": initialize_worker, "timeout": 30}
        for metadata_mode in (False, True):
            config = self._config(metadata_mode=metadata_mode)
            with self._checkpoint_loader(config) as loader:
                next(loader)
                state = loader.state_dict()
                expected = list(loader)
            self.assertTrue(expected, "Checkpoint must have remaining samples to replay.")
            with self.subTest(metadata_mode=metadata_mode):
                with self._checkpoint_loader(config, dataloader_kwargs={**worker_options, **extra_options}) as resumed:
                    resumed.load_state_dict(state)
                    self.assertEqual(list(resumed), expected, "Worker settings must not change checkpoint replay.")

    def test_checkpoint_restore_allows_changed_packing_config(self) -> None:
        """Feature: DataLoader checkpoint compatibility.
        Description: Resume both metadata modes with changed packing settings and optional legacy fingerprints.
        Expectation: Compatible reader state restores the remaining samples without configuration hashing.
        """
        for metadata_mode in (False, True):
            config = self._config(metadata_mode=metadata_mode)
            with self._checkpoint_loader(config) as loader:
                next(loader)
                state = loader.state_dict()
                expected = list(loader)
            for removed_field in ("config_fingerprint", "topology_fingerprint", "last_plan_id"):
                self.assertNotIn(removed_field, state)
            self.assertTrue(expected, "Checkpoint must have remaining samples to replay.")
            changed = replace(config, seq_len=16, min_balance_gain=0.5)
            for legacy in (False, True):
                saved = dict(
                    state, config_fingerprint="legacy-config", topology_fingerprint="legacy-layout",
                    last_plan_id="legacy-plan",
                ) if legacy else state
                with self.subTest(metadata_mode=metadata_mode, legacy=legacy), self._checkpoint_loader(
                        changed,
                ) as resumed:
                    resumed.load_state_dict(saved)
                    self.assertEqual(list(resumed), expected, "Compatible packing settings must preserve replay.")

    def test_checkpoint_restore_rejects_changed_reader_seed(self) -> None:
        """Feature: Deterministic reader checkpoint replay.
        Description: Restore online and metadata readers with a different worker seed.
        Expectation: Reader seed validation remains active without a loader configuration fingerprint.
        """
        for metadata_mode in (False, True):
            config = self._config(metadata_mode=metadata_mode)
            with self._checkpoint_loader(config) as loader:
                next(loader)
                state = loader.state_dict()
            changed = replace(config, seed=config.seed + 1)
            with self.subTest(metadata_mode=metadata_mode), self._checkpoint_loader(changed) as resumed:
                with self.assertRaisesRegex(ValueError, "checkpoint seed"):
                    resumed.load_state_dict(state)

    def test_checkpoint_restore_rejects_another_rank(self) -> None:
        """Feature: Rank-local checkpoint ownership.
        Description: Restore a checkpoint saved by a different global rank in both loading modes.
        Expectation: Rank ownership is checked before restoring any reader progress.
        """
        for metadata_mode in (False, True):
            config = self._config(metadata_mode=metadata_mode)
            with self._checkpoint_loader(config) as loader:
                next(loader)
                state = loader.state_dict()
            state["global_rank"] = 1
            with self.subTest(metadata_mode=metadata_mode), self._checkpoint_loader(config) as resumed:
                with self.assertRaisesRegex(ValueError, "checkpoint global_rank"):
                    resumed.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
