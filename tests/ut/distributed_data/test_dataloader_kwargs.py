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

import inspect
import unittest
from unittest.mock import patch

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
)
from hyper_parallel.distributed_data.api import _config_fingerprint, _normalize_dataloader_kwargs


class _StandaloneMesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


def _metadata_fn(sample: dict[str, int]) -> SampleMetadata:
    """Derive deterministic packing metadata for one sample."""
    return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])


def _worker_init_fn(worker_id: int) -> None:
    """Provide a module-level, spawn-compatible worker initializer."""
    del worker_id


class TestDataLoaderKwargs(unittest.TestCase):
    """Verify DataLoader execution options without ceding sample ownership."""

    @staticmethod
    def _config(**kwargs: object) -> DistributedDatasetConfig:
        """Build the smallest valid distributed-data configuration."""
        return DistributedDatasetConfig(seq_len=8, local_batch_size=1, **kwargs)

    @staticmethod
    def _samples() -> list[dict[str, int]]:
        """Return a mapping-style, sidecar-compatible test Dataset."""
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
            "worker_init_fn": _worker_init_fn,
            "multiprocessing_context": "spawn",
            "pin_memory_device": "npu",
            "in_order": True,
        }

    def test_public_builder_exposes_keyword_only_dataloader_kwargs(self) -> None:
        """The public API should accept optional DataLoader kwargs by keyword."""
        parameter = inspect.signature(build_distributed_dataloader).parameters["dataloader_kwargs"]

        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(parameter.default)

    def test_dataloader_kwargs_override_config_without_mutating_either_input(self) -> None:
        """Explicit kwargs should win over legacy config fields on a copied mapping."""
        config = self._config(
            num_workers=1,
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=False,
        )
        supplied = self._execution_options()
        original = supplied.copy()

        normalized, _ = _normalize_dataloader_kwargs(config, supplied)

        for name, expected in original.items():
            self.assertIs(normalized[name], expected)
        self.assertEqual(supplied, original)
        self.assertEqual(config.num_workers, 1)
        self.assertFalse(config.pin_memory)
        self.assertEqual(config.prefetch_factor, 2)
        self.assertFalse(config.persistent_workers)

    def test_online_source_forwards_execution_options_to_native_dataloader(self) -> None:
        """Online Dataset Readers should pass normalized options to PyTorch."""
        supplied = self._execution_options()
        with patch("hyper_parallel.distributed_data.dataset_reader.DataLoader") as dataloader_type:
            build_distributed_dataloader(
                self._samples(),
                _StandaloneMesh(),
                self._config(),
                metadata_fn=_metadata_fn,
                dataloader_kwargs=supplied,
            )

        forwarded = dataloader_type.call_args.kwargs
        for name, expected in supplied.items():
            self.assertIs(forwarded[name], expected)
        self.assertIsNone(forwarded["batch_size"])
        self.assertIn("sampler", forwarded)
        self.assertIn("collate_fn", forwarded)
        self.assertIn("generator", forwarded)

    def test_sidecar_reader_forwards_execution_options_to_native_dataloader(self) -> None:
        """Plan-aware sidecar reads should use the same PyTorch worker options."""
        samples = self._samples()
        metadata = [_metadata_fn(sample) for sample in samples]
        supplied = self._execution_options()
        with patch("hyper_parallel.distributed_data.sidecar.DataLoader") as dataloader_type:
            build_distributed_dataloader(
                samples,
                _StandaloneMesh(),
                self._config(),
                metadata=metadata,
                dataloader_kwargs=supplied,
            )

        forwarded = dataloader_type.call_args.kwargs
        for name, expected in supplied.items():
            self.assertIs(forwarded[name], expected)
        self.assertIsNone(forwarded["batch_size"])
        self.assertIn("sampler", forwarded)
        self.assertIn("collate_fn", forwarded)
        self.assertIn("generator", forwarded)

    def test_rejects_framework_managed_and_unknown_options(self) -> None:
        """Users must not replace sample routing internals or arbitrary options."""
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
                    _normalize_dataloader_kwargs(self._config(), supplied)

    def test_rejects_invalid_worker_option_combinations(self) -> None:
        """Invalid combinations should fail before native DataLoader iteration."""
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
                    _normalize_dataloader_kwargs(self._config(), supplied)

    def test_equivalent_context_forms_and_custom_worker_init_share_fingerprint(self) -> None:
        """Stable fingerprints should ignore context objects and callable addresses."""
        config = self._config(num_workers=1)
        _, string_fingerprint = _normalize_dataloader_kwargs(
            config,
            {
                "multiprocessing_context": "spawn",
                "worker_init_fn": _worker_init_fn,
            },
        )
        _, context_fingerprint = _normalize_dataloader_kwargs(
            config,
            {
                "multiprocessing_context": torch.multiprocessing.get_context("spawn"),
                "worker_init_fn": _worker_init_fn,
            },
        )

        self.assertEqual(string_fingerprint, context_fingerprint)
        common = {
            "dataset_reader_ranks": (0,),
            "planner_rank": 0,
            "sidecar_mode": False,
            "communication_device_type": None,
            "uses_default_pack": True,
            "uses_default_collate": True,
        }
        string_config_fingerprint = _config_fingerprint(
            config,
            dataloader_fingerprint=string_fingerprint,
            **common,
        )
        context_config_fingerprint = _config_fingerprint(
            config,
            dataloader_fingerprint=context_fingerprint,
            **common,
        )
        self.assertEqual(string_config_fingerprint, context_config_fingerprint)

        _, no_callback_fingerprint = _normalize_dataloader_kwargs(
            config,
            {"multiprocessing_context": "spawn"},
        )
        self.assertNotEqual(string_fingerprint, no_callback_fingerprint)

    def test_equivalent_config_and_kwargs_worker_options_share_fingerprint(self) -> None:
        """Only effective worker settings should participate in build compatibility."""
        config_options = {
            "num_workers": 2,
            "pin_memory": True,
            "prefetch_factor": 3,
            "persistent_workers": True,
        }
        config = self._config(**config_options)
        override_config = self._config()
        _, config_fingerprint = _normalize_dataloader_kwargs(config, None)
        _, override_fingerprint = _normalize_dataloader_kwargs(override_config, config_options)

        self.assertEqual(config_fingerprint, override_fingerprint)
        common = {
            "dataset_reader_ranks": (0,),
            "planner_rank": 0,
            "sidecar_mode": False,
            "communication_device_type": None,
            "uses_default_pack": True,
            "uses_default_collate": True,
        }
        self.assertEqual(
            _config_fingerprint(config, dataloader_fingerprint=config_fingerprint, **common),
            _config_fingerprint(override_config, dataloader_fingerprint=override_fingerprint, **common),
        )

        _, integer_timeout = _normalize_dataloader_kwargs(config, {"timeout": 1})
        _, float_timeout = _normalize_dataloader_kwargs(config, {"timeout": 1.0})
        self.assertEqual(integer_timeout, float_timeout)


if __name__ == "__main__":
    unittest.main()
