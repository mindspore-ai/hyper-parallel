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
"""Replacement construction must not wait for parked peers or reload initial weights."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from hyper_parallel import DeviceMesh, Shard, distribute_tensor
from hyper_parallel.core.fault_tolerance.groups import GroupRegistry
from hyper_parallel.data.indexed import indexed_data_config
from hyper_parallel.data.parallel.dataloader_parallel import DataLoaderParallelContext, build_dataset_for_dataloader
from hyper_parallel.data.tools import io as indexed_io
from hyper_parallel.models._transformers.model_builder import _materialize_and_load_model


class TestReplacementBootstrap(unittest.TestCase):
    """Exercise local slicing and storage materialization with real CPU tensors."""

    def test_tp_source_rank_communication_is_skipped_only_for_replacement(self):
        mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("tp",), _init_backend=False)
        tensor = torch.arange(16).reshape(4, 4)
        with patch("hyper_parallel.core.dtensor.dtensor._distribute_tensor_with_communication",
                   side_effect=RuntimeError("peer collective")) as collective:
            with self.assertRaisesRegex(RuntimeError, "peer collective"):
                distribute_tensor(tensor, mesh, [Shard(0)], src_data_rank=0)
            registry = GroupRegistry()
            registry.reboot = True
            with registry:
                restored = distribute_tensor(tensor, mesh, [Shard(0)], src_data_rank=0)
            torch.testing.assert_close(restored.to_local(), tensor[:2])
            collective.assert_called_once()

    def test_meta_skeleton_materializes_without_initial_load_or_random_initialization(self):
        model = torch.nn.Linear(4, 4, device="meta")
        registry = GroupRegistry()
        registry.reboot = True
        with patch("hyper_parallel.models._transformers.model_builder.HuggingFaceCheckpointer") as checkpoint, \
                patch("hyper_parallel.models._transformers.model_builder._initialize_model_weights") as initialize:
            with registry:
                result = _materialize_and_load_model(model, is_meta_device=True, device=torch.device("cpu"),
                                                     load_base_model=True, pretrained_path="unused",
                                                     weights_mapping=None)
            self.assertIs(result, model)
            self.assertEqual(model.weight.device.type, "cpu")
            self.assertEqual(tuple(model.weight.shape), (4, 4))
            checkpoint.assert_not_called()
            initialize.assert_not_called()


class TestReplacementDataBootstrap(unittest.TestCase):
    """Rebuild Trainer data components without collectives against parked peers."""

    def test_indexed_paths_rebuild_locally_in_the_same_order(self):
        """Feature: Local indexed path reconstruction without a cached manifest.

        Description: Resolve unknown filename formats during original and replacement bootstrap.
        Expectation: Both workers use deterministic local discovery without collective participation.
        """
        registry = GroupRegistry()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("first", "second"):
                (root / name).mkdir()
                (root / name / "sample.bin").touch()
            (root / "top.bin").touch()
            top, first, second = (str(root / name) for name in ("top", "first/sample", "second/sample"))
            with patch.object(indexed_data_config, "dist") as distributed:
                distributed.is_initialized.return_value = True
                distributed.get_rank.return_value = 1
                distributed.get_world_size.return_value = 2
                distributed.all_gather_object.side_effect = AssertionError("parked peer")
                with registry:
                    original = indexed_data_config.resolve_data_paths(str(root))
                self.assertEqual(original, ["1", top, "1", first, "1", second])
                registry.reboot = True
                with registry:
                    restored = indexed_data_config.resolve_data_paths(str(root))
                self.assertEqual(restored, original)
                distributed.all_gather_object.assert_not_called()

    def test_replacement_explicit_blend_needs_no_store_manifest(self):
        """Feature: Explicit indexed blend bootstrap without cached data.

        Description: Reconstruct explicit weighted prefixes that need no directory discovery.
        Expectation: The configured order is preserved without Store records or collectives.
        """
        registry = GroupRegistry()
        registry.reboot = True
        with patch.object(indexed_data_config, "dist") as distributed:
            distributed.is_initialized.return_value = True
            distributed.get_rank.return_value = 1
            paths = ["30", "missing-dataset-a", "70", "missing-dataset-b"]
            with registry:
                self.assertEqual(indexed_data_config.resolve_data_paths(paths), paths)
            distributed.all_gather_object.assert_not_called()

    def test_replacement_dataset_build_preserves_input_ownership_without_barrier(self):
        registry = GroupRegistry()
        registry.reboot = True
        for owner, index_cache in ((True, False), (False, False), (False, True)):
            with self.subTest(owner=owner, index_cache=index_cache):
                context = DataLoaderParallelContext(distributed_enabled=True, data_index_cache=index_cache,
                                                    build_on_rank=lambda: owner, build_cache_on_rank=Mock(),
                                                    barrier=Mock(side_effect=AssertionError("parked peer")))
                dataset = object()
                factory = Mock(return_value=dataset)
                with registry:
                    actual = build_dataset_for_dataloader(factory, context, barrier_needed=True)
                self.assertIs(actual, dataset if owner or index_cache else None)
                self.assertEqual(factory.call_count, int(owner or index_cache))
                context.build_cache_on_rank.assert_not_called()
                context.barrier.assert_not_called()

    def test_original_dataset_build_keeps_cache_barrier_order(self):
        events = []
        context = DataLoaderParallelContext(distributed_enabled=True, build_on_rank=lambda: True,
                                            build_cache_on_rank=lambda: False,
                                            barrier=lambda: events.append("barrier"))
        with GroupRegistry():
            build_dataset_for_dataloader(lambda: events.append("build"), context, barrier_needed=True)
        self.assertEqual(events, ["barrier", "build"])

    def test_replacement_downloads_missing_index_atomically_without_barrier(self):
        registry = GroupRegistry()
        registry.reboot = True
        for prefix in ("s3://", "msc://"):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cache" / "data.idx"
                destinations = []

                def _download(*args):
                    destination = Path(args[-1])
                    destinations.append(destination)
                    self.assertFalse(path.exists())
                    self.assertNotEqual(destination, path)
                    destination.write_bytes(b"complete-index")

                with patch.object(indexed_io.torch, "distributed") as distributed, \
                        patch.object(indexed_io, "HAS_BOTO3", True), patch.object(indexed_io, "HAS_MSC", True), \
                        patch.object(indexed_io, "boto3") as boto3, \
                        patch.object(indexed_io, "multi_storage_client") as msc:
                    distributed.is_initialized.return_value = True
                    distributed.get_rank.return_value = 1
                    boto3.client.return_value.download_file.side_effect = _download
                    msc.download_file.side_effect = _download
                    with registry:
                        indexed_io._cache_index_file(prefix + "bucket/data.idx", str(path))
                    distributed.barrier.assert_not_called()
                self.assertEqual(path.read_bytes(), b"complete-index")
                self.assertEqual(len(destinations), 1)
                self.assertFalse(destinations[0].parent.exists())

    def test_cached_index_still_synchronizes_original_workers_only(self):
        """Feature: Deferred indexed-data startup after replacement construction.

        Description: Open a cached index after sealing the replacement registry.
        Expectation: Original workers synchronize and the replacement does not wait.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.idx"
            path.write_bytes(b"cached")
            registry = GroupRegistry()
            with patch.object(indexed_io.torch, "distributed") as distributed, \
                    patch.object(indexed_io, "HAS_BOTO3", True), patch.object(indexed_io, "boto3") as boto3:
                distributed.is_initialized.return_value = True
                distributed.get_rank.return_value = 1
                with registry:
                    indexed_io._cache_index_file("s3://bucket/data.idx", str(path))
                distributed.barrier.assert_called_once()
                distributed.barrier.reset_mock()
                registry.reboot = True
                registry.sealed = True
                with registry:
                    indexed_io._cache_index_file("s3://bucket/data.idx", str(path))
                distributed.barrier.assert_not_called()
                boto3.client.assert_not_called()

    def test_failed_download_does_not_publish_a_partial_index(self):
        registry = GroupRegistry()
        registry.reboot = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.idx"

            def _download(bucket, key, destination):
                del bucket, key
                Path(destination).write_bytes(b"partial")
                raise OSError("download interrupted")

            with patch.object(indexed_io.torch, "distributed") as distributed, \
                    patch.object(indexed_io, "HAS_BOTO3", True), patch.object(indexed_io, "boto3") as boto3:
                distributed.is_initialized.return_value = True
                boto3.client.return_value.download_file.side_effect = _download
                with registry, self.assertRaisesRegex(OSError, "download interrupted"):
                    indexed_io._cache_index_file("s3://bucket/data.idx", str(path))
                distributed.barrier.assert_not_called()
                boto3.client.return_value.close.assert_called_once()
            self.assertFalse(path.exists())
            self.assertFalse(list(Path(directory).iterdir()))
