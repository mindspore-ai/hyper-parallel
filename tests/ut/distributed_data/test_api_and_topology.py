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
"""Tests for the public dynamic-packing API and named-mesh topology."""

import inspect
import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

from hyper_parallel import distributed_data
from hyper_parallel.distributed_data import DistributedDatasetConfig, SampleMetadata, build_distributed_dataloader
from hyper_parallel.distributed_data.locality import _create_locality_groups, _fixed_rank_groups
from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data.topology import DataTopology


class TestDistributedDataPublicApi(unittest.TestCase):
    """Verify that the public API is expressed in training-facing terms."""

    def test_config_uses_sequence_and_local_batch_sizing(self) -> None:
        """Feature: Distributed data configuration.
        Description: Inspect sequence, local-batch, and sharding fields.
        Expectation: Training-facing fields replace removed raw-read sizing concepts.
        """
        field_names = {field.name for field in fields(DistributedDatasetConfig)}

        self.assertIn("seq_len", field_names)
        self.assertIn("local_batch_size", field_names)
        self.assertNotIn("enable_dp_balance", field_names)
        self.assertNotIn("double_buffer", field_names)
        self.assertIn("dataset_already_sharded", field_names)
        self.assertNotIn("raw_sample_size", field_names)
        self.assertNotIn("micro_batch_num", field_names)
        self.assertNotIn("buffer_size_multiplier", field_names)
        self.assertNotIn("shuffle", field_names)
        self.assertIn("metadata_mode", field_names)
        for name in ("num_workers", "pin_memory", "prefetch_factor", "persistent_workers"):
            self.assertNotIn(name, field_names, f"{name} must be configured through dataloader_kwargs.")

        config = DistributedDatasetConfig(seq_len=32_768, local_batch_size=4)
        self.assertFalse(hasattr(config, "enable_dp_balance"))
        self.assertFalse(config.dataset_already_sharded)
        self.assertFalse(config.metadata_mode)
        self.assertFalse(hasattr(config, "raw_sample_size"))
        self.assertFalse(hasattr(config, "micro_batch_num"))
        self.assertFalse(hasattr(config, "buffer_size_multiplier"))

    def test_builder_accepts_a_raw_dataset_and_constructor_callbacks(self) -> None:
        """Feature: Distributed DataLoader builder API.
        Description: Inspect the raw Dataset and constructor callback parameters.
        Expectation: Loading, packing, and collation remain separate contracts.
        """
        signature = inspect.signature(build_distributed_dataloader)
        parameters = signature.parameters

        self.assertEqual(tuple(parameters)[:3], ("dataset", "mesh", "config"))
        self.assertEqual(parameters["metadata_fn"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters["metadata"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters["pack_fn"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters["collate_fn"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters["device"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertNotIn("communication_device", parameters)
        self.assertIsNone(parameters["metadata_fn"].default)
        self.assertIsNone(parameters["metadata"].default)
        self.assertIsNone(parameters["pack_fn"].default)
        self.assertIsNone(parameters["collate_fn"].default)
        self.assertNotIn("raw_sample_size", parameters)
        self.assertNotIn("micro_batch_num", parameters)

    def test_public_default_callbacks_preserve_nested_batch_boundaries(self) -> None:
        """Feature: Default data construction callbacks.
        Description: Pack raw samples and collate multiple planned sequences.
        Expectation: Immutable packing-bin and local-batch boundaries are preserved.
        """
        raw_samples = [{"id": 0}, {"id": 1}]
        packed = distributed_data.default_pack_fn(raw_samples, seq_len=32)

        self.assertEqual(packed, ({"id": 0}, {"id": 1}))
        self.assertIsInstance(packed, tuple)
        collated = distributed_data.default_collate_fn([packed, ({"id": 2},)])
        self.assertEqual(collated, (({"id": 0}, {"id": 1}), ({"id": 2},)))
        self.assertIsInstance(collated, tuple)

    def test_backend_selection_is_internal(self) -> None:
        """Feature: Fixed control-plane backend.
        Description: Inspect the public configuration fields.
        Expectation: Backend selection is no longer a caller-facing option.
        """
        field_names = {field.name for field in fields(DistributedDatasetConfig)}
        self.assertNotIn("cpu_backend", field_names)
        self.assertNotIn("payload_backend", field_names)

    def test_dataset_already_sharded_must_be_boolean(self) -> None:
        """Feature: Dataset Reader sharding.
        Description: Configure the reader-stride switch with a truthy non-boolean value.
        Expectation: Configuration validation rejects the invalid value.
        """
        with self.assertRaisesRegex(ValueError, "dataset_already_sharded must be boolean"):
            DistributedDatasetConfig(seq_len=32, local_batch_size=1, dataset_already_sharded=1)

    def test_metadata_mode_must_be_boolean(self) -> None:
        """Feature: Explicit metadata source selection.
        Description: Configure metadata mode with a truthy non-boolean value.
        Expectation: Configuration validation rejects the invalid value.
        """
        with self.assertRaisesRegex(ValueError, "metadata_mode must be boolean"):
            DistributedDatasetConfig(seq_len=32, local_batch_size=1, metadata_mode=1)

    def test_plan_broadcast_defaults_on_and_requires_boolean(self) -> None:
        """Feature: Configurable plan broadcast.
        Description: Use the default, disable broadcast, and supply invalid flag values.
        Expectation: Central planning remains the default and only boolean values are accepted.
        """
        self.assertTrue(DistributedDatasetConfig(seq_len=32, local_batch_size=1).broadcast_plan)
        self.assertFalse(DistributedDatasetConfig(seq_len=32, local_batch_size=1, broadcast_plan=False).broadcast_plan)
        for value in (0, 1, "false", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "broadcast_plan must be boolean"):
                DistributedDatasetConfig(seq_len=32, local_batch_size=1, broadcast_plan=value)

    def test_balance_logging_defaults_off_and_requires_boolean(self) -> None:
        """Feature: Opt-in balancing reports.
        Description: Omit the switch, enable it, and supply non-boolean values.
        Expectation: Logging defaults off and accepts only boolean configuration.
        """
        self.assertFalse(DistributedDatasetConfig(seq_len=32, local_batch_size=1).log_balance)
        self.assertTrue(DistributedDatasetConfig(seq_len=32, local_batch_size=1, log_balance=True).log_balance)
        for value in (0, 1, "false", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "log_balance must be boolean"):
                DistributedDatasetConfig(seq_len=32, local_batch_size=1, log_balance=value)

    def test_text_repacking_config_defaults_and_constraints(self) -> None:
        """Feature: Built-in text repacking configuration.
        Description: Inspect defaults and reject invalid flags, PAD IDs and offline metadata.
        Expectation: Repacking is opt-in and unsupported configurations fail at construction.
        """
        config = DistributedDatasetConfig(seq_len=12, local_batch_size=1)
        self.assertFalse(config.repack_sequences)
        self.assertIsNone(config.pad_token_id)
        for value in (0, 1, "true", None):
            with self.subTest(repack=value), self.assertRaisesRegex(ValueError, "repack_sequences must be boolean"):
                DistributedDatasetConfig(seq_len=12, local_batch_size=1, repack_sequences=value)
        for value in (-1, True, 1.5, "151645"):
            with self.subTest(pad=value), self.assertRaisesRegex(ValueError, "pad_token_id"):
                DistributedDatasetConfig(seq_len=12, local_batch_size=1, pad_token_id=value)
        with self.assertRaisesRegex(ValueError, "repack_sequences requires online metadata"):
            DistributedDatasetConfig(seq_len=12, local_batch_size=1, repack_sequences=True, metadata_mode=True)

    def test_balance_group_size_is_positive_and_topology_divisible(self) -> None:
        """Feature: Local balancing groups.
        Description: Configure fixed-size groups over the root mesh rank order.
        Expectation: Valid groups partition the mesh and invalid sizes fail clearly.
        """
        config = DistributedDatasetConfig(seq_len=32, local_batch_size=1, balance_group_size=2)
        self.assertEqual(config.balance_group_size, 2)
        topology = DataTopology.from_layout(
            mesh_shape=(4,), mesh_dim_names=("dp",), rank_list=(0, 1, 2, 3), global_rank=0,
        )
        self.assertEqual(_fixed_rank_groups(topology, config.balance_group_size), ((0, 1), (2, 3)))
        with self.assertRaisesRegex(ValueError, "balance_group_size must be a positive integer"):
            DistributedDatasetConfig(seq_len=32, local_batch_size=1, balance_group_size=0)
        with self.assertRaisesRegex(ValueError, "must divide the root mesh size"):
            _fixed_rank_groups(topology, 3)

    def test_locality_groups_use_nodes_or_fixed_rank_order(self) -> None:
        """Feature: Local balancing group discovery.
        Description: Build node-based and fixed-size groups with mocked WORLD communication.
        Expectation: Node discovery groups matching hosts; fixed groups need no startup gather.
        """
        mesh = SimpleNamespace(mesh_shape=(4,), mesh_dim_names=("dp",), rank_list=(0, 1, 2, 3))
        for group_size in (None, 2):
            with self.subTest(group_size=group_size), patch(
                    "hyper_parallel.distributed_data.locality.dist",
            ) as distributed, patch(
                    "hyper_parallel.distributed_data.locality.all_gather_control_object",
                    return_value=[(0, "node-a"), (1, "node-b"), (2, "node-a"), (3, "node-b")],
            ) as gather, patch.dict("os.environ", {"GROUP_RANK": "node-a"}):
                distributed.is_available.return_value = True
                distributed.is_initialized.return_value = True
                distributed.get_rank.return_value = 0
                distributed.get_world_size.return_value = 4

                _, groups = _create_locality_groups(
                    mesh, balance_group_size=group_size, communication_backend="gloo",
                )

                if group_size is None:
                    gather.assert_called_once()
                    self.assertEqual(gather.call_args.args[0], (0, "node-a"))
                    expected_groups = [[0, 1, 2, 3], [0, 2], [1, 3]]
                    self.assertEqual(groups.data_plane_ranks, (0, 2))
                else:
                    gather.assert_not_called()
                    expected_groups = [[0, 1], [2, 3]]
                    self.assertEqual(groups.data_plane_ranks, (0, 1))
                self.assertEqual(
                    [call.kwargs["ranks"] for call in distributed.new_group.call_args_list], expected_groups,
                    "Every rank must create the same process groups in the same order.",
                )


class TestDistributedDataBuildState(unittest.TestCase):
    """Keep mode-specific construction and direct validation failures explicit."""

    @staticmethod
    def _mesh() -> SimpleNamespace:
        """Return a standalone mesh that needs no distributed initialization."""
        return SimpleNamespace(mesh_shape=(1,), mesh_dim_names=("dp",), rank_list=(0,))

    def test_sampler_modes_yield_sampled_data(self) -> None:
        """Feature: Dataset Reader construction.
        Description: Build native sampler loaders in online and metadata modes.
        Expectation: Both modes yield all sampled data in order.
        """
        samples = [0, 1]
        metadata = [SampleMetadata(pack_tokens=4, sample_id=index) for index in samples]
        for metadata_mode in (False, True):
            config = DistributedDatasetConfig(seq_len=8, local_batch_size=1, metadata_mode=metadata_mode)
            callbacks = {"metadata": metadata} if metadata_mode else {"metadata_fn": metadata.__getitem__}
            with self.subTest(metadata_mode=metadata_mode):
                sampler = build_dataset_batch_sampler(
                    total_samples=2, micro_batch_size=1, global_batch_size=1, dp_world_size=1, dp_rank=0,
                )
                loader = build_distributed_dataloader(
                    samples, self._mesh(), config, batch_sampler=sampler, **callbacks,
                    device="cpu", cost_model=lambda metadata: metadata.cost,
                )
                self.assertEqual(list(loader), [(0,), (1,)])

    def test_metadata_length_mismatch_raises_before_group_creation(self) -> None:
        """Feature: Local metadata alignment.
        Description: Supply fewer metadata entries than Dataset samples.
        Expectation: The builder rejects the mismatch before starting collectives.
        """
        sampler = build_dataset_batch_sampler(
            total_samples=2, micro_batch_size=1, global_batch_size=1, dp_world_size=1, dp_rank=0,
        )
        with patch("hyper_parallel.distributed_data.api.create_data_groups") as create_groups:
            with self.assertRaisesRegex(ValueError, "metadata must align with the mapping Dataset"):
                build_distributed_dataloader(
                    [0, 1], self._mesh(),
                    DistributedDatasetConfig(seq_len=8, local_batch_size=1, metadata_mode=True),
                    batch_sampler=sampler, metadata=[SampleMetadata(1)], device="cpu",
                    cost_model=lambda metadata: metadata.cost,
                )
            create_groups.assert_not_called()

    def test_invalid_config_raises_before_group_creation(self) -> None:
        """Feature: Local configuration validation.
        Description: Supply invalid configuration objects to the builder.
        Expectation: Errors are raised before group creation.
        """
        for config in (None, {}, object()):
            with self.subTest(config=config), patch(
                    "hyper_parallel.distributed_data.api.create_data_groups",
            ) as create_groups:
                with self.assertRaisesRegex(ValueError, "config must be DistributedDatasetConfig"):
                    build_distributed_dataloader([], self._mesh(), config)
                create_groups.assert_not_called()

    def test_partial_build_failures_raise_before_group_creation(self) -> None:
        """Feature: Local builder input validation.
        Description: Supply invalid loader options, devices, and metadata combinations.
        Expectation: Local errors are raised before group creation.
        """
        cases = (
            ({"dataloader_kwargs": {"num_workers": -1}, "metadata_fn": lambda _: SampleMetadata(1)},
             "num_workers", False),
            ({"device": "invalid-device", "metadata_fn": lambda _: SampleMetadata(1)},
             "communication_device", False),
            ({"metadata": [SampleMetadata(pack_tokens=1)]}, "Metadata mode requires batch_sampler", True),
        )
        for sharded in (False, True):
            for kwargs, message, metadata_mode in cases:
                config = DistributedDatasetConfig(
                    seq_len=8, local_batch_size=1, dataset_already_sharded=sharded,
                    metadata_mode=metadata_mode,
                )
                with self.subTest(sharded=sharded, message=message), patch(
                        "hyper_parallel.distributed_data.api.create_data_groups",
                ) as create_groups:
                    with self.assertRaisesRegex(ValueError, message):
                        build_distributed_dataloader([0, 1], self._mesh(), config, **kwargs)
                    create_groups.assert_not_called()

    def test_metadata_mode_requires_the_configured_metadata_source(self) -> None:
        """Feature: Explicit metadata mode.
        Description: Supply metadata sources that conflict with the configured mode.
        Expectation: The builder rejects each incompatible or missing source.
        """
        metadata = [SampleMetadata(pack_tokens=1), SampleMetadata(pack_tokens=1)]
        cases = (
            (
                DistributedDatasetConfig(seq_len=8, local_batch_size=1, metadata_mode=True),
                {"metadata_fn": lambda _: metadata[0]},
                "metadata_mode=True requires precomputed metadata",
            ),
            (
                DistributedDatasetConfig(seq_len=8, local_batch_size=1, metadata_mode=True),
                {},
                "metadata_mode=True requires metadata",
            ),
            (
                DistributedDatasetConfig(seq_len=8, local_batch_size=1),
                {"metadata": metadata},
                "metadata_mode=False requires metadata_fn",
            ),
            (
                DistributedDatasetConfig(seq_len=8, local_batch_size=1),
                {},
                "metadata_mode=False requires metadata_fn",
            ),
        )
        for config, callbacks, message in cases:
            with self.subTest(metadata_mode=config.metadata_mode, message=message):
                sampler = build_dataset_batch_sampler(
                    total_samples=2, micro_batch_size=1, global_batch_size=1, dp_world_size=1, dp_rank=0,
                )
                with self.assertRaisesRegex(ValueError, message):
                    build_distributed_dataloader(
                        [0, 1], self._mesh(), config, batch_sampler=sampler, **callbacks,
                        device="cpu", cost_model=lambda sample_metadata: sample_metadata.cost,
                    )

    def test_native_sampler_rejects_local_balance_groups(self) -> None:
        """Feature: Local balancing groups.
        Description: Pass a local group size to the native sampler path.
        Expectation: Native samplers reject the local-only option explicitly.
        """
        config = DistributedDatasetConfig(
            seq_len=8, local_batch_size=1, balance_group_size=1, communication_backend="gloo",
        )
        sampler = build_dataset_batch_sampler(
            total_samples=1, micro_batch_size=1, global_batch_size=1, dp_world_size=1, dp_rank=0,
        )
        with self.assertRaisesRegex(ValueError, "only supported for local balancing"):
            build_distributed_dataloader(
                [0], self._mesh(), config, batch_sampler=sampler,
                metadata_fn=lambda _: SampleMetadata(1), device="cpu",
                cost_model=lambda metadata: metadata.cost,
            )


class TestDataTopology(unittest.TestCase):
    """Verify constructor ownership for a two-way DP, four-way MP mesh."""

    @staticmethod
    def _topology(global_rank: int) -> DataTopology:
        return DataTopology.from_layout(
            mesh_shape=(2, 4),
            mesh_dim_names=("dp", "mp"),
            rank_list=tuple(range(8)),
            global_rank=global_rank,
            dp_dim_names=("dp",),
        )

    def test_dp2_mp4_has_two_constructors_and_two_mp_groups(self) -> None:
        """Feature: Data topology derivation.
        Description: Build an eight-rank topology with two-way DP and four-way MP.
        Expectation: Ranks 0 and 4 own their respective model-replica batches.
        """
        expected_mp_groups = ((0, 1, 2, 3), (4, 5, 6, 7))

        for global_rank in range(8):
            with self.subTest(global_rank=global_rank):
                topology = self._topology(global_rank)
                expected_data_rank = global_rank // 4

                self.assertEqual(topology.data_parallel_size, 2)
                self.assertEqual(topology.constructor_ranks, (0, 4))
                self.assertEqual(topology.model_parallel_rank_groups, expected_mp_groups)
                self.assertEqual(topology.data_rank, expected_data_rank)
                self.assertEqual(topology.constructor_rank, (0, 4)[expected_data_rank])
                self.assertEqual(topology.model_parallel_ranks, expected_mp_groups[expected_data_rank])
                self.assertEqual(topology.is_constructor, global_rank in (0, 4))

    def test_rank_one_still_knows_every_group_but_consumes_rank_zero_batch(self) -> None:
        """Feature: Model-group topology visibility.
        Description: Inspect topology from a non-constructor rank.
        Expectation: The rank derives every group and consumes its constructor's batch.
        """
        topology = self._topology(1)

        self.assertEqual(topology.constructor_ranks, (0, 4))
        self.assertEqual(topology.model_parallel_rank_groups, ((0, 1, 2, 3), (4, 5, 6, 7)))
        self.assertEqual(topology.model_parallel_ranks, (0, 1, 2, 3))
        self.assertEqual(topology.constructor_rank, 0)
        self.assertFalse(topology.is_constructor)


if __name__ == "__main__":
    unittest.main()
