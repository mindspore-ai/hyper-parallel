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

from hyper_parallel import distributed_data
from hyper_parallel.distributed_data import DistributedDatasetConfig, build_distributed_dataloader
from hyper_parallel.distributed_data.topology import DataTopology
from tests.common.mark_utils import arg_mark


class TestDistributedDataPublicApi(unittest.TestCase):
    """Verify that the public API is expressed in training-facing terms."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_config_uses_sequence_and_local_batch_sizing(self) -> None:
        """Feature: Distributed data configuration.
        Description: Inspect sequence, local-batch, buffering, and sharding fields.
        Expectation: Training-facing fields replace removed raw-read sizing concepts.
        """
        field_names = {field.name for field in fields(DistributedDatasetConfig)}

        self.assertIn("seq_len", field_names)
        self.assertIn("local_batch_size", field_names)
        self.assertIn("double_buffer", field_names)
        self.assertIn("dataset_already_sharded", field_names)
        self.assertNotIn("raw_sample_size", field_names)
        self.assertNotIn("micro_batch_num", field_names)

        config = DistributedDatasetConfig(seq_len=32_768, local_batch_size=4)
        self.assertFalse(config.double_buffer)
        self.assertFalse(config.dataset_already_sharded)
        self.assertFalse(hasattr(config, "raw_sample_size"))
        self.assertFalse(hasattr(config, "micro_batch_num"))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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
        self.assertEqual(parameters["communication_device"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(parameters["metadata_fn"].default)
        self.assertIsNone(parameters["metadata"].default)
        self.assertIsNone(parameters["pack_fn"].default)
        self.assertIsNone(parameters["collate_fn"].default)
        self.assertNotIn("raw_sample_size", parameters)
        self.assertNotIn("micro_batch_num", parameters)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_buffer_multiplier_must_be_finite(self) -> None:
        """Feature: Read-ahead buffer sizing.
        Description: Configure non-finite buffer size multipliers.
        Expectation: Invalid targets fail before distributed collectives.
        """
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "buffer_size_multiplier"):
                    DistributedDatasetConfig(
                        seq_len=32,
                        local_batch_size=1,
                        buffer_size_multiplier=value,
                    )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_arbitrarily_large_integer_buffer_multiplier_is_valid(self) -> None:
        """Feature: Exact integer buffer sizing.
        Description: Configure an arbitrarily large integer multiplier.
        Expectation: Validation accepts the exact integer without float overflow.
        """
        huge_multiplier = 10 ** 1000

        config = DistributedDatasetConfig(
            seq_len=32,
            local_batch_size=1,
            buffer_size_multiplier=huge_multiplier,
        )

        self.assertEqual(config.buffer_size_multiplier, huge_multiplier)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_control_backend_must_support_cpu_object_collectives(self) -> None:
        """Feature: Control-plane backend validation.
        Description: Configure an accelerator-only backend for object collectives.
        Expectation: The invalid control backend is rejected.
        """
        with self.assertRaisesRegex(ValueError, "cpu_backend must support CPU tensors"):
            DistributedDatasetConfig(seq_len=32, local_batch_size=1, cpu_backend="hccl")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_double_buffer_must_be_boolean(self) -> None:
        """Feature: Host double buffering.
        Description: Configure the overlap switch with a truthy non-boolean value.
        Expectation: Configuration validation rejects the invalid value.
        """
        with self.assertRaisesRegex(ValueError, "double_buffer must be boolean"):
            DistributedDatasetConfig(seq_len=32, local_batch_size=1, double_buffer=1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_dataset_already_sharded_must_be_boolean(self) -> None:
        """Feature: Dataset Reader sharding.
        Description: Configure the reader-stride switch with a truthy non-boolean value.
        Expectation: Configuration validation rejects the invalid value.
        """
        with self.assertRaisesRegex(ValueError, "dataset_already_sharded must be boolean"):
            DistributedDatasetConfig(seq_len=32, local_batch_size=1, dataset_already_sharded=1)


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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
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
