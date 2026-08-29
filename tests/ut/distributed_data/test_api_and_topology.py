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


class TestDistributedDataPublicApi(unittest.TestCase):
    """Verify that the public API is expressed in training-facing terms."""

    def test_config_uses_sequence_and_local_batch_sizing(self) -> None:
        """The config must not expose the removed raw-read sizing concepts."""
        field_names = {field.name for field in fields(DistributedDatasetConfig)}

        self.assertIn("seq_len", field_names)
        self.assertIn("local_batch_size", field_names)
        self.assertNotIn("raw_sample_size", field_names)
        self.assertNotIn("micro_batch_num", field_names)

        config = DistributedDatasetConfig(seq_len=32_768, local_batch_size=4)
        self.assertFalse(hasattr(config, "raw_sample_size"))
        self.assertFalse(hasattr(config, "micro_batch_num"))

    def test_builder_accepts_a_raw_dataset_and_constructor_callbacks(self) -> None:
        """The builder contract must separate raw loading, packing, and collation."""
        signature = inspect.signature(build_distributed_dataloader)
        parameters = signature.parameters

        self.assertEqual(tuple(parameters)[:3], ("dataset", "mesh", "config"))
        self.assertEqual(parameters["metadata_fn"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters["pack_fn"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters["collate_fn"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(parameters["pack_fn"].default)
        self.assertIsNone(parameters["collate_fn"].default)
        self.assertNotIn("raw_sample_size", parameters)
        self.assertNotIn("micro_batch_num", parameters)

    def test_public_default_callbacks_preserve_nested_batch_boundaries(self) -> None:
        """Default construction returns immutable bins and an immutable local batch."""
        raw_samples = [{"id": 0}, {"id": 1}]
        packed = distributed_data.default_pack_fn(raw_samples, seq_len=32)

        self.assertEqual(packed, ({"id": 0}, {"id": 1}))
        self.assertIsInstance(packed, tuple)
        collated = distributed_data.default_collate_fn([packed, ({"id": 2},)])
        self.assertEqual(collated, (({"id": 0}, {"id": 1}), ({"id": 2},)))
        self.assertIsInstance(collated, tuple)

    def test_buffer_multiplier_must_be_finite(self) -> None:
        """Invalid read-ahead targets must fail before distributed collectives."""
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "buffer_size_multiplier"):
                    DistributedDatasetConfig(
                        seq_len=32,
                        local_batch_size=1,
                        buffer_size_multiplier=value,
                    )

    def test_arbitrarily_large_integer_buffer_multiplier_is_valid(self) -> None:
        """Integer validation must not overflow by coercing an exact integer to float."""
        huge_multiplier = 10 ** 1000

        config = DistributedDatasetConfig(
            seq_len=32,
            local_batch_size=1,
            buffer_size_multiplier=huge_multiplier,
        )

        self.assertEqual(config.buffer_size_multiplier, huge_multiplier)


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
        """Ranks 0 and 4 construct data for their respective MP replicas."""
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
        """Every rank derives groups globally even when it is not their member."""
        topology = self._topology(1)

        self.assertEqual(topology.constructor_ranks, (0, 4))
        self.assertEqual(topology.model_parallel_rank_groups, ((0, 1, 2, 3), (4, 5, 6, 7)))
        self.assertEqual(topology.model_parallel_ranks, (0, 1, 2, 3))
        self.assertEqual(topology.constructor_rank, 0)
        self.assertFalse(topology.is_constructor)


if __name__ == "__main__":
    unittest.main()
