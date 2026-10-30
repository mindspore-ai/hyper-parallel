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
"""Geometry checks independent of accelerator execution or process groups."""

import itertools
import math
import random
import unittest
from types import SimpleNamespace

from hyper_parallel.components.checkpoint._weight_check_geometry import (
    box_union_volume, local_block_box, tile_box,
)


class TestWeightCheckGeometry(unittest.TestCase):
    """Verify shard intersections and coverage against enumerated coordinates."""

    def test_tiles_are_bounded_disjoint_and_complete(self) -> None:
        """Feature: Tiling; Description: Split arbitrary rectangles; Expectation: Exact disjoint coverage."""
        for shape in ((), (0,), (1,), (11,), (3, 7), (2, 3, 7)):
            for limit in (1, 2, 8, 64):
                with self.subTest(shape=shape, limit=limit):
                    points = []
                    for offsets, lengths in tile_box((0,) * len(shape), shape, limit):
                        self.assertLessEqual(math.prod(lengths), limit)
                        points.extend(itertools.product(*(range(start, start + size)
                                                          for start, size in zip(offsets, lengths))))
                    expected = set(itertools.product(*(range(size) for size in shape)))
                    self.assertEqual(set(points), expected)
                    self.assertEqual(len(points), len(expected))

    def test_union_volume_handles_replicas_overlaps_and_scalars(self) -> None:
        """Feature: Coverage.

        Description: Compare overlapping boxes with coordinate enumeration.
        Expectation: Count replicated or overlapping elements only once.
        """
        generator = random.Random(42)
        for dimensions in (1, 2, 3):
            for _ in range(50):
                boxes = []
                points = set()
                for _ in range(8):
                    offsets = tuple(generator.randrange(4) for _ in range(dimensions))
                    lengths = tuple(generator.randrange(4) for _ in range(dimensions))
                    boxes.append((offsets, lengths))
                    points.update(itertools.product(*(range(start, start + size)
                                                      for start, size in zip(offsets, lengths))))
                self.assertEqual(box_union_volume(boxes + boxes), len(points))
        self.assertEqual(box_union_volume([((), ()), ((), ())]), 1)
        self.assertEqual(box_union_volume([]), 0)

    def test_transposed_expert_shard_intersection(self) -> None:
        """Feature: Expert layouts.

        Description: Undo expert selection and transpose.
        Expectation: Recover the correct HF rectangle.
        """
        block = SimpleNamespace(
            offsets=(0, 0), lengths=(4, 6), base=(2, 0, 0), steps=((0, 0, 1), (0, 1, 0)),
        )
        self.assertEqual(local_block_box(block, (2, 1, 0), (1, 3, 4)), ((0, 1), (4, 3)))
        self.assertIsNone(local_block_box(block, (3, 0, 0), (1, 6, 4)))
        self.assertIsNone(local_block_box(block, (2, 6, 0), (1, 0, 4)))

    def test_scalar_selection_checks_fixed_model_coordinates(self) -> None:
        """Feature: Scalar weights; Description: Select a fixed vector entry; Expectation: Check shard membership."""
        block = SimpleNamespace(offsets=(), lengths=(), base=(3,), steps=())
        self.assertEqual(local_block_box(block, (2,), (2,)), ((), ()))
        self.assertIsNone(local_block_box(block, (0,), (2,)))

    def test_nonrectangular_mapping_fails_explicitly(self) -> None:
        """Feature: Unsupported conversion; Description: Flatten two axes; Expectation: Reject instead of skipping."""
        block = SimpleNamespace(offsets=(0, 0), lengths=(2, 3), base=(0,), steps=((3,), (1,)))
        with self.assertRaisesRegex(ValueError, "non-rectangular"):
            local_block_box(block, (0,), (6,))
