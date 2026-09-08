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
"""Unit tests for ``hyper_parallel.core.utils.shape_utils``.

``compute_local_shape_and_global_offset`` is balanced shard math: given a
tensor shape, a mesh, and a placement, return the per-rank local shape.
Coverage pins:

1. Even split — ``global_size % num_devices == 0`` → every rank gets the
   even quotient on the sharded axis.
2. Uneven split — first ``remainder`` ranks get ``quotient + 1``, the rest
   get ``quotient`` (balanced ``Shard`` semantics).
3. ``"None"`` axis entries are no-ops (replicated dim left untouched).
4. Multi-dim placement — sharding two tensor dims via two different mesh
   axes does not interfere; chunk math is applied per dim independently.

The mesh / layout chain is mocked so the test stays hardware-agnostic.
"""
# pylint: disable=protected-access
import os
import unittest
from typing import Any
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.utils import shape_utils  # pylint: disable=wrong-import-position


class _FakeShardMesh:
    """Mesh stand-in that returns canned ``(num_devices, local_rank)`` per axis.

    Named ``_FakeShardMesh`` to distinguish it from other mesh test doubles.

    Args:
        axis_info: Map from axis name (e.g. ``"dp"``) to ``(num_devices, local_rank)``.
    """

    def __init__(self, axis_info: dict[str, tuple[int, int]]) -> None:
        """Store mesh size and local rank information by axis."""
        self._info = axis_info

    def get_device_num_along_axis(self, axis: str) -> int:
        """Return the configured device count for an axis."""
        return self._info[axis][0]

    def get_local_rank(self, axis: str) -> int:
        """Return the configured local rank for an axis."""
        return self._info[axis][1]


class _FakeLayout:
    """Stub layout exposing only the fields ``compute_local_shape_and_global_offset`` reads."""

    def __init__(self, alias_tensor_map: Any, mesh: _FakeShardMesh) -> None:
        """Store the tensor map and mesh exposed to the function under test."""
        self.alias_tensor_map = alias_tensor_map
        self.mesh = mesh

    def placement_to_tensor_map(self, ndim: int) -> None:  # pylint: disable=unused-argument
        """Real Layout populates ``alias_tensor_map`` from Placement objects here.

        We pre-populate ``alias_tensor_map`` in the constructor, so this is a no-op.
        """


def _make_total_layout(layout_to_return):
    """Build a ``total_layout`` callable that returns ``layout_to_return`` for any call."""

    def _callable(*_args, **_kwargs):
        return layout_to_return
    return _callable


class TestComputeLocalShape(unittest.TestCase):
    """Pin the chunk-math contract used to derive every DTensor's local shape."""

    def _run(self, global_shape, alias_tensor_map, axis_info, placement, *, alias=True):
        """Drive ``compute_local_shape_and_global_offset`` with mocked layout/mesh."""
        layout = _FakeLayout(alias_tensor_map=alias_tensor_map, mesh=_FakeShardMesh(axis_info))
        # Patch Layout.from_device_mesh to return a "total_layout" callable, and
        # patch _is_alias_placements to flip between the two code paths.
        with patch.object(shape_utils, "Layout") as mock_layout_cls, \
             patch("hyper_parallel.core.dtensor.dtensor._is_alias_placements", return_value=alias):
            mock_layout_cls.from_device_mesh.return_value = _make_total_layout(layout)
            return shape_utils.compute_local_shape_and_global_offset(
                global_shape, device_mesh=None, placement=placement,
            )

    def test_even_split_along_single_axis(self):
        """``global=8 / dp=4`` → every rank gets 2 on that axis."""
        for rank in range(4):
            with self.subTest(rank=rank):
                slice_shape = self._run(
                    global_shape=(8, 16),
                    alias_tensor_map=("dp", "None"),
                    axis_info={"dp": (4, rank)},
                    placement=("dp", "None"),
                    alias=True,
                )
                self.assertEqual(slice_shape, [2, 16], (
                    f"Even split should give 2 on every rank, got {slice_shape} on rank={rank}"
                ))

    def test_uneven_split_uses_balanced_shard_geometry(self):
        """``global=10 / num_devices=4`` → ranks 0,1 get 3 each; ranks 2,3 get 2 each."""
        expected_per_rank = {0: 3, 1: 3, 2: 2, 3: 2}
        for rank, expected in expected_per_rank.items():
            with self.subTest(rank=rank):
                slice_shape = self._run(
                    global_shape=(10,),
                    alias_tensor_map=("tp",),
                    axis_info={"tp": (4, rank)},
                    placement=("tp",),
                    alias=True,
                )
                self.assertEqual(slice_shape, [expected], (
                    f"Uneven split mismatch on rank={rank}: "
                    f"expected={expected}, got={slice_shape[0]}"
                ))

    def test_ceil_chunk_keeps_fixed_size_until_trailing_shard(self):
        """FSDP ceil-chunk geometry differs from balanced Shard geometry."""
        expected_geometries = {
            (10, 4): ((3, 0), (3, 3), (3, 6), (1, 9)),
            (6, 4): ((2, 0), (2, 2), (2, 4), (0, 6)),
        }
        for (global_size, shard_count), rank_geometries in expected_geometries.items():
            for shard_rank, (expected_size, expected_offset) in enumerate(rank_geometries):
                with self.subTest(
                    global_size=global_size,
                    shard_count=shard_count,
                    shard_rank=shard_rank,
                ):
                    local_shape, global_offset = (
                        shape_utils.compute_local_shape_and_global_offset_by_ceil_chunk(
                            (global_size, 3),
                            shard_dim=0,
                            shard_count=shard_count,
                            shard_rank=shard_rank,
                        )
                    )
                    self.assertEqual(local_shape, [expected_size, 3])
                    self.assertEqual(global_offset, [expected_offset, 0])

    def test_replicate_axis_keeps_full_shape(self):
        """``alias_tensor_map`` entries equal to ``\"None\"`` leave the dim untouched."""
        slice_shape = self._run(
            global_shape=(8, 12),
            alias_tensor_map=("None", "None"),
            axis_info={},
            placement=("None", "None"),
            alias=True,
        )
        self.assertEqual(slice_shape, [8, 12], (
            f"Replicated tensor must keep full shape, got {slice_shape}"
        ))

    def test_multi_dim_shard_applies_per_dim_independently(self):
        """Two tensor dims sharded along two mesh axes: math runs per-dim, no cross-talk."""
        # dim 0 along "dp" (4 devices, rank 1 → falls below remainder=2, gets 3)
        # dim 1 along "tp" (2 devices, rank 0 → even, gets 6)
        slice_shape = self._run(
            global_shape=(10, 12),
            alias_tensor_map=("dp", "tp"),
            axis_info={"dp": (4, 1), "tp": (2, 0)},
            placement=("dp", "tp"),
            alias=True,
        )
        self.assertEqual(slice_shape, [3, 6], (
            f"Multi-dim shard mismatch: expected [3, 6], got {slice_shape}"
        ))

    def test_tuple_alias_runs_all_subaxes_sequentially(self):
        """A nested-tuple entry (e.g. ``(\"dp\", \"cp\")``) chunks the dim once per sub-axis."""
        # dim 0 sharded along both "dp" (2 devices) and "cp" (2 devices).
        # Sequential: 8 / 2 = 4 (dp step), then 4 / 2 = 2 (cp step).
        slice_shape = self._run(
            global_shape=(8,),
            alias_tensor_map=(("dp", "cp"),),
            axis_info={"dp": (2, 0), "cp": (2, 0)},
            placement=(("dp", "cp"),),
            alias=True,
        )
        self.assertEqual(slice_shape, [2], (
            f"Nested-tuple alias must apply both axes sequentially, got {slice_shape}"
        ))


if __name__ == "__main__":
    unittest.main()
