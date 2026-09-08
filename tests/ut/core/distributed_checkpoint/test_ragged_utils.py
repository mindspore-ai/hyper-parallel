# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Unit tests for RaggedShard distributed checkpoint geometry."""
import os
from unittest.mock import patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.distributed_checkpoint.ragged_utils import (
    _decompose_flat_interval,
    compute_ragged_boxes,
    create_ragged_write_items,
    get_ragged_box_tensor,
)
from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardSavePlanner
from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import RaggedShard
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


def _make_rank_zero_ragged_tensor() -> DTensor:
    """Build a rank-zero shard whose flat interval crosses an N-D row boundary."""
    _DEVICE_MESH_MAP.clear()
    EXISTING_COMM_GROUPS.clear()
    with patch(
        "hyper_parallel.core.dtensor.device_mesh.platform.get_rank",
        return_value=0,
    ):
        mesh = Layout((2,), ("ragged",), init_backend=False).mesh
        return DTensor.from_local(
            torch.arange(48),
            mesh,
            (RaggedShard(dims=(0, 1), local_units=(1, 3)),),
            shape=(6, 4, 8),
        )


def test_decompose_flat_interval_preserves_row_major_order():
    """Split one contiguous flat interval into ordered regular boxes."""
    boxes = _decompose_flat_interval((3, 4), 2, 8)

    assert boxes == (
        ((0, 2), (1, 2)),
        ((1, 0), (1, 4)),
    )


def test_decompose_flat_interval_exhaustively_preserves_values():
    """Every small flat interval is represented once and in row-major order."""
    for shape in ((2, 3), (2, 3, 2), (3, 2, 2)):
        full = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
        for flat_start in range(full.numel() + 1):
            for flat_end in range(flat_start, full.numel() + 1):
                pieces = []
                for offsets, sizes in _decompose_flat_interval(shape, flat_start, flat_end):
                    slices = tuple(
                        slice(offset, offset + size)
                        for offset, size in zip(offsets, sizes)
                    )
                    pieces.extend(full[slices].reshape(-1).tolist())
                assert pieces == list(range(flat_start, flat_end))


def test_compute_ragged_boxes_tracks_local_flat_ranges():
    """Map each regular box back to a contiguous segment of local flat storage."""
    tensor = _make_rank_zero_ragged_tensor()

    boxes = compute_ragged_boxes(tensor)

    assert boxes == (
        ((0, 0, 0), (1, 4, 8), 0, 32),
        ((1, 0, 0), (1, 2, 8), 32, 48),
    )


def test_ragged_write_items_and_box_views_use_logical_offsets():
    """Create one WriteItem and matching local tensor view for every box."""
    tensor = _make_rank_zero_ragged_tensor()

    items = create_ragged_write_items("weight", tensor)
    pieces = [get_ragged_box_tensor(tensor, item.index) for item in items]

    assert [item.index.offset for item in items] == [(0, 0, 0), (1, 0, 0)]
    assert [item.tensor_data["chunk"].sizes for item in items] == [
        (1, 4, 8),
        (1, 2, 8),
    ]
    assert tuple(pieces[0].shape) == (1, 4, 8)
    assert tuple(pieces[1].shape) == (1, 2, 8)
    assert torch.equal(pieces[0].reshape(-1), torch.arange(32))
    assert torch.equal(pieces[1].reshape(-1), torch.arange(32, 48))


def test_ragged_state_dict_keeps_save_plan_cache_enabled():
    """Ragged geometry can use the standard save plan cache."""
    planner = StandardSavePlanner(enable_plan_caching=True)
    planner.configure_planner({"weight": _make_rank_zero_ragged_tensor()})
    assert planner._enable_plan_caching

    planner.configure_planner({"weight": torch.ones(2, 3)})
    assert planner._enable_plan_caching


def test_ragged_state_dict_without_collectives_keeps_cache_disabled():
    """Without plan collectives, RaggedShard remains uncached like other tensors."""
    planner = StandardSavePlanner(enable_plan_caching=True)
    planner.configure_planner(
        {"weight": _make_rank_zero_ragged_tensor()},
        use_collectives=False,
    )
    assert not planner._enable_plan_caching
