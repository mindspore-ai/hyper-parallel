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
"""Unit tests for fully_shard list support on MindSpore platform (no NPU required)."""
import os
import unittest
from unittest.mock import patch, MagicMock

import pytest

# Skip entire module if mindspore is not installed (avoids import failure)
pytest.importorskip("mindspore")

# Force mindspore platform before any hyper_parallel imports
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from mindspore import nn as ms_nn

from hyper_parallel.core.fully_shard.api import (
    _get_root_modules,
    _validate_module_for_fully_shard,
    fully_shard,
)
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.platform import PlatformType


def _default_mp_policy():
    """Create default MixedPrecisionPolicy for tests."""
    return MixedPrecisionPolicy(
        param_dtype=None,
        reduce_dtype=None,
        output_dtype=None,
    )


def _make_parent_cell():
    """Create a parent Cell with child."""

    class ParentCell(ms_nn.Cell):
        """Parent cell containing child cell."""

        def __init__(self):
            super().__init__()
            self.child = ms_nn.Dense(4, 4)

        def construct(self, x):
            return self.child(x)

    return ParentCell


class TestGetRootModulesMindSpore(unittest.TestCase):
    """Unit tests for _get_root_modules with MindSpore Cells."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

    def test_sibling_cells_both_roots(self):
        """Two sibling cells (neither contains the other) are both roots."""
        dense1 = ms_nn.Dense(4, 4)
        dense2 = ms_nn.Dense(4, 4)
        roots = _get_root_modules([dense1, dense2])
        self.assertEqual(len(roots), 2)
        self.assertIn(dense1, roots)
        self.assertIn(dense2, roots)

    def test_parent_child_only_parent_root(self):
        """When list contains parent and child, only parent is root."""
        ParentCell = _make_parent_cell()
        parent = ParentCell()
        child = parent.child
        roots = _get_root_modules([parent, child])
        self.assertEqual(len(roots), 1)
        self.assertIn(parent, roots)
        self.assertNotIn(child, roots)

    def test_single_cell_is_root(self):
        """Single cell in list is root."""
        cell = ms_nn.Dense(4, 4)
        roots = _get_root_modules([cell])
        self.assertEqual(len(roots), 1)
        self.assertIn(cell, roots)

    def test_three_siblings_all_roots(self):
        """Three sibling cells are all roots."""
        c1, c2, c3 = ms_nn.Dense(4, 4), ms_nn.Dense(4, 4), ms_nn.Dense(4, 4)
        roots = _get_root_modules([c1, c2, c3])
        self.assertEqual(len(roots), 3)
        self.assertIn(c1, roots)
        self.assertIn(c2, roots)
        self.assertIn(c3, roots)


class TestValidateModuleForFullyShardMindSpore(unittest.TestCase):
    """Unit tests for _validate_module_for_fully_shard with MindSpore."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

    def test_single_cell_valid(self):
        """Single Cell passes validation."""
        cell = ms_nn.Dense(4, 4)
        _validate_module_for_fully_shard(cell, PlatformType.MINDSPORE)

    def test_list_of_cells_valid(self):
        """List of Cells passes validation (MindSpore supports list)."""
        cells = [ms_nn.Dense(4, 4), ms_nn.Dense(4, 4)]
        _validate_module_for_fully_shard(cells, PlatformType.MINDSPORE)

    def test_empty_list_raises(self):
        """Empty list raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _validate_module_for_fully_shard([], PlatformType.MINDSPORE)
        self.assertIn("empty list", str(ctx.exception))

    def test_list_with_non_cell_raises(self):
        """List containing non-Cell raises ValueError."""
        cell = ms_nn.Dense(4, 4)
        with self.assertRaises(ValueError) as ctx:
            _validate_module_for_fully_shard([cell, 1], PlatformType.MINDSPORE)
        self.assertIn("index 1", str(ctx.exception))

    def test_non_cell_raises(self):
        """Non-Cell input raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _validate_module_for_fully_shard("not a cell", PlatformType.MINDSPORE)
        self.assertIn("nn.cell", str(ctx.exception))


class TestFullyShardListAPIMindSpore(unittest.TestCase):
    """Unit tests for fully_shard list support on MindSpore (mocked to avoid NPU/dist)."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

    def _create_mock_mesh(self):
        """Create mock DeviceMesh to avoid distributed init."""
        mesh = MagicMock()
        mesh.ndim = 1
        return mesh

    @patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh")
    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_single_cell_returns_cell(self, mock_platform, mock_get_device):
        """fully_shard with single cell returns the same cell (in-place)."""
        mock_platform.platform_type = PlatformType.MINDSPORE
        mock_get_device.return_value = "npu"
        mesh = self._create_mock_mesh()
        cell = ms_nn.Dense(4, 4)
        with patch(
            "hyper_parallel.core.fully_shard.api.HSDPModule.hsdp_init",
            lambda self, *a, **k: None,
        ):
            result = fully_shard(
                cell,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=_default_mp_policy(),
            )
        self.assertIs(result, cell)
        self.assertIsInstance(result, ms_nn.Cell)

    @patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh")
    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_list_returns_list(self, mock_platform, mock_get_device):
        """fully_shard with list returns the same list (in-place)."""
        mock_platform.platform_type = PlatformType.MINDSPORE
        mock_get_device.return_value = "npu"
        mesh = self._create_mock_mesh()
        dense1 = ms_nn.Dense(4, 4)
        dense2 = ms_nn.Dense(4, 4)
        cells_list = [dense1, dense2]
        with patch.object(
            type(dense1),
            "hsdp_init",
            lambda self, *a, **k: None,
            create=True,
        ):
            with patch(
                "hyper_parallel.core.fully_shard.api.HSDPModule.hsdp_init",
                lambda self, *a, **k: None,
            ):
                result = fully_shard(
                    cells_list,
                    mesh=mesh,
                    reshard_after_forward=True,
                    mp_policy=_default_mp_policy(),
                )
        self.assertIs(result, cells_list)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    @patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh")
    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_list_root_filtering(self, mock_platform, mock_get_device):
        """fully_shard with parent+child list filters to root only."""
        mock_platform.platform_type = PlatformType.MINDSPORE
        mock_get_device.return_value = "npu"
        mesh = self._create_mock_mesh()
        ParentCell = _make_parent_cell()
        parent = ParentCell()
        child = parent.child
        cells_list = [parent, child]
        with patch(
            "hyper_parallel.core.fully_shard.api.HSDPModule.hsdp_init",
            lambda self, *a, **k: None,
        ):
            result = fully_shard(
                cells_list,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=_default_mp_policy(),
            )
        self.assertIs(result, cells_list)
        self.assertEqual(len(result), 2)

    @patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh")
    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_two_sibling_cells_second_params_in_hsdp_state(
        self, mock_platform, mock_get_device
    ):
        """fully_shard([cell1, cell2]) puts both cells' params into hsdp_state.hsdp_params."""
        mock_platform.platform_type = PlatformType.MINDSPORE
        mock_get_device.return_value = "CPU"
        mesh = self._create_mock_mesh()
        cell1 = ms_nn.Dense(4, 4)
        cell2 = ms_nn.Dense(4, 4)
        result = fully_shard(
            [cell1, cell2],
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=_default_mp_policy(),
        )
        self.assertIs(result, [cell1, cell2])
        self.assertTrue(hasattr(cell1, "hsdp_scheduler"), "cell1 should have hsdp_scheduler")
        hsdp_state = cell1.hsdp_scheduler.hsdp_state
        self.assertEqual(len(hsdp_state.modules), 2)
        self.assertIn(cell1, hsdp_state.modules)
        self.assertIn(cell2, hsdp_state.modules)
        # Each Dense(4,4) has weight and bias -> 4 params total; both cells must be in hsdp_params
        self.assertGreaterEqual(
            len(hsdp_state.hsdp_params),
            4,
            "hsdp_state.hsdp_params must include params from both cells (2 Dense -> 4 params)",
        )

    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_empty_list_raises(self, mock_platform):
        """fully_shard with empty list raises ValueError."""
        mock_platform.platform_type = PlatformType.MINDSPORE
        mesh = self._create_mock_mesh()
        with self.assertRaises(ValueError) as ctx:
            fully_shard(
                [],
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=_default_mp_policy(),
            )
        self.assertIn("empty list", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
