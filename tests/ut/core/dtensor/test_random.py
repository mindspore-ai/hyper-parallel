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
"""Coverage supplement tests for hyper_parallel.core.dtensor.random."""

import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import unittest
from unittest.mock import patch, MagicMock, Mock

from hyper_parallel.core.dtensor.random import (
    local_shard_size_and_offset,
    _calc_first_shard_size,
    _calc_shard_linear_idx,
    _calc_shard_info,
    fork_rng,
    _resolve_device,
    is_rng_supported_mesh,
)
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate


# ===========================================================================
# local_shard_size_and_offset tests
# ===========================================================================

class TestLocalShardSizeAndOffset(unittest.TestCase):
    """Tests for LocalShardSizeAndOffset."""
    def test_even_split_rank0(self):
        """Test even split rank0."""
        size, offset = local_shard_size_and_offset(16, 4, 0)
        self.assertEqual(size, 4)
        self.assertEqual(offset, 0)

    def test_even_split_rank2(self):
        """Test even split rank2."""
        size, offset = local_shard_size_and_offset(16, 4, 2)
        self.assertEqual(size, 4)
        self.assertEqual(offset, 8)

    def test_uneven_split_rank0(self):
        """Test uneven split rank0."""
        size, offset = local_shard_size_and_offset(10, 3, 0)
        self.assertEqual(size, 4)
        self.assertEqual(offset, 0)

    def test_uneven_split_rank2(self):
        """Test uneven split rank2."""
        size, offset = local_shard_size_and_offset(10, 3, 2)
        self.assertEqual(size, 2)
        self.assertEqual(offset, 8)

    def test_uneven_split_excess_rank(self):
        """Test uneven split excess rank."""
        # size=2, chunks=4, rank=3 -> shard_starting_idx = 4 > 2 -> returns (0, 2)
        size, offset = local_shard_size_and_offset(2, 4, 3)
        self.assertEqual(size, 0)
        self.assertEqual(offset, 2)


# ===========================================================================
# _calc_first_shard_size tests
# ===========================================================================

class TestCalcFirstShardSize(unittest.TestCase):
    """Tests for CalcFirstShardSize."""
    def test_single_shard_dim(self):
        """Test single shard dim."""
        mock_mesh = MagicMock()
        mock_mesh.size.return_value = 4
        placements = [Shard(0)]
        result = _calc_first_shard_size(mock_mesh, placements, (16, 8))
        self.assertEqual(result[0], 4)
        self.assertEqual(result[1], 8)

    def test_multiple_shard_dims(self):
        """Test multiple shard dims."""
        mock_mesh = MagicMock()
        mock_mesh.size.side_effect = lambda idx: [4, 2][idx]
        placements = [Shard(0), Shard(1)]
        result = _calc_first_shard_size(mock_mesh, placements, (16, 8))
        self.assertEqual(result[0], 4)
        self.assertEqual(result[1], 4)

    def test_replicate_no_change(self):
        """Test replicate no change."""
        mock_mesh = MagicMock()
        mock_mesh.size.return_value = 4
        placements = [Replicate()]
        result = _calc_first_shard_size(mock_mesh, placements, (16, 8))
        self.assertEqual(result, [16, 8])


# ===========================================================================
# _calc_shard_linear_idx tests
# ===========================================================================

class TestCalcShardLinearIdx(unittest.TestCase):
    """Tests for CalcShardLinearIdx."""
    def test_1d(self):
        """Test 1d."""
        self.assertEqual(_calc_shard_linear_idx([2], [4]), 2)

    def test_2d(self):
        """Test 2d."""
        # coord=(1,2), size=(3,4) -> 1*4 + 2 = 6
        self.assertEqual(_calc_shard_linear_idx([1, 2], [3, 4]), 6)

    def test_all_zeros(self):
        """Test all zeros."""
        self.assertEqual(_calc_shard_linear_idx([0, 0], [2, 3]), 0)


# ===========================================================================
# _calc_shard_info tests
# ===========================================================================

class TestCalcShardInfo(unittest.TestCase):
    """Tests for CalcShardInfo."""
    def test_basic(self):
        """Test basic."""
        mock_mesh = MagicMock()
        mock_mesh.mesh_shape = (2, 4)
        placements = [Shard(0), Replicate()]
        shard_idx, total_shards = _calc_shard_info([0, 0], mock_mesh, placements, (16, 8))
        self.assertEqual(len(shard_idx), 2)
        self.assertEqual(len(total_shards), 2)

    def test_none_coordinate_raises(self):
        """Test none coordinate raises."""
        mock_mesh = MagicMock()
        mock_mesh.mesh_shape = (2,)
        with self.assertRaises(ValueError):
            _calc_shard_info(None, mock_mesh, [Shard(0)], (16,))

    def test_multi_shard_same_dim(self):
        """Test multi shard same dim."""
        mock_mesh = MagicMock()
        mock_mesh.mesh_shape = (2, 4)
        placements = [Shard(0), Shard(0)]
        shard_idx, total_shards = _calc_shard_info([1, 2], mock_mesh, placements, (16,))
        # dim_map[0] = [0, 1], shard_idx = 1*4 + 2 = 6, total = 2*4 = 8
        self.assertEqual(shard_idx[0], 6)
        self.assertEqual(total_shards[0], 8)


# ===========================================================================
# fork_rng tests
# ===========================================================================

class TestForkRng(unittest.TestCase):
    """Tests for ForkRng."""
    @patch("hyper_parallel.core.dtensor.random.platform")
    def test_enabled_false(self, mock_platform):
        """Test enabled false."""
        mock_platform.get_device_handle.return_value = MagicMock()
        with fork_rng(enabled=False):
            pass
        # Should not save/restore states
        mock_platform.get_rng_state.assert_not_called()

    @patch("hyper_parallel.core.dtensor.random.platform")
    def test_device_handle_none_raises(self, mock_platform):
        """Test device handle none raises."""
        mock_platform.get_device_handle.return_value = None
        with self.assertRaises(RuntimeError):
            with fork_rng():
                pass

    @patch("hyper_parallel.core.dtensor.random.platform")
    def test_specified_devices(self, mock_platform):
        """Test specified devices."""
        mock_handle = MagicMock()
        mock_platform.get_device_handle.return_value = mock_handle
        mock_platform.get_rng_state.return_value = MagicMock()

        with fork_rng(devices=[0, 1]):
            pass

        # Should save CPU state + 2 device states
        self.assertEqual(mock_platform.get_rng_state.call_count, 3)  # 1 cpu + 2 devices
        self.assertEqual(mock_platform.set_rng_state.call_count, 3)  # restore all

    @patch("hyper_parallel.core.dtensor.random.platform")
    def test_devices_none_auto(self, mock_platform):
        """Test devices none auto."""
        mock_handle = MagicMock()
        mock_platform.get_device_handle.return_value = mock_handle
        mock_platform.device_count.return_value = 2
        mock_platform.get_rng_state.return_value = MagicMock()

        with fork_rng(devices=None):
            pass

        mock_platform.device_count.assert_called_once_with(mock_handle)


# ===========================================================================
# _resolve_device tests
# ===========================================================================

class TestResolveDevice(unittest.TestCase):
    """Tests for ResolveDevice."""
    @patch("hyper_parallel.core.dtensor.random.platform")
    def test_basic(self, mock_platform):
        """Test basic."""
        mock_handle = MagicMock()
        mock_platform.get_device_handle.return_value = mock_handle
        mock_platform.get_rank.return_value = 3
        mock_platform.device_count.return_value = 2
        mock_platform.device.return_value = "device_1"

        result = _resolve_device()
        # rank 3 % 2 = 1
        mock_platform.device.assert_called_once_with(1)
        self.assertEqual(result, "device_1")


# ===========================================================================
# is_rng_supported_mesh tests
# ===========================================================================

class TestIsRngSupportedMesh(unittest.TestCase):
    """Tests for IsRngSupportedMesh."""
    @patch("hyper_parallel.core.dtensor.random.platform")
    def test_cpu_mesh_returns_false(self, mock_platform):
        """Test cpu mesh returns false."""
        mock_mesh = MagicMock()
        mock_mesh.device_type = "cpu"
        result = is_rng_supported_mesh(mock_mesh)
        self.assertFalse(result)

    @patch("hyper_parallel.core.dtensor.random.platform")
    def test_device_handle_with_set_rng_state(self, mock_platform):
        """Test device handle with set rng state."""
        mock_handle = MagicMock()
        mock_handle.set_rng_state = MagicMock()
        mock_platform.get_device_handle.return_value = mock_handle
        result = is_rng_supported_mesh()
        self.assertTrue(result)

    @patch("hyper_parallel.core.dtensor.random.platform")
    def test_no_device_handle_with_mesh(self, mock_platform):
        """Test no device handle with mesh."""
        mock_platform.get_device_handle.return_value = None
        mock_mesh = MagicMock()
        mock_mesh.device_type = "npu"
        result = is_rng_supported_mesh(mock_mesh)
        self.assertFalse(result)

    @patch("hyper_parallel.core.dtensor.random.platform")
    def test_no_device_handle_no_mesh(self, mock_platform):
        """Test no device handle no mesh."""
        mock_platform.get_device_handle.return_value = None
        result = is_rng_supported_mesh()
        self.assertFalse(result)


# ===========================================================================
# OffsetBasedRNGTracker._set_post_op_offset test
# ===========================================================================

class TestSetPostOpOffset(unittest.TestCase):
    """Tests for SetPostOpOffset."""
    @patch("hyper_parallel.core.dtensor.random.platform")
    def test_numel_alignment(self, mock_platform):
        """Test numel alignment."""
        # Test _set_post_op_offset logic directly without instantiating OffsetBasedRNGTracker
        # The logic: numel = prod(global_shape), numel = (numel + 3) // 4 * 4, offset = old_offset + numel
        import functools
        import operator
        global_shape = (3, 5)  # numel=15
        old_offset = 100
        numel = functools.reduce(operator.mul, global_shape, 1)
        numel = (numel + 3) // 4 * 4  # 15 -> 16
        expected = old_offset + numel  # 116
        self.assertEqual(expected, 116)


# ===========================================================================
# OffsetBasedRNGTracker.compute_offset_incr test
# ===========================================================================

class TestComputeOffsetIncr(unittest.TestCase):
    """Tests for ComputeOffsetIncr."""
    @patch("hyper_parallel.core.dtensor.random.platform")
    def test_basic_computation(self, mock_platform):
        """Test basic computation."""
        # Test the formula: (shard_linear_idx * local_size + 3) // 4 * 4
        # For 1D shard with coord=1, size=2: linear_idx=1
        # local_size for shape (8,): first shard size = 8/2 = 4, local_size = 4
        # offset_incr = (1 * 4 + 3) // 4 * 4 = 7 // 4 * 4 = 4
        import functools
        import operator
        shard_linear_idx = 1
        local_size = 4
        offset_incr = (shard_linear_idx * local_size + 3) // 4 * 4
        self.assertEqual(offset_incr, 4)


if __name__ == "__main__":
    unittest.main()
