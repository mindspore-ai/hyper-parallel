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
"""Coverage supplement tests for hyper_parallel.core.dtensor.tensor_redistribution."""

import os
import weakref
from types import SimpleNamespace


import unittest
from unittest.mock import patch, MagicMock, Mock

import torch
import numpy as np

from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.dtensor import DTensor, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.layout import Layout, infer_slice_area_by_rank
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard, StridedShard
from hyper_parallel.core.dtensor.tensor_redistribution import TensorRedistribution
from hyper_parallel.core.fully_shard.state_dict_utils import get_model_state_dict
from hyper_parallel.core.utils.communication import EXISTING_COMM_GROUPS
from tests.common.mark_utils import arg_mark


def _setup_mock_dm_platform(platform_mock):
    """Configure a mock platform for DeviceMesh construction."""
    platform_mock.get_rank.return_value = 0
    platform_mock.get_world_size.return_value = 8
    platform_mock.Tensor = torch.Tensor
    mock_group = Mock()
    mock_group.group_name = "mock_group"

    def _split_group_side_effect(split_ranks, parent_pg=None, timeout=None, pg_options=None, group_desc=None):
        """ split group side effect."""
        for sr in split_ranks:
            key = str(tuple(sorted(sr)))
            EXISTING_COMM_GROUPS[key] = mock_group
        return mock_group

    platform_mock.split_group.side_effect = _split_group_side_effect
    platform_mock.get_created_group.return_value = None
    platform_mock.get_process_group_ranks.return_value = list(range(8))

    def mock_tensor_to_numpy(tensor: object) -> object:
        """Mock tensor to numpy."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    platform_mock.tensor_to_numpy.side_effect = mock_tensor_to_numpy


class _MockedTestCase(unittest.TestCase):
    """Base class that patches both device_mesh and tensor_redistribution platforms."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.patcher_dm = patch("hyper_parallel.core.dtensor.device_mesh.dist", MagicMock())
        self.patcher_dm_tensor = patch("hyper_parallel.core.dtensor.device_mesh.Tensor", torch.Tensor)
        self.patcher_tr = patch("hyper_parallel.core.dtensor.tensor_redistribution._utils")

        self.mock_dm_platform = self.patcher_dm.start()
        self.patcher_dm_tensor.start()
        self.mock_tr_platform = self.patcher_tr.start()

        _setup_mock_dm_platform(self.mock_dm_platform)
        # Also setup tr platform for rank
        self.mock_tr_platform.get_rank.return_value = 0

        _DEVICE_MESH_MAP.clear()
        EXISTING_COMM_GROUPS.clear()

        self.addCleanup(self.patcher_dm.stop)
        self.addCleanup(self.patcher_dm_tensor.stop)
        self.addCleanup(self.patcher_tr.stop)
        self.addCleanup(_DEVICE_MESH_MAP.clear)
        self.addCleanup(EXISTING_COMM_GROUPS.clear)

    def _make_layout(self, mesh_shape, alias_name, rank_list=None):
        """ make layout."""
        return Layout(mesh_shape, alias_name, rank_list=rank_list, init_backend=False)


class TestUnevenShardRedistribution(unittest.TestCase):
    """Exercise real layout/DTensor logic, mocking only communication."""

    def setUp(self) -> None:
        """Keep rank-specific mesh caches isolated."""
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        self.addCleanup(_DEVICE_MESH_MAP.clear)
        self.addCleanup(_LAYOUT_CACHE.clear)
        self.addCleanup(EXISTING_COMM_GROUPS.clear)

    @staticmethod
    def _layout(mesh_shape, placements, ndim, ranks):
        """Build explicit layouts without creating process groups."""
        layout = Layout(mesh_shape, tuple(f"axis{i}" for i in range(len(mesh_shape))),
                        rank_list=ranks, init_backend=False)
        layout.set_placements(placements)
        layout.placement_to_tensor_map(ndim)
        return layout

    def _check_full_tensor(self, full, mesh_shape, placements, ranks):
        """Compare all simulated ranks with the same unsharded reference."""
        for rank in ranks:
            self._check_rank_full_tensor(full, mesh_shape, placements, ranks, rank)

    def _check_rank_full_tensor(self, full, mesh_shape, placements, ranks, rank):
        """Check one rank with independent layout and collective mock state."""
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()
        with patch("hyper_parallel.core.dtensor.device_mesh.dist") as mesh_dist:
            mesh_dist.get_rank.return_value = rank
            source = self._layout(mesh_shape, placements, full.ndim, ranks)
        mappings = list(source.tensor_map)

        def _slice_for(peer):
            """Construct the reference shard for a peer."""
            area = infer_slice_area_by_rank(
                mesh_shape, mappings, ranks.index(peer), full.shape, source.uneven_shard_mesh_dims,
            )
            return full[tuple(slice(start, end) for start, end in area)].clone()

        local = _slice_for(rank).requires_grad_()
        dtensor = DTensor.from_local_with_layout(local, source, shape=tuple(full.shape))
        original_map = source.tensor_map
        original_placements = source.placements

        def _current_group():
            """Find the group for the next shard axis being gathered."""
            dim = next(index for index, mapping in enumerate(mappings) if mapping != -1)
            axes = mappings[dim] if isinstance(mappings[dim], tuple) else (mappings[dim],)
            return source.mesh.get_devices_for_axis(len(mesh_shape) - 1 - axes[-1], rank)

        def _gather(outputs, padded, group):
            """Mock rank-ordered communication without replacing layout logic."""
            self.assertEqual(group, "mock_group")
            dim = next(index for index, mapping in enumerate(mappings) if mapping != -1)
            rank_list = sorted(_current_group())
            self.assertEqual(len(outputs), len(rank_list))
            for output, peer in zip(outputs, rank_list):
                value = _slice_for(peer)
                buffer = value.new_zeros(padded.shape)
                buffer.narrow(dim, 0, value.shape[dim]).copy_(value)
                if peer == rank:
                    torch.testing.assert_close(padded, buffer, rtol=0, atol=0)
                output.copy_(buffer)
            axes = list(mappings[dim] if isinstance(mappings[dim], tuple) else (mappings[dim],))
            axes.pop()
            mappings[dim] = tuple(axes) if axes else -1

        with patch("hyper_parallel.core.dtensor.tensor_redistribution._utils.create_group",
                   return_value="mock_group"), patch(
                       "hyper_parallel.core.dtensor.tensor_redistribution.dist.all_gather",
                       side_effect=_gather,
                   ), patch(
                       "hyper_parallel.core.dtensor.tensor_redistribution.dist.get_process_group_ranks",
                       side_effect=lambda group: sorted(_current_group()),
                   ), torch.no_grad():
            result = dtensor.full_tensor()
        torch.testing.assert_close(result, full, rtol=0, atol=0)
        self.assertFalse(result.requires_grad)
        self.assertIsNone(result.grad_fn)
        # Verify the gather owns storage instead of returning a view.
        self.assertIsNone(result._base)  # pylint: disable=protected-access
        self.assertIsNone(local.grad)
        self.assertEqual(result.untyped_storage().nbytes(), result.numel() * result.element_size())
        self.assertEqual(source.tensor_map, original_map)
        self.assertEqual(source.placements, original_placements)
        self.assertEqual(tuple(dtensor.shape), tuple(full.shape))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_full_tensor_ceil_shards_and_empty_ranks(self):
        """Feature: Uneven full tensor reconstruction.

        Description: Gather ceil shards including 48/32 and empty ranks.
        Expectation: Values match exactly; output owns compact detached storage.
        """
        for size, count in ((48, 32), (5, 4), (2, 4), (8, 4), (0, 4), (5, 1)):
            with self.subTest(size=size, count=count):
                self._check_full_tensor(torch.arange(size, dtype=torch.float32), (count,),
                                        (Shard(0, uneven_shard=True),), tuple(range(count)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_full_tensor_mixed_mesh_and_nonzero_dimension(self):
        """Feature: Mixed-mesh uneven export.

        Description: Exercise replica axes, TP+FSDP and reordered ranks.
        Expectation: Reconstruction preserves values and source metadata.
        """
        cases = (
            ((2, 2), (Replicate(), Shard(0, uneven_shard=True))),
            ((2, 2), (Shard(1), Shard(0, uneven_shard=True))),
            ((2, 2), (Shard(0), Shard(0, uneven_shard=True))),
            ((2, 2), (StridedShard(0, split_factor=2, uneven_shard=True), Shard(0))),
        )
        for mesh_shape, placements in cases:
            with self.subTest(placements=placements):
                self._check_full_tensor(torch.arange(30, dtype=torch.float32).reshape(5, 6),
                                        mesh_shape, placements, (3, 1, 2, 0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_get_model_state_dict_cpu_offload(self):
        """Feature: Full-state CPU export.

        Description: Export an uneven parameter through the public state API.
        Expectation: Only rank zero retains unpadded CPU weights.
        """
        source = self._layout((4,), (Shard(0, uneven_shard=True),), 1, (0, 1, 2, 3))
        tensor = DTensor.from_local_with_layout(torch.tensor([0., 1.]), source, shape=(5,))
        model = SimpleNamespace(state_dict=lambda: {"A_log": tensor})
        options = SimpleNamespace(full_state_dict=True, cpu_offload=True,
                                  ignore_frozen_params=False, broadcast_from_rank0=False)

        def _gather(outputs, unused_padded, group):
            """Populate ordinary all-gather output buffers in place."""
            del group, unused_padded
            for output, values in zip(outputs, ([0., 1.], [2., 3.], [4., 0.], [0., 0.])):
                output.copy_(torch.tensor(values))

        with patch("hyper_parallel.core.dtensor.tensor_redistribution._utils.create_group"), patch(
            "hyper_parallel.core.dtensor.tensor_redistribution.dist.all_gather", side_effect=_gather,
        ), patch(
            "hyper_parallel.core.dtensor.tensor_redistribution.dist.get_process_group_ranks",
            return_value=[0, 1, 2, 3],
        ), patch("hyper_parallel.core.fully_shard.state_dict_utils.dist") as state_dist:
            state_dist.get_rank.return_value = 0
            state = get_model_state_dict(model, options=options)
            torch.testing.assert_close(state["A_log"], torch.arange(5, dtype=torch.float32), rtol=0, atol=0)
            self.assertEqual(state["A_log"].device.type, "cpu")
            state_dist.get_rank.return_value = 1
            self.assertEqual(get_model_state_dict(model, options=options), {})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_rejects_unsupported_target_and_invalid_local_shape(self):
        """Feature: Uneven redistribution validation.

        Description: Supply unsupported targets and padded local storage.
        Expectation: Both invalid cases raise descriptive exceptions.
        """
        source = self._layout((2,), (Shard(0, uneven_shard=True),), 1, (0, 1))
        tensor = DTensor.from_local_with_layout(torch.arange(3.), source, shape=(5,))
        with self.assertRaisesRegex(NotImplementedError, "fully replicated target"):
            TensorRedistribution().redistribution(tensor, source)
        invalid = DTensor.from_local_with_layout(torch.arange(4.), source, shape=(5,))
        with self.assertRaisesRegex(ValueError, "local shape"):
            invalid.full_tensor()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_export_rejects_partial_nonzero_uneven_axis_and_autograd(self):
        """Feature: Export-only boundary.

        Description: Request Partial, nonzero uneven axes or autograd.
        Expectation: Reject before communicating, rather than detach silently.
        """
        source = self._layout((2, 2), (Partial(), Shard(0, uneven_shard=True)), 1, (0, 1, 2, 3))
        tensor = DTensor.from_local_with_layout(torch.tensor([1., 2., 3.]), source, shape=(5,))
        with patch("hyper_parallel.core.dtensor.tensor_redistribution.dist.all_gather") as gather:
            with self.assertRaisesRegex(NotImplementedError, "Partial"):
                tensor.full_tensor()
            source = self._layout((2,), (Shard(1, uneven_shard=True),), 2, (0, 1))
            tensor = DTensor.from_local_with_layout(torch.ones(2, 3), source, shape=(2, 5))
            with self.assertRaisesRegex(NotImplementedError, "dim=0"):
                tensor.full_tensor()
            source = self._layout((2,), (Shard(0, uneven_shard=True),), 1, (0, 1))
            tensor = DTensor.from_local_with_layout(torch.ones(3, requires_grad=True), source, shape=(5,))
            with self.assertRaisesRegex(RuntimeError, "not differentiable"):
                tensor.full_tensor()
            gather.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_export_does_not_retain_communication_storage(self):
        """Feature: Communication buffer lifetime.

        Description: Keep the export result alive after the helper returns.
        Expectation: Temporary buffers die; output cannot alias input storage.
        """
        references = []

        def _gather(outputs, padded, group):
            """Keep only weak references to communication tensors and storage."""
            del group
            for buffer in [padded, *outputs]:
                references.extend((weakref.ref(buffer), weakref.ref(buffer.untyped_storage())))
            outputs[0].copy_(torch.tensor([1., 2., 3.]))
            outputs[1].copy_(torch.tensor([4., 5., 0.]))

        local = torch.tensor([4., 5.], requires_grad=True)
        with patch("hyper_parallel.core.dtensor.tensor_redistribution._utils.create_group"), patch(
            "hyper_parallel.core.dtensor.tensor_redistribution.dist.get_process_group_ranks", return_value=[0, 1],
        ), patch("hyper_parallel.core.dtensor.tensor_redistribution.dist.all_gather", new=_gather):
            result = TensorRedistribution._padded_all_gather(local, [3, 2], 0, [0, 1])
        torch.testing.assert_close(result, torch.arange(1., 6.), rtol=0, atol=0)
        self.assertTrue(all(reference() is None for reference in references))
        self.assertFalse(result.requires_grad)
        # Verify the gather owns storage instead of returning a view.
        self.assertIsNone(result._base)  # pylint: disable=protected-access
        single = TensorRedistribution._padded_all_gather(local, [2], 0, [0])
        self.assertNotEqual(single.data_ptr(), local.data_ptr())
        with torch.no_grad():
            local.zero_()
        torch.testing.assert_close(single, torch.tensor([4., 5.]), rtol=0, atol=0)


# ===========================================================================
# TensorRedistribution method tests
# ===========================================================================

class TestConstructReshape(_MockedTestCase):
    """Tests for ConstructReshape."""
    def test_basic(self):
        """Test basic."""
        tr = TensorRedistribution()
        x = MagicMock()
        x.view.return_value = "reshaped"
        result = tr._construct_reshape(x, 2, 3)
        x.view.assert_called_once_with((2, 3))
        self.assertEqual(result, "reshaped")


class TestConstructAllConcat(_MockedTestCase):
    """Tests for ConstructAllConcat."""
    def test_basic(self):
        """Test basic."""
        tr = TensorRedistribution()
        x = MagicMock()
        self.mock_tr_platform.create_group.return_value = "group"
        self.mock_tr_platform.differentiable_all_gather_concat.return_value = "gathered"

        result = tr._construct_all_concat(x, 0, 1, 2, 3)
        # rank_list = (0, 1, 2), concat_dim = 3
        self.mock_tr_platform.create_group.assert_called_once_with((0, 1, 2))
        self.mock_tr_platform.differentiable_all_gather_concat.assert_called_once()
        self.assertEqual(result, "gathered")


class TestConstructStridedSlice(_MockedTestCase):
    """Tests for ConstructStridedSlice."""
    def test_basic(self):
        """Test basic."""
        tr = TensorRedistribution()
        x = MagicMock()
        self.mock_tr_platform.construct_strided_slice.return_value = "sliced"
        # args: begin0, begin1, end0, end1, stride0, stride1
        result = tr._construct_strided_slice(x, 0, 0, 4, 4, 1, 1)
        self.mock_tr_platform.construct_strided_slice.assert_called_once()
        self.assertEqual(result, "sliced")


class TestConstructAllSplit(_MockedTestCase):
    """Tests for ConstructAllSplit."""
    def test_basic(self):
        """Test basic."""
        from hyper_parallel.core.dtensor import tensor_redistribution as tr_mod
        tr = TensorRedistribution()
        tr.rank_id = 0
        x = torch.arange(16).reshape(2, 8)

        # args: (split_dim, split_size, group); rank 0 -> index 0 of the split.
        with patch.object(tr_mod.torch, "chunk", wraps=torch.chunk) as mock_chunk:
            result = tr._construct_all_split(x, 1, 4, [0, 1, 2, 3])
        mock_chunk.assert_called_once_with(x, 4, dim=1)
        self.assertTrue(torch.equal(result, x.chunk(4, dim=1)[0]))


class TestConstructAllToAll(_MockedTestCase):
    """Tests for ConstructAllToAll."""
    def test_same_dim(self):
        """Test same dim."""
        tr = TensorRedistribution()
        # Create a real tensor for shape inspection
        x = torch.randn(8, 4)
        self.mock_tr_platform.create_group.return_value = "group"
        self.mock_tr_platform.differentiable_all_to_all.return_value = torch.randn(8, 4)

        # split_dim == concat_dim
        result = tr._construct_all_to_all(x, 0, 0, 2, [0, 1])
        self.mock_tr_platform.create_group.assert_called_once()
        self.mock_tr_platform.differentiable_all_to_all.assert_called_once()

    def test_different_dim(self):
        """Test different dim."""
        tr = TensorRedistribution()
        x = torch.randn(8, 4)
        self.mock_tr_platform.create_group.return_value = "group"
        self.mock_tr_platform.differentiable_all_to_all.return_value = torch.randn(8, 4)

        result = tr._construct_all_to_all(x, 0, 1, 2, [0, 1])
        self.mock_tr_platform.create_group.assert_called_once()

    def test_uneven_split_raises(self):
        """Test uneven split raises."""
        tr = TensorRedistribution()
        x = torch.randn(7, 4)  # 7 can't be evenly split into 2
        self.mock_tr_platform.create_group.return_value = "group"

        with self.assertRaises(ValueError):
            tr._construct_all_to_all(x, 0, 1, 2, [0, 1])


# ===========================================================================
# _apply_eazy_redistribute tests
# ===========================================================================

class TestApplyEazyRedistribute(_MockedTestCase):
    """Tests for ApplyEazyRedistribute."""
    def test_same_layout_returns_true(self):
        """Test same layout returns true."""
        tr = TensorRedistribution()

        src = MagicMock()
        src.mesh_shape = (2, 4)
        src.rank_list = (0, 1, 2, 3, 4, 5, 6, 7)
        src.tensor_map = (1, -1)

        dst = MagicMock()
        dst.mesh_shape = (2, 4)
        dst.rank_list = (0, 1, 2, 3, 4, 5, 6, 7)
        dst.tensor_map = (-1, 0)

        self.assertTrue(tr._apply_eazy_redistribute(src, dst))

    def test_different_mesh_shape_returns_false(self):
        """Test different mesh shape returns false."""
        tr = TensorRedistribution()

        src = MagicMock()
        src.mesh_shape = (2, 4)
        src.rank_list = (0, 1, 2, 3, 4, 5, 6, 7)

        dst = MagicMock()
        dst.mesh_shape = (4, 2)
        dst.rank_list = (0, 1, 2, 3, 4, 5, 6, 7)

        self.assertFalse(tr._apply_eazy_redistribute(src, dst))

    def test_different_rank_list_returns_false(self):
        """Test different rank list returns false."""
        tr = TensorRedistribution()

        src = MagicMock()
        src.mesh_shape = (2, 4)
        src.rank_list = (0, 1, 2, 3, 4, 5, 6, 7)

        dst = MagicMock()
        dst.mesh_shape = (2, 4)
        dst.rank_list = (7, 6, 5, 4, 3, 2, 1, 0)

        self.assertFalse(tr._apply_eazy_redistribute(src, dst))

    def test_different_tensor_map_len_returns_false(self):
        """Test different tensor map len returns false."""
        tr = TensorRedistribution()

        src = MagicMock()
        src.mesh_shape = (2, 4)
        src.rank_list = (0, 1, 2, 3, 4, 5, 6, 7)
        src.tensor_map = (1, -1)

        dst = MagicMock()
        dst.mesh_shape = (2, 4)
        dst.rank_list = (0, 1, 2, 3, 4, 5, 6, 7)
        dst.tensor_map = (1, -1, 0)

        self.assertFalse(tr._apply_eazy_redistribute(src, dst))


# ===========================================================================
# _allreduce_along_dev_dim tests
# ===========================================================================

class TestAllreduceAlongDevDim(_MockedTestCase):
    """Tests for AllreduceAlongDevDim."""
    def _make_mock_layout(self):
        """ make mock layout."""
        layout = MagicMock()
        layout.mesh_shape = (2, 4)
        layout.alias_name = ("dp", "mp")
        layout.get_comm_group_by_axis.return_value = "group"
        return layout

    def test_sum_op(self):
        """Test sum op."""
        layout = self._make_mock_layout()
        x = torch.randn(4, 4)
        self.mock_tr_platform.differentiable_all_reduce.return_value = x

        result = TensorRedistribution._allreduce_along_dev_dim(x, 'sum', layout, 'dp')
        self.mock_tr_platform.differentiable_all_reduce.assert_called_once_with(x, 'sum', 'group')

    def test_avg_op(self):
        """Test avg op."""
        layout = self._make_mock_layout()
        x = torch.randn(4, 4)
        self.mock_tr_platform.differentiable_all_reduce.return_value = x

        result = TensorRedistribution._allreduce_along_dev_dim(x, 'avg', layout, 'dp')
        self.mock_tr_platform.differentiable_all_reduce.assert_called_once_with(x, 'sum', 'group')

    def test_all_op(self):
        """Test all op."""
        layout = self._make_mock_layout()
        x = torch.tensor([True, False, True])
        self.mock_tr_platform.tensor_type_cast.return_value = torch.tensor([1, 0, 1])
        self.mock_tr_platform.differentiable_all_reduce.return_value = torch.tensor([1, 0, 1])

        result = TensorRedistribution._allreduce_along_dev_dim(x, 'all', layout, 'dp')
        self.mock_tr_platform.tensor_type_cast.assert_called_once()
        self.mock_tr_platform.differentiable_all_reduce.assert_called_once()

    def test_zero_dim_tensor(self):
        """Test zero dim tensor."""
        layout = self._make_mock_layout()
        x = torch.tensor(5.0)  # 0-dim
        self.mock_tr_platform.differentiable_all_reduce.return_value = torch.tensor([5.0])

        result = TensorRedistribution._allreduce_along_dev_dim(x, 'sum', layout, 'dp')
        self.assertEqual(result.dim(), 0)


# ===========================================================================
# _construct_layout_tuple_for_transform_operator_list tests
# ===========================================================================

class TestConstructLayoutTuple(_MockedTestCase):
    """Tests for ConstructLayoutTuple."""
    def test_basic(self):
        """Test basic."""
        from hyper_parallel.core.dtensor.tensor_redistribution import (
            _construct_layout_tuple_for_transform_operator_list,
        )
        from_layout = MagicMock()
        from_layout.to_dict.return_value = {
            "mesh_shape": (2, 4),
            "tensor_map": (1, -1),
            "interleaved_parallel": False,
            "alias_name": ("dp", "mp"),
            "rank_list": (0, 1, 2, 3, 4, 5, 6, 7),
        }
        to_layout = MagicMock()
        to_layout.to_dict.return_value = {
            "mesh_shape": (2, 4),
            "tensor_map": (-1, 0),
            "interleaved_parallel": False,
            "alias_name": ("dp", "mp"),
            "rank_list": (0, 1, 2, 3, 4, 5, 6, 7),
        }
        from_tuple, to_tuple = _construct_layout_tuple_for_transform_operator_list(
            from_layout, to_layout, (16, 8)
        )
        self.assertEqual(from_tuple[0], (2, 4))
        self.assertEqual(from_tuple[1], (1, -1))
        self.assertEqual(to_tuple[0], (2, 4))
        self.assertEqual(to_tuple[1], (-1, 0))
        # Both use from_full_shape
        self.assertEqual(from_tuple[2], [16, 8])
        self.assertEqual(to_tuple[2], [16, 8])


# ===========================================================================
# TensorRedistribution operator mapping tests
# ===========================================================================

class TestTensorRedistributionInit(_MockedTestCase):
    """Tests for TensorRedistributionInit."""
    def test_operator_mapping(self):
        """Test operator mapping."""
        tr = TensorRedistribution()
        self.assertIn("Reshape", tr._construct_op_operator)
        self.assertIn("AllConcat", tr._construct_op_operator)
        self.assertIn("StridedSlice", tr._construct_op_operator)
        self.assertIn("all_concat", tr._construct_op_operator)
        self.assertIn("all_split", tr._construct_op_operator)
        self.assertIn("all_to_all", tr._construct_op_operator)
        self.assertFalse(tr.is_init)
        self.assertIsNone(tr.rank_id)


class TestConstructAllConcatNew(_MockedTestCase):
    """Tests for ConstructAllConcatNew."""
    def test_static_method(self):
        """Test static method."""
        x = MagicMock()
        self.mock_tr_platform.create_group.return_value = "group"
        self.mock_tr_platform.differentiable_all_gather_concat.return_value = "gathered"

        # args: (concat_dim, concat_size, group_rank_list)
        result = TensorRedistribution._construct_all_concat_new(x, 0, 4, [0, 1, 2, 3])
        self.mock_tr_platform.create_group.assert_called_once_with([0, 1, 2, 3])
        self.mock_tr_platform.differentiable_all_gather_concat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
