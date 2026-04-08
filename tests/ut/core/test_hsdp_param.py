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
"""Unit tests for common HSDPParamV2 layout and group helpers."""

import os
import unittest
from unittest.mock import MagicMock, patch

import torch

# This core helper test builds torch DTensor/DeviceMesh objects directly, so it
# must pin the torch backend before importing hyper_parallel platform-bound types.
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.core.fully_shard.hsdp_param import (
    HSDPParamV2,
    _GROUP_INFO_CACHE,
    _build_group_info_from_rank_list,
)
from hyper_parallel.core.fully_shard.hsdp_utils import FullyShardParamMode, GroupInfo
from hyper_parallel.core.fully_shard.utils import DDPMeshInfo, FSDPMeshInfo
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestHSDPParamCoreHelpers(unittest.TestCase):
    """Test shared layout/group utilities used by platform-specific HSDP params."""

    def tearDown(self):
        _GROUP_INFO_CACHE.clear()
        EXISTING_COMM_GROUPS.clear()

    def _make_mesh(self, mesh, mesh_dim_names):
        """Create a DeviceMesh with mocked rank helpers so UT does not need dist init."""
        with patch("hyper_parallel.core.dtensor.device_mesh.platform") as mock_platform:
            mock_platform.get_rank.return_value = 0
            mock_platform.get_world_size.return_value = int(torch.tensor(mesh).numel())
            mock_platform.tensor_to_numpy.side_effect = (
                lambda tensor: tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor
            )
            return DeviceMesh("cpu", mesh, mesh_dim_names=mesh_dim_names, _init_backend=False)

    def _make_param_v2(self) -> HSDPParamV2:
        """Allocate a raw HSDPParamV2 object for helper-method testing."""
        param_v2 = object.__new__(HSDPParamV2)
        param_v2.uses_param_shard = True
        param_v2.is_sharded = True
        return param_v2

    @patch("hyper_parallel.core.fully_shard.hsdp_param.DeviceMesh.concatenate")
    def test_get_base_spmd_placements_concatenates_mesh_for_dtensor_unified(self, mock_concatenate):
        """DTENSOR_UNIFIED should prepend DP replicate dims before original DTensor placements."""
        fsdp_mesh = self._make_mesh([0, 1], ("fsdp",))
        orig_mesh = self._make_mesh([0, 1], ("tp",))
        unified_mesh = MagicMock(spec=DeviceMesh)
        mock_concatenate.return_value = unified_mesh

        param_v2 = self._make_param_v2()
        param_v2.param_mode = FullyShardParamMode.DTENSOR_UNIFIED
        param_v2.mesh_info = MagicMock(spec=DDPMeshInfo)
        param_v2.mesh_info.mesh = fsdp_mesh
        param_v2._orig_param_is_dtensor = True
        param_v2._orig_dtensor_mesh = orig_mesh
        param_v2._orig_dtensor_placements = (Shard(0),)

        placements = HSDPParamV2._get_base_spmd_placements(param_v2)

        mock_concatenate.assert_called_once_with([fsdp_mesh, orig_mesh])
        self.assertIs(param_v2._spmd_mesh, unified_mesh)
        self.assertEqual(placements, (Replicate(), Shard(0)))

    def test_get_base_spmd_placements_keeps_original_mesh_for_dtensor_compat(self):
        """DTENSOR_COMPAT should keep the existing DTensor mesh without concatenation."""
        compat_mesh = self._make_mesh([0, 1], ("tp",))

        param_v2 = self._make_param_v2()
        param_v2.param_mode = FullyShardParamMode.DTENSOR_COMPAT
        param_v2.mesh_info = MagicMock(spec=DDPMeshInfo)
        param_v2.mesh_info.mesh = compat_mesh
        param_v2._orig_param_is_dtensor = True
        param_v2._orig_dtensor_mesh = compat_mesh
        param_v2._orig_dtensor_placements = (Replicate(),)

        placements = HSDPParamV2._get_base_spmd_placements(param_v2)

        self.assertIs(param_v2._spmd_mesh, compat_mesh)
        self.assertEqual(placements, (Replicate(),))

    def test_apply_data_parallel_placements_writes_explicit_fsdp_axis(self):
        """Base helper should write the fully_shard placement onto the configured mesh axis."""
        param_v2 = self._make_param_v2()
        param_v2.param_mode = FullyShardParamMode.LOCAL_PARAM
        param_v2.mesh_info = object.__new__(FSDPMeshInfo)
        param_v2._orig_param_is_dtensor = False
        param_v2._spmd_mesh = MagicMock()
        param_v2._spmd_mesh.ndim = 3
        param_v2._spmd_shard_mesh_dim = 1

        result = HSDPParamV2._apply_data_parallel_placements(
            param_v2,
            [Replicate(), Replicate(), Shard(1)],
            Shard(0),
        )

        self.assertEqual(result, (Replicate(), Shard(0), Shard(1)))

    @patch("hyper_parallel.core.fully_shard.hsdp_param.platform.split_group", side_effect=RuntimeError("no pg"))
    @patch("hyper_parallel.core.fully_shard.hsdp_param.platform.create_group")
    def test_layout_driven_group_info_falls_back_to_rank_list(self, mock_create_group, mock_split_group):
        """When mesh groups are unavailable, layout-driven group info should derive an explicit rank list."""
        del mock_split_group
        mesh = self._make_mesh(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
            ("dp", "fsdp", "tp"),
        )
        mock_create_group.return_value = "layout-group"

        param_v2 = self._make_param_v2()
        param_v2.param_mode = FullyShardParamMode.LOCAL_PARAM
        param_v2._spmd_mesh = mesh
        param_v2._spmd_placements = (Replicate(), Shard(0), Replicate())
        param_v2._spmd_shard_mesh_dim = 1

        group_info = HSDPParamV2._build_layout_driven_group_info(param_v2)

        mock_create_group.assert_called_once_with([0, 1, 4, 5])
        self.assertEqual(group_info.group, "layout-group")
        self.assertEqual(group_info.rank_size, 4)
        self.assertEqual(group_info.group_name, str((0, 1, 4, 5)))

    @patch("hyper_parallel.core.fully_shard.hsdp_param.platform.split_group")
    def test_layout_driven_group_info_uses_split_group_for_multi_axis_replicate(self, mock_split_group):
        """Multi-axis layout groups should be materialized through split_group, not mesh flattening."""
        mesh = self._make_mesh(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
            ("dp", "fsdp", "tp"),
        )
        mock_split_group.return_value = "split-layout-group"

        param_v2 = self._make_param_v2()
        param_v2.param_mode = FullyShardParamMode.LOCAL_PARAM
        param_v2._spmd_mesh = mesh
        param_v2._spmd_placements = (Replicate(), Shard(0), Replicate())
        param_v2._spmd_shard_mesh_dim = 1

        group_info = HSDPParamV2._build_layout_driven_group_info(param_v2)

        mock_split_group.assert_called_once_with(split_ranks=[[0, 1, 4, 5], [2, 3, 6, 7]])
        self.assertEqual(group_info.group, "split-layout-group")
        self.assertEqual(group_info.rank_size, 4)

    def test_layout_driven_group_info_returns_invalid_without_replicate_axes(self):
        """A layout with no non-shard replicate axis should not create an all-reduce group."""
        param_v2 = self._make_param_v2()
        param_v2.param_mode = FullyShardParamMode.LOCAL_PARAM
        param_v2._spmd_mesh = MagicMock()
        param_v2._spmd_placements = (Shard(0),)
        param_v2._spmd_shard_mesh_dim = 0

        group_info = HSDPParamV2._build_layout_driven_group_info(param_v2)

        self.assertEqual(group_info.rank_size, 1)
        self.assertIsNone(group_info.group)

    @patch("hyper_parallel.core.fully_shard.hsdp_param.platform.create_group")
    def test_build_group_info_from_rank_list_reuses_cache(self, mock_create_group):
        """Explicit rank-list groups should reuse the cached communication group."""
        mock_create_group.return_value = "cached-group"

        group_info = _build_group_info_from_rank_list("fully_shard_unsharded_group", [3, 1, 2])
        cached_group_info = _build_group_info_from_rank_list("fully_shard_unsharded_group", [2, 3, 1])

        mock_create_group.assert_called_once_with([1, 2, 3])
        self.assertEqual(group_info.group, "cached-group")
        self.assertEqual(cached_group_info.group, "cached-group")
        self.assertEqual(group_info.group_name, str((1, 2, 3)))

    def test_init_group_infos_tracks_shard_and_dp_sizes(self):
        """Group initialization should record sharded and unsharded rank sizes consistently."""
        param_v2 = self._make_param_v2()
        param_v2.mesh_info = object.__new__(FSDPMeshInfo)
        param_v2.mesh_info.shard_process_group = "shard-group"
        param_v2.mesh_info.shard_mesh_size = 2

        with patch.object(
            HSDPParamV2,
            "_build_layout_driven_group_info",
            return_value=GroupInfo("unsharded-group", "unsharded-pg", 4),
        ):
            HSDPParamV2._init_group_infos(param_v2)

        self.assertEqual(param_v2.sharded_group_info.rank_size, 2)
        self.assertEqual(param_v2.unsharded_group_info.rank_size, 4)
        self.assertEqual(param_v2.shard_size, 2)
        self.assertEqual(param_v2.dp_size, 4)
        self.assertEqual(param_v2.rank_size, 8)

    def test_normalize_unsharded_grad_to_local_reduces_partial_dtensor(self):
        """Partial DTensor gradients should reduce before converting to the local tensor."""
        mesh = self._make_mesh([0, 1], ("tp",))
        partial_grad = DTensor.from_local(torch.ones(2), mesh, (Partial("sum"),))
        reduced_grad = DTensor.from_local(torch.arange(2, dtype=torch.float32), mesh, (Replicate(),))

        param_v2 = self._make_param_v2()
        param_v2._orig_dtensor_mesh = mesh
        param_v2._orig_dtensor_placements = (Replicate(),)

        with patch.object(DTensor, "reduce_partial", autospec=True, return_value=reduced_grad) as mock_reduce, patch.object(
            DTensor, "redistribute", autospec=True
        ) as mock_redistribute:
            grad = HSDPParamV2._normalize_unsharded_grad_to_local(param_v2, partial_grad)

        mock_reduce.assert_called_once_with(partial_grad)
        mock_redistribute.assert_not_called()
        self.assertTrue(torch.equal(grad, reduced_grad.to_local()))

    def test_normalize_unsharded_grad_to_local_redistributes_mismatched_layout(self):
        """DTensor gradients should redistribute back to the original managed layout before local use."""
        target_mesh = self._make_mesh([0, 1], ("tp",))
        grad_mesh = self._make_mesh([[0, 1], [2, 3]], ("dp", "tp"))
        mismatched_grad = DTensor.from_local(torch.ones(2), grad_mesh, (Replicate(), Shard(0)))
        redistributed_grad = DTensor.from_local(torch.arange(2, dtype=torch.float32), target_mesh, (Replicate(),))

        param_v2 = self._make_param_v2()
        param_v2._orig_dtensor_mesh = target_mesh
        param_v2._orig_dtensor_placements = (Replicate(),)

        with patch.object(DTensor, "redistribute", autospec=True, return_value=redistributed_grad) as mock_redistribute:
            grad = HSDPParamV2._normalize_unsharded_grad_to_local(
                param_v2,
                mismatched_grad,
                reduce_partial_dtensor=False,
            )

        mock_redistribute.assert_called_once_with(mismatched_grad, target_mesh, (Replicate(),))
        self.assertTrue(torch.equal(grad, redistributed_grad.to_local()))


if __name__ == "__main__":
    unittest.main()
