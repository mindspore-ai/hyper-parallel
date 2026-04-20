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
"""Unit tests for MindSpore fully_shard DTensor-aware param handling."""
# pylint: disable=protected-access

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.core.fully_shard.hsdp_utils import FullyShardParamMode, GroupInfo, ShardedState
from hyper_parallel.core.fully_shard.utils import FSDPMeshInfo
from hyper_parallel.platform.mindspore.fully_shard._version_utils import copy_without_bumping_version
from hyper_parallel.platform.mindspore.fully_shard.param import MindSporeHSDPParamV2


class FakeDTensorPayload:
    """Lightweight DTensor-like object accepted by unwrap_dtensor_param()."""

    def __init__(self, mesh="mesh-a", placements=("tp",)):
        self._device_mesh = mesh
        self._placements = placements
        self._local_tensor = object()

    @property
    def device_mesh(self):
        return self._device_mesh

    @property
    def placements(self):
        return self._placements

    def to_local(self):
        return self._local_tensor


class TestMindSporeParam(unittest.TestCase):
    """Cover DTensor-aware constructor and helper behavior."""

    def _make_module_info(self):
        module = MagicMock()
        module.register_load_state_dict_post_hook.return_value = "hook-handle"
        return SimpleNamespace(
            module=module,
            param_name="weight",
            shared_modules=[],
            shared_param_names=[],
        )

    def test_param_init_requires_resolved_param_mode(self):
        """Construction should fail until state/scheduler resolves the parameter mode."""
        mesh_info = MagicMock(spec=FSDPMeshInfo)

        with self.assertRaisesRegex(AssertionError, "param_mode must be resolved"):
            MindSporeHSDPParamV2(
                param=MagicMock(),
                module_info=self._make_module_info(),
                mesh_info=mesh_info,
                device="npu",
                param_mode=None,
            )

    @patch.object(MindSporeHSDPParamV2, "_init_group_infos")
    @patch.object(MindSporeHSDPParamV2, "_init_sharded_param")
    def test_param_init_tracks_dtensor_metadata_and_group_init(self, mock_init_sharded_param, mock_init_group_infos):
        """DTensor-managed params should record original mesh/layout metadata up front."""
        mesh_info = MagicMock(spec=FSDPMeshInfo)
        mesh_info.shard_mesh_dim = 0
        mesh_info.replicate_mesh_dim = 1
        param = FakeDTensorPayload(mesh="tp-mesh", placements=("tp",))

        hsdp_param = MindSporeHSDPParamV2(
            param=param,
            module_info=self._make_module_info(),
            mesh_info=mesh_info,
            device="npu",
            param_mode=FullyShardParamMode.DTENSOR_UNIFIED,
        )

        mock_init_sharded_param.assert_called_once_with(param, None)
        mock_init_group_infos.assert_called_once()
        self.assertTrue(hsdp_param._orig_param_is_dtensor)
        self.assertEqual(hsdp_param._orig_dtensor_mesh, "tp-mesh")
        self.assertEqual(hsdp_param._orig_dtensor_placements, ("tp",))
        self.assertEqual(hsdp_param._spmd_shard_mesh_dim, 0)
        self.assertEqual(hsdp_param._spmd_replicate_mesh_dim, 1)
        self.assertTrue(hsdp_param.uses_param_shard)

    def test_unsharded_grad_data_normalizes_dtensor_grad(self):
        """Gradient accessor should normalize DTensor grads back to local tensors."""
        hsdp_param = object.__new__(MindSporeHSDPParamV2)
        hsdp_param._unsharded_param = SimpleNamespace(grad="dtensor-grad")
        hsdp_param._to_local_unsharded_grad = MagicMock(return_value="local-grad")

        grad = hsdp_param.unsharded_grad_data

        hsdp_param._to_local_unsharded_grad.assert_called_once_with("dtensor-grad")
        self.assertEqual(grad, "local-grad")

    def test_get_data_parallel_shard_placement_writes_strided_shard_for_same_dim_layout(self):
        """Same-dim TP/FSDP layouts should materialize a StridedShard on the explicit fully_shard axis."""
        hsdp_param = object.__new__(MindSporeHSDPParamV2)
        hsdp_param._spmd_shard_mesh_dim = 1
        hsdp_param._spmd_mesh = SimpleNamespace(mesh_shape=(2, 4, 2))

        placement = MindSporeHSDPParamV2._get_data_parallel_shard_placement(
            hsdp_param,
            [Replicate(), Replicate(), Shard(0)],
            Shard(0),
        )

        self.assertEqual(placement, StridedShard(0, split_factor=2))

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.Parameter")
    @patch(
        "hyper_parallel.platform.mindspore.fully_shard.param."
        "MindSporeHSDPParamV2._get_unsharded_param_from_all_gather_output"
    )
    def test_init_unsharded_param_preserves_dtensor_wrapper(
        self,
        mock_get_unsharded_param,
        mock_parameter,
    ):
        """DTensor-managed params should create the wrapper once and then refresh it in place."""
        hsdp_param = object.__new__(MindSporeHSDPParamV2)
        hsdp_param._orig_param_is_dtensor = True
        hsdp_param._unsharded_param = None
        hsdp_param.sharded_param = SimpleNamespace(name="weight", requires_grad=True)
        mock_get_unsharded_param.return_value = "dtensor-unsharded"
        wrapped_parameter = MagicMock(name="wrapped-parameter")
        mock_parameter.return_value = wrapped_parameter

        MindSporeHSDPParamV2.init_unsharded_param(hsdp_param)

        mock_get_unsharded_param.assert_called_once_with()
        mock_parameter.assert_called_once_with(
            "dtensor-unsharded",
            name="weight",
            requires_grad=True,
        )
        self.assertIs(hsdp_param._unsharded_param, wrapped_parameter)

        mock_parameter.reset_mock()
        mock_get_unsharded_param.reset_mock()
        mock_get_unsharded_param.return_value = "next-dtensor-unsharded"
        wrapped_parameter.grad = "old-grad"

        MindSporeHSDPParamV2.init_unsharded_param(hsdp_param)

        mock_parameter.assert_not_called()
        wrapped_parameter.set_data.assert_called_once_with("next-dtensor-unsharded")
        self.assertIsNone(wrapped_parameter.grad)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.Parameter")
    @patch(
        "hyper_parallel.platform.mindspore.fully_shard.param."
        "MindSporeHSDPParamV2._get_unsharded_param_from_all_gather_output"
    )
    def test_init_unsharded_param_preserves_frozen_requires_grad_on_creation(
        self,
        mock_get_unsharded_param,
        mock_parameter,
    ):
        """Frozen local params should pass requires_grad=False when creating the unsharded Parameter."""
        hsdp_param = object.__new__(MindSporeHSDPParamV2)
        hsdp_param._orig_param_is_dtensor = False
        hsdp_param._unsharded_param = None
        hsdp_param.sharded_param = SimpleNamespace(name="frozen_weight", requires_grad=False)
        mock_get_unsharded_param.return_value = "frozen-local"
        created_parameter = MagicMock(name="created-parameter")
        mock_parameter.return_value = created_parameter

        MindSporeHSDPParamV2.init_unsharded_param(hsdp_param)

        mock_get_unsharded_param.assert_called_once_with()
        mock_parameter.assert_called_once_with(
            [],
            name="frozen_weight",
            requires_grad=False,
        )
        self.assertIs(hsdp_param._unsharded_param, created_parameter)
        self.assertEqual(created_parameter.data, "frozen-local")

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.set_requires_grad_if_needed")
    @patch(
        "hyper_parallel.platform.mindspore.fully_shard.param."
        "MindSporeHSDPParamV2._get_unsharded_param_from_all_gather_output"
    )
    def test_init_unsharded_param_refreshes_existing_local_parameter_data(
        self,
        mock_get_unsharded_param,
        mock_set_requires_grad_if_needed,
    ):
        """Existing local Parameters should be refreshed in place with the latest unpacked tensor."""
        hsdp_param = object.__new__(MindSporeHSDPParamV2)
        hsdp_param._orig_param_is_dtensor = False
        hsdp_param.sharded_param = SimpleNamespace(name="weight", requires_grad=True)
        existing_param = MagicMock(name="unsharded_param")
        existing_param.grad = "old-grad"
        hsdp_param._unsharded_param = existing_param
        mock_get_unsharded_param.return_value = "new-local"

        MindSporeHSDPParamV2.init_unsharded_param(hsdp_param)

        mock_get_unsharded_param.assert_called_once_with()
        mock_set_requires_grad_if_needed.assert_called_once_with(hsdp_param.sharded_param, existing_param)
        self.assertEqual(existing_param.data, "new-local")
        self.assertIsNone(existing_param.grad)

    def test_get_unsharded_param_data_uses_cast_all_gather_input_on_no_comm_path(self):
        """The no-communication path should preserve dtype casts already applied in all_gather_inputs."""
        hsdp_param = object.__new__(MindSporeHSDPParamV2)
        hsdp_param.is_sharded = False
        hsdp_param.sharded_state = MagicMock()
        hsdp_param.all_gather_outputs = []
        hsdp_param.reset_sharded_param = MagicMock()
        cast_input = MagicMock()
        cast_input.numel.return_value = 8
        cast_input.dtype = "float16"
        cast_input.device = "npu:0"
        output = MagicMock()

        hsdp_param.init_all_gather_outputs = MagicMock(
            side_effect=lambda **kwargs: setattr(
                hsdp_param, "all_gather_outputs", [output]
            )
        )
        hsdp_param.alloc_all_gather_outputs = MagicMock()

        with patch.object(
            MindSporeHSDPParamV2,
            "all_gather_inputs",
            new_callable=PropertyMock,
            return_value=[cast_input],
        ):
            gathered, handle = MindSporeHSDPParamV2._get_unsharded_param_data(hsdp_param)

        hsdp_param.init_all_gather_outputs.assert_called_once_with(
            all_gather_input_numels=[8],
            all_gather_input_dtypes=["float16"],
            world_size=1,
            device="npu",
        )
        hsdp_param.alloc_all_gather_outputs.assert_called_once()
        output.copy_.assert_not_called()
        output.data.copy_.assert_called_once_with(cast_input)
        self.assertIs(gathered, output)
        self.assertIsNone(handle)

    def test_copy_without_bumping_version_prefers_data_alias(self):
        """Shared helper should write through ``dst.data``."""
        dst = MagicMock(name="dst")
        src = MagicMock(name="src")

        copy_without_bumping_version(dst, src)

        dst.copy_.assert_not_called()
        dst.data.copy_.assert_called_once_with(src)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_grad_supports_same_dim_strided_non_dim0_layout(self, mock_reduce_scatter):
        """same-dim StridedShard(dim!=0) should reuse the non-dim0 chunk-cat packing path."""
        hsdp_param = object.__new__(MindSporeHSDPParamV2)
        hsdp_param.sharded_state = ShardedState.UNSHARDED
        hsdp_param.unsharded_accumulated_grad = None
        hsdp_param._unsharded_param = SimpleNamespace(
            grad=ms.Tensor(np.arange(32, dtype=np.float32).reshape(4, 8))
        )
        hsdp_param.is_sharded = True
        hsdp_param.mesh_info = MagicMock(spec=FSDPMeshInfo)
        hsdp_param.mesh_info.shard_process_group = "pg"
        hsdp_param.mesh_info.shard_mesh_size = 2
        hsdp_param.enable_fsdp_shard = True
        hsdp_param.hsdp_placement = Shard(1)
        hsdp_param._orig_size = (4, 8)
        hsdp_param._orig_param_is_dtensor = True
        hsdp_param._orig_dtensor_placements = (Shard(1),)
        hsdp_param._spmd_shard_mesh_dim = 0
        hsdp_param._spmd_placements = (StridedShard(1, split_factor=2), Shard(1))

        reduced_grad, _ = MindSporeHSDPParamV2.reduce_scatter_grad(hsdp_param, async_op=False)

        expected_packed = np.concatenate(
            np.array_split(np.arange(32, dtype=np.float32).reshape(4, 8), 2, axis=1),
            axis=0,
        ).reshape(-1)
        self.assertEqual(reduced_grad.numel(), 16)
        np.testing.assert_allclose(mock_reduce_scatter.call_args.args[1].asnumpy(), expected_packed)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.all_reduce")
    def test_all_reduce_grad_uses_layout_driven_unsharded_group(self, mock_all_reduce):
        """
        Gradient all-reduce should follow layout-driven group bookkeeping.

        This should no longer depend on ``mesh_info.replicate_process_group``.
        """
        raw_grad = MagicMock(name="raw-grad")
        normalized_grad = MagicMock(name="normalized-grad")
        mock_all_reduce.return_value = "reduce-handle"

        hsdp_param = object.__new__(MindSporeHSDPParamV2)
        hsdp_param.unsharded_accumulated_grad = None
        hsdp_param.unsharded_group_info = GroupInfo("unsharded-group", "layout-group", 4)
        hsdp_param._to_local_unsharded_grad = MagicMock(return_value=normalized_grad)

        reduced_grad, handle = MindSporeHSDPParamV2.all_reduce_grad(
            hsdp_param,
            grad=raw_grad,
            async_op=True,
            reduce_op="sum",
        )

        hsdp_param._to_local_unsharded_grad.assert_called_once_with(raw_grad)
        mock_all_reduce.assert_called_once_with(
            normalized_grad,
            op="sum",
            group="layout-group",
            async_op=True,
        )
        self.assertIs(reduced_grad, normalized_grad)
        self.assertEqual(handle, "reduce-handle")


if __name__ == "__main__":
    unittest.main()
