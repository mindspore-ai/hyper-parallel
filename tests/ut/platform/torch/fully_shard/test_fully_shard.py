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
"""TorchHSDPParamV2 Core Functions Unit Tests.

Covers parameter sharding init, unshard/wait, state transitions, reduce_scatter_grad,
all_reduce_grad, reset_sharded_param, and _get_unsharded_param_data. All tests mock
DTensor/Layout and distributed calls; no NPU required.
"""
from contextlib import nullcontext
import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from torch import nn

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.platform.torch.fully_shard.param import (
    TorchHSDPParamV2,
    _copy_without_bumping_version,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.fully_shard.hsdp_utils import ShardedState, ParamModuleInfo
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerContext
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, StridedShard
from hyper_parallel.core.fully_shard.utils import (
    MixedPrecisionPolicy,
    HSDPMeshInfo,
    FSDPMeshInfo,
    DDPMeshInfo,
    CPUOffloadPolicy,
    CommFusionPolicy,
    SourceShardMetaInfo,
)
from hyper_parallel.core.fully_shard.hsdp_utils import (
    FullyShardParamMode,
    infer_fully_shard_param_mode,
    get_managed_modules_parameters,
    get_rank_list_for_axes,
)
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.hsdp_param import (
    _build_group_info_from_rank_list,
    _GROUP_INFO_CACHE,
)
from hyper_parallel.platform.torch.fully_shard.state import TorchHSDPStateV2
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, get_torch_platform
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2


platform = get_torch_platform()
Tensor = platform.Tensor


class TestTorchHSDPParamV2(unittest.TestCase):
    """Test core functions of TorchHSDPParamV2 (init, unshard, to_sharded, grad ops)."""

    def setUp(self):
        """Set up the test environment with common test objects.

        Creates mock objects, parameters, module info, mesh info, sharding policy,
        and mixed precision policy for use in test methods.
        """
        # Create mesh information
        self.mesh_info = MagicMock(spec=FSDPMeshInfo)
        self.mesh_info.mesh = MagicMock()
        self.mesh_info.mesh.ndim = 1
        self.mesh_info.mesh.mesh_shape = (2,)
        self.mesh_info.mesh.mesh_dim_names = ("fsdp",)
        self.mesh_info.mesh.rank = 0
        self.mesh_info.mesh.rank_list = (0, 1)
        self.mesh_info.mesh.get_group.return_value = MagicMock()
        self.mesh_info.shard_mesh_dim = 0
        self.mesh_info.replicate_mesh_dim = None
        self.mesh_info.shard_mesh_rank = 0
        self.mesh_info.shard_mesh_size = 2
        self.mesh_info.shard_process_group = MagicMock()

        # FIXME: should be npu:0
        self.device = torch.device("cpu")
        self.param = torch.nn.Parameter(torch.randn(16, 16).to(self.device))
        sharded_shape = (self.param.shape[0] // self.mesh_info.shard_mesh_size, self.param.shape[1])
        self.sharded_param_data = Tensor(np.random.randn(*sharded_shape).astype(np.float32))

        # Create module information
        self.module = MagicMock()
        self.module_info = ParamModuleInfo(
            module=self.module,
            param_name="weight",
            shared_modules=[],
            shared_param_names=[]
        )

        # Create sharding policy
        self.shard_placement_fn = lambda p: Shard(0)

        # Create mixed precision policy
        self.mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True
        )

    def _create_mock_dtensor(self, mock_dtensor_from_local, sharded_param_data):
        """Create a mock DTensor instance with common settings."""
        mock_dtensor_instance = MagicMock(spec=DTensor)
        mock_dtensor_instance._local_tensor = self.sharded_param_data
        mock_dtensor_instance.untyped_storage.return_value = sharded_param_data.untyped_storage()
        mock_dtensor_instance.detach.return_value = mock_dtensor_instance
        mock_dtensor_instance.requires_grad_.return_value = mock_dtensor_instance
        mock_dtensor_instance.requires_grad = False
        mock_dtensor_from_local.return_value = mock_dtensor_instance

        return mock_dtensor_instance

    @staticmethod
    def _build_fake_dtensor(mesh, placements):
        """Build a lightweight fake DTensor for unit tests and mocked meshes."""
        if isinstance(mesh, DeviceMesh) and not isinstance(mesh, MagicMock):
            return DTensor.from_local(torch.zeros(1), mesh, placements)
        local_tensor = torch.zeros(1)
        dtensor = torch.Tensor._make_subclass(
            DTensor,
            local_tensor,
            require_grad=False,
        )
        dtensor._local_tensor = local_tensor  # pylint: disable=W0212
        dtensor._device_mesh = mesh  # pylint: disable=W0212
        dtensor._placements = placements  # pylint: disable=W0212
        return dtensor

    def _create_param_v2(self, **kwargs):
        """Create a TorchHSDPParamV2 instance with default parameters."""
        return TorchHSDPParamV2(
            param=kwargs.get('param', self.param),
            module_info=kwargs.get('module_info', self.module_info),
            mesh_info=kwargs.get('mesh_info', self.mesh_info),
            shard_placement_fn=kwargs.get('shard_placement_fn', self.shard_placement_fn),
            mp_policy=kwargs.get('mp_policy', self.mp_policy),
            offload_policy=kwargs.get('offload_policy'),
            device=kwargs.get('device', self.device),
        )

    def _simulate_unsharded_state(self, param_v2):
        """Simulate the unsharded state for a parameter."""
        param_v2._unsharded_param = MagicMock()
        param_v2.unsharded_param_buffers = [MagicMock()]
        param_v2.sharded_state = ShardedState.UNSHARDED

    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    def test_init_sharded_param(self, mock_dtensor_from_local, mock_sharded_local_tensor):
        """Test parameter sharding initialization.

        description: Create TorchHSDPParamV2 with mocked Layout/DTensor; check init state.
        expectation: is_sharded True, sharded_state SHARDED, hsdp_placement Shard(0).
        feature: fully_shard param init.

        Args:
            mock_dtensor_from_local: Unused mock parameter
            mock_sharded_local_tensor: Mock for _sharded_local_tensor method
        """
        mock_dtensor_instance = self._create_mock_dtensor(mock_dtensor_from_local, self.sharded_param_data)
        param_v2 = self._create_param_v2()

        # Verify initialization
        self.assertEqual(param_v2.sharded_state, ShardedState.SHARDED)
        self.assertEqual(param_v2.hsdp_placement, Shard(0))
        self.assertEqual(
            param_v2.sharded_size,
            (self.param.shape[0] // self.mesh_info.shard_mesh_size, self.param.shape[1]),
        )
        self.assertEqual(param_v2.sharded_param._hsdp_param_initialized, True)

    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    def test_init_sharded_param_below_threshold(self, mock_dtensor_from_local,
                                                mock_sharded_local_tensor):
        """Test initialization when parameter size is small.

        description: Init sharded param with small parameter; current impl shards all.
        expectation: is_sharded is True, sharded_state is SHARDED.
        feature: fully_shard param init (below-threshold Replicate not yet implemented).

        Args:
            mock_dtensor_from_local: Unused mock parameter
            mock_sharded_local_tensor: Mock for _sharded_local_tensor method
        """
        # Create small parameter
        param_data = torch.randn(4, 4).to(self.device)
        small_param = torch.nn.Parameter(param_data)
        self._create_mock_dtensor(mock_dtensor_from_local, param_data)
        param_v2 = self._create_param_v2(param=small_param)

        # Current behavior: all params are sharded (below-threshold not yet implemented)
        self.assertEqual(param_v2.sharded_state, ShardedState.SHARDED)
        self.assertEqual(param_v2.shard_world_size, self.mesh_info.shard_mesh_size)

    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    def test_disable_param_shard_skips_storage_sharding(
        self, mock_dtensor_from_local, mock_sharded_local_tensor
    ):
        """Verify DDPMeshInfo keeps replicate_params unsharded on the explicit mesh."""
        mock_dtensor_from_local.return_value = self.param.detach()
        replicate_mesh_info = MagicMock(spec=DDPMeshInfo)
        replicate_mesh_info.mesh = self.mesh_info.mesh
        replicate_mesh_info.shard_mesh_dim = None
        replicate_mesh_info.replicate_mesh_dim = 0
        replicate_mesh_info.replicate_mesh_size = self.mesh_info.shard_mesh_size
        replicate_mesh_info.replicate_process_group = MagicMock()

        param_v2 = TorchHSDPParamV2(
            param=self.param,
            module_info=self.module_info,
            mesh_info=replicate_mesh_info,
            shard_placement_fn=self.shard_placement_fn,
            mp_policy=self.mp_policy,
            device=self.device,
        )

        self.assertTrue(param_v2.is_replicate_param)
        self.assertEqual(param_v2.shard_world_size, 1)
        self.assertEqual(param_v2.replicate_world_size, self.mesh_info.shard_mesh_size)
        self.assertEqual(param_v2.sharded_size, self.param.shape)

    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    @patch.object(TorchHSDPParamV2, '_get_unsharded_param_data')
    def test_unshard_and_wait(self, mock_get_unsharded, mock_dtensor_from_local,
                              mock_sharded_local_tensor):
        """Test the unshard and wait_for_unshard process.

        description: Call unshard(async_op=True) then wait_for_unshard(); mock all_gather.
        expectation: _get_unsharded_param_data called, handle.wait, init_unsharded_param, to_unsharded.
        feature: fully_shard param unshard lifecycle.

        Args:
            mock_get_unsharded: Mock for _get_unsharded_param_data method
            mock_dtensor_from_local: Unused mock parameter
            mock_sharded_local_tensor: Mock for _sharded_local_tensor method
        """
        mock_dtensor_instance = self._create_mock_dtensor(mock_dtensor_from_local, self.sharded_param_data)
        mock_sharded_local_tensor.fsdp_pre_all_gather = True
        mock_sharded_local_tensor.fsdp_post_all_gather = True

        # Create parameter
        param_v2 = self._create_param_v2()

        # Set mock
        mock_handle = MagicMock()

        def _launch_unshard(async_op):
            self.assertTrue(async_op)
            param_v2.allgather_comm_ctx.allgather_handle = mock_handle

        mock_get_unsharded.side_effect = _launch_unshard

        # Call unshard
        param_v2.unshard(async_op=True)

        # Verify unshard call
        mock_get_unsharded.assert_called_once_with(async_op=True)
        self.assertIs(param_v2.allgather_comm_ctx.allgather_handle, mock_handle)

        # Call wait_for_unshard
        with patch.object(param_v2, "init_unsharded_param") as mock_init_unsharded, \
                patch.object(param_v2, "to_unsharded") as mock_to_unsharded:
            param_v2.wait_for_unshard()

            # Verify waiting and state transition
            mock_handle.wait.assert_called_once()
            mock_init_unsharded.assert_called_once()
            mock_to_unsharded.assert_called_once()
            self.assertIsNone(param_v2.allgather_comm_ctx.allgather_handle)

    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    def test_state_transitions(self, mock_dtensor_from_local, mock_sharded_local_tensor):
        """Test parameter state transitions.

        description: After unsharded state, call to_sharded(); verify _setattr and free.
        expectation: _setattr_on_modules and free_unsharded_param called, sharded_state SHARDED.
        feature: fully_shard param state transition to_sharded.

        Args:
            mock_dtensor_from_local: Unused mock parameter
            mock_sharded_local_tensor: Mock for _sharded_local_tensor method
        """
        mock_dtensor_instance = self._create_mock_dtensor(mock_dtensor_from_local, self.sharded_param_data)

        # Create parameter
        param_v2 = self._create_param_v2()

        # Simulate state after unshard
        self._simulate_unsharded_state(param_v2)

        # Test to_sharded
        with patch.object(param_v2, '_setattr_on_modules') as mock_setattr, \
                patch.object(param_v2, 'free_unsharded_param') as mock_free:
            param_v2.to_sharded()

            # Verify state transition
            mock_setattr.assert_called_once_with(param_v2.sharded_param)
            mock_free.assert_called_once()
            self.assertEqual(param_v2.sharded_state, ShardedState.SHARDED)

    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    def test_reduce_scatter_grad(self, mock_dtensor_from_local, mock_sharded_local_tensor):
        """Test gradient reduce-scatter operation.

        description: Set unsharded grad, call reduce_scatter_grad(async_op=True); mock dist.
        expectation: reduce_scatter_tensor called, _reduce_scatter_output size and handle set.
        feature: fully_shard param reduce_scatter_grad.

        Args:
            mock_dtensor_from_local: Unused mock parameter
            mock_sharded_local_tensor: Mock for _sharded_local_tensor method
        """
        mock_dtensor_instance = self._create_mock_dtensor(mock_dtensor_from_local, self.sharded_param_data)

        # Create parameter
        param_v2 = self._create_param_v2()

        # Simulate state after unshard
        self._simulate_unsharded_state(param_v2)
        param_v2._unsharded_param.grad = torch.zeros(self.param.shape).to(self.device)
        # Set mock
        mock_handle = MagicMock()
        # Call reduce_scatter_grad
        with patch("hyper_parallel.platform.torch.fully_shard.param.dist.reduce_scatter_tensor") as mock_reduce_scatter:
            mock_reduce_scatter.return_value = mock_handle
            param_v2.reduce_scatter_grad(async_op=True)
            # Verify call
            mock_reduce_scatter.assert_called_once()
            self.assertEqual(
                param_v2.reduce_scatter_comm_ctx.reduce_scatter_output.numel(),
                self.param.numel() // self.mesh_info.shard_mesh_size,
            )
            self.assertIs(param_v2.reduce_scatter_comm_ctx.reduce_scatter_handle, mock_handle)

    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    def test_all_reduce_grad(self, mock_dtensor_from_local, mock_sharded_local_tensor):
        """Test gradient all-reduce operation.

        description: Use HSDPMeshInfo and call all_reduce_grad(grad, async_op=True); mock dist.
        expectation: all_reduce called, _all_reduce_output shape/dtype and handle set.
        feature: fully_shard param all_reduce_grad (HSDP path).

        Args:
            mock_dtensor_from_local: Unused mock parameter
            mock_sharded_local_tensor: Mock for _sharded_local_tensor method
        """
        mock_dtensor_instance = self._create_mock_dtensor(mock_dtensor_from_local, self.sharded_param_data)

        # Create HSDP mesh information
        hsdp_mesh_info = MagicMock(spec=HSDPMeshInfo)
        hsdp_mesh_info.mesh = MagicMock()
        hsdp_mesh_info.mesh.ndim = 2
        hsdp_mesh_info.mesh.mesh_shape = (2, 2)
        hsdp_mesh_info.mesh.mesh_dim_names = ("dp", "fsdp")
        hsdp_mesh_info.shard_mesh_dim = 1
        hsdp_mesh_info.replicate_mesh_dim = 0
        hsdp_mesh_info.shard_mesh_rank = 0
        hsdp_mesh_info.shard_mesh_size = 2
        hsdp_mesh_info.shard_process_group = MagicMock()
        hsdp_mesh_info.replicate_mesh_size = 2
        hsdp_mesh_info.replicate_process_group = MagicMock()

        # Create parameter
        param_v2 = self._create_param_v2(mesh_info=hsdp_mesh_info)

        # Simulate gradient
        grad = torch.zeros(size=self.param.shape).to(self.device)
        with patch("hyper_parallel.platform.torch.fully_shard.param.dist.all_reduce") as mock_all_reduce:
            mock_handle = MagicMock()
            mock_all_reduce.return_value = mock_handle
            grad_dtype = torch.bfloat16
            param_v2.reduce_dtype = grad_dtype
            param_v2.reduce_scatter_comm_ctx.reduce_scatter_output = grad
            param_v2.all_reduce_grad(async_op=True)
            # Verify call
            mock_all_reduce.assert_called_once()
            self.assertEqual(param_v2.all_reduce_comm_ctx.all_reduce_output.shape, grad.shape)
            self.assertEqual(param_v2.all_reduce_comm_ctx.all_reduce_output.dtype, grad_dtype)
            self.assertIs(param_v2.all_reduce_comm_ctx.all_reduce_handle, mock_handle)

    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    @patch.object(TorchHSDPParamV2, "_update_shardedparam_storage_forcely")
    @patch('hyper_parallel.core.dtensor.layout.Layout')
    def test_reset_sharded_param(
        self,
        mock_layout,
        mock_update_storage,
        mock_dtensor_from_local,
        mock_sharded_local_tensor,
    ):
        """Test resetting sharded parameters.

        description: Create param_v2, call reset_sharded_param(); check _sharded_param_data.
        expectation: _sharded_param_data equals original sharded data (view -1).
        feature: fully_shard param reset_sharded_param.

        Args:
            mock_layout: Mock for Layout class
            mock_dtensor_from_local: Mock for DTensor class
            mock_sharded_local_tensor: Mock for _sharded_local_tensor method
        """
        sharded_param_data = self.sharded_param_data
        mock_dtensor_instance = self._create_mock_dtensor(mock_dtensor_from_local, sharded_param_data)

        # Create parameter
        param_v2 = self._create_param_v2()

        param_v2.reset_sharded_param()
        # Verify reset operation
        self.assertTrue(torch.all(param_v2._sharded_param_data == sharded_param_data.view(-1)))
        mock_update_storage.assert_called_once_with()

    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    @patch('hyper_parallel.core.dtensor.layout.Layout')
    def test_get_unsharded_param_data(self, mock_layout, mock_dtensor_from_local,
                                      mock_sharded_local_tensor):
        """Test getting unsharded parameter data.

        description: Call _get_unsharded_param_data(async_op=True); mock all_gather_into_tensor.
        expectation: alloc_unsharded_param_buffers and all_gather called, return handle matches mock.
        feature: fully_shard param _get_unsharded_param_data.

        Args:
            mock_layout: Mock for Layout class
            mock_dtensor_from_local: Mock for DTensor class
            mock_sharded_local_tensor: Mock for _sharded_local_tensor method
        """
        mock_dtensor_instance = self._create_mock_dtensor(mock_dtensor_from_local, self.sharded_param_data)

        # Create parameter
        param_v2 = self._create_param_v2()

        mock_handle = MagicMock()
        with patch('hyper_parallel.platform.torch.fully_shard.param.dist.all_gather_into_tensor') as mock_all_gather, \
                patch.object(param_v2, 'alloc_unsharded_param_buffers') as mock_alloc_outputs:
            mock_all_gather.return_value = mock_handle
            param_v2._get_unsharded_param_data(async_op=True)
            mock_alloc_outputs.assert_called_once()
            mock_all_gather.assert_called_once()
            self.assertIs(param_v2.allgather_comm_ctx.allgather_handle, mock_handle)

    def test_copy_without_bumping_version_uses_unsafe_preserve(self):
        """No-comm all-gather copies should preserve tensor version counters."""
        dst = MagicMock()
        src = MagicMock()

        with patch(
            "hyper_parallel.platform.torch.fully_shard.param."
            "torch.autograd._unsafe_preserve_version_counter",
            return_value=nullcontext(),
        ) as mock_preserve:
            _copy_without_bumping_version(dst, src)

        mock_preserve.assert_called_once_with(dst)
        dst.copy_.assert_called_once_with(src)

    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    @patch('hyper_parallel.core.dtensor.layout.Layout')
    def test_to_sharded_post_forward(self, mock_layout, mock_dtensor_from_local,
                                     mock_sharded_local_tensor):
        """Test transition to sharded state after forward (to_sharded).

        description: Call to_sharded() after unsharded state; verify state and mocks.
        expectation: _setattr_on_modules and free_unsharded_param called, sharded_state is SHARDED.
        feature: fully_shard param lifecycle (post-forward reshard).

        Args:
            mock_layout: Mock for Layout class.
            mock_dtensor_from_local: Mock for DTensor.from_local.
            mock_sharded_local_tensor: Mock for _sharded_local_tensor.
        """
        self._create_mock_dtensor(mock_dtensor_from_local, self.sharded_param_data)
        param_v2 = self._create_param_v2()
        self._simulate_unsharded_state(param_v2)

        with patch.object(param_v2, '_setattr_on_modules') as mock_setattr, \
                patch.object(param_v2, 'free_unsharded_param') as mock_free:
            param_v2.to_sharded()

            mock_setattr.assert_called_once()
            mock_free.assert_called_once()
            self.assertEqual(param_v2.sharded_state, ShardedState.SHARDED)

    def test_apply_data_parallel_placements_writes_fsdp_shard_to_explicit_dp_axis(self):
        """Verify FSDP placement is written to the explicit DP shard axis."""
        param_v2 = object.__new__(TorchHSDPParamV2)
        param_v2.mesh_info = object.__new__(FSDPMeshInfo)
        param_v2._orig_param_is_dtensor = False
        param_v2._spmd_shard_mesh_dim = 1
        param_v2._spmd_mesh = MagicMock()
        param_v2._spmd_mesh.ndim = 3
        param_v2._spmd_mesh.mesh_shape = (2, 4, 2)

        # Input:
        # - Unified SPMD placements already contain TP sharding on mesh axis 2.
        # - The explicit FSDP shard axis is mesh axis 1.
        # Expected output:
        # - Mesh axis 1 becomes StridedShard(dim=0, split_factor=2).
        placements = [Replicate(), Replicate(), Shard(0)]
        result = TorchHSDPParamV2._apply_data_parallel_placements(param_v2, placements, Shard(0))

        self.assertEqual(result, (Replicate(), StridedShard(0, split_factor=2), Shard(0)))

    def test_get_base_spmd_placements_prefixes_dp_mesh_for_dtensor_param(self):
        """Verify DTensor parameters preserve their model-parallel placements after the DP prefix."""
        dp_mesh = MagicMock(spec=DeviceMesh)
        dp_mesh.ndim = 1
        orig_mesh = MagicMock(spec=DeviceMesh)
        unified_mesh = MagicMock(spec=DeviceMesh)
        param_v2 = object.__new__(TorchHSDPParamV2)
        param_v2.mesh_info = MagicMock(spec=FSDPMeshInfo)
        param_v2.mesh_info.mesh = dp_mesh
        param_v2.source_shard_info = SourceShardMetaInfo(
            orig_mesh,
            (Replicate(),),
            origin_is_dtensor=True,
        )

        with patch.object(DeviceMesh, "concatenate", return_value=unified_mesh) as mock_concatenate:
            placements = TorchHSDPParamV2._get_base_spmd_placements(param_v2)

        mock_concatenate.assert_called_once_with([dp_mesh, orig_mesh])
        self.assertIs(param_v2._spmd_mesh, unified_mesh)
        self.assertEqual(placements, (Replicate(), Replicate()))

    def test_get_base_spmd_placements_accepts_plain_parameter_tp_metadata(self):
        """Dual-mode metadata should produce the same unified source layout without a DTensor origin."""
        dp_mesh = MagicMock(spec=DeviceMesh)
        dp_mesh.ndim = 1
        tp_mesh = MagicMock(spec=DeviceMesh)
        unified_mesh = MagicMock(spec=DeviceMesh)
        param_v2 = object.__new__(TorchHSDPParamV2)
        param_v2.mesh_info = MagicMock(spec=FSDPMeshInfo)
        param_v2.mesh_info.mesh = dp_mesh
        param_v2.source_shard_info = SourceShardMetaInfo(tp_mesh, (Shard(1),), origin_is_dtensor=False)
        param_v2._orig_param_is_dtensor = False

        with patch.object(DeviceMesh, "concatenate", return_value=unified_mesh) as mock_concatenate:
            placements = TorchHSDPParamV2._get_base_spmd_placements(param_v2)

        mock_concatenate.assert_called_once_with([dp_mesh, tp_mesh])
        self.assertIs(param_v2._spmd_mesh, unified_mesh)
        self.assertEqual(placements, (Replicate(), Shard(1)))

    def test_unsharded_grad_data_returns_plain_tensor_grad(self):
        """Verify gradient communication consumes the ordinary Tensor attached by Hyper autograd."""
        grad = torch.randn(4, 4)
        param_v2 = object.__new__(TorchHSDPParamV2)
        param_v2._unsharded_param = MagicMock()
        param_v2._unsharded_param.grad = grad

        self.assertIs(param_v2.unsharded_grad_data, grad)

    @patch("hyper_parallel.core.fully_shard.hsdp_param.platform._create_group")
    def test_build_group_info_from_rank_list_reuses_platform_group_cache(self, mock_create_group):
        """Verify explicit rank-list groups are cached through platform global state."""
        _GROUP_INFO_CACHE.clear()
        EXISTING_COMM_GROUPS.clear()
        mock_create_group.return_value = "cached-pg"

        group_info = _build_group_info_from_rank_list("fully_shard_unsharded_group", [3, 1, 2])
        cached_group_info = _build_group_info_from_rank_list("fully_shard_unsharded_group", [2, 3, 1])

        mock_create_group.assert_called_once_with([1, 2, 3])
        self.assertEqual(group_info.group, "cached-pg")
        self.assertEqual(cached_group_info.group, "cached-pg")
        self.assertEqual(EXISTING_COMM_GROUPS[str((1, 2, 3))], "cached-pg")
        _GROUP_INFO_CACHE.clear()
        EXISTING_COMM_GROUPS.clear()

    @patch("hyper_parallel.platform.torch.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_grad_supports_non_dim0_fully_shard_placement(self, mock_reduce_scatter):
        """Verify reduce_scatter_grad packs non-dim0 FSDP gradients via chunk-cat."""
        param_v2 = object.__new__(TorchHSDPParamV2)
        param_v2.sharded_state = ShardedState.UNSHARDED
        param_v2.unsharded_accumulated_grad = None
        param_v2.gradient_scaling_factor = None
        param_v2._unsharded_param = MagicMock()
        grad = torch.arange(16, dtype=torch.float32).view(4, 4)
        param_v2._unsharded_param.grad = grad
        param_v2.reduce_dtype = None
        param_v2.reduce_scatter_comm_ctx = SimpleNamespace(
            reduce_scatter_output=None,
            reduce_scatter_handle=None,
        )
        param_v2.mesh_info = MagicMock(spec=FSDPMeshInfo)
        param_v2.mesh_info.shard_process_group = MagicMock()
        param_v2.shard_world_size = 2
        param_v2.hsdp_placement = Shard(1)
        param_v2._spmd_placements = (Shard(1),)
        param_v2._orig_size = grad.shape
        mock_reduce_scatter.return_value = MagicMock()

        TorchHSDPParamV2.reduce_scatter_grad(param_v2, async_op=False)

        expected_packed = torch.cat(torch.chunk(grad, 2, dim=1), dim=0).reshape(-1)
        reduced_grad = param_v2.reduce_scatter_comm_ctx.reduce_scatter_output
        self.assertEqual(reduced_grad.numel(), grad.numel() // param_v2.shard_world_size)
        self.assertTrue(torch.equal(mock_reduce_scatter.call_args.args[1], expected_packed))

    @patch("hyper_parallel.platform.torch.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_grad_supports_same_dim_strided_non_dim0_layout(self, mock_reduce_scatter):
        """Verify same-dim StridedShard(dim!=0) reuses the non-dim0 chunk-cat packing path."""
        param_v2 = object.__new__(TorchHSDPParamV2)
        param_v2.sharded_state = ShardedState.UNSHARDED
        param_v2.unsharded_accumulated_grad = None
        param_v2.gradient_scaling_factor = None
        param_v2._unsharded_param = MagicMock()
        grad = torch.arange(32, dtype=torch.float32).view(4, 8)
        param_v2._unsharded_param.grad = grad
        param_v2.reduce_dtype = None
        param_v2.reduce_scatter_comm_ctx = SimpleNamespace(
            reduce_scatter_output=None,
            reduce_scatter_handle=None,
        )
        param_v2.mesh_info = MagicMock(spec=FSDPMeshInfo)
        param_v2.mesh_info.shard_process_group = MagicMock()
        param_v2.shard_world_size = 2
        param_v2.hsdp_placement = Shard(1)
        param_v2._orig_size = grad.shape
        param_v2.source_shard_info = SourceShardMetaInfo(
            MagicMock(),
            (Shard(1),),
            origin_is_dtensor=True,
        )
        param_v2._spmd_shard_mesh_dim = 0
        param_v2._spmd_placements = (StridedShard(1, split_factor=2), Shard(1))
        mock_reduce_scatter.return_value = MagicMock()

        TorchHSDPParamV2.reduce_scatter_grad(param_v2, async_op=False)

        expected_packed = torch.cat(torch.chunk(grad, 2, dim=1), dim=0).reshape(-1)
        reduced_grad = param_v2.reduce_scatter_comm_ctx.reduce_scatter_output
        self.assertEqual(reduced_grad.numel(), grad.numel() // param_v2.shard_world_size)
        self.assertTrue(torch.equal(mock_reduce_scatter.call_args.args[1], expected_packed))

    def test_to_accumulated_grad_if_needed_preserves_grad_without_reduce_dtype(self):
        """Verify no-sync paths still move local grad into accumulated buffer even without dtype conversion."""
        param_v2 = object.__new__(TorchHSDPParamV2)
        grad = torch.ones(2, 2, dtype=torch.float32)
        param_v2._unsharded_param = MagicMock()
        param_v2._unsharded_param.grad = grad
        param_v2.reduce_dtype = None
        param_v2.unsharded_accumulated_grad = None

        TorchHSDPParamV2.to_accumulated_grad_if_needed(param_v2)

        self.assertIsNone(param_v2._unsharded_param.grad)
        self.assertTrue(torch.equal(param_v2.unsharded_accumulated_grad, grad))


class TestFullyShardMeshUtils(unittest.TestCase):
    """Unit tests for fully_shard mesh helpers."""

    @staticmethod
    def _build_fake_dtensor(mesh, placements):
        """Build a lightweight fake DTensor for mesh helper unit tests."""
        if isinstance(mesh, DeviceMesh) and not isinstance(mesh, MagicMock):
            return DTensor.from_local(torch.zeros(1), mesh, placements)
        local_tensor = torch.zeros(1)
        dtensor = torch.Tensor._make_subclass(
            DTensor,
            local_tensor,
            require_grad=False,
        )
        dtensor._local_tensor = local_tensor  # pylint: disable=W0212
        dtensor._device_mesh = mesh  # pylint: disable=W0212
        dtensor._placements = placements  # pylint: disable=W0212
        return dtensor

    @staticmethod
    def _build_param_mesh_info(mesh, parameter=None):
        """Build parameter-owned mesh metadata through the refactored Torch state."""
        state = object.__new__(TorchHSDPStateV2)
        state.mesh = mesh
        state.raw_replicate_params = set()
        target_parameter = object() if parameter is None else parameter
        return TorchHSDPStateV2._build_param_mesh_info(state, target_parameter)

    def test_native_dtensor_requires_state_metadata(self):
        """TorchHSDPParamV2 should reject native DTensor input without owning-state metadata."""
        mesh = MagicMock(spec=DeviceMesh)
        parameter = self._build_fake_dtensor(mesh, (Replicate(),))

        with self.assertRaisesRegex(ValueError, "origin_is_dtensor"):
            TorchHSDPParamV2(
                param=parameter,
                module_info=MagicMock(),
                mesh_info=MagicMock(),
                device=torch.device("cpu"),
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform.get_rank", return_value=0)
    def test_device_mesh_concatenate_concatenates_explicit_dp_and_tp_meshes(self, mock_get_rank):
        """Verify concatenate rebuilds the original root mesh from DP and TP sub-meshes."""
        root_mesh = DeviceMesh(
            "cpu",
            np.array([[0, 1], [2, 3]]),
            mesh_dim_names=("dp", "tp"),
            _init_backend=False,
        )
        dp_mesh = root_mesh["dp"]
        tp_mesh = root_mesh["tp"]

        unified_mesh = DeviceMesh.concatenate([dp_mesh, tp_mesh])

        self.assertEqual(unified_mesh.mesh_dim_names, ("dp", "tp"))
        self.assertEqual(unified_mesh.to_hash(), root_mesh.to_hash())

    def test_infer_fully_shard_param_mode_for_tp_mesh(self):
        """Verify TP-only meshes still distinguish LOCAL_PARAM, DTENSOR_COMPAT, and DTENSOR_UNIFIED."""
        mesh = MagicMock(spec=DeviceMesh)
        mesh.mesh_dim_names = ("tp",)
        self.assertEqual(infer_fully_shard_param_mode(mesh, []), FullyShardParamMode.LOCAL_PARAM)
        dtensor_param = self._build_fake_dtensor(mesh, (Replicate(),))
        self.assertEqual(
            infer_fully_shard_param_mode(None, [dtensor_param]),
            FullyShardParamMode.DTENSOR_COMPAT,
        )
        self.assertEqual(
            infer_fully_shard_param_mode(mesh, [dtensor_param]),
            FullyShardParamMode.DTENSOR_UNIFIED,
        )

    def test_build_data_parallel_mesh_info_uses_explicit_1d_mesh_for_fsdp(self):
        """Verify an explicit 1D mesh builds FSDPMeshInfo with shard dim 0."""
        mesh = MagicMock(spec=DeviceMesh)
        mesh.ndim = 1
        mesh.mesh_shape = (4,)
        mesh.get_group.return_value = MagicMock()

        with patch("hyper_parallel.core.fully_shard.utils.get_group_local_rank", return_value=0):
            mesh_info = self._build_param_mesh_info(mesh)

        self.assertIsInstance(mesh_info, FSDPMeshInfo)
        self.assertEqual(mesh_info.shard_mesh_dim, 0)

    def test_build_data_parallel_mesh_info_uses_explicit_2d_mesh_for_hsdp(self):
        """Verify an explicit 2D mesh builds HSDPMeshInfo with DP and shard dims."""
        mesh = MagicMock(spec=DeviceMesh)
        mesh.ndim = 2
        mesh.mesh_shape = (2, 4)
        mesh.get_group.return_value = MagicMock()
        with patch("hyper_parallel.core.fully_shard.utils.get_group_local_rank", return_value=0):
            mesh_info = self._build_param_mesh_info(mesh)

        self.assertIsInstance(mesh_info, HSDPMeshInfo)
        self.assertEqual(mesh_info.replicate_mesh_dim, 0)
        self.assertEqual(mesh_info.shard_mesh_dim, 1)

    def test_fully_shard_rejects_mesh_none_for_dtensor_params(self):
        """Verify DTensor parameters require an explicit data-parallel mesh."""
        mesh = MagicMock(spec=DeviceMesh)
        mesh.ndim = 1
        mesh.mesh_dim_names = ("tp",)
        mesh.mesh_shape = (8,)
        module = nn.Module()
        module.register_parameter(
            "weight",
            nn.Parameter(self._build_fake_dtensor(mesh, (Replicate(),))),
        )
        with patch("hyper_parallel.core.fully_shard.api._validate_module_for_fully_shard"), \
                patch("hyper_parallel.core.fully_shard.api._extend_module_with_hsdp_interface"), \
                patch("hyper_parallel.core.fully_shard.api.platform.get_world_size", return_value=8), \
                patch("hyper_parallel.core.fully_shard.api.init_device_mesh", return_value=MagicMock()), \
                self.assertRaisesRegex(ValueError, "not support mesh=None"):
            fully_shard(module, mesh=None)

    def test_get_managed_modules_parameters_skips_already_initialized_nested_params(self):
        """Verify nested fully_shard ignores parameters already initialized by inner wrappers."""

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2, 2))

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.outer_weight = nn.Parameter(torch.randn(2, 2))

        module = Outer()
        nested_mesh = MagicMock(spec=DeviceMesh)
        nested_dtensor = self._build_fake_dtensor(nested_mesh, (Replicate(),))
        with torch.no_grad():
            module.inner.weight.data = nested_dtensor
        module.inner.weight._hsdp_param_initialized = True

        params = get_managed_modules_parameters((module,))

        self.assertEqual(params, [module.outer_weight])

    def test_nested_fully_shard_with_mesh_none_creates_default_mesh_for_unmanaged_outer_params(self):
        """Verify outer mesh=None ignores inner fully_shard params and still allocates a default mesh."""

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2, 2))

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.outer_weight = nn.Parameter(torch.randn(2, 2))

        module = Outer()
        nested_mesh = MagicMock(spec=DeviceMesh)
        nested_dtensor = self._build_fake_dtensor(nested_mesh, (Replicate(),))
        with torch.no_grad():
            module.inner.weight.data = nested_dtensor
        module.inner.weight._hsdp_param_initialized = True

        default_mesh = MagicMock(spec=DeviceMesh)

        def _attach_mock_hsdp(mod):
            mod.hsdp_init = MagicMock()

        with patch(
            "hyper_parallel.core.fully_shard.api._validate_module_for_fully_shard"
        ), patch(
            "hyper_parallel.core.fully_shard.api._extend_module_with_hsdp_interface",
            side_effect=_attach_mock_hsdp,
        ), patch(
            "hyper_parallel.core.fully_shard.api.platform.get_world_size",
            return_value=8,
        ), patch(
            "hyper_parallel.core.fully_shard.api.init_device_mesh",
            return_value=default_mesh,
        ) as mock_init_device_mesh, patch(
            "hyper_parallel.core.fully_shard.api._get_device_from_mesh",
            return_value=torch.device("cpu"),
        ):
            fully_shard(module, mesh=None)

        mock_init_device_mesh.assert_called_once_with(
            device_type="npu",
            mesh_shape=(8,),
        )
        module.hsdp_init.assert_called_once()
        self.assertIs(module.hsdp_init.call_args.args[2], default_mesh)

    def test_build_data_parallel_mesh_info_rejects_mesh_with_more_than_2_dims(self):
        """Verify parameter mesh-info construction rejects explicit meshes with rank greater than 2."""
        mesh = MagicMock(spec=DeviceMesh)
        mesh.ndim = 3
        mesh.mesh_dim_names = ("tp", "dp", "fsdp")
        mesh.mesh_shape = (2, 4, 8)
        with self.assertRaisesRegex(ValueError, "only supports explicit 1D DP/FSDP meshes or 2D HSDP meshes"):
            self._build_param_mesh_info(mesh)

    def test_scheduler_get_managed_params_skips_ignored_params(self):
        """Verify HSDPSchedulerV2._get_managed_params filters ignored params without AttributeError."""
        scheduler = object.__new__(HSDPSchedulerV2)
        module = nn.Module()
        keep_param = nn.Parameter(torch.randn(2, 2))
        ignore_param = nn.Parameter(torch.randn(2, 2))
        module.register_parameter("keep_weight", keep_param)
        module.register_parameter("ignore_weight", ignore_param)
        scheduler.modules = (module,)
        scheduler.ignored_params = {ignore_param}

        params = HSDPSchedulerV2._get_managed_params(scheduler)

        self.assertEqual(params, [keep_param])

    @patch("hyper_parallel.core.fully_shard.hsdp_param.platform._create_group", return_value=MagicMock())
    def test_build_group_info_from_rank_list_reuses_cached_group(self, mock_create_group):
        """Verify identical rank lists reuse a cached process group instead of recreating it."""
        _GROUP_INFO_CACHE.clear()
        EXISTING_COMM_GROUPS.clear()
        _build_group_info_from_rank_list("fully_shard_unsharded_group", [2, 0])
        _build_group_info_from_rank_list("fully_shard_unsharded_group", [0, 2])

        mock_create_group.assert_called_once_with([0, 2])
        _GROUP_INFO_CACHE.clear()
        EXISTING_COMM_GROUPS.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform.get_rank", return_value=0)
    def test_get_rank_list_for_axes_honors_explicit_rank(self, mock_get_rank):
        """Verify get_rank_list_for_axes uses the provided rank instead of mesh.rank."""
        mesh = DeviceMesh(
            "cpu",
            np.array([[0, 1], [2, 3]]),
            mesh_dim_names=("dp", "tp"),
            _init_backend=False,
        )

        rank_list = get_rank_list_for_axes(mesh, [0], rank=1)

        self.assertEqual(rank_list, [1, 3])

    @patch("hyper_parallel.platform.torch.fully_shard.state.TorchHSDPParamV2")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform.get_rank", return_value=0)
    def test_state_builds_param_metadata_per_parameter(self, mock_get_rank, mock_hsdp_param_cls):
        """Verify managed parameters receive parameter-specific DP and source-layout metadata."""
        mesh = DeviceMesh(
            "cpu",
            np.array([0, 1]),
            mesh_dim_names=("fsdp",),
            _init_backend=False,
        )

        module = nn.Module()
        module.register_parameter("local_weight", nn.Parameter(torch.randn(4, 4)))
        module.register_parameter(
            "dt_weight",
            nn.Parameter(self._build_fake_dtensor(mesh, (Replicate(),))),
        )

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        )

        mock_instance = MagicMock()
        mock_instance._orig_param_is_dtensor = False
        mock_instance.source_shard_info = None
        mock_instance.sharded_param.requires_grad = True
        mock_hsdp_param_cls.return_value = mock_instance

        with patch.object(mesh, "get_group", return_value=None), \
                patch("hyper_parallel.core.fully_shard.utils.get_group_local_rank", return_value=0), \
                patch.object(TorchHSDPStateV2, "_move_states_to_device"), \
                patch.object(TorchHSDPStateV2, "_validate_cpu_offload_params"), \
                patch(
                    "hyper_parallel.platform.torch.fully_shard.state._get_param_module_infos",
                    return_value=[MagicMock(), MagicMock()],
                ):
            state = TorchHSDPStateV2(
                (module,),
                mesh,
                None,
                CommFusionPolicy(),
                mp_policy,
                None,
                set(),
                {module.local_weight},
                MagicMock(),
                HSDPSchedulerContext(),
                torch.device("cpu"),
            )

        passed_mesh_infos = [call.args[2] for call in mock_hsdp_param_cls.call_args_list]
        passed_source_shard_infos = [
            call.kwargs["source_shard_info"] for call in mock_hsdp_param_cls.call_args_list
        ]
        self.assertEqual(state.hsdp_params, [mock_instance, mock_instance])
        self.assertIsInstance(passed_mesh_infos[0], DDPMeshInfo)
        self.assertIs(passed_mesh_infos[0].mesh, mesh)
        self.assertIsInstance(passed_mesh_infos[1], FSDPMeshInfo)
        self.assertIs(passed_mesh_infos[1].mesh, mesh)
        self.assertIsNone(passed_source_shard_infos[0])
        self.assertIsInstance(passed_source_shard_infos[1], SourceShardMetaInfo)
        self.assertIs(passed_source_shard_infos[1].mesh, mesh)
        self.assertEqual(passed_source_shard_infos[1].placements, (Replicate(),))
        self.assertTrue(passed_source_shard_infos[1].origin_is_dtensor)

    @patch("hyper_parallel.platform.torch.fully_shard.state.TorchHSDPParamV2")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform.get_rank", return_value=0)
    def test_state_passes_source_shard_info_to_each_parameter(self, mock_get_rank, mock_hsdp_param_cls):
        """The state should pass parameter-identity metadata into each Torch wrapper."""
        mesh = DeviceMesh(
            "cpu",
            np.array([0, 1]),
            mesh_dim_names=("fsdp",),
            _init_backend=False,
        )
        module = nn.Module()
        parameter = nn.Parameter(torch.randn(4, 4))
        module.register_parameter("weight", parameter)
        metadata = SourceShardMetaInfo(mesh, (Replicate(),))
        mock_instance = MagicMock()
        mock_instance.source_shard_info = metadata
        mock_instance.sharded_param.requires_grad = True
        mock_hsdp_param_cls.return_value = mock_instance

        with patch.object(mesh, "get_group", return_value=None), \
                patch("hyper_parallel.core.fully_shard.utils.get_group_local_rank", return_value=0), \
                patch.object(TorchHSDPStateV2, "_move_states_to_device"), \
                patch.object(TorchHSDPStateV2, "_validate_cpu_offload_params"), \
                patch(
                    "hyper_parallel.platform.torch.fully_shard.state._get_param_module_infos",
                    return_value=[MagicMock()],
                ):
            TorchHSDPStateV2(
                (module,),
                mesh,
                None,
                CommFusionPolicy(),
                MixedPrecisionPolicy(),
                None,
                set(),
                set(),
                MagicMock(),
                HSDPSchedulerContext(),
                torch.device("cpu"),
                source_shard_infos={parameter: metadata},
            )

        self.assertIs(mock_hsdp_param_cls.call_args.kwargs["source_shard_info"], metadata)

    @patch("hyper_parallel.platform.torch.fully_shard.state.DDPMeshInfo")
    def test_replicate_param_flattens_2d_mesh_for_ddp(self, mock_ddp_mesh_info):
        """A replicate parameter on a 2D HSDP mesh should all-reduce over the flattened mesh."""
        state = object.__new__(TorchHSDPStateV2)
        parameter = object()
        flattened_mesh = object()
        state.mesh = MagicMock(ndim=2)
        state.mesh.flatten.return_value = flattened_mesh
        state.raw_replicate_params = {parameter}

        result = state._build_param_mesh_info(parameter)

        state.mesh.flatten.assert_called_once_with()
        mock_ddp_mesh_info.assert_called_once_with(
            mesh=flattened_mesh,
            replicate_mesh_dim=0,
        )
        self.assertIs(result, mock_ddp_mesh_info.return_value)

    def test_state_resolves_default_reduce_op_per_state(self):
        """Verify default gradient reduction op is chosen once for the whole fully_shard state."""
        state = object.__new__(TorchHSDPStateV2)
        local_param = MagicMock()
        local_param.source_shard_info = None
        dtensor_param = MagicMock()
        dtensor_param.source_shard_info = SourceShardMetaInfo(MagicMock(), (Shard(0),), origin_is_dtensor=True)

        state.hsdp_params = [local_param]
        self.assertEqual(
            TorchHSDPStateV2._resolve_default_reduce_op(state),
            torch.distributed.ReduceOp.AVG,
        )

        state.hsdp_params = [local_param, dtensor_param]
        self.assertEqual(
            TorchHSDPStateV2._resolve_default_reduce_op(state),
            torch.distributed.ReduceOp.AVG,
        )

        state.hsdp_params = [dtensor_param]
        self.assertEqual(
            TorchHSDPStateV2._resolve_default_reduce_op(state),
            torch.distributed.ReduceOp.SUM,
        )

    def test_state_resolves_sum_for_plain_parameters_with_tp_metadata(self):
        """Dual-mode metadata should select SUM independently of parameter tensor type."""
        state = object.__new__(TorchHSDPStateV2)
        first = MagicMock()
        second = MagicMock()
        first.source_shard_info = SourceShardMetaInfo(MagicMock(), (Shard(0),))
        second.source_shard_info = SourceShardMetaInfo(MagicMock(), (Replicate(),))
        state.hsdp_params = [first, second]

        self.assertEqual(
            TorchHSDPStateV2._resolve_default_reduce_op(state),
            torch.distributed.ReduceOp.SUM,
        )

    def test_state_init_mp_dtypes_initializes_replicate_params_independently(self):
        """Verify _init_mp_dtypes leaves dtype ownership with each managed parameter."""
        state = object.__new__(TorchHSDPStateV2)
        state.hsdp_params = []
        state.replicate_params = []
        state.mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        )

        replicate_param = MagicMock()
        replicate_param.sharded_param.requires_grad = True
        replicate_param.orig_dtype = torch.float16
        replicate_param.reduce_dtype = torch.bfloat16
        replicate_param.init_dtype_attrs = MagicMock()
        state.hsdp_params.append(replicate_param)

        TorchHSDPStateV2._init_mp_dtypes(state)

        replicate_param.init_dtype_attrs.assert_called_once_with(state.mp_policy)

    def test_state_init_mp_dtypes_accepts_mixed_managed_dtypes(self):
        """Verify _init_mp_dtypes accepts mixed dtypes across managed parameters."""
        state = object.__new__(TorchHSDPStateV2)
        state.hsdp_params = []
        state.replicate_params = []
        state.mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        )

        hsdp_param = MagicMock()
        hsdp_param.sharded_param.requires_grad = True
        hsdp_param.orig_dtype = torch.float32
        hsdp_param.reduce_dtype = torch.float32
        hsdp_param.init_dtype_attrs = MagicMock()
        state.hsdp_params.append(hsdp_param)

        replicate_param = MagicMock()
        replicate_param.sharded_param.requires_grad = True
        replicate_param.orig_dtype = torch.float16
        replicate_param.reduce_dtype = torch.float32
        replicate_param.init_dtype_attrs = MagicMock()
        state.hsdp_params.append(replicate_param)

        TorchHSDPStateV2._init_mp_dtypes(state)

    @patch("hyper_parallel.platform.torch.fully_shard.state.TorchHSDPParamV2")
    def test_state_skips_ignored_params_during_param_init(self, mock_hsdp_param_cls):
        """Verify ignored_params are excluded from TorchHSDPParamV2 initialization."""
        mesh = MagicMock(spec=DeviceMesh)
        mesh.ndim = 1
        mesh.mesh_shape = (2,)
        mesh.mesh_dim_names = ("fsdp",)

        module = nn.Module()
        keep_param = nn.Parameter(torch.randn(4, 4))
        ignore_param = nn.Parameter(torch.randn(4, 4))
        module.register_parameter("keep_weight", keep_param)
        module.register_parameter("ignore_weight", ignore_param)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        )

        mock_instance = MagicMock()
        mock_instance._orig_param_is_dtensor = False
        mock_instance.source_shard_info = None
        mock_instance.sharded_param.requires_grad = True
        mock_hsdp_param_cls.return_value = mock_instance

        with patch.object(TorchHSDPStateV2, "_move_states_to_device"), \
                patch.object(TorchHSDPStateV2, "_validate_cpu_offload_params"), \
                patch("hyper_parallel.core.fully_shard.utils.get_group_local_rank", return_value=0), \
                patch(
                    "hyper_parallel.platform.torch.fully_shard.state._get_param_module_infos",
                    return_value=[MagicMock()],
                ):
            TorchHSDPStateV2(
                (module,),
                mesh,
                None,
                CommFusionPolicy(),
                mp_policy,
                None,
                {ignore_param},
                set(),
                MagicMock(),
                HSDPSchedulerContext(),
                torch.device("cpu"),
            )

        self.assertEqual(mock_hsdp_param_cls.call_count, 1)
        self.assertIs(mock_hsdp_param_cls.call_args.args[0], keep_param)

    @patch("hyper_parallel.platform.torch.fully_shard.state.TorchHSDPParamV2")
    def test_state_forwards_shard_placement_fn_during_param_init(self, mock_hsdp_param_cls):
        """Verify module-level fully_shard forwards config.shard_placement_fn into TorchHSDPParamV2."""
        mesh = MagicMock(spec=DeviceMesh)
        mesh.ndim = 1
        mesh.mesh_shape = (2,)
        mesh.mesh_dim_names = ("fsdp",)

        module = nn.Module()
        keep_param = nn.Parameter(torch.randn(4, 4))
        module.register_parameter("weight", keep_param)

        def shard_placement_fn(param):
            del param
            return Shard(1)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        )

        mock_instance = MagicMock()
        mock_instance._orig_param_is_dtensor = False
        mock_instance.source_shard_info = None
        mock_instance.sharded_param.requires_grad = True
        mock_hsdp_param_cls.return_value = mock_instance

        with patch.object(TorchHSDPStateV2, "_move_states_to_device"), \
                patch.object(TorchHSDPStateV2, "_validate_cpu_offload_params"), \
                patch("hyper_parallel.core.fully_shard.utils.get_group_local_rank", return_value=0), \
                patch(
                    "hyper_parallel.platform.torch.fully_shard.state._get_param_module_infos",
                    return_value=[MagicMock()],
                ):
            TorchHSDPStateV2(
                (module,),
                mesh,
                shard_placement_fn,
                CommFusionPolicy(),
                mp_policy,
                None,
                set(),
                set(),
                MagicMock(),
                HSDPSchedulerContext(),
                torch.device("cpu"),
            )

        self.assertEqual(mock_hsdp_param_cls.call_count, 1)
        self.assertIs(mock_hsdp_param_cls.call_args.args[0], keep_param)
        self.assertIs(mock_hsdp_param_cls.call_args.kwargs["shard_placement_fn"], shard_placement_fn)

    def test_state_validate_no_meta_params_checks_replicate_params(self):
        """Verify _validate_no_meta_params checks replicated entries in the unified parameter list."""
        state = object.__new__(TorchHSDPStateV2)

        replicate_param = MagicMock()
        replicate_param._param_fqn = "replicate_weight"
        replicate_param.sharded_param.device.type = "meta"
        state.hsdp_params = [replicate_param]

        with self.assertRaisesRegex(RuntimeError, "replicate_weight"):
            TorchHSDPStateV2._validate_no_meta_params(state)

    def test_state_validate_cpu_offload_params_checks_replicate_params(self):
        """Verify CPU offload validation covers replicated entries in the unified parameter list."""
        state = object.__new__(TorchHSDPStateV2)
        state.offload_policy = CPUOffloadPolicy()

        replicate_param = MagicMock()
        replicate_param._param_fqn = "replicate_weight"
        replicate_param.sharded_param.device.type = "cuda"
        state.hsdp_params = [replicate_param]

        with self.assertRaisesRegex(RuntimeError, "replicate_weight"):
            TorchHSDPStateV2._validate_cpu_offload_params(state)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform.get_rank", return_value=0)
    def test_device_mesh_concatenate_rejects_mismatched_root_meshes(self, mock_get_rank):
        """Verify concatenate rejects meshes coming from different root rank sets."""
        dp_mesh = DeviceMesh(
            "cpu",
            np.array([[0, 1], [2, 3]]),
            mesh_dim_names=("dp", "fsdp"),
            _init_backend=False,
        )
        tp_mesh = DeviceMesh(
            "cpu",
            np.array([[4, 5], [6, 7]]),
            mesh_dim_names=("tp", "ep"),
            _init_backend=False,
        )

        # Input:
        # - DP mesh and TP mesh come from different root rank sets.
        # Expected output:
        # - DeviceMesh.concatenate rejects the combination with ValueError.
        with self.assertRaisesRegex(ValueError, "share the same root mesh"):
            DeviceMesh.concatenate([dp_mesh, tp_mesh])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform.get_rank", return_value=0)
    def test_device_mesh_concatenate_rejects_root_mesh_without_dp_prefix_order(self, mock_get_rank):
        """Verify concatenate rejects sub-meshes that violate the root mesh dimension order."""
        root_mesh = DeviceMesh(
            "cpu",
            np.array([[0, 1], [2, 3]]),
            mesh_dim_names=("tp", "dp"),
            _init_backend=False,
        )
        dp_mesh = root_mesh["dp"]
        tp_mesh = root_mesh["tp"]

        with self.assertRaisesRegex(ValueError, "follow the root mesh order"):
            DeviceMesh.concatenate([dp_mesh, tp_mesh])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform.get_rank", return_value=0)
    def test_layout_deepcopy_preserves_submesh_root_for_concatenate(self, mock_get_rank):
        """Verify deepcopy keeps sub-mesh root references usable by concatenate."""
        root_mesh = DeviceMesh(
            "cpu",
            np.array([[0, 1], [2, 3]]),
            mesh_dim_names=("dp", "tp"),
            _init_backend=False,
        )
        dp_mesh = root_mesh["dp"]
        tp_mesh = root_mesh["tp"]

        copied_layout = copy.deepcopy(Layout.from_device_mesh(tp_mesh))
        unified_mesh = DeviceMesh.concatenate([dp_mesh, copied_layout.mesh])

        self.assertEqual(copied_layout.mesh._get_root_mesh().to_hash(), root_mesh.to_hash())  # pylint: disable=W0212
        self.assertEqual(unified_mesh.mesh_dim_names, ("dp", "tp"))
        self.assertEqual(unified_mesh.to_hash(), root_mesh.to_hash())

if __name__ == '__main__':
    unittest.main()
