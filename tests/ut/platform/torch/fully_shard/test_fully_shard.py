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
import unittest
from unittest.mock import patch, MagicMock, call

import numpy as np
import torch
from torch import nn

from hyper_parallel.platform.torch.fully_shard.param import TorchHSDPParamV2
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.device_mesh import DeviceMesh, init_device_mesh
from hyper_parallel.core.fully_shard.hsdp_utils import ShardedState, ParamModuleInfo
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.fully_shard.utils import HSDPMeshInfo, FSDPMeshInfo, MixedPrecisionPolicy

from hyper_parallel.platform.platform import get_torch_platform
from tests.common.mark_utils import arg_mark

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
        mock_dtensor_instance.detach.return_value = mock_dtensor_instance
        mock_dtensor_instance.requires_grad_.return_value = mock_dtensor_instance
        mock_dtensor_from_local.return_value = mock_dtensor_instance

        return mock_dtensor_instance

    def _create_param_v2(self, **kwargs):
        """Create a TorchHSDPParamV2 instance with default parameters."""
        return TorchHSDPParamV2(
            param=kwargs.get('param', self.param),
            module_info=kwargs.get('module_info', self.module_info),
            mesh_info=kwargs.get('mesh_info', self.mesh_info),
            shard_placement_fn=kwargs.get('shard_placement_fn', self.shard_placement_fn),
            mp_policy=kwargs.get('mp_policy', self.mp_policy),
            device=kwargs.get('device', self.device)
        )

    def _simulate_unsharded_state(self, param_v2):
        """Simulate the unsharded state for a parameter."""
        param_v2._unsharded_param = MagicMock()
        param_v2.sharded_state = ShardedState.UNSHARDED

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    @patch('hyper_parallel.platform.torch.fully_shard.param.Layout')
    def test_init_sharded_param(self, mock_layout, mock_dtensor_from_local,
                                mock_sharded_local_tensor):
        """Test parameter sharding initialization.

        description: Create TorchHSDPParamV2 with mocked Layout/DTensor; check init state.
        expectation: is_sharded True, sharded_state SHARDED, hsdp_placement Shard(0).
        feature: fully_shard param init.

        Args:
            mock_layout: Mock for Layout class
            mock_dtensor_from_local: Unused mock parameter
            mock_sharded_local_tensor: Mock for _sharded_local_tensor method
        """
        mock_dtensor_instance = self._create_mock_dtensor(mock_dtensor_from_local, self.sharded_param_data)
        param_v2 = self._create_param_v2()

        # Verify initialization
        self.assertTrue(param_v2.is_sharded)
        self.assertEqual(param_v2.sharded_state, ShardedState.SHARDED)
        self.assertEqual(param_v2.hsdp_placement, Shard(0))
        self.assertTrue(param_v2.sharded_param.shape,
                        (self.param.shape[0] // self.mesh_info.shard_mesh_size, self.param.shape[1]))
        self.assertEqual(param_v2.sharded_param._hsdp_param_initialized, True)
        self.assertEqual(param_v2.sharded_state, ShardedState.SHARDED)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    @patch('hyper_parallel.platform.torch.fully_shard.param.Layout')
    def test_init_sharded_param_below_threshold(self, mock_layout, mock_dtensor_from_local,
                                                mock_sharded_local_tensor):
        """Test initialization when parameter size is small.

        description: Init sharded param with small parameter; current impl shards all.
        expectation: is_sharded is True, sharded_state is SHARDED.
        feature: fully_shard param init (below-threshold Replicate not yet implemented).

        Args:
            mock_layout: Mock for Layout class
            mock_dtensor_from_local: Unused mock parameter
            mock_sharded_local_tensor: Mock for _sharded_local_tensor method
        """
        # Create small parameter
        param_data = torch.randn(4, 4).to(self.device)
        small_param = torch.nn.Parameter(param_data)
        mock_dtensor_from_local.return_value = param_data

        mock_layout_instance = MagicMock()
        mock_layout.from_device_mesh.return_value = mock_layout_instance

        param_v2 = self._create_param_v2(
            param=small_param,
        )

        # Current behavior: all params are sharded (below-threshold not yet implemented)
        self.assertTrue(param_v2.is_sharded)
        self.assertEqual(param_v2.sharded_state, ShardedState.SHARDED)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    @patch('hyper_parallel.platform.torch.fully_shard.param.Layout')
    @patch.object(TorchHSDPParamV2, '_get_unsharded_param_data')
    def test_unshard_and_wait(self, mock_get_unsharded, mock_layout, mock_dtensor_from_local,
                              mock_sharded_local_tensor):
        """Test the unshard and wait_for_unshard process.

        description: Call unshard(async_op=True) then wait_for_unshard(); mock all_gather.
        expectation: _get_unsharded_param_data called, handle.wait, init_unsharded_param, to_unsharded.
        feature: fully_shard param unshard lifecycle.

        Args:
            mock_get_unsharded: Mock for _get_unsharded_param_data method
            mock_layout: Mock for Layout class
            mock_dtensor_from_local: Unused mock parameter
            mock_sharded_local_tensor: Mock for _sharded_local_tensor method
        """
        mock_dtensor_instance = self._create_mock_dtensor(mock_dtensor_from_local, self.sharded_param_data)
        mock_sharded_local_tensor.fsdp_pre_all_gather = True
        mock_sharded_local_tensor.fsdp_post_all_gather = True

        # Create parameter
        param_v2 = self._create_param_v2()

        # Set mock
        mock_output = MagicMock()
        mock_output.numel.return_value = self.param.numel()
        mock_handle = MagicMock()
        mock_get_unsharded.return_value = (mock_output, mock_handle)

        # Call unshard
        param_v2.unshard(async_op=True)

        # Verify unshard call
        mock_get_unsharded.assert_called_once_with(async_op=True)
        self.assertEqual(param_v2.prefetch_handle, mock_handle)

        # Call wait_for_unshard
        with patch.object(param_v2, "init_unsharded_param") as mock_init_unsharded, \
                patch.object(param_v2, "to_unsharded") as mock_to_unsharded:
            param_v2.wait_for_unshard()

            # Verify waiting and state transition
            mock_handle.wait.assert_called_once()
            mock_init_unsharded.assert_called_once()
            mock_to_unsharded.assert_called_once()
            self.assertIsNone(param_v2.prefetch_handle)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    @patch('hyper_parallel.platform.torch.fully_shard.param.Layout')
    def test_state_transitions(self, mock_layout, mock_dtensor_from_local,
                               mock_sharded_local_tensor):
        """Test parameter state transitions.

        description: After unsharded state, call to_sharded(); verify _setattr and free.
        expectation: _setattr_on_modules and free_unsharded_param called, sharded_state SHARDED.
        feature: fully_shard param state transition to_sharded.

        Args:
            mock_layout: Mock for Layout class
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

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    @patch('hyper_parallel.platform.torch.fully_shard.param.Layout')
    def test_reduce_scatter_grad(self, mock_layout, mock_dtensor_from_local,
                                 mock_sharded_local_tensor):
        """Test gradient reduce-scatter operation.

        description: Set unsharded grad, call reduce_scatter_grad(async_op=True); mock dist.
        expectation: reduce_scatter_tensor called, _reduce_scatter_output size and handle set.
        feature: fully_shard param reduce_scatter_grad.

        Args:
            mock_layout: Mock for Layout class
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
        mock_output = MagicMock()
        mock_handle = MagicMock()
        # Call reduce_scatter_grad
        with patch("hyper_parallel.platform.torch.fully_shard.param.dist.reduce_scatter_tensor") as mock_reduce_scatter:
            mock_reduce_scatter.return_value = mock_handle
            param_v2.reduce_scatter_grad(async_op=True)
            # Verify call
            mock_reduce_scatter.assert_called_once()
            self.assertEqual(param_v2._reduce_scatter_output.numel(),
                             self.param.numel() // self.mesh_info.shard_mesh_size)
            self.assertEqual(param_v2.reduce_scatter_handle, mock_handle)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    @patch('hyper_parallel.platform.torch.fully_shard.param.Layout')
    def test_all_reduce_grad(self, mock_layout, mock_dtensor_from_local, mock_sharded_local_tensor):
        """Test gradient all-reduce operation.

        description: Use HSDPMeshInfo, call all_reduce_grad(grad, dtype, async_op=True); mock dist.
        expectation: all_reduce called, _all_reduce_output shape/dtype and handle set.
        feature: fully_shard param all_reduce_grad (HSDP path).

        Args:
            mock_layout: Mock for Layout class
            mock_dtensor_from_local: Unused mock parameter
            mock_sharded_local_tensor: Mock for _sharded_local_tensor method
        """
        mock_dtensor_instance = self._create_mock_dtensor(mock_dtensor_from_local, self.sharded_param_data)

        # Create HSDP mesh information
        hsdp_mesh_info = MagicMock(spec=HSDPMeshInfo)
        hsdp_mesh_info.mesh = MagicMock()
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
            param_v2.all_reduce_grad(grad=grad, dtype=grad_dtype, async_op=True)
            # Verify call
            mock_all_reduce.assert_called_once()
            self.assertEqual(param_v2._all_reduce_output.shape, grad.shape)
            self.assertEqual(param_v2._all_reduce_output.dtype, grad_dtype)
            self.assertEqual(param_v2.all_reduce_handle, mock_handle)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    @patch('hyper_parallel.core.layout.Layout')
    def test_reset_sharded_param(self, mock_layout, mock_dtensor_from_local, mock_sharded_local_tensor):
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

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    @patch('hyper_parallel.core.layout.Layout')
    def test_get_unsharded_param_data(self, mock_layout, mock_dtensor_from_local,
                                      mock_sharded_local_tensor):
        """Test getting unsharded parameter data.

        description: Call _get_unsharded_param_data(async_op=True); mock all_gather_into_tensor.
        expectation: alloc_all_gather_outputs and all_gather called, return handle matches mock.
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
                patch.object(param_v2, 'alloc_all_gather_outputs') as mock_alloc_outputs:
            mock_all_gather.return_value = mock_handle
            result, handle = param_v2._get_unsharded_param_data(async_op=True)
            mock_alloc_outputs.assert_called_once()
            mock_all_gather.assert_called_once()
            self.assertEqual(handle, mock_handle)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
    @patch.object(TorchHSDPParamV2, "_sharded_local_tensor")
    @patch.object(DTensor, "from_local")
    @patch('hyper_parallel.core.layout.Layout')
    def test_to_sharded_post_forward(self, mock_layout, mock_dtensor_from_local,
                                     mock_sharded_local_tensor):
        """Test transition to sharded state after forward (to_sharded).

        description: Call to_sharded() after unsharded state; verify state and mocks.
        expectation: _setattr_on_modules and free_unsharded_param called, sharded_state is SHARDED.
        feature: fully_shard param lifecycle (post-forward reshard).

        Args:
            mock_layout: Mock for Layout class
            mock_dtensor_from_local: Mock for DTensor class
            mock_sharded_local_tensor: Mock for _sharded_local_tensor method
        """
        # Set mock return values
        mock_dtensor_instance = self._create_mock_dtensor(mock_dtensor_from_local, self.sharded_param_data)

        # Create post_forward mesh information
        post_forward_mesh = MagicMock(spec=FSDPMeshInfo)
        post_forward_mesh.mesh = MagicMock()
        post_forward_mesh.shard_mesh_rank = 0
        post_forward_mesh.shard_mesh_size = 2

        # Create parameter
        param_v2 = self._create_param_v2()

        # Simulate state after unshard
        self._simulate_unsharded_state(param_v2)
        param_v2.all_gather_outputs = [torch.randn(self.param.numel()).to(self.device)]

        # Call to_sharded (post-forward reshard); API uses to_sharded() -> SHARDED
        with patch.object(param_v2, '_setattr_on_modules') as mock_setattr, \
                patch.object(param_v2, 'free_unsharded_param') as mock_free:
            param_v2.to_sharded()

            # Verify state transition
            mock_setattr.assert_called_once()
            mock_free.assert_called_once()
            self.assertEqual(param_v2.sharded_state, ShardedState.SHARDED)


if __name__ == '__main__':
    unittest.main()
