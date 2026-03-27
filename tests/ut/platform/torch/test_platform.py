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
"""Unit tests for TorchPlatform core functionality.

This module contains comprehensive unit tests for the core business logic
of the TorchPlatform class, covering device management, distributed
communication primitives, parameter handling, and tensor operations.
"""
import os
import unittest
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import torch

from hyper_parallel.platform.torch.platform import TorchPlatform
from hyper_parallel.platform.torch.dtensor import DTensorBase


class TestTorchPlatformCore(unittest.TestCase):
    """Unit tests for TorchPlatform core functionality.
    
    Tests cover device and distributed environment management,
    distributed communication primitives, parameter management,
    and tensor operations.
    """

    def setUp(self):
        """Set up test fixtures before each test method.
        
        Configures the environment and initializes the TorchPlatform instance.
        """
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.platform = TorchPlatform()

    @mock.patch('hyper_parallel.platform.torch.platform.TorchPlatform.get_device_handle')
    def test_device_type(self, mock_get_device_handle):
        """Test device type detection logic.
        
        Verifies that the platform correctly identifies different device types
        (NPU and CUDA) based on the device handle.
        
        Args:
            mock_get_device_handle: Mock for the get_device_handle method.
        """
        # Test NPU device
        mock_get_device_handle.return_value = torch.npu
        self.assertEqual(self.platform.device_type(), "npu")

        # Test CUDA device
        mock_get_device_handle.return_value = torch.cuda
        self.assertEqual(self.platform.device_type(), "cuda")

    @mock.patch('hyper_parallel.platform.torch.platform._get_default_group')
    @mock.patch('torch.distributed.init_process_group')
    def test_init_process_group(self, mock_init, mock_get_default):
        """Test distributed process group initialization logic.
        
        Verifies that the platform properly initializes the distributed
        environment when not already initialized, and avoids reinitialization
        when already initialized.
        
        Args:
            mock_init: Mock for torch.distributed.init_process_group.
            mock_get_default: Mock for _get_default_group function.
        """
        # Test initialization when not initialized
        mock_get_default.side_effect = RuntimeError("No default group")
        TorchPlatform.init_process_group(backend="hccl", init_method="env://")
        mock_init.assert_called_once()

        # Test no reinitialization when already initialized
        mock_get_default.reset_mock()
        mock_get_default.side_effect = None
        mock_init.reset_mock()
        mock_get_default.return_value = MagicMock()
        TorchPlatform.init_process_group()
        mock_init.assert_not_called()

    @mock.patch('hyper_parallel.platform.torch.platform.TorchPlatform.get_rank')
    @mock.patch('torch.distributed.new_group')
    def test_split_group(self, mock_new_group, mock_get_rank):
        """Test process group splitting logic.
        
        Verifies that the platform correctly splits the default process group
        into subgroups based on provided rank lists.
        
        Args:
            mock_new_group: Mock for torch.distributed.new_group.
            mock_get_rank: Mock for TorchPlatform.get_rank.
        """
        mock_get_rank.return_value = 2
        mock_group = MagicMock()
        mock_new_group.return_value = mock_group

        split_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]
        result = TorchPlatform.split_group(split_ranks=split_ranks)

        self.assertEqual(mock_new_group.call_count, 4)
        self.assertEqual(result, mock_group)

        # Test exception cases
        with self.assertRaises(ValueError):
            TorchPlatform.split_group(split_ranks=[])

        with self.assertRaises(ValueError):
            TorchPlatform.split_group(split_ranks=None)

    @mock.patch('torch.distributed.nn.functional.all_gather')
    def test_differentiable_all_gather_concat(self, mock_all_gather):
        """Test differentiable all_gather and concatenation logic.
        
        Verifies that the platform correctly performs all_gather operation
        and concatenates results along the specified dimension.
        
        Args:
            mock_all_gather: Mock for torch.distributed.nn.functional.all_gather.
        """
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mock_output = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        ]
        mock_all_gather.return_value = mock_output

        result = TorchPlatform.differentiable_all_gather_concat(
            tensor, group=None, concat_size=2, concat_dim=0
        )

        self.assertEqual(result.shape, (4, 2))
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        self.assertTrue(torch.allclose(result, expected))

    @mock.patch('torch.distributed.nn.functional.all_reduce')
    def test_differentiable_all_reduce(self, mock_all_reduce):
        """Test differentiable all_reduce logic.
        
        Verifies that the platform correctly performs all_reduce operation
        with the specified reduction operation.
        
        Args:
            mock_all_reduce: Mock for torch.distributed.nn.functional.all_reduce.
        """
        tensor = torch.tensor([1.0, 2.0, 3.0])
        mock_result = torch.tensor([2.0, 4.0, 6.0])
        mock_all_reduce.return_value = mock_result

        # Test string operation type
        result = TorchPlatform.differentiable_all_reduce(tensor, op='sum', group=None)
        self.assertTrue(torch.allclose(result, mock_result))

    @mock.patch('torch.distributed.nn.functional.reduce_scatter')
    @mock.patch('torch.chunk')
    @mock.patch('torch.empty')
    def test_differentiable_reduce_scatter(self, mock_empty, mock_chunk, mock_reduce_scatter):
        """Test differentiable reduce_scatter logic.
        
        Verifies that the platform correctly performs reduce_scatter operation
        with both sum and average reduction operations.
        
        Args:
            mock_empty: Mock for torch.empty.
            mock_chunk: Mock for torch.chunk.
            mock_reduce_scatter: Mock for torch.distributed.nn.functional.reduce_scatter.
        """
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mock_chunks = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        mock_empty_tensor = torch.tensor([0.0, 0.0])
        mock_result = torch.tensor([3.0, 7.0])

        mock_chunk.return_value = mock_chunks
        mock_empty.return_value = mock_empty_tensor
        mock_reduce_scatter.return_value = mock_result

        # Test sum operation
        result = TorchPlatform.differentiable_reduce_scatter(
            tensor, dev_num=2, axis=0, op='sum', group=None
        )
        self.assertTrue(torch.allclose(result, mock_result))

        # Test avg operation (needs additional division by device count)
        mock_reduce_scatter.return_value = torch.tensor([6.0, 14.0])  # sum result
        result = TorchPlatform.differentiable_reduce_scatter(
            tensor, dev_num=2, axis=0, op='avg', group=None
        )
        expected_avg = torch.tensor([3.0, 7.0])  # sum / 2
        self.assertTrue(torch.allclose(result, expected_avg))

    @mock.patch('torch.distributed.all_reduce')
    def test_all_reduce_non_contiguous(self, mock_all_reduce):
        """Test all_reduce handling of non-contiguous tensors.
        
        Verifies that the platform correctly handles non-contiguous tensors
        by converting them to contiguous before performing all_reduce.
        
        Args:
            mock_all_reduce: Mock for torch.distributed.all_reduce.
        """
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        non_contiguous = tensor[:, ::2]  # Create non-contiguous tensor
        mock_group_info = MagicMock()
        mock_group_info.group = None
        mock_handle = MagicMock()
        mock_all_reduce.return_value = mock_handle

        result, _ = TorchPlatform.all_reduce(non_contiguous, mock_group_info, async_op=False)

        # Verify tensor is converted to contiguous
        self.assertTrue(result.is_contiguous())
        mock_all_reduce.assert_called_once()

    def test_search_parameter_by_name(self):
        """Test parameter search by name logic.
        
        Verifies that the platform correctly searches for parameters
        in nested model structures using dot notation.
        """

        # Create nested model structure
        class InnerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner_weight = torch.nn.Parameter(torch.randn(2, 2))

        class OuterModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(2, 2))
                self.inner = InnerModel()

        model = OuterModel()

        # Test direct parameter
        result = TorchPlatform.search_parameter_by_name(model, "weight")
        self.assertEqual(result, (model, "weight", model.weight))

        # Test nested parameter
        result = TorchPlatform.search_parameter_by_name(model, "inner.inner_weight")
        self.assertEqual(result, (model.inner, "inner_weight", model.inner.inner_weight))

        # Test self. prefix
        result = TorchPlatform.search_parameter_by_name(model, "self.weight")
        self.assertEqual(result, (model, "weight", model.weight))

        # Test non-existent parameter
        result = TorchPlatform.search_parameter_by_name(model, "non_existent")
        self.assertIsNone(result)

    def test_update_parameter_by_name(self):
        """Test parameter update by name logic.
        
        Verifies that the platform correctly updates model parameters
        using the result from search_parameter_by_name.
        """

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(2, 2))

        model = SimpleModel()
        old_param = model.weight
        new_param = torch.nn.Parameter(torch.randn(2, 2))

        search_result = TorchPlatform.search_parameter_by_name(model, "weight")
        TorchPlatform.update_parameter_by_name(model, search_result, new_param)

        self.assertIsNot(model.weight, old_param)
        self.assertIs(model.weight, new_param)

    @mock.patch('hyper_parallel.platform.torch.platform.Parameter')
    @mock.patch('hyper_parallel.core.dtensor.dtensor.DTensor.from_local')
    @mock.patch('hyper_parallel.core.dtensor.layout._get_slice_tensor_by_layout')
    def test_set_layout_into_parameter(self, mock_get_slice, mock_dtensor_from_local, mock_parameter):
        """Test parameter layout setting logic.
        
        Verifies that the platform correctly sets tensor layouts into parameters
        and handles error cases appropriately.
        
        Args:
            mock_get_slice: Mock for _get_slice_tensor_by_layout function.
            mock_dtensor_from_local: Mock for DTensor.from_local method.
            mock_parameter: Mock for Parameter class.
        """
        param = MagicMock()
        param.requires_grad = True
        mock_layout = MagicMock()
        mock_slice = torch.randn(2, 4)
        mock_dtensor = MagicMock()
        mock_new_param = MagicMock()

        mock_get_slice.return_value = mock_slice
        mock_dtensor_from_local.return_value = mock_dtensor
        mock_parameter.return_value = mock_new_param

        result = TorchPlatform.set_layout_into_parameter(param, mock_layout)

        # Verify call flow
        mock_get_slice.assert_called_once_with(param, mock_layout)
        mock_dtensor_from_local.assert_called_once_with(mock_slice, mock_layout.mesh, mock_layout.alias_placements)
        mock_parameter.assert_called_once_with(mock_dtensor, requires_grad=True)
        self.assertEqual(result, mock_new_param)

        # Test case where parameter is already a DTensor
        with self.assertRaises(ValueError):
            from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=import-outside-toplevel

            TorchPlatform.set_layout_into_parameter(MagicMock(spec=DTensor), mock_layout)

    def test_cast_fp_tensor(self):
        """Test floating-point tensor type casting logic.
        
        Verifies that the platform correctly casts tensors to different
        floating-point types and handles edge cases.
        """
        # Test different floating point type conversions
        tensor32 = torch.randn(2, 2, dtype=torch.float32)
        result = self.platform.cast_fp_tensor(torch.float16, tensor32)
        self.assertEqual(result.dtype, torch.float16)

        # Test no conversion for same type
        result = self.platform.cast_fp_tensor(torch.float32, tensor32)
        self.assertIs(result, tensor32)  # Should return the same object

        # Test no conversion for non-floating point types
        tensor_int = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = self.platform.cast_fp_tensor(torch.float32, tensor_int)
        self.assertIs(result, tensor_int)  # Should return the same object

        # Test no conversion for non-tensor types
        non_tensor = [1, 2, 3]
        result = self.platform.cast_fp_tensor(torch.float32, non_tensor)
        self.assertIs(result, non_tensor)  # Should return the same object

    def test_apply_to_tensors(self):
        """Test recursive tensor processing logic.
        
        Verifies that the platform correctly applies functions recursively
        to tensors within nested data structures.
        """
        # Create test data with different container types
        test_data = {
            "tensor1": torch.tensor([1.0, 2.0]),
            "list": [torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])],
            "nested": {
                "tensor2": torch.tensor([7.0, 8.0]),
                "value": 10
            },
            "value": 5
        }

        # Define processing function: multiply tensors by 2
        def double_tensor(t):
            return t * 2

        # Apply processing function
        result = self.platform.apply_to_tensors(double_tensor, test_data)

        # Verify results
        self.assertTrue(torch.allclose(result["tensor1"], torch.tensor([2.0, 4.0])))
        self.assertTrue(torch.allclose(result["list"][0], torch.tensor([6.0, 8.0])))
        self.assertTrue(torch.allclose(result["list"][1], torch.tensor([10.0, 12.0])))
        self.assertTrue(torch.allclose(result["nested"]["tensor2"], torch.tensor([14.0, 16.0])))
        self.assertEqual(result["nested"]["value"], 10)  # Non-tensor values remain unchanged
        self.assertEqual(result["value"], 5)  # Non-tensor values remain unchanged
