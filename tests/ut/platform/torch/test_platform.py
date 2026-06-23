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
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

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

    def test_dtensor_data_setter_updates_wrapper_and_local_tensor(self):
        """Assigning ``dtensor.data = x`` should synchronize wrapper and local tensor payloads."""
        class FakeDataDescriptor:
            def __init__(self):
                self.set_calls = []

            def __get__(self, obj, objtype=None):
                return "fake-data"

            def __set__(self, obj, value):
                self.set_calls.append((obj, value))

        fake_descriptor = FakeDataDescriptor()
        fake_tensor_cls = SimpleNamespace(data=fake_descriptor)
        fake_dtensor = SimpleNamespace(_local_tensor=object())

        with patch("hyper_parallel.platform.torch.dtensor.Tensor", fake_tensor_cls):
            DTensorBase.data.fset(fake_dtensor, "payload")

        self.assertEqual(
            fake_descriptor.set_calls,
            [(fake_dtensor, "payload"), (fake_dtensor._local_tensor, "payload")],
        )

    def test_dtensor_data_setter_uses_local_tensor_for_dtensor_input(self):
        """Assigning another DTensor should propagate its local shard payload."""
        class FakeDataDescriptor:
            def __init__(self):
                self.set_calls = []

            def __get__(self, obj, objtype=None):
                return "fake-data"

            def __set__(self, obj, value):
                self.set_calls.append((obj, value))

        class FakeInputDTensor:
            def __init__(self):
                self._local_tensor = "local-shard"

            def to_local(self):
                return self._local_tensor

        fake_descriptor = FakeDataDescriptor()
        fake_tensor_cls = SimpleNamespace(data=fake_descriptor)
        fake_dtensor = SimpleNamespace(_local_tensor=object())
        input_dtensor = FakeInputDTensor()

        with patch("hyper_parallel.platform.torch.dtensor.Tensor", fake_tensor_cls):
            with patch("hyper_parallel.platform.torch.dtensor.DTensorBase", FakeInputDTensor):
                DTensorBase.data.fset(fake_dtensor, input_dtensor)

        self.assertEqual(
            fake_descriptor.set_calls,
            [(fake_dtensor, "local-shard"), (fake_dtensor._local_tensor, "local-shard")],
        )

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

    @mock.patch('torch.distributed.all_gather_into_tensor')
    def test_all_gather_single_returns_output_and_handle(self, mock_all_gather):
        """Test direct all_gather_single wrapper with async handle propagation."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mock_group = MagicMock()
        fake_handle = MagicMock()

        def fake_all_gather(output, input_tensor, group=None, async_op=False):
            self.assertIs(group, mock_group)
            self.assertTrue(async_op)
            output.copy_(torch.cat([input_tensor, input_tensor], dim=0))
            return fake_handle

        mock_all_gather.side_effect = fake_all_gather

        output, handle = TorchPlatform.all_gather_single(
            tensor,
            [4, 2],
            mock_group,
            async_op=True,
        )

        self.assertIs(handle, fake_handle)
        self.assertEqual(tuple(output.shape), (4, 2))
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(torch.allclose(output, expected))

    @mock.patch('torch.distributed.reduce_scatter_tensor')
    def test_reduce_scatter_single_returns_output_and_handle(self, mock_reduce_scatter):
        """Test direct reduce_scatter_single wrapper with async handle propagation."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        mock_group = MagicMock()
        fake_handle = MagicMock()

        def fake_reduce_scatter(output, input_tensor, group=None, async_op=False):
            self.assertIs(group, mock_group)
            self.assertTrue(async_op)
            output.copy_(input_tensor[: output.shape[0]])
            return fake_handle

        mock_reduce_scatter.side_effect = fake_reduce_scatter

        output, handle = TorchPlatform.reduce_scatter_single(
            tensor,
            [2, 2],
            mock_group,
            async_op=True,
        )

        self.assertIs(handle, fake_handle)
        self.assertEqual(tuple(output.shape), (2, 2))
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(torch.allclose(output, expected))

    @mock.patch('torch.distributed.P2POp')
    def test_p2p_op_maps_op_type_string_to_callable(self, mock_p2p_op):
        """p2p_op should translate the 'isend'/'irecv' string to the dist callable."""
        tensor = torch.tensor([1.0, 2.0])
        mock_group = MagicMock()
        sentinel = MagicMock()
        mock_p2p_op.return_value = sentinel

        send_desc = TorchPlatform.p2p_op("isend", tensor, 3, mock_group)
        self.assertIs(send_desc, sentinel)
        mock_p2p_op.assert_called_with(torch.distributed.isend, tensor, 3, mock_group)

        TorchPlatform.p2p_op("irecv", tensor, 5)
        mock_p2p_op.assert_called_with(torch.distributed.irecv, tensor, 5, None)

    def test_p2p_op_rejects_unknown_op_type(self):
        """p2p_op should raise on an op_type other than 'isend'/'irecv'."""
        with self.assertRaises(ValueError):
            TorchPlatform.p2p_op("scatter", torch.tensor([1.0]), 0)

    def test_batch_isend_irecv_empty_returns_none(self):
        """An empty op list should short-circuit to None (no launch)."""
        self.assertIsNone(TorchPlatform.batch_isend_irecv([]))

    @mock.patch('torch.distributed.batch_isend_irecv')
    def test_batch_isend_irecv_wraps_works_into_single_wait_handle(self, mock_batch):
        """The returned handle's wait() should wait every per-op work once."""
        work_a, work_b = MagicMock(), MagicMock()
        mock_batch.return_value = [work_a, work_b]

        ops = [MagicMock(), MagicMock()]
        handle = TorchPlatform.batch_isend_irecv(ops)
        mock_batch.assert_called_once_with(ops)

        work_a.wait.assert_not_called()
        handle.wait()
        work_a.wait.assert_called_once_with()
        work_b.wait.assert_called_once_with()

    def test_differentiable_async_allgather_wait_immediate_backward(self):
        """Async all-gather wait should return a real gradient when handle_box is None."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        work = MagicMock()
        work.wait = MagicMock()
        out_perm = torch.cat([x.detach(), x.detach()], dim=0)
        fake_handle = MagicMock()

        def fake_reduce_scatter(output, input_tensor, group=None, async_op=False):
            self.assertTrue(async_op)
            output.copy_(input_tensor[: output.shape[0]])
            return fake_handle

        with mock.patch('torch.distributed.reduce_scatter_tensor', side_effect=fake_reduce_scatter):
            output = TorchPlatform.differentiable_async_allgather_wait(
                x,
                work,
                out_perm,
                group=MagicMock(),
                world_size=2,
                gather_dim=0,
                handle_box=None,
            )
            output.sum().backward()

        self.assertEqual(work.wait.call_count, 1)
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))

    def test_differentiable_async_allgather_wait_defers_backward_handle(self):
        """Async all-gather wait should defer reverse reduce-scatter when handle_box is provided."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        work = MagicMock()
        work.wait = MagicMock()
        out_perm = torch.cat([x.detach(), x.detach()], dim=0)
        fake_handle = MagicMock()
        handle_box = []

        def fake_reduce_scatter(output, input_tensor, group=None, async_op=False):
            self.assertTrue(async_op)
            output.copy_(input_tensor[: output.shape[0]])
            return fake_handle

        with mock.patch('torch.distributed.reduce_scatter_tensor', side_effect=fake_reduce_scatter):
            output = TorchPlatform.differentiable_async_allgather_wait(
                x,
                work,
                out_perm,
                group=MagicMock(),
                world_size=2,
                gather_dim=0,
                handle_box=handle_box,
            )
            output.sum().backward()

        self.assertEqual(work.wait.call_count, 1)
        self.assertEqual(len(handle_box), 1)
        handle, output_buffer, gather_dim = handle_box[0]
        self.assertIs(handle, fake_handle)
        self.assertEqual(gather_dim, 0)
        self.assertTrue(torch.allclose(output_buffer, torch.ones_like(output_buffer)))
        self.assertTrue(torch.allclose(x.grad, torch.zeros_like(x)))

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
