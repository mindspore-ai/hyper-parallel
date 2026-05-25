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
"""Unit tests for torch fully_shard parameter helper paths."""
# pylint: disable=protected-access

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=wrong-import-position
import torch

from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.hsdp_utils import GroupInfo, ShardedState
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.torch.fully_shard.param import TorchHSDPParamV2


def _new_param():
    """Create an uninitialized TorchHSDPParamV2 with common test fields."""
    hsdp_param = object.__new__(TorchHSDPParamV2)
    hsdp_param.all_gather_outputs = []
    hsdp_param.device = torch.device("cpu")
    hsdp_param.shard_size = 2
    hsdp_param.dp_size = 1
    hsdp_param.is_sharded = True
    hsdp_param.hsdp_placement = Shard(0)
    hsdp_param._orig_size = torch.Size((4,))
    hsdp_param._orig_param_is_dtensor = False
    hsdp_param._orig_dtensor_mesh = None
    hsdp_param._orig_dtensor_placements = None
    hsdp_param._sharded_param_data = torch.tensor([1.0, 2.0])
    hsdp_param.sharded_size = torch.Size((2,))
    hsdp_param.contiguous_sharded_stride = (1,)
    hsdp_param._sharding_spec = SimpleNamespace(mesh="mesh", placements=(Shard(0),))
    hsdp_param.sharded_param = SimpleNamespace(
        shape=torch.Size((2,)),
        grad=None,
        requires_grad=True,
        _local_tensor=torch.tensor([1.0, 2.0]),
    )
    hsdp_param._unsharded_param = SimpleNamespace(grad=None)
    hsdp_param.unsharded_accumulated_grad = None
    hsdp_param.mp_policy = MixedPrecisionPolicy()
    hsdp_param.offload_to_cpu = False
    hsdp_param.pin_memory = False
    hsdp_param.prefetch_handle = None
    hsdp_param._reduce_scatter_output = None
    hsdp_param.reduce_scatter_handle = None
    hsdp_param._grad = torch.ones(2)
    hsdp_param._all_reduce_output = None
    hsdp_param.all_reduce_handle = None
    hsdp_param.unsharded_group_info = GroupInfo("invalid", None, 1)
    hsdp_param.sharded_group_info = GroupInfo("invalid", None, 1)
    hsdp_param.sharded_state = ShardedState.SHARDED
    return hsdp_param


class TestTorchHSDPParamHelpers(unittest.TestCase):
    """Cover parameter helper behavior without constructing real device meshes."""

    def test_init_all_gather_outputs_reuse_and_force_recreate(self):
        """All-gather buffers should be reused unless recreation is requested."""
        hsdp_param = _new_param()
        existing = torch.empty(1)
        hsdp_param.all_gather_outputs = [existing]

        hsdp_param.init_all_gather_outputs([2], [torch.float32], 2, torch.device("cpu"))
        self.assertIs(hsdp_param.all_gather_outputs[0], existing)

        hsdp_param.init_all_gather_outputs(
            [2, 1], [torch.float32, torch.float16], 2, torch.device("cpu"), force_recreate=True
        )
        self.assertEqual([t.numel() for t in hsdp_param.all_gather_outputs], [4, 2])
        self.assertEqual(hsdp_param.all_gather_outputs[1].dtype, torch.float16)

    def test_get_unsharded_param_from_all_gather_output_plain_and_dtensor(self):
        """All-gather output should restore plain tensors and DTensor wrappers."""
        hsdp_param = _new_param()
        hsdp_param.all_gather_outputs = [torch.arange(4, dtype=torch.float32)]

        unsharded = hsdp_param._get_unsharded_param_from_all_gather_output()
        torch.testing.assert_close(unsharded, torch.arange(4, dtype=torch.float32))

        hsdp_param._orig_param_is_dtensor = True
        hsdp_param._orig_dtensor_mesh = "mesh"
        hsdp_param._orig_dtensor_placements = (Shard(0),)
        with patch("hyper_parallel.platform.torch.fully_shard.param.DTensor.from_local", return_value="dtensor") as mock_from:
            self.assertEqual(hsdp_param._get_unsharded_param_from_all_gather_output(), "dtensor")
        mock_from.assert_called_once()

    def test_get_unsharded_param_from_all_gather_output_requires_single_buffer(self):
        """All-gather output restoration should require exactly one buffer."""
        hsdp_param = _new_param()

        with self.assertRaisesRegex(AssertionError, "Expected 1 all_gather_output"):
            hsdp_param._get_unsharded_param_from_all_gather_output()

    def test_output_wait_and_clear_helpers(self):
        """Output accessors should wait handles once and clear cached tensors."""
        hsdp_param = _new_param()
        hsdp_param._reduce_scatter_output = "rs"
        hsdp_param.reduce_scatter_handle = MagicMock()
        hsdp_param._all_reduce_output = "ar"
        hsdp_param.all_reduce_handle = MagicMock()

        self.assertEqual(hsdp_param.reduce_scatter_output(), "rs")
        self.assertEqual(hsdp_param.all_reduce_output(), "ar")
        self.assertIsNone(hsdp_param.reduce_scatter_handle)
        self.assertIsNone(hsdp_param.all_reduce_handle)
        hsdp_param.clear_reduce_scatter_output()
        hsdp_param.clear_all_reduce_output()
        self.assertIsNone(hsdp_param._reduce_scatter_output)
        self.assertIsNone(hsdp_param._all_reduce_output)

    def test_apply_reduced_grad_assigns_and_accumulates(self):
        """Reduced gradients should assign new grads and accumulate existing grads."""
        hsdp_param = _new_param()
        hsdp_param.to_sharded_dtensor = MagicMock(side_effect=lambda tensor: SimpleNamespace(_local_tensor=tensor.clone()))
        hsdp_param._unsharded_param.grad = torch.ones(2)

        need_sync = hsdp_param.apply_reduced_grad(torch.tensor([1.0, 2.0]), torch.float16)

        self.assertFalse(need_sync)
        torch.testing.assert_close(hsdp_param.sharded_param.grad._local_tensor, torch.tensor([1.0, 2.0], dtype=torch.float16))
        self.assertIsNone(hsdp_param._unsharded_param.grad)

        hsdp_param._unsharded_param.grad = torch.ones(2)
        hsdp_param.apply_reduced_grad(torch.tensor([3.0, 4.0]), None)
        torch.testing.assert_close(
            hsdp_param.sharded_param.grad._local_tensor,
            torch.tensor([4.0, 6.0], dtype=torch.float16),
        )

    def test_apply_reduced_grad_uses_main_grad(self):
        """Reduced gradients should accumulate into fp32 main grad when enabled."""
        hsdp_param = _new_param()
        hsdp_param.mp_policy = MixedPrecisionPolicy(apply_grad_on_fp32_main_grad=True)
        hsdp_param.sharded_param.main_grad = SimpleNamespace(_local_tensor=torch.ones(2))
        hsdp_param.to_sharded_dtensor = MagicMock(side_effect=lambda tensor: SimpleNamespace(_local_tensor=tensor.clone()))

        hsdp_param.apply_reduced_grad(torch.tensor([2.0, 3.0]), torch.float32)

        torch.testing.assert_close(hsdp_param.sharded_param.main_grad._local_tensor, torch.tensor([3.0, 4.0]))
        self.assertIsNone(hsdp_param.sharded_param.grad)

    @patch("hyper_parallel.platform.torch.fully_shard.param.dist.all_reduce")
    def test_all_reduce_grad_single_rank_and_mocked_multi_rank(self, mock_all_reduce):
        """All-reduce should skip single-rank groups and launch for multi-rank groups."""
        hsdp_param = _new_param()
        grad = torch.ones(2)

        reduced, handle = hsdp_param.all_reduce_grad(grad=grad)
        self.assertIs(reduced, grad)
        self.assertIsNone(handle)
        mock_all_reduce.assert_not_called()

        hsdp_param.unsharded_group_info = GroupInfo("group", "process-group", 2)
        hsdp_param.dp_size = 2
        mock_all_reduce.return_value = "handle"
        reduced, handle = hsdp_param.all_reduce_grad(grad=grad, dtype=torch.float16)
        self.assertEqual(handle, "handle")
        self.assertEqual(reduced.dtype, torch.float16)
        mock_all_reduce.assert_called_once()

    def test_reset_sharded_param_meta_and_shape_mismatch(self):
        """Reset should materialize meta params and reject mismatched shard shapes."""
        hsdp_param = _new_param()
        meta_param = torch.nn.Parameter(torch.empty(2, device="meta"))
        hsdp_param._module_info = SimpleNamespace(module=SimpleNamespace(weight=meta_param), param_name="weight")
        hsdp_param.sharded_param = meta_param

        hsdp_param.reset_sharded_param()

        bad_param = torch.nn.Parameter(torch.ones(3))
        hsdp_param._module_info.module.weight = bad_param
        hsdp_param.sharded_param = SimpleNamespace(_local_tensor=torch.ones(2), _hsdp_param_initialized=True)
        hsdp_param._sharded_param_data = torch.ones(2)
        with self.assertRaisesRegex(AssertionError, "Expected sharded_size"):
            hsdp_param.reset_sharded_param()


if __name__ == "__main__":
    unittest.main()
