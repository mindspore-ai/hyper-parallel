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
"""Unit tests for MindSpore fully_shard mixed-precision behavior."""
# pylint: disable=protected-access

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("mindspore")
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_for_fully_shard,
)

ensure_mindspore_platform_for_fully_shard()

import mindspore as ms

from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.fully_shard.param import MindSporeHSDPParamV2
from hyper_parallel.platform.mindspore.fully_shard.state import MindSporeHSDPStateV2
from tests.ut.platform.mindspore.fully_shard.conftest import MindSporeFullyShardUnitTest


def _bare_param(policy=None):
    """Create a lightweight parameter wrapper for dtype and gradient tests."""
    hsdp_param = object.__new__(MindSporeHSDPParamV2)
    hsdp_param.mp_policy = policy or MixedPrecisionPolicy()
    hsdp_param.orig_dtype = ms.float32
    hsdp_param.param_dtype = None
    hsdp_param.reduce_dtype = None
    hsdp_param.offload_to_cpu = False
    hsdp_param.pin_memory = False
    hsdp_param.sharded_size = (2,)
    hsdp_param.sharded_param = SimpleNamespace(
        dtype=ms.float32,
        grad=None,
        main_grad=None,
        _local_tensor=ms.Tensor([0.0, 0.0], ms.float32),
    )
    hsdp_param.unsharded_accumulated_grad = None
    hsdp_param._unsharded_param = SimpleNamespace(grad=None)
    hsdp_param.to_sharded_dtensor = MagicMock(side_effect=lambda tensor: SimpleNamespace(_local_tensor=tensor))
    hsdp_param._sharded_param_storage_dtype = MagicMock(return_value=ms.float32)
    return hsdp_param


class TestParamDtypes(MindSporeFullyShardUnitTest):
    """Test parameter-owned mixed-precision metadata."""

    def test_init_dtype_attrs_normalizes_redundant_casts(self):
        """Param/original and reduce/param equality should disable redundant casts."""
        hsdp_param = _bare_param()
        policy = MixedPrecisionPolicy(param_dtype=ms.float32, reduce_dtype=ms.float32)

        hsdp_param.init_dtype_attrs(policy)

        self.assertEqual(hsdp_param.orig_dtype, ms.float32)
        self.assertIsNone(hsdp_param.param_dtype)
        self.assertIsNone(hsdp_param.reduce_dtype)

    def test_reduce_comm_dtype_prefers_policy_then_gradient(self):
        """Communication dtype should be resolved by each parameter independently."""
        hsdp_param = _bare_param()
        grad = ms.Tensor([1.0], ms.float16)
        hsdp_param.reduce_dtype = ms.float32
        self.assertEqual(hsdp_param.reduce_comm_dtype(grad), ms.float32)
        hsdp_param.reduce_dtype = None
        self.assertEqual(hsdp_param.reduce_comm_dtype(grad), ms.float16)

    def test_state_initializes_each_parameter_without_uniform_dtype_constraint(self):
        """Per-dtype buckets allow heterogeneous parameter dtypes in one state."""
        param_a = MagicMock()
        param_b = MagicMock()
        state = object.__new__(MindSporeHSDPStateV2)
        state.hsdp_params = [param_a, param_b]
        state.mp_policy = MixedPrecisionPolicy(param_dtype=ms.float16)

        state._init_mp_dtypes()

        param_a.init_dtype_attrs.assert_called_once_with(state.mp_policy)
        param_b.init_dtype_attrs.assert_called_once_with(state.mp_policy)


class TestGradientApplication(MindSporeFullyShardUnitTest):
    """Test casting, assignment, accumulation, and CPU-offload reporting."""

    def test_apply_reduced_grad_casts_and_assigns_dtensor(self):
        """Reduced gradients should match parameter storage dtype before assignment."""
        hsdp_param = _bare_param()
        reduced_grad = ms.Tensor([1.0, 2.0], ms.float16)

        need_synchronize = hsdp_param.apply_reduced_grad(reduced_grad)

        self.assertFalse(need_synchronize)
        assigned = hsdp_param.to_sharded_dtensor.call_args.args[0]
        self.assertEqual(assigned.dtype, ms.float32)
        np.testing.assert_allclose(assigned.asnumpy(), np.array([1.0, 2.0], dtype=np.float32))

    def test_apply_reduced_grad_accumulates_without_view_inplace(self):
        """MindSpore accumulation should replace the gradient with a mint.add result."""
        hsdp_param = _bare_param()
        hsdp_param.sharded_param.grad = SimpleNamespace(
            _local_tensor=ms.Tensor([1.0, 2.0], ms.float32)
        )

        hsdp_param.apply_reduced_grad(ms.Tensor([3.0, 4.0], ms.float32))

        accumulated = hsdp_param.to_sharded_dtensor.call_args.args[0]
        np.testing.assert_allclose(accumulated.asnumpy(), np.array([4.0, 6.0], dtype=np.float32))

    def test_apply_reduced_grad_assigns_fp32_main_grad(self):
        """FP32-main-grad policy should leave ``param.grad`` empty."""
        policy = MixedPrecisionPolicy(apply_grad_on_fp32_main_grad=True)
        hsdp_param = _bare_param(policy)

        hsdp_param.apply_reduced_grad(ms.Tensor([1.0, 2.0], ms.float32))

        self.assertIsNone(hsdp_param.sharded_param.grad)
        self.assertIsNotNone(hsdp_param.sharded_param.main_grad)

    def test_apply_reduced_grad_reports_cpu_offload_sync(self):
        """Non-blocking CPU offload should tell the caller to synchronize."""
        hsdp_param = _bare_param()
        hsdp_param.offload_to_cpu = True
        hsdp_param.pin_memory = True
        reduced_grad = MagicMock()
        reduced_grad.reshape.return_value.narrow.return_value.view.return_value = reduced_grad
        reduced_grad.dtype = ms.float32
        reduced_grad.to.return_value = MagicMock(name="cpu-grad")

        need_synchronize = hsdp_param.apply_reduced_grad(reduced_grad)

        self.assertTrue(need_synchronize)
        reduced_grad.to.assert_called_once_with("cpu", non_blocking=True)


class TestGradientAccumulation(MindSporeFullyShardUnitTest):
    """Test full-gradient accumulation before communication."""

    def test_to_accumulated_grad_casts_and_clears_live_grad(self):
        """No-sync should retain a reduce-dtype gradient and clear the live reference."""
        hsdp_param = _bare_param()
        hsdp_param.reduce_dtype = ms.float32
        hsdp_param._unsharded_param.grad = ms.Tensor([1.0, 2.0], ms.float16)

        hsdp_param.to_accumulated_grad_if_needed()

        self.assertEqual(hsdp_param.unsharded_accumulated_grad.dtype, ms.float32)
        self.assertIsNone(hsdp_param._unsharded_param.grad)

    def test_accumulate_unsharded_grad_uses_mint_add(self):
        """A later micro-step should merge into the retained full gradient."""
        hsdp_param = _bare_param()
        hsdp_param.unsharded_accumulated_grad = ms.Tensor([1.0, 2.0], ms.float32)
        hsdp_param._unsharded_param.grad = ms.Tensor([3.0, 4.0], ms.float32)
        hsdp_param._to_local_unsharded_grad = MagicMock(
            return_value=hsdp_param._unsharded_param.grad
        )

        hsdp_param.accumulate_unsharded_grad_if_needed()

        np.testing.assert_allclose(
            hsdp_param.unsharded_accumulated_grad.asnumpy(),
            np.array([4.0, 6.0], dtype=np.float32),
        )
        self.assertIsNone(hsdp_param._unsharded_param.grad)


class TestCommunicationDtypes(MindSporeFullyShardUnitTest):
    """Test mint collective dtypes and result contexts without hardware."""

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_uses_parameter_reduce_dtype(self, mock_reduce_scatter):
        """Per-parameter RS should cast to its configured communication dtype."""
        hsdp_param = _bare_param()
        hsdp_param._unsharded_param = SimpleNamespace(
            grad=ms.Tensor([1.0, 2.0, 3.0, 4.0], ms.float16)
        )
        hsdp_param._to_local_unsharded_grad = MagicMock(side_effect=lambda grad: grad)
        hsdp_param.reduce_dtype = ms.float32
        hsdp_param.gradient_scaling_factor = None
        hsdp_param.is_sharded = True
        hsdp_param.shard_world_size = 2
        hsdp_param.mesh_info = SimpleNamespace(shard_process_group="fsdp")
        hsdp_param.hsdp_placement = SimpleNamespace(dim=0)
        hsdp_param._orig_size = (4,)
        hsdp_param.reduce_scatter_comm_ctx = SimpleNamespace(
            reduce_scatter_output=None,
            reduce_scatter_handle=None,
        )
        handle = MagicMock()
        mock_reduce_scatter.return_value = handle
        with patch(
            "hyper_parallel.platform.mindspore.fully_shard.param.FSDPMeshInfo",
            type(hsdp_param.mesh_info),
        ):
            hsdp_param.reduce_scatter_grad(reduce_op="sum")

        self.assertEqual(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output.dtype, ms.float32)
        self.assertIs(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_handle, handle)
        self.assertEqual(mock_reduce_scatter.call_args.kwargs["op"], "sum")

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.all_reduce")
    def test_all_reduce_uses_replicate_mesh_and_context(self, mock_all_reduce):
        """HSDP AR should consume the RS output and cache async work."""
        hsdp_param = _bare_param()
        hsdp_param.reduce_scatter_comm_ctx = SimpleNamespace(
            reduce_scatter_output=ms.Tensor([1.0, 2.0], ms.float32),
            reduce_scatter_handle=None,
        )
        hsdp_param.all_reduce_comm_ctx = SimpleNamespace(
            all_reduce_output=None,
            all_reduce_handle=None,
        )
        hsdp_param.replicate_world_size = 2
        hsdp_param.mesh_info = SimpleNamespace(replicate_process_group="dp")
        handle = MagicMock()
        mock_all_reduce.return_value = handle
        with patch(
            "hyper_parallel.platform.mindspore.fully_shard.param.DDPMeshInfo",
            type(hsdp_param.mesh_info),
        ):
            hsdp_param.all_reduce_grad(reduce_op="avg")

        self.assertIs(hsdp_param.all_reduce_comm_ctx.all_reduce_handle, handle)
        self.assertEqual(mock_all_reduce.call_args.kwargs["op"], "avg")
        self.assertEqual(mock_all_reduce.call_args.kwargs["group"], "dp")


if __name__ == "__main__":
    unittest.main()
