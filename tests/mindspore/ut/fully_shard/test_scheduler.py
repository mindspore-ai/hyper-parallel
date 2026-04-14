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
"""Unit tests for MindSpore fully_shard scheduler compatibility behavior."""

# pylint: disable=protected-access

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import mindspore as ms

from hyper_parallel.platform.mindspore.fully_shard.scheduler import MindSporeHSDPSchedulerV2


def _make_scheduler():
    """Create a lightweight scheduler without invoking the full constructor."""
    scheduler = object.__new__(MindSporeHSDPSchedulerV2)
    scheduler.modules = []
    scheduler.platform = "platform"
    scheduler.device = "npu"
    scheduler.config = SimpleNamespace(mesh=None)
    scheduler.mesh = None
    scheduler._get_managed_params = MagicMock(return_value=[])
    return scheduler


class FakeMesh:
    """Minimal mesh stub exposing only the hash used by compatibility mode."""

    def __init__(self, mesh_hash):
        self._mesh_hash = mesh_hash

    def to_hash(self):
        return self._mesh_hash


class TestMindSporeScheduler(unittest.TestCase):
    """Test scheduler compatibility-mode mesh resolution and hook wrapping."""

    @patch("hyper_parallel.platform.mindspore.fully_shard.scheduler.MindSporeHSDPStateV2")
    @patch("hyper_parallel.platform.mindspore.fully_shard.scheduler.DDPMeshInfo")
    def test_new_cell_state_uses_compat_mesh_for_mesh_none(self, mock_ddp_mesh_info, mock_state_ctor):
        """mesh=None should reuse the shared DTensor mesh carried by managed parameters."""
        scheduler = _make_scheduler()
        compat_mesh = FakeMesh("mesh-hash")
        scheduler._get_managed_params.return_value = ["p0", "p1"]
        mock_ddp_mesh_info.return_value = "compat-mesh-info"

        with patch(
            "hyper_parallel.platform.mindspore.fully_shard.scheduler.get_dtensor_managed_mesh",
            side_effect=[compat_mesh, compat_mesh],
        ):
            MindSporeHSDPSchedulerV2._new_cell_state(scheduler)

        mock_ddp_mesh_info.assert_called_once_with(mesh=compat_mesh, replicate_mesh_dim=0)
        mock_state_ctor.assert_called_once_with(
            scheduler.modules,
            "compat-mesh-info",
            scheduler.config,
            scheduler.platform,
            scheduler.device,
        )
        self.assertEqual(scheduler.mesh_info, "compat-mesh-info")

    def test_new_cell_state_rejects_mixed_compat_meshes(self):
        """mesh=None compatibility mode should reject DTensor params with different meshes."""
        scheduler = _make_scheduler()
        mesh_a = FakeMesh("mesh-a")
        mesh_b = FakeMesh("mesh-b")
        scheduler._get_managed_params.return_value = ["p0", "p1"]

        with patch(
            "hyper_parallel.platform.mindspore.fully_shard.scheduler.get_dtensor_managed_mesh",
            side_effect=[mesh_a, mesh_b],
        ), self.assertRaisesRegex(ValueError, "share the same mesh"):
            MindSporeHSDPSchedulerV2._new_cell_state(scheduler)

    @patch("hyper_parallel.platform.mindspore.fully_shard.scheduler.PostBackwardFunction.apply")
    def test_register_post_backward_hook_wraps_full_pytree(self, mock_apply):
        """PostBackwardFunction should see the full flattened args/kwargs pytree, not only grad-requiring tensors."""
        scheduler = _make_scheduler()
        grad_tensor = ms.Tensor([1.0], ms.float32)
        grad_tensor.requires_grad = True
        wrapped_tensor = ms.Tensor([2.0], ms.float32)
        mock_apply.return_value = (wrapped_tensor, "wrapped-b", "wrapped-k")

        args, kwargs = MindSporeHSDPSchedulerV2._register_post_backward_hook(
            scheduler,
            args=(grad_tensor, "arg-b"),
            kwargs={"kw": "arg-k"},
        )

        mock_apply.assert_called_once_with(scheduler, grad_tensor, "arg-b", "arg-k")
        self.assertTrue(args[0].requires_grad)
        self.assertEqual(args[0].asnumpy().tolist(), [2.0])
        self.assertEqual(args[1], "wrapped-b")
        self.assertEqual(kwargs, {"kw": "wrapped-k"})


if __name__ == "__main__":
    unittest.main()
