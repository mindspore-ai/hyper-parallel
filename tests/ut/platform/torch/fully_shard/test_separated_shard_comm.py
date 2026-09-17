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
# ==========================================================================
"""Tests for the Torch-only HSDP shard communicator switch."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from torch import nn

from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.core.fully_shard.utils import FSDPMeshInfo
from hyper_parallel.platform.torch.fully_shard.param import TorchHSDPParamV2
from hyper_parallel.platform.torch.fully_shard.state import TorchHSDPStateV2


class _Module(HSDPModule, nn.Module):
    def __init__(self, source_group, size=2):
        nn.Module.__init__(self)
        mesh_info = object.__new__(FSDPMeshInfo)
        mesh_info.shard_process_group = source_group
        param = object.__new__(TorchHSDPParamV2)
        param.mesh_info = mesh_info
        param.shard_world_size = size
        mesh_info.reduce_scatter_process_group = source_group
        state = object.__new__(TorchHSDPStateV2)
        state.hsdp_params = [param]
        state.is_shard = True
        scheduler_ctx = SimpleNamespace(lazy_init_done=False, separated_shard_groups={})
        state.scheduler_ctx = scheduler_ctx
        self.hsdp_scheduler = SimpleNamespace(
            hsdp_state=state, scheduler_ctx=scheduler_ctx
        )


class TestSeparatedShardComm(unittest.TestCase):
    """Exercise the public recursive API without initializing distributed hardware."""

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_backend", return_value="hccl")
    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1])
    def test_group_reuse_toggle_and_mesh_preservation(self, _ranks, _backend, new_group):
        """Layers share one RS group while AG and the DeviceMesh stay unchanged."""
        original, separate = object(), object()
        new_group.return_value = separate
        model = _Module(original)
        model.child = _Module(original)
        model.set_separated_shard_comm()
        new_group.assert_not_called()
        model.set_separated_shard_comm(True)
        model.set_separated_shard_comm(True)
        self.assertEqual(new_group.call_count, 2)
        new_group.assert_any_call(ranks=[0, 1], backend="hccl", use_local_synchronization=True)
        for module in (model, model.child):
            param = module.hsdp_scheduler.hsdp_state.hsdp_params[0]
            self.assertIs(param.mesh_info.reduce_scatter_process_group, separate)
            self.assertIs(param.mesh_info.shard_process_group, original)
        model.set_separated_shard_comm(False)
        self.assertIs(
            model.child.hsdp_scheduler.hsdp_state.hsdp_params[0].mesh_info.reduce_scatter_process_group,
            original,
        )
        model.set_separated_shard_comm(True)
        self.assertEqual(new_group.call_count, 2)

    @patch("torch.distributed.new_group")
    def test_validation_precedes_all_mutation(self, new_group):
        """An initialized child rejects the entire request before groups are created."""
        model = _Module(object())
        model.child = _Module(object())
        for invalid in (None, 1, "true"):
            with self.assertRaises(ValueError):
                model.set_separated_shard_comm(invalid)
        model.child.hsdp_scheduler.scheduler_ctx.lazy_init_done = True
        with self.assertRaises(ValueError):
            model.set_separated_shard_comm(True)
        new_group.assert_not_called()

    @patch("torch.distributed.new_group")
    def test_size_one_does_not_allocate(self, new_group):
        """Local shards do not require another communicator."""
        model = _Module(object(), size=1)
        model.set_separated_shard_comm(True)
        new_group.assert_not_called()

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_backend", return_value="hccl")
    @patch("torch.distributed.get_process_group_ranks", return_value=[2, 3])
    def test_second_list_root_controls_shared_state(self, _ranks, _backend, new_group):
        """The non-first root of a list-wrapped unit may configure its shared state."""
        first = _Module(object())
        second = _Module(object())
        second.hsdp_scheduler = first.hsdp_scheduler
        second.set_separated_shard_comm(True)
        param = first.hsdp_scheduler.hsdp_state.hsdp_params[0]
        self.assertIs(param.mesh_info.reduce_scatter_process_group, new_group.return_value)
