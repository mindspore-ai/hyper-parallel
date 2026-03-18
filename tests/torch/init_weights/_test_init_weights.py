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
"""Test: init_empty_weights -> fully_shard -> materialize -> init, verify consistency."""

import os
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch
from torch import nn
import torch.distributed as dist
import torch_npu  # pylint: disable=W0611

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.init_weights import init_empty_weights
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.utils import init_dist


class SimpleNet(nn.Module):
    """Test network with nn.Linear, nn.Parameter, and registered buffer."""
    def __init__(self, hidden_size=32):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.extra_weight = nn.Parameter(torch.ones(hidden_size, hidden_size))
        self.register_buffer("scale", torch.ones(hidden_size))

    def forward(self, x):
        x = self.linear2(torch.relu(self.linear1(x)))
        x = torch.matmul(x, self.extra_weight)
        return x * self.scale


def _create_sharded_model(rank, world_size, include_buffers=False):
    """Create a sharded SimpleNet: init_empty_weights -> fully_shard -> materialize."""
    device = torch.device(f"npu:{rank}")

    with init_empty_weights(include_buffers=include_buffers):
        model = SimpleNet(32)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    model = fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)

    with torch.no_grad():
        for p in model.parameters():
            assert isinstance(p, DTensor), f"param expected to be DTensor after fully_shard, got {type(p)}"
            local = p._local_tensor  # pylint: disable=W0212
            assert local.is_meta, f"param expected on meta after init_empty_weights, got {local.device}"
            p._local_tensor = torch.empty(local.shape, dtype=local.dtype, device=device)  # pylint: disable=W0212
        if include_buffers:
            for module in model.modules():
                for buf_name, buf in module._buffers.items():  # pylint: disable=W0212
                    if buf is None:
                        continue
                    assert buf.is_meta, f"buffer '{buf_name}' expected on meta, got {buf.device}"
                    module._buffers[buf_name] = torch.empty(buf.shape, dtype=buf.dtype, device=device)  # pylint: disable=W0212

    return model


def _assert_shards_differ(model, rank, world_size):
    """Assert DTensor params hold different shards across ranks, and buffers are identical across ranks."""
    device = torch.device(f"npu:{rank}")
    for name, val in model.state_dict().items():
        if isinstance(val, DTensor):
            local = val.to_local()
            gathered_list = [torch.zeros_like(local) for _ in range(world_size)]
            dist.all_gather(gathered_list, local)
            for other_rank in range(world_size):
                if other_rank != rank:
                    assert not torch.equal(
                        local,
                        gathered_list[other_rank]
                    ), f"param '{name}' rank {rank} == rank {other_rank}"
        else:
            assert not val.is_meta, f"buffer '{name}' is still on meta device"
            assert val.device == device, f"buffer '{name}' expected on {device}, got {val.device}"
            gathered_list = [torch.zeros_like(val) for _ in range(world_size)]
            dist.all_gather(gathered_list, val)
            for other_rank in range(world_size):
                if other_rank != rank:
                    assert torch.equal(
                        val,
                        gathered_list[other_rank]
                    ), f"buffer '{name}' rank {rank} != rank {other_rank}, expected identical"


def test_init_weights_consistency():
    """
    Feature: init_empty_weights -> fully_shard -> in-place init
    Description: Verify in-place init (normal_, uniform_) on params and buffers
                 with different ranks holding different shards.
    Expectation: run successfully
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    model = _create_sharded_model(rank, world_size, include_buffers=True)

    torch.manual_seed(42)
    for p in model.parameters():
        nn.init.normal_(p, mean=0.0, std=0.02)
    for buf in model.buffers():
        nn.init.uniform_(buf)

    _assert_shards_differ(model, rank, world_size)


def test_init_weights_with_randn_like():
    """
    Feature: init_empty_weights -> fully_shard -> non-in-place random ops
    Description: Verify non-in-place random ops (randn_like, rand_like)
                 dispatch correctly on DTensors and buffers.
    Expectation: run successfully
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()

    model = _create_sharded_model(rank, world_size, include_buffers=True)

    torch.manual_seed(42)
    with torch.no_grad():
        for p in model.parameters():
            new_dt = torch.randn_like(p)
            p._local_tensor = new_dt.to_local() * 0.02  # pylint: disable=W0212
        for buf in model.buffers():
            buf.data = torch.rand_like(buf)

    _assert_shards_differ(model, rank, world_size)
