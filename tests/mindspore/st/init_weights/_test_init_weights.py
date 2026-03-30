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
"""Test: init_empty_weights -> fully_shard -> materialize -> init (MindSpore, multi-card)."""

import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

# pylint: disable=C0413
import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, mint
from mindspore.communication import get_group_size, get_rank
from mindspore.common.initializer import initializer

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.init_weights import init_empty_weights
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy

D.init()


_NPU = "Ascend"


class SimpleNet(nn.Cell):
    """Test network with mint.nn.Linear, Parameter, and registered buffer."""

    def __init__(self, hidden_size=32):
        super().__init__()
        self.linear1 = mint.nn.Linear(hidden_size, hidden_size)
        self.linear2 = mint.nn.Linear(hidden_size, hidden_size)
        self.extra_weight = ms.Parameter(initializer("ones", (hidden_size, hidden_size), ms.float32))
        self.register_buffer("scale", mint.ones((hidden_size,)))

    def construct(self, x):
        x = self.linear2(mint.relu(self.linear1(x)))
        x = mint.matmul(x, self.extra_weight)
        return x * self.scale


def _materialize_meta_locals(model):
    """Allocate real NPU storage for meta DTensor params after fully_shard."""
    for p in model.trainable_params():
        if isinstance(p, DTensor):
            local = p.to_local()
            if getattr(local, "is_meta", False):
                p.set_data(mint.empty(local.shape, dtype=local.dtype, device=_NPU))


def _create_sharded_model():
    """Create a sharded SimpleNet: init_empty_weights -> fully_shard -> materialize params."""
    world_size = get_group_size()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=True,
    )

    with init_empty_weights():
        model = SimpleNet(32)
        model = fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)

    for p in model.trainable_params():
        assert isinstance(p, DTensor), f"param expected DTensor after fully_shard, got {type(p)}"
        local = p.to_local()
        assert getattr(local, "is_meta", False), (
            f"param expected on meta after init_empty_weights, got {getattr(local, 'device', local)}"
        )

    _materialize_meta_locals(model)
    return model


def _assert_shards_differ(model, rank, world_size):
    """Assert DTensor trainable params hold different shards across ranks."""
    for param in model.trainable_params():
        assert isinstance(param, DTensor), f"trainable param expected DTensor, got {type(param)}"
        local = param.to_local()
        gathered = [mint.zeros_like(local) for _ in range(world_size)]
        mint.distributed.all_gather(gathered, local)
        for other_rank in range(world_size):
            if other_rank != rank:
                assert not (local == gathered[other_rank]).all(), (
                    f"param '{param.name}' rank {rank} == rank {other_rank}"
                )


def test_init_weights_with_randn_like():
    """
    Feature: init_empty_weights -> fully_shard -> randn_like style fill
    Description: Random fill on local shards via randn_like.
    Expectation: run successfully
    """
    rank = get_rank()

    model = _create_sharded_model()

    ms.manual_seed(42)
    for p in model.trainable_params():
        if isinstance(p, DTensor):
            new_dt = mint.randn_like(p)
            local_value = new_dt.to_local() * 0.02
            p.set_data(local_value)

    _assert_shards_differ(model, rank, get_group_size())
