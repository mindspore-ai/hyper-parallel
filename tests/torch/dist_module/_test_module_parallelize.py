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
"""2-card worker tests for Module.parallelize (torchrun entry)."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import  # side effect: register Ascend NPU

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.dmodule.module import Module
from hyper_parallel.dmodule.sharding import ShardingConfig
from hyper_parallel.dmodule.types import MeshAxisName
from tests.torch.utils import init_dist


class ToyLinear(Module):
    """Minimal linear module with TP sharding on weight and output."""

    @dataclass(kw_only=True, slots=True)  # pylint: disable=unexpected-keyword-arg
    class Config(Module.Config):
        in_features: int = 8
        out_features: int = 4

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._sharding_config = ShardingConfig(
            state_shardings={"weight": {MeshAxisName.TP: Shard(0)}},
            # Keys must match forward positional arg names (see Module._redistribute_inputs).
            in_dst_shardings={"x": {MeshAxisName.TP: Replicate()}},
        )
        device = torch.device("npu", dist.get_rank() if dist.is_initialized() else 0)
        self.weight = torch.nn.Parameter(
            torch.arange(config.in_features * config.out_features, dtype=torch.float32, device=device)
            .reshape(config.out_features, config.in_features)
            / 10.0
        )

    def forward(self, x):  # pylint: disable=method-hidden
        """Compute sharded linear; ``Module.parallelize`` wraps this at runtime."""
        return torch.nn.functional.linear(x, self.weight, None)


def test_parallelize_toy_linear_tp2():
    """
    Feature: Module.parallelize on 2-rank TP mesh
    Description: shard weight, replicate input, shard output; forward matches single-rank math
    Expectation: weight and output are DTensor; local forward matches sharded linear math.
    """
    init_dist()
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2,),
        mesh_dim_names=("tp",),
    )
    mod = ToyLinear(ToyLinear.Config())
    mod.parallelize(mesh)

    assert isinstance(mod.weight, DTensor)
    weight_local = mod.weight.to_local()
    x = torch.ones(2, mod.config.in_features, device=weight_local.device)
    out = mod(x)
    assert isinstance(out, DTensor)

    local_out = out.to_local()
    expected = torch.nn.functional.linear(x, weight_local, None)
    assert torch.allclose(local_out, expected, atol=1e-5)

    # Each rank holds an out_features shard; local sum should be positive.
    assert local_out.sum().item() > 0


def test_parallelize_idempotent_guard():
    """Second parallelize() call must raise."""
    init_dist()
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2,),
        mesh_dim_names=("tp",),
    )
    mod = ToyLinear(ToyLinear.Config())
    mod.parallelize(mesh)
    try:
        mod.parallelize(mesh)
        raise AssertionError("expected ValueError on second parallelize")
    except ValueError as exc:
        assert "already been parallelized" in str(exc)
