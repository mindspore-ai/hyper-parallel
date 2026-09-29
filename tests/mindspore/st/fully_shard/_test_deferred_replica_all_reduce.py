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
"""Numerical regression for deferred HSDP all-reduce, independent of a training stack."""
from unittest.mock import patch

import mindspore as ms
import numpy as np
from mindspore import Tensor, nn
from mindspore.communication import get_rank, get_group_size, init

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from hyper_parallel.platform.mindspore.fully_shard import param as param_mod


class LinearLoss(nn.Cell):
    """One parameter with an exactly representable, input-dependent gradient."""

    def __init__(self) -> None:
        """Initialize the same weight on every rank."""
        super().__init__()
        self.linear = nn.Dense(4, 4, has_bias=False, weight_init="ones")

    def construct(self, inputs: Tensor) -> Tensor:
        """Return a scalar whose weight gradient is the input in every row."""
        return self.linear(inputs).sum()


def _check_configuration(shard_size: int, reduce_op: str, main_grad: bool) -> None:
    """Validate both replica-only and reduce-scatter plus replica reduction."""
    rank = get_rank()
    mesh = init_device_mesh("npu", (4 // shard_size, shard_size, 2),
                            mesh_dim_names=("replicate", "shard", "ep"))
    with SkipDTensorDispatch():
        model = LinearLoss()
    model = fully_shard(
        model, mesh=mesh[("replicate", "shard")], comm_fusion=False,
        mp_policy=MixedPrecisionPolicy(param_dtype=ms.float32, reduce_dtype=ms.float32,
                                       apply_grad_on_fp32_main_grad=main_grad),
    )
    model.set_reduce_op_type(reduce_op)
    model.set_gradient_scaling_factor(0.125)
    peers = list(range(rank % 2, 8, 2))
    for step in range(2):
        model.zero_grad()
        with patch.object(param_mod.dist, "all_reduce", wraps=param_mod.dist.all_reduce) as collective:
            with SkipDTensorDispatch():
                for micro in range(4):
                    # Reduce-scatter remains enabled; only replica AR is deferred.
                    model.set_requires_all_reduce(micro == 3)
                    inputs = Tensor(np.full((1, 4), rank + micro + step + 1, np.float32))
                    model(inputs).backward()
                    assert collective.call_count == int(micro == 3), (shard_size, step, micro)
        weight = model.linear.weight
        grad = weight.main_grad if main_grad else weight.grad
        local_grad = grad.to_local().asnumpy()
        expected = sum(peer + micro + step + 1 for peer in peers for micro in range(4)) * 0.125
        if reduce_op == "avg":
            expected /= len(peers)
        np.testing.assert_array_equal(local_grad, np.full(local_grad.shape, expected, np.float32))
        # Identical gradients applied to identical shards must preserve replicas.
        with SkipDTensorDispatch(), get_platform().no_grad():
            local_weight = weight.to_local()
            local_weight.copy_(local_weight - grad.to_local() * 0.125)


def test_deferred_replica_all_reduce() -> None:
    """Check the analytic gradient and one collective per step on all eight ranks."""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_deterministic(True)
    init()
    assert get_group_size() == 8
    enable_mindspore_backward_compat()
    for shard_size in (1, 2):
        for reduce_op in ("sum", "avg"):
            for main_grad in (False, True):
                _check_configuration(shard_size, reduce_op, main_grad)
