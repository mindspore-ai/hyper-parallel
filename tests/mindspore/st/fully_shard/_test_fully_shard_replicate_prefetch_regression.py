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
"""MindSpore ST for fully_shard replicate_params + backward prefetch regression."""
import mindspore as ms
import numpy as np

from mindspore import Tensor, mint, nn, Parameter
from mindspore.communication import get_rank, init

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.hsdp_utils import ShardedState, get_hsdp_state
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from tests.mindspore.st.common_net import SlimLeNet16

ms.set_seed(42)
ms.set_deterministic(True)
enable_mindspore_backward_compat()


def _setup_prefetch(net) -> None:
    """Configure one-hop forward/backward prefetch on the wrapped dense layers."""
    wrapped_modules = [
        net.dense_relu_sequential[0],
        net.dense_relu_sequential[2],
        net.dense_relu_sequential[4],
    ]

    for idx, layer in enumerate(wrapped_modules[:-1]):
        layer.set_modules_to_forward_prefetch([wrapped_modules[idx + 1]])

    wrapped_modules.reverse()
    for idx, layer in enumerate(wrapped_modules[:-1]):
        layer.set_modules_to_backward_prefetch([wrapped_modules[idx + 1]])


def _assert_mixed_state_for_prefetched_child(module) -> None:
    """Assert the child module is in the mixed state that used to trigger the bug."""
    hsdp_state = get_hsdp_state(module)
    assert hsdp_state is not None, "Expected fully_shard child to expose hsdp_state"

    replicate_states = [param.sharded_state for param in hsdp_state.replicate_params]
    sharded_states = [param.sharded_state for param in hsdp_state.sharded_hsdp_params]

    assert replicate_states, "Expected at least one replicate_param in the prefetched child"
    assert sharded_states, "Expected at least one sharded param in the prefetched child"
    assert all(state == ShardedState.UNSHARDED for state in replicate_states), (
        f"Expected replicate_params to stay UNSHARDED after forward, got {replicate_states}"
    )
    assert all(state == ShardedState.SHARDED for state in sharded_states), (
        f"Expected non-replicated params to be SHARDED after forward, got {sharded_states}"
    )


class ReplicateStateNet(nn.Cell):
    """Small net with a TP-sharded replicate state updated in forward."""

    def __init__(self, tp_mesh):
        super().__init__()
        self.dense = nn.Dense(32, 4, weight_init="normal", bias_init="zeros")
        state = Parameter(Tensor(np.zeros((32,), np.float32)), name="max_logits_val", requires_grad=False)
        self.max_logits_val = Parameter(
            distribute_tensor(state, device_mesh=tp_mesh, placements=(Shard(0),)),
            name="max_logits_val",
            requires_grad=False,
        )

    def construct(self, x):
        self.max_logits_val.copy_(mint.ones_like(self.max_logits_val) * 7.0)
        return self.dense(x)


def test_ms_fully_shard_replicate_params_backward_prefetch_regression():
    """Regression test for replicate_params + backward prefetch on MindSpore fully_shard.

    Feature: fully_shard replicate_params with backward prefetch.
    Description: Wrap child dense layers and the root module with fully_shard, mark
        each dense weight as ``replicate_params`` and enable one-hop forward/backward
        prefetch. After forward, the prefetched child should enter the mixed state
        that previously triggered duplicate ``unshard()`` during backward prefetch:
        replicated weights remain ``UNSHARDED`` while sharded biases are resharded.
    Expectation: ``loss.backward()`` succeeds without re-entering ``unshard()`` on an
        already-``UNSHARDED`` replicate param, and gradients remain usable by the optimizer.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank_id = get_rank()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("dp",))
    net = SlimLeNet16()
    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False,
    )
    replicate_params = {param for param in net.trainable_params() if "weight" in param.name}

    for layer_idx in (0, 2, 4):
        fully_shard(
            net.dense_relu_sequential[layer_idx],
            mesh=mesh,
            mp_policy=mp_policy,
            replicate_params=replicate_params,
        )
    fully_shard(net, mesh=mesh, mp_policy=mp_policy, replicate_params=replicate_params)
    _setup_prefetch(net)

    optimizer = nn.Adam(net.trainable_params(), learning_rate=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    data = Tensor(np.random.randn(8, 1, 28, 28).astype(np.float32))
    label = Tensor(np.random.randint(0, 16, size=(8,)).astype(np.int32))

    net.zero_grad()
    logits = net(data)
    _assert_mixed_state_for_prefetched_child(net.dense_relu_sequential[2])

    loss = loss_fn(logits, label)
    loss.backward()

    grads = tuple(param.grad for param in net.trainable_params())
    assert all(grad is not None for grad in grads), "Expected all fully_shard params to materialize gradients"
    with SkipDTensorDispatch():
        optimizer(grads)

    if rank_id == 0:
        print(f"rank: {rank_id}, regression step loss: {float(loss.asnumpy())}")


def test_ms_hsdp_replicate_dtensor_state_visible_after_backward():
    """End-to-end regression for TP DTensor replicate state under HSDP.

    Feature: HSDP ``replicate_params`` with a TP-sharded DTensor state parameter.
    Description: ``ReplicateStateNet`` updates ``max_logits_val`` in forward,
        then HSDP switches the module back to the sharded parameter object after
        backward. The callback-style read from the module must observe the
        updated value instead of the initial zeros.
    Expectation: ``max_logits_val`` remains greater than zero after backward.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank_id = get_rank()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    net = ReplicateStateNet(mesh["tp"])
    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=False,
    )
    fully_shard(net, mesh=mesh, mp_policy=mp_policy, replicate_params={net.max_logits_val})

    loss_fn = nn.CrossEntropyLoss()
    data = Tensor(np.random.randn(8, 32).astype(np.float32))
    label = Tensor(np.random.randint(0, 4, size=(8,)).astype(np.int32))

    net.zero_grad()
    loss = loss_fn(net(data), label)
    loss.backward()

    max_logits = net.max_logits_val
    assert float(max_logits.sum().asnumpy()) > 0, "Expected callback-style max_logits_val read to be non-zero"
    if rank_id == 0:
        print(f"rank: {rank_id}, max_logits_sum_after_backward: {float(max_logits.sum().asnumpy())}")
