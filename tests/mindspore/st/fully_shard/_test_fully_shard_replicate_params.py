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
"""MindSpore ST for fully_shard replicate_params.

``replicate_params`` keeps the listed parameters un-sharded on every rank and reduces their
gradients with all-reduce (ZeRO-1 style) instead of reduce-scatter. This test covers a model
that mixes replicate weights (all-reduce) and sharded biases (reduce-scatter) under forward
and backward prefetch. A separate case verifies that a TP-sharded, non-trainable state can be
ignored when its forward performs an in-place update.
"""
# pylint: disable=wrong-import-position
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import numpy as np
import mindspore as ms
from mindspore import Parameter, Tensor, nn
from mindspore.common.api import _no_grad
from mindspore.communication import get_group_size, get_rank, init

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.hsdp_utils import ShardedState, get_hsdp_state
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from tests.mindspore.st.common_net import SlimLeNet16
from tests.mindspore.st.fully_shard._fsdp_precision_common import assert_shard_matches_reference, _to_numpy

_SEED = 42
_GLOBAL_BS = 64
_NUM_STEPS = 4
_LR = 0.01
_NUM_CLASSES = 16
_INPUT_SHAPE = (1, 28, 28)
_RTOL = 1e-4
_ATOL = 1e-5
_LOSS_FN = nn.CrossEntropyLoss(reduction="sum")


def _fp32_policy():
    return MixedPrecisionPolicy(param_dtype=ms.float32, reduce_dtype=ms.float32,
                                output_dtype=ms.float32, cast_forward_inputs=False)


def _setup_prefetch(net):
    """Wire one-hop forward and backward prefetch across the three wrapped Dense layers."""
    layers = [net.dense_relu_sequential[0], net.dense_relu_sequential[2], net.dense_relu_sequential[4]]
    for cur, nxt in zip(layers, layers[1:]):
        cur.set_modules_to_forward_prefetch([nxt])
    for cur, prev in zip(layers[1:], layers):
        cur.set_modules_to_backward_prefetch([prev])


def _assert_mixed_prefetch_state(module):
    """After forward, the prefetched child must return all managed params to SHARDED."""
    hsdp_state = get_hsdp_state(module)
    assert hsdp_state is not None, "Expected fully_shard child to expose hsdp_state"
    managed_states = [param.sharded_state for param in hsdp_state.hsdp_params]
    assert managed_states, "Expected the prefetched child to manage parameters"
    assert all(state == ShardedState.SHARDED for state in managed_states), (
        f"Expected all managed parameters to be SHARDED after forward, got {managed_states}"
    )


def _build_replicate_fully_shard_net(mesh):
    """SlimLeNet16 with weights kept as replicate_params and biases sharded, plus prefetch."""
    ms.set_seed(_SEED)
    net = SlimLeNet16()
    replicate_params = {param for param in net.trainable_params() if "weight" in param.name}
    for idx in (0, 2, 4):
        fully_shard(net.dense_relu_sequential[idx], mesh=mesh, mp_policy=_fp32_policy(),
                    replicate_params=replicate_params)
    fully_shard(net, mesh=mesh, mp_policy=_fp32_policy(), replicate_params=replicate_params)
    net.set_reduce_op_type("sum")
    _setup_prefetch(net)
    root_state = get_hsdp_state(net)
    assert len(root_state.hsdp_params) == 1
    output_scale_param = root_state.hsdp_params[0]
    assert net.output_scale_weight.shape[0] % mesh.shape[-1] != 0
    assert output_scale_param.is_replicate_param and output_scale_param.shard_world_size == 1
    return net


def test_ms_fully_shard_with_replicate_params():
    """
    Feature: fully_shard with replicate_params (all-reduce weights) mixed with sharded biases, under prefetch.
    Description: Train the mixed model and an identical single-card reference for several steps; on the first
                 forward assert the prefetched child stays in the replicate-UNSHARDED / sharded-SHARDED state.
    Expectation: Per-rank loss matches; replicate grads match the full reference grad, sharded grads match the shard.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_deterministic(True)
    enable_mindspore_backward_compat()
    init()
    rank = get_rank()
    world_size = get_group_size()
    assert _GLOBAL_BS % world_size == 0, f"global_bs {_GLOBAL_BS} not divisible by world_size {world_size}"
    mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    ms.set_seed(_SEED)
    reference_net = SlimLeNet16()
    fsdp_net = _build_replicate_fully_shard_net(mesh)
    reference_optimizer = nn.SGD(reference_net.trainable_params(), learning_rate=_LR)
    fsdp_optimizer = nn.SGD(fsdp_net.trainable_params(), learning_rate=_LR)

    local_bs = _GLOBAL_BS // world_size
    rank_slice = slice(rank * local_bs, (rank + 1) * local_bs)
    for step in range(_NUM_STEPS):
        image_np = np.random.default_rng(2026 + step).standard_normal((_GLOBAL_BS, *_INPUT_SHAPE)).astype(np.float32)
        label_np = np.random.default_rng(7 + step).integers(0, _NUM_CLASSES, size=(_GLOBAL_BS,)).astype(np.int32)
        images, labels = Tensor(image_np), Tensor(label_np)

        fsdp_net.zero_grad()
        fsdp_logits = fsdp_net(images[rank_slice])
        if step == 0:
            _assert_mixed_prefetch_state(fsdp_net.dense_relu_sequential[2])
        fsdp_loss = _LOSS_FN(fsdp_logits, labels[rank_slice])
        fsdp_loss.backward()
        fsdp_grads = tuple(param.grad for param in fsdp_net.trainable_params())

        for param in reference_net.trainable_params():
            param.grad = None
        _LOSS_FN(reference_net(images), labels).backward()
        reference_grads = tuple(param.grad for param in reference_net.trainable_params())
        with _no_grad():
            reference_loss = float(_to_numpy(_LOSS_FN(reference_net(images[rank_slice]), labels[rank_slice])))

        fsdp_loss_value = float(_to_numpy(fsdp_loss))
        assert np.allclose(reference_loss, fsdp_loss_value, rtol=_RTOL, atol=_ATOL), (
            f"replicate_params, rank {rank}, step {step}, loss: expected {reference_loss}, got {fsdp_loss_value}"
        )
        for idx, (full_grad, local_grad) in enumerate(zip(reference_grads, fsdp_grads)):
            full_np, local_np = _to_numpy(full_grad), _to_numpy(local_grad)
            # replicate params hold the full grad on every rank (compare full); sharded params hold a dim-0 shard
            replicated = local_np.shape[0] == full_np.shape[0]
            shard_size = 1 if replicated else mesh.shape[-1]
            shard_coord = 0 if replicated else mesh.get_coordinate()[-1]
            assert_shard_matches_reference(f"replicate_params step {step}", rank, f"grad {idx}",
                                           full_np, local_np, shard_size, shard_coord)

        with SkipDTensorDispatch(), _no_grad():
            fsdp_optimizer(fsdp_grads)
        with _no_grad():
            reference_optimizer(reference_grads)

    print(f"[Rank {rank}] replicate_params precision passed: world_size={world_size}, steps={_NUM_STEPS}")


class ReplicateStateNet(nn.Cell):
    """Small net with a TP-sharded, non-trainable state updated in forward."""

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
        self.max_logits_val.fill_(7.0)
        return self.dense(x)


def test_ms_fully_shard_ignored_dtensor_state():
    """
    Feature: ignored_params holding a TP-sharded DTensor state on a (dp, tp) mesh.
    Description: max_logits_val is sharded on the TP sub-mesh and updated in forward with no-grad + in-place.
                 It is excluded from fully_shard because the unsharded-parameter lifecycle does not support
                 forward-side in-place state updates.
    Expectation: max_logits_val reads back greater than zero after backward.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank_id = get_rank()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    net = ReplicateStateNet(mesh["tp"])
    fully_shard(net, mesh=mesh["dp"], mp_policy=_fp32_policy(), ignored_params={net.max_logits_val})

    loss_fn = nn.CrossEntropyLoss()
    data = Tensor(np.random.randn(8, 32).astype(np.float32))
    label = Tensor(np.random.randint(0, 4, size=(8,)).astype(np.int32))
    net.zero_grad()
    loss_fn(net(data), label).backward()

    max_logits_sum = float(net.max_logits_val.sum().asnumpy())
    assert max_logits_sum > 0, "Expected callback-style max_logits_val read to be non-zero after backward"
    if rank_id == 0:
        print(f"ignored_params DTensor state after backward: {max_logits_sum}")
