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
"""Pure fully_shard precision cases (MindSpore ST).

A fully_shard SlimLeNet16 and an identical single-card reference (built via
init_empty_weights + in-process load) are trained for several optimizer steps on the same
data; each step accumulates gradients in sharded form (grad sync on every micro-batch) and
the reference's full gradient is compared against each rank's gradient shard (see
``_fsdp_precision_common``). No external checkpoint / dataset. Forward/backward prefetch and
gradient accumulation are on in every case; recompute / comm_fusion / mesh vary so a single
failing case points at one feature. The full-form (deferred-sync) accumulation is guarded in
``_test_fully_shard_simu_pp``.
"""
# pylint: disable=wrong-import-position
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import numpy as np
import mindspore as ms
from mindspore import Tensor, nn
from mindspore.common.api import _no_grad
from mindspore.communication import get_group_size, get_rank, init

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.activation_checkpoint import (
    CheckpointPolicy, SwapManager, checkpoint_wrapper, swap_wrapper,
)
from hyper_parallel.core.dtensor.init_weights import init_empty_weights
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from tests.mindspore.st.common_net import SlimLeNet16
from tests.mindspore.st.fully_shard._fsdp_precision_common import assert_shard_matches_reference, _to_numpy

# Training config
_SEED = 1
_GLOBAL_BS = 64
_MICRO_STEP = 4
_NUM_STEPS = 4
_LR = 0.01
_NUM_CLASSES = 16
_INPUT_SHAPE = (1, 28, 28)
_RTOL = 1e-4
_ATOL = 1e-5
_LOSS_FN = nn.CrossEntropyLoss(reduction="sum")

# FSDP config
_FSDP_SHARDING_SIZE = 2


def _setup_prefetch(net):
    """Wire one-hop forward/backward prefetch across the three sharded Dense layers."""
    layers = [net.dense_relu_sequential[0], net.dense_relu_sequential[2], net.dense_relu_sequential[4]]
    for cur, nxt in zip(layers, layers[1:]):
        cur.set_modules_to_forward_prefetch([nxt])
    for cur, prev in zip(layers[1:], layers):
        cur.set_modules_to_backward_prefetch([prev])


def _apply_recompute(net):
    """Wrap the sharded Dense layers with activation recompute / swap policies."""
    def must_recompute(ctx, op, *args, **kwargs):  # pylint: disable=W0613
        return CheckpointPolicy.MUST_RECOMPUTE

    def must_swap(ctx, op, *args, **kwargs):  # pylint: disable=W0613
        return CheckpointPolicy.MUST_SWAP

    net.dense_relu_sequential[0] = swap_wrapper(net.dense_relu_sequential[0])
    net.dense_relu_sequential[1] = checkpoint_wrapper(net.dense_relu_sequential[1], policy_fn=must_recompute)
    net.dense_relu_sequential[2] = checkpoint_wrapper(net.dense_relu_sequential[2], policy_fn=must_swap)
    net.dense_relu_sequential[3] = checkpoint_wrapper(net.dense_relu_sequential[3])
    for i in range(len(net.dense_relu_sequential) - 1):
        SwapManager().set_forward_prefetch_layer(net.dense_relu_sequential[i], net.dense_relu_sequential[i + 1])


def _fully_sharded_modules(net):
    """Return root and child modules wrapped by fully_shard in this test."""
    return (net, net.dense_relu_sequential[0], net.dense_relu_sequential[2], net.dense_relu_sequential[4])


def _set_reduce_op_type(net, reduce_op_type):
    """Apply reduce op type to every fully_shard state in this nested test model."""
    for mod in _fully_sharded_modules(net):
        mod.set_reduce_op_type(reduce_op_type)


def _assert_comm_fusion_state(net, enabled):
    """Validate every fully_shard module is wired to the expected comm_fusion path."""
    found_fused_group = False
    for mod in _fully_sharded_modules(net):
        state = mod.hsdp_scheduler.hsdp_state
        assert state.config.comm_fusion == enabled, (
            f"{type(mod).__name__} comm_fusion mismatch: expected {enabled}, got {state.config.comm_fusion}"
        )
        param_group = getattr(state, "param_group", None)  # absent when comm_fusion is off
        if not enabled:
            assert param_group is None
        elif state.hsdp_params:
            assert param_group is not None, f"{type(mod).__name__} has sharded params but no fused group"
            found_fused_group = True
    if enabled:
        assert found_fused_group, "expected at least one fused param_group when comm_fusion=True"


def _build_fully_shard_net(state_dict, mesh, *, recompute, comm_fusion):
    """Build a fully_shard SlimLeNet16 matching the reference via init_empty_weights + in-process load."""
    mp_policy = MixedPrecisionPolicy(param_dtype=ms.float32, reduce_dtype=ms.float32,
                                     output_dtype=ms.float32, cast_forward_inputs=False)
    with init_empty_weights():
        net = SlimLeNet16()
    with ms.DeviceCtx("meta"):
        for idx in (0, 2, 4):
            fully_shard(net.dense_relu_sequential[idx], mesh=mesh, mp_policy=mp_policy, comm_fusion=comm_fusion)
        fully_shard(net, mesh=mesh, mp_policy=mp_policy, comm_fusion=comm_fusion)
    _assert_comm_fusion_state(net, comm_fusion)
    net.load_state_dict(state_dict, strict=True)
    _set_reduce_op_type(net, "sum")
    _setup_prefetch(net)
    if recompute:
        _apply_recompute(net)
    return net


def _step_batch(step, world_size):
    """Deterministic global (images, labels) batch for one step, identical on every rank."""
    images = np.random.default_rng(2026 + step).standard_normal((_GLOBAL_BS, *_INPUT_SHAPE)).astype(np.float32)
    labels = np.random.default_rng(7 + step).integers(0, _NUM_CLASSES, size=(_GLOBAL_BS,)).astype(np.int32)
    assert _GLOBAL_BS % world_size == 0, f"global_bs {_GLOBAL_BS} not divisible by world_size {world_size}"
    return Tensor(images), Tensor(labels)


def _fully_shard_backward(net, images, labels):
    """Accumulate one step's gradients in sharded form over micro-batches; return (rank loss, grads)."""
    net.zero_grad()
    micro = images.shape[0] // _MICRO_STEP
    assert micro > 0, f"micro batch must be positive, got local batch {images.shape[0]}"
    loss_total = None
    for i in range(_MICRO_STEP):
        loss = _LOSS_FN(net(images[i * micro: (i + 1) * micro]), labels[i * micro: (i + 1) * micro])
        loss.backward()
        loss_total = loss if loss_total is None else loss_total + loss
    grads = tuple(param.grad for param in net.trainable_params())
    return float(_to_numpy(loss_total)), grads


def _reference_backward(net, images, labels, rank_slice):
    """Full-batch backward plus this rank's loss on its slice; return (rank loss, grads)."""
    for param in net.trainable_params():
        param.grad = None
    _LOSS_FN(net(images), labels).backward()
    grads = tuple(param.grad for param in net.trainable_params())
    with _no_grad():
        rank_loss = float(_to_numpy(_LOSS_FN(net(images[rank_slice]), labels[rank_slice])))
    return rank_loss, grads


def run_precision_case(*, case_name, hsdp=False, recompute=False, comm_fusion=False):
    """Train a fully_shard SlimLeNet16 and a single-card reference in step, comparing loss + grad shards."""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_deterministic(True)
    enable_mindspore_backward_compat()
    init()
    rank = get_rank()
    world_size = get_group_size()
    assert _GLOBAL_BS % world_size == 0, f"global_bs {_GLOBAL_BS} not divisible by world_size {world_size}"

    if hsdp:
        replicate_size = world_size // _FSDP_SHARDING_SIZE
        mesh = init_device_mesh(device_type="npu", mesh_shape=(replicate_size, _FSDP_SHARDING_SIZE),
                                mesh_dim_names=("dp", "op"))
    else:
        mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    shard_size = mesh.shape[-1]
    shard_coord = mesh.get_coordinate()[-1]

    ms.set_seed(_SEED)
    reference_net = SlimLeNet16()
    state_dict = {name: Tensor(param.asnumpy().copy(), ms.float32)
                  for name, param in reference_net.parameters_dict().items()}
    fsdp_net = _build_fully_shard_net(state_dict, mesh, recompute=recompute, comm_fusion=comm_fusion)
    fsdp_optimizer = nn.SGD(fsdp_net.trainable_params(), learning_rate=_LR)
    reference_optimizer = nn.SGD(reference_net.trainable_params(), learning_rate=_LR)

    local_bs = _GLOBAL_BS // world_size
    rank_slice = slice(rank * local_bs, (rank + 1) * local_bs)
    for step in range(_NUM_STEPS):
        images, labels = _step_batch(step, world_size)
        fsdp_loss, fsdp_grads = _fully_shard_backward(fsdp_net, images[rank_slice], labels[rank_slice])
        reference_loss, reference_grads = _reference_backward(reference_net, images, labels, rank_slice)

        assert np.allclose(reference_loss, fsdp_loss, rtol=_RTOL, atol=_ATOL), (
            f"{case_name}, rank {rank}, step {step}, loss: expected {reference_loss}, got {fsdp_loss}"
        )
        for idx, (full_grad, local_grad) in enumerate(zip(reference_grads, fsdp_grads)):
            assert_shard_matches_reference(f"{case_name} step {step}", rank, f"grad {idx}",
                                           _to_numpy(full_grad), _to_numpy(local_grad), shard_size, shard_coord)

        with SkipDTensorDispatch(), _no_grad():
            fsdp_optimizer(fsdp_grads)
        with _no_grad():
            reference_optimizer(reference_grads)

    print(f"[Rank {rank}] {case_name} passed: world_size={world_size}, mesh={mesh.shape}, "
          f"steps={_NUM_STEPS}, recompute={recompute}, comm_fusion={comm_fusion}")


def test_ms_fully_shard_with_gradient_accumulation():
    """
    Feature: 1D dp fully_shard with prefetch and sharded-form gradient accumulation.
    Description: Train a fully_shard SlimLeNet16 and an identical single-card reference for several
                 optimizer steps, accumulating gradients in sharded form over micro-batches each step.
    Expectation: Per-rank loss and every parameter's gradient shard match the reference at every step.
    """
    run_precision_case(case_name="fully_shard with gradient accumulation")


def test_ms_fully_shard_with_recompute():
    """
    Feature: 1D dp fully_shard with activation recompute.
    Description: Same multi-step training comparison as the base case, with layers wrapped for recompute / swap.
    Expectation: Per-rank loss and gradient shards match the single-card reference at every step.
    """
    run_precision_case(case_name="fully_shard with recompute", recompute=True)


def test_ms_fully_shard_with_comm_fusion():
    """
    Feature: 1D dp fully_shard with communication fusion.
    Description: Same multi-step training comparison with comm_fusion (fused gradient reduction) enabled.
    Expectation: Per-rank loss and gradient shards match the single-card reference at every step.
    """
    run_precision_case(case_name="fully_shard with comm fusion", comm_fusion=True)


def test_ms_fully_shard_with_recompute_and_comm_fusion():
    """
    Feature: 1D dp fully_shard with recompute and communication fusion together.
    Description: Same multi-step training comparison with both recompute and comm_fusion enabled.
    Expectation: Per-rank loss and gradient shards match the single-card reference at every step.
    """
    run_precision_case(case_name="fully_shard with recompute and comm fusion", recompute=True, comm_fusion=True)


def test_ms_hsdp_with_recompute():
    """
    Feature: 2D HSDP (dp x op) fully_shard with recompute.
    Description: Same multi-step training comparison on a 2D HSDP mesh with recompute enabled.
    Expectation: Per-rank loss and gradient shards match the single-card reference at every step.
    """
    run_precision_case(case_name="HSDP with recompute", hsdp=True, recompute=True)


def test_ms_hsdp_with_recompute_and_comm_fusion():
    """
    Feature: 2D HSDP fully_shard with recompute and communication fusion together.
    Description: Same multi-step training comparison on a 2D HSDP mesh with recompute and comm_fusion enabled.
    Expectation: Per-rank loss and gradient shards match the single-card reference at every step.
    """
    run_precision_case(case_name="HSDP with recompute and comm fusion", hsdp=True, recompute=True, comm_fusion=True)
