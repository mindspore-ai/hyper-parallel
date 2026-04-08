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
"""Precision test: fully_shard(list[submodules]) vs standalone (MindSpore ST).

Validates grouped forward hooks (PyTorch FSDP2-aligned): one PostBackward boundary
per list unit per step. Compares loss and ``dense1`` weight grad shard to a
non-sharded reference trained with the same seed and optimizer steps.
"""
import numpy as np
import mindspore as ms
from mindspore import Tensor, mint, nn
from mindspore.communication import get_rank, get_group_size, init

from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat

ms.set_seed(42)
ms.set_deterministic(True)

HIDDEN = 16
OUT = 8
BATCH = 8
LR = 0.01
EPOCHS = 2
STEPS = 4


class TinyDensePairBlock(nn.Cell):
    """Two serial Dense layers (wrapped as one list FSDP unit in the dist case)."""

    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.dense1 = nn.Dense(hidden_size, hidden_size, has_bias=False)
        self.dense2 = nn.Dense(hidden_size, output_size, has_bias=False)

    def construct(self, x: Tensor) -> Tensor:
        """Forward through two Dense layers.

        Args:
            x: Input activations.
        """
        x = self.dense1(x)
        return self.dense2(x)


class ListUnitNet(nn.Cell):
    """pre -> dense1 -> dense2 -> sum (same topology as Torch list precision test)."""

    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.pre = nn.Dense(hidden_size, hidden_size, has_bias=False)
        self.block = TinyDensePairBlock(hidden_size, output_size)

    def construct(self, x: Tensor) -> Tensor:
        """Forward: pre, block, then reduce to scalar loss.

        Args:
            x: Input activations.
        """
        x = self.pre(x)
        x = self.block(x)
        return mint.sum(x)


def _fixed_input():
    rng = np.random.RandomState(0)
    return Tensor(rng.randn(BATCH, HIDDEN).astype(np.float32))


def _clear_trainable_grads(net) -> None:
    """Plain ``nn.Cell`` has no ``zero_grad``; clear grads before each step.

    After ``fully_shard(net, ...)``, the root cell mixes in ``HSDPModule`` and
    ``net.zero_grad()`` is valid (see ``_train_fully_shard_list``).
    """
    for p in net.trainable_params():
        p.grad = None


def _get_dense1_grad_shard(net) -> Tensor:
    """Return ``block.dense1.weight.grad`` local tensor (DTensor)."""
    for p in net.trainable_params():
        if p.name.endswith("block.dense1.weight") or "dense1.weight" in p.name:
            grad = p.grad
            assert isinstance(grad, DTensor), type(grad)
            return grad.to_local()
    raise AssertionError("block.dense1.weight grad not found")


def _train_reference():
    """Single full-replica training (no fully_shard) for numerical target."""
    # Match ``fully_shard()``: PyNative loss.backward() needs torch-style tensor API.
    enable_mindspore_backward_compat()
    ms.set_seed(42)
    x = _fixed_input()
    net = ListUnitNet(HIDDEN, OUT)
    opt = nn.SGD(net.trainable_params(), learning_rate=LR)
    last_loss = None
    last_g1 = None
    for _ in range(EPOCHS):
        for _ in range(STEPS):
            _clear_trainable_grads(net)
            loss = net(x)
            loss.backward()
            last_loss = float(loss.asnumpy())
            last_g1 = None
            for p in net.trainable_params():
                if "dense1.weight" in p.name:
                    last_g1 = p.grad.asnumpy().copy()
            grads = tuple(p.grad for p in net.trainable_params())
            opt(grads)
    assert last_g1 is not None
    return last_loss, last_g1


def _train_fully_shard_list(mesh):
    """Nested fully_shard + list unit (exercises grouped hooks)."""
    ms.set_seed(42)
    x = _fixed_input()
    net = ListUnitNet(HIDDEN, OUT)
    mp_policy = MixedPrecisionPolicy()
    fsdp_kw = {"mesh": mesh, "mp_policy": mp_policy}
    fully_shard(net.pre, **fsdp_kw)
    fully_shard(
        [net.block.dense1, net.block.dense2],
        **fsdp_kw,
        reshard_after_forward=False,
    )
    fully_shard(net, **fsdp_kw)
    assert net.block.dense1.hsdp_scheduler is net.block.dense2.hsdp_scheduler

    opt = nn.SGD(net.trainable_params(), learning_rate=LR)
    last_loss = None
    last_local = None
    for _ in range(EPOCHS):
        for _ in range(STEPS):
            net.zero_grad()
            loss = net(x)
            loss.backward()
            last_loss = float(loss.asnumpy())
            last_local = _get_dense1_grad_shard(net).asnumpy()
            grads = tuple(p.grad for p in net.trainable_params())
            for g in grads:
                assert isinstance(g, DTensor), type(g)
            with SkipDTensorDispatch():
                opt(grads)
    return last_loss, last_local


def test_ms_fully_shard_list_unit_precision():
    """
    Feature: fully_shard(list) precision vs standalone MindSpore training.
    Description: Nested ``fully_shard([d1,d2], reshard_after_forward=False)`` list unit;
        compare final loss and ``dense1`` grad slice to a non-sharded reference.
    Expectation: Run success on all ranks.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    init()
    rank = get_rank()
    ws = get_group_size()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(ws,), mesh_dim_names=("dp",))
    shard = ws

    ref_loss, ref_g1 = _train_reference()
    dist_loss, dist_local = _train_fully_shard_list(mesh)

    assert np.allclose(ref_loss, dist_loss, rtol=1e-3, atol=1e-3), (ref_loss, dist_loss)

    stride = HIDDEN // shard
    off = (rank % shard) * stride
    expected = ref_g1[off : off + stride, :]
    assert np.allclose(expected, dist_local, rtol=1e-3, atol=1e-3), (
        rank,
        expected.shape,
        dist_local.shape,
    )
