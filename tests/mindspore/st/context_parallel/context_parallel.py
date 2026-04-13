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
"""MindSpore distributed integration tests for ContextParallel and AsyncContextParallel."""
# pylint: disable=wrong-import-position
import os

import mindspore as ms
import mindspore.communication.management as D
import numpy as np
import pytest
from mindspore import mint, nn

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "mindspore")

from hyper_parallel import init_device_mesh, ContextParallel, AsyncContextParallel


def _init_dist(expected_world_size: int):
    """Initialize distributed runtime in PyNative mode and validate world size."""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    D.init()
    rank = D.get_rank()
    world_size = D.get_group_size()
    if world_size != expected_world_size:
        pytest.skip(f"Expected world_size={expected_world_size}, but got {world_size}")
    return rank, world_size


def _assert_close(actual, expected, rank, label, atol=1e-4, rtol=1e-4):
    """Compare two tensors after converting them to numpy."""
    actual_np = actual.asnumpy()
    expected_np = expected.asnumpy()
    max_diff = float(np.max(np.abs(actual_np - expected_np)))
    print(f"[Rank {rank}] {label} max_diff={max_diff:.3e}")
    assert np.allclose(actual_np, expected_np, atol=atol, rtol=rtol), (
        f"[Rank {rank}] {label} mismatch: max_diff={max_diff:.3e}"
    )


def _make_full_input(batch: int, seq_len: int, hidden_size: int) -> ms.Tensor:
    """Create a deterministic full-sequence input tensor."""
    values = np.sin(np.arange(batch * seq_len * hidden_size, dtype=np.float32) / 7.0)
    return ms.Tensor(values.reshape(batch, seq_len, hidden_size))


def _make_weight_matrix(hidden_size: int, offset: float) -> np.ndarray:
    """Create a deterministic projection matrix."""
    values = np.cos(np.arange(hidden_size * hidden_size, dtype=np.float32) / 11.0 + offset)
    return (values.reshape(hidden_size, hidden_size) / hidden_size).astype(np.float32)


class _Projection(nn.Cell):
    """Deterministic projection that reshapes to BSHD."""

    def __init__(  # pylint: disable=unused-argument
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        weight_data: np.ndarray,
        name: str,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.weight = ms.Parameter(ms.Tensor(weight_data), name=f"{name}_weight")

    def construct(self, x):
        out = mint.matmul(x, self.weight)
        shape = x.shape
        return out.reshape(shape[0], shape[1], self.num_heads, self.head_dim)


class _BshdSelfAttention(nn.Cell):
    """Simple BSHD self-attention used for CP parity tests."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = 1.0 / np.sqrt(float(head_dim))

    def construct(self, q, k, v):
        q_heads = q.permute((0, 2, 1, 3))
        k_heads = k.permute((0, 2, 1, 3))
        v_heads = v.permute((0, 2, 1, 3))
        scores = mint.matmul(q_heads, k_heads.permute((0, 1, 3, 2))) * self.scale
        probs = ms.ops.softmax(scores, axis=-1)
        out = mint.matmul(probs, v_heads)
        return out.permute((0, 2, 1, 3)).contiguous()


class _TinyCpModel(nn.Cell):
    """Minimal QKV + attention model for ContextParallel integration tests."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.q_proj = _Projection(
            hidden_size,
            num_heads,
            head_dim,
            _make_weight_matrix(hidden_size, 0.1),
            "q",
        )
        self.k_proj = _Projection(
            hidden_size,
            num_heads,
            head_dim,
            _make_weight_matrix(hidden_size, 0.2),
            "k",
        )
        self.v_proj = _Projection(
            hidden_size,
            num_heads,
            head_dim,
            _make_weight_matrix(hidden_size, 0.3),
            "v",
        )
        self.attn = _BshdSelfAttention(head_dim)

    def construct(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.attn(q, k, v)


def _prepare_inputs(
    rank: int,
    world_size: int,
    batch: int = 1,
    seq_len: int = 8,
    num_heads: int = 4,
    head_dim: int = 4,
):
    """Create deterministic full and local inputs for the current rank."""
    hidden_size = num_heads * head_dim
    full_x = _make_full_input(batch, seq_len, hidden_size)
    local_s = seq_len // world_size
    start = rank * local_s
    end = start + local_s
    local_x = full_x[:, start:end, :]
    return full_x, local_x, hidden_size, local_s


def _build_local_reference(rank: int, local_s: int, ref_full):
    """Slice the local sequence window from the reference full output."""
    start = rank * local_s
    end = start + local_s
    return ref_full[:, start:end, :, :]


def _sum_across_ranks(tensor: ms.Tensor) -> ms.Tensor:
    """Sum a tensor across all ranks in the default communication group."""
    return ms.ops.AllReduce()(tensor)


def test_context_parallel_ulysses_forward():
    """
    Feature: MindSpore ContextParallel Ulysses forward parity.
    Description: Compare 2-card local CP output against a single-card full-sequence reference.
    Expectation: Outputs match on each rank.
    """
    rank, world_size = _init_dist(expected_world_size=2)
    full_x, local_x, hidden_size, local_s = _prepare_inputs(rank, world_size)
    mesh = init_device_mesh("npu", (world_size,), mesh_dim_names=("cp",))

    model_cp = _TinyCpModel(hidden_size, num_heads=4, head_dim=4)
    ContextParallel(seq_dim=1, head_dim=2).apply(model_cp.attn, mesh)
    cp_out = model_cp(local_x)

    model_ref = _TinyCpModel(hidden_size, num_heads=4, head_dim=4)
    ref_out = _build_local_reference(rank, local_s, model_ref(full_x))
    _assert_close(cp_out, ref_out, rank, "sync_context_parallel_ulysses_forward")


def test_async_context_parallel_ulysses_forward():
    """
    Feature: MindSpore AsyncContextParallel Ulysses forward parity.
    Description: Compare 2-card Async CP output against a single-card full-sequence reference.
    Expectation: Outputs match on each rank.
    """
    rank, world_size = _init_dist(expected_world_size=2)
    full_x, local_x, hidden_size, local_s = _prepare_inputs(rank, world_size)
    mesh = init_device_mesh("npu", (world_size,), mesh_dim_names=("cp",))

    model_cp = _TinyCpModel(hidden_size, num_heads=4, head_dim=4)
    AsyncContextParallel(seq_dim=1, head_dim=2).apply(
        module=model_cp.attn,
        device_mesh=mesh,
        q_proj=model_cp.q_proj,
        k_proj=model_cp.k_proj,
        v_proj=model_cp.v_proj,
    )
    cp_out = model_cp(local_x)

    model_ref = _TinyCpModel(hidden_size, num_heads=4, head_dim=4)
    ref_out = _build_local_reference(rank, local_s, model_ref(full_x))
    _assert_close(cp_out, ref_out, rank, "async_context_parallel_ulysses_forward")


def test_async_context_parallel_ulysses_forward_repeat():
    """
    Feature: MindSpore AsyncContextParallel repeated forward parity.
    Description: Run the same 2-card async forward twice to ensure slot state is cleared between iterations.
    Expectation: Both forward results match the single-card reference.
    """
    rank, world_size = _init_dist(expected_world_size=2)
    full_x, local_x, hidden_size, local_s = _prepare_inputs(rank, world_size)
    mesh = init_device_mesh("npu", (world_size,), mesh_dim_names=("cp",))

    model_cp = _TinyCpModel(hidden_size, num_heads=4, head_dim=4)
    AsyncContextParallel(seq_dim=1, head_dim=2).apply(
        module=model_cp.attn,
        device_mesh=mesh,
        q_proj=model_cp.q_proj,
        k_proj=model_cp.k_proj,
        v_proj=model_cp.v_proj,
    )

    model_ref = _TinyCpModel(hidden_size, num_heads=4, head_dim=4)
    ref_out = _build_local_reference(rank, local_s, model_ref(full_x))

    first_out = model_cp(local_x)
    second_out = model_cp(local_x)
    _assert_close(first_out, ref_out, rank, "async_context_parallel_ulysses_forward_repeat:first")
    _assert_close(second_out, ref_out, rank, "async_context_parallel_ulysses_forward_repeat:second")
    _assert_close(second_out, first_out, rank, "async_context_parallel_ulysses_forward_repeat:stable")


def test_async_context_parallel_ulysses_backward():
    """
    Feature: MindSpore AsyncContextParallel Ulysses backward parity.
    Description: Compare 2-card local Async CP parameter grads against a single-card local-slice reference loss.
    Expectation: Parameter gradients match on each rank.
    """
    rank, world_size = _init_dist(expected_world_size=2)
    full_x, local_x, hidden_size, local_s = _prepare_inputs(rank, world_size)
    mesh = init_device_mesh("npu", (world_size,), mesh_dim_names=("cp",))

    model_cp = _TinyCpModel(hidden_size, num_heads=4, head_dim=4)
    AsyncContextParallel(seq_dim=1, head_dim=2).apply(
        module=model_cp.attn,
        device_mesh=mesh,
        q_proj=model_cp.q_proj,
        k_proj=model_cp.k_proj,
        v_proj=model_cp.v_proj,
    )

    def cp_forward(inp):
        return ms.ops.sum(model_cp(inp))

    _, cp_grads = ms.value_and_grad(
        cp_forward, grad_position=None, weights=model_cp.trainable_params(), has_aux=False
    )(local_x)

    model_ref = _TinyCpModel(hidden_size, num_heads=4, head_dim=4)
    start = rank * local_s
    end = start + local_s

    def ref_forward(inp):
        return ms.ops.sum(model_ref(inp)[:, start:end, :, :])

    _, ref_local_grads = ms.value_and_grad(
        ref_forward, grad_position=None, weights=model_ref.trainable_params(), has_aux=False
    )(full_x)

    model_ref_full = _TinyCpModel(hidden_size, num_heads=4, head_dim=4)

    def ref_full_forward(inp):
        return ms.ops.sum(model_ref_full(inp))

    _, ref_full_grads = ms.value_and_grad(
        ref_full_forward, grad_position=None, weights=model_ref_full.trainable_params(), has_aux=False
    )(full_x)

    for param, cp_grad, ref_local_grad, ref_full_grad in zip(
        model_cp.trainable_params(), cp_grads, ref_local_grads, ref_full_grads
    ):
        assert not np.isnan(cp_grad.asnumpy()).any(), f"{param.name} grad contains NaN"
        assert not np.isinf(cp_grad.asnumpy()).any(), f"{param.name} grad contains Inf"
        if param.name.startswith("q_proj"):
            grad_to_compare = cp_grad
            ref_grad = ref_local_grad
        else:
            grad_to_compare = _sum_across_ranks(cp_grad)
            ref_grad = ref_full_grad
        _assert_close(
            grad_to_compare,
            ref_grad,
            rank,
            f"async_context_parallel_ulysses_backward:{param.name}",
            atol=5e-4,
            rtol=5e-4,
        )


def test_async_context_parallel_ulysses_backward_repeat():
    """
    Feature: MindSpore AsyncContextParallel repeated backward stability.
    Description: Execute async backward twice on the same 2-card input without parameter updates.
    Expectation: Repeated gradients are finite and numerically stable across iterations.
    """
    rank, world_size = _init_dist(expected_world_size=2)
    full_x, local_x, hidden_size, local_s = _prepare_inputs(rank, world_size)
    mesh = init_device_mesh("npu", (world_size,), mesh_dim_names=("cp",))

    model_cp = _TinyCpModel(hidden_size, num_heads=4, head_dim=4)
    AsyncContextParallel(seq_dim=1, head_dim=2).apply(
        module=model_cp.attn,
        device_mesh=mesh,
        q_proj=model_cp.q_proj,
        k_proj=model_cp.k_proj,
        v_proj=model_cp.v_proj,
    )

    def cp_forward(inp):
        return ms.ops.sum(model_cp(inp))

    _, first_grads = ms.value_and_grad(
        cp_forward, grad_position=None, weights=model_cp.trainable_params(), has_aux=False
    )(local_x)
    _, second_grads = ms.value_and_grad(
        cp_forward, grad_position=None, weights=model_cp.trainable_params(), has_aux=False
    )(local_x)

    model_ref = _TinyCpModel(hidden_size, num_heads=4, head_dim=4)
    start = rank * local_s
    end = start + local_s

    def ref_forward(inp):
        return ms.ops.sum(model_ref(inp)[:, start:end, :, :])

    _, ref_local_grads = ms.value_and_grad(
        ref_forward, grad_position=None, weights=model_ref.trainable_params(), has_aux=False
    )(full_x)

    model_ref_full = _TinyCpModel(hidden_size, num_heads=4, head_dim=4)

    def ref_full_forward(inp):
        return ms.ops.sum(model_ref_full(inp))

    _, ref_full_grads = ms.value_and_grad(
        ref_full_forward, grad_position=None, weights=model_ref_full.trainable_params(), has_aux=False
    )(full_x)

    for param, first_grad, second_grad, ref_local_grad, ref_full_grad in zip(
        model_cp.trainable_params(), first_grads, second_grads, ref_local_grads, ref_full_grads
    ):
        assert not np.isnan(first_grad.asnumpy()).any(), f"{param.name} first grad contains NaN"
        assert not np.isinf(first_grad.asnumpy()).any(), f"{param.name} first grad contains Inf"
        assert not np.isnan(second_grad.asnumpy()).any(), f"{param.name} second grad contains NaN"
        assert not np.isinf(second_grad.asnumpy()).any(), f"{param.name} second grad contains Inf"
        if param.name.startswith("q_proj"):
            first_compare = first_grad
            second_compare = second_grad
            ref_grad = ref_local_grad
        else:
            first_compare = _sum_across_ranks(first_grad)
            second_compare = _sum_across_ranks(second_grad)
            ref_grad = ref_full_grad
        _assert_close(
            first_compare,
            ref_grad,
            rank,
            f"async_context_parallel_ulysses_backward_repeat:first:{param.name}",
            atol=5e-4,
            rtol=5e-4,
        )
        _assert_close(
            second_compare,
            ref_grad,
            rank,
            f"async_context_parallel_ulysses_backward_repeat:second:{param.name}",
            atol=5e-4,
            rtol=5e-4,
        )
        _assert_close(
            second_compare,
            first_compare,
            rank,
            f"async_context_parallel_ulysses_backward_repeat:stable:{param.name}",
            atol=5e-4,
            rtol=5e-4,
        )


def test_async_context_parallel_hybrid_forward():
    """
    Feature: MindSpore AsyncContextParallel hybrid forward parity.
    Description: Compare 4-card hybrid Async CP output against a single-card full-sequence reference.
    Expectation: Outputs match on each rank.
    """
    rank, world_size = _init_dist(expected_world_size=4)
    full_x, local_x, hidden_size, local_s = _prepare_inputs(rank, world_size)
    mesh = init_device_mesh("npu", (world_size,), mesh_dim_names=("cp",))

    model_cp = _TinyCpModel(hidden_size, num_heads=4, head_dim=4)
    AsyncContextParallel(seq_dim=1, head_dim=2, ulysses_degree=2).apply(
        module=model_cp.attn,
        device_mesh=mesh,
        q_proj=model_cp.q_proj,
        k_proj=model_cp.k_proj,
        v_proj=model_cp.v_proj,
    )
    cp_out = model_cp(local_x)

    model_ref = _TinyCpModel(hidden_size, num_heads=4, head_dim=4)
    ref_out = _build_local_reference(rank, local_s, model_ref(full_x))
    _assert_close(cp_out, ref_out, rank, "async_context_parallel_hybrid_forward", atol=5e-4, rtol=5e-4)
