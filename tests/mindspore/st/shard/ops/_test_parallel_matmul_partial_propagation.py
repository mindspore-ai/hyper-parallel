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
"""Test chained matmul Partial propagation under TP and DP×TP configurations."""
import numpy as np
import mindspore as ms
from mindspore import mint, ops, Tensor
import mindspore.communication.management as D

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate


def setup_module():
    """Initialize the distributed environment."""
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    D.init()


# ---------------------------------------------------------------------------
# Test 1: matmul, tp=2
# ---------------------------------------------------------------------------

def test_chained_matmul_partial_propagation_tp2():
    """
    Feature: Chained matmul Partial propagation TP=2.
    Description: (x @ A.T) @ (B.T) with row-wise TP, pre-transposed inputs.
    Expectation: full_tensor() matches single-card gold.
    """
    np.random.seed(42)
    rank_dim, in_dim, out_dim, batch = 8, 16, 32, 4
    tp_size = D.get_group_size()
    mesh = init_device_mesh("npu", (tp_size,), mesh_dim_names=("tp",))

    x_f = np.random.randn(batch, in_dim).astype(np.float32)
    a_f = (np.random.randn(rank_dim, in_dim) * 0.1).astype(np.float32)
    b_f = (np.random.randn(out_dim, rank_dim) * 0.1).astype(np.float32)

    gold = x_f @ a_f.T @ b_f.T

    a_t_full = Tensor(a_f.T)
    b_t_full = Tensor(b_f.T)
    x_full = Tensor(x_f)

    x_dt = distribute_tensor(x_full, mesh, (Shard(1),))
    a_t_dt = distribute_tensor(a_t_full, mesh, (Shard(0),))
    b_t_dt = distribute_tensor(b_t_full, mesh, (Replicate(),))

    tmp = mint.matmul(x_dt, a_t_dt)
    out = mint.matmul(tmp, b_t_dt)
    result = out.redistribute(device_mesh=mesh, placements=(Replicate(),)).to_local()

    assert np.allclose(result.asnumpy(), gold, rtol=1e-4, atol=1e-4), (
        f"mm_tp2 mismatch: got {result.asnumpy().sum():.5f}, expected {gold.sum():.5f}"
    )


# ---------------------------------------------------------------------------
# Test 2: matmul, dp=2 × tp=2
# ---------------------------------------------------------------------------

def test_chained_matmul_dp_tp_partial_propagation():
    """
    Feature: Chained matmul Partial propagation under DP×TP.
    Description: (x @ A.T) @ (B.T) with 2D mesh (dp, tp), pre-transposed.
    Expectation: full_tensor() matches single-card gold.
    """
    np.random.seed(43)
    rank_dim, in_dim, out_dim, batch = 8, 16, 32, 8
    world = D.get_group_size()
    tp_size = 2
    dp_size = world // tp_size
    mesh = init_device_mesh("npu", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

    x_f = np.random.randn(batch, in_dim).astype(np.float32)
    a_f = (np.random.randn(rank_dim, in_dim) * 0.1).astype(np.float32)
    b_f = (np.random.randn(out_dim, rank_dim) * 0.1).astype(np.float32)

    gold = x_f @ a_f.T @ b_f.T

    a_t_full = Tensor(a_f.T)
    b_t_full = Tensor(b_f.T)
    x_full = Tensor(x_f)

    x_dt = distribute_tensor(x_full, mesh, (Shard(0), Shard(1)))
    a_t_dt = distribute_tensor(a_t_full, mesh, (Replicate(), Shard(0)))
    b_t_dt = distribute_tensor(b_t_full, mesh, (Replicate(), Replicate()))

    tmp = mint.matmul(x_dt, a_t_dt)
    out = mint.matmul(tmp, b_t_dt)
    result = out.redistribute(device_mesh=mesh, placements=(Replicate(), Replicate())).to_local()

    assert np.allclose(result.asnumpy(), gold, rtol=1e-4, atol=1e-4), (
        f"mm_dp2_tp2 mismatch: got {result.asnumpy().sum():.5f}, expected {gold.sum():.5f}"
    )


# ---------------------------------------------------------------------------
# Test 3: linear, tp=2
# ---------------------------------------------------------------------------

def test_linear_partial_propagation_tp2():
    """
    Feature: LinearDistributedOp propagates Partial from input x.
    Description: matmul → ops.dense (no bias) chain under TP=2, pre-transposed.
    Expectation: full_tensor() matches single-card gold.
    """
    np.random.seed(44)
    hid_dim, in_dim, out_dim, batch = 12, 16, 32, 4
    tp_size = D.get_group_size()
    mesh = init_device_mesh("npu", (tp_size,), mesh_dim_names=("tp",))

    x_f = np.random.randn(batch, in_dim).astype(np.float32)
    a_f = (np.random.randn(hid_dim, in_dim) * 0.1).astype(np.float32)
    w_f = (np.random.randn(out_dim, hid_dim) * 0.1).astype(np.float32)

    gold = x_f @ a_f.T @ w_f.T

    a_t_full = Tensor(a_f.T)
    x_full = Tensor(x_f)
    w_full = Tensor(w_f)

    x_dt = distribute_tensor(x_full, mesh, (Shard(1),))
    a_t_dt = distribute_tensor(a_t_full, mesh, (Shard(0),))
    w_dt = distribute_tensor(w_full, mesh, (Replicate(),))

    tmp = mint.matmul(x_dt, a_t_dt)
    out = ops.dense(tmp, w_dt)
    result = out.redistribute(device_mesh=mesh, placements=(Replicate(),)).to_local()

    assert np.allclose(result.asnumpy(), gold, rtol=1e-4, atol=1e-4), (
        f"linear_tp2 mismatch: got {result.asnumpy().sum():.5f}, expected {gold.sum():.5f}"
    )
