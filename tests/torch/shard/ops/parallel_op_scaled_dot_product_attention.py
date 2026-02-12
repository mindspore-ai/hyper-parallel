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
"""test torch dtensor with scaled_dot_product_attention distributed operator."""

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.dtensor import distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist

np.random.seed(42)

BATCH_SIZE = 8
NUM_HEADS = 16
SEQ_LEN = 256
HEAD_DIM = 64
DTYPE = torch.float16
SCALE = 1.0 / math.sqrt(HEAD_DIM)

global_query_np = np.random.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM).astype(np.float16)
global_key_np = np.random.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM).astype(np.float16)
global_value_np = np.random.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM).astype(np.float16)


def bnsd_tensors():
    """Create BNSD [B, N, S, D] tensors on NPU from global numpy arrays."""
    q = torch.from_numpy(global_query_np).npu()
    k = torch.from_numpy(global_key_np).npu()
    v = torch.from_numpy(global_value_np).npu()
    return q, k, v


def gqa_tensors(q_heads=NUM_HEADS, kv_heads=4):
    """Create Q/K/V tensors for grouped-query attention (Q has more heads than KV)."""
    q = torch.from_numpy(global_query_np[:, :q_heads, :, :]).npu()
    k = torch.from_numpy(global_key_np[:, :kv_heads, :, :]).npu()
    v = torch.from_numpy(global_value_np[:, :kv_heads, :, :]).npu()
    return q, k, v


def run_standalone(is_causal=False, attn_mask=None, scale=SCALE,
                   dropout_p=0.0, enable_gqa=False):
    """Run standalone SDPA as ground truth."""
    q, k, v = bnsd_tensors()
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
    )


def assert_close(actual, expected, atol=1e-2, rtol=1e-2, msg="Output mismatch"):
    """Assert two tensors are numerically close."""
    ok = np.allclose(
        actual.cpu().float().numpy(),
        expected.cpu().float().numpy(),
        atol=atol, rtol=rtol,
    )
    assert ok, msg


def test_sdpa_replicate():
    """Verify SDPA with fully replicated tensors matches standalone execution."""
    init_dist()
    expected = run_standalone()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(1,), mesh_dim_names=("dp",))
    q, k, v = bnsd_tensors()
    dq = DTensor.from_local(q, mesh, (Replicate(),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
    assert result.to_local().shape == expected.shape
    assert_close(result.to_local(), expected)


def test_sdpa_dp():
    """Verify SDPA with data parallelism (batch sharding) matches standalone."""
    init_dist()
    expected = run_standalone()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bnsd_tensors()
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE // 8, NUM_HEADS, SEQ_LEN, HEAD_DIM)

    gathered = result.redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_sdpa_mp():
    """Verify SDPA with head parallelism (head sharding) matches standalone."""
    init_dist()
    expected = run_standalone()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("mp",))
    q, k, v = bnsd_tensors()
    placements = (Shard(1),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE, NUM_HEADS // 8, SEQ_LEN, HEAD_DIM)

    gathered = result.redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_sdpa_sp():
    """Verify SDPA with sequence parallelism produces correct local shape."""
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bnsd_tensors()
    dq = distribute_tensor(q, mesh, (Shard(2),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN // 8, HEAD_DIM)


def test_sdpa_dp_mp_2d():
    """Verify SDPA with DP + MP on 2D mesh matches standalone."""
    init_dist()
    expected = run_standalone()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))
    q, k, v = bnsd_tensors()
    placements = (Shard(0), Shard(1))
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE // 4, NUM_HEADS // 2, SEQ_LEN, HEAD_DIM)

    gathered = result.redistribute(mesh, (Replicate(), Replicate())).to_local()
    assert_close(gathered, expected)


def test_sdpa_sp_mp_2d():
    """Verify SDPA with SP + MP on 2D mesh produces correct local shape."""
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("sp", "mp"))
    q, k, v = bnsd_tensors()
    dq = distribute_tensor(q, mesh, (Shard(2), Shard(1)))
    dk = distribute_tensor(k, mesh, (Replicate(), Shard(1)))
    dv = distribute_tensor(v, mesh, (Replicate(), Shard(1)))

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE, NUM_HEADS // 2, SEQ_LEN // 4, HEAD_DIM)


def test_sdpa_sp_causal():
    """Verify SDPA SP with is_causal=True constructs correct chunk causal mask."""
    init_dist()
    expected = run_standalone(is_causal=True)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bnsd_tensors()
    dq = distribute_tensor(q, mesh, (Shard(2),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = F.scaled_dot_product_attention(dq, dk, dv, is_causal=True, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN // 8, HEAD_DIM)

    gathered = result.redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_sdpa_sp_explicit_mask():
    """Verify SDPA SP with explicit attn_mask slices mask to local Q range."""
    init_dist()
    mask = torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool).npu().tril(diagonal=0)
    expected = run_standalone(attn_mask=mask)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bnsd_tensors()
    dq = distribute_tensor(q, mesh, (Shard(2),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = F.scaled_dot_product_attention(dq, dk, dv, attn_mask=mask, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN // 8, HEAD_DIM)

    gathered = result.redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_sdpa_error_kv_strategy_mismatch():
    """Verify SDPA rejects mismatched Key and Value sharding strategies."""
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("mp",))
    q, k, v = bnsd_tensors()
    dq = distribute_tensor(q, mesh, (Shard(1),))
    dk = distribute_tensor(k, mesh, (Shard(1),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    with pytest.raises(ValueError, match="Key and Value must have identical"):
        F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)


def test_sdpa_error_kv_seq_sharding():
    """Verify SDPA rejects KV sequence sharding without Ring Attention."""
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bnsd_tensors()
    placements = (Shard(2),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    with pytest.raises(NotImplementedError, match="KV sequence sharding"):
        F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)


def test_sdpa_custom_scale():
    """Verify SDPA with custom scale=0.125 matches standalone execution."""
    init_dist()
    custom_scale = 0.125
    expected = run_standalone(scale=custom_scale)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bnsd_tensors()
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=custom_scale)
    gathered = result.redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_sdpa_dropout():
    """Verify SDPA with dropout_p=0.1 produces correct output shape."""
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bnsd_tensors()
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = F.scaled_dot_product_attention(dq, dk, dv, dropout_p=0.1, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE // 8, NUM_HEADS, SEQ_LEN, HEAD_DIM)


def test_sdpa_enable_gqa():
    """Verify SDPA with enable_gqa=True and Q 16 heads, KV 4 heads."""
    init_dist()
    q, k, v = gqa_tensors(q_heads=NUM_HEADS, kv_heads=4)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    dq = distribute_tensor(q, mesh, (Shard(0),))
    dk = distribute_tensor(k, mesh, (Shard(0),))
    dv = distribute_tensor(v, mesh, (Shard(0),))

    result = F.scaled_dot_product_attention(dq, dk, dv, enable_gqa=True, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE // 8, NUM_HEADS, SEQ_LEN, HEAD_DIM)


def test_sdpa_sp_correctness():
    """Verify SDPA SP gathered output matches standalone execution."""
    init_dist()
    expected = run_standalone()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bnsd_tensors()
    dq = distribute_tensor(q, mesh, (Shard(2),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
    gathered = result.redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)
