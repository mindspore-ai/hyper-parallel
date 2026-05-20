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
from typing import Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from tests.torch.utils import init_backend, to_device

# pylint: disable=E1102

try:
    import torch_npu  # pylint: disable=W0611
    _DEVICE_TYPE = "npu"
except ImportError:
    _DEVICE_TYPE = "cpu"

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


def bnsd_tensors_d() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create BNSD [B, N, S, D] tensors on the current device.

    Returns:
        Tuple of (query, key, value) tensors on the target device.
    """
    q = to_device(torch.from_numpy(global_query_np.copy()), _DEVICE_TYPE)
    k = to_device(torch.from_numpy(global_key_np.copy()), _DEVICE_TYPE)
    v = to_device(torch.from_numpy(global_value_np.copy()), _DEVICE_TYPE)
    return q, k, v


def gqa_tensors_d(q_heads: int = NUM_HEADS, kv_heads: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Q/K/V tensors for grouped-query attention on the current device.

    Args:
        q_heads: Number of query heads.
        kv_heads: Number of key/value heads.

    Returns:
        Tuple of (query, key, value) tensors on the target device.
    """
    q = to_device(torch.from_numpy(global_query_np[:, :q_heads, :, :].copy()), _DEVICE_TYPE)
    k = to_device(torch.from_numpy(global_key_np[:, :kv_heads, :, :].copy()), _DEVICE_TYPE)
    v = to_device(torch.from_numpy(global_value_np[:, :kv_heads, :, :].copy()), _DEVICE_TYPE)
    return q, k, v


def run_standalone_d(is_causal: bool = False,
                     attn_mask: Optional[torch.Tensor] = None,
                     scale: float = SCALE, dropout_p: float = 0.0,
                     enable_gqa: bool = False) -> torch.Tensor:
    """Run standalone SDPA as ground truth on the current device.

    Args:
        is_causal: Whether to apply causal masking.
        attn_mask: Optional attention mask.
        scale: Scaling factor for dot products.
        dropout_p: Dropout probability.
        enable_gqa: Whether to enable grouped-query attention.

    Returns:
        SDPA output tensor on the target device.
    """
    q, k, v = bnsd_tensors_d()
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale, enable_gqa=enable_gqa,
    )


def assert_close(actual: torch.Tensor, expected: torch.Tensor,
                 atol: float = 1e-2, rtol: float = 1e-2,
                 msg: str = "Output mismatch") -> None:
    """Assert two tensors are numerically close.

    Args:
        actual: Actual output tensor.
        expected: Expected output tensor.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        msg: Error message on failure.
    """
    ok = np.allclose(
        actual.cpu().float().numpy(),
        expected.cpu().float().numpy(),
        atol=atol, rtol=rtol,
    )
    assert ok, msg


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_sdpa_replicate() -> None:
    """Test SDPA with fully replicated tensors."""
    init_backend(_DEVICE_TYPE)
    expected = run_standalone_d()

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(1,), mesh_dim_names=("dp",))
    q, k, v = bnsd_tensors_d()
    dq = DTensor.from_local(q, mesh, (Replicate(),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
    assert result.to_local().shape == expected.shape
    assert_close(result.to_local(), expected)


def test_sdpa_dp() -> None:
    """Test SDPA with data parallelism (batch sharding)."""
    init_backend(_DEVICE_TYPE)
    expected = run_standalone_d()

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(4,), mesh_dim_names=("dp",))
    q, k, v = bnsd_tensors_d()
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE // 4, NUM_HEADS, SEQ_LEN, HEAD_DIM)

    gathered = result.redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_sdpa_mp() -> None:
    """Test SDPA with head parallelism (head sharding)."""
    init_backend(_DEVICE_TYPE)
    expected = run_standalone_d()

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(4,), mesh_dim_names=("mp",))
    q, k, v = bnsd_tensors_d()
    placements = (Shard(1),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE, NUM_HEADS // 4, SEQ_LEN, HEAD_DIM)

    gathered = result.redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_sdpa_sp() -> None:
    """Test SDPA with sequence parallelism."""
    init_backend(_DEVICE_TYPE)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(4,), mesh_dim_names=("sp",))
    q, k, v = bnsd_tensors_d()
    dq = distribute_tensor(q, mesh, (Shard(2),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN // 4, HEAD_DIM)


def test_sdpa_dp_mp_2d() -> None:
    """Test SDPA with DP + MP on 2D mesh."""
    init_backend(_DEVICE_TYPE)
    expected = run_standalone_d()

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "mp"))
    q, k, v = bnsd_tensors_d()
    placements = (Shard(0), Shard(1))
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE // 2, NUM_HEADS // 2, SEQ_LEN, HEAD_DIM)

    gathered = result.redistribute(mesh, (Replicate(), Replicate())).to_local()
    assert_close(gathered, expected)


def test_sdpa_sp_mp_2d() -> None:
    """Test SDPA with SP + MP on 2D mesh."""
    init_backend(_DEVICE_TYPE)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("sp", "mp"))
    q, k, v = bnsd_tensors_d()
    dq = distribute_tensor(q, mesh, (Shard(2), Shard(1)))
    dk = distribute_tensor(k, mesh, (Replicate(), Shard(1)))
    dv = distribute_tensor(v, mesh, (Replicate(), Shard(1)))

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE, NUM_HEADS // 2, SEQ_LEN // 2, HEAD_DIM)


def test_sdpa_sp_causal() -> None:
    """Test SDPA SP with is_causal=True."""
    init_backend(_DEVICE_TYPE)
    expected = run_standalone_d(is_causal=True)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(4,), mesh_dim_names=("sp",))
    q, k, v = bnsd_tensors_d()
    dq = distribute_tensor(q, mesh, (Shard(2),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = F.scaled_dot_product_attention(dq, dk, dv, is_causal=True, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN // 4, HEAD_DIM)

    gathered = result.redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_sdpa_sp_explicit_mask() -> None:
    """Test SDPA SP with explicit attn_mask."""
    init_backend(_DEVICE_TYPE)
    mask = to_device(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool).tril(diagonal=0), _DEVICE_TYPE)
    expected = run_standalone_d(attn_mask=mask)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(4,), mesh_dim_names=("sp",))
    q, k, v = bnsd_tensors_d()
    dq = distribute_tensor(q, mesh, (Shard(2),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = F.scaled_dot_product_attention(dq, dk, dv, attn_mask=mask, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN // 4, HEAD_DIM)

    gathered = result.redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_sdpa_custom_scale() -> None:
    """Test SDPA with custom scale."""
    init_backend(_DEVICE_TYPE)
    custom_scale = 0.125
    expected = run_standalone_d(scale=custom_scale)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(4,), mesh_dim_names=("dp",))
    q, k, v = bnsd_tensors_d()
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=custom_scale)
    gathered = result.redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_sdpa_dropout() -> None:
    """Test SDPA with dropout."""
    init_backend(_DEVICE_TYPE)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(4,), mesh_dim_names=("dp",))
    q, k, v = bnsd_tensors_d()
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = F.scaled_dot_product_attention(dq, dk, dv, dropout_p=0.1, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE // 4, NUM_HEADS, SEQ_LEN, HEAD_DIM)


def test_sdpa_enable_gqa() -> None:
    """Test SDPA with enable_gqa=True."""
    init_backend(_DEVICE_TYPE)
    q, k, v = gqa_tensors_d(q_heads=NUM_HEADS, kv_heads=4)

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(4,), mesh_dim_names=("dp",))
    dq = distribute_tensor(q, mesh, (Shard(0),))
    dk = distribute_tensor(k, mesh, (Shard(0),))
    dv = distribute_tensor(v, mesh, (Shard(0),))

    result = F.scaled_dot_product_attention(dq, dk, dv, enable_gqa=True, scale=SCALE)
    assert result.to_local().shape == (BATCH_SIZE // 4, NUM_HEADS, SEQ_LEN, HEAD_DIM)


def test_sdpa_sp_correctness() -> None:
    """Test SDPA SP gathered output correctness."""
    init_backend(_DEVICE_TYPE)
    expected = run_standalone_d()

    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(4,), mesh_dim_names=("sp",))
    q, k, v = bnsd_tensors_d()
    dq = distribute_tensor(q, mesh, (Shard(2),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
    gathered = result.redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


# ---------------------------------------------------------------------------
# NPU-only error tests
# ---------------------------------------------------------------------------

def test_sdpa_error_kv_strategy_mismatch() -> None:
    """Verify SDPA rejects mismatched Key and Value sharding strategies."""
    init_backend("npu")

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("mp",))
    q = to_device(torch.from_numpy(global_query_np.copy()), "npu")
    k = to_device(torch.from_numpy(global_key_np.copy()), "npu")
    v = to_device(torch.from_numpy(global_value_np.copy()), "npu")
    dq = distribute_tensor(q, mesh, (Shard(1),))
    dk = distribute_tensor(k, mesh, (Shard(1),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    with pytest.raises(ValueError, match="Key and Value must have identical"):
        F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)


def test_sdpa_error_kv_seq_sharding() -> None:
    """Verify SDPA rejects KV sequence sharding without Ring Attention."""
    init_backend("npu")

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4,), mesh_dim_names=("sp",))
    q = to_device(torch.from_numpy(global_query_np.copy()), "npu")
    k = to_device(torch.from_numpy(global_key_np.copy()), "npu")
    v = to_device(torch.from_numpy(global_value_np.copy()), "npu")
    placements = (Shard(2),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    with pytest.raises(NotImplementedError, match="KV sequence sharding"):
        F.scaled_dot_product_attention(dq, dk, dv, scale=SCALE)
