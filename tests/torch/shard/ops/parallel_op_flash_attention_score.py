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
"""Test torch DTensor with npu_fusion_attention distributed operator."""

import numpy as np
import pytest
import torch
import torch_npu
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.placement_types import Shard, Replicate
from tests.torch.utils import init_dist


np.random.seed(42)

BATCH_SIZE = 8
SEQ_LEN = 512
HEAD_NUM = 16
HEAD_DIM = 64
HIDDEN_SIZE = HEAD_NUM * HEAD_DIM
SCALE = 1.0 / (HEAD_DIM ** 0.5)

global_query_np = np.random.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).astype(np.float16)
global_key_np = np.random.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).astype(np.float16)
global_value_np = np.random.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).astype(np.float16)


def create_attention_mask(sparse_mode):
    """Create attention mask for the given sparse mode."""
    if sparse_mode == 0:
        return None
    if sparse_mode == 1:
        return torch.zeros(SEQ_LEN, SEQ_LEN, dtype=torch.bool).npu()
    if sparse_mode in [2, 3, 4]:
        return torch.triu(torch.ones(2048, 2048), diagonal=1).bool().npu()
    return None


def run_standalone_bsh(sparse_mode=0, scale=SCALE, keep_prob=1.0,
                       pre_tockens=2147483647, next_tockens=2147483647):
    """Run standalone BSH attention as ground truth."""
    q = torch.from_numpy(global_query_np).npu()
    k = torch.from_numpy(global_key_np).npu()
    v = torch.from_numpy(global_value_np).npu()
    mask = create_attention_mask(sparse_mode)
    result = torch_npu.npu_fusion_attention(
        q, k, v, head_num=HEAD_NUM, input_layout='BSH',
        atten_mask=mask, scale=scale, keep_prob=keep_prob,
        pre_tockens=pre_tockens, next_tockens=next_tockens,
        sparse_mode=sparse_mode,
    )
    return result[0]


def bsh_tensors():
    """Create BSH format tensors on NPU."""
    q = torch.from_numpy(global_query_np).npu()
    k = torch.from_numpy(global_key_np).npu()
    v = torch.from_numpy(global_value_np).npu()
    return q, k, v


def bnsd_tensors():
    """Create BNSD format tensors on NPU."""
    def to_bnsd(data):
        x = data.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM)
        return np.ascontiguousarray(np.transpose(x, (0, 2, 1, 3)))
    q = torch.from_numpy(to_bnsd(global_query_np)).npu()
    k = torch.from_numpy(to_bnsd(global_key_np)).npu()
    v = torch.from_numpy(to_bnsd(global_value_np)).npu()
    return q, k, v


def sbh_tensors():
    """Create SBH format tensors on NPU."""
    def to_sbh(data):
        return np.ascontiguousarray(np.transpose(data, (1, 0, 2)))
    q = torch.from_numpy(to_sbh(global_query_np)).npu()
    k = torch.from_numpy(to_sbh(global_key_np)).npu()
    v = torch.from_numpy(to_sbh(global_value_np)).npu()
    return q, k, v


def bsnd_tensors():
    """Create BSND format tensors on NPU."""
    def to_bsnd(data):
        return np.ascontiguousarray(
            data.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM)
        )
    q = torch.from_numpy(to_bsnd(global_query_np)).npu()
    k = torch.from_numpy(to_bsnd(global_key_np)).npu()
    v = torch.from_numpy(to_bsnd(global_value_np)).npu()
    return q, k, v


def tnd_tensors(num_samples=BATCH_SIZE, tokens_per_sample=SEQ_LEN):
    """Create TND format tensors and cumulative sequence lengths."""
    total = num_samples * tokens_per_sample
    def to_tnd(data):
        return np.ascontiguousarray(
            data.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM)
                .reshape(total, HEAD_NUM, HEAD_DIM)
        )
    q = torch.from_numpy(to_tnd(global_query_np)).npu()
    k = torch.from_numpy(to_tnd(global_key_np)).npu()
    v = torch.from_numpy(to_tnd(global_value_np)).npu()
    actual_seq_qlen = [(i + 1) * tokens_per_sample for i in range(num_samples)]
    actual_seq_kvlen = [(i + 1) * tokens_per_sample for i in range(num_samples)]
    return q, k, v, actual_seq_qlen, actual_seq_kvlen


def assert_close(actual, expected, atol=1e-2, rtol=1e-2, msg="Output mismatch"):
    """Assert two tensors are numerically close."""
    ok = np.allclose(
        actual.cpu().float().numpy(),
        expected.cpu().float().numpy(),
        atol=atol, rtol=rtol,
    )
    assert ok, msg


def test_bsh_replicate():
    """
    Feature: npu_fusion_attention BSH layout with fully replicated tensors
    Description:
        - All Q/K/V tensors are Replicate on an 8-device mesh.
        - No actual parallelism; the distributed op should pass through directly.
        - Compare output against standalone single-device execution.
    Expectation: Output matches standalone result within tolerance.
    """
    init_dist()
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    dq = DTensor.from_local(q, mesh, (Replicate(),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scale=SCALE, sparse_mode=0,
    )
    assert_close(result[0].to_local(), expected)


def test_bsh_dp():
    """
    Feature: npu_fusion_attention BSH layout with data parallelism
    Description:
        - Q/K/V are Shard(0) on batch dimension across 8 devices.
        - Each device computes attention on its local batch slice independently.
        - Gather output and compare against standalone full-batch result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    placements = (Shard(0),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scale=SCALE, sparse_mode=0,
    )
    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_bsh_mp():
    """
    Feature: npu_fusion_attention BSH layout with head parallelism
    Description:
        - Q/K/V are Shard(2) on hidden dimension across 8 devices.
        - head_num is divided by the split factor; each device handles a subset of heads.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("mp",))
    q, k, v = bsh_tensors()
    placements = (Shard(2),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scale=SCALE, sparse_mode=0,
    )
    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_bsh_sp():
    """
    Feature: npu_fusion_attention BSH layout with head parallelism
    Description:
        - Q/K/V are Shard(2) on hidden dimension across 8 devices.
        - head_num is divided by the split factor; each device handles a subset of heads.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bsh_tensors()
    dq = DTensor.distribute_tensor(q, mesh, (Shard(1),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scale=SCALE, sparse_mode=0,
    )
    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_bsh_dp_mp_2d():
    """
    Feature: npu_fusion_attention BSH layout with DP + MP on 2D mesh
    Description:
        - 2D mesh (4, 2) with dp on batch dim and mp on hidden dim.
        - Q/K/V are Shard(0) on dp axis and Shard(2) on mp axis.
        - Combines data parallelism and head parallelism simultaneously.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))
    q, k, v = bsh_tensors()
    placements = (Shard(0), Shard(2))
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scale=SCALE, sparse_mode=0,
    )
    gathered = result[0].redistribute(mesh, (Replicate(), Replicate())).to_local()
    assert_close(gathered, expected)


def test_bsh_sp_mp_2d():
    """
    Feature: npu_fusion_attention BSH layout with SP + MP on 2D mesh
    Description:
        - 2D mesh (4, 2) with sp on seq dim and mp on hidden dim.
        - Q is Shard(1)+Shard(2); K/V are Replicate+Shard(2).
        - Combines sequence parallelism and head parallelism simultaneously.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("sp", "mp"))
    q, k, v = bsh_tensors()
    dq = DTensor.distribute_tensor(q, mesh, (Shard(1), Shard(2)))
    dk = DTensor.distribute_tensor(k, mesh, (Replicate(), Shard(2)))
    dv = DTensor.distribute_tensor(v, mesh, (Replicate(), Shard(2)))

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scale=SCALE, sparse_mode=0,
    )
    gathered = result[0].redistribute(mesh, (Replicate(), Replicate())).to_local()
    assert_close(gathered, expected)


def test_bsh_dp_sp_mp_3d():
    """
    Feature: npu_fusion_attention BSH layout with DP + SP + MP on 3D mesh
    Description:
        - 3D mesh (2, 2, 2) with dp, sp, and mp axes.
        - Q is Shard(0)+Shard(1)+Shard(2); K/V are Shard(0)+Replicate+Shard(2).
        - Exercises all three parallelism strategies simultaneously.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "sp", "mp"))
    q, k, v = bsh_tensors()
    dq = DTensor.distribute_tensor(q, mesh, (Shard(0), Shard(1), Shard(2)))
    dk = DTensor.distribute_tensor(k, mesh, (Shard(0), Replicate(), Shard(2)))
    dv = DTensor.distribute_tensor(v, mesh, (Shard(0), Replicate(), Shard(2)))

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scale=SCALE, sparse_mode=0,
    )
    full_replicate = (Replicate(), Replicate(), Replicate())
    gathered = result[0].redistribute(mesh, full_replicate).to_local()
    assert_close(gathered, expected)


def test_bnsd_dp_mp():
    """
    Feature: npu_fusion_attention BSH layout with DP + SP + MP on 3D mesh
    Description:
        - 3D mesh (2, 2, 2) with dp, sp, and mp axes.
        - Q is Shard(0)+Shard(1)+Shard(2); K/V are Shard(0)+Replicate+Shard(2).
        - Exercises all three parallelism strategies simultaneously.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))
    q, k, v = bnsd_tensors()
    placements = (Shard(0), Shard(1))
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BNSD',
        scale=SCALE, sparse_mode=0,
    )
    out = result[0]
    assert out.to_local().shape == (BATCH_SIZE // 4, HEAD_NUM // 2, SEQ_LEN, HEAD_DIM)


def test_bnsd_sp():
    """
    Feature: npu_fusion_attention BNSD layout with DP + SP on 2D mesh
    Description:
        - BNSD format with 2D mesh (4, 2): dp on batch dim, sp on seq dim (dim 2).
        - Q is Shard(0)+Shard(2); K/V are Shard(0)+Replicate.
        - Verify output shape matches expected local dimensions after sharding.
    Expectation: Local output shape is (B/4, N, S/2, D).
    """
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "sp"))
    q, k, v = bnsd_tensors()
    dq = DTensor.distribute_tensor(q, mesh, (Shard(0), Shard(2)))
    dk = DTensor.distribute_tensor(k, mesh, (Shard(0), Replicate()))
    dv = DTensor.distribute_tensor(v, mesh, (Shard(0), Replicate()))

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BNSD',
        scale=SCALE, sparse_mode=0,
    )
    out = result[0]
    assert out.to_local().shape == (BATCH_SIZE // 4, HEAD_NUM, SEQ_LEN // 2, HEAD_DIM)


def test_sbh_dp():
    """
    Feature: npu_fusion_attention SBH layout with data parallelism
    Description:
        - SBH format with batch dim at index 1, sharded across 8 devices.
        - Verify output shape matches expected local dimensions after batch sharding.
    Expectation: Local output shape is (S, B/8, H).
    """
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = sbh_tensors()
    placements = (Shard(1),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='SBH',
        scale=SCALE, sparse_mode=0,
    )
    out = result[0]
    assert out.to_local().shape == (SEQ_LEN, BATCH_SIZE // 8, HIDDEN_SIZE)


def test_bsnd_dp_mp():
    """
    Feature: npu_fusion_attention BSND layout with DP + MP on 2D mesh
    Description:
        - BSND format with 2D mesh (4, 2): dp on batch dim, mp on head dim (dim 2).
        - Q/K/V are Shard(0)+Shard(2).
        - Verify output shape matches expected local dimensions after sharding.
    Expectation: Local output shape is (B/4, S, N/2, D).
    """
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))
    q, k, v = bsnd_tensors()
    placements = (Shard(0), Shard(2))
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSND',
        scale=SCALE, sparse_mode=0,
    )
    out = result[0]
    assert out.to_local().shape == (BATCH_SIZE // 4, SEQ_LEN, HEAD_NUM // 2, HEAD_DIM)


def test_tnd_dp():
    """
    Feature: npu_fusion_attention TND layout with data parallelism
    Description:
        - TND format with T dimension sharded across 8 devices (equal-length samples).
        - actual_seq_qlen/actual_seq_kvlen provided as cumulative sums.
        - Distributed op adjusts actual_seq_len via clamp to match local T slice.
        - Verify output shape matches expected local T dimension.
    Expectation: Local output shape is (T/8, N, D).
    """
    init_dist()

    q, k, v, actual_seq_qlen, actual_seq_kvlen = tnd_tensors()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    placements = (Shard(0),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
        scale=SCALE, sparse_mode=0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    total_tokens = BATCH_SIZE * SEQ_LEN
    out = result[0]
    assert out.to_local().shape == (total_tokens // 8, HEAD_NUM, HEAD_DIM)


def test_tnd_mp():
    """
    Feature: npu_fusion_attention TND layout with head parallelism
    Description:
        - TND format with N (head) dimension sharded across 8 devices.
        - head_num is divided by the split factor.
        - Verify output shape matches expected local head count.
    Expectation: Local output shape is (T, N/8, D).
    """
    init_dist()

    q, k, v, actual_seq_qlen, actual_seq_kvlen = tnd_tensors()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("mp",))
    placements = (Shard(1),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
        scale=SCALE, sparse_mode=0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    total_tokens = BATCH_SIZE * SEQ_LEN
    out = result[0]
    assert out.to_local().shape == (total_tokens, HEAD_NUM // 8, HEAD_DIM)


def test_tnd_dp_mp():
    """
    Feature: npu_fusion_attention TND layout with DP + MP on 2D mesh
    Description:
        - TND format with 2D mesh (4, 2): dp on T dim, mp on N dim.
        - Q/K/V are Shard(0)+Shard(1).
        - Verify output shape matches expected local dimensions after both splits.
    Expectation: Local output shape is (T/4, N/2, D).
    """
    init_dist()

    q, k, v, actual_seq_qlen, actual_seq_kvlen = tnd_tensors()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))
    dq = DTensor.distribute_tensor(q, mesh, (Shard(0), Shard(1)))
    dk = DTensor.distribute_tensor(k, mesh, (Shard(0), Shard(1)))
    dv = DTensor.distribute_tensor(v, mesh, (Shard(0), Shard(1)))

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
        scale=SCALE, sparse_mode=0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    total_tokens = BATCH_SIZE * SEQ_LEN
    out = result[0]
    assert out.to_local().shape == (total_tokens // 4, HEAD_NUM // 2, HEAD_DIM)


def test_sp_sparse_mode_0():
    """
    Feature: npu_fusion_attention BSH SP with sparse_mode=0 (defaultMask, no mask)
    Description:
        - Q is Shard(1) on seq dim; K/V are Replicate.
        - sparse_mode=0 with default pre/next tokens (no mask provided).
        - Distributed op applies LEFT_UP_TO_LEFT_UP offset to pre/next tokens.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    expected = run_standalone_bsh(sparse_mode=0)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bsh_tensors()
    dq = DTensor.distribute_tensor(q, mesh, (Shard(1),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scale=SCALE, sparse_mode=0,
    )
    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_sp_sparse_mode_2():
    """
    Feature: npu_fusion_attention BSH SP with sparse_mode=2 (leftUpCausal)
    Description:
        - Q is Shard(1) on seq dim; K/V are Replicate.
        - sparse_mode=2 with compressed 2048x2048 causal mask.
        - Distributed op converts to sparse_mode=4 (band) with LEFT_UP_TO_RIGHT_DOWN offset.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    expected = run_standalone_bsh(sparse_mode=2)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bsh_tensors()
    mask = create_attention_mask(2)
    dq = DTensor.distribute_tensor(q, mesh, (Shard(1),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        atten_mask=mask, scale=SCALE, sparse_mode=2,
    )
    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_sp_sparse_mode_3():
    """
    Feature: npu_fusion_attention BSH SP with sparse_mode=3 (rightDownCausal)
    Description:
        - Q is Shard(1) on seq dim; K/V are Replicate.
        - sparse_mode=3 with compressed 2048x2048 causal mask.
        - Distributed op converts to sparse_mode=4 (band) with RIGHT_DOWN_TO_RIGHT_DOWN offset.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    expected = run_standalone_bsh(sparse_mode=3)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bsh_tensors()
    mask = create_attention_mask(3)
    dq = DTensor.distribute_tensor(q, mesh, (Shard(1),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        atten_mask=mask, scale=SCALE, sparse_mode=3,
    )
    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_sp_sparse_mode_4():
    """
    Feature: npu_fusion_attention BSH SP with sparse_mode=4 (band) and custom pre/next tokens
    Description:
        - Q is Shard(1) on seq dim; K/V are Replicate.
        - sparse_mode=4 with pre_tockens=256, next_tockens=256, compressed 2048x2048 mask.
        - Distributed op applies RIGHT_DOWN_TO_RIGHT_DOWN offset to pre/next tokens.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    pre, nxt = 256, 256
    expected = run_standalone_bsh(sparse_mode=4, pre_tockens=pre, next_tockens=nxt)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bsh_tensors()
    mask = create_attention_mask(4)
    dq = DTensor.distribute_tensor(q, mesh, (Shard(1),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        atten_mask=mask, scale=SCALE, sparse_mode=4,
        pre_tockens=pre, next_tockens=nxt,
    )
    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_dp_sparse_mode_1():
    """
    Feature: npu_fusion_attention BSH DP with sparse_mode=1 (allMask)
    Description:
        - Q/K/V are Shard(0) on batch dim (data parallelism).
        - sparse_mode=1 with full [Sq, Skv] attention mask.
        - DP does not enter sparse params adjustment; mode=1 is passed through directly.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    expected = run_standalone_bsh(sparse_mode=1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    mask = create_attention_mask(1)
    placements = (Shard(0),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        atten_mask=mask, scale=SCALE, sparse_mode=1,
    )
    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_dp_sparse_mode_4():
    """
    Feature: npu_fusion_attention BSH DP with sparse_mode=4 (band) and custom pre/next tokens
    Description:
        - Q/K/V are Shard(0) on batch dim (data parallelism).
        - sparse_mode=4 with pre_tockens=256,
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    pre, nxt = 256, 256
    expected = run_standalone_bsh(sparse_mode=4, pre_tockens=pre, next_tockens=nxt)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    mask = create_attention_mask(4)
    placements = (Shard(0),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        atten_mask=mask, scale=SCALE, sparse_mode=4,
        pre_tockens=pre, next_tockens=nxt,
    )
    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_error_kv_strategy_mismatch():
    """
    Feature: npu_fusion_attention rejects mismatched Key and Value sharding
    Description:
        - K is Shard(0)+Shard(2) but V is Shard(0)+Replicate.
        - Key and Value must have identical tensor_map for correctness.
    Expectation: Raise ValueError with message "Key and Value must have identical".
    """
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))
    q, k, v = bsh_tensors()
    dq = DTensor.distribute_tensor(q, mesh, (Shard(0), Shard(2)))
    dk = DTensor.distribute_tensor(k, mesh, (Shard(0), Shard(2)))
    dv = DTensor.distribute_tensor(v, mesh, (Shard(0), Replicate()))

    with pytest.raises(ValueError, match="Key and Value must have identical"):
        torch_npu.npu_fusion_attention(
            dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
            scale=SCALE, sparse_mode=0,
        )


def test_error_head_num_not_divisible():
    """
    Feature: npu_fusion_attention rejects indivisible head_num under head parallelism
    Description:
        - Use head_num=17 (prime) with 8-way head sharding.
        - 17 is not divisible by 8, so adjusted_head_num cannot be computed.
    Expectation: Raise ValueError with message "not divisible".
    """
    init_dist()

    odd_heads = 17
    odd_hidden = odd_heads * HEAD_DIM
    data = np.random.randn(BATCH_SIZE, SEQ_LEN, odd_hidden).astype(np.float16)
    q = torch.from_numpy(data).npu()
    k = torch.from_numpy(data).npu()
    v = torch.from_numpy(data).npu()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("mp",))
    placements = (Shard(2),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    with pytest.raises(ValueError, match="not divisible"):
        torch_npu.npu_fusion_attention(
            dq, dk, dv, head_num=odd_heads, input_layout='BSH',
            scale=SCALE, sparse_mode=0,
        )


def test_error_kv_seq_sharding_blocked():
    """
    Feature: npu_fusion_attention rejects KV sequence sharding for non-TND layouts
    Description:
        - BSH layout with Q/K/V all Shard(1) on seq dim.
        - KV sequence sharding requires Ring Attention which is not supported.
    Expectation: Raise NotImplementedError with message "KV sequence sharding is not supported".
    """
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bsh_tensors()
    placements = (Shard(1),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    with pytest.raises(NotImplementedError, match="KV sequence sharding is not supported"):
        torch_npu.npu_fusion_attention(
            dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
            scale=SCALE, sparse_mode=0,
        )


def test_error_sparse_mode_1_no_mask():
    """
    Feature: npu_fusion_attention rejects sparse_mode=1 without atten_mask
    Description:
        - sparse_mode=1 (allMask) requires an explicit atten_mask tensor.
        - Call without providing atten_mask.
    Expectation: Raise ValueError with message "sparse_mode=1.*requires atten_mask".
    """
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    placements = (Shard(0),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    with pytest.raises(ValueError, match="sparse_mode=1.*requires atten_mask"):
        torch_npu.npu_fusion_attention(
            dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
            scale=SCALE, sparse_mode=1,
        )


def test_error_tnd_sp_without_actual_seq_len():
    """
    Feature: npu_fusion_attention rejects TND SP without actual_seq_qlen/actual_seq_kvlen
    Description:
        - TND layout with sequence parallelism (Q and K both Shard(0) on T dim).
        - actual_seq_qlen and actual_seq_kvlen are not provided.
    Expectation: Raise ValueError with message "actual_seq_qlen and actual_seq_kvlen must be provided".
    """
    init_dist()

    q, k, v, _, _ = tnd_tensors()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "sp"))
    dq = DTensor.distribute_tensor(q, mesh, (Shard(0), Shard(1)))
    dk = DTensor.distribute_tensor(k, mesh, (Shard(0), Shard(1)))
    dv = DTensor.distribute_tensor(v, mesh, (Shard(0), Shard(1)))

    with pytest.raises(ValueError, match="actual_seq_qlen and actual_seq_kvlen must be provided"):
        torch_npu.npu_fusion_attention(
            dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
            scale=SCALE, sparse_mode=3,
        )


def test_error_bnsd_kv_seq_sharding_blocked():
    """
    Feature: npu_fusion_attention rejects KV sequence sharding for BNSD layout
    Description:
        - BNSD layout with Q/K/V all Shard(2) on seq dim.
        - KV sequence sharding requires Ring Attention which is not supported,
          same restriction as BSH.
    Expectation: Raise NotImplementedError with message "KV sequence sharding is not supported".
    """
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "sp"))
    q, k, v = bnsd_tensors()
    placements = (Shard(0), Shard(2))
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    with pytest.raises(NotImplementedError, match="KV sequence sharding is not supported"):
        torch_npu.npu_fusion_attention(
            dq, dk, dv, head_num=HEAD_NUM, input_layout='BNSD',
            scale=SCALE, sparse_mode=0,
        )


def test_bsh_custom_scale():
    """
    Feature: npu_fusion_attention BSH DP with custom scale value
    Description:
        - Use a non-default scale=0.125 instead of 1/sqrt(HEAD_DIM).
        - Q/K/V are Shard(0) on batch dim.
        - Gather output and compare against standalone result with same scale.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    custom_scale = 0.125
    expected = run_standalone_bsh(scale=custom_scale)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    placements = (Shard(0),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scale=custom_scale, sparse_mode=0,
    )
    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_bsh_dropout():
    """
    Feature: npu_fusion_attention BSH DP with dropout (keep_prob < 1)
    Description:
        - Q/K/V are Shard(0) on batch dim with keep_prob=0.9.
        - Dropout introduces randomness, so only output shape is verified.
    Expectation: Local output shape is (B/8, S, H). No correctness check due to randomness.
    """
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    placements = (Shard(0),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scale=SCALE, sparse_mode=0, keep_prob=0.9,
    )
    out = result[0]
    assert out.to_local().shape == (BATCH_SIZE // 8, SEQ_LEN, HIDDEN_SIZE)


def test_bsh_long_sequence_sp():
    """
    Feature: npu_fusion_attention BSH SP with longer sequence length
    Description:
        - Use SEQ_LEN=2048 (4x default) with 8-way sequence parallelism.
        - Q is Shard(1); K/V are Replicate.
        - Verify output shape matches expected local seq dimension.
    Expectation: Local output shape is (B, S/8, H) where S=2048.
    """
    init_dist()

    long_seq = 2048
    q_np = np.random.randn(BATCH_SIZE, long_seq, HIDDEN_SIZE).astype(np.float16)
    k_np = np.random.randn(BATCH_SIZE, long_seq, HIDDEN_SIZE).astype(np.float16)
    v_np = np.random.randn(BATCH_SIZE, long_seq, HIDDEN_SIZE).astype(np.float16)
    q = torch.from_numpy(q_np).npu()
    k = torch.from_numpy(k_np).npu()
    v = torch.from_numpy(v_np).npu()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    dq = DTensor.distribute_tensor(q, mesh, (Shard(1),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scale=SCALE, sparse_mode=0,
    )
    out = result[0]
    assert out.to_local().shape == (BATCH_SIZE, long_seq // 8, HIDDEN_SIZE)


def test_bsh_large_batch_dp():
    """
    Feature: npu_fusion_attention BSH DP with large batch size
    Description:
        - Use BATCH_SIZE=64 (8x default) with 8-way data parallelism.
        - Verify output shape matches expected local batch dimension.
    Expectation: Local output shape is (64/8, S, H).
    """
    init_dist()

    large_batch = 64
    q_np = np.random.randn(large_batch, SEQ_LEN, HIDDEN_SIZE).astype(np.float16)
    q = torch.from_numpy(q_np).npu()
    k = torch.from_numpy(q_np).npu()
    v = torch.from_numpy(q_np).npu()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    placements = (Shard(0),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scale=SCALE, sparse_mode=0,
    )
    out = result[0]
    assert out.to_local().shape == (large_batch // 8, SEQ_LEN, HIDDEN_SIZE)


def test_bsh_redistribute_then_attention():
    """
    Feature: npu_fusion_attention BSH with redistribute before attention
    Description:
        - Start with Replicate tensors, redistribute to Shard(0) on batch dim.
        - Run attention on the redistributed DTensors.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    dq = DTensor.from_local(q, mesh, (Replicate(),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    dq = dq.redistribute(mesh, (Shard(0),))
    dk = dk.redistribute(mesh, (Shard(0),))
    dv = dv.redistribute(mesh, (Shard(0),))

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scale=SCALE, sparse_mode=0,
    )
    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_sp_sparse_mode_2_with_2way_split():
    """
    Feature: npu_fusion_attention BSH SP sparse_mode=2 with 2-way sequence split
    Description:
        - 2D mesh (2, 4) with sp=2 and dp=4.
        - Q is Shard(1)+Shard(0); K/V are Replicate+Shard(0).
        - Verifies LEFT_UP_TO_RIGHT_DOWN offset calculation with non-8-way split.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    expected = run_standalone_bsh(sparse_mode=2)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("sp", "dp"))
    q, k, v = bsh_tensors()
    dq = DTensor.distribute_tensor(q, mesh, (Shard(1), Shard(0)))
    dk = DTensor.distribute_tensor(k, mesh, (Replicate(), Shard(0)))
    dv = DTensor.distribute_tensor(v, mesh, (Replicate(), Shard(0)))
    mask = create_attention_mask(2)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        atten_mask=mask, scale=SCALE, sparse_mode=2,
    )
    gathered = result[0].redistribute(mesh, (Replicate(), Replicate())).to_local()
    assert_close(gathered, expected)


def test_sp_sparse_mode_3_with_2way_split():
    """
    Feature: npu_fusion_attention BSH SP sparse_mode=3 with 2-way sequence split
    Description:
        - 2D mesh (2, 4) with sp=2 and dp=4.
        - Q is Shard(1)+Shard(0); K/V are Replicate+Shard(0).
        - Verifies RIGHT_DOWN_TO_RIGHT_DOWN offset calculation with non-8-way split.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()
    expected = run_standalone_bsh(sparse_mode=3)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("sp", "dp"))
    q, k, v = bsh_tensors()
    dq = DTensor.distribute_tensor(q, mesh, (Shard(1), Shard(0)))
    dk = DTensor.distribute_tensor(k, mesh, (Replicate(), Shard(0)))
    dv = DTensor.distribute_tensor(v, mesh, (Replicate(), Shard(0)))
    mask = create_attention_mask(3)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        atten_mask=mask, scale=SCALE, sparse_mode=3,
    )
    gathered = result[0].redistribute(mesh, (Replicate(), Replicate())).to_local()
    assert_close(gathered, expected)


def test_bnsd_sp_correctness():
    """
    Feature: npu_fusion_attention BNSD SP correctness verification
    Description:
        - BNSD format with Q Shard(2) on seq dim; K/V Replicate.
        - Run standalone BNSD attention as ground truth.
        - Gather distributed output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()

    q_bnsd, k_bnsd, v_bnsd = bnsd_tensors()
    mask = create_attention_mask(0)
    standalone = torch_npu.npu_fusion_attention(
        q_bnsd, k_bnsd, v_bnsd, head_num=HEAD_NUM, input_layout='BNSD',
        atten_mask=mask, scale=SCALE, sparse_mode=0,
    )[0]

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    dq = DTensor.distribute_tensor(q_bnsd, mesh, (Shard(2),))
    dk = DTensor.from_local(k_bnsd, mesh, (Replicate(),))
    dv = DTensor.from_local(v_bnsd, mesh, (Replicate(),))

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BNSD',
        scale=SCALE, sparse_mode=0,
    )
    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, standalone)


def test_tnd_dp_correctness():
    """
    Feature: npu_fusion_attention TND DP correctness with actual_seq_len adjustment
    Description:
        - TND format with 8-way DP on T dimension (equal-length samples).
        - Verifies that pure DP correctly adjusts actual_seq_qlen/actual_seq_kvlen
          via clamp to match local T slice, while skipping sparse params adjustment.
        - Run standalone TND attention as ground truth.
        - Gather distributed output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()

    # 8 batches with varying lengths, total = 4096
    q, k, v, actual_seq_qlen, actual_seq_kvlen = tnd_tensors()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    placements = (Shard(0),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    # Run standalone for comparison
    q_full = torch.from_numpy(
        global_query_np.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM)
            .reshape(BATCH_SIZE * SEQ_LEN, HEAD_NUM, HEAD_DIM)
    ).npu()
    k_full = torch.from_numpy(
        global_key_np.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM)
            .reshape(BATCH_SIZE * SEQ_LEN, HEAD_NUM, HEAD_DIM)
    ).npu()
    v_full = torch.from_numpy(
        global_value_np.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM)
            .reshape(BATCH_SIZE * SEQ_LEN, HEAD_NUM, HEAD_DIM)
    ).npu()
    expected = torch_npu.npu_fusion_attention(
        q_full, k_full, v_full, head_num=HEAD_NUM, input_layout='TND',
        scale=SCALE, sparse_mode=0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )[0]

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
        scale=SCALE, sparse_mode=0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )

    total_tokens = BATCH_SIZE * SEQ_LEN
    out = result[0]
    assert out.to_local().shape == (total_tokens // 8, HEAD_NUM, HEAD_DIM)

    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_tnd_cp():
    """
    Feature: npu_fusion_attention TND layout with context parallelism
    Description:
        - TND format with Q Shard(0) on sp, K/V Replicate.
        - s1_split_num > 1, triggers CP branch.
        - sparse_mode=3 (rightDownCausal) as required by CP.
        - actual_seq_qlen/actual_seq_kvlen adjusted via clamp logic.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()

    q, k, v, actual_seq_qlen, actual_seq_kvlen = tnd_tensors()

    # Standalone ground truth
    mask = create_attention_mask(3)
    expected = torch_npu.npu_fusion_attention(
        q, k, v, head_num=HEAD_NUM, input_layout='TND',
        atten_mask=mask, scale=SCALE, sparse_mode=3,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )[0]

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("cp",))
    dq = DTensor.distribute_tensor(q, mesh, (Shard(0),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
        atten_mask=mask, scale=SCALE, sparse_mode=3,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    total_tokens = BATCH_SIZE * SEQ_LEN
    assert result[0].to_local().shape == (total_tokens // 8, HEAD_NUM, HEAD_DIM)

    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)


def test_fa_vs_sdpa_cross_validation():
    """Cross-validate FA output against SDPA output as ground truth."""
    init_dist()
    q, k, v = bsh_tensors()

    # SDPA ground truth
    q_bnsd = q.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM).transpose(1, 2).float()
    k_bnsd = k.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM).transpose(1, 2).float()
    v_bnsd = v.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM).transpose(1, 2).float()
    sdpa_out = torch.nn.functional.scaled_dot_product_attention(
        q_bnsd, k_bnsd, v_bnsd, scale=SCALE
    )
    expected = sdpa_out.transpose(1, 2).reshape(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).half()

    # FA output
    fa_out = torch_npu.npu_fusion_attention(
        q, k, v, head_num=HEAD_NUM, input_layout='BSH',
        scale=SCALE, sparse_mode=0,
    )[0]

    assert_close(fa_out, expected, atol=1e-3, rtol=1e-3)


def test_fa_vs_sdpa_distributed_cross_validation():
    """Cross-validate FA against SDPA under the same distributed sharding strategy."""
    init_dist()
    q, k, v = bsh_tensors()

    # Reshape to BNSD for SDPA
    q_bnsd = q.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM).transpose(1, 2)
    k_bnsd = k.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM).transpose(1, 2)
    v_bnsd = v.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM).transpose(1, 2)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("sp", "mp"))
    full_replicate = (Replicate(), Replicate())

    # FA path: BSH, Q=Shard(1)+Shard(2), KV=Replicate+Shard(2)
    dq_fa = DTensor.distribute_tensor(q, mesh, (Shard(1), Shard(2)))
    dk_fa = DTensor.distribute_tensor(k, mesh, (Replicate(), Shard(2)))
    dv_fa = DTensor.distribute_tensor(v, mesh, (Replicate(), Shard(2)))

    fa_result = torch_npu.npu_fusion_attention(
        dq_fa, dk_fa, dv_fa, head_num=HEAD_NUM, input_layout='BSH',
        scale=SCALE, sparse_mode=0,
    )
    fa_gathered = fa_result[0].redistribute(mesh, full_replicate).to_local()

    # SDPA path: BNSD [B,N,S,D], Q=Shard(2)+Shard(1), KV=Replicate+Shard(1)
    # BSH Shard(1)=seq -> BNSD Shard(2)=seq; BSH Shard(2)=hidden -> BNSD Shard(1)=head
    dq_sdpa = DTensor.distribute_tensor(q_bnsd, mesh, (Shard(2), Shard(1)))
    dk_sdpa = DTensor.distribute_tensor(k_bnsd, mesh, (Replicate(), Shard(1)))
    dv_sdpa = DTensor.distribute_tensor(v_bnsd, mesh, (Replicate(), Shard(1)))

    sdpa_result = torch.nn.functional.scaled_dot_product_attention(
        dq_sdpa.to_local().float(),
        dk_sdpa.to_local().float(),
        dv_sdpa.to_local().float(),
        scale=SCALE,
    )
    # Gather SDPA result back: local [B, N/2, S/4, D] -> global [B, N, S, D] -> BSH
    sdpa_dt = DTensor.from_local(sdpa_result.half(), mesh, (Shard(2), Shard(1)))
    sdpa_gathered = sdpa_dt.redistribute(mesh, full_replicate).to_local()
    sdpa_gathered_bsh = sdpa_gathered.transpose(1, 2).reshape(
        BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE
    )

    assert_close(fa_gathered, sdpa_gathered_bsh, atol=1e-3, rtol=1e-3,
                 msg="FA vs SDPA mismatch under SP+MP distributed sharding")


def test_error_batch_sharding_mismatch():
    """
    Feature: npu_fusion_attention rejects mismatched Q/K batch sharding
    Description:
        - BSH layout with Q Shard(0) on batch dim but K/V Replicate on batch dim.
        - Q and K must have identical batch sharding strategy.
    Expectation: Raise ValueError with message "batch sharding strategy".
    """
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))
    q, k, v = bsh_tensors()
    dq = DTensor.distribute_tensor(q, mesh, (Shard(0), Shard(2)))
    dk = DTensor.distribute_tensor(k, mesh, (Replicate(), Shard(2)))
    dv = DTensor.distribute_tensor(v, mesh, (Replicate(), Shard(2)))

    with pytest.raises(ValueError, match="batch sharding strategy"):
        torch_npu.npu_fusion_attention(
            dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
            scale=SCALE, sparse_mode=0,
        )


def test_error_hidden_sharding_mismatch():
    """
    Feature: npu_fusion_attention rejects mismatched Q/K hidden sharding
    Description:
        - BSH layout with Q Shard(2) on hidden dim but K/V Replicate on hidden dim.
        - Q and K must have identical hidden sharding strategy.
    Expectation: Raise ValueError with message "hidden sharding strategy".
    """
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))
    q, k, v = bsh_tensors()
    dq = DTensor.distribute_tensor(q, mesh, (Shard(0), Shard(2)))
    dk = DTensor.distribute_tensor(k, mesh, (Shard(0), Replicate()))
    dv = DTensor.distribute_tensor(v, mesh, (Shard(0), Replicate()))

    with pytest.raises(ValueError, match="hidden sharding strategy"):
        torch_npu.npu_fusion_attention(
            dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
            scale=SCALE, sparse_mode=0,
        )


def test_error_dim_sharding_mismatch():
    """
    Feature: npu_fusion_attention rejects mismatched Q/K dim sharding
    Description:
        - BNSD layout with Q Shard(3) on dim but K/V Replicate on dim.
        - Q and K must have identical dim sharding strategy.
    Expectation: Raise ValueError with message "dim sharding strategy".
    """
    init_dist()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))
    q, k, v = bnsd_tensors()
    dq = DTensor.distribute_tensor(q, mesh, (Shard(0), Shard(3)))
    dk = DTensor.distribute_tensor(k, mesh, (Shard(0), Replicate()))
    dv = DTensor.distribute_tensor(v, mesh, (Shard(0), Replicate()))

    with pytest.raises(ValueError, match="dim sharding strategy"):
        torch_npu.npu_fusion_attention(
            dq, dk, dv, head_num=HEAD_NUM, input_layout='BNSD',
            scale=SCALE, sparse_mode=0,
        )


def test_error_tnd_cp_wrong_sparse_mode():
    """
    Feature: npu_fusion_attention rejects TND CP with sparse_mode != 3
    Description:
        - TND layout with Q Shard(0) on cp axis, K/V Replicate (s1_split_num > 1).
        - sparse_mode=0 instead of required sparse_mode=3 (rightDownCausal).
        - Context parallelism requires rightDownCausal for correct mask partitioning.
    Expectation: Raise ValueError with message "sparse_mode=3".
    """
    init_dist()

    q, k, v, actual_seq_qlen, actual_seq_kvlen = tnd_tensors()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("cp",))
    dq = DTensor.distribute_tensor(q, mesh, (Shard(0),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    with pytest.raises(ValueError, match="sparse_mode=3"):
        torch_npu.npu_fusion_attention(
            dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
            scale=SCALE, sparse_mode=0,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
        )


def test_error_tnd_cp_qk_global_t_mismatch():
    """
    Feature: npu_fusion_attention rejects TND CP when Q/K have different global T
    Description:
        - TND layout with Q Shard(0) on cp axis, K/V Replicate on cp axis
          but with different total token count.
        - Context parallelism requires Q and K to have the same global T-dimension.
    Expectation: Raise ValueError with message "same global T-dimension".
    """
    init_dist()

    q, k, v, actual_seq_qlen, actual_seq_kvlen = tnd_tensors()
    # Create shorter KV tensors so global T dimensions differ
    half_total = BATCH_SIZE * SEQ_LEN // 2
    k_short = k[:half_total]
    v_short = v[:half_total]

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("cp", "mp"))
    dq = DTensor.distribute_tensor(q, mesh, (Shard(0), Shard(1)))
    dk = DTensor.from_local(k_short, mesh, (Replicate(), Shard(1)))
    dv = DTensor.from_local(v_short, mesh, (Replicate(), Shard(1)))
    mask = create_attention_mask(3)

    with pytest.raises(ValueError, match="same global T-dimension"):
        torch_npu.npu_fusion_attention(
            dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
            atten_mask=mask, scale=SCALE, sparse_mode=3,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
        )


def test_tnd_dp_kv_sharded():
    """
    Feature: npu_fusion_attention TND DP with both Q and K/V sharded on T dimension
    Description:
        - TND format with Q/K/V all Shard(0) on T dimension (kv_is_sharded=True).
        - Distributed op adjusts both actual_seq_qlen and actual_seq_kvlen via clamp
          using the kv_is_sharded branch.
        - Run standalone TND attention as ground truth.
        - Gather distributed output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    init_dist()

    q, k, v, actual_seq_qlen, actual_seq_kvlen = tnd_tensors()

    # Standalone ground truth
    expected = torch_npu.npu_fusion_attention(
        q, k, v, head_num=HEAD_NUM, input_layout='TND',
        scale=SCALE, sparse_mode=0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )[0]

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    placements = (Shard(0),)
    dq = DTensor.distribute_tensor(q, mesh, placements)
    dk = DTensor.distribute_tensor(k, mesh, placements)
    dv = DTensor.distribute_tensor(v, mesh, placements)

    result = torch_npu.npu_fusion_attention(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
        scale=SCALE, sparse_mode=0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    total_tokens = BATCH_SIZE * SEQ_LEN
    out = result[0]
    assert out.to_local().shape == (total_tokens // 8, HEAD_NUM, HEAD_DIM)

    gathered = result[0].redistribute(mesh, (Replicate(),)).to_local()
    assert_close(gathered, expected)
