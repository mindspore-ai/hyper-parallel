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

"""Test MindSpore DTensor with flash_attention_score distributed operator."""

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor, ops
from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.dtensor import distribute_tensor
from hyper_parallel.core.placement_types import Shard, Replicate


def setup_module():
    ms.set_device("Ascend")
    D.init()


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
        return Tensor(np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.bool_))
    if sparse_mode in [2, 3, 4]:
        mask_np = np.triu(np.ones((2048, 2048)), k=1).astype(np.bool_)
        return Tensor(mask_np)
    return None


def run_standalone_bsh(sparse_mode=0, scalar_value=SCALE, keep_prob=1.0,
                       pre_tokens=2147483647, next_tokens=2147483647):
    """Run standalone BSH attention as ground truth."""
    q = Tensor(global_query_np)
    k = Tensor(global_key_np)
    v = Tensor(global_value_np)
    mask = create_attention_mask(sparse_mode)
    result = ops.flash_attention_score(
        q, k, v, head_num=HEAD_NUM, input_layout='BSH',
        attn_mask=mask, scalar_value=scalar_value, keep_prob=keep_prob,
        pre_tokens=pre_tokens, next_tokens=next_tokens,
        sparse_mode=sparse_mode,
    )
    return result


def bsh_tensors():
    """Create BSH format tensors."""
    q = Tensor(global_query_np)
    k = Tensor(global_key_np)
    v = Tensor(global_value_np)
    return q, k, v


def bnsd_tensors():
    """Create BNSD format tensors."""
    def to_bnsd(data):
        x = data.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM)
        return np.ascontiguousarray(np.transpose(x, (0, 2, 1, 3)))
    q = Tensor(to_bnsd(global_query_np))
    k = Tensor(to_bnsd(global_key_np))
    v = Tensor(to_bnsd(global_value_np))
    return q, k, v


def sbh_tensors():
    """Create SBH format tensors."""
    def to_sbh(data):
        return np.ascontiguousarray(np.transpose(data, (1, 0, 2)))
    q = Tensor(to_sbh(global_query_np))
    k = Tensor(to_sbh(global_key_np))
    v = Tensor(to_sbh(global_value_np))
    return q, k, v


def bsnd_tensors():
    """Create BSND format tensors."""
    def to_bsnd(data):
        return np.ascontiguousarray(
            data.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM)
        )
    q = Tensor(to_bsnd(global_query_np))
    k = Tensor(to_bsnd(global_key_np))
    v = Tensor(to_bsnd(global_value_np))
    return q, k, v


def tnd_tensors(num_samples=BATCH_SIZE, tokens_per_sample=SEQ_LEN):
    """Create TND format tensors and cumulative sequence lengths."""
    total = num_samples * tokens_per_sample
    def to_tnd(data):
        return np.ascontiguousarray(
            data.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM)
                .reshape(total, HEAD_NUM, HEAD_DIM)
        )
    q = Tensor(to_tnd(global_query_np))
    k = Tensor(to_tnd(global_key_np))
    v = Tensor(to_tnd(global_value_np))
    actual_seq_qlen = [(i + 1) * tokens_per_sample for i in range(num_samples)]
    actual_seq_kvlen = [(i + 1) * tokens_per_sample for i in range(num_samples)]
    return q, k, v, actual_seq_qlen, actual_seq_kvlen


def scaled_dot_product_attention(q, k, v, scalar_value):
    """Manual SDPA implementation for cross-validation. Input shape: [B, N, S, D]."""
    attn_weights = ops.matmul(q, k.swapaxes(-2, -1)) * scalar_value
    attn_weights = ops.softmax(attn_weights, axis=-1)
    return ops.matmul(attn_weights, v)


def test_bsh_replicate():
    """
    Feature: flash_attention_score BSH layout with fully replicated tensors
    Description:
        - All Q/K/V tensors are Replicate on an 8-device mesh.
        - No actual parallelism; the distributed op should pass through directly.
        - Compare output against standalone single-device execution.
    Expectation: Output matches standalone result within tolerance.
    """
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    dq = DTensor.from_local(q, mesh, (Replicate(),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scalar_value=SCALE, sparse_mode=0,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_bsh_dp():
    """
    Feature: flash_attention_score BSH layout with data parallelism
    Description:
        - Q/K/V are Shard(0) on batch dimension across 8 devices.
        - Each device computes attention on its local batch slice independently.
        - Gather output and compare against standalone full-batch result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scalar_value=SCALE, sparse_mode=0,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_bsh_mp():
    """
    Feature: flash_attention_score BSH layout with head parallelism
    Description:
        - Q/K/V are Shard(2) on hidden dimension across 8 devices.
        - head_num is divided by the split factor; each device handles a subset of heads.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("mp",))
    q, k, v = bsh_tensors()
    placements = (Shard(2),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scalar_value=SCALE, sparse_mode=0,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_bsh_sp():
    """
    Feature: flash_attention_score BSH layout with sequence parallelism
    Description:
        - Q is Shard(1) on seq dimension; K/V are Replicate.
        - Distributed op adjusts sparse parameters for local sequence slice.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bsh_tensors()
    dq = distribute_tensor(q, mesh, (Shard(1),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scalar_value=SCALE, sparse_mode=0,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_bsh_dp_mp_2d():
    """
    Feature: flash_attention_score BSH layout with DP + MP on 2D mesh
    Description:
        - 2D mesh (4, 2) with dp on batch dim and mp on hidden dim.
        - Q/K/V are Shard(0) on dp axis and Shard(2) on mp axis.
        - Combines data parallelism and head parallelism simultaneously.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))
    q, k, v = bsh_tensors()
    placements = (Shard(0), Shard(2))
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scalar_value=SCALE, sparse_mode=0,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_bsh_sp_mp_2d():
    """
    Feature: flash_attention_score BSH layout with SP + MP on 2D mesh
    Description:
        - 2D mesh (4, 2) with sp on seq dim and mp on hidden dim.
        - Q is Shard(1)+Shard(2); K/V are Replicate+Shard(2).
        - Combines sequence parallelism and head parallelism simultaneously.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("sp", "mp"))
    q, k, v = bsh_tensors()
    dq = distribute_tensor(q, mesh, (Shard(1), Shard(2)))
    dk = distribute_tensor(k, mesh, (Replicate(), Shard(2)))
    dv = distribute_tensor(v, mesh, (Replicate(), Shard(2)))

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scalar_value=SCALE, sparse_mode=0,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_bsh_dp_sp_mp_3d():
    """
    Feature: flash_attention_score BSH layout with DP + SP + MP on 3D mesh
    Description:
        - 3D mesh (2, 2, 2) with dp, sp, and mp axes.
        - Q is Shard(0)+Shard(1)+Shard(2); K/V are Shard(0)+Replicate+Shard(2).
        - Exercises all three parallelism strategies simultaneously.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    expected = run_standalone_bsh()

    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "sp", "mp")
    )
    q, k, v = bsh_tensors()
    dq = distribute_tensor(q, mesh, (Shard(0), Shard(1), Shard(2)))
    dk = distribute_tensor(k, mesh, (Shard(0), Replicate(), Shard(2)))
    dv = distribute_tensor(v, mesh, (Shard(0), Replicate(), Shard(2)))

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scalar_value=SCALE, sparse_mode=0,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_bnsd_dp_mp():
    """
    Feature: flash_attention_score BNSD layout with DP + MP on 2D mesh
    Description:
        - BNSD format with 2D mesh (4, 2): dp on batch dim, mp on head dim (dim 1).
        - Q/K/V are Shard(0)+Shard(1).
        - Verify output shape matches expected local dimensions after sharding.
    Expectation: Local output shape is (B/4, N/2, S, D).
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))
    q, k, v = bnsd_tensors()
    placements = (Shard(0), Shard(1))
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BNSD',
        scalar_value=SCALE, sparse_mode=0,
    )
    assert result.to_local().shape == (BATCH_SIZE // 4, HEAD_NUM // 2, SEQ_LEN, HEAD_DIM)


def test_bnsd_sp():
    """
    Feature: flash_attention_score BNSD layout with DP + SP on 2D mesh
    Description:
        - BNSD format with 2D mesh (4, 2): dp on batch dim, sp on seq dim (dim 2).
        - Q is Shard(0)+Shard(2); K/V are Shard(0)+Replicate.
        - Verify output shape matches expected local dimensions after sharding.
    Expectation: Local output shape is (B/4, N, S/2, D).
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "sp"))
    q, k, v = bnsd_tensors()
    dq = distribute_tensor(q, mesh, (Shard(0), Shard(2)))
    dk = distribute_tensor(k, mesh, (Shard(0), Replicate()))
    dv = distribute_tensor(v, mesh, (Shard(0), Replicate()))

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BNSD',
        scalar_value=SCALE, sparse_mode=0,
    )
    assert result.to_local().shape == (BATCH_SIZE // 4, HEAD_NUM, SEQ_LEN // 2, HEAD_DIM)


def test_sbh_dp():
    """
    Feature: flash_attention_score SBH layout with data parallelism
    Description:
        - SBH format with batch dim at index 1, sharded across 8 devices.
        - Verify output shape matches expected local dimensions after batch sharding.
    Expectation: Local output shape is (S, B/8, H).
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = sbh_tensors()
    placements = (Shard(1),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='SBH',
        scalar_value=SCALE, sparse_mode=0,
    )
    assert result.to_local().shape == (SEQ_LEN, BATCH_SIZE // 8, HIDDEN_SIZE)


def test_bsnd_dp_mp():
    """
    Feature: flash_attention_score BSND layout with DP + MP on 2D mesh
    Description:
        - BSND format with 2D mesh (4, 2): dp on batch dim, mp on head dim (dim 2).
        - Q/K/V are Shard(0)+Shard(2).
        - Verify output shape matches expected local dimensions after sharding.
    Expectation: Local output shape is (B/4, S, N/2, D).
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))
    q, k, v = bsnd_tensors()
    placements = (Shard(0), Shard(2))
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSND',
        scalar_value=SCALE, sparse_mode=0,
    )
    assert result.to_local().shape == (BATCH_SIZE // 4, SEQ_LEN, HEAD_NUM // 2, HEAD_DIM)


def test_tnd_dp():
    """
    Feature: flash_attention_score TND layout with data parallelism
    Description:
        - TND format with T dimension sharded across 8 devices (equal-length samples).
        - actual_seq_qlen/actual_seq_kvlen provided as cumulative sums.
        - Distributed op adjusts actual_seq_len via clamp to match local T slice.
        - Verify output shape matches expected local T dimension.
    Expectation: Local output shape is (T/8, N, D).
    """
    q, k, v, actual_seq_qlen, actual_seq_kvlen = tnd_tensors()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
        scalar_value=SCALE, sparse_mode=0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    total_tokens = BATCH_SIZE * SEQ_LEN
    assert result.to_local().shape == (total_tokens // 8, HEAD_NUM, HEAD_DIM)


def test_tnd_mp():
    """
    Feature: flash_attention_score TND layout with head parallelism
    Description:
        - TND format with N (head) dimension sharded across 8 devices.
        - head_num is divided by the split factor.
        - Verify output shape matches expected local head count.
    Expectation: Local output shape is (T, N/8, D).
    """
    q, k, v, actual_seq_qlen, actual_seq_kvlen = tnd_tensors()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("mp",))
    placements = (Shard(1),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
        scalar_value=SCALE, sparse_mode=0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    total_tokens = BATCH_SIZE * SEQ_LEN
    assert result.to_local().shape == (total_tokens, HEAD_NUM // 8, HEAD_DIM)


def test_tnd_dp_mp():
    """
    Feature: flash_attention_score TND layout with DP + MP on 2D mesh
    Description:
        - TND format with 2D mesh (4, 2): dp on T dim, mp on N dim.
        - Q/K/V are Shard(0)+Shard(1).
        - Verify output shape matches expected local dimensions after both splits.
    Expectation: Local output shape is (T/4, N/2, D).
    """
    q, k, v, actual_seq_qlen, actual_seq_kvlen = tnd_tensors()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(4, 2), mesh_dim_names=("dp", "mp"))
    dq = distribute_tensor(q, mesh, (Shard(0), Shard(1)))
    dk = distribute_tensor(k, mesh, (Shard(0), Shard(1)))
    dv = distribute_tensor(v, mesh, (Shard(0), Shard(1)))

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
        scalar_value=SCALE, sparse_mode=0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    total_tokens = BATCH_SIZE * SEQ_LEN
    assert result.to_local().shape == (total_tokens // 4, HEAD_NUM // 2, HEAD_DIM)


def test_sp_sparse_mode_0():
    """
    Feature: flash_attention_score BSH SP with sparse_mode=0 (defaultMask, no mask)
    Description:
        - Q is Shard(1) on seq dim; K/V are Replicate.
        - sparse_mode=0 with default pre/next tokens (no mask provided).
        - Distributed op applies LEFT_UP_TO_LEFT_UP offset to pre/next tokens.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    expected = run_standalone_bsh(sparse_mode=0)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bsh_tensors()
    dq = distribute_tensor(q, mesh, (Shard(1),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scalar_value=SCALE, sparse_mode=0,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_sp_sparse_mode_2():
    """
    Feature: flash_attention_score BSH SP with sparse_mode=2 (leftUpCausal)
    Description:
        - Q is Shard(1) on seq dim; K/V are Replicate.
        - sparse_mode=2 with compressed 2048x2048 causal mask.
        - Distributed op converts to sparse_mode=4 (band) with LEFT_UP_TO_RIGHT_DOWN offset.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    expected = run_standalone_bsh(sparse_mode=2)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bsh_tensors()
    mask = create_attention_mask(2)
    dq = distribute_tensor(q, mesh, (Shard(1),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        attn_mask=mask, scalar_value=SCALE, sparse_mode=2,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_sp_sparse_mode_3():
    """
    Feature: flash_attention_score BSH SP with sparse_mode=3 (rightDownCausal)
    Description:
        - Q is Shard(1) on seq dim; K/V are Replicate.
        - sparse_mode=3 with compressed 2048x2048 causal mask.
        - Distributed op converts to sparse_mode=4 (band) with RIGHT_DOWN_TO_RIGHT_DOWN offset.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    expected = run_standalone_bsh(sparse_mode=3)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bsh_tensors()
    mask = create_attention_mask(3)
    dq = distribute_tensor(q, mesh, (Shard(1),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        attn_mask=mask, scalar_value=SCALE, sparse_mode=3,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_sp_sparse_mode_4():
    """
    Feature: flash_attention_score BSH SP with sparse_mode=4 (band) and custom pre/next tokens
    Description:
        - Q is Shard(1) on seq dim; K/V are Replicate.
        - sparse_mode=4 with pre_tokens=256, next_tokens=256, compressed 2048x2048 mask.
        - Distributed op applies RIGHT_DOWN_TO_RIGHT_DOWN offset to pre/next tokens.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    pre, nxt = 256, 256
    expected = run_standalone_bsh(sparse_mode=4, pre_tokens=pre, next_tokens=nxt)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    q, k, v = bsh_tensors()
    mask = create_attention_mask(4)
    dq = distribute_tensor(q, mesh, (Shard(1),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        attn_mask=mask, scalar_value=SCALE, sparse_mode=4,
        pre_tokens=pre, next_tokens=nxt,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_dp_sparse_mode_1():
    """
    Feature: flash_attention_score BSH DP with sparse_mode=1 (allMask)
    Description:
        - Q/K/V are Shard(0) on batch dim (data parallelism).
        - sparse_mode=1 with full [Sq, Skv] attention mask.
        - DP does not enter sparse params adjustment; mode=1 is passed through directly.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    expected = run_standalone_bsh(sparse_mode=1)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    mask = create_attention_mask(1)
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        attn_mask=mask, scalar_value=SCALE, sparse_mode=1,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_dp_sparse_mode_4():
    """
    Feature: flash_attention_score BSH DP with sparse_mode=4 (band) and custom pre/next tokens
    Description:
        - Q/K/V are Shard(0) on batch dim (data parallelism).
        - sparse_mode=4 with pre_tokens=256, next_tokens=256.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    pre, nxt = 256, 256
    expected = run_standalone_bsh(sparse_mode=4, pre_tokens=pre, next_tokens=nxt)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    mask = create_attention_mask(4)
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        attn_mask=mask, scalar_value=SCALE, sparse_mode=4,
        pre_tokens=pre, next_tokens=nxt,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_bsh_custom_scale():
    """
    Feature: flash_attention_score BSH DP with custom scalar_value
    Description:
        - Use a non-default scalar_value=0.125 instead of 1/sqrt(HEAD_DIM).
        - Q/K/V are Shard(0) on batch dim.
        - Gather output and compare against standalone result with same scalar_value.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    custom_scale = 0.125
    expected = run_standalone_bsh(scalar_value=custom_scale)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scalar_value=custom_scale, sparse_mode=0,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_bsh_dropout():
    """
    Feature: flash_attention_score BSH DP with dropout (keep_prob < 1)
    Description:
        - Q/K/V are Shard(0) on batch dim with keep_prob=0.9.
        - Dropout introduces randomness, so only output shape is verified.
    Expectation: Local output shape is (B/8, S, H). No correctness check due to randomness.
    """
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scalar_value=SCALE, sparse_mode=0, keep_prob=0.9,
    )
    assert result.to_local().shape == (BATCH_SIZE // 8, SEQ_LEN, HIDDEN_SIZE)


def test_bsh_long_sequence_sp():
    """
    Feature: flash_attention_score BSH SP with longer sequence length
    Description:
        - Use SEQ_LEN=2048 (4x default) with 8-way sequence parallelism.
        - Q is Shard(1); K/V are Replicate.
        - Verify output shape matches expected local seq dimension.
    Expectation: Local output shape is (B, S/8, H) where S=2048.
    """
    long_seq = 2048
    q_np = np.random.randn(BATCH_SIZE, long_seq, HIDDEN_SIZE).astype(np.float16)
    k_np = np.random.randn(BATCH_SIZE, long_seq, HIDDEN_SIZE).astype(np.float16)
    v_np = np.random.randn(BATCH_SIZE, long_seq, HIDDEN_SIZE).astype(np.float16)
    q = Tensor(q_np)
    k = Tensor(k_np)
    v = Tensor(v_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    dq = distribute_tensor(q, mesh, (Shard(1),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scalar_value=SCALE, sparse_mode=0,
    )
    assert result.to_local().shape == (BATCH_SIZE, long_seq // 8, HIDDEN_SIZE)


def test_bsh_large_batch_dp():
    """
    Feature: flash_attention_score BSH DP with large batch size
    Description:
        - Use BATCH_SIZE=64 (8x default) with 8-way data parallelism.
        - Verify output shape matches expected local batch dimension.
    Expectation: Local output shape is (64/8, S, H).
    """
    large_batch = 64
    q_np = np.random.randn(large_batch, SEQ_LEN, HIDDEN_SIZE).astype(np.float16)
    q = Tensor(q_np)
    k = Tensor(q_np)
    v = Tensor(q_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scalar_value=SCALE, sparse_mode=0,
    )
    assert result.to_local().shape == (large_batch // 8, SEQ_LEN, HIDDEN_SIZE)


def test_bsh_redistribute_then_attention():
    """
    Feature: flash_attention_score BSH with redistribute before attention
    Description:
        - Start with Replicate tensors, redistribute to Shard(0) on batch dim.
        - Run attention on the redistributed DTensors.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    expected = run_standalone_bsh()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    q, k, v = bsh_tensors()
    dq = DTensor.from_local(q, mesh, (Replicate(),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    dq = dq.redistribute(mesh, (Shard(0),))
    dk = dk.redistribute(mesh, (Shard(0),))
    dv = dv.redistribute(mesh, (Shard(0),))

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        scalar_value=SCALE, sparse_mode=0,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_sp_sparse_mode_2_with_2way_split():
    """
    Feature: flash_attention_score BSH SP sparse_mode=2 with 2-way sequence split
    Description:
        - 2D mesh (2, 4) with sp=2 and dp=4.
        - Q is Shard(1)+Shard(0); K/V are Replicate+Shard(0).
        - Verifies LEFT_UP_TO_RIGHT_DOWN offset calculation with non-8-way split.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    expected = run_standalone_bsh(sparse_mode=2)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("sp", "dp"))
    q, k, v = bsh_tensors()
    dq = distribute_tensor(q, mesh, (Shard(1), Shard(0)))
    dk = distribute_tensor(k, mesh, (Replicate(), Shard(0)))
    dv = distribute_tensor(v, mesh, (Replicate(), Shard(0)))
    mask = create_attention_mask(2)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        attn_mask=mask, scalar_value=SCALE, sparse_mode=2,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_sp_sparse_mode_3_with_2way_split():
    """
    Feature: flash_attention_score BSH SP sparse_mode=3 with 2-way sequence split
    Description:
        - 2D mesh (2, 4) with sp=2 and dp=4.
        - Q is Shard(1)+Shard(0); K/V are Replicate+Shard(0).
        - Verifies RIGHT_DOWN_TO_RIGHT_DOWN offset calculation with non-8-way split.
        - Gather output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    expected = run_standalone_bsh(sparse_mode=3)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("sp", "dp"))
    q, k, v = bsh_tensors()
    dq = distribute_tensor(q, mesh, (Shard(1), Shard(0)))
    dk = distribute_tensor(k, mesh, (Replicate(), Shard(0)))
    dv = distribute_tensor(v, mesh, (Replicate(), Shard(0)))
    mask = create_attention_mask(3)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BSH',
        attn_mask=mask, scalar_value=SCALE, sparse_mode=3,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_bnsd_sp_correctness():
    """
    Feature: flash_attention_score BNSD SP correctness verification
    Description:
        - BNSD format with Q Shard(2) on seq dim; K/V Replicate.
        - Run standalone BNSD attention as ground truth.
        - Gather distributed output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    q_bnsd, k_bnsd, v_bnsd = bnsd_tensors()
    standalone = ops.flash_attention_score(
        q_bnsd, k_bnsd, v_bnsd, head_num=HEAD_NUM, input_layout='BNSD',
        scalar_value=SCALE, sparse_mode=0,
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("sp",))
    dq = distribute_tensor(q_bnsd, mesh, (Shard(2),))
    dk = DTensor.from_local(k_bnsd, mesh, (Replicate(),))
    dv = DTensor.from_local(v_bnsd, mesh, (Replicate(),))

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='BNSD',
        scalar_value=SCALE, sparse_mode=0,
    )
    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), standalone.asnumpy(), 1e-2, 1e-2)


def test_tnd_dp_correctness():
    """
    Feature: flash_attention_score TND DP correctness with actual_seq_len adjustment
    Description:
        - TND format with 8-way DP on T dimension (equal-length samples).
        - Run standalone TND attention as ground truth.
        - Gather distributed output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    q, k, v, actual_seq_qlen, actual_seq_kvlen = tnd_tensors()

    q_full = Tensor(
        global_query_np.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM)
            .reshape(BATCH_SIZE * SEQ_LEN, HEAD_NUM, HEAD_DIM)
    )
    k_full = Tensor(
        global_key_np.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM)
            .reshape(BATCH_SIZE * SEQ_LEN, HEAD_NUM, HEAD_DIM)
    )
    v_full = Tensor(
        global_value_np.reshape(BATCH_SIZE, SEQ_LEN, HEAD_NUM, HEAD_DIM)
            .reshape(BATCH_SIZE * SEQ_LEN, HEAD_NUM, HEAD_DIM)
    )
    expected = ops.flash_attention_score(
        q_full, k_full, v_full, head_num=HEAD_NUM, input_layout='TND',
        scalar_value=SCALE, sparse_mode=0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
        scalar_value=SCALE, sparse_mode=0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )

    total_tokens = BATCH_SIZE * SEQ_LEN
    assert result.to_local().shape == (total_tokens // 8, HEAD_NUM, HEAD_DIM)

    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_tnd_cp():
    """
    Feature: flash_attention_score TND layout with context parallelism
    Description:
        - TND format with Q Shard(0) on cp, K/V Replicate.
        - s1_split_num > 1, triggers CP branch.
        - sparse_mode=3 (rightDownCausal) as required by CP.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    q, k, v, actual_seq_qlen, actual_seq_kvlen = tnd_tensors()

    mask = create_attention_mask(3)
    expected = ops.flash_attention_score(
        q, k, v, head_num=HEAD_NUM, input_layout='TND',
        attn_mask=mask, scalar_value=SCALE, sparse_mode=3,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("cp",))
    dq = distribute_tensor(q, mesh, (Shard(0),))
    dk = DTensor.from_local(k, mesh, (Replicate(),))
    dv = DTensor.from_local(v, mesh, (Replicate(),))

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
        attn_mask=mask, scalar_value=SCALE, sparse_mode=3,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    total_tokens = BATCH_SIZE * SEQ_LEN
    assert result.to_local().shape == (total_tokens // 8, HEAD_NUM, HEAD_DIM)

    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)


def test_tnd_dp_kv_sharded():
    """
    Feature: flash_attention_score TND DP with both Q and K/V sharded on T dimension
    Description:
        - TND format with Q/K/V all Shard(0) on T dimension (kv_is_sharded=True).
        - Run standalone TND attention as ground truth.
        - Gather distributed output and compare against standalone result.
    Expectation: Gathered output matches standalone result within tolerance.
    """
    q, k, v, actual_seq_qlen, actual_seq_kvlen = tnd_tensors()

    expected = ops.flash_attention_score(
        q, k, v, head_num=HEAD_NUM, input_layout='TND',
        scalar_value=SCALE, sparse_mode=0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    placements = (Shard(0),)
    dq = distribute_tensor(q, mesh, placements)
    dk = distribute_tensor(k, mesh, placements)
    dv = distribute_tensor(v, mesh, placements)

    result = ops.flash_attention_score(
        dq, dk, dv, head_num=HEAD_NUM, input_layout='TND',
        scalar_value=SCALE, sparse_mode=0,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    total_tokens = BATCH_SIZE * SEQ_LEN
    assert result.to_local().shape == (total_tokens // 8, HEAD_NUM, HEAD_DIM)

    output = result.full_tensor()
    assert np.allclose(output.asnumpy(), expected.asnumpy(), 1e-2, 1e-2)
