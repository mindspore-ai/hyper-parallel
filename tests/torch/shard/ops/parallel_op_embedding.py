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
"""test torch dtensor with distributed embedding"""

import numpy as np
import torch
import torch.nn.functional as F
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import _build_layout, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global

np.random.seed(42)
vocab_size = 32
embed_dim = 64
batch_size = 8
seq_len = 16

# Unified random inputs for standalone (single-device) execution
standalone_input_np = np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype(np.int64)
standalone_weight_np = np.random.randn(vocab_size, embed_dim).astype(np.float32)


def test_distributed_embedding_data_parallel():
    """
    Feature: dtensor + F.embedding Data Parallel
    Description:
        - Input sharded on Batch dim, Weight replicated on all dims.
        - Verify numerical equivalence and correct Layout output (inherit batch shard, replicate embed dim).
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()
    standalone_output = F.embedding(standalone_input, standalone_weight)

    # Distributed Setup: Mesh (dp, tp)
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # dp axis is used to shard the batch dimension, tp axis is fully replicated
    x_placements = (Shard(0), Replicate())
    w_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    # Op dispatching will intercept this call
    dist_output = F.embedding(dist_input, dist_weight)

    # Layout Validation
    # Input sharded dim 0 on dp, weight replicated. Output tensor (3D) is sharded dim 0 on dp.
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"DP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical Validation (Gather local results back to global and compare)
    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-4), \
        "DP Embedding output mismatch between standalone and distributed execution"


def test_distributed_embedding_column_parallel():
    """
    Feature: dtensor + F.embedding Column Parallel
    Description:
        - Weight sharded on Embedding dim (tp axis).
        - Output dimension should carry the Shard(2) layout on the tp axis.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()
    standalone_output = F.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # Input: dp shards batch
    x_placements = (Shard(0), Replicate())
    # Weight: tp shards embed dim (dim=1)
    w_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.embedding(dist_input, dist_weight)

    # Layout Validation: Output is 3D, dp shards dim 0 (batch), tp shards dim 2 (embed_dim)
    expected_layout = _build_layout(mesh, (Shard(0), Shard(2)), 3)
    assert dist_output.layout == expected_layout, \
        f"CP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical Validation
    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-4), \
        "CP Embedding output mismatch between standalone and distributed execution"


def test_distributed_embedding_row_parallel():
    """
    Feature: dtensor + F.embedding Row Parallel
    Description:
        - Weight sharded on Vocab dim (tp axis).
        - Output layout should generate a Partial('sum') state on tp axis.
        - Verify row masking implementation correctly sums to match standalone.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()
    standalone_output = F.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # Input: dp shards batch
    x_placements = (Shard(0), Replicate())
    # Weight: tp shards vocab dim (dim=0)
    w_placements = (Replicate(), Shard(0))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.embedding(dist_input, dist_weight)

    # Layout Validation: Output dims are not physically sharded by tp, but tp holds a Partial('sum') state
    expected_layout = _build_layout(mesh, (Shard(0), Partial('sum')), 3)
    assert dist_output.layout == expected_layout, \
        f"RP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical Validation
    # Row Parallel generates Partial status, which needs reduction (AllReduce) before gathering to global
    dist_output_reduced = dist_output.reduce_partial()
    gathered_output = local_to_global(dist_output_reduced)

    assert torch.allclose(standalone_output, gathered_output, atol=1e-4), \
        "RP Embedding output mismatch between standalone and distributed execution"


def test_distributed_embedding_dp_cp():
    """
    Feature: dtensor + F.embedding Data Parallel and Column Parallel (2D Parallelism)
    Description:
        - Input sharded on Batch dim via dp axis.
        - Weight sharded on Embedding dim via tp axis.
        - Output layout should be sharded on Batch dim and Embedding dim.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()
    standalone_output = F.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # dp shards batch (dim 0)
    x_placements = (Shard(0), Replicate())
    # tp shards embed dim (dim 1)
    w_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.embedding(dist_input, dist_weight)

    # Layout Validation: Output is 3D, dp shards dim 0 (batch), tp shards dim 2 (embed_dim)
    expected_layout = _build_layout(mesh, (Shard(0), Shard(2)), 3)
    assert dist_output.layout == expected_layout, \
        f"DP+CP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical Validation
    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-4), \
        "DP+CP Embedding output mismatch between standalone and distributed execution"


def test_distributed_embedding_dp_rp():
    """
    Feature: dtensor + F.embedding Data Parallel and Row Parallel (2D Parallelism)
    Description:
        - Input sharded on Batch dim via dp axis.
        - Weight sharded on Vocab dim via tp axis.
        - Output layout should be sharded on Batch dim and contain Partial('sum') on tp axis.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()
    standalone_output = F.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # dp shards batch (dim 0)
    x_placements = (Shard(0), Replicate())
    # tp shards vocab dim (dim 0)
    w_placements = (Replicate(), Shard(0))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.embedding(dist_input, dist_weight)

    # Layout Validation: Output is 3D, dp shards dim 0, tp carries Partial('sum')
    expected_layout = _build_layout(mesh, (Shard(0), Partial('sum')), 3)
    assert dist_output.layout == expected_layout, \
        f"DP+RP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical Validation (Requires reducing partial sum across tp axis)
    dist_output_reduced = dist_output.reduce_partial()
    gathered_output = local_to_global(dist_output_reduced)

    assert torch.allclose(standalone_output, gathered_output, atol=1e-4), \
        "DP+RP Embedding output mismatch between standalone and distributed execution"


def test_distributed_embedding_sequence_parallel():
    """
    Feature: dtensor + F.embedding Sequence Parallel
    Description:
        - Input sharded on Sequence dim via dp axis.
        - Weight replicated on all dims.
        - Output should be sharded only on Sequence dim.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()
    standalone_output = F.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # dp shards sequence (dim 1)
    x_placements = (Shard(1), Replicate())
    # Weight fully replicated
    w_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.embedding(dist_input, dist_weight)

    # Layout Validation: Output is 3D, dp shards dim 1 (seq_len)
    expected_layout = _build_layout(mesh, (Shard(1), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"SP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical Validation
    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-4), \
        "SP Embedding output mismatch between standalone and distributed execution"


def test_distributed_embedding_sp_cp():
    """
    Feature: dtensor + F.embedding Sequence Parallel and Column Parallel (2D Parallelism)
    Description:
        - Input sharded on Sequence dim via dp axis.
        - Weight sharded on Embedding dim via tp axis.
        - Output should be sharded on both Sequence dim and Embedding dim.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()
    standalone_output = F.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # dp shards sequence (dim 1)
    x_placements = (Shard(1), Replicate())
    # tp shards embed dim (dim 1)
    w_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.embedding(dist_input, dist_weight)

    # Layout Validation: Output is 3D, dp shards dim 1 (seq_len), tp shards dim 2 (embed_dim)
    expected_layout = _build_layout(mesh, (Shard(1), Shard(2)), 3)
    assert dist_output.layout == expected_layout, \
        f"SP+CP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical Validation
    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-4), \
        "SP+CP Embedding output mismatch between standalone and distributed execution"


def test_distributed_embedding_sp_rp():
    """
    Feature: dtensor + F.embedding Sequence Parallel and Row Parallel (2D Parallelism)
    Description:
        - Input sharded on Sequence dim via dp axis.
        - Weight sharded on Vocab dim via tp axis.
        - Output layout should be sharded on Sequence dim and contain Partial('sum') on tp axis.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()
    standalone_output = F.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # dp shards sequence (dim 1)
    x_placements = (Shard(1), Replicate())
    # tp shards vocab dim (dim 0)
    w_placements = (Replicate(), Shard(0))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.embedding(dist_input, dist_weight)

    # Layout Validation: Output is 3D, dp shards dim 1, tp carries Partial('sum')
    expected_layout = _build_layout(mesh, (Shard(1), Partial('sum')), 3)
    assert dist_output.layout == expected_layout, \
        f"SP+RP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical Validation (Requires reducing partial sum across tp axis)
    dist_output_reduced = dist_output.reduce_partial()
    gathered_output = local_to_global(dist_output_reduced)

    assert torch.allclose(standalone_output, gathered_output, atol=1e-4), \
        "SP+RP Embedding output mismatch between standalone and distributed execution"

def test_distributed_embedding_weight_2d_sharding():
    """
    Feature: dtensor + F.embedding Weight 2D Sharding
    Description:
        - Input is fully replicated.
        - Weight is sharded on both Vocab dim (dp axis) and Embed dim (tp axis) simultaneously.
        - Output should inherit Embed dim sharding on tp axis, and carry Partial('sum') on dp axis.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()
    standalone_output = F.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # Input is replicated everywhere to isolate weight sharding effects
    x_placements = (Replicate(), Replicate())
    # Weight: dp shards vocab dim (dim 0), tp shards embed dim (dim 1)
    w_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.embedding(dist_input, dist_weight)

    # Layout Validation: Output is 3D.
    # dp (dim 0 of mesh) has Partial('sum') due to vocab sharding on dp.
    # tp (dim 1 of mesh) shards output's embed_dim (dim 2) due to weight embed sharding on tp.
    expected_layout = _build_layout(mesh, (Partial('sum'), Shard(2)), 3)
    assert dist_output.layout == expected_layout, \
        f"Weight 2D Sharding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical Validation (Requires reducing partial sum across dp axis)
    dist_output_reduced = dist_output.reduce_partial()
    gathered_output = local_to_global(dist_output_reduced)

    assert torch.allclose(standalone_output, gathered_output, atol=1e-4), \
        "Weight 2D Sharding output mismatch between standalone and distributed execution"


def test_distributed_embedding_weight_row_parallel_only():
    """
    Feature: dtensor + F.embedding Weight Row Parallel Only
    Description:
        - Input is fully replicated.
        - Weight is sharded on Vocab dim (tp axis) only.
        - Output should be replicated on physical dimensions but contain Partial('sum') on tp axis.
    Expectation: Success with correct layout and numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()
    standalone_output = F.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    # Input is replicated everywhere
    x_placements = (Replicate(), Replicate())
    # dp replicates, tp shards vocab dim (dim 0)
    w_placements = (Replicate(), Shard(0))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.embedding(dist_input, dist_weight)

    # Layout Validation: Output is 3D, tp carries Partial('sum'), dp replicates
    expected_layout = _build_layout(mesh, (Replicate(), Partial('sum')), 3)
    assert dist_output.layout == expected_layout, \
        f"Weight RP Only layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical Validation
    dist_output_reduced = dist_output.reduce_partial()
    gathered_output = local_to_global(dist_output_reduced)

    assert torch.allclose(standalone_output, gathered_output, atol=1e-4), \
        "Weight RP Only output mismatch between standalone and distributed execution"
def test_distributed_embedding_cp_padding_and_scale():
    """
    Feature: dtensor + F.embedding Column Parallel with padding_idx & scale_grad_by_freq
    Description:
        - Verify native compatibility for padding_idx and scale_grad_by_freq in Column-Sharding.
    Expectation: Success with correct numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()

    padding_idx = 2
    standalone_output = F.embedding(
        standalone_input, standalone_weight,
        padding_idx=padding_idx, scale_grad_by_freq=True
    )


    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    x_placements = (Shard(0), Replicate())
    w_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    # Pass via kwargs
    dist_output = F.embedding(
        dist_input, dist_weight,
        padding_idx=padding_idx, scale_grad_by_freq=True
    )

    gathered_output = local_to_global(dist_output)
    assert torch.allclose(standalone_output, gathered_output, atol=1e-4), \
        "CP Embedding with kwargs output mismatch between standalone and distributed execution"


def test_distributed_embedding_cp_max_norm_error():
    """
    Feature: dtensor + F.embedding Column Parallel with max_norm
    Description:
        - Verify max_norm raises ValueError in Column-Sharding since local norm != global norm.
    Expectation: Raises ValueError.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    x_placements = (Shard(0), Replicate())
    w_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    try:
        F.embedding(dist_input, dist_weight, max_norm=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for max_norm in Column Parallel")

def test_distributed_embedding_rp_padding():
    """
    Feature: dtensor + F.embedding Row Parallel with padding_idx
    Description:
        - Verify padding_idx is correctly mapped to local relative index or ignored in Row-Sharding.
    Expectation: Success with correct numerical equivalence.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()

    # vocab_size is 32, tp split by 4 parts (8 tokens per partition).
    # padding_idx = 10 -> Falls inside rank 1 (8-15) and correctly offset locally.
    padding_idx = 10
    standalone_output = F.embedding(standalone_input, standalone_weight, padding_idx=padding_idx)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    x_placements = (Shard(0), Replicate())
    w_placements = (Replicate(), Shard(0))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = F.embedding(dist_input, dist_weight, padding_idx=padding_idx)

    dist_output_reduced = dist_output.reduce_partial()
    gathered_output = local_to_global(dist_output_reduced)

    assert torch.allclose(standalone_output, gathered_output, atol=1e-4), \
        "RP Embedding with padding_idx output mismatch between standalone and distributed execution"


def test_distributed_embedding_rp_kwargs_error():
    """
    Feature: dtensor + F.embedding Row Parallel with max_norm / scale_grad_by_freq
    Description:
        - Verify max_norm and scale_grad_by_freq raise ValueError in Row-Sharding
          due to static index masking conflicts.
    Expectation: Raises ValueError for both parameters.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))

    x_placements = (Shard(0), Replicate())
    w_placements = (Replicate(), Shard(0))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    try:
        F.embedding(dist_input, dist_weight, max_norm=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for max_norm in Row Parallel")

    try:
        F.embedding(dist_input, dist_weight, scale_grad_by_freq=True)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for scale_grad_by_freq in Row Parallel")

def test_distributed_embedding_args_positional():
    """
    Feature: dtensor + F.embedding Positional Args
    Description:
        - Verify padding_idx, max_norm, scale_grad_by_freq as positional arguments.
        - Ensure args list manipulation in get_expand_impl covers padding_idx correctly.
    Expectation: Success with correct numerical equivalence or expected exceptions.
    """
    init_dist()

    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("dp", "tp"))
    x_placements = (Shard(0), Replicate())

    # 1. CP with padding_idx and scale_grad_by_freq as positional
    w_placements_cp = (Replicate(), Shard(1))
    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight_cp = distribute_tensor(standalone_weight, mesh, w_placements_cp)

    # Signature: F.embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq)
    standalone_output_cp = F.embedding(standalone_input, standalone_weight, 2, None, 2.0, True)
    dist_output_cp = F.embedding(dist_input, dist_weight_cp, 2, None, 2.0, True)

    gathered_output_cp = local_to_global(dist_output_cp)
    assert torch.allclose(standalone_output_cp, gathered_output_cp, atol=1e-4), \
        "Positional Args CP output mismatch"

    # 2. RP with padding_idx as positional
    w_placements_rp = (Replicate(), Shard(0))
    dist_weight_rp = distribute_tensor(standalone_weight, mesh, w_placements_rp)

    standalone_output_rp = F.embedding(standalone_input, standalone_weight, 10, None, 2.0, False)
    dist_output_rp = F.embedding(dist_input, dist_weight_rp, 10, None, 2.0, False)

    dist_output_rp_reduced = dist_output_rp.reduce_partial()
    gathered_output_rp = local_to_global(dist_output_rp_reduced)
    assert torch.allclose(standalone_output_rp, gathered_output_rp, atol=1e-4), \
        "Positional Args RP padding_idx output mismatch"

    # 3. RP with max_norm positional (should raise error)
    try:
        F.embedding(dist_input, dist_weight_rp, None, 1.0, 2.0, False)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for RP positional max_norm")

    # 4. RP with scale_grad_by_freq positional (should raise error)
    try:
        F.embedding(dist_input, dist_weight_rp, None, None, 2.0, True)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for RP positional scale_grad_by_freq")
