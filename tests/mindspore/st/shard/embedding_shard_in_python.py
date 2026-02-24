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
"""
test embedding shard in python
"""

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor, _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial


def setup_module():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("Ascend")
    D.init()


# Global Configurations
np.random.seed(42)
vocab_size = 32
embed_dim = 64
batch_size = 8
seq_len = 16

base_mesh_shape = (2, 4)
base_alias_name = ("dp", "tp")

# Unified random inputs for standalone (single-device) execution
standalone_input_np = np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype(np.int64)
standalone_weight_np = np.random.randn(vocab_size, embed_dim).astype(np.float32)


def test_distributed_embedding_data_parallel():
    """
    Feature: dtensor + mint.nn.functional.embedding Data Parallel
    Description:
        - Input sharded on Batch dim, Weight replicated on all dims.
        - Verify numerical equivalence and correct Layout output (inherit batch shard, replicate embed dim).
    Expectation: Success with correct layout and numerical equivalence.
    """
    ms.set_seed(42)

    standalone_input = Tensor(standalone_input_np)
    standalone_weight = Tensor(standalone_weight_np)
    standalone_output = ms.mint.nn.functional.embedding(standalone_input, standalone_weight)

    # Distributed Setup: Mesh (dp, tp)
    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)

    # dp axis is used to shard the batch dimension, tp axis is fully replicated
    x_placements = (Shard(0), Replicate())
    w_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    # Op dispatching will intercept this call
    dist_output = ms.mint.nn.functional.embedding(dist_input, dist_weight)

    # Layout Validation
    expected_layout = _build_layout(mesh, (Shard(0), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"DP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical Validation (Gather local results back to global and compare)
    gathered_output = dist_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), gathered_output.asnumpy(), atol=1e-3, rtol=1e-3), \
        "DP Embedding output mismatch between standalone and distributed execution"


def test_distributed_embedding_column_parallel():
    """
    Feature: dtensor + mint.nn.functional.embedding Column Parallel
    Description:
        - Weight sharded on Embedding dim (tp axis).
        - Output dimension should carry the Shard(2) layout on the tp axis.
    Expectation: Success with correct layout and numerical equivalence.
    """
    ms.set_seed(42)

    standalone_input = Tensor(standalone_input_np)
    standalone_weight = Tensor(standalone_weight_np)
    standalone_output = ms.mint.nn.functional.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)

    # Input: dp shards batch
    x_placements = (Shard(0), Replicate())
    # Weight: tp shards embed dim (dim=1)
    w_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = ms.mint.nn.functional.embedding(dist_input, dist_weight)

    # Layout Validation: Output is 3D, dp shards dim 0 (batch), tp shards dim 2 (embed_dim)
    expected_layout = _build_layout(mesh, (Shard(0), Shard(2)), 3)
    assert dist_output.layout == expected_layout, \
        f"CP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical Validation
    gathered_output = dist_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), gathered_output.asnumpy(), atol=1e-3, rtol=1e-3), \
        "CP Embedding output mismatch between standalone and distributed execution"


def test_distributed_embedding_row_parallel():
    """
    Feature: dtensor + mint.nn.functional.embedding Row Parallel
    Description:
        - Weight sharded on Vocab dim (tp axis).
        - Output layout should generate a Partial('sum') state on tp axis.
        - Verify row masking implementation correctly sums to match standalone.
    Expectation: Success with correct layout and numerical equivalence.
    """
    ms.set_seed(42)

    standalone_input = Tensor(standalone_input_np)
    standalone_weight = Tensor(standalone_weight_np)
    standalone_output = ms.mint.nn.functional.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)

    # Input: dp shards batch
    x_placements = (Shard(0), Replicate())
    # Weight: tp shards vocab dim (dim=0)
    w_placements = (Replicate(), Shard(0))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = ms.mint.nn.functional.embedding(dist_input, dist_weight)

    # Layout Validation: Output dims are not physically sharded by tp, but tp holds a Partial('sum') state
    expected_layout = _build_layout(mesh, (Shard(0), Partial('sum')), 3)
    assert dist_output.layout == expected_layout, \
        f"RP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    # Numerical Validation: requires reduction before gathering to global
    dist_output_reduced = dist_output.reduce_partial()
    gathered_output = dist_output_reduced.full_tensor()

    assert np.allclose(standalone_output.asnumpy(), gathered_output.asnumpy(), atol=1e-3, rtol=1e-3), \
        "RP Embedding output mismatch between standalone and distributed execution"


def test_distributed_embedding_dp_cp():
    """
    Feature: dtensor + mint.nn.functional.embedding Data Parallel and Column Parallel
    Description:
        - Input sharded on Batch dim via dp axis.
        - Weight sharded on Embedding dim via tp axis.
    Expectation: Success with correct layout and numerical equivalence.
    """
    ms.set_seed(42)

    standalone_input = Tensor(standalone_input_np)
    standalone_weight = Tensor(standalone_weight_np)
    standalone_output = ms.mint.nn.functional.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)

    x_placements = (Shard(0), Replicate())
    w_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = ms.mint.nn.functional.embedding(dist_input, dist_weight)

    expected_layout = _build_layout(mesh, (Shard(0), Shard(2)), 3)
    assert dist_output.layout == expected_layout, \
        f"DP+CP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = dist_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), gathered_output.asnumpy(), atol=1e-3, rtol=1e-3), \
        "DP+CP Embedding output mismatch"


def test_distributed_embedding_dp_rp():
    """
    Feature: dtensor + mint.nn.functional.embedding Data Parallel and Row Parallel
    Description:
        - Input sharded on Batch dim via dp axis.
        - Weight sharded on Vocab dim via tp axis.
    Expectation: Success with correct layout and numerical equivalence.
    """
    ms.set_seed(42)

    standalone_input = Tensor(standalone_input_np)
    standalone_weight = Tensor(standalone_weight_np)
    standalone_output = ms.mint.nn.functional.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)

    x_placements = (Shard(0), Replicate())
    w_placements = (Replicate(), Shard(0))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = ms.mint.nn.functional.embedding(dist_input, dist_weight)

    expected_layout = _build_layout(mesh, (Shard(0), Partial('sum')), 3)
    assert dist_output.layout == expected_layout, \
        f"DP+RP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    dist_output_reduced = dist_output.reduce_partial()
    gathered_output = dist_output_reduced.full_tensor()

    assert np.allclose(standalone_output.asnumpy(), gathered_output.asnumpy(), atol=1e-3, rtol=1e-3), \
        "DP+RP Embedding output mismatch"


def test_distributed_embedding_sequence_parallel():
    """
    Feature: dtensor + mint.nn.functional.embedding Sequence Parallel
    Description:
        - Input sharded on Sequence dim via dp axis.
        - Weight replicated on all dims.
    Expectation: Success with correct layout and numerical equivalence.
    """
    ms.set_seed(42)

    standalone_input = Tensor(standalone_input_np)
    standalone_weight = Tensor(standalone_weight_np)
    standalone_output = ms.mint.nn.functional.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)

    x_placements = (Shard(1), Replicate())
    w_placements = (Replicate(), Replicate())

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = ms.mint.nn.functional.embedding(dist_input, dist_weight)

    expected_layout = _build_layout(mesh, (Shard(1), Replicate()), 3)
    assert dist_output.layout == expected_layout, \
        f"SP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = dist_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), gathered_output.asnumpy(), atol=1e-3, rtol=1e-3), \
        "SP Embedding output mismatch"


def test_distributed_embedding_sp_cp():
    """
    Feature: dtensor + mint.nn.functional.embedding Sequence Parallel and Column Parallel
    Description:
        - Input sharded on Sequence dim via dp axis.
        - Weight sharded on Embedding dim via tp axis.
    Expectation: Success with correct layout and numerical equivalence.
    """
    ms.set_seed(42)

    standalone_input = Tensor(standalone_input_np)
    standalone_weight = Tensor(standalone_weight_np)
    standalone_output = ms.mint.nn.functional.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)

    x_placements = (Shard(1), Replicate())
    w_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = ms.mint.nn.functional.embedding(dist_input, dist_weight)

    expected_layout = _build_layout(mesh, (Shard(1), Shard(2)), 3)
    assert dist_output.layout == expected_layout, \
        f"SP+CP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    gathered_output = dist_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), gathered_output.asnumpy(), atol=1e-3, rtol=1e-3), \
        "SP+CP Embedding output mismatch"


def test_distributed_embedding_sp_rp():
    """
    Feature: dtensor + mint.nn.functional.embedding Sequence Parallel and Row Parallel
    Description:
        - Input sharded on Sequence dim via dp axis.
        - Weight sharded on Vocab dim via tp axis.
    Expectation: Success with correct layout and numerical equivalence.
    """
    ms.set_seed(42)

    standalone_input = Tensor(standalone_input_np)
    standalone_weight = Tensor(standalone_weight_np)
    standalone_output = ms.mint.nn.functional.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)

    x_placements = (Shard(1), Replicate())
    w_placements = (Replicate(), Shard(0))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = ms.mint.nn.functional.embedding(dist_input, dist_weight)

    expected_layout = _build_layout(mesh, (Shard(1), Partial('sum')), 3)
    assert dist_output.layout == expected_layout, \
        f"SP+RP Embedding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    dist_output_reduced = dist_output.reduce_partial()
    gathered_output = dist_output_reduced.full_tensor()

    assert np.allclose(standalone_output.asnumpy(), gathered_output.asnumpy(), atol=1e-3, rtol=1e-3), \
        "SP+RP Embedding output mismatch"


def test_distributed_embedding_weight_2d_sharding():
    """
    Feature: dtensor + mint.nn.functional.embedding Weight 2D Sharding
    Description:
        - Input is fully replicated.
        - Weight is sharded on both Vocab dim (dp axis) and Embed dim (tp axis) simultaneously.
    Expectation: Success with correct layout and numerical equivalence.
    """
    ms.set_seed(42)

    standalone_input = Tensor(standalone_input_np)
    standalone_weight = Tensor(standalone_weight_np)
    standalone_output = ms.mint.nn.functional.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)

    x_placements = (Replicate(), Replicate())
    w_placements = (Shard(0), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = ms.mint.nn.functional.embedding(dist_input, dist_weight)

    expected_layout = _build_layout(mesh, (Partial('sum'), Shard(2)), 3)
    assert dist_output.layout == expected_layout, \
        f"Weight 2D Sharding layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    dist_output_reduced = dist_output.reduce_partial()
    gathered_output = dist_output_reduced.full_tensor()

    assert np.allclose(standalone_output.asnumpy(), gathered_output.asnumpy(), atol=1e-3, rtol=1e-3), \
        "Weight 2D Sharding output mismatch"


def test_distributed_embedding_weight_row_parallel_only():
    """
    Feature: dtensor + mint.nn.functional.embedding Weight Row Parallel Only
    Description:
        - Input is fully replicated.
        - Weight is sharded on Vocab dim (tp axis) only.
    Expectation: Success with correct layout and numerical equivalence.
    """
    ms.set_seed(42)

    standalone_input = Tensor(standalone_input_np)
    standalone_weight = Tensor(standalone_weight_np)
    standalone_output = ms.mint.nn.functional.embedding(standalone_input, standalone_weight)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)

    x_placements = (Replicate(), Replicate())
    w_placements = (Replicate(), Shard(0))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = ms.mint.nn.functional.embedding(dist_input, dist_weight)

    expected_layout = _build_layout(mesh, (Replicate(), Partial('sum')), 3)
    assert dist_output.layout == expected_layout, \
        f"Weight RP Only layout mismatch: expected {expected_layout}, got {dist_output.layout}"

    dist_output_reduced = dist_output.reduce_partial()
    gathered_output = dist_output_reduced.full_tensor()

    assert np.allclose(standalone_output.asnumpy(), gathered_output.asnumpy(), atol=1e-3, rtol=1e-3), \
        "Weight RP Only output mismatch"


def test_distributed_embedding_cp_padding_and_scale():
    """
    Feature: dtensor + mint.nn.functional.embedding Column Parallel with padding_idx & scale_grad_by_freq
    Description:
        - Verify native compatibility for padding_idx and scale_grad_by_freq in Column-Sharding.
    Expectation: Success with correct numerical equivalence.
    """
    ms.set_seed(42)

    standalone_input = Tensor(standalone_input_np)
    standalone_weight = Tensor(standalone_weight_np)

    padding_idx = 2
    standalone_output = ms.mint.nn.functional.embedding(
        standalone_input, standalone_weight,
        padding_idx=padding_idx, scale_grad_by_freq=True
    )

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)

    x_placements = (Shard(0), Replicate())
    w_placements = (Replicate(), Shard(1))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    # Pass via kwargs
    dist_output = ms.mint.nn.functional.embedding(
        dist_input, dist_weight,
        padding_idx=padding_idx, scale_grad_by_freq=True
    )

    gathered_output = dist_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), gathered_output.asnumpy(), atol=1e-3, rtol=1e-3), \
        "CP Embedding with kwargs output mismatch"


def test_distributed_embedding_rp_padding():
    """
    Feature: dtensor + mint.nn.functional.embedding Row Parallel with padding_idx
    Description:
        - Verify padding_idx is correctly mapped to local relative index or ignored in Row-Sharding.
    Expectation: Success with correct numerical equivalence.
    """
    ms.set_seed(42)

    standalone_input = Tensor(standalone_input_np)
    standalone_weight = Tensor(standalone_weight_np)

    padding_idx = 10
    standalone_output = ms.mint.nn.functional.embedding(standalone_input, standalone_weight, padding_idx=padding_idx)

    mesh = init_device_mesh(device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name)

    x_placements = (Shard(0), Replicate())
    w_placements = (Replicate(), Shard(0))

    dist_input = distribute_tensor(standalone_input, mesh, x_placements)
    dist_weight = distribute_tensor(standalone_weight, mesh, w_placements)

    dist_output = ms.mint.nn.functional.embedding(dist_input, dist_weight, padding_idx=padding_idx)

    dist_output_reduced = dist_output.reduce_partial()
    gathered_output = dist_output_reduced.full_tensor()

    assert np.allclose(standalone_output.asnumpy(), gathered_output.asnumpy(), atol=1e-3, rtol=1e-3), \
        "RP Embedding with padding_idx output mismatch"
