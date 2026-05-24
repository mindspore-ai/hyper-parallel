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
"""test gatherd shard in python"""

from typing import Any
import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor, ops
from hyper_parallel import init_device_mesh, shard_module
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan


def setup_module() -> None:
    """Initialize the distributed environment for the test module."""
    ms.set_device("Ascend")
    D.init()


class GatherDNet(ms.nn.Cell):
    """GatherD composed of GatherD and ReLUs"""

    def __init__(self, device_mesh=None, relu_strategy=None) -> None:
        """Initialize."""
        super().__init__()
        self.gatherd = ops.GatherD()
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None and device_mesh is not None:
            sharding_plan = ShardingPlan(
                input_plan={"input": relu_strategy},
            )
            shard_module(self.relu, device_mesh=device_mesh, sharding_plan=sharding_plan)

    def construct(self, x: Any, dim: Any, index: Any) -> object:
        """Forward computation."""
        out = self.gatherd(x, dim, index)
        out = self.relu(out)
        out = out + 1
        return out


base_mesh_shape = (2, 2)
base_alias_name = ("dp", "mp")


def test_gatherd_data_parallel_dim0_1():
    """
    Feature: GatherD in python shard.
    Description: Input is sharded on batch, and index is fully replicated for dim=0 gather.
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    n, m, k = 16, 32, 64
    x = Tensor(np.random.randn(n, m, k).astype(np.float32))
    index = Tensor(np.random.randint(0, n, size=(n, m, k)).astype(np.int32))
    dim = 0

    standalone_net = GatherDNet()
    standalone_output = standalone_net(x, dim, index)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    # Match UT data_parallel_dim0: batch-sharded input, replicated index.
    x_placements = (Shard(0), Replicate())
    index_placements = (Replicate(), Replicate())

    x_local = distribute_tensor(x, mesh, x_placements)
    index_local = distribute_tensor(index, mesh, index_placements)

    relu_placements = (Replicate(), Replicate())

    parallel_net = GatherDNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, dim, index_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gatherd_data_parallel_dim1_2():
    """
    Feature: GatherD in python shard.
    Description: Input is sharded on sequence, and index is fully replicated for dim=1 gather.
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    # 3D tensors: [batch, seq, hidden]
    n, m, k = 16, 32, 64
    x = Tensor(np.random.randn(n, m, k).astype(np.float32))
    index = Tensor(np.random.randint(0, m, size=(n, m, k)).astype(np.int32))
    dim = 1

    # Standalone
    standalone_net = GatherDNet()
    standalone_output = standalone_net(x, dim, index)

    # Parallel
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    # Match UT data_parallel_dim1: sequence-sharded input, replicated index.
    x_placements = (Replicate(), Shard(1))
    index_placements = (Replicate(), Replicate())

    x_local = distribute_tensor(x, mesh, x_placements)
    index_local = distribute_tensor(index, mesh, index_placements)

    relu_placements = (Replicate(), Replicate())

    parallel_net = GatherDNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, dim, index_local)

    # Validate
    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gatherd_row_parallel_3():
    """
    Feature: GatherD in python shard.
    Description: Input is batch-sharded and index is sharded on batch for dim=0 gather.
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    # 3D tensors: [batch, seq, hidden]
    n, m, k = 16, 32, 64
    x = Tensor(np.random.randn(n, m, k).astype(np.float32))
    index = Tensor(np.random.randint(0, n, size=(n, m, k)).astype(np.int32))
    dim = 0
    # Standalone
    standalone_net = GatherDNet()
    standalone_output = standalone_net(x, dim, index)

    # Parallel
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    # Match UT row_parallel: input batch-sharded, index sharded on dim0.
    x_placements = (Shard(0), Replicate())
    index_placements = (Replicate(), Shard(0))

    x_local = distribute_tensor(x, mesh, x_placements)
    index_local = distribute_tensor(index, mesh, index_placements)

    relu_placements = (Replicate(), Shard(0))

    parallel_net = GatherDNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, dim, index_local)

    # Validate
    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gatherd_input_multi_shard_4():
    """
    Feature: GatherD in python shard.
    Description: Input is sharded on both batch and sequence, while index matches the input
                 sharding on the non-gather axis and stays replicated on the gather axis.
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    n, m = 16, 32
    x = Tensor(np.random.randn(n, m).astype(np.float32))
    index = Tensor(np.random.randint(0, n, size=(n, m)).astype(np.int32))
    dim = 0

    # Standalone
    standalone_net = GatherDNet()
    standalone_output = standalone_net(x, dim, index)

    # Parallel
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Shard(0), Shard(1))
    index_placements = (Replicate(), Shard(1))

    x_local = distribute_tensor(x, mesh, x_placements)
    index_local = distribute_tensor(index, mesh, index_placements)

    relu_placements = (Replicate(), Shard(1))

    parallel_net = GatherDNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, dim, index_local)

    # Validate
    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gatherd_3d_mesh_input_multi_shard_5():
    """
    Feature: GatherD in python shard.
    Description: Input is replicated on non-dim axes and sharded on the gather axis by tp,
                 while index is sharded on the same gather axis by cp.
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    # 3D tensors: [batch, seq, hidden]
    n, m, k = 16, 32, 64
    x = Tensor(np.random.randn(n, m, k).astype(np.float32))
    index = Tensor(np.random.randint(0, m, size=(n, m, k)).astype(np.int32))
    dim = 1

    standalone_net = GatherDNet()
    standalone_output = standalone_net(x, dim, index)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "cp")
    )

    x_placements = (Replicate(), Shard(1), Replicate())
    index_placements = (Replicate(), Replicate(), Shard(1))

    x_local = distribute_tensor(x, mesh, x_placements)
    index_local = distribute_tensor(index, mesh, index_placements)

    relu_placements = (Replicate(), Replicate(), Shard(1))

    parallel_net = GatherDNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, dim, index_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gatherd_3d_mesh_matched_non_dim_shard_6():
    """
    Feature: GatherD in python shard.
    Description: Input and index match on non-gather axes in a 3D mesh, while the gather axis is
                 sharded on different mesh axes.
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    n, m, k = 16, 32, 64
    x = Tensor(np.random.randn(n, m, k).astype(np.float32))
    index = Tensor(np.random.randint(0, m, size=(n, m, k)).astype(np.int32))
    dim = 1

    standalone_net = GatherDNet()
    standalone_output = standalone_net(x, dim, index)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(2, 2, 2),
        mesh_dim_names=("dp", "tp", "cp")
    )

    x_placements = (Shard(0), Shard(1), Replicate())
    index_placements = (Shard(0), Replicate(), Shard(1))

    x_local = distribute_tensor(x, mesh, x_placements)
    index_local = distribute_tensor(index, mesh, index_placements)

    relu_placements = (Shard(0), Replicate(), Shard(1))

    parallel_net = GatherDNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, dim, index_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


_GATHER_ND_MESH_SHAPE = (2, 2, 2)
_GATHER_ND_ALIAS = ("dp", "cp", "tp")


class GatherNdNet(ms.nn.Cell):
    """GatherNd composed of GatherNd and ReLUs"""

    def __init__(self, device_mesh=None, relu_strategy=None) -> None:
        """Initialize."""
        super().__init__()
        self.gathernd = ops.GatherNd()
        self.relu = ms.nn.ReLU()
        if relu_strategy is not None and device_mesh is not None:
            sharding_plan = ShardingPlan(
                input_plan={"input": relu_strategy},
            )
            shard_module(self.relu, device_mesh=device_mesh, sharding_plan=sharding_plan)

    def construct(self, x: Any, indices: Any) -> object:
        """Forward computation."""
        out = self.gathernd(x, indices)
        out = self.relu(out)
        out = out + 1
        return out


def test_gathernd_partial_model_parallel_1():
    """
    Feature: GatherNd in python shard.
    Description: Test GatherNd with indices sharded by tp (last dim not sharded), params replicated.
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    n, m = 16, 64
    x = Tensor(np.random.randn(n, m).astype(np.float32))
    idx_np = np.stack([np.arange(n, dtype=np.int32), np.ones([n], dtype=np.int32)], axis=1)
    indices = Tensor(idx_np)

    standalone_net = GatherNdNet()
    standalone_output = standalone_net(x, indices)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=_GATHER_ND_MESH_SHAPE,
        mesh_dim_names=_GATHER_ND_ALIAS
    )

    x_placements = (Replicate(), Replicate(), Replicate())
    indices_placements = (Replicate(), Replicate(), Shard(0))

    x_local = distribute_tensor(x, mesh, x_placements)
    indices_local = distribute_tensor(indices, mesh, indices_placements)

    relu_placements = (Replicate(), Replicate(), Shard(0))

    parallel_net = GatherNdNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, indices_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gathernd_partial_data_parallel_2():
    """
    Feature: GatherNd in python shard.
    Description: Test GatherNd with indices sharded by dp (last dim not sharded), params replicated.
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    n, m = 16, 64
    x = Tensor(np.random.randn(n, m).astype(np.float32))
    idx_np = np.stack([np.arange(n, dtype=np.int32), np.ones([n], dtype=np.int32)], axis=1)
    indices = Tensor(idx_np)

    standalone_net = GatherNdNet()
    standalone_output = standalone_net(x, indices)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=_GATHER_ND_MESH_SHAPE,
        mesh_dim_names=_GATHER_ND_ALIAS
    )

    x_placements = (Replicate(), Replicate(), Replicate())
    indices_placements = (Shard(0), Replicate(), Replicate())

    x_local = distribute_tensor(x, mesh, x_placements)
    indices_local = distribute_tensor(indices, mesh, indices_placements)

    relu_placements = (Shard(0), Replicate(), Replicate())

    parallel_net = GatherNdNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, indices_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gathernd_params_plain_tensor():
    """
    Feature: GatherNd in python shard.
    Description: Test GatherNd where params is plain Tensor (not DTensor), indices sharded by tp.
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    n, m = 16, 64
    x = Tensor(np.random.randn(n, m).astype(np.float32))
    idx_np = np.stack([np.arange(n, dtype=np.int32), np.ones([n], dtype=np.int32)], axis=1)
    indices = Tensor(idx_np)

    standalone_net = GatherNdNet()
    standalone_output = standalone_net(x, indices)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=_GATHER_ND_MESH_SHAPE,
        mesh_dim_names=_GATHER_ND_ALIAS
    )

    indices_placements = (Replicate(), Replicate(), Shard(0))
    indices_local = distribute_tensor(indices, mesh, indices_placements)

    relu_placements = (Replicate(), Replicate(), Shard(0))

    parallel_net = GatherNdNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x, indices_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gathernd_k1_trailing_dims_shard_3():
    """
    Feature: GatherNd in python shard.
    Description: Test GatherNd with K=1, params sharded on trailing dims, indices sharded on indices[:-1].
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    n, m, k = 16, 8, 32
    x = Tensor(np.random.randn(n, m, k).astype(np.float32))
    indices = Tensor(np.arange(n, dtype=np.int32).reshape(n, 1))

    standalone_net = GatherNdNet()
    standalone_output = standalone_net(x, indices)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=_GATHER_ND_MESH_SHAPE,
        mesh_dim_names=_GATHER_ND_ALIAS
    )

    x_placements = (Replicate(), Shard(1), Shard(2))
    indices_placements = (Shard(0), Replicate(), Replicate())

    x_local = distribute_tensor(x, mesh, x_placements)
    indices_local = distribute_tensor(indices, mesh, indices_placements)

    relu_placements = (Shard(0), Shard(1), Shard(2))

    parallel_net = GatherNdNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, indices_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gathernd_k2_trailing_dim_shard_4():
    """
    Feature: GatherNd in python shard.
    Description: Test GatherNd with K=2, params sharded on trailing dims, indices sharded on indices[:-1].
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    n, m, k = 16, 8, 32
    x = Tensor(np.random.randn(n, m, k).astype(np.float32))
    idx_np = np.stack([np.arange(n, dtype=np.int32), np.zeros([n], dtype=np.int32)], axis=1)
    indices = Tensor(idx_np)

    standalone_net = GatherNdNet()
    standalone_output = standalone_net(x, indices)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=_GATHER_ND_MESH_SHAPE,
        mesh_dim_names=_GATHER_ND_ALIAS
    )

    x_placements = (Replicate(), Shard(2), Replicate())
    indices_placements = (Replicate(), Replicate(), Shard(0))

    x_local = distribute_tensor(x, mesh, x_placements)
    indices_local = distribute_tensor(indices, mesh, indices_placements)

    relu_placements = (Replicate(), Shard(1), Shard(0))

    parallel_net = GatherNdNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, indices_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_gathernd_k3_no_trailing_dims_5():
    """
    Feature: GatherNd in python shard.
    Description: Test GatherNd with K=3 and input_rank=3, params replicated, indices sharded on indices[:-1].
    Expectation: Run success and match standalone.
    """
    ms.set_seed(1)
    np.random.seed(1)

    n, m, k = 16, 8, 32
    x = Tensor(np.random.randn(n, m, k).astype(np.float32))

    idx0 = np.arange(n, dtype=np.int32)
    idx1 = np.zeros([n], dtype=np.int32)
    idx2 = np.zeros([n], dtype=np.int32)
    indices = Tensor(np.stack([idx0, idx1, idx2], axis=1))

    standalone_net = GatherNdNet()
    standalone_output = standalone_net(x, indices)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=_GATHER_ND_MESH_SHAPE,
        mesh_dim_names=_GATHER_ND_ALIAS
    )

    x_placements = (Replicate(), Replicate(), Replicate())
    indices_placements = (Replicate(), Replicate(), Shard(0))

    x_local = distribute_tensor(x, mesh, x_placements)
    indices_local = distribute_tensor(indices, mesh, indices_placements)

    relu_placements = (Replicate(), Replicate(), Shard(0))

    parallel_net = GatherNdNet(device_mesh=mesh, relu_strategy=relu_placements)
    parallel_output = parallel_net(x_local, indices_local)

    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)
