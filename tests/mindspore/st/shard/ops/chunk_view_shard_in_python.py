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
ChunkView operator integration tests.
"""
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor, ops
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate


def setup_module():
    """Setup module - initialize distributed environment."""
    ms.set_device("Ascend")
    D.init()


base_mesh_shape = (2, 2)
base_alias_name = ("dp", "tp")


class ChunkViewNet(ms.nn.Cell):
    """Network for testing ChunkView operator."""
    def __init__(self, chunks, dim=0):
        super().__init__()
        self.chunks = chunks
        self.dim = dim
    def construct(self, x):
        return ops.auto_generate.chunk_view_op(x, self.chunks, self.dim)


def compare_results(standalone_output, parallel_output, rtol=1e-5, atol=1e-8):
    """
    Compare standalone and parallel outputs.
    
    Args:
        standalone_output: Output from standalone execution.
        parallel_output: Output from distributed execution.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
    
    Raises:
        AssertionError: If outputs do not match.
    """
    if isinstance(standalone_output, (list, tuple)):
        assert len(standalone_output) == len(parallel_output), (
            f"Output count mismatch: standalone={len(standalone_output)}, "
            f"parallel={len(parallel_output)}"
        )
        for i, (s_out, p_out) in enumerate(zip(standalone_output, parallel_output)):
            s_np = s_out.asnumpy() if isinstance(s_out, Tensor) else s_out
            p_np = p_out.asnumpy() if isinstance(p_out, Tensor) else p_out
            assert np.allclose(s_np, p_np, rtol=rtol, atol=atol), (
                f"Output {i} mismatch:\nStandalone:\n{s_np}\nParallel:\n{p_np}"
            )
    else:
        s_np = standalone_output.asnumpy() if isinstance(standalone_output, Tensor) else standalone_output
        p_np = parallel_output.asnumpy() if isinstance(parallel_output, Tensor) else parallel_output
        assert np.allclose(s_np, p_np, rtol=rtol, atol=atol), (
            f"Output mismatch:\nStandalone:\n{s_np}\nParallel:\n{p_np}"
        )


def test_chunk_view_data_parallel1():
    """
    Feature: ChunkView operator with data parallel.
    Description: Test chunk_view with data parallel sharding.
    Expectation: Run success and match standalone result.
    """
    ms.set_seed(1)
    np.random.seed(1)
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )
    x_placements = (Shard(0), Replicate())
    d, m = 8, 16
    chunks = 4
    dim = 1
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    standalone_net = ChunkViewNet(chunks, dim)
    standalone_output = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = ChunkViewNet(chunks, dim)
    parallel_output = parallel_net(x_local)
    parallel_output_full = [out.full_tensor() for out in parallel_output]
    compare_results(standalone_output, parallel_output_full)


def test_chunk_view_model_parallel2():
    """
    Feature: ChunkView operator with model parallel.
    Description: Test chunk_view with model parallel sharding.
    Expectation: Run success and match standalone result.
    """
    ms.set_seed(2)
    np.random.seed(2)
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )
    x_placements = (Replicate(), Shard(1))
    d, m = 8, 16
    chunks = 2
    dim = 0
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    standalone_net = ChunkViewNet(chunks, dim)
    standalone_output = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = ChunkViewNet(chunks, dim)
    parallel_output = parallel_net(x_local)
    parallel_output_full = [out.full_tensor() for out in parallel_output]
    compare_results(standalone_output, parallel_output_full)


def test_chunk_view_negative_dim3():
    """
    Feature: ChunkView operator with negative dimension.
    Description: Test chunk_view with negative dimension index.
    Expectation: Run success and match standalone result.
    """
    ms.set_seed(5)
    np.random.seed(5)
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )
    x_placements = (Shard(0), Replicate())
    d, m= 8, 16
    chunks = 2
    dim = -1
    x = Tensor(np.random.randn(d, m).astype(np.float32))
    standalone_net = ChunkViewNet(chunks, dim)
    standalone_output = standalone_net(x)
    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_net = ChunkViewNet(chunks, dim)
    parallel_output = parallel_net(x_local)
    parallel_output_full = [out.full_tensor() for out in parallel_output]
    compare_results(standalone_output, parallel_output_full)
