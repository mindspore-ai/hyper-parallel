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
Integration tests for HistcExt distributed operator.
"""

import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate


def setup_module():
    """Setup module - initialize MindSpore and distributed environment."""
    ms.set_device("Ascend")
    D.init()


base_mesh_shape = (2, 2)
base_alias_name = ("dp", "tp")


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
    if isinstance(standalone_output, Tensor):
        standalone_output = standalone_output.asnumpy()
    if isinstance(parallel_output, Tensor):
        parallel_output = parallel_output.asnumpy()

    assert np.allclose(standalone_output, parallel_output, rtol=rtol, atol=atol), (
        f"Standalone output:\n{standalone_output}\n"
        f"Parallel output:\n{parallel_output}"
    )


def test_histc_ext_data_parallel1():
    """
    Feature: HistcExt operator with data parallel.
    Description: Test HistcExt with input sharded on batch dimension.
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
    x_np = np.random.randn(d, m).astype(np.float32)
    x = Tensor(x_np)

    bins = 50
    min_val = -3.0
    max_val = 3.0

    standalone_output = ms.mint.histc(x, bins=bins, min=min_val, max=max_val)

    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_output = ms.mint.histc(x_local, bins=bins, min=min_val, max=max_val)
    parallel_output = parallel_output.full_tensor()

    compare_results(standalone_output, parallel_output)


def test_histc_ext_model_parallel2():
    """
    Feature: HistcExt operator with model parallel.
    Description: Test HistcExt with input sharded on feature dimension.
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
    x_np = np.random.randn(d, m).astype(np.float32)
    x = Tensor(x_np)

    bins = 100
    min_val = 0.0
    max_val = 5.0

    standalone_output = ms.mint.histc(x, bins=bins, min=min_val, max=max_val)

    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_output = ms.mint.histc(x_local, bins=bins, min=min_val, max=max_val)
    parallel_output = parallel_output.full_tensor()

    compare_results(standalone_output, parallel_output)


def test_histc_ext_hybrid_parallel3():
    """
    Feature: HistcExt operator with hybrid parallel.
    Description: Test HistcExt with input sharded on multiple dimensions.
    Expectation: Run success and match standalone result.
    """
    ms.set_seed(3)
    np.random.seed(3)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Shard(0), Shard(1))

    d, m = 8, 16
    x_np = np.random.randn(d, m).astype(np.float32)
    x = Tensor(x_np)

    bins = 75
    min_val = -5.0
    max_val = 5.0

    standalone_output = ms.mint.histc(x, bins=bins, min=min_val, max=max_val)

    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_output = ms.mint.histc(x_local, bins=bins, min=min_val, max=max_val)
    parallel_output = parallel_output.full_tensor()

    compare_results(standalone_output, parallel_output)


def test_histc_ext_all_replicated4():
    """
    Feature: HistcExt operator with all replicated.
    Description: Test HistcExt with input not sharded.
    Expectation: Run success and match standalone result.
    """
    ms.set_seed(4)
    np.random.seed(4)

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=base_mesh_shape,
        mesh_dim_names=base_alias_name
    )

    x_placements = (Replicate(), Replicate())

    d, m = 8, 16
    x_np = np.random.randn(d, m).astype(np.float32)
    x = Tensor(x_np)

    bins = 100
    min_val = 0.0
    max_val = 3.0

    standalone_output = ms.mint.histc(x, bins=bins, min=min_val, max=max_val)

    x_local = distribute_tensor(x, mesh, x_placements)
    parallel_output = ms.mint.histc(x_local, bins=bins, min=min_val, max=max_val)
    parallel_output = parallel_output.full_tensor()

    compare_results(standalone_output, parallel_output)
