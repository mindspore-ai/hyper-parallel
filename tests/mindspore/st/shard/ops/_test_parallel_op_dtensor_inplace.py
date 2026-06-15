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
"""Distributed worker tests for ``DTensor.copy_`` / ``zero_`` / ``fill_``.

Verifies that the autograd-invisible in-place ops on ``DTensor`` produce
results numerically equivalent to the same operations on a standalone
single-device tensor."""

# pylint: disable=protected-access

import numpy as np

import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


def setup_module() -> None:
    """Initialize the distributed environment for the test module."""
    ms.set_device("Ascend")
    D.init()


base_mesh_shape = (2,)
base_alias_name = ("dp",)


def test_dtensor_copy_same_placement():
    """copy_ between two same-shape same-placement DTensors matches standalone."""
    np.random.seed(1)

    mesh = init_device_mesh(
        device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name
    )

    dst_np = np.zeros((8, 4), dtype=np.float32)
    src_np = np.random.randn(8, 4).astype(np.float32)

    standalone_dst = Tensor(dst_np.copy())
    standalone_src = Tensor(src_np)
    standalone_dst.copy_(standalone_src)

    dist_dst = distribute_tensor(Tensor(dst_np.copy()), mesh, (Shard(0),))
    dist_src = distribute_tensor(Tensor(src_np), mesh, (Shard(0),))
    dist_dst.copy_(dist_src)

    gathered = dist_dst.full_tensor()
    assert np.allclose(standalone_dst.asnumpy(), gathered.asnumpy(), 1e-5, 1e-6), (
        f"DTensor.copy_ same-placement mismatch: "
        f"standalone={standalone_dst.asnumpy()}, parallel={gathered.asnumpy()}"
    )


def test_dtensor_copy_dtype_cast():
    """fp16 src copied into fp32 dst preserves dst dtype and casts values."""
    np.random.seed(2)

    mesh = init_device_mesh(
        device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name
    )

    dst_np = np.zeros((8, 4), dtype=np.float32)
    src_np = np.random.randn(8, 4).astype(np.float16)

    standalone_dst = Tensor(dst_np.copy())
    standalone_src = Tensor(src_np)
    standalone_dst.copy_(standalone_src)

    dist_dst = distribute_tensor(Tensor(dst_np.copy()), mesh, (Shard(0),))
    dist_src = distribute_tensor(Tensor(src_np), mesh, (Shard(0),))
    dist_dst.copy_(dist_src)

    assert dist_dst._local_tensor.dtype == ms.float32, (
        f"copy_ should preserve dst dtype: expected Float32, "
        f"got {dist_dst._local_tensor.dtype}"
    )
    gathered = dist_dst.full_tensor()
    assert np.allclose(standalone_dst.asnumpy(), gathered.asnumpy(), 1e-3, 1e-3), (
        f"DTensor.copy_ dtype-cast mismatch: "
        f"standalone={standalone_dst.asnumpy()}, parallel={gathered.asnumpy()}"
    )


def test_dtensor_copy_scalar_broadcast():
    """0-d Replicate src broadcasts into Shard dst (placement relaxation case)."""
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name
    )

    dst_np = np.zeros((8, 4), dtype=np.float32)
    scalar_val = 7.0

    standalone_dst = Tensor(dst_np.copy())
    standalone_dst.copy_(Tensor(scalar_val, dtype=ms.float32))

    dist_dst = distribute_tensor(Tensor(dst_np.copy()), mesh, (Shard(0),))
    dist_scalar = distribute_tensor(
        Tensor(scalar_val, dtype=ms.float32), mesh, (Replicate(),)
    )
    dist_dst.copy_(dist_scalar)

    gathered = dist_dst.full_tensor()
    assert np.allclose(standalone_dst.asnumpy(), gathered.asnumpy(), 1e-5, 1e-6), (
        f"DTensor.copy_ scalar-broadcast mismatch: "
        f"standalone={standalone_dst.asnumpy()}, parallel={gathered.asnumpy()}"
    )


def test_dtensor_zero():
    """zero_ resets local shard; full tensor matches standalone zero_."""
    np.random.seed(3)

    mesh = init_device_mesh(
        device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name
    )

    src_np = np.random.randn(8, 4).astype(np.float32)

    standalone = Tensor(src_np.copy())
    standalone.zero_()

    dist_tensor = distribute_tensor(Tensor(src_np.copy()), mesh, (Shard(0),))
    dist_tensor.zero_()

    gathered = dist_tensor.full_tensor()
    assert np.allclose(standalone.asnumpy(), gathered.asnumpy(), 1e-6, 1e-7), (
        f"DTensor.zero_ mismatch: "
        f"standalone={standalone.asnumpy()}, parallel={gathered.asnumpy()}"
    )


def test_dtensor_fill():
    """fill_ writes value into local shard; full tensor matches standalone fill_."""
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=base_mesh_shape, mesh_dim_names=base_alias_name
    )

    dst_np = np.zeros((8, 4), dtype=np.float32)
    fill_val = 4.25

    standalone = Tensor(dst_np.copy())
    standalone.fill_(fill_val)

    dist_tensor = distribute_tensor(Tensor(dst_np.copy()), mesh, (Replicate(),))
    dist_tensor.fill_(fill_val)

    gathered = dist_tensor.full_tensor()
    assert np.allclose(standalone.asnumpy(), gathered.asnumpy(), 1e-6, 1e-7), (
        f"DTensor.fill_ mismatch: "
        f"standalone={standalone.asnumpy()}, parallel={gathered.asnumpy()}"
    )
