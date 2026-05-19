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
"""MindSpore distributed ST for mindspore.ops.matmul_reduce_scatter (MC2 fusion).

Reference computation uses primitive ops:
  local matmul(x1_local, x2_local) → partial_out → ReduceScatter → output

MC2 operator handles communication internally; HyperParallel is responsible for:
  - Extracting local tensors from DTensor inputs (preprocess / to_local).
  - Inferring output DTensor layouts (infer_layout).

Numerical comparison: full_tensor() of the output DTensor vs np.matmul of global inputs,
using float16 (10-bit mantissa, ~8x more precise than bfloat16) with atol=0.1 / rtol=0.02.

Note on dtype choice: MRS reference is float32 numpy (unlike AGM which uses bf16 primitives
as reference). With bfloat16 (7-bit mantissa + ReduceScatter cross-rank sum), max_diff was
~0.5, requiring atol=0.5. Switching to float16 reduces expected max_diff to ~0.065,
allowing the tighter atol=0.1.
"""
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
import mindspore.ops as ms_ops
from mindspore import Tensor
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate


def setup_module():
    """Initialize MindSpore context and communication."""
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    D.init()


np.random.seed(42)
ms.set_seed(42)
ms.set_deterministic(True)

# Global tensor dimensions — satisfy CANN constraints for MatmulReduceScatter:
# K is sharded by world_size, so k_local = K / world_size must be ≥ 256.
# With world_size=2: K ≥ 512. M must be a multiple of world_size.
M = 128
K = 512
N = 512


def _fp16(arr: np.ndarray) -> Tensor:
    """Convert float32 numpy array to MindSpore float16 Tensor."""
    return Tensor(arr.astype(np.float32), ms.float16)


def _to_f32np(t: Tensor) -> np.ndarray:
    """Convert MindSpore tensor (any dtype) to float32 numpy array."""
    return t.astype(ms.float32).asnumpy()


def _make_inputs():
    """Generate global float32 numpy arrays for x1 and x2."""
    x1_np = np.random.randn(M, K).astype(np.float32)
    x2_np = np.random.randn(K, N).astype(np.float32)
    return x1_np, x2_np


def _allclose(a: np.ndarray, b: np.ndarray, tag: str):
    # float16 has 10-bit mantissa (~8x more precise than bfloat16).
    # With k_local=256 products per rank + ReduceScatter sum, expected max_diff ~0.065
    # for output values ~20-40 → atol=0.1, rtol=0.02.
    assert np.allclose(a, b, atol=0.1, rtol=0.02), (
        f"{tag}: max_diff={np.abs(a - b).max():.4f}, "
        f"actual shape={a.shape}, ref shape={b.shape}"
    )


def test_mrs_tp_basic():
    """
    Feature: MatmulReduceScatter TP — x1 Shard(1) on tp (k-dim), x2 Shard(0) on tp (k-dim).
    Description:
        - 2-rank tp mesh; x1 global (128, 256), x2 global (256, 512).
        - x1 Shard(1): each rank holds (128, 128) — k-dim halved.
        - x2 Shard(0): each rank holds (128, 512) — k-dim halved.
        - Reference: full matmul(x1_global, x2_global) → (128, 512); each rank gets M/2 rows.
        - Verifies output full_tensor matches global reference within float16 tolerance.
    Expectation: Output matches reference within float16 tolerance.
    """
    x1_np, x2_np = _make_inputs()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("tp",))
    group = mesh.get_group("tp")
    world_size = 2

    x1_dtensor = distribute_tensor(_fp16(x1_np), mesh, (Shard(1),))
    x2_dtensor = distribute_tensor(_fp16(x2_np), mesh, (Shard(0),))

    out_dtensor = ms_ops.matmul_reduce_scatter(x1_dtensor, x2_dtensor, group, world_size)

    ref_np = np.matmul(x1_np, x2_np)
    out_full = _to_f32np(out_dtensor.full_tensor())
    _allclose(out_full, ref_np, "mrs_tp_basic output")


def test_mrs_trans_x2_true():
    """
    Feature: MatmulReduceScatter with trans_x2=True.
    Description:
        - 2-rank tp mesh; x1 Shard(1) on k, x2 physical shape (N, K) with Shard(1) on k-dim.
        - Reference: full matmul(x1_global, x2_t_global.T) — x2_t physical (512, 256), transposed is (256, 512).
        - C++ impl transposes x2 before calling CANN.
    Expectation: Output matches reference within float16 tolerance.
    """
    x1_np = np.random.randn(M, K).astype(np.float32)
    x2_t_np = np.random.randn(N, K).astype(np.float32)  # physical shape (N, K) when trans_x2=True

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("tp",))
    group = mesh.get_group("tp")
    world_size = 2

    x1_dtensor = distribute_tensor(_fp16(x1_np), mesh, (Shard(1),))
    # x2 physical (N, K): Shard(1) shards k-dim on tp
    x2_dtensor = distribute_tensor(_fp16(x2_t_np), mesh, (Shard(1),))

    out_dtensor = ms_ops.matmul_reduce_scatter(
        x1_dtensor, x2_dtensor, group, world_size, trans_x2=True
    )

    # trans_x2=True: C++ transposes x2 before matmul; reference is matmul(x1, x2_t.T)
    ref_np = np.matmul(x1_np, x2_t_np.T)  # x2_t.T shape: (K, N) → output (M, N)
    out_full = _to_f32np(out_dtensor.full_tensor())
    _allclose(out_full, ref_np, "mrs_trans_x2_true output")


def test_mrs_large_m():
    """
    Feature: MatmulReduceScatter with larger M dimension.
    Description:
        - 2-rank tp mesh; x1 global (256, 512) Shard(1) on k, x2 global (512, 512) Shard(0) on k.
        - Verifies output correctness with M=256 (larger than test_mrs_tp_basic which uses M=128).
    Expectation: Output matches reference within float16 tolerance.
    """
    x1_np = np.random.randn(256, K).astype(np.float32)
    x2_np = np.random.randn(K, N).astype(np.float32)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("tp",))
    group = mesh.get_group("tp")
    world_size = 2

    x1_dtensor = distribute_tensor(_fp16(x1_np), mesh, (Shard(1),))
    x2_dtensor = distribute_tensor(_fp16(x2_np), mesh, (Shard(0),))

    out_dtensor = ms_ops.matmul_reduce_scatter(x1_dtensor, x2_dtensor, group, world_size)

    ref_np = np.matmul(x1_np, x2_np)
    out_full = _to_f32np(out_dtensor.full_tensor())
    _allclose(out_full, ref_np, "mrs_large_m output")


def test_mrs_dp_tp_basic():
    """
    Feature: MatmulReduceScatter on 2D (dp=2, tp=2) mesh.
    Description:
        - 4-rank mesh (dp=2, tp=2); x1 Shard(1) on tp only; x2 Shard(0) on tp only.
        - x1 global (128, 256); each rank holds (128, 128) — halved by tp.
        - Reference: full matmul(x1_global, x2_global) → (128, 512); each tp-rank gets M/2 rows.
        - Verifies forward output correctness on 2D mesh.
    Expectation: Output matches reference within float16 tolerance.
    """
    x1_np, x2_np = _make_inputs()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    tp_group = mesh.get_group("tp")
    world_size = 2

    x1_dtensor = distribute_tensor(_fp16(x1_np), mesh, (Replicate(), Shard(1)))
    x2_dtensor = distribute_tensor(_fp16(x2_np), mesh, (Replicate(), Shard(0)))

    out_dtensor = ms_ops.matmul_reduce_scatter(x1_dtensor, x2_dtensor, tp_group, world_size)

    ref_np = np.matmul(x1_np, x2_np)
    out_full = _to_f32np(out_dtensor.full_tensor())
    _allclose(out_full, ref_np, "mrs_dp_tp_basic output")


def test_mrs_mp2_np2_tp2():
    """
    Feature: MatmulReduceScatter with m, n, k each sharded on a distinct mesh axis.
    Description:
        - 8-rank (mp=2, np=2, tp=2) mesh.
        - x1 (M, K) with placements (Shard(0), Replicate(), Shard(1)):
          m sharded on mp, k sharded on tp; each rank holds (M/2, K/2).
        - x2 (K, N) with placements (Replicate(), Shard(1), Shard(0)):
          n sharded on np, k sharded on tp; each rank holds (K/2, N/2).
        - ReduceScatter on tp_group (world_size=2): sums k partial and scatters m;
          output m is jointly sharded on (tp, mp), n sharded on np;
          each rank holds (M/4, N/2).
        - full_tensor() reconstructs (M, N) via AllGather on both m and n dimensions.
    Expectation: Output matches reference within float16 tolerance.
    """
    x1_np, x2_np = _make_inputs()

    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("mp", "np", "tp")
    )
    tp_group = mesh.get_group("tp")
    world_size = 2

    x1_dtensor = distribute_tensor(_fp16(x1_np), mesh, (Shard(0), Replicate(), Shard(1)))
    x2_dtensor = distribute_tensor(_fp16(x2_np), mesh, (Replicate(), Shard(1), Shard(0)))

    out_dtensor = ms_ops.matmul_reduce_scatter(x1_dtensor, x2_dtensor, tp_group, world_size)

    ref_np = np.matmul(x1_np, x2_np)
    out_full = _to_f32np(out_dtensor.full_tensor())
    _allclose(out_full, ref_np, "mrs_mp2_np2_tp2 output")
