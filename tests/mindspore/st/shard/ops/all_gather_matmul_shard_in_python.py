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
"""MindSpore distributed ST for mindspore.ops.all_gather_matmul (MC2 fusion).

Reference computation uses primitive ops:
  AllGather(x1_local) → gathered_x1 → matmul(gathered_x1, x2_local)

MC2 operator handles communication internally; HyperParallel is responsible for:
  - Extracting local tensors from DTensor inputs (preprocess / to_local).
  - Inferring output DTensor layouts (infer_layout).

Numerical comparison uses float16 with atol=1e-2 / rtol=1e-2.

Note on dtype choice: AGM reference is computed with float16 primitives (AllGather + matmul),
so MC2 output and reference share the same numerical path — errors cancel and max_diff < 1e-3.
float16 (10-bit mantissa) is preferred over bfloat16 for higher precision.
"""
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
import mindspore.ops as ms_ops
from mindspore import Tensor
from mindspore.ops.function.comm_func import all_gather_into_tensor
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

# Global tensor dimensions — satisfy CANN constraints: K ∈ [256, 65535), M is multiple of world_size.
M = 128
K = 256
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
    assert np.allclose(a, b, atol=1e-2, rtol=1e-2), (
        f"{tag}: max_diff={np.abs(a - b).max():.4f}, "
        f"actual shape={a.shape}, ref shape={b.shape}"
    )


def test_agm_tp_input_shard0_x2_replicate():
    """
    Feature: AllGatherMatmul TP — x1 Shard(0) on tp, x2 Replicate, gather_output=True.
    Description:
        - 2-rank tp mesh; x1 global (128, 256), each rank holds (64, 256).
        - x2 global (256, 512), replicated on each rank.
        - Reference: AllGather(x1_local) → (128, 256); matmul → (128, 512).
        - Verifies output shape, values, and gather_out shape/values.
    Expectation: Output matches reference within float16 tolerance.
    """
    x1_np, x2_np = _make_inputs()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("tp",))
    group = mesh.get_group("tp")
    world_size = 2

    x1_dtensor = distribute_tensor(_fp16(x1_np), mesh, (Shard(0),))
    x2_dtensor = distribute_tensor(_fp16(x2_np), mesh, (Replicate(),))

    out_dtensor, gather_dtensor = ms_ops.all_gather_matmul(
        x1_dtensor, x2_dtensor, group, world_size, gather_output=True
    )

    x1_local = x1_dtensor.to_local()
    gathered_x1, _ = all_gather_into_tensor(None, x1_local, group=group)
    ref_output = ms_ops.matmul(gathered_x1, x2_dtensor.to_local())

    out_full = _to_f32np(out_dtensor.full_tensor())
    ref_np = _to_f32np(ref_output)
    _allclose(out_full, ref_np, "agm_tp_input_shard0_x2_replicate output")

    gather_full = _to_f32np(gather_dtensor.full_tensor())
    gathered_ref = _to_f32np(gathered_x1)
    _allclose(gather_full, gathered_ref, "agm_tp_input_shard0_x2_replicate gather_out")


def test_agm_tp_input_shard0_x2_shard1():
    """
    Feature: AllGatherMatmul TP — x1 Shard(0) on tp, x2 Shard(1) on tp (n-dim sharded).
    Description:
        - 2-rank tp mesh; x1 global (128, 256), x2 global (256, 512).
        - x1 Shard(0): each rank holds (64, 256).
        - x2 Shard(1): each rank holds (256, 256) — n-dim halved.
        - Reference: AllGather(x1_local); matmul → (128, 256) per rank.
        - Full output reconstructed by gathering along n-dim.
    Expectation: Gathered output matches reference within float16 tolerance.
    """
    x1_np, x2_np = _make_inputs()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("tp",))
    group = mesh.get_group("tp")
    world_size = 2

    x1_dtensor = distribute_tensor(_fp16(x1_np), mesh, (Shard(0),))
    x2_dtensor = distribute_tensor(_fp16(x2_np), mesh, (Shard(1),))

    out_dtensor, _ = ms_ops.all_gather_matmul(
        x1_dtensor, x2_dtensor, group, world_size, gather_output=False
    )

    x1_local = x1_dtensor.to_local()
    gathered_x1, _ = all_gather_into_tensor(None, x1_local, group=group)
    ref_output = ms_ops.matmul(gathered_x1, _fp16(x2_np))

    out_full = _to_f32np(out_dtensor.full_tensor())
    ref_np = _to_f32np(ref_output)
    _allclose(out_full, ref_np, "agm_tp_input_shard0_x2_shard1 output")


def test_agm_trans_x2_true():
    """
    Feature: AllGatherMatmul with trans_x2=True.
    Description:
        - 2-rank tp mesh; x1 Shard(0), x2 physical shape (N, K) with Shard(0) on n-dim.
        - Reference: AllGather(x1_local); matmul(gathered_x1, x2_local.T).
        - C++ impl transposes x2 before calling CANN.
    Expectation: Output matches reference within float16 tolerance.
    """
    x1_np = np.random.randn(M, K).astype(np.float32)
    x2_t_np = np.random.randn(N, K).astype(np.float32)  # physical shape (N, K) when trans_x2=True

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("tp",))
    group = mesh.get_group("tp")
    world_size = 2

    x1_dtensor = distribute_tensor(_fp16(x1_np), mesh, (Shard(0),))
    # x2 shape (N, K): Shard(0) shards n-dim on tp
    x2_dtensor = distribute_tensor(_fp16(x2_t_np), mesh, (Shard(0),))

    out_dtensor, _ = ms_ops.all_gather_matmul(
        x1_dtensor, x2_dtensor, group, world_size, gather_output=False, trans_x2=True
    )

    x1_local = x1_dtensor.to_local()
    gathered_x1, _ = all_gather_into_tensor(None, x1_local, group=group)
    # trans_x2=True: C++ transposes x2 before matmul; reference uses x2_t.T = (K, N) sliced portion
    x2_t_full = _fp16(x2_t_np)
    ref_output = ms_ops.matmul(gathered_x1, x2_t_full.T)

    out_full = _to_f32np(out_dtensor.full_tensor())
    ref_np = _to_f32np(ref_output)
    _allclose(out_full, ref_np, "agm_trans_x2_true output")


def test_agm_gather_output_false():
    """
    Feature: AllGatherMatmul with gather_output=False.
    Description:
        - 2-rank tp mesh; x1 Shard(0), x2 Replicate.
        - gather_output=False: CANN returns empty tensor for gather_out.
        - Verifies output values and that gather_out is empty (shape[0]==0).
    Expectation: Output matches reference within float16 tolerance; gather_out tensor is empty.
    """
    x1_np, x2_np = _make_inputs()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("tp",))
    group = mesh.get_group("tp")
    world_size = 2

    x1_dtensor = distribute_tensor(_fp16(x1_np), mesh, (Shard(0),))
    x2_dtensor = distribute_tensor(_fp16(x2_np), mesh, (Replicate(),))

    out_dtensor, gather_dtensor = ms_ops.all_gather_matmul(
        x1_dtensor, x2_dtensor, group, world_size, gather_output=False
    )

    x1_local = x1_dtensor.to_local()
    gathered_x1, _ = all_gather_into_tensor(None, x1_local, group=group)
    ref_output = ms_ops.matmul(gathered_x1, x2_dtensor.to_local())

    out_full = _to_f32np(out_dtensor.full_tensor())
    ref_np = _to_f32np(ref_output)
    _allclose(out_full, ref_np, "agm_gather_output_false output")

    gather_local = gather_dtensor.to_local()
    assert gather_local.shape[0] == 0, (
        f"gather_output=False: expected gather_out shape[0]==0, "
        f"got shape={gather_local.shape}"
    )


def test_agm_dp_tp_input_shard0():
    """
    Feature: AllGatherMatmul on 2D (dp=2, tp=2) mesh.
    Description:
        - 4-rank mesh (dp=2, tp=2); x1 Shard(0) on tp only; x2 Replicate.
        - x1 global (128, 256); each rank holds (64, 256) — halved by tp.
        - Reference: AllGather on tp axis → (128, 256); matmul → (128, 512).
    Expectation: Output matches reference within float16 tolerance.
    """
    x1_np, x2_np = _make_inputs()

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    tp_group = mesh.get_group("tp")
    world_size = 2

    x1_dtensor = distribute_tensor(_fp16(x1_np), mesh, (Replicate(), Shard(0)))
    x2_dtensor = distribute_tensor(_fp16(x2_np), mesh, (Replicate(), Replicate()))

    out_dtensor, _ = ms_ops.all_gather_matmul(
        x1_dtensor, x2_dtensor, tp_group, world_size, gather_output=False
    )

    x1_local = x1_dtensor.to_local()
    gathered_x1, _ = all_gather_into_tensor(None, x1_local, group=tp_group)
    ref_output = ms_ops.matmul(gathered_x1, x2_dtensor.to_local())

    out_full = _to_f32np(out_dtensor.full_tensor())
    ref_np = _to_f32np(ref_output)
    _allclose(out_full, ref_np, "agm_dp_tp_input_shard0 output")


def test_agm_k_sharded_mp2_tp2():
    """
    Feature: AllGatherMatmul with m-dim sharded on mp and k-dim sharded on tp.
    Description:
        - 4-rank (mp=2, tp=2) mesh; x1 (Shard(0), Shard(1)), x2 (Replicate(), Shard(0)).
        - K_SHARDED=512 so k_local=256 per tp rank, satisfying CANN k>=256 constraint.
        - AllGather on mp_group gathers m-dim; k sharded on tp produces Partial output on tp.
        - full_tensor() resolves Partial via AllReduce on tp to yield correct (M, N) result.
    Expectation: Output matches float16 matmul reference within tolerance.
    """
    k_sharded = 512
    x1_np = np.random.randn(M, k_sharded).astype(np.float32)
    x2_np = np.random.randn(k_sharded, N).astype(np.float32)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("mp", "tp"))
    mp_group = mesh.get_group("mp")
    world_size = 2

    x1_dtensor = distribute_tensor(_fp16(x1_np), mesh, (Shard(0), Shard(1)))
    x2_dtensor = distribute_tensor(_fp16(x2_np), mesh, (Replicate(), Shard(0)))

    out_dtensor, _ = ms_ops.all_gather_matmul(
        x1_dtensor, x2_dtensor, mp_group, world_size, gather_output=False
    )

    # k-sharded path: CANN splits k across tp ranks and AllReduces partial sums.
    # Accumulated float16 rounding differs from a single full-k matmul, so use
    # float32 numpy reference with the same loose tolerance as MRS k-sharded tests.
    ref_np = np.matmul(x1_np, x2_np)
    out_full = _to_f32np(out_dtensor.full_tensor())
    assert np.allclose(out_full, ref_np, atol=0.1, rtol=0.02), (
        f"agm_k_sharded_mp2_tp2 output: max_diff={np.abs(out_full - ref_np).max():.4f}, "
        f"actual shape={out_full.shape}, ref shape={ref_np.shape}"
    )


def test_agm_k_sharded_mp2_np2_tp2():
    """
    Feature: AllGatherMatmul with m, n, k each sharded on a distinct mesh axis.
    Description:
        - 8-rank (mp=2, np=2, tp=2) mesh.
        - x1 (M, K_SHARDED) with placements (Shard(0), Replicate(), Shard(1)):
          m sharded on mp, k sharded on tp; each rank holds (M/2, K/2).
        - x2 (K_SHARDED, N_SHARDED) with placements (Replicate(), Shard(1), Shard(0)):
          n sharded on np, k sharded on tp; each rank holds (K/2, N/2).
        - K_SHARDED=512, N_SHARDED=512 so k_local=256 satisfying CANN k>=256 constraint.
        - AllGather on mp_group gathers m-dim; k on tp → Partial on tp; n on np → Shard on n.
        - full_tensor() resolves Partial (AllReduce on tp) and Shard (AllGather on np).
    Expectation: Output matches float16 matmul reference within tolerance.
    """
    k_sharded = 512
    n_sharded = 512
    x1_np = np.random.randn(M, k_sharded).astype(np.float32)
    x2_np = np.random.randn(k_sharded, n_sharded).astype(np.float32)

    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("mp", "np", "tp")
    )
    mp_group = mesh.get_group("mp")
    world_size = 2

    x1_dtensor = distribute_tensor(_fp16(x1_np), mesh, (Shard(0), Replicate(), Shard(1)))
    x2_dtensor = distribute_tensor(_fp16(x2_np), mesh, (Replicate(), Shard(1), Shard(0)))

    out_dtensor, _ = ms_ops.all_gather_matmul(
        x1_dtensor, x2_dtensor, mp_group, world_size, gather_output=False
    )

    # k-sharded path: same rounding argument as test_agm_k_sharded_mp2_tp2;
    # use float32 numpy reference with loose tolerance.
    ref_np = np.matmul(x1_np, x2_np)
    out_full = _to_f32np(out_dtensor.full_tensor())
    assert np.allclose(out_full, ref_np, atol=0.1, rtol=0.02), (
        f"agm_k_sharded_mp2_np2_tp2 output: max_diff={np.abs(out_full - ref_np).max():.4f}, "
        f"actual shape={out_full.shape}, ref shape={ref_np.shape}"
    )
