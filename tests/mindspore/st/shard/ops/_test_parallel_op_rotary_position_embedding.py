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
"""MindSpore distributed ST for RotaryPositionEmbedding operator.

Formula: y = x * cos + x_rotate * sin, where x_rotate rotates within the last
(D) dimension.  All B/N/S positions are element-wise independent, so they can
be freely sharded.  D must remain replicated.

Shapes used throughout (BNSD layout):
  B=4, N=4, S=16, D=64  →  B*N=16 ≤ S*8=128 (CANN float16 constraint). ✓

cos/sin broadcast variants:
  full-shape: (B, N, S, D)  — shards on the same dims as x.
  broadcast:  (1, 1, S, D)  — replicated on B/N; may shard on S when x shards S.

Precision tolerance:
  atol=rtol=1e-3.  RPE is pure element-wise with no cross-rank reduction in the
  forward pass, so distributed and standalone invoke the same NPU kernel on the
  same data — expected max_diff ≈ 0.  For backward Partial("sum") gradients the
  all-reduce adds at most one float16 rounding step (epsilon_f16 ≈ 1e-3), so
  1e-3 is both correct and tight.

mode values exercised:
  0=rotate_half, 1=rotate_interleaved, 2=quarter, 3=interleave-half.
  CANN backward is only supported for mode=0 and mode=1.  mode=2/3 are
  validated in forward-only tests (_run_standalone_fwd, no ms.grad call).
"""
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor
from mindspore.ops import rotary_position_embedding

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial


def setup_module() -> None:
    """Initialize MindSpore context and communication."""
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    D.init()


np.random.seed(42)
ms.set_seed(42)
ms.set_deterministic(True)

B = 4
N = 4
S = 16
D_DIM = 64

_INPUT_NAMES = ("x", "cos", "sin")
_TOL = {"atol": 1e-3, "rtol": 1e-3}


def _np_to_f16(arr: np.ndarray) -> Tensor:
    """Convert float32 numpy array to MindSpore float16 Tensor."""
    return Tensor(arr, ms.float16)


def _to_f32np(t) -> np.ndarray:
    """Convert any MindSpore tensor to float32 numpy array."""
    return t.astype(ms.float32).asnumpy()


def _make_rpe(mode: int):
    """Return a 3-argument RPE callable with the given rotation mode captured.

    Args:
        mode: Rotation mode (0=rotate_half, 1=rotate_interleaved,
              2=quarter, 3=interleave-half).

    Returns:
        Callable (x, cos, sin) -> y that ms.grad can differentiate for mode=0/1.
    """
    def _fn(x, cos, sin):
        return rotary_position_embedding(x, cos, sin, mode=mode)
    return _fn


def _run_standalone_fwd(x_np, cos_np, sin_np, mode=0):
    """Compute standalone RPE forward pass only.

    Used for mode=2 and mode=3 whose CANN backward kernels are not available.

    Args:
        x_np: float32 ndarray, shape (B, N, S, D).
        cos_np: float32 ndarray, shape (B, N, S, D) or (1, 1, S, D).
        sin_np: float32 ndarray, same shape as cos_np.
        mode: RPE rotation mode.

    Returns:
        fwd_np: float32 ndarray, forward output.
    """
    return _to_f32np(_make_rpe(mode)(_np_to_f16(x_np), _np_to_f16(cos_np), _np_to_f16(sin_np)))


def _run_standalone(x_np, cos_np, sin_np, mode=0):
    """Compute standalone RPE forward and backward.

    Use only for mode=0 and mode=1, which have CANN backward support.

    Args:
        x_np: float32 ndarray, shape (B, N, S, D).
        cos_np: float32 ndarray, shape (B, N, S, D) or (1, 1, S, D).
        sin_np: float32 ndarray, same shape as cos_np.
        mode: RPE rotation mode (must be 0 or 1).

    Returns:
        tuple: (fwd_np, (grad_x_np, grad_cos_np, grad_sin_np)) — all float32.
    """
    x = _np_to_f16(x_np)
    cos = _np_to_f16(cos_np)
    sin = _np_to_f16(sin_np)
    fn = _make_rpe(mode)
    fwd_np = _to_f32np(fn(x, cos, sin))
    grads = ms.grad(fn, (0, 1, 2))(x, cos, sin)
    return fwd_np, tuple(_to_f32np(g) for g in grads)


def _get_grad_placements(d_inputs: tuple) -> list:
    """Derive the correct DTensor placement for each input's gradient.

    Rule: if any input is Shard on mesh dimension d, then a Replicate input on
    that dimension receives gradient contributions from all ranks → Partial("sum").
    Shard inputs keep their Shard placement.

    Args:
        d_inputs: Tuple of DTensor inputs (dx, dcos, dsin).

    Returns:
        list of placement tuples, one per input.
    """
    n_dims = len(d_inputs[0].layout.placements)
    dim_has_shard = [
        any(isinstance(d_inp.layout.placements[d], Shard) for d_inp in d_inputs)
        for d in range(n_dims)
    ]
    result = []
    for d_inp in d_inputs:
        grad_placements = tuple(
            Partial("sum") if isinstance(p, Replicate) and dim_has_shard[dim_idx] else p
            for dim_idx, p in enumerate(d_inp.layout.placements)
        )
        result.append(grad_placements)
    return result


def _assert_fwd(dist_out, ref_np, tag):
    """Assert gathered distributed forward output matches standalone reference."""
    dist_np = _to_f32np(dist_out.full_tensor())
    assert np.allclose(dist_np, ref_np, **_TOL), (
        f"{tag} fwd mismatch: "
        f"max_diff={np.abs(dist_np - ref_np).max()}"
    )


def _assert_bwd(raw_grads, d_inputs, ref_grads, tag):
    """Assert all input gradients match standalone reference.

    Reconstructs each gradient as a DTensor using placements from
    _get_grad_placements (handles both Shard and Partial("sum") cases), gathers
    via full_tensor(), and compares to ref_grads.

    Args:
        raw_grads: Tuple of local gradient tensors from ms.grad.
        d_inputs: Tuple of DTensor inputs (dx, dcos, dsin).
        ref_grads: Tuple of float32 reference gradient ndarrays.
        tag: Human-readable label for assertion messages.
    """
    grad_placements = _get_grad_placements(d_inputs)
    for name, raw_g, d_inp, ref, g_pl in zip(
            _INPUT_NAMES, raw_grads, d_inputs, ref_grads, grad_placements):
        dist_g = (
            DTensor.from_local(raw_g, d_inp.layout.mesh, g_pl)
            .full_tensor()
            .astype(ms.float32)
            .asnumpy()
        )
        assert np.allclose(dist_g, ref, **_TOL), (
            f"{tag} backward grad_{name} mismatch: "
            f"max_diff={np.abs(dist_g - ref).max()}"
        )


# ---------------------------------------------------------------------------
# 2-card tests
# ---------------------------------------------------------------------------

def test_rpe_replicated():
    """
    Feature: RotaryPositionEmbedding (MindSpore) all replicated, 2-card dp mesh.
    Description:
        - 1-D dp mesh (2 cards); x/cos/sin all Replicate.
        - cos/sin shape (1, 1, S, D) — broadcast over B and N.
        - float16, mode=0 (rotate_half).
    Expectation: Distributed forward matches standalone within tolerance.
    """
    mode = 0
    x_np = np.random.randn(B, N, S, D_DIM).astype(np.float32)
    cos_np = np.random.randn(1, 1, S, D_DIM).astype(np.float32)
    sin_np = np.random.randn(1, 1, S, D_DIM).astype(np.float32)
    ref_fwd = _run_standalone_fwd(x_np, cos_np, sin_np, mode=mode)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    dx = distribute_tensor(_np_to_f16(x_np), mesh, (Replicate(),))
    dcos = distribute_tensor(_np_to_f16(cos_np), mesh, (Replicate(),))
    dsin = distribute_tensor(_np_to_f16(sin_np), mesh, (Replicate(),))

    _assert_fwd(_make_rpe(mode)(dx, dcos, dsin), ref_fwd, "replicated mode=0")


def test_rpe_dp_b():
    """
    Feature: RotaryPositionEmbedding (MindSpore) B-dim data parallel, 2-card dp mesh.
    Description:
        - 1-D dp mesh (2 cards); x Shard(0); cos/sin Replicate, broadcast (1,1,S,D).
        - float16, mode=1 (rotate_interleaved).
        - Forward verified; all three gradients verified:
            grad_x → Shard(0); grad_cos/sin → Partial("sum") via _get_grad_placements.
    Expectation: Distributed forward and all gradients match standalone within tolerance.
    """
    mode = 1
    x_np = np.random.randn(B, N, S, D_DIM).astype(np.float32)
    cos_np = np.random.randn(1, 1, S, D_DIM).astype(np.float32)
    sin_np = np.random.randn(1, 1, S, D_DIM).astype(np.float32)
    ref_fwd, ref_grads = _run_standalone(x_np, cos_np, sin_np, mode=mode)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))
    dx = distribute_tensor(_np_to_f16(x_np), mesh, (Shard(0),))
    dcos = distribute_tensor(_np_to_f16(cos_np), mesh, (Replicate(),))
    dsin = distribute_tensor(_np_to_f16(sin_np), mesh, (Replicate(),))
    d_inputs = (dx, dcos, dsin)

    fn = _make_rpe(mode)
    _assert_fwd(fn(*d_inputs), ref_fwd, "dp_b mode=1")

    raw_grads = ms.grad(fn, (0, 1, 2))(*d_inputs)
    _assert_bwd(raw_grads, d_inputs, ref_grads, "dp_b mode=1")


def test_rpe_tp_n():
    """
    Feature: RotaryPositionEmbedding (MindSpore) N-dim tensor parallel, 2-card tp mesh.
    Description:
        - 1-D tp mesh (2 cards); x/cos/sin all Shard(1).
        - cos/sin full shape (B, N, S, D) — same sharding as x.
        - float16, mode=0 (rotate_half).
        - Forward and all three gradients verified; all grad placements = Shard(1).
    Expectation: Distributed forward and gradients match standalone within tolerance.
    """
    mode = 0
    x_np = np.random.randn(B, N, S, D_DIM).astype(np.float32)
    cos_np = np.random.randn(B, N, S, D_DIM).astype(np.float32)
    sin_np = np.random.randn(B, N, S, D_DIM).astype(np.float32)
    ref_fwd, ref_grads = _run_standalone(x_np, cos_np, sin_np, mode=mode)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("tp",))
    dx = distribute_tensor(_np_to_f16(x_np), mesh, (Shard(1),))
    dcos = distribute_tensor(_np_to_f16(cos_np), mesh, (Shard(1),))
    dsin = distribute_tensor(_np_to_f16(sin_np), mesh, (Shard(1),))
    d_inputs = (dx, dcos, dsin)

    fn = _make_rpe(mode)
    _assert_fwd(fn(*d_inputs), ref_fwd, "tp_n mode=0")

    raw_grads = ms.grad(fn, (0, 1, 2))(*d_inputs)
    _assert_bwd(raw_grads, d_inputs, ref_grads, "tp_n mode=0")


# ---------------------------------------------------------------------------
# 4-card tests
# ---------------------------------------------------------------------------

def test_rpe_dp_tp():
    """
    Feature: RotaryPositionEmbedding (MindSpore) 2-D dp×tp mesh, B by dp, N by tp.
    Description:
        - 4-card 2-D mesh (dp=2, tp=2); x/cos/sin all (Shard(0), Shard(1)).
        - cos/sin full shape (B, N, S, D) — same placements as x.
        - float16, mode=0 (rotate_half).
        - Forward and all three gradients verified; all grad placements = (Shard(0), Shard(1)).
    Expectation: Distributed fwd and all gradients match standalone within tolerance.
    """
    mode = 0
    x_np = np.random.randn(B, N, S, D_DIM).astype(np.float32)
    cos_np = np.random.randn(B, N, S, D_DIM).astype(np.float32)
    sin_np = np.random.randn(B, N, S, D_DIM).astype(np.float32)
    ref_fwd, ref_grads = _run_standalone(x_np, cos_np, sin_np, mode=mode)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    placements = (Shard(0), Shard(1))
    dx = distribute_tensor(_np_to_f16(x_np), mesh, placements)
    dcos = distribute_tensor(_np_to_f16(cos_np), mesh, placements)
    dsin = distribute_tensor(_np_to_f16(sin_np), mesh, placements)
    d_inputs = (dx, dcos, dsin)

    fn = _make_rpe(mode)
    _assert_fwd(fn(*d_inputs), ref_fwd, "dp_tp mode=0")

    raw_grads = ms.grad(fn, (0, 1, 2))(*d_inputs)
    _assert_bwd(raw_grads, d_inputs, ref_grads, "dp_tp mode=0")


def test_rpe_dp_sp():
    """
    Feature: RotaryPositionEmbedding (MindSpore) 2-D dp×sp mesh, B by dp, S by sp.
    Description:
        - 4-card 2-D mesh (dp=2, sp=2); x (Shard(0), Shard(2)).
        - cos/sin broadcast (1, 1, S, D): dp Replicate, sp Shard(2).
          Local S slices of cos/sin align with the corresponding S slices of x.
        - float16, mode=2 (quarter); forward only (CANN backward not available for mode=2).
    Expectation: Distributed forward matches standalone within tolerance.
    """
    mode = 2
    x_np = np.random.randn(B, N, S, D_DIM).astype(np.float32)
    cos_np = np.random.randn(1, 1, S, D_DIM).astype(np.float32)
    sin_np = np.random.randn(1, 1, S, D_DIM).astype(np.float32)
    ref_fwd = _run_standalone_fwd(x_np, cos_np, sin_np, mode=mode)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "sp"))
    dx = distribute_tensor(_np_to_f16(x_np), mesh, (Shard(0), Shard(2)))
    dcos = distribute_tensor(_np_to_f16(cos_np), mesh, (Replicate(), Shard(2)))
    dsin = distribute_tensor(_np_to_f16(sin_np), mesh, (Replicate(), Shard(2)))

    _assert_fwd(_make_rpe(mode)(dx, dcos, dsin), ref_fwd, "dp_sp mode=2")


def test_rpe_tp_sp():
    """
    Feature: RotaryPositionEmbedding (MindSpore) 2-D tp×sp mesh, N by tp, S by sp.
    Description:
        - 4-card 2-D mesh (tp=2, sp=2); x/cos/sin full shape (B, N, S, D).
          Placements: (Shard(1), Shard(2)) — tp shards N, sp shards S.
        - float16, mode=3 (interleave-half); forward only (CANN backward not available for mode=3).
    Expectation: Distributed forward matches standalone within tolerance.
    """
    mode = 3
    x_np = np.random.randn(B, N, S, D_DIM).astype(np.float32)
    cos_np = np.random.randn(B, N, S, D_DIM).astype(np.float32)
    sin_np = np.random.randn(B, N, S, D_DIM).astype(np.float32)
    ref_fwd = _run_standalone_fwd(x_np, cos_np, sin_np, mode=mode)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("tp", "sp"))
    placements = (Shard(1), Shard(2))
    dx = distribute_tensor(_np_to_f16(x_np), mesh, placements)
    dcos = distribute_tensor(_np_to_f16(cos_np), mesh, placements)
    dsin = distribute_tensor(_np_to_f16(sin_np), mesh, placements)

    _assert_fwd(_make_rpe(mode)(dx, dcos, dsin), ref_fwd, "tp_sp mode=3")


def test_rpe_dp_tp_cos_full():
    """
    Feature: RotaryPositionEmbedding (MindSpore) 2-D dp×tp mesh, broadcast cos/sin.
    Description:
        - 4-card 2-D mesh (dp=2, tp=2); x (Shard(0), Shard(1)).
        - cos/sin broadcast (1, 1, S, D): both Replicate on dp and tp.
          Validates that the kernel broadcasts B=1/N=1 cos/sin against the
          sharded x without requiring cos/sin to be co-sharded.
        - float16, mode=2 (quarter); forward only (CANN backward not available for mode=2).
    Expectation: Distributed forward matches standalone within tolerance.
    """
    mode = 2
    x_np = np.random.randn(B, N, S, D_DIM).astype(np.float32)
    cos_np = np.random.randn(1, 1, S, D_DIM).astype(np.float32)
    sin_np = np.random.randn(1, 1, S, D_DIM).astype(np.float32)
    ref_fwd = _run_standalone_fwd(x_np, cos_np, sin_np, mode=mode)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    dx = distribute_tensor(_np_to_f16(x_np), mesh, (Shard(0), Shard(1)))
    dcos = distribute_tensor(_np_to_f16(cos_np), mesh, (Replicate(), Replicate()))
    dsin = distribute_tensor(_np_to_f16(sin_np), mesh, (Replicate(), Replicate()))

    _assert_fwd(_make_rpe(mode)(dx, dcos, dsin), ref_fwd, "dp_tp_cos_full mode=2")


# ---------------------------------------------------------------------------
# 8-card tests
# ---------------------------------------------------------------------------

def test_rpe_dp_tp_sp():
    """
    Feature: RotaryPositionEmbedding (MindSpore) 3-D dp×tp×sp mesh, B/N/S all sharded.
    Description:
        - 8-card 3-D mesh (dp=2, tp=2, sp=2).
        - x (Shard(0), Shard(1), Shard(2)): B by dp, N by tp, S by sp.
        - cos/sin broadcast (1, 1, S, D): dp/tp Replicate, sp Shard(2).
          Each rank's cos/sin S-slice aligns with its x S-slice.
        - float16, mode=3 (interleave-half); forward only (CANN backward not available for mode=3).
    Expectation: Distributed forward matches standalone within tolerance.
    """
    mode = 3
    x_np = np.random.randn(B, N, S, D_DIM).astype(np.float32)
    cos_np = np.random.randn(1, 1, S, D_DIM).astype(np.float32)
    sin_np = np.random.randn(1, 1, S, D_DIM).astype(np.float32)
    ref_fwd = _run_standalone_fwd(x_np, cos_np, sin_np, mode=mode)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "tp", "sp"))
    dx = distribute_tensor(_np_to_f16(x_np), mesh, (Shard(0), Shard(1), Shard(2)))
    dcos = distribute_tensor(_np_to_f16(cos_np), mesh, (Replicate(), Replicate(), Shard(2)))
    dsin = distribute_tensor(_np_to_f16(sin_np), mesh, (Replicate(), Replicate(), Shard(2)))

    _assert_fwd(_make_rpe(mode)(dx, dcos, dsin), ref_fwd, "dp_tp_sp mode=3")
