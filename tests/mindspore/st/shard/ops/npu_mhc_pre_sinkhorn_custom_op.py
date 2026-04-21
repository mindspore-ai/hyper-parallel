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
"""MindSpore ST for npu_mhc_pre_sinkhorn custom op."""
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate, Partial
from hyper_parallel.custom_ops.experimental import npu_mhc_pre_sinkhorn


def setup_module():
    """Initialize MindSpore context and communication."""
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    D.init()


np.random.seed(42)
ms.set_seed(42)
ms.set_deterministic(True)

_FWD_NAMES = ("h_in", "h_post", "h_res")
_INPUT_NAMES = ("x", "phi", "alpha", "bias")


def _generate_inputs(layout="BSND"):
    """Generate random inputs for npu_mhc_pre_sinkhorn."""
    # pylint: disable=invalid-name
    BATCH = 2
    SEQ = 4
    N = 4
    C = 1280
    FUSION_SIZE = N * N + 2 * N  # 24
    BATCH_SEQ = (BATCH, SEQ) if layout == "BSND" else (BATCH * SEQ,)

    _x_np = np.random.randn(*BATCH_SEQ, N, C).astype(np.float16)
    _phi_np = np.random.randn(FUSION_SIZE, N * C).astype(np.float32)
    _alpha_np = np.random.randn(3).astype(np.float32)
    _bias_np = np.random.randn(FUSION_SIZE).astype(np.float32)
    return _x_np, _phi_np, _alpha_np, _bias_np


def _foward_fn(x, phi, alpha, bias):
    return npu_mhc_pre_sinkhorn(x, phi, alpha, bias)[0].sum()




def _run_standalone(x_np, phi_np, alpha_np, bias_np):
    """Run npu_mhc_pre_sinkhorn forward and backward on full tensors.

    Returns:
        tuple: A pair (fwd_out, grad_out) where:
            fwd_out is (h_in, h_post, h_res) — first three forward outputs as float32 ndarrays;
            grad_out is (grad_x, grad_phi, grad_alpha, grad_bias).
    """
    x = Tensor(x_np, ms.bfloat16)
    phi = Tensor(phi_np)
    alpha = Tensor(alpha_np)
    bias = Tensor(bias_np)

    out = npu_mhc_pre_sinkhorn(x, phi, alpha, bias)
    fwd_out = tuple(out[i].astype(ms.float32).asnumpy() for i in range(len(_FWD_NAMES)))

    grad = ms.grad(_foward_fn, (0, 1, 2, 3))(x, phi, alpha, bias)
    grad_out = tuple(g.astype(ms.float32).asnumpy() for g in grad)

    return fwd_out, grad_out


def _get_grad_placements(d_inputs: tuple) -> list:
    """Derive the correct DTensor placement for each input's gradient."""
    n_dims = len(d_inputs[0].layout.placements)
    dim_has_shard = [
        any(isinstance(d_inp.layout.placements[d], Shard) for d_inp in d_inputs)
        for d in range(n_dims)
    ]
    result = []
    for d_inp in d_inputs:
        mesh_placements = d_inp.layout.placements
        grad_placements = tuple(
            Partial("sum") if isinstance(p, Replicate) and dim_has_shard[dim_idx] else p
            for dim_idx, p in enumerate(mesh_placements)
        )
        result.append(grad_placements)
    return result


def _assert_fwd(results, ref_out, tag):
    """Assert all forward outputs match the reference within tolerance."""
    for i, (name, ref) in enumerate(zip(_FWD_NAMES, ref_out)):
        dist = results[i].full_tensor().astype(ms.float32).asnumpy()
        assert np.allclose(dist, ref, atol=1e-3, rtol=1e-3), (
            f"{tag} forward {name} mismatch: "
            f"max_diff={np.abs(dist - ref).max()}"
        )


def _assert_bwd(raw_grads, d_inputs, ref_grad, tag):
    """Assert all backward gradients match the reference within tolerance."""
    grad_placements = _get_grad_placements(d_inputs)
    for name, raw_g, d_inp, ref, g_placements in zip(
            _INPUT_NAMES, raw_grads, d_inputs, ref_grad, grad_placements):
        dist_g = (
            DTensor.from_local(raw_g, d_inp.layout.mesh, g_placements)
            .full_tensor()
            .astype(ms.float32)
            .asnumpy()
        )
        assert np.allclose(dist_g, ref, atol=1e-3, rtol=1e-3), (
            f"{tag} backward grad_{name} mismatch: "
            f"max_diff={np.abs(dist_g - ref).max()}"
        )




def test_mhc_pre_sinkhorn_bsnd_replicated():
    """
    Feature: npu_mhc_pre_sinkhorn BSND forward/backward replicated on 2-device mesh.
    Description:
        - All inputs replicated; all three forward outputs and all four backward gradients
          compared against standalone single-card results.
    Expectation: Distributed outputs match standalone within tolerance.
    """
    x_np, phi_np, alpha_np, bias_np = _generate_inputs(layout="BSND")
    ref_out, ref_grad = _run_standalone(x_np, phi_np, alpha_np, bias_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))

    dx = distribute_tensor(Tensor(x_np, ms.bfloat16), mesh, (Replicate(),))
    dphi = distribute_tensor(Tensor(phi_np), mesh, (Replicate(),))
    dalpha = distribute_tensor(Tensor(alpha_np), mesh, (Replicate(),))
    dbias = distribute_tensor(Tensor(bias_np), mesh, (Replicate(),))

    results = npu_mhc_pre_sinkhorn(dx, dphi, dalpha, dbias)
    _assert_fwd(results, ref_out, "Replicated")

    raw_grads = ms.grad(_foward_fn, (0, 1, 2, 3))(dx, dphi, dalpha, dbias)
    _assert_bwd(raw_grads, (dx, dphi, dalpha, dbias), ref_grad, "Replicated")


def test_mhc_pre_sinkhorn_bsnd_dp():
    """
    Feature: npu_mhc_pre_sinkhorn BSND forward/backward with B-dim data parallel.
    Description:
        - 2-device dp mesh; x sharded on B (dim 0); phi/alpha/bias replicated.
    Expectation: full_tensor() outputs match standalone within tolerance.
    """
    x_np, phi_np, alpha_np, bias_np = _generate_inputs(layout="BSND")
    ref_out, ref_grad = _run_standalone(x_np, phi_np, alpha_np, bias_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))

    dx = distribute_tensor(Tensor(x_np, ms.bfloat16), mesh, (Shard(0),))
    dphi = distribute_tensor(Tensor(phi_np), mesh, (Replicate(),))
    dalpha = distribute_tensor(Tensor(alpha_np), mesh, (Replicate(),))
    dbias = distribute_tensor(Tensor(bias_np), mesh, (Replicate(),))

    results = npu_mhc_pre_sinkhorn(dx, dphi, dalpha, dbias)
    _assert_fwd(results, ref_out, "DP")

    raw_grads = ms.grad(_foward_fn, (0, 1, 2, 3))(dx, dphi, dalpha, dbias)
    _assert_bwd(raw_grads, (dx, dphi, dalpha, dbias), ref_grad, "DP")


# def test_mhc_pre_sinkhorn_tnd_dp_cp_tp():
#     """Temporarily commented out - TND dp_cp_tp case for npu_mhc_pre_sinkhorn."""
#     pass


def test_mhc_pre_sinkhorn_bsnd_dp_cp_tp():
    """
    Feature: npu_mhc_pre_sinkhorn BSND forward/backward with data parallel on B, cp_tp on S.
    Description:
        - 8-device dp mesh; x sharded on B and S; phi/alpha/bias replicated.
    Expectation: full_tensor() outputs match standalone within tolerance.
    """
    x_np, phi_np, alpha_np, bias_np = _generate_inputs(layout="BSND")
    ref_out, ref_grad = _run_standalone(x_np, phi_np, alpha_np, bias_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "tp"))

    dx = distribute_tensor(Tensor(x_np, ms.bfloat16), mesh, (Shard(0), Shard(1), Shard(1)))
    dphi = distribute_tensor(Tensor(phi_np), mesh, (Replicate(), Replicate(), Replicate()))
    dalpha = distribute_tensor(Tensor(alpha_np), mesh, (Replicate(), Replicate(), Replicate()))
    dbias = distribute_tensor(Tensor(bias_np), mesh, (Replicate(), Replicate(), Replicate()))

    results = npu_mhc_pre_sinkhorn(dx, dphi, dalpha, dbias)
    _assert_fwd(results, ref_out, "DP/CP/TP")

    raw_grads = ms.grad(_foward_fn, (0, 1, 2, 3))(dx, dphi, dalpha, dbias)
    _assert_bwd(raw_grads, (dx, dphi, dalpha, dbias), ref_grad, "DP/CP/TP")
