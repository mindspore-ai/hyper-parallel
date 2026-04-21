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
"""MindSpore ST for npu_mhc_post custom op."""
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import Tensor
from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.custom_ops.experimental import npu_mhc_post


def setup_module():
    """Initialize MindSpore context and communication."""
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    D.init()


np.random.seed(42)
ms.set_seed(42)
ms.set_deterministic(True)

_INPUT_NAMES = ("x", "h_res", "h_out", "h_post")


def _generate_inputs(layout="BSND"):
    """Generate random inputs for npu_mhc_post."""
    # pylint: disable=invalid-name
    BATCH = 2
    SEQ = 4
    N = 4
    D_DIM = 1280
    BATCH_SEQ = (BATCH, SEQ) if layout == "BSND" else (BATCH * SEQ,)

    _x_np = np.random.randn(*BATCH_SEQ, N, D_DIM).astype(np.float16)
    _h_res_np = np.random.randn(*BATCH_SEQ, N, N).astype(np.float32)
    _h_out_np = np.random.randn(*BATCH_SEQ, D_DIM).astype(np.float16)
    _h_post_np = np.random.randn(*BATCH_SEQ, N).astype(np.float32)
    return _x_np, _h_res_np, _h_out_np, _h_post_np


def _foward_fn(x, h_res, h_out, h_post):
    return npu_mhc_post(x, h_res, h_out, h_post).sum()




def _run_standalone(x_np, h_res_np, h_out_np, h_post_np):
    """Run npu_mhc_post forward and backward on full tensors.

    Returns:
        tuple: A pair (fwd_out, grad_out) where:
            fwd_out is the single forward output as a float32 ndarray;
            grad_out is (grad_x, grad_h_res, grad_h_out, grad_h_post).
    """
    x = Tensor(x_np)
    h_res = Tensor(h_res_np)
    h_out = Tensor(h_out_np)
    h_post = Tensor(h_post_np)

    fwd_out = npu_mhc_post(x, h_res, h_out, h_post).asnumpy().astype(np.float32)

    grad = ms.grad(_foward_fn, (0, 1, 2, 3))(x, h_res, h_out, h_post)
    grad_out = tuple(g.asnumpy().astype(np.float32) for g in grad)

    return fwd_out, grad_out


def _assert_bwd(raw_grads, d_inputs, ref_grad, tag):
    """Assert all backward gradients match the reference within tolerance."""
    for name, raw_g, d_inp, ref in zip(_INPUT_NAMES, raw_grads, d_inputs, ref_grad):
        dist_g = (
            DTensor.from_local(raw_g, d_inp.layout.mesh, d_inp.layout.alias_placements)
            .full_tensor()
            .asnumpy()
            .astype(np.float32)
        )
        assert np.allclose(dist_g, ref, atol=1e-3, rtol=1e-3), (
            f"{tag} backward grad_{name} mismatch: "
            f"max_diff={np.abs(dist_g - ref).max()}"
        )




def test_mhc_post_bsnd_replicated():
    """
    Feature: npu_mhc_post BSND forward/backward replicated on 2-device mesh.
    Description:
        - All inputs replicated; forward output and all four backward gradients
          compared against standalone.
    Expectation: Distributed outputs match standalone within tolerance.
    """
    x_np, h_res_np, h_out_np, h_post_np = _generate_inputs(layout="BSND")
    ref_out, ref_grad = _run_standalone(x_np, h_res_np, h_out_np, h_post_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))

    dx = distribute_tensor(Tensor(x_np), mesh, (Replicate(),))
    dh_res = distribute_tensor(Tensor(h_res_np), mesh, (Replicate(),))
    dh_out = distribute_tensor(Tensor(h_out_np), mesh, (Replicate(),))
    dh_post = distribute_tensor(Tensor(h_post_np), mesh, (Replicate(),))

    dist_out = npu_mhc_post(dx, dh_res, dh_out, dh_post).full_tensor().asnumpy().astype(np.float32)
    assert np.allclose(dist_out, ref_out, atol=1e-3, rtol=1e-3), (
        f"Replicated forward mismatch: max_diff={np.abs(dist_out - ref_out).max()}"
    )

    raw_grads = ms.grad(_foward_fn, (0, 1, 2, 3))(dx, dh_res, dh_out, dh_post)
    _assert_bwd(raw_grads, (dx, dh_res, dh_out, dh_post), ref_grad, "Replicated")


def test_mhc_post_bsnd_dp():
    """
    Feature: npu_mhc_post BSND forward/backward with B-dim data parallel.
    Description:
        - 2-device dp mesh; x/h_res/h_out/h_post sharded on B (dim 0).
    Expectation: full_tensor() outputs match standalone within tolerance.
    """
    x_np, h_res_np, h_out_np, h_post_np = _generate_inputs(layout="BSND")
    ref_out, ref_grad = _run_standalone(x_np, h_res_np, h_out_np, h_post_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("dp",))

    dx = distribute_tensor(Tensor(x_np), mesh, (Shard(0),))
    dh_res = distribute_tensor(Tensor(h_res_np), mesh, (Shard(0),))
    dh_out = distribute_tensor(Tensor(h_out_np), mesh, (Shard(0),))
    dh_post = distribute_tensor(Tensor(h_post_np), mesh, (Shard(0),))

    dist_out = npu_mhc_post(dx, dh_res, dh_out, dh_post).full_tensor().asnumpy().astype(np.float32)
    assert np.allclose(dist_out, ref_out, atol=1e-3, rtol=1e-3), (
        f"DP forward mismatch: max_diff={np.abs(dist_out - ref_out).max()}"
    )

    raw_grads = ms.grad(_foward_fn, (0, 1, 2, 3))(dx, dh_res, dh_out, dh_post)
    _assert_bwd(raw_grads, (dx, dh_res, dh_out, dh_post), ref_grad, "DP")


def test_mhc_post_bsnd_dp_cp_tp():
    """
    Feature: npu_mhc_post BSND forward/backward with B-dim and S-dim parallelism.
    Description:
        - 8-device dp_cp_tp mesh; x/h_res/h_out/h_post sharded on B (dim 0) and S (dim 1).
    Expectation: full_tensor() outputs match standalone within tolerance.
    """
    x_np, h_res_np, h_out_np, h_post_np = _generate_inputs(layout="BSND")
    ref_out, ref_grad = _run_standalone(x_np, h_res_np, h_out_np, h_post_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2, 2), mesh_dim_names=("dp", "cp", "tp"))

    dx = distribute_tensor(Tensor(x_np), mesh, (Shard(0), Shard(1), Shard(1)))
    dh_res = distribute_tensor(Tensor(h_res_np), mesh, (Shard(0), Shard(1), Shard(1)))
    dh_out = distribute_tensor(Tensor(h_out_np), mesh, (Shard(0), Shard(1), Shard(1)))
    dh_post = distribute_tensor(Tensor(h_post_np), mesh, (Shard(0), Shard(1), Shard(1)))

    dist_out = npu_mhc_post(dx, dh_res, dh_out, dh_post).full_tensor().asnumpy().astype(np.float32)
    assert np.allclose(dist_out, ref_out, atol=1e-3, rtol=1e-3), (
        f"BSND/DP/CP/TP forward mismatch: max_diff={np.abs(dist_out - ref_out).max()}"
    )

    raw_grads = ms.grad(_foward_fn, (0, 1, 2, 3))(dx, dh_res, dh_out, dh_post)
    _assert_bwd(raw_grads, (dx, dh_res, dh_out, dh_post), ref_grad, "BSND/DP/CP/TP")


def test_mhc_post_tnd_dp_cp_tp():
    """
    Feature: npu_mhc_post TND forward/backward with T-dim parallelism.
    Description:
        - 4-device dp_cp_tp mesh; x/h_res/h_out/h_post sharded on T (dim 0).
    Expectation: full_tensor() outputs match standalone within tolerance.
    """
    x_np, h_res_np, h_out_np, h_post_np = _generate_inputs(layout="TND")
    ref_out, ref_grad = _run_standalone(x_np, h_res_np, h_out_np, h_post_np)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "cp_tp"))

    dx = distribute_tensor(Tensor(x_np), mesh, (Shard(0), Shard(0)))
    dh_res = distribute_tensor(Tensor(h_res_np), mesh, (Shard(0), Shard(0)))
    dh_out = distribute_tensor(Tensor(h_out_np), mesh, (Shard(0), Shard(0)))
    dh_post = distribute_tensor(Tensor(h_post_np), mesh, (Shard(0), Shard(0)))

    dist_out = npu_mhc_post(dx, dh_res, dh_out, dh_post).full_tensor().asnumpy().astype(np.float32)
    assert np.allclose(dist_out, ref_out, atol=1e-3, rtol=1e-3), (
        f"TND/DP/CP_TP forward mismatch: max_diff={np.abs(dist_out - ref_out).max()}"
    )

    raw_grads = ms.grad(_foward_fn, (0, 1, 2, 3))(dx, dh_res, dh_out, dh_post)
    _assert_bwd(raw_grads, (dx, dh_res, dh_out, dh_post), ref_grad, "TND/DP/CP_TP")
