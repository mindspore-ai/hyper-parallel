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
"""Distributed tests for clip_grad_norm_ FSDP2-aligned precision."""

from collections.abc import Callable
from unittest.mock import patch

import numpy as np  # pylint: disable=C0413
import pytest
import torch  # pylint: disable=C0413
import torch.distributed as dist  # pylint: disable=C0413
from torch import nn  # pylint: disable=C0413
import torch_npu  # pylint: disable=C0413,W0611

from hyper_parallel import init_device_mesh, SkipDTensorDispatch  # pylint: disable=C0413
from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0413
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.fully_shard.api import fully_shard  # pylint: disable=C0413
from hyper_parallel.core.fully_shard.utils import (  # pylint: disable=C0413
    MixedPrecisionPolicy,
)
from hyper_parallel.core.utils import clip_grad as clip_grad_module
from hyper_parallel.core.utils.clip_grad import clip_grad_norm_  # pylint: disable=C0413

from tests.torch.common_net import DenseNet, DenseMutiLayerNet  # pylint: disable=C0413
from tests.torch.utils import init_dist  # pylint: disable=C0413


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SEED = 42
_RTOL = 1e-3
_ATOL = 1e-3
_LR = 0.01
_MAX_NORM = 1.0
_STEPS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _close(a, b):
    """Check two tensors/scalars are close within unified tolerances."""
    va = (a.cpu().detach().float().numpy()
          if isinstance(a, torch.Tensor) else float(a))
    vb = (b.cpu().detach().float().numpy()
          if isinstance(b, torch.Tensor) else float(b))
    return np.allclose(va, vb, rtol=_RTOL, atol=_ATOL)


def _get_fsdp_kwargs(mesh):
    """Standard fully_shard kwargs for tests."""
    mp = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.float32,
        output_dtype=torch.float32, cast_forward_inputs=True,
    )
    return {
        "mesh": mesh, "reshard_after_forward": True,
        "shard_placement_fn": None, "mp_policy": mp,
        "offload_policy": None, "ignored_params": None,
    }


def _gather_full(model, attr="grad"):
    """All-gather managed tensors to reconstruct full values.

    Replicated parameters are part of ``hsdp_params`` and have shard world
    size one, so their local tensor is already the global value.
    """
    result = []
    for module in model.modules():
        if not hasattr(module, "hsdp_scheduler"):
            continue
        hsdp_state = module.hsdp_scheduler.hsdp_state
        if hsdp_state is None:
            continue
        for hp in hsdp_state.hsdp_params:
            if not hp.sharded_param.requires_grad:
                continue
            if attr == "grad":
                tensor = hp.sharded_param.grad
                if tensor is None:
                    continue
            else:
                tensor = hp.sharded_param
            local = (
                tensor._local_tensor  # pylint: disable=W0212
                if isinstance(tensor, DTensor) else tensor
            )
            if hp.shard_world_size > 1:
                gathered = [
                    torch.zeros_like(local)
                    for _ in range(hp.shard_world_size)
                ]
                dist.all_gather(
                    gathered, local,
                    group=hp.mesh_info.shard_process_group,
                )
                full = torch.cat(gathered, dim=0).view(
                    hp._orig_size  # pylint: disable=W0212
                )
            else:
                full = local.clone()
            result.append(full)
    return result


def _save_grads(model):
    """Save a snapshot of every param's local grad."""
    return {
        p: p.grad.clone()
        for p in model.parameters() if p.grad is not None
    }


def _restore_grads(model, saved):
    """Restore grads from a snapshot."""
    for p in model.parameters():
        if p in saved:
            p.grad = saved[p].clone()
        else:
            p.grad = None


def _ref_clip(full_grads, max_norm, norm_type):
    """Clip full grads via nn.utils reference, return (norm, clipped)."""
    ref_params = [nn.Parameter(g) for g in full_grads]
    for rp, fg in zip(ref_params, full_grads):
        rp.grad = fg.clone()
    ref_norm = torch.nn.utils.clip_grad_norm_(
        ref_params, max_norm, norm_type=norm_type,
    )
    ref_clipped = [rp.grad.clone() for rp in ref_params]
    return ref_norm, ref_clipped


def _sgd_step(model, lr):
    """Manual SGD step on a fully_shard model."""
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            if isinstance(p, DTensor):
                grad_local = (
                    p.grad._local_tensor  # pylint: disable=W0212
                    if isinstance(p.grad, DTensor) else p.grad
                )
                p._local_tensor.sub_(  # pylint: disable=W0212
                    lr * grad_local,
                )
            else:
                p.data.sub_(lr * p.grad)


def _zero_grads(model):
    """Clear all gradients."""
    for p in model.parameters():
        p.grad = None


# ===================================================================
# Test A – 5-step training loop: FSDP2-aligned clipped grads
# ===================================================================

def test_clip_grad_norm_training_5step():  # pylint: disable=R0914
    """Run 5 training steps; verify clipped grads match nn.utils at every step.

    At each step:
      1. Forward → loss → backward
      2. Gather full grads → nn.utils clip (reference)
      3. Our clip_grad_norm_ on sharded grads
      4. Gather our clipped grads → compare against reference
      5. SGD step → zero grads

    If clipped grads match at every step, parameters evolve identically,
    so losses are guaranteed to match between HSDP and FSDP2 paths.
    """
    init_dist()

    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(4, 2),
        mesh_dim_names=("dp", "op"),
    )
    fsdp_kwargs = _get_fsdp_kwargs(mesh)
    world_size = len(mesh.rank_list)

    model = DenseNet(32, 64, has_bias=True)
    model = fully_shard(model, **fsdp_kwargs)
    model.set_reduce_op_type("sum")

    torch.manual_seed(_SEED)

    for step in range(_STEPS):
        # -- Forward + backward --
        x = torch.rand(4, 32).npu()
        with SkipDTensorDispatch():
            out = model(x)
            loss = out.sum()
            loss.backward(torch.tensor(1.0 / world_size).npu())

        # -- Reference: clip full grads via nn.utils (= FSDP1 path) --
        saved = _save_grads(model)
        full_grads = _gather_full(model, "grad")
        ref_norm, ref_clipped = _ref_clip(full_grads, _MAX_NORM, 2.0)

        # -- Our clip on sharded grads --
        _restore_grads(model, saved)
        with SkipDTensorDispatch():
            our_norm = clip_grad_norm_(model, _MAX_NORM, norm_type=2.0)

        # -- Check 1: return value == global norm (torchtitan-aligned) --
        assert _close(our_norm, ref_norm), (
            f"Step {step} return norm mismatch vs torch reference: "
            f"got={our_norm.item():.6e}, "
            f"ref={ref_norm.item():.6e}"
        )

        # -- Check 2: clipped grads match torch reference --
        our_clipped = _gather_full(model, "grad")
        for i, (rg, og) in enumerate(zip(ref_clipped, our_clipped)):
            assert _close(og, rg), (
                f"Step {step} grad[{i}] mismatch: "
                f"max_diff={torch.max(torch.abs(og - rg)).item():.2e}"
            )

        # -- SGD step + zero grads --
        _sgd_step(model, _LR)
        _zero_grads(model)


# ===================================================================
# Test B – main_grad support (apply_grad_on_fp32_main_grad)
# ===================================================================

def _check_main_grad():  # pylint: disable=R0914
    """Verify clip_grad_norm_ reads param.main_grad when param.grad is None.

    Simulates ``apply_grad_on_fp32_main_grad=True`` where the training
    loop stores the real gradient in ``param.main_grad`` and sets
    ``param.grad = None``.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(4, 2),
        mesh_dim_names=("dp", "op"),
    )
    fsdp_kwargs = _get_fsdp_kwargs(mesh)
    world_size = len(mesh.rank_list)

    model = DenseNet(32, 64, has_bias=True)
    model = fully_shard(model, **fsdp_kwargs)
    model.set_reduce_op_type("sum")

    torch.manual_seed(_SEED)
    x = torch.rand(4, 32).npu()
    with SkipDTensorDispatch():
        out = model(x)
        loss = out.sum()
        loss.backward(torch.tensor(1.0 / world_size).npu())

    # Gather full grads BEFORE moving to main_grad
    full_grads = _gather_full(model, "grad")

    # Move grads from .grad to .main_grad (simulates fp32 main_grad path)
    for p in model.parameters():
        if p.grad is not None:
            p.main_grad = p.grad.clone()
            p.grad = None

    # Our clip — should read main_grad
    max_norm = 0.01
    with SkipDTensorDispatch():
        our_norm = clip_grad_norm_(model, max_norm, norm_type=2.0)

    # Reference clip on full grads
    ref_norm, ref_clipped = _ref_clip(full_grads, max_norm, 2.0)

    # Check 1: return value == global norm (torchtitan-aligned)
    assert _close(our_norm, ref_norm), (
        f"main_grad return norm mismatch: "
        f"got={our_norm.item():.6e}, "
        f"ref={ref_norm.item():.6e}"
    )

    # Verify clipped main_grad values match reference
    for p in model.parameters():
        if hasattr(p, "main_grad") and p.main_grad is not None:
            p.grad = p.main_grad
    our_clipped = _gather_full(model, "grad")
    for i, (rg, og) in enumerate(zip(ref_clipped, our_clipped, strict=True)):
        assert _close(og, rg), f"main_grad clipped grad[{i}] mismatch"

    # Cleanup
    for p in model.parameters():
        if hasattr(p, "main_grad"):
            del p.main_grad
        p.grad = None


# ===================================================================
# Test C – Frozen params
# ===================================================================

def _check_frozen_params():
    """Verify clip_grad_norm_ correctly skips frozen (requires_grad=False) params."""
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(4, 2),
        mesh_dim_names=("dp", "op"),
    )
    fsdp_kwargs = _get_fsdp_kwargs(mesh)
    world_size = len(mesh.rank_list)

    model = DenseNet(32, 64, has_bias=True)
    model = fully_shard(model, **fsdp_kwargs)
    model.set_reduce_op_type("sum")

    torch.manual_seed(_SEED)
    x = torch.rand(4, 32).npu()
    with SkipDTensorDispatch():
        out = model(x)
        loss = out.sum()
        loss.backward(torch.tensor(1.0 / world_size).npu())

    # Freeze the first param
    params = list(model.parameters())
    assert len(params) >= 2, "Need at least 2 params for freeze test"

    # Gather full grads of unfrozen params (skip first)
    unfrozen_full_grads = _gather_full(model, "grad")[1:]

    saved_grad = params[0].grad
    params[0].grad = None
    params[0].requires_grad_(False)

    max_norm = 0.01
    with SkipDTensorDispatch():
        our_norm = clip_grad_norm_(model, max_norm, norm_type=2.0)

    # Reference: clip only unfrozen grads
    ref_norm, ref_clipped = _ref_clip(unfrozen_full_grads, max_norm, 2.0)

    # Check 1: return value == global norm (torchtitan-aligned)
    assert _close(our_norm, ref_norm), (
        f"frozen return norm mismatch: "
        f"got={our_norm.item():.6e}, "
        f"ref={ref_norm.item():.6e}"
    )

    # Check 2: clipped grads match torch reference
    our_clipped = _gather_full(model, "grad")
    for i, (rg, og) in enumerate(zip(ref_clipped, our_clipped, strict=True)):
        assert _close(og, rg), f"Frozen: clipped grad[{i}] mismatch"

    # Restore
    params[0].requires_grad_(True)
    params[0].grad = saved_grad


# ===================================================================
# Test D-replicate – Mixed FSDP-sharded + replicate_params over-count
# ===================================================================

def _check_replicate_params():  # pylint: disable=R0914
    """Replicate-grad norms must not be all-reduced over the shard group.

    Regression: ``_total_norm_fsdp2_aligned`` previously stacked
    per-tensor norms across ALL grad groups and did ONE all_reduce
    over the first non-empty group's shard dims. Replicated grads
    (already identical on every shard rank) were counted once per shard
    rank instead of once in the global norm.

    Setup chooses every layer's bias as ``replicate_params`` so the
    replicate contribution is large enough to fail the old code but
    small enough that the weights still dominate.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(8,),
        mesh_dim_names=("dp",),
    )
    fsdp_kwargs = _get_fsdp_kwargs(mesh)
    world_size = len(mesh.rank_list)

    torch.manual_seed(_SEED)

    model = DenseMutiLayerNet(64, layer_num=4, has_bias=True)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.1)

    # Biases stay replicated; weights go through the FSDP shard path.
    fsdp_kwargs["replicate_params"] = {
        layer.bias for layer in model.layers
    }

    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    model = fully_shard(model, **fsdp_kwargs)
    model.set_reduce_op_type("sum")

    x = torch.rand(4, 64).npu()
    with SkipDTensorDispatch():
        out = model(x)
        loss = out
        loss.backward(torch.tensor(1.0 / world_size).npu())

    saved = _save_grads(model)
    full_grads = _gather_full(model, "grad")
    ref_norm, ref_clipped = _ref_clip(full_grads, _MAX_NORM, 2.0)

    _restore_grads(model, saved)
    with SkipDTensorDispatch():
        our_norm = clip_grad_norm_(model, _MAX_NORM, norm_type=2.0)

    rank = dist.get_rank()
    assert _close(our_norm, ref_norm), (
        f"[rank={rank}] replicate over-count regression: "
        f"got={our_norm.item():.6e}, ref={ref_norm.item():.6e}, "
        f"shard_world_size={world_size}"
    )

    our_clipped = _gather_full(model, "grad")
    for i, (rg, og) in enumerate(zip(ref_clipped, our_clipped, strict=True)):
        assert _close(og, rg), (
            f"[rank={rank}] replicate clipped grad[{i}] mismatch: "
            f"max_diff={torch.max(torch.abs(og - rg)).item():.2e}"
        )


# ===================================================================
# Test D – Multi-grad-group parameter ordering
# ===================================================================

def _check_multi_group():  # pylint: disable=R0914
    """Verify stack order stability across multiple grad groups.

    Regression guard for param / grad_group iteration order: when
    parameters span multiple fully_shard units (multiple grad_groups),
    norm stacking must follow original parameter order, not
    grad_group iteration order. Float32 addition is non-associative,
    so different orders can produce 1-ULP diffs on non-rank-0 ranks.

    Check on all ranks explicitly — the regression does not manifest
    on rank 0 in general.
    """
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(4, 2),
        mesh_dim_names=("dp", "op"),
    )
    fsdp_kwargs = _get_fsdp_kwargs(mesh)
    world_size = len(mesh.rank_list)

    torch.manual_seed(_SEED)

    # Multi-layer net (8 layers × weight+bias = 16 params) creates
    # multiple grad_groups, exercising the param-order regression path.
    # Randomize per-param weights so per-param norms differ across layers.
    model = DenseMutiLayerNet(64, layer_num=8, has_bias=True)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    model = fully_shard(model, **fsdp_kwargs)
    model.set_reduce_op_type("sum")

    x = torch.rand(4, 64).npu()
    with SkipDTensorDispatch():
        out = model(x)
        loss = out
        loss.backward(torch.tensor(1.0 / world_size).npu())

    saved = _save_grads(model)
    full_grads = _gather_full(model, "grad")
    ref_norm, ref_clipped = _ref_clip(full_grads, _MAX_NORM, 2.0)

    _restore_grads(model, saved)
    with SkipDTensorDispatch():
        our_norm = clip_grad_norm_(model, _MAX_NORM, norm_type=2.0)

    rank = dist.get_rank()

    # Check 1: return value == torch reference (FSDP2-aligned global norm)
    assert _close(our_norm, ref_norm), (
        f"[rank={rank}] multi-group norm mismatch: "
        f"got={our_norm.item():.6e}, ref={ref_norm.item():.6e}"
    )

    # Check 2: clipped grads match torch reference on ALL ranks
    our_clipped = _gather_full(model, "grad")
    for i, (rg, og) in enumerate(zip(ref_clipped, our_clipped, strict=True)):
        assert _close(og, rg), (
            f"[rank={rank}] multi-group grad[{i}] mismatch: "
            f"max_diff={torch.max(torch.abs(og - rg)).item():.2e}"
        )


def _check_shared_shard_group_order() -> None:  # pylint: disable=too-many-locals
    """Watch the actual norm input order and collective count for two mesh objects."""
    meshes = (
        init_device_mesh("npu", (4, 2), mesh_dim_names=("dp", "op")),
        init_device_mesh("npu", (4, 2), mesh_dim_names=("replica", "shard")),
    )
    assert meshes[0] is not meshes[1], f"Expected distinct meshes, got {meshes}"
    shard = meshes[0].get_local_rank("op")
    full_grads = [torch.arange(1, 5, device="npu", dtype=torch.float32) * factor for factor in (1, 8, 2, 4)]
    reference_norm, reference_grads = _ref_clip(full_grads, _MAX_NORM, 2.0)
    parameters = []
    local_grads = []
    placements = (Replicate(), Shard(0))
    for index, full in enumerate(full_grads):
        mesh = meshes[index % len(meshes)]
        local = full.chunk(2)[shard].clone()
        param = nn.Parameter(DTensor.from_local(torch.zeros_like(local), mesh, placements))
        param.grad = DTensor.from_local(local, mesh, placements)
        parameters.append(param)
        local_grads.append(param.grad.to_local())

    # Norm rounding alone cannot reliably detect a permuted stack on every backend.
    per_tensor_norms = clip_grad_module._per_tensor_norms  # pylint: disable=protected-access
    with (
        patch.object(clip_grad_module, "_per_tensor_norms", wraps=per_tensor_norms) as norms,
        patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce,
        SkipDTensorDispatch(),
    ):
        actual_norm = clip_grad_norm_(parameters, _MAX_NORM)
    assert norms.call_count == 1, f"Expected one norm stack, got {norms.call_count}"
    order = [grad.data_ptr() for grad in norms.call_args.args[0]]
    expected_order = [grad.data_ptr() for grad in local_grads]
    assert order == expected_order, f"Norm input order {order}, expected {expected_order}"
    assert reduce.call_count == 1, f"Expected one shard norm all-reduce, got {reduce.call_count}"
    reduced_ranks = dist.get_process_group_ranks(reduce.call_args.kwargs["group"])
    expected_ranks = dist.get_process_group_ranks(meshes[0].get_group("op"))
    assert reduced_ranks == expected_ranks, f"Reduced over {reduced_ranks}, expected {expected_ranks}"
    torch.testing.assert_close(actual_norm, reference_norm, rtol=1e-6, atol=1e-6)
    for local, expected in zip(local_grads, reference_grads, strict=True):
        torch.testing.assert_close(local, expected.chunk(2)[shard], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("check", [
    pytest.param(_check_main_grad, id="main-grad"),
    pytest.param(_check_frozen_params, id="frozen"),
    pytest.param(_check_replicate_params, id="replicate"),
    pytest.param(_check_multi_group, id="multi-group"),
    pytest.param(_check_shared_shard_group_order, id="shared-shard-group-order"),
])
def test_clip_grad_norm_edge_cases(check: Callable[[], None]) -> None:
    """Run each gradient path against the Torch reference on the same eight ranks."""
    init_dist()
    check()
    dist.barrier()
