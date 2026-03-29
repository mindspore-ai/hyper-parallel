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
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import numpy as np  # pylint: disable=C0413
import torch  # pylint: disable=C0413
import torch.distributed as dist  # pylint: disable=C0413
from torch import nn  # pylint: disable=C0413
import torch_npu  # pylint: disable=C0413,W0611

from hyper_parallel import init_device_mesh, SkipDTensorDispatch  # pylint: disable=C0413
from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0413
from hyper_parallel.core.fully_shard.api import fully_shard  # pylint: disable=C0413
from hyper_parallel.core.utils.clip_grad import clip_grad_norm_  # pylint: disable=C0413
from hyper_parallel.core.fully_shard.utils import (  # pylint: disable=C0413
    MixedPrecisionPolicy,
)

from tests.torch.common_net import DenseNet  # pylint: disable=C0413
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
    """All-gather sharded tensors to reconstruct full values."""
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
            if hp.is_sharded:
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


def _local_norm(model, norm_type=2.0):
    """Compute local combined norm from this rank's grad shards.

    Matches the return value of clip_grad_norm_ for finite p-norms,
    which returns the local combined norm (pre-eps, pre-reduce)
    to match FSDP2's _NormPartial._local_tensor behavior.
    """
    norms = []
    for p in model.parameters():
        grad = getattr(p, "main_grad", None)
        if grad is None:
            grad = p.grad
        if grad is None:
            continue
        local_g = (
            grad._local_tensor  # pylint: disable=W0212
            if isinstance(grad, DTensor) else grad
        )
        norms.append(torch.linalg.vector_norm(local_g, norm_type))
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms), norm_type)


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

        # -- Reference: clip full grads via nn.utils --
        saved = _save_grads(model)
        full_grads = _gather_full(model, "grad")
        _, ref_clipped = _ref_clip(full_grads, _MAX_NORM, 2.0)

        # -- Our clip on sharded grads --
        _restore_grads(model, saved)
        with SkipDTensorDispatch():
            clip_grad_norm_(model, _MAX_NORM, norm_type=2.0)

        # -- Precision check: clipped grads must match --
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

def test_clip_grad_norm_main_grad():  # pylint: disable=R0914
    """Verify clip_grad_norm_ reads param.main_grad when param.grad is None.

    Simulates ``apply_grad_on_fp32_main_grad=True`` where the training
    loop stores the real gradient in ``param.main_grad`` and sets
    ``param.grad = None``.
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
        clip_grad_norm_(model, max_norm, norm_type=2.0)

    # Reference clip on full grads
    _, ref_clipped = _ref_clip(full_grads, max_norm, 2.0)

    # Verify clipped main_grad values match reference
    for p in model.parameters():
        if hasattr(p, "main_grad") and p.main_grad is not None:
            p.grad = p.main_grad
    our_clipped = _gather_full(model, "grad")
    for i, (rg, og) in enumerate(zip(ref_clipped, our_clipped)):
        assert _close(og, rg), f"main_grad clipped grad[{i}] mismatch"

    # Cleanup
    for p in model.parameters():
        if hasattr(p, "main_grad"):
            del p.main_grad
        p.grad = None


# ===================================================================
# Test C – Frozen params
# ===================================================================

def test_clip_grad_norm_frozen_params():
    """Verify clip_grad_norm_ correctly skips frozen (requires_grad=False) params."""
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
        clip_grad_norm_(model, max_norm, norm_type=2.0)

    # Reference: clip only unfrozen grads
    _, ref_clipped = _ref_clip(unfrozen_full_grads, max_norm, 2.0)

    # Verify clipped unfrozen grads match
    our_clipped = _gather_full(model, "grad")
    for i, (rg, og) in enumerate(zip(ref_clipped, our_clipped)):
        assert _close(og, rg), f"Frozen: clipped grad[{i}] mismatch"

    # Restore
    params[0].requires_grad_(True)
    params[0].grad = saved_grad
