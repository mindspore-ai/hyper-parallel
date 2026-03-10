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
"""E2E tests for distributed clip_grad_norm_ API."""
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import numpy as np  # pylint: disable=C0413
import torch  # pylint: disable=C0413
import torch.distributed as dist  # pylint: disable=C0413
from torch import nn  # pylint: disable=C0413
import torch_npu  # pylint: disable=C0413,W0611

from hyper_parallel import init_device_mesh, SkipDTensorDispatch  # pylint: disable=C0413
from hyper_parallel.core.dtensor import DTensor  # pylint: disable=C0413
from hyper_parallel.core.fully_shard.api import fully_shard  # pylint: disable=C0413
from hyper_parallel.core.placement_types import Partial, Shard  # pylint: disable=C0413
from hyper_parallel.core.utils.clip_grad import clip_grad_norm_  # pylint: disable=C0413
from hyper_parallel.platform.torch.fully_shard.utils import (  # pylint: disable=C0413
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


def _build_model_and_backward(mesh, fsdp_kwargs):
    """Build fully_shard model, run forward+backward, return model."""
    model = DenseNet(32, 64, has_bias=True)
    model = fully_shard(model, **fsdp_kwargs)
    model.set_reduce_op_type("sum")
    torch.manual_seed(_SEED)
    x = torch.rand(4, 32).npu()
    with SkipDTensorDispatch():
        out = model(x)
        loss = out.sum()
        loss.backward(torch.tensor(1.0 / len(mesh.rank_list)).npu())
    return model


def _gather_full(model, attr="grad"):
    """All-gather sharded tensors to reconstruct full values.

    Args:
        model: A fully_shard model.
        attr: ``"grad"`` for gradients, ``"data"`` for parameters.
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


def _assert_ranks_agree(norm_val, label):
    """Verify every rank computed the same total_norm."""
    t = norm_val.clone().float()
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    for i, n in enumerate(gathered):
        assert _close(n, gathered[0]), (
            f"[{label}] rank-{i} norm {n.item():.6f} != "
            f"rank-0 norm {gathered[0].item():.6f}"
        )


# ===================================================================
# Test A – Comprehensive semantics + E2E SGD closed loop
# ===================================================================

def test_clip_grad_norm_comprehensive():  # pylint: disable=R0914,R0915
    """Comprehensive semantics: norm parity, grad parity, E2E SGD step.

    Three representative combos (not Cartesian product):
      1. HSDP 2D (4,2) + L2   + model API            + E2E SGD step
      2. HSDP 2D (4,2) + Linf + model.parameters() API
      3. FSDP 1D (8,)  + L2   + model API

    Every combo verifies total_norm and clipped-gradient parity with
    torch.nn.utils.clip_grad_norm_ on all-gathered full gradients.
    Combo 1 additionally verifies post-SGD-step parameter parity.
    """
    init_dist()

    # ---- HSDP 2D (Replicate + Shard) ----
    mesh_2d = init_device_mesh(
        device_type="npu", mesh_shape=(4, 2),
        mesh_dim_names=("dp", "op"),
    )

    # -- Combo 1: HSDP + L2 + model + E2E SGD step --
    model = _build_model_and_backward(mesh_2d, _get_fsdp_kwargs(mesh_2d))
    saved = _save_grads(model)
    full_grads = _gather_full(model, "grad")

    _restore_grads(model, saved)
    ref_norm, ref_clipped = _ref_clip(full_grads, 0.01, 2.0)
    assert ref_norm > 0.01, (
        f"Setup: L2 norm {ref_norm.item():.4f} must exceed max_norm=0.01"
    )
    with SkipDTensorDispatch():
        our_norm = clip_grad_norm_(model, 0.01, norm_type=2.0)
    assert _close(our_norm, ref_norm), (
        f"Combo1 norm: {our_norm.item():.6f} vs {ref_norm.item():.6f}"
    )
    _assert_ranks_agree(our_norm, "Combo1-L2")
    our_clipped = _gather_full(model, "grad")
    for i, (rg, og) in enumerate(zip(ref_clipped, our_clipped)):
        assert _close(og, rg), f"Combo1 grad[{i}] mismatch"

    # E2E closed loop: manual SGD step on sharded model, verify full params
    full_params_before = _gather_full(model, "data")
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            # In HSDP backward, p is DTensor but p.grad is plain
            # local-shard tensor.  Extract local tensors explicitly.
            if isinstance(p, DTensor):
                grad_local = (
                    p.grad._local_tensor  # pylint: disable=W0212
                    if isinstance(p.grad, DTensor) else p.grad
                )
                p._local_tensor.sub_(  # pylint: disable=W0212
                    _LR * grad_local,
                )
            else:
                p.data.sub_(_LR * p.grad)
    full_params_after = _gather_full(model, "data")
    for i, (before, clipped, after) in enumerate(
        zip(full_params_before, ref_clipped, full_params_after),
    ):
        expected = before - _LR * clipped
        assert _close(after, expected), (
            f"Combo1 E2E param[{i}] mismatch"
        )

    # -- Combo 2: HSDP + Linf + parameters() API --
    model = _build_model_and_backward(mesh_2d, _get_fsdp_kwargs(mesh_2d))
    saved = _save_grads(model)
    full_grads = _gather_full(model, "grad")

    _restore_grads(model, saved)
    ref_norm, ref_clipped = _ref_clip(full_grads, 0.01, float("inf"))
    assert ref_norm > 0.01, "Setup: clipping must trigger"
    with SkipDTensorDispatch():
        our_norm = clip_grad_norm_(
            model.parameters(), 0.01, norm_type=float("inf"),
        )
    assert _close(our_norm, ref_norm), (
        f"Combo2 norm: {our_norm.item():.6f} vs {ref_norm.item():.6f}"
    )
    _assert_ranks_agree(our_norm, "Combo2-Linf")
    our_clipped = _gather_full(model, "grad")
    for i, (rg, og) in enumerate(zip(ref_clipped, our_clipped)):
        assert _close(og, rg), f"Combo2 grad[{i}] mismatch"

    # ---- FSDP 1D (pure Shard, 8 ranks) ----
    mesh_1d = init_device_mesh(
        device_type="npu", mesh_shape=(8,),
        mesh_dim_names=("dp",),
    )

    # -- Combo 3: FSDP 1D + L2 + model API --
    model = _build_model_and_backward(mesh_1d, _get_fsdp_kwargs(mesh_1d))
    saved = _save_grads(model)
    full_grads = _gather_full(model, "grad")

    _restore_grads(model, saved)
    ref_norm, ref_clipped = _ref_clip(full_grads, 0.5, 2.0)
    assert ref_norm > 0.5, "Setup: clipping must trigger"
    with SkipDTensorDispatch():
        our_norm = clip_grad_norm_(model, 0.5, norm_type=2.0)
    assert _close(our_norm, ref_norm), (
        f"Combo3 norm: {our_norm.item():.6f} vs {ref_norm.item():.6f}"
    )
    _assert_ranks_agree(our_norm, "Combo3-L2")
    our_clipped = _gather_full(model, "grad")
    for i, (rg, og) in enumerate(zip(ref_clipped, our_clipped)):
        assert _close(og, rg), f"Combo3 grad[{i}] mismatch"


# ===================================================================
# Test B – TP+FSDP: multi-shard-dim with Partial pre-reduction
# ===================================================================

def test_clip_grad_norm_partial_shard():  # pylint: disable=R0914,R0915
    """Verify clip_grad_norm_ with Partial + Shard placements (TP+FSDP).

    Manually constructs DTensor parameters with
    ``[Partial("sum"), Shard(0)]`` placements on a (tp=2, dp=4) mesh
    to simulate a TP+FSDP scenario without requiring full TP training
    infrastructure.

    Sub-sections:
      (a) Single param with Partial("sum") + Shard — norm & clip parity
      (b) Multi Partial params in same coalesce group — norm parity
      (c) Partial("avg") placement — AVG fallback norm parity

    The Partial dimension models TP's output gradient (partial sum),
    the Shard dimension models FSDP's weight sharding.

    Verifies:
      - total_norm matches nn.utils reference on the full gradient
      - clipped gradients match reference after accounting for
        Partial reduction and Shard gathering
      - all ranks agree on total_norm
    """
    init_dist()
    device = torch.device("npu")

    # ---- Mesh: (tp=2, dp=4) → 8 ranks ----
    mesh = init_device_mesh(
        device_type="npu", mesh_shape=(2, 4),
        mesh_dim_names=("tp", "dp"),
    )
    tp_size = 2
    dp_size = 4

    rank = dist.get_rank()
    # Mesh layout [[0,1,2,3],[4,5,6,7]]:
    #   dim 0 (tp): groups {0,4},{1,5},{2,6},{3,7}
    #   dim 1 (dp): groups {0,1,2,3},{4,5,6,7}
    dp_rank = rank % dp_size

    # ---- Deterministic full gradient (same on all ranks) ----
    torch.manual_seed(_SEED)
    full_grad = torch.randn(32, 64, device=device)

    # Each dp rank holds a contiguous row-shard of the full gradient.
    shard_rows = 32 // dp_size  # 8
    full_shard = full_grad[
        dp_rank * shard_rows : (dp_rank + 1) * shard_rows
    ].clone()

    # Partial(SUM): each tp rank holds shard / tp_size so that
    # SUM across tp group = full_shard.
    local_partial = full_shard / tp_size

    # ---- Build DTensor param + grad ----
    full_weight = torch.ones(32, 64, device=device)
    local_weight = full_weight[
        dp_rank * shard_rows : (dp_rank + 1) * shard_rows
    ].clone()

    placements = [Partial("sum"), Shard(0)]
    param = nn.Parameter(
        DTensor.from_local(local_weight, mesh, placements),
    )
    param.grad = DTensor.from_local(
        local_partial, mesh, placements,
    )

    # ---- Our clip_grad_norm_ ----
    max_norm = 0.01
    with SkipDTensorDispatch():
        our_norm = clip_grad_norm_([param], max_norm, norm_type=2.0)

    # ---- Reference: nn.utils on full (unreduced, ungathered) grad ----
    ref_param = nn.Parameter(full_weight.clone())
    ref_param.grad = full_grad.clone()
    ref_norm = torch.nn.utils.clip_grad_norm_(
        [ref_param], max_norm, norm_type=2.0,
    )
    ref_clipped = ref_param.grad.clone()

    assert ref_norm > max_norm, (
        f"Setup: norm {ref_norm.item():.4f} must exceed max_norm"
    )
    assert _close(our_norm, ref_norm), (
        f"TP+FSDP norm: {our_norm.item():.6f} vs {ref_norm.item():.6f}"
    )
    _assert_ranks_agree(our_norm, "TP+FSDP-Partial")

    # ---- Verify clipped gradient ----
    # After clip, param.grad._local_tensor holds the clipped local
    # partial shard.  To compare against the reference, extract the
    # corresponding shard of ref_clipped and divide by tp_size.
    clipped_local = (
        param.grad._local_tensor  # pylint: disable=W0212
        if isinstance(param.grad, DTensor) else param.grad
    )
    ref_shard = ref_clipped[
        dp_rank * shard_rows : (dp_rank + 1) * shard_rows
    ]
    expected_local = ref_shard / tp_size
    assert _close(clipped_local, expected_local), (
        f"TP+FSDP clipped grad mismatch on rank {rank}"
    )

    # ---- (b) Multi Partial params in same coalesce group ----
    # Two params with identical (mesh, partial_info) are coalesced
    # into one buffer.  Verify the total norm matches nn.utils on
    # the concatenated full gradients.
    torch.manual_seed(_SEED + 10)
    full_grad_b1 = torch.randn(32, 64, device=device)
    full_grad_b2 = torch.randn(16, 32, device=device)
    shard_rows_b2 = 16 // dp_size  # 4

    local_partial_b1 = full_grad_b1[
        dp_rank * shard_rows : (dp_rank + 1) * shard_rows
    ].clone() / tp_size
    local_partial_b2 = full_grad_b2[
        dp_rank * shard_rows_b2 : (dp_rank + 1) * shard_rows_b2
    ].clone() / tp_size

    w_b1 = torch.ones(shard_rows, 64, device=device)
    w_b2 = torch.ones(shard_rows_b2, 32, device=device)

    param_b1 = nn.Parameter(
        DTensor.from_local(w_b1, mesh, placements),
    )
    param_b2 = nn.Parameter(
        DTensor.from_local(w_b2, mesh, placements),
    )
    param_b1.grad = DTensor.from_local(
        local_partial_b1, mesh, placements,
    )
    param_b2.grad = DTensor.from_local(
        local_partial_b2, mesh, [Partial("sum"), Shard(0)],
    )

    max_norm_b = 0.01
    with SkipDTensorDispatch():
        our_norm_b = clip_grad_norm_(
            [param_b1, param_b2], max_norm_b, norm_type=2.0,
        )

    ref_p1 = nn.Parameter(torch.ones(32, 64, device=device))
    ref_p2 = nn.Parameter(torch.ones(16, 32, device=device))
    ref_p1.grad = full_grad_b1.clone()
    ref_p2.grad = full_grad_b2.clone()
    ref_norm_b = torch.nn.utils.clip_grad_norm_(
        [ref_p1, ref_p2], max_norm_b, norm_type=2.0,
    )
    assert _close(our_norm_b, ref_norm_b), (
        f"Multi-Partial norm: {our_norm_b.item():.6f} vs "
        f"{ref_norm_b.item():.6f}"
    )
    _assert_ranks_agree(our_norm_b, "Multi-Partial-coalesce")

    # ---- (c) Partial("avg") placement ----
    # Verify AVG reduction (with SUM+divide fallback) produces
    # the same norm as nn.utils on the full gradient.
    torch.manual_seed(_SEED + 20)
    full_grad_c = torch.randn(32, 64, device=device)
    local_avg_c = full_grad_c[
        dp_rank * shard_rows : (dp_rank + 1) * shard_rows
    ].clone()

    placements_avg = [Partial("avg"), Shard(0)]
    w_c = torch.ones(shard_rows, 64, device=device)
    param_c = nn.Parameter(
        DTensor.from_local(w_c, mesh, placements_avg),
    )
    param_c.grad = DTensor.from_local(
        local_avg_c, mesh, placements_avg,
    )

    max_norm_c = 0.01
    with SkipDTensorDispatch():
        our_norm_c = clip_grad_norm_(
            [param_c], max_norm_c, norm_type=2.0,
        )

    ref_pc = nn.Parameter(torch.ones(32, 64, device=device))
    ref_pc.grad = full_grad_c.clone()
    ref_norm_c = torch.nn.utils.clip_grad_norm_(
        [ref_pc], max_norm_c, norm_type=2.0,
    )
    assert _close(our_norm_c, ref_norm_c), (
        f"Partial-avg norm: {our_norm_c.item():.6f} vs "
        f"{ref_norm_c.item():.6f}"
    )
    _assert_ranks_agree(our_norm_c, "Partial-avg")


# ===================================================================
# Test C – Edge cases (single card)
# ===================================================================

def test_clip_grad_norm_edge_cases():  # pylint: disable=R0914
    """Edge cases on a single card with plain (non-DTensor) parameters.

    Sub-sections:
      (a) Unusual norm_types: 0 (manual ref), 1 and -inf (vs nn.utils)
      (b) foreach variants: None, False, True — same result
      (c) error_if_nonfinite + mixed fp16/fp32 dtype promotion
    """
    init_dist()
    device = torch.device("npu")

    # Deterministic plain params + grads
    torch.manual_seed(_SEED)
    p1 = nn.Parameter(torch.randn(8, 8, device=device))
    p2 = nn.Parameter(torch.randn(4, device=device))
    p1.grad = torch.randn(8, 8, device=device)
    p2.grad = torch.randn(4, device=device)
    params = [p1, p2]

    def _save():
        return [(p.grad.clone(), p.data.clone()) for p in params]

    def _restore(snap):
        for p, (g, unused_data) in zip(params, snap):  # pylint: disable=W0612
            p.grad = g.clone()

    snap = _save()

    # (a) norm_type=0: our impl returns total non-zero element count
    _restore(snap)
    expected_l0 = sum(
        torch.count_nonzero(p.grad).float() for p in params
    )
    our_l0 = clip_grad_norm_(params, 1000.0, norm_type=0.0)
    assert _close(our_l0, expected_l0), (
        f"L0: {our_l0.item():.1f} vs {expected_l0.item():.1f}"
    )

    # norm_type=1 and -inf: compare against nn.utils
    for norm_type in [1.0, float("-inf")]:
        # Reference: fresh dummy params
        _restore(snap)
        dummy = [nn.Parameter(torch.zeros_like(p)) for p in params]
        for d, p in zip(dummy, params):
            d.grad = p.grad.clone()
        ref_norm = torch.nn.utils.clip_grad_norm_(
            dummy, 1000.0, norm_type=norm_type,
        )
        # Ours
        _restore(snap)
        our_norm = clip_grad_norm_(params, 1000.0, norm_type=norm_type)
        assert _close(our_norm, ref_norm), (
            f"norm_type={norm_type}: {our_norm.item():.6f} vs "
            f"{ref_norm.item():.6f}"
        )

    # (b) foreach: all variants produce the same clipped result
    _restore(snap)
    ref_norm_b, _ = _ref_clip(
        [p.grad.clone() for p in params], 0.01, 2.0,
    )
    assert ref_norm_b > 0.01, "Setup: clipping must trigger"
    for foreach in [None, False, True]:
        _restore(snap)
        our_norm = clip_grad_norm_(
            params, 0.01, norm_type=2.0, foreach=foreach,
        )
        assert _close(our_norm, ref_norm_b), (
            f"foreach={foreach}: {our_norm.item():.6f} vs "
            f"{ref_norm_b.item():.6f}"
        )

    # (c) error_if_nonfinite + mixed dtype promotion
    torch.manual_seed(_SEED + 1)
    p_f16 = nn.Parameter(
        torch.randn(4, 4, device=device, dtype=torch.float16),
    )
    p_f32 = nn.Parameter(torch.randn(4, 4, device=device))
    p_f16.grad = torch.randn(4, 4, device=device, dtype=torch.float16)
    p_f32.grad = torch.randn(4, 4, device=device)
    mixed = [p_f16, p_f32]

    # Normal: result dtype should be float32 (promoted from fp16+fp32)
    norm_ok = clip_grad_norm_(mixed, 1.0, norm_type=2.0)
    assert norm_ok.isfinite(), f"Mixed norm not finite: {norm_ok.item()}"
    assert norm_ok.dtype == torch.float32, (
        f"Expected float32, got {norm_ok.dtype}"
    )

    # Inject inf → error_if_nonfinite=True → RuntimeError
    p_f16.grad = torch.randn(4, 4, device=device, dtype=torch.float16)
    p_f32.grad = torch.randn(4, 4, device=device)
    p_f32.grad.view(-1)[0] = float("inf")
    raised = False
    try:
        clip_grad_norm_(
            mixed, 1.0, norm_type=2.0, error_if_nonfinite=True,
        )
    except RuntimeError:
        raised = True
    assert raised, "Expected RuntimeError for inf grad"

    # Inject nan → error_if_nonfinite=False → non-finite norm
    p_f16.grad = torch.randn(4, 4, device=device, dtype=torch.float16)
    p_f32.grad = torch.randn(4, 4, device=device)
    p_f16.grad.view(-1)[0] = float("nan")
    norm_nan = clip_grad_norm_(
        mixed, 1.0, norm_type=2.0, error_if_nonfinite=False,
    )
    assert not norm_nan.isfinite(), (
        f"Expected non-finite norm, got {norm_nan.item()}"
    )


# ===================================================================
# Test D – Empty / sparse gradients (no deadlock)
# ===================================================================

def test_clip_grad_norm_empty_grads():  # pylint: disable=R0914,R0915
    """Verify no deadlock with empty or sparse gradients.

    Sub-sections:
      (a) All ranks: grad=None → norm=0.0, dtype=float32
      (b) FSDP 1D: even ranks null bias grad → no deadlock, ranks agree
      (c) HSDP 2D: symmetric null weight grad → no deadlock, ranks agree
      (d) Partial placement: asymmetric grad=None across tp group →
          deadlock regression for zero-participation fix
      (e) Mixed precision (fp16) + Partial + asymmetric grad=None →
          coalesced buffer dtype-safety regression
    """
    rank, _ = init_dist()

    # ---- (a) All grads None ----
    mesh_2d = init_device_mesh(
        device_type="npu", mesh_shape=(4, 2),
        mesh_dim_names=("dp", "op"),
    )
    model = _build_model_and_backward(mesh_2d, _get_fsdp_kwargs(mesh_2d))
    for p in model.parameters():
        p.grad = None
    with SkipDTensorDispatch():
        norm_a = clip_grad_norm_(model, 1.0, norm_type=2.0)
    assert _close(norm_a, torch.tensor(0.0)), (
        f"All-None: expected 0.0, got {norm_a.item()}"
    )
    assert norm_a.dtype == torch.float32, (
        f"All-None dtype: expected float32, got {norm_a.dtype}"
    )

    # ---- (b) & (c): Sparse grads — verify no-deadlock and cross-rank
    # consistency only.  Numerical correctness of the norm under full
    # gradients is covered by Test A's reference comparison.

    # ---- (b) FSDP 1D: even ranks null bias ----
    mesh_1d = init_device_mesh(
        device_type="npu", mesh_shape=(8,),
        mesh_dim_names=("dp",),
    )
    model = _build_model_and_backward(mesh_1d, _get_fsdp_kwargs(mesh_1d))
    params = list(model.parameters())
    if len(params) > 1 and rank % 2 == 0:
        params[1].grad = None
    with SkipDTensorDispatch():
        norm_b = clip_grad_norm_(model, 1.0, norm_type=2.0)
    assert norm_b.isfinite() and norm_b >= 0
    _assert_ranks_agree(norm_b, "FSDP-1D-sparse")

    # ---- (c) HSDP 2D: symmetric null weight across shard groups ----
    # Use rank % 2 == 0 so that every shard group ({0,1},{2,3},...)
    # loses the same shard position → all groups compute the same
    # reduced norm and _assert_ranks_agree holds.
    model = _build_model_and_backward(mesh_2d, _get_fsdp_kwargs(mesh_2d))
    params = list(model.parameters())
    if rank % 2 == 0:
        params[0].grad = None
    with SkipDTensorDispatch():
        norm_c = clip_grad_norm_(model, 1.0, norm_type=2.0)
    assert norm_c.isfinite() and norm_c >= 0
    _assert_ranks_agree(norm_c, "HSDP-2D-sparse")

    # ---- (d) Partial placement: asymmetric grad=None (deadlock
    # regression).  On a (tp=2, dp=4) mesh with [Partial, Shard],
    # only tp_rank==0 carries a gradient.  Without the zero-
    # participation fix, tp_rank==1 skips the Partial all_reduce
    # that tp_rank==0 enters, causing a collective deadlock.
    mesh_tp = init_device_mesh(
        device_type="npu", mesh_shape=(2, 4),
        mesh_dim_names=("tp", "dp"),
    )
    dp_size = 4
    dp_rank = rank % dp_size
    tp_rank = rank // dp_size
    device = torch.device("npu")

    torch.manual_seed(_SEED)
    full_grad = torch.randn(32, 64, device=device)
    shard_rows = 32 // dp_size
    local_weight = torch.ones(shard_rows, 64, device=device)

    placements_d = [Partial("sum"), Shard(0)]
    param_d = nn.Parameter(
        DTensor.from_local(local_weight, mesh_tp, placements_d),
    )
    if tp_rank == 0:
        local_partial = full_grad[
            dp_rank * shard_rows : (dp_rank + 1) * shard_rows
        ].clone() / 2
        param_d.grad = DTensor.from_local(
            local_partial, mesh_tp, placements_d,
        )
    else:
        param_d.grad = None

    with SkipDTensorDispatch():
        norm_d = clip_grad_norm_([param_d], 1.0, norm_type=2.0)
    assert norm_d.isfinite() and norm_d >= 0, (
        f"Partial-grad-None: expected finite norm, got {norm_d.item()}"
    )
    # Cross-rank agreement within each dp group: ranks sharing the
    # same tp_rank (same row of the mesh) went through the same
    # Shard all_reduce and must compute identical norms.
    t_d = norm_d.clone().float()
    all_norms = [
        torch.zeros_like(t_d) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_norms, t_d)
    for r in range(dist.get_world_size()):
        if r // dp_size == tp_rank:
            assert _close(all_norms[r], t_d), (
                f"Partial-grad-None: rank {r} norm "
                f"{all_norms[r].item():.6f} != rank {rank} "
                f"norm {t_d.item():.6f}"
            )

    # ---- (e) Mixed precision + Partial + asymmetric grad=None ----
    # Same topology as (d) but gradient is fp16.  Verifies that the
    # coalesced buffer correctly casts fp16 grads to fp32 before the
    # collective, preventing dtype mismatch with the fp32 zeros
    # contributed by grad-free ranks.
    torch.manual_seed(_SEED + 2)
    full_grad_e = torch.randn(32, 64, device=device)
    local_weight_e = torch.ones(shard_rows, 64, device=device)

    param_e = nn.Parameter(
        DTensor.from_local(local_weight_e, mesh_tp, placements_d),
    )
    if tp_rank == 0:
        local_partial_e = full_grad_e[
            dp_rank * shard_rows : (dp_rank + 1) * shard_rows
        ].clone().to(torch.float16) / 2
        param_e.grad = DTensor.from_local(
            local_partial_e, mesh_tp, placements_d,
        )
    else:
        param_e.grad = None

    with SkipDTensorDispatch():
        norm_e = clip_grad_norm_([param_e], 1.0, norm_type=2.0)
    assert norm_e.isfinite() and norm_e >= 0, (
        f"MixedPrec-Partial: expected finite norm, got {norm_e.item()}"
    )
    t_e = norm_e.clone().float()
    all_norms_e = [
        torch.zeros_like(t_e) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_norms_e, t_e)
    for r in range(dist.get_world_size()):
        if r // dp_size == tp_rank:
            assert _close(all_norms_e[r], t_e), (
                f"MixedPrec-Partial: rank {r} norm "
                f"{all_norms_e[r].item():.6f} != rank {rank} "
                f"norm {t_e.item():.6f}"
            )
