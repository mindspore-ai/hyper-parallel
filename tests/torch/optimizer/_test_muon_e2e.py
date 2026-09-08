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
"""End-to-end Muon optimizer cases (torchrun workers on Ascend).

Precision self-consistency contract: every sharding scenario feeds the optimizer
logically identical initial weights, gradients and optimizer states, only the
sharding rules differ. After each of the 10 steps the updated sharded weights
(and the sharded momentum / exp_avg states) are gathered back into the full
single-card view and asserted against a plain single-card reference trajectory
computed in the same worker.
"""
# pylint: disable=W0611,C0413,C0412,W0613,W0612
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from torch import nn

from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh, distribute_tensor
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.core.optimizer import get_hyper_optimizer
# pylint: disable=protected-access
from hyper_parallel.core.optimizer.muon import (
    Muon,
    MuonPostUpdateContext,
    NSInputTransform,
    zeropower_via_newtonschulz5,
    _NS_ASYM5_COEFFS,
    _NS_LEGACY_COEFFS,
)
# pylint: enable=protected-access
from tests.torch.utils import init_dist

_DIM = 64
_FFN_DIM = 128
_VOCAB = 128
_NUM_EXPERTS = 8
_STEPS = 10
_INIT_SEED = 7
_GRAD_SEED = 2026
_INPUT_SEED = 99
_GRAD_SCALE = 0.05

_LR = 0.01
_WEIGHT_DECAY = 0.1
_MATCHED_ADAMW_RMS = 0.2
_MOMENTUM = 0.9
_NS_STEPS = 5
_ADAMW_LR = 1e-3
_ADAMW_WEIGHT_DECAY = 0.01

# Muon updates go through bf16 Newton-Schulz, and batched matmul grouping may
# differ between sharding layouts, so weights use a bf16-appropriate tolerance.
_MUON_RTOL = 1e-2
_MUON_ATOL = 1e-3
# Optimizer states and AdamW-updated params are elementwise fp32 on identical
# logical values, so they should match the reference almost exactly.
_STATE_RTOL = 1e-4
_STATE_ATOL = 1e-6


class AttentionBlock(nn.Module):  # pylint: disable=abstract-method
    """Fused-QKV attention projections (2D Muon matrices)."""

    def __init__(self, dim: int) -> None:
        """Create fused qkv and output projection weights."""
        super().__init__()
        self.qkv = nn.Parameter(torch.randn(dim, 3 * dim) * 0.02)
        self.o_proj = nn.Parameter(torch.randn(dim, dim) * 0.02)


class MoEBlock(nn.Module):  # pylint: disable=abstract-method
    """Router (2D) and stacked expert weights (3D) for batched-NS coverage."""

    def __init__(self, dim: int, ffn_dim: int, num_experts: int) -> None:
        """Create the router and the stacked expert up/down weights."""
        super().__init__()
        self.router = nn.Parameter(torch.randn(dim, num_experts) * 0.02)
        self.expert_up = nn.Parameter(torch.randn(num_experts, dim, ffn_dim) * 0.02)
        self.expert_down = nn.Parameter(torch.randn(num_experts, ffn_dim, dim) * 0.02)


class MuonE2EModel(nn.Module):
    """Attention + MoE + norms covering the Muon / AdamW parameter classes."""

    def __init__(self, dim: int = _DIM, ffn_dim: int = _FFN_DIM, vocab: int = _VOCAB,
                 num_experts: int = _NUM_EXPERTS) -> None:
        """Create embedding, attention, MoE and norm parameters."""
        super().__init__()
        self.dim = dim
        self.embed = nn.Parameter(torch.randn(vocab, dim) * 0.02)
        self.attn = AttentionBlock(dim)
        self.moe = MoEBlock(dim, ffn_dim, num_experts)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one attention + MoE block and return logits against the embedding.

        Args:
            x: Input hidden states.

        Returns:
            Output logits projected by the embedding weight.
        """
        attn_out = torch.matmul(torch.matmul(x, self.attn.qkv)[..., :self.dim], self.attn.o_proj)
        hidden = self.norm1(x + attn_out)
        gates = torch.softmax(torch.matmul(hidden, self.moe.router), dim=-1)
        expert_hidden = torch.matmul(hidden.unsqueeze(1), self.moe.expert_up)
        expert_out = torch.matmul(expert_hidden, self.moe.expert_down)
        moe_out = (expert_out * gates.unsqueeze(-1)).sum(dim=-2)
        out = self.norm2(hidden + moe_out)
        return torch.matmul(out, self.embed.t())


def _build_model():
    """Build the model with a fixed seed so every rank holds identical logical weights."""
    torch.manual_seed(_INIT_SEED)
    return MuonE2EModel().npu()


def _split_params(model):
    """Route matrix parameters to Muon and 1D parameters (norms) to AdamW."""
    muon_params = [param for param in model.parameters() if param.ndim >= 2]
    adamw_params = [param for param in model.parameters() if param.ndim < 2]
    return muon_params, adamw_params


def _make_optimizer(model, **muon_overrides):
    """Build the chained Muon + AdamW optimizer used by the e2e cases."""
    muon_params, adamw_params = _split_params(model)
    muon_kwargs = {
        "muon_lr": _LR,
        "muon_weight_decay": _WEIGHT_DECAY,
        "muon_matched_adamw_rms": _MATCHED_ADAMW_RMS,
        "muon_momentum": _MOMENTUM,
        "muon_ns_steps": _NS_STEPS,
    }
    for key, value in muon_overrides.items():
        muon_kwargs[key if key.startswith("muon_") else f"muon_{key}"] = value
    return get_hyper_optimizer(
        model=model,
        muon_params=[{"params": muon_params}],
        adamw_params=[{"params": adamw_params}],
        muon_kwargs=muon_kwargs,
        adamw_kwargs={"adamw_lr": _ADAMW_LR, "adamw_weight_decay": _ADAMW_WEIGHT_DECAY},
    )


def _full_grads(step, model):
    """Deterministic per-step logical gradients, identical on every rank."""
    generator = torch.Generator().manual_seed(_GRAD_SEED + step)
    return {
        name: torch.randn(tuple(param.shape), generator=generator) * _GRAD_SCALE
        for name, param in model.named_parameters()
    }


def _assign_gradients(model, grads):
    """Assign gradients, slicing the same logical gradient into each parameter's layout."""
    for name, param in model.named_parameters():
        full_grad = grads[name].npu()
        if isinstance(param, DTensor):
            grad = distribute_tensor(full_grad, param.device_mesh, param.placements)
            assert grad.to_local().shape == param.to_local().shape, (
                f"grad local shape {tuple(grad.to_local().shape)} != param local shape "
                f"{tuple(param.to_local().shape)} for {name}"
            )
            param.grad = grad
        else:
            param.grad = full_grad


def _logical_value(tensor):
    """Gather a (possibly DTensor) value back into the full single-card view."""
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


def _logical_optimizer_state(tensor, param):
    """Gather local optimizer state using its parameter's logical layout."""
    if not isinstance(param, DTensor):
        return _logical_value(tensor)
    local_tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    state_dtensor = DTensor.from_local(
        local_tensor,
        param.device_mesh,
        param.placements,
        shape=tuple(param.shape),
    )
    return state_dtensor.full_tensor()


def _optimizer_step(optimizer):
    """Run one optimizer step through the production DTensor bypass boundary."""
    with SkipDTensorDispatch():
        optimizer.step()


def _assert_close(ref_value, actual_value, label, rtol, atol):
    """Assert two logical tensors match within tolerance, reporting the max diff."""
    ref_np = ref_value.detach().float().cpu().numpy()
    actual_np = actual_value.detach().float().cpu().numpy()
    if not np.allclose(ref_np, actual_np, rtol=rtol, atol=atol):
        max_diff = float(np.abs(ref_np - actual_np).max())
        raise AssertionError(
            f"{label}: max abs diff {max_diff} exceeds rtol={rtol}, atol={atol}"
        )


def _snapshot_step(model, muon_optimizer, adamw_optimizer):
    """Capture full weights and optimizer states after one step."""
    params = {}
    momentum = {}
    exp_avg = {}
    for name, param in model.named_parameters():
        params[name] = param.detach().clone()
        if param.ndim >= 2:
            momentum[name] = muon_optimizer.state[param]["momentum_buffer"].detach().clone()
        else:
            exp_avg[name] = adamw_optimizer.state[param]["exp_avg"].detach().clone()
    return {"params": params, "momentum": momentum, "exp_avg": exp_avg}


def _run_reference_trajectory():
    """Plain single-card reference: no sharding, one logical view of weights/grads/states."""
    model = _build_model()
    optimizer = _make_optimizer(model)
    muon_optimizer = optimizer.optimizers_dict["muon"]
    adamw_optimizer = optimizer.optimizers_dict["adamw"]
    trajectory = []
    for step in range(_STEPS):
        _assign_gradients(model, _full_grads(step, model))
        _optimizer_step(optimizer)
        trajectory.append(_snapshot_step(model, muon_optimizer, adamw_optimizer))
    return trajectory


def _assert_matches_reference(model, optimizer, reference):
    """Run 10 sharded steps and assert weights/states match the reference every step."""
    muon_optimizer = optimizer.optimizers_dict["muon"]
    adamw_optimizer = optimizer.optimizers_dict["adamw"]
    for step in range(_STEPS):
        _assign_gradients(model, _full_grads(step, model))
        _optimizer_step(optimizer)
        ref_snapshot = reference[step]
        # HSDP keeps Muon state only on each parameter's replica owner. Use the
        # checkpoint synchronization path while validating the full logical state,
        # then release temporary non-owner copies before the next optimizer step.
        muon_optimizer._broadcast_state_fused_for_ckpt()  # pylint: disable=protected-access
        try:
            for name, param in model.named_parameters():
                is_muon = param.ndim >= 2
                rtol, atol = (_MUON_RTOL, _MUON_ATOL) if is_muon else (_STATE_RTOL, _STATE_ATOL)
                _assert_close(
                    ref_snapshot["params"][name], _logical_value(param),
                    f"step {step} param {name}", rtol, atol,
                )
                if is_muon:
                    _assert_close(
                        ref_snapshot["momentum"][name],
                        _logical_optimizer_state(muon_optimizer.state[param]["momentum_buffer"], param),
                        f"step {step} momentum {name}", _STATE_RTOL, _STATE_ATOL,
                    )
                else:
                    _assert_close(
                        ref_snapshot["exp_avg"][name],
                        _logical_optimizer_state(adamw_optimizer.state[param]["exp_avg"], param),
                        f"step {step} exp_avg {name}", _STATE_RTOL, _STATE_ATOL,
                    )
        finally:
            muon_optimizer.cleanup_synced_state()


def _run_sharding_parity_case(shard_model_fn, muon_overrides=None):
    """Shared driver: reference trajectory vs one sharded scenario, asserted per step."""
    init_dist()
    reference = _run_reference_trajectory()
    model = shard_model_fn(_build_model())
    optimizer = _make_optimizer(model, **(muon_overrides or {}))
    _assert_matches_reference(model, optimizer, reference)


def _shard_full_8p(model):
    """Shard every parameter on a flat 8-card mesh."""
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    return fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=MixedPrecisionPolicy())


def _shard_hybrid_2x4(model):
    """Shard on a (replicate=2, shard=4) mesh so Muon dedups NS compute across replicas."""
    mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("replicate", "shard"))
    return fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=MixedPrecisionPolicy())


def _shard_mixed_meshes(model):
    """Different sharding rules per module: attention on 8 cards, experts on (2, 4), rest replicated."""
    mesh_dp8 = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    mesh_2x4 = init_device_mesh(device_type="npu", mesh_shape=(2, 4), mesh_dim_names=("replicate", "shard"))
    model.attn = fully_shard(model.attn, mesh=mesh_dp8, reshard_after_forward=True,
                             mp_policy=MixedPrecisionPolicy())
    model.moe = fully_shard(model.moe, mesh=mesh_2x4, reshard_after_forward=True,
                            mp_policy=MixedPrecisionPolicy())
    return model


def test_muon_e2e_full_shard_8p():
    """
    Feature: Muon e2e precision self-consistency under full sharding.
    Description:
        All weights (embedding, attention, router, 3D experts) sharded on an 8-card mesh,
        norm weights sharded for AdamW. Identical logical weights/grads/states as the
        single-card reference; gather and assert after each of the 10 steps.
    Expectation: Every step matches the single-card reference within tolerance.
    """
    _run_sharding_parity_case(_shard_full_8p)


def test_muon_e2e_hybrid_shard_replicate_8p():
    """
    Feature: Muon e2e precision self-consistency under hybrid sharding with replica dedup.
    Description:
        Params sharded on a (replicate=2, shard=4) mesh with hsdp_replica_count=2, so NS
        compute is split across replica peers and broadcast back.
    Expectation: Every step matches the single-card reference within tolerance.
    """
    _run_sharding_parity_case(_shard_hybrid_2x4, {"hsdp_replica_count": 2})


def test_muon_e2e_mixed_mesh_per_module_8p():
    """
    Feature: Muon e2e precision self-consistency with mixed sharding rules in one model.
    Description:
        Attention weights sharded on 8 cards, expert weights sharded on a (2, 4) mesh,
        embedding and norm weights replicated: multiple HSDP comm groups in one optimizer.
    Expectation: Every step matches the single-card reference within tolerance.
    """
    _run_sharding_parity_case(_shard_mixed_meshes)


def test_muon_e2e_forward_backward_smoke_8p():
    """
    Feature: Muon end-to-end training smoke under fully_shard.
    Description:
        Real forward/backward on an 8-card fully-sharded attention+MoE model with the
        chained Muon + AdamW optimizer; gradients come from autograd, not manual setup.
    Expectation: Losses stay finite and decrease over the steps.
    """
    init_dist()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,), mesh_dim_names=("dp",))
    model = fully_shard(_build_model(), mesh=mesh, reshard_after_forward=True,
                        mp_policy=MixedPrecisionPolicy())
    optimizer = _make_optimizer(model)
    generator = torch.Generator().manual_seed(_INPUT_SEED)
    data = torch.randn(8, _DIM, generator=generator).npu()

    losses = []
    for _ in range(_STEPS):
        logits = model(data)
        loss = (logits * logits).mean()
        loss.backward()
        _optimizer_step(optimizer)
        optimizer.zero_grad()
        loss_value = _logical_value(loss.detach())
        losses.append(float(loss_value))

    assert all(np.isfinite(losses)), f"non-finite loss in {losses}"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def _run_plain_trajectory(**muon_overrides):
    """Single-card trajectory helper for the callback parity cases."""
    model = _build_model()
    optimizer = _make_optimizer(model, **muon_overrides)
    trajectory = []
    for step in range(_STEPS):
        _assign_gradients(model, _full_grads(step, model))
        _optimizer_step(optimizer)
        trajectory.append({name: param.detach().clone() for name, param in model.named_parameters()})
    return trajectory


def _assert_trajectories_equal(reference, actual, label, rtol=_STATE_RTOL, atol=_STATE_ATOL):
    """Assert two full trajectories match step by step for every parameter."""
    for step in range(_STEPS):
        for name in reference[step]:
            _assert_close(reference[step][name], actual[step][name],
                          f"{label} step {step} param {name}", rtol, atol)


def test_muon_custom_ns_coefficients_parity():
    """
    Feature: Muon custom Newton-Schulz coefficients.
    Description:
        ns_variant="custom" fed with the built-in asym5 / legacy coefficient tables must
        reproduce the corresponding built-in variant step by step.
    Expectation: Custom-coefficient trajectory matches the built-in variant trajectory.
    """
    init_dist()
    for variant, coefficients in (
            ("asym5", list(_NS_ASYM5_COEFFS)),
            ("legacy", [_NS_LEGACY_COEFFS] * _NS_STEPS),
    ):
        reference = _run_plain_trajectory(ns_variant=variant)
        custom = _run_plain_trajectory(ns_variant="custom", ns_coefficients=coefficients)
        _assert_trajectories_equal(reference, custom, f"custom-ns-vs-{variant}")


def test_muon_zeropower_fn_parity():
    """
    Feature: Muon zeropower_fn callback.
    Description:
        A zeropower_fn that delegates to the default Newton-Schulz implementation must be
        invoked every step and must reproduce the default trajectory.
    Expectation: Callback invoked; trajectory matches the default run.
    """
    init_dist()
    calls = []

    def zeropower_fn(ns_input: torch.Tensor, steps: int) -> torch.Tensor:
        """Delegate to the default Newton-Schulz and record the invocation.

        Args:
            ns_input: Batched Newton-Schulz input.
            steps: Number of Newton-Schulz iterations.

        Returns:
            The approximate orthogonal update.
        """
        calls.append(tuple(ns_input.shape))
        return zeropower_via_newtonschulz5(ns_input, steps=steps, ns_variant="asym5")

    reference = _run_plain_trajectory()
    actual = _run_plain_trajectory(zeropower_fn=zeropower_fn)
    assert len(calls) >= _STEPS, f"zeropower_fn called {len(calls)} times, expected >= {_STEPS}"
    _assert_trajectories_equal(reference, actual, "zeropower_fn")


def test_muon_momentum_update_fn_parity():
    """
    Feature: Muon momentum_update_fn callback.
    Description:
        A momentum_update_fn reimplementing the native momentum math must be invoked once
        per step and must reproduce the default trajectory.
    Expectation: Callback invoked per step; trajectory matches the default run.
    """
    init_dist()
    call_args = []

    def momentum_update_fn(grads: list, bufs: list, momentum1: float, momentum2: float,
                           nesterov: bool) -> list:
        """Reimplement the native momentum math and record the invocation.

        Args:
            grads: Local gradient tensors.
            bufs: Local momentum buffers.
            momentum1: Momentum coefficient used to form the update.
            momentum2: Momentum coefficient used to update the state.
            nesterov: Whether to apply the Nesterov correction.

        Returns:
            Momentum-adjusted Newton-Schulz inputs.
        """
        call_args.append((momentum1, momentum2, nesterov, len(grads)))
        updates = []
        for grad, buf in zip(grads, bufs):
            update = grad + momentum1 * buf
            buf.mul_(momentum2).add_(grad)
            if nesterov:
                update = update * momentum1 + grad
            updates.append(update)
        return updates

    reference = _run_plain_trajectory()
    actual = _run_plain_trajectory(momentum_update_fn=momentum_update_fn)
    assert len(call_args) == _STEPS, f"momentum_update_fn called {len(call_args)} times"
    assert all(args[2] for args in call_args), "nesterov flag not forwarded to momentum_update_fn"
    _assert_trajectories_equal(reference, actual, "momentum_update_fn")


def test_muon_apply_lr_in_update_parity():
    """
    Feature: Muon apply_lr_in_update mode.
    Description:
        Folding -lr into the update scale (apply_lr_in_update=True) must produce the same
        trajectory as the default separate -lr application, up to float reordering.
    Expectation: Both trajectories match within bf16 tolerance.
    """
    init_dist()
    reference = _run_plain_trajectory()
    actual = _run_plain_trajectory(apply_lr_in_update=True)
    _assert_trajectories_equal(reference, actual, "apply_lr_in_update", rtol=_MUON_RTOL, atol=_MUON_ATOL)


def test_muon_zero_rms_scale_mode():
    """
    Feature: Muon zero_rms_scale_mode with matched_adamw_rms=0.
    Description:
        mode "zero": updates are scaled to 0, so only weight decay changes the weights.
        mode "use_lr": the scale is 1.0, so the step is -lr * NS(momentum); validated
        against a manually computed Newton-Schulz expectation.
    Expectation: Both modes match their analytic expectations.
    """
    init_dist()

    # mode "zero": only decoupled weight decay may move the Muon parameters.
    model = _build_model()
    initial = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer = _make_optimizer(model, matched_adamw_rms=0.0, zero_rms_scale_mode="zero")
    for step in range(_STEPS):
        _assign_gradients(model, _full_grads(step, model))
        _optimizer_step(optimizer)
    decay = (1 - _LR * _WEIGHT_DECAY) ** _STEPS
    for name, param in model.named_parameters():
        if param.ndim >= 2:
            _assert_close(initial[name] * decay, param.detach(),
                          f"zero_rms zero param {name}", _STATE_RTOL, _STATE_ATOL)

    # mode "use_lr": replicate momentum + NS by hand and compare every step.
    model = _build_model()
    optimizer = _make_optimizer(model, matched_adamw_rms=0.0, zero_rms_scale_mode="use_lr")
    momentum_bufs = {}
    for step in range(_STEPS):
        grads = _full_grads(step, model)
        _assign_gradients(model, grads)
        expected = {}
        for name, param in model.named_parameters():
            if param.ndim < 2:
                continue
            grad = grads[name].npu()
            buf = momentum_bufs.get(name, torch.zeros_like(grad))
            update = grad + _MOMENTUM * buf
            buf = buf * _MOMENTUM + grad
            momentum_bufs[name] = buf
            update = update * _MOMENTUM + grad  # nesterov
            ns_input = update.to(torch.bfloat16)
            ns_output = zeropower_via_newtonschulz5(
                ns_input.reshape(-1, *ns_input.shape[-2:]), steps=_NS_STEPS, ns_variant="asym5")
            expected[name] = (param.detach() * (1 - _LR * _WEIGHT_DECAY)
                              - _LR * ns_output.reshape(param.shape).float())
        _optimizer_step(optimizer)
        for name, param in model.named_parameters():
            if param.ndim >= 2:
                _assert_close(expected[name], param.detach(),
                              f"zero_rms use_lr step {step} param {name}", _MUON_RTOL, _MUON_ATOL)


def test_muon_ns_transform_fn_qkv_split_parity():
    """
    Feature: Muon ns_transform_fn reversible NS transform.
    Description:
        Splitting the fused qkv weight into q/k/v views via NSInputTransform must match a
        run where q, k, v are separate parameters, proving restore() writes updates back
        to the right slices.
    Expectation: Every q/k/v slice matches the separate-parameter trajectory per step.
    """
    init_dist()
    dim = _DIM

    def ns_transform_fn(param_fqn: str, working_input: torch.Tensor) -> "NSInputTransform | None":
        """Split the fused qkv weight into q/k/v views for separate NS updates.

        Args:
            param_fqn: Fully qualified parameter name.
            working_input: Contiguous Newton-Schulz input.

        Returns:
            A reversible qkv transform for the fused weight, otherwise ``None``.
        """
        if param_fqn != "attn.qkv":
            return None
        views = [working_input[:, index * dim:(index + 1) * dim] for index in range(3)]

        def restore(updates: list, output: torch.Tensor) -> None:
            """Write the per-slice NS updates back into the fused weight views.

            Args:
                updates: Newton-Schulz updates for q, k, and v.
                output: Destination tensor for the restored fused update.
            """
            for view, update in zip(views, updates):
                view.copy_(update.contiguous().view_as(view))

        return NSInputTransform(tensors=views, restore=restore)

    fused_model = _build_model()
    fused_optimizer = _make_optimizer(fused_model, ns_transform_fn=ns_transform_fn)

    generator = torch.Generator().manual_seed(_GRAD_SEED)
    qkv_grads = [torch.randn(tuple(fused_model.attn.qkv.shape), generator=generator) * _GRAD_SCALE
                 for _ in range(_STEPS)]

    qkv_init = _build_model().attn.qkv.detach()
    separate_params = [
        nn.Parameter(qkv_init[:, 0 * dim:1 * dim].clone()),
        nn.Parameter(qkv_init[:, 1 * dim:2 * dim].clone()),
        nn.Parameter(qkv_init[:, 2 * dim:3 * dim].clone()),
    ]
    separate_optimizer = Muon(separate_params, lr=_LR, weight_decay=_WEIGHT_DECAY,
                              matched_adamw_rms=_MATCHED_ADAMW_RMS, momentum=_MOMENTUM,
                              ns_steps=_NS_STEPS)

    for step in range(_STEPS):
        for name, param in fused_model.named_parameters():
            if name == "attn.qkv":
                param.grad = qkv_grads[step].npu().clone()
            else:
                param.grad = torch.zeros_like(param)
        _optimizer_step(fused_optimizer)

        full_grad = qkv_grads[step].npu()
        for index, param in enumerate(separate_params):
            param.grad = full_grad[:, index * dim:(index + 1) * dim].clone()
        _optimizer_step(separate_optimizer)

        for index, param in enumerate(separate_params):
            _assert_close(
                param.detach(),
                fused_model.attn.qkv.detach()[:, index * dim:(index + 1) * dim],
                f"ns_transform_fn step {step} qkv slice {index}", _MUON_RTOL, _MUON_ATOL,
            )


def test_muon_post_update_fn_context():
    """
    Feature: Muon post_update_fn callback.
    Description:
        The callback must fire once per Muon parameter per step, with the parameter already
        updated in place and a context carrying fqn, logical shape, lr, weight_decay, step.
    Expectation: Full parameter/step coverage with correct context values.
    """
    init_dist()
    records = []

    def post_update_fn(param: torch.nn.Parameter, local_tensor: torch.Tensor,
                       context: MuonPostUpdateContext) -> None:
        """Record the invocation and assert the context contents.

        Args:
            param: Parameter associated with the callback.
            local_tensor: Updated local parameter tensor.
            context: Muon update metadata.
        """
        records.append((context.param_fqn, context.step))
        assert context.lr == _LR, f"unexpected lr in context: {context.lr}"
        assert context.weight_decay == _WEIGHT_DECAY, f"unexpected weight_decay: {context.weight_decay}"
        assert tuple(context.logical_shape) == tuple(param.shape), (
            f"logical_shape {context.logical_shape} != param shape {tuple(param.shape)}"
        )
        assert local_tensor.untyped_storage().data_ptr() == param.data.untyped_storage().data_ptr(), (
            "post_update_fn must observe the updated in-place parameter storage"
        )

    model = _build_model()
    optimizer = _make_optimizer(model, post_update_fn=post_update_fn)
    for step in range(_STEPS):
        _assign_gradients(model, _full_grads(step, model))
        _optimizer_step(optimizer)

    muon_names = {name for name, param in model.named_parameters() if param.ndim >= 2}
    expected = {(name, step) for name in muon_names for step in range(1, _STEPS + 1)}
    assert set(records) == expected, (
        f"post_update_fn coverage mismatch, missing: {expected - set(records)}, "
        f"unexpected: {set(records) - expected}"
    )
