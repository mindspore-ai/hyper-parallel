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
"""Module acceptance: independent formulas, all VJPs, checkpoint state and one SGD step."""

# Torch's builtin linear has no callable signature in the installed pylint stubs.
# pylint: disable=not-callable

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.modules import MHCPost, PipelinedMhc, SharedExpert, SwiGLUMLP
from hyper_parallel.components.modules.common.rms_norm import RMSNormModule, OffsetRMSNormModule
from hyper_parallel.components.modules.grouped_experts import GroupedExperts as LegacyGroupedExperts
from hyper_parallel.components.modules.attention.grouped_query_attention.grouped_query_attention import (
    apply_gqa_position,
)
from hyper_parallel.components.modules.attention.multi_latent_attention.multi_latent_attention import apply_mla_rope
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig, GroupedExpertsModule,
)
from hyper_parallel.components.modules.moe.router import MegaGateModule
from hyper_parallel.components.ops import ops
from tests.common.op_precision import PrecisionCase, assert_precision
from tests.common.op_precision_inputs import Inputs


@dataclass(frozen=True)
class ModulePrecisionCase:
    """A real production member and a source-order reference over its checkpoint state."""

    name: str
    module: nn.Module
    reference: Callable[..., Any]
    args: tuple[torch.Tensor, ...]
    variants: tuple[tuple[str, str], ...]
    dtype: torch.dtype
    expert_parameters: tuple[str, ...] = ()


def _rms(module, x):
    values = x.float()
    normalized = values / (values.square().sum(-1, keepdim=True) / x.shape[-1] + module.eps).sqrt()
    scale = 1 + module.weight.float() if isinstance(module, OffsetRMSNormModule) else module.weight
    if module.rms_norm_cast_order == "before_weight":
        return scale * normalized.to(x.dtype)
    return (scale.float() * normalized).to(x.dtype)


def _mlp(module, x):
    hidden = F.linear(x, module.linear_fc1.weight, module.linear_fc1.bias)
    gate, up = hidden.chunk(2, -1)
    if module.activation_dtype == "float32":
        gate, up = gate.float(), up.float()
    if module.swiglu_limit > 0:
        gate = gate.clamp(max=module.swiglu_limit)
        up = up.clamp(-module.swiglu_limit, module.swiglu_limit)
    activated = (F.silu(gate) * up).to(x.dtype)
    return F.linear(activated, module.linear_fc2.weight, module.linear_fc2.bias)


def _experts(module, x, indices, weights):
    # Iterate experts in original token order; no shared permutation or grouped GEMM.
    flat = x.reshape(-1, x.shape[-1])
    output = flat * 0 + (module.gate_up_proj[:0].sum() + module.down_proj[:0].sum() + weights.sum() * 0)
    for expert in range(module.config.num_experts):
        tokens, slots = torch.where(indices == expert)
        gate, up = F.linear(flat[tokens], module.gate_up_proj[expert]).chunk(2, -1)
        if module.config.swiglu_limit is not None:
            bound = module.config.swiglu_limit
            gate = gate.float().clamp(max=bound)
            up = up.float().clamp(-bound, bound)
        hidden = (F.silu(gate) * up).to(x.dtype)
        probabilities = weights[tokens, slots, None]
        if module.config.routing_weight_mode == "before_fc2":
            hidden = (hidden * probabilities).to(hidden.dtype)
        values = F.linear(hidden, module.down_proj[expert])
        if module.config.routing_weight_mode == "output":
            values = values * probabilities.to(values.dtype)
        output = output.index_add(0, tokens, values)
    return output.reshape_as(x)


def _legacy_experts(module, x, indices, weights):
    linear_owner = SimpleNamespace(config=module.config,
                                   gate_up_proj=module.gate_up_proj.transpose(1, 2),
                                   down_proj=module.down_proj.transpose(1, 2))
    return _experts(linear_owner, x, indices, weights)


def _post(module, x, residual, post, combine):
    streams = residual.unflatten(-1, (module.num_stream, x.shape[-1]))
    mixed = torch.einsum("bsij,bsih->bsjh", combine, streams.to(combine.dtype))
    return (mixed + post[..., None] * x[..., None, :]).flatten(2).to(x.dtype)


def _pipelined(module, x):
    flat = x.flatten(2).float()
    norm = flat / (flat.square().mean(-1, keepdim=True) + module.input_norm.eps).sqrt()
    mixed = F.linear(norm * module.input_norm.weight, module.fn.float())
    count = module.hc_mult
    pre, post, residual = (mixed * torch.cat((module.scale[0].expand(count), module.scale[1].expand(count),
                                            module.scale[2].expand(count * count))) + module.base).split(
                                                (count, count, count * count), -1)
    matrix = residual.unflatten(-1, (count, count))
    matrix = (matrix - matrix.amax(-1, keepdim=True)).exp()
    matrix = matrix / matrix.sum(-1, keepdim=True) + module.hc_eps
    matrix = matrix / (matrix.sum(-2, keepdim=True) + module.hc_eps)
    for _ in range(module.hc_sinkhorn_iters - 1):
        matrix = matrix / (matrix.sum(-1, keepdim=True) + module.hc_eps)
        matrix = matrix / (matrix.sum(-2, keepdim=True) + module.hc_eps)
    return pre.sigmoid() + module.hc_eps, 2 * post.sigmoid(), matrix


def _router(module, x, image_mask):
    logits = F.linear(x.reshape(-1, x.shape[-1]).float(), module.weight.float())
    scores = torch.logaddexp(logits, torch.zeros_like(logits)).sqrt()
    bias = torch.where(image_mask.reshape(-1, 1), module.bias_vl, module.bias)
    indices = (scores + bias).topk(module.top_k, sorted=False).indices
    weights = scores.gather(-1, indices)
    return logits, weights / (weights.sum(-1, keepdim=True) + 1e-20) * module.routed_scaling_factor, indices


def module_precision_cases(dtype: torch.dtype, *, hardware: bool = False) -> list[ModulePrecisionCase]:
    """Build deterministic basic/expert modules without changing the caller's random state."""
    inputs = Inputs(dtype)
    cases = []
    width = 128 if hardware else 8
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7201)
        for offset in (False, True):
            for order in ("before_weight", "after_weight"):
                settings = SimpleNamespace(hidden_size=width, eps=1e-6)
                member = (OffsetRMSNormModule(config=settings, cast_order=order, fp32_weight=True) if offset
                          else RMSNormModule(config=settings, cast_order=order, fp32_weight=True))
                nn.init.uniform_(member.weight, 0.1, 0.9)
                variant = "standard" if order == "before_weight" else "fp32_affine"
                cases.append(ModulePrecisionCase(f"rms_{offset}_{order}", member, _rms,
                                                 (inputs.tensor(2, 5, width),), (("rms_norm", variant),), dtype))
        for factory, name in ((SwiGLUMLP, "mlp"), (SharedExpert, "shared")):
            for limit, activation_dtype in ((0.0, "input"), (0.25, "float32"), (0.25, "input")):
                settings = SimpleNamespace(hidden_size=width, intermediate_size=2 * width, bias=True,
                                           swiglu_limit=limit, activation_dtype=activation_dtype,
                                           num_layers=1, init_method_std=0.1, return_bias_tuple=False)
                member = factory(config=settings).to(dtype=dtype)
                # SharedExpert owns staged projection wrappers; its reference uses the actual linear owners.
                reference = _shared if name == "shared" else _mlp
                variant = "clamped_fp32" if activation_dtype == "float32" else "standard"
                cases.append(ModulePrecisionCase(f"{name}_{limit}_{activation_dtype}", member, reference,
                                                 (inputs.tensor(2, 5, width),), (("swiglu", variant),), dtype))
        for mode in ("before_fc2", "output"):
            for empty in (False, True):
                member = GroupedExpertsModule(config=GroupedExpertsConfig(
                    4, width, 2 * width, routing_weight_mode=mode, num_layers=1, init_method_std=0.1,
                )).to(dtype=dtype)
                ids = torch.tensor([[2, 0], [0, 2], [2, 0]])[:0 if empty else 3]
                cases.append(ModulePrecisionCase(f"experts_{mode}_empty_{empty}", member, _experts,
                                                 (inputs.tensor(ids.shape[0], width), ids,
                                                  inputs.tensor(ids.shape[0], 2, dtype=torch.float32)),
                                                 (("grouped_matmul", "standard"), ("swiglu", "standard"))
                                                 + (() if empty else (("moe_token_permute", "standard"),
                                                                      ("moe_token_unpermute", "standard"))),
                                                 dtype, ("gate_up_proj", "down_proj")))
        for mode in ("before_fc2", "output"):
            source = GroupedExpertsModule(config=GroupedExpertsConfig(
                4, width, 2 * width, routing_weight_mode=mode, num_layers=1, init_method_std=0.1,
            )).to(dtype=dtype)
            source.router_gating_in_fp32 = mode == "output"
            member = LegacyGroupedExperts(module=source)
            cases.append(ModulePrecisionCase(f"legacy_experts_{mode}", member, _legacy_experts,
                                             (inputs.tensor(3, width), torch.tensor([[2, 0], [0, 2], [2, 0]]),
                                              inputs.tensor(3, 2, dtype=torch.float32)),
                                             (("grouped_matmul", "standard"), ("swiglu", "standard"),
                                              ("moe_token_permute", "standard"),
                                              ("moe_token_unpermute", "standard")), dtype,
                                             ("gate_up_proj", "down_proj")))
        post = MHCPost(config=SimpleNamespace(num_stream=4))
        cases.append(ModulePrecisionCase("mhc_post", post, _post,
                                         (inputs.tensor(2, 5, width), inputs.tensor(2, 5, 4 * width),
                                          inputs.tensor(2, 5, 4, dtype=torch.float32),
                                          inputs.tensor(2, 5, 4, 4, dtype=torch.float32)),
                                         (("mhc_post", "standard"),), dtype))
        settings = SimpleNamespace(num_stream=4, hidden_size=width, sinkhorn_iters=20, hc_eps=1e-6, norm_eps=1e-6)
        cases.append(ModulePrecisionCase("pipelined_mhc", PipelinedMhc(config=settings), _pipelined,
                                         (inputs.tensor(2, 5, 4, width),), (("sinkhorn", "standard"),), dtype))
        router_width = 256 if hardware else width
        settings = SimpleNamespace(hidden_size=router_width, num_experts=128 if hardware else 6,
                                   top_k=8 if hardware else 2,
                                   scoring_func="sqrtsoftplus", routed_scaling_factor=1.7,
                                   vision_enabled=True, initializer_range=0.1)
        router = MegaGateModule(config=settings).to(dtype=dtype)
        with torch.no_grad():
            router.bias.copy_(torch.linspace(-8, 8, router.num_experts))
            router.bias_vl.copy_(-router.bias)
        cases.append(ModulePrecisionCase("mega_gate", router, _router,
                                         (inputs.tensor(2, 5, router_width), torch.arange(10).reshape(2, 5) % 2 == 0),
                                         (("moe_router", "deepseek_v41"),), dtype))
    return cases


def _shared(module, x):
    fc1, bias = module.linear_fc1(x)
    if bias is not None:
        fc1 = fc1 + bias
    gate, up = fc1.chunk(2, -1)
    if module.activation_dtype == "float32":
        gate, up = gate.float(), up.float()
    if module.swiglu_limit > 0:
        gate, up = gate.clamp(max=module.swiglu_limit), up.clamp(-module.swiglu_limit, module.swiglu_limit)
    result, bias = module.linear_fc2((F.silu(gate) * up).to(x.dtype))
    return result if bias is None else result + bias


def _flatten(outputs):
    return tuple(outputs) if isinstance(outputs, (tuple, list)) else (outputs,)


def _evaluate(case, module, args, reference, recompute):
    run_args = tuple(arg.detach().clone().requires_grad_(arg.requires_grad) for arg in args)
    saved_args = tuple(arg.detach().clone() for arg in run_args)
    parameters = dict(module.named_parameters())
    saved = {name: value.detach().clone() for name, value in module.state_dict().items()}
    operation = (lambda *values: case.reference(module, *values)) if reference else module
    outputs = _flatten(checkpoint(operation, *run_args, use_reentrant=False) if recompute else operation(*run_args))
    generator = torch.Generator().manual_seed(9203)
    upstream = [torch.randn(output.shape, generator=generator).to(output.device) for output in outputs
                if output.is_floating_point() and output.requires_grad]
    terms = [output.float().mul(gradient).sum() for output, gradient in zip(
        (value for value in outputs if value.is_floating_point() and value.requires_grad), upstream)]
    sum(terms).backward()
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    for value, original in zip(run_args, saved_args):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    gradients = {name: parameter.grad for name, parameter in parameters.items()}
    gradients.update({f"input.{index}": value.grad for index, value in enumerate(run_args) if value.requires_grad})
    if parameters:
        torch.optim.SGD(parameters.values(), lr=0.03125).step()
    return outputs, gradients, dict(module.state_dict())


def _copy_member(module: nn.Module) -> nn.Module:
    # Family provenance contains immutable mapping proxies; share metadata, never tensor state.
    metadata = [getattr(child, "_hp_family_build_info", None) for child in module.modules()]
    return deepcopy(module, {id(value): value for value in metadata if value is not None})


def check_module_case(
    case: ModulePrecisionCase, *, selections: Mapping[str, str] | None = None, device: str = "cpu",
    recompute: bool = False, compile_module: bool = False,
) -> None:
    """Check outputs, every input/parameter VJP, unused experts, one update and state reload.

    Native runs compare to source-order formulas. Fused runs compare to same-device
    native. Backend configuration stays unchanged through backward and is always restored.
    """
    previous = ops.get_backends()
    native = {family: "native" for family, _ in case.variants}
    selections = native if selections is None else dict(selections)
    expected_module = _copy_member(case.module).to(device)
    actual_module = _copy_member(case.module).to(device)
    args = tuple(arg.to(device) for arg in case.args)
    try:
        ops.set_backends(native)
        expected = _evaluate(case, expected_module, args, selections == native, False)
        ops.set_backends(selections)
        if recompute:
            # Validate optional kernels before checkpoint installs saved-tensor hooks.
            _evaluate(case, _copy_member(case.module).to(device), args, False, False)
            ops.set_backends(selections)
        target = torch.compile(actual_module, backend="aot_eager") if compile_module else actual_module
        actual = _evaluate(case, target, args, False, recompute)
        used = ops.used_backends()
        for family, variant in case.variants:
            # Empty experts use graph-connected empty matmuls instead of unsupported fused GMM.
            if case.name.startswith("experts_") and case.args[0].shape[0] == 0 and family == "grouped_matmul":
                continue
            required = (family, variant, selections.get(family, "native"))
            if required not in used:
                raise AssertionError(f"{case.name}: used={used}, expected={required}")
        if len(actual[0]) != len(expected[0]):
            raise AssertionError(f"{case.name}: outputs={len(actual[0])}, expected={len(expected[0])}")
        for index, (value, reference) in enumerate(zip(actual[0], expected[0])):
            if value.requires_grad != reference.requires_grad:
                raise AssertionError(f"{case.name}: requires_grad={value.requires_grad}, "
                                     f"expected={reference.requires_grad}")
            assert_precision(value, reference, dtype=case.dtype, label=f"{case.name}/output.{index}")
        for name, reference in expected[1].items():
            actual_name = f"_orig_mod.{name}" if compile_module and not name.startswith("input.") else name
            value = actual[1][actual_name]
            if (value is None) != (reference is None):
                raise AssertionError(f"{case.name}/{name}: actual={value}, expected={reference}")
            if value is not None:
                assert_precision(value, reference, dtype=case.dtype, backward=True, label=f"{case.name}/{name}")
                if name in case.expert_parameters:
                    for expert, (left, right) in enumerate(zip(value, reference)):
                        assert_precision(left, right, dtype=case.dtype, backward=True, label=f"{name}/expert.{expert}")
        state = {name.removeprefix("_orig_mod."): value for name, value in actual[2].items()}
        for name, value in state.items():
            assert_precision(value, expected[2][name], dtype=case.dtype, label=f"{case.name}/update/{name}")
        restored = _copy_member(case.module).to(device)
        restored.load_state_dict(state, strict=True)
        for value, reference in zip(_flatten(restored(*args)), _flatten(actual_module(*args))):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
    finally:
        ops.set_backends(previous)


def _position_call(x, cos, sin, *, helper, interleaved):
    return helper(x, (cos, sin), interleaved=interleaved)


def _position_reference(x, cos, sin, *, interleaved):
    width = cos.shape[-1]
    if not width:
        return x
    prefix, tail = x[..., :width], x[..., width:]
    if interleaved:
        even, odd = prefix[..., ::2], prefix[..., 1::2]
        cosine, sine = cos[..., :width // 2].unsqueeze(2), sin[..., :width // 2].unsqueeze(2)
        result = torch.cat((even * cosine - odd * sine, odd * cosine + even * sine), -1)
    else:
        left, right = prefix.chunk(2, -1)
        result = prefix * cos.unsqueeze(2) + torch.cat((-right, left), -1) * sin.unsqueeze(2)
    return torch.cat((result, tail), -1)


def position_precision_cases(dtype: torch.dtype) -> list[PrecisionCase]:
    """Exercise actual GQA/MLA position adapters, including partial and identity rotations."""
    inputs = Inputs(dtype)
    result = []
    for helper in (apply_gqa_position, apply_mla_rope):
        for interleaved in (False, True):
            for width in (0, 64, 128):
                angles = inputs.tensor(1, 5, width // 2, grad=False).repeat(1, 1, 2)
                operation = partial(_position_call, helper=helper, interleaved=interleaved)
                reference = partial(_position_reference, interleaved=interleaved)
                result.append(PrecisionCase(
                    "rope", "interleaved_single" if interleaved else "half", f"{helper.__name__}_{interleaved}_{width}",
                    operation, reference, (inputs.tensor(2, 5, 3, 128), angles.cos(), angles.sin()),
                    precision_dtype=dtype,
                ))
    return result
