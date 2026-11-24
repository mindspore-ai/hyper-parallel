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
"""Real A5 training with V4.1 clamped experts and the existing quantization strategies."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

from hyper_parallel.components.functional.moe_token_permute import moe_token_permute
from hyper_parallel.components.functional.moe_token_unpermute import moe_token_unpermute
from hyper_parallel.components.modules.low_precision_grouped_experts import (
    LowPrecisionGroupedExperts,
    _swiglu as _clamped_swiglu,
)
from hyper_parallel.components.modules.mxfp8_grouped_swiglu import FusedMXFP8GroupedSwiGLU
from hyper_parallel.components.quantization.functional.base_gmm_func import _GroupedLinearFunction
from hyper_parallel.components.quantization.functional.mxfp8_fused_grouped_swiglu import fused_grouped_swiglu
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import MXFP8GroupedLinear
from hyper_parallel.components.quantization.functional.strategy_factory import build_low_precision_strategy
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.components.quantization.tensor import QuantizedTensor
from hyper_parallel.distributed.expert_parallel.experts import _argsort_keys
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.models._transformers.model_builder import apply_model_infrastructure, instantiate_infrastructure
from hyper_parallel.trainer.config.parallelism import normalize_distributed_setup_overrides
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.qat_family import deepseek_experts_model, deepseek_qat_config, install_family
from tests.torch.expert_parallel._test_family_mxfp8_local import CountingMXOps, _device


def test_deepseek_fused_mxfp8() -> None:
    """Train native V4.1 clipped experts with both real CANN fusion kernels."""
    _device()
    torch.manual_seed(971)
    model = deepseek_experts_model().to(device="npu", dtype=torch.bfloat16)
    source = model.experts
    install_family(model, deepseek_qat_config(fused=True))
    expert = model.experts
    optimizer = torch.optim.SGD(expert.parameters(), lr=1e-4)
    counts = torch.tensor([16, 0, 16, 32], device="npu", dtype=torch.int64)
    inputs = torch.randn(64, 128, device="npu", dtype=torch.bfloat16) * 64
    initial = [parameter.detach().clone() for parameter in expert.parameters()]
    chunks = inputs.split([16, 0, 16, 32])
    expected = torch.cat([
        source._apply_gate(chunk @ gate_up.T) @ down.T  # pylint: disable=protected-access
        for chunk, gate_up, down in zip(chunks, source.gate_up_proj, source.down_proj)
    ])
    runtime = expert.quantizer.npu_ops
    losses = []
    saved = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        """Observe that swap/recompute hooks see physical tensors only.

        Args:
            tensor: Physical saved activation or quantized payload buffer.
        """
        assert isinstance(tensor, torch.Tensor) and not isinstance(tensor, QuantizedTensor)
        saved.append(tuple(tensor.shape))
        return tensor

    with patch.object(runtime, "swiglu_mx_quant_dual_axis", wraps=runtime.swiglu_mx_quant_dual_axis) as forward, \
         patch.object(runtime, "swiglu_backward_mx_quant_dual_axis",
                      wraps=runtime.swiglu_backward_mx_quant_dual_axis) as backward:
        for step in range(10):
            optimizer.zero_grad(set_to_none=True)
            states = inputs.detach().clone().requires_grad_()
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                actual = expert.forward_expert_major(states, counts)
                loss = actual.float().square().mean()
            if step == 0:
                relative_l2 = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
                assert relative_l2 < 0.12, relative_l2
            loss.backward()
            assert torch.isfinite(loss)
            assert torch.isfinite(states.grad).all()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in expert.parameters())
            optimizer.step()
            losses.append(loss.item())
        assert forward.call_count == backward.call_count == 10
        assert all(call.kwargs["clamp_limit"] == source.limit for call in forward.call_args_list)
        assert all(call.kwargs["clamp_limit"] == source.limit for call in backward.call_args_list)
    assert all(not torch.equal(before, after) for before, after in zip(initial, expert.parameters()))
    assert saved
    print(json.dumps({"deepseek_clamp": source.limit, "fused": True, "steps": 10,
                      "losses": losses, "bf16_forward_relative_l2": relative_l2,
                      "fused_forward_calls": 10, "fused_backward_calls": 10,
                      "physical_saved_tensors": len(saved)}), flush=True)


@pytest.mark.parametrize("fake,weight_format", [(False, "mxfp8"), (False, "mxfp4"), (True, "mxfp4")])
def test_deepseek_qat_family(fake: bool, weight_format: str) -> None:
    """Compare local training against native activation and the unmodified strategy.

    Args:
        fake: Select fake W4A8 rather than native quantization.
        weight_format: Named scheme's weight format; activation format is MXFP8.
    """
    _device()
    torch.manual_seed(971)
    config = deepseek_qat_config(fake=fake, weight_format=weight_format)
    model = deepseek_experts_model().to(device="npu", dtype=torch.bfloat16)
    source = model.experts
    reference_weights = nn.ParameterList([nn.Parameter(p.detach().clone()) for p in source.parameters()])
    install_family(model, config)
    expert = model.experts
    actual_weights = (expert.gate_up_proj, expert.down_proj)
    policy = config.training.low_precision.resolve_dtype_scheme("experts")
    strategy = build_low_precision_strategy(policy, tile_shapes=tuple(tuple(p.shape) for p in reference_weights))
    assert type(expert.grouped_linear) is type(strategy)
    calls = []
    original = expert.grouped_linear.grouped_matmul

    def observed(*args: Any, **kwargs: Any) -> torch.Tensor:
        """Record real strategy dispatch while preserving device execution."""
        calls.append(kwargs["layout"])
        return original(*args, **kwargs)

    expert.grouped_linear.grouped_matmul = observed
    optimizer = torch.optim.SGD(expert.parameters(), lr=1e-4)
    reference_optimizer = torch.optim.SGD(reference_weights.parameters(), lr=1e-4)
    groups = torch.tensor([16, 0, 16, 32], device="npu", dtype=torch.int64)
    # Large projections exercise clipping; ordinary initialization alone rarely reaches limit=10.
    inputs = torch.randn(64, 128, device="npu", dtype=torch.bfloat16) * 64
    initial = [p.detach().clone() for p in actual_weights]
    losses = []
    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        actual_input = inputs.detach().clone().requires_grad_()
        expected_input = inputs.detach().clone().requires_grad_()
        actual = expert.forward_expert_major(actual_input, groups)
        projected = _GroupedLinearFunction.apply(expected_input, reference_weights[0], groups, strategy, 1)
        if step == 0:
            gate, up = projected.chunk(2, dim=-1)
            assert (gate > source.limit).any()
            assert (up > source.limit).any() and (up < -source.limit).any()
        # Use the native activation hook so the reference keeps identical quantization groups and order.
        activated = source._apply_gate(projected)  # pylint: disable=protected-access
        expected = _GroupedLinearFunction.apply(activated, reference_weights[1], groups, strategy, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        loss, expected_loss = actual.float().square().mean(), expected.float().square().mean()
        loss.backward()
        expected_loss.backward()
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0, atol=0)
        for actual_weight, expected_weight in zip(actual_weights, reference_weights):
            torch.testing.assert_close(actual_weight.grad, expected_weight.grad.transpose(-2, -1), rtol=0, atol=0)
        optimizer.step()
        reference_optimizer.step()
        for actual_weight, expected_weight in zip(actual_weights, reference_weights):
            torch.testing.assert_close(actual_weight, expected_weight.transpose(-2, -1), rtol=0, atol=0)
        assert torch.isfinite(loss)
        losses.append(loss.item())
    assert all(not torch.equal(before, after) for before, after in zip(initial, actual_weights))
    assert len(calls) == 60
    assert {"NN", "NT", "TN"}.issubset(calls)
    print(json.dumps({"deepseek_clamp": source.limit, "fake": fake, "weight_format": weight_format,
                      "strategy": type(strategy).__name__, "steps": 10, "gmm_calls": len(calls),
                      "losses": losses, "reference": "exact outputs, input/weight gradients and updates"}), flush=True)


class _FixedV41Router(nn.Module):
    """Fixed router satisfying the DeepSeek-V4.1 gate call and attribute contract."""

    def __init__(self, num_experts: int = 4) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.expert_bias_update_groups = ()
        self.indices = None
        self.scores = None

    def forward(
        self, hidden_states: torch.Tensor, image_mask: Any = None, token_mask: Any = None,
        sequence_ids: Any = None, num_sequences: Any = None, sequence_partition_groups: tuple = (),
    ) -> tuple:
        """Return fixed expert assignments and differentiable routing weights."""
        del hidden_states, image_mask, token_mask, sequence_ids, num_sequences, sequence_partition_groups
        return None, self.scores, self.indices


class _ZeroSharedExpert(nn.Module):
    """Deterministic zero shared-expert branch isolating routed EP parity."""

    @staticmethod
    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        """Return a zero tensor matching the input dtype and shape."""
        return hidden_states * 0.0


class _V41MoEBlock(nn.Module):
    """Minimal V4.1 MoE container exposing gate, experts, shared_experts, and is_hash."""

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.config = config
        self.is_hash = False
        self.gate = _FixedV41Router(num_experts=config.num_local_experts)
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = _ZeroSharedExpert()

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Run text-only routed and shared expert forward."""
        del input_ids
        _, weights, indices = self.gate(hidden_states)
        routed = self.experts(hidden_states.reshape(-1, hidden_states.shape[-1]), indices, weights)
        return routed.reshape_as(hidden_states) + self.shared_experts(hidden_states)


class _V41TinyMoEModel(nn.Module):
    """Minimal model wrapping one DeepSeek-V4.1 MoE block for real EP planner tests."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            architectures=["DeepseekV41ForCausalLM"], model_type="deepseek_v41",
            n_routed_experts=4, num_experts_per_tok=2, hidden_size=128,
            num_attention_heads=4, num_key_value_heads=1, tie_word_embeddings=False,
            hidden_act="silu", swiglu_limit=10.0,
        )
        expert_cfg = DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg
            hidden_size=128, intermediate_size=256, num_local_experts=4,
            num_experts_per_tok=2, swiglu_limit=10.0,
        )
        self.mlp = _V41MoEBlock(expert_cfg)
        for parameter in self.mlp.experts.parameters():
            nn.init.normal_(parameter, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward local activations through the V4.1 MoE block."""
        return self.mlp(hidden_states)


def _v41_ep_routing(case: str, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Construct balanced, empty-expert, and empty-receiver-rank routing tables."""
    routes, scores = [], []
    for rank in range(2):
        if case == "balanced":
            indices = [[(token + rank) % 4, (token + rank + 2) % 4] for token in range(32)]
        elif case == "empty_experts":
            indices = [[0, 2]] * 32
        else:
            indices = [[0, 1]] * 32
        routes.append(torch.tensor(indices, device=device, dtype=torch.int64))
        values = [[0.25 + (token % 2) * 0.125, 0.5 + rank * 0.125] for token in range(32)]
        scores.append(torch.tensor(values, device=device, dtype=torch.bfloat16, requires_grad=True))
    return routes, scores


def _v41_ep_reference(
    reference_experts: nn.Module, inputs: list[torch.Tensor], routes: list[torch.Tensor],
    scores: list[torch.Tensor], *, fused: bool, limit: float,
) -> list[torch.Tensor]:
    """Reproduce two-rank EP dispatch, local V4.1 MXFP8 math, and combine without collectives."""
    dispatched, unpermute_indices, dispatch_indices, splits = [], [], [], []
    for states, indices in zip(inputs, routes):
        flat_indices = indices.flatten()
        permuted, inverse = moe_token_permute(states.reshape(-1, 128), indices.to(torch.int32))
        order = torch.empty_like(inverse, dtype=torch.int64)
        order.scatter_(0, inverse.to(torch.int64), torch.arange(inverse.numel(), device=states.device))
        ordered_indices = flat_indices[order]
        unpermute_indices.append(inverse)
        dispatch_indices.append(ordered_indices)
        dispatched.append(permuted)
        splits.append([(ordered_indices // 2 == owner).sum().item() for owner in range(2)])
    sender_outputs = [[], []]
    quantizer = MXFP8Quantizer()
    gmm_strategy = MXFP8GroupedLinear()
    for owner in range(2):
        state_parts, index_parts = [], []
        for sender in range(2):
            begin = sum(splits[sender][:owner])
            end = begin + splits[sender][owner]
            state_parts.append(dispatched[sender][begin:end])
            index_parts.append(dispatch_indices[sender][begin:end] - owner * 2)
        received = torch.cat(state_parts)
        local_indices = torch.cat(index_parts)
        order = _argsort_keys(local_indices, bound=2)
        counts = torch.stack([(local_indices == index).sum() for index in range(2)]).to(torch.int64)
        expert_slice = slice(owner * 2, (owner + 1) * 2)
        gate_up = reference_experts.gate_up_proj[expert_slice]
        down = reference_experts.down_proj[expert_slice]
        sorted_in = received[order]
        if fused:
            sorted_out = fused_grouped_swiglu(
                sorted_in, gate_up, down, counts, quantizer, clamp_limit=limit)
        else:
            proj = _GroupedLinearFunction.apply(
                sorted_in, gate_up.transpose(-2, -1), counts, gmm_strategy, 1)
            activated = _clamped_swiglu(proj, limit=limit)
            sorted_out = _GroupedLinearFunction.apply(
                activated, down.transpose(-2, -1), counts, gmm_strategy, 1)
        received_output = torch.empty_like(sorted_out)
        received_output[order] = sorted_out
        returned = received_output.split([splits[sender][owner] for sender in range(2)])
        for sender, values in enumerate(returned):
            sender_outputs[sender].append(values)
    outputs = []
    for sender, parts in enumerate(sender_outputs):
        returned = torch.cat(parts)
        output = moe_token_unpermute(returned, unpermute_indices[sender], scores[sender])
        outputs.append(output.reshape_as(inputs[sender]))
    return outputs


def _build_v41_ep_model(source: nn.Module, *, fused: bool, device: torch.device) -> nn.Module:
    """Build the V4.1 MoE model with enabled low_precision catalog and EP=2."""
    raw = deepseek_qat_config(weight_format="mxfp8", fused=fused).to_dict()
    raw["training"]["low_precision"]["enabled"] = True
    raw["plan_overrides"][0]["match"] = "mlp.experts"
    raw["plan_overrides"].append({
        "match": "mlp.forward",
        "when": "ep",
        "region_dispatch": False,
        "replace": {
            "kind": "method",
            "implementation": {
                "_target_": (
                    "hyper_parallel.models.deepseek_v41.adapter.distributed."
                    "moe_engram_expert_parallel.deepseek_v41_ep_compute_fn"
                ),
                "use_grouped_gemm": True,
                "fix_router": False,
            },
        },
    })
    config = resolve_config(raw)
    mesh = MeshContext(dp_size=2, dp_replicate_size=2, ep_size=2)
    mesh.build_meshs("npu", 2)
    setup = DistributedSetup(mesh_context=mesh, low_precision_config=config.training.low_precision)
    normalize_distributed_setup_overrides(setup, config)
    planner, fsdp_manager = instantiate_infrastructure(setup, device)
    return apply_model_infrastructure(
        source, mesh=mesh, sharding_planner=planner, fsdp2_manager=fsdp_manager,
        distributed_setup=setup, device=device,
    )


def _run_v41_ep_step(
    model: nn.Module, reference: nn.Module, optimizer: torch.optim.Optimizer,
    ref_optimizer: torch.optim.Optimizer, *, step: int, case: str, rank: int,
    fused: bool, limit: float, device: torch.device,
) -> float:
    """Execute one EP=2 training step and verify outputs, gradients, and updates."""
    optimizer.zero_grad(set_to_none=True)
    ref_optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(9100 + step)
    # Scale inputs so projections exceed swiglu_limit=10.0 and exercise clipping.
    inputs = [(torch.randn(1, 32, 128, device=device, dtype=torch.bfloat16) * 64).requires_grad_()
              for _ in range(2)]
    upstream = [torch.randn_like(value) for value in inputs]
    routes, scores = _v41_ep_routing(case, device)
    actual_input = inputs[rank].detach().clone().requires_grad_()
    actual_scores = scores[rank].detach().clone().requires_grad_()
    model.mlp.gate.indices, model.mlp.gate.scores = routes[rank], actual_scores

    output = model(actual_input)
    outputs = _v41_ep_reference(reference, inputs, routes, scores, fused=fused, limit=limit)
    torch.testing.assert_close(output, outputs[rank], rtol=0, atol=0)

    loss = (output.float() * upstream[rank].float()).sum()
    ref_loss = sum((val.float() * grad.float()).sum() for val, grad in zip(outputs, upstream))
    loss.backward()
    ref_loss.backward()

    expert = model.mlp.experts
    torch.testing.assert_close(actual_input.grad, inputs[rank].grad, rtol=0, atol=0)
    torch.testing.assert_close(actual_scores.grad, scores[rank].grad, rtol=0, atol=0)
    for actual, full in zip(expert.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, full.grad[rank * 2:(rank + 1) * 2], rtol=0, atol=0)
    if case == "empty_experts":
        assert all(torch.count_nonzero(param.grad[1]) == 0 for param in expert.parameters())
    if case == "empty_rank" and rank == 1:
        assert all(torch.count_nonzero(param.grad) == 0 for param in expert.parameters())

    optimizer.step()
    ref_optimizer.step()
    for actual, full in zip(expert.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, full[rank * 2:(rank + 1) * 2], rtol=0, atol=0)
    return loss.item()


@pytest.mark.parametrize("fused", [True, False])
def test_deepseek_v41_mxfp8_ep(fused: bool) -> None:
    """Verify fused and non-fused V4.1 clipped MXFP8 experts under EP=2 on Ascend A5."""
    device, backend = _device()
    if not dist.is_initialized():
        dist.init_process_group("hccl")
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    torch.manual_seed(2026)
    source = _V41TinyMoEModel().to(device=device, dtype=torch.bfloat16)
    limit = float(source.mlp.experts.limit)
    unsharded = copy.deepcopy(source)
    ref_cfg = deepseek_qat_config(weight_format="mxfp8", fused=True).to_dict()
    ref_cfg["plan_overrides"][0]["match"] = "mlp.experts"
    install_family(unsharded, resolve_config(ref_cfg))
    reference = unsharded.mlp.experts

    model = _build_v41_ep_model(source, fused=fused, device=device)
    expert = model.mlp.experts
    assert isinstance(expert, FusedMXFP8GroupedSwiGLU if fused else LowPrecisionGroupedExperts)
    assert expert.local_expert_count == 2
    assert expert.gate_up_proj.shape == (2, 128, 512)
    assert expert.down_proj.shape == (2, 256, 128)
    for actual, full in zip(expert.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, full[rank * 2:(rank + 1) * 2], rtol=0, atol=0)

    ops = CountingMXOps(backend)
    if fused:
        expert.quantizer = MXFP8Quantizer(npu_ops=ops)
    else:
        expert.grouped_linear.quantizer = MXFP8Quantizer(npu_ops=ops)

    optimizer = torch.optim.SGD(expert.parameters(), lr=1e-3, foreach=False, fused=False)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3, foreach=False, fused=False)
    initial = [param.detach().clone() for param in expert.parameters()]
    local_tokens, losses = [], []
    hook = expert.register_forward_pre_hook(lambda _mod, args: local_tokens.append(args[0].shape[0]))

    for step, case in enumerate(("balanced", "empty_experts", "empty_rank")):
        loss_val = _run_v41_ep_step(
            model, reference, optimizer, ref_optimizer,
            step=step, case=case, rank=rank, fused=fused, limit=limit, device=device,
        )
        if case == "empty_rank" and rank == 1:
            assert local_tokens[-1] == 0
        losses.append(loss_val)

    hook.remove()
    assert all(not torch.equal(before, after) for before, after in zip(initial, expert.parameters()))
    assert len(ops.calls) > 0
    if fused:
        expected_calls = 2 if rank == 1 else 3
        assert ops.fused_forward == expected_calls
        assert ops.fused_backward == expected_calls
    print(json.dumps({
        "v41_ep": 2, "rank": rank, "fused": fused, "clamp_limit": limit,
        "local_tokens": local_tokens, "gmm_calls": len(ops.calls),
        "fusion_calls": [ops.fused_forward, ops.fused_backward], "losses": losses,
    }), flush=True)
    dist.barrier()
