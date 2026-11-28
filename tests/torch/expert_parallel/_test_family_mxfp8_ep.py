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
"""Ascend A5 EP=2 acceptance through native GroupedExperts ModuleFamily across all low-precision formats."""

import copy
import json

import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F

from hyper_parallel.components.functional.moe_token_permute import moe_token_permute
from hyper_parallel.components.functional.moe_token_unpermute import moe_token_unpermute
from hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts import (
    EPLowPrecisionGroupedExperts,
    MoonEPLowPrecisionGroupedExperts,
)
from hyper_parallel.components.quantization.functional.mxfp8_fused_grouped_swiglu import fused_grouped_swiglu
from hyper_parallel.components.quantization.functional.mxfp8_projection import mxfp8_projection
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.distributed.expert_parallel.experts import _argsort_keys
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.models._transformers.model_builder import apply_model_infrastructure, instantiate_infrastructure
from hyper_parallel.trainer.config.parallelism import normalize_distributed_setup_overrides
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.grouped_experts_family import TinyMoE, family_config, replace_experts
from tests.torch.expert_parallel._test_family_mxfp8_local import CountingMXOps, _device, _metric

_STEPS = ("balanced", "empty_experts", "empty_rank")


def _local_tensor(param: torch.Tensor) -> torch.Tensor:
    """Return the rank-local physical shard for DTensor or plain Tensor parameters."""
    return param.to_local() if isinstance(param, DTensor) else param


def _build_ep_model(source: TinyMoE, raw_config: dict, device: torch.device):
    """Apply native ModuleFamily replacement, Shard(0) planner sharding, and EPContext binding."""
    config = resolve_config(raw_config)
    mesh = MeshContext(dp_size=2, dp_replicate_size=2, ep_size=2)
    mesh.build_meshs("npu", 2)
    setup = DistributedSetup(mesh_context=mesh, low_precision_config=config.training.low_precision)
    normalize_distributed_setup_overrides(setup, config)
    planner, fsdp_manager = instantiate_infrastructure(setup, device)
    return apply_model_infrastructure(
        source, mesh=mesh, sharding_planner=planner, fsdp2_manager=fsdp_manager,
        distributed_setup=setup, device=device,
    )


def _routing(case: str, device: torch.device):
    """Use nonuniform, nonunit scores so applying scores twice cannot pass."""
    routes, scores = [], []
    for rank in range(2):
        if case == "balanced":
            indices = [[(token + rank) % 4, (token + rank + 2) % 4] for token in range(40)]
        elif case == "empty_experts":
            indices = [[0, 2]] * 40
        else:
            indices = [[0, 1]] * 40
        routes.append(torch.tensor(indices, device=device, dtype=torch.int64))
        values = [[0.125 + (token % 3) * 0.125, 0.625 + rank * 0.125] for token in range(40)]
        scores.append(torch.tensor(values, device=device, dtype=torch.bfloat16, requires_grad=True))
    return routes, scores


def _reference_local(inputs, weights, counts, quantizer, mode):
    """Evaluate an owner-sized group using canonical [C, 2I, H] / [C, H, I] reference weights."""
    gate_up, down = weights
    gate_up_t, down_t = gate_up.transpose(-2, -1), down.transpose(-2, -1)
    if mode == "bf16":
        outputs = []
        for block, first, second in zip(inputs.split(counts.tolist()), gate_up_t, down_t):
            gate, up = (block @ first).chunk(2, -1)
            outputs.append((F.silu(gate) * up) @ second)
        return torch.cat(outputs) if outputs else inputs.new_empty((0, down_t.shape[-1]))
    if mode == "fused":
        return fused_grouped_swiglu(inputs, gate_up_t, down_t, counts, quantizer)
    projected = mxfp8_projection(inputs, gate_up_t, counts, quantizer)
    gate, up = projected.chunk(2, -1)
    return mxfp8_projection(F.silu(gate) * up, down_t, counts, quantizer)


def _reference(experts, inputs, routes, scores, mode):
    """Reproduce EPContext slot dispatch, local expert grouping, and combine order without collectives."""
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
        counts = torch.stack([(local_indices == index).sum() for index in range(2)])
        expert_slice = slice(owner * 2, (owner + 1) * 2)
        weights = experts.gate_up_proj[expert_slice], experts.down_proj[expert_slice]
        sorted_output = _reference_local(received[order], weights, counts, quantizer, mode)
        received_output = torch.empty_like(sorted_output)
        received_output[order] = sorted_output
        returned = received_output.split([splits[sender][owner] for sender in range(2)])
        for sender, values in enumerate(returned):
            sender_outputs[sender].append(values)
    outputs = []
    for sender, parts in enumerate(sender_outputs):
        returned = torch.cat(parts)
        output = moe_token_unpermute(returned, unpermute_indices[sender], scores[sender])
        outputs.append(output.reshape_as(inputs[sender]))
    return outputs


def _exact(actual, expected):
    """Require bitwise-equivalent arithmetic after aligning token and group order."""
    torch.testing.assert_close(_local_tensor(actual), _local_tensor(expected), rtol=0, atol=0)


def _check_partition(experts, reference, rank):
    """Inspect physical Shard(0) parameters and compare their initial contents."""
    assert isinstance(experts, EPLowPrecisionGroupedExperts)
    gate_local = _local_tensor(experts.gate_up_proj)
    down_local = _local_tensor(experts.down_proj)
    assert gate_local.shape == (2, 512, 128)
    assert down_local.shape == (2, 128, 256)
    for actual, full in zip((gate_local, down_local), reference.parameters()):
        _exact(actual, full[rank * 2:(rank + 1) * 2])


@pytest.mark.parametrize("fused", [False, True])
def test_family_mxfp8_ep(fused: bool) -> None:
    """Train both MXFP8 implementations through native ModuleFamily EP=2 without method overrides."""
    device, backend = _device()
    if not dist.is_initialized():
        dist.init_process_group("hccl")
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    torch.manual_seed(793)
    source = TinyMoE(dtype=torch.bfloat16).to(device)
    unsharded = copy.deepcopy(source)
    replace_experts(unsharded, family_config())
    reference = unsharded.mlp.experts

    raw = family_config(use_mxfp8=True, use_fused_mxfp8=fused, overlap_shared_expert=False).to_dict()
    model = _build_ep_model(source, raw, device)
    expert = model.mlp.experts
    ops = CountingMXOps(backend)
    expert.quantizer = MXFP8Quantizer(npu_ops=ops)
    _check_partition(expert, reference, rank)

    local_tokens: list[int] = []
    orig_compute = expert.compute_experts

    def tracked_compute(hidden_states, tokens_per_expert, gate_up_proj, down_proj, routing_weights=None):
        local_tokens.append(int(hidden_states.shape[0]))
        return orig_compute(hidden_states, tokens_per_expert, gate_up_proj, down_proj, routing_weights)

    expert.compute_experts = tracked_compute
    optimizer = torch.optim.SGD(expert.parameters(), lr=0.1, foreach=False, fused=False)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, foreach=False, fused=False)
    losses = []

    for step, case in enumerate(_STEPS):
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(7930 + step)
        inputs = [torch.randn(1, 40, 128, dtype=torch.bfloat16).to(device).requires_grad_() for _ in range(2)]
        upstream = [torch.randn_like(value) for value in inputs]
        routes, scores = _routing(case, device)
        actual_input = inputs[rank].detach().clone().requires_grad_()
        actual_scores = scores[rank].detach().clone().requires_grad_()
        model.mlp.gate.indices, model.mlp.gate.scores = routes[rank], actual_scores

        output = model(actual_input)
        outputs = _reference(reference, inputs, routes, scores, "fused" if fused else "plain")
        _exact(output, outputs[rank])
        if step == 0:
            bf16_outputs = _reference(
                reference, [v.detach() for v in inputs], routes, [s.detach() for s in scores], "bf16",
            )
            _metric(f"EP2/fused={fused}/bf16/rank={rank}/output", output, bf16_outputs[rank])

        loss = (output.float() * upstream[rank].float()).sum()
        ref_loss = sum((value.float() * grad.float()).sum() for value, grad in zip(outputs, upstream))
        loss.backward()
        ref_loss.backward()

        _exact(actual_input.grad, inputs[rank].grad)
        _exact(actual_scores.grad, scores[rank].grad)
        for actual_p, full_p in zip(expert.parameters(), reference.parameters()):
            _exact(_local_tensor(actual_p.grad), full_p.grad[rank * 2:(rank + 1) * 2])
        if case == "empty_rank" and rank == 1:
            assert local_tokens[-1] == 0
        optimizer.step()
        ref_optimizer.step()
        _check_partition(expert, reference, rank)
        losses.append(loss.item())

    assert len(local_tokens) == 3
    assert len(ops.calls) > 0
    if fused:
        expected_calls = 2 if rank == 1 else 3
        assert ops.fused_forward == expected_calls
        assert ops.fused_backward == expected_calls
    print(json.dumps({
        "ep": 2, "rank": rank, "fused": fused, "local_tokens": local_tokens,
        "losses": losses, "gmm_calls": len(ops.calls),
    }), flush=True)
    dist.barrier()


@pytest.mark.parametrize(
    "scheme_name,scheme_cfg,moonep,fused",
    [
        ("w8a8", {"weight_format": "mxfp8", "act_format": "mxfp8"}, False, False),
        ("w4a8", {"weight_format": "mxfp4", "act_format": "mxfp8", "is_fake_quantize": False}, False, False),
        ("fake_w4a8", {"weight_format": "mxfp4", "act_format": "mxfp8", "is_fake_quantize": True}, False, False),
        ("hif8", {"weight_format": "hif8", "act_format": "hif8"}, False, False),
        ("mxfp8_moonep", {"weight_format": "mxfp8", "act_format": "mxfp8"}, True, False),
        ("fused_mxfp8_moonep", {"weight_format": "mxfp8", "act_format": "mxfp8"}, True, True),
        ("w4a8_moonep", {"weight_format": "mxfp4", "act_format": "mxfp8", "is_fake_quantize": False}, True, False),
        ("fake_w4a8_moonep", {"weight_format": "mxfp4", "act_format": "mxfp8", "is_fake_quantize": True}, True, False),
        ("hif8_moonep", {"weight_format": "hif8", "act_format": "hif8"}, True, False),
    ],
)
def test_all_schemes_native_ep_and_moonep_a5(
    scheme_name: str, scheme_cfg: dict, moonep: bool, fused: bool,
) -> None:
    """Verify MXFP8, Fused MXFP8, W4A8, Fake-W4A8, and HiF8 under native EP=2 and MoonEP on Ascend A5."""
    device, _ = _device()
    if not dist.is_initialized():
        dist.init_process_group("hccl")
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    torch.manual_seed(2026)
    source = TinyMoE(dtype=torch.bfloat16).to(device)
    bf16_ref = copy.deepcopy(source)
    replace_experts(bf16_ref, family_config())

    if fused:
        raw = family_config(
            use_mxfp8=True,
            use_fused_mxfp8=True,
            moonep=moonep,
            overlap_shared_expert=False,
        ).to_dict()
    else:
        raw = family_config(
            low_precision_dtype_scheme="experts",
            moonep=moonep,
            overlap_shared_expert=False,
        ).to_dict()
        raw["training"]["low_precision"] = {"dtype_schemes": {"experts": scheme_cfg}}
    model = _build_ep_model(source, raw, device)
    expert = model.mlp.experts
    expected_cls = MoonEPLowPrecisionGroupedExperts if moonep else EPLowPrecisionGroupedExperts
    assert isinstance(expert, expected_cls)

    max_rel_l2 = 0.25 if scheme_cfg["weight_format"] == "mxfp4" else 0.10
    cases = ("balanced", "empty_rank") if moonep else ("balanced",)
    for step_idx, routing_case in enumerate(cases):
        expert.zero_grad(set_to_none=True)
        routes, scores = _routing(routing_case, device)
        torch.manual_seed(9900 + step_idx * 10 + rank)
        inp = torch.randn(1, 40, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
        ref_inp = inp.detach().clone().requires_grad_()
        model.mlp.gate.indices, model.mlp.gate.scores = routes[rank], scores[rank]
        bf16_ref.mlp.gate.indices, bf16_ref.mlp.gate.scores = routes[rank], scores[rank].detach()

        out = model(inp)
        ref_out = bf16_ref(ref_inp)
        rel_l2 = ((out.float() - ref_out.float()).norm() / ref_out.float().norm()).item()
        assert rel_l2 < max_rel_l2, f"{scheme_name}/{routing_case} rel_l2={rel_l2}"
        loss = out.float().square().mean()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(inp.grad).all()
        assert all(p.grad is not None and torch.isfinite(_local_tensor(p.grad)).all() for p in expert.parameters())
        print(json.dumps({
            "scheme": scheme_name, "moonep": moonep, "case": routing_case, "rank": rank,
            "bf16_rel_l2": round(rel_l2, 6), "loss": round(loss.item(), 6),
        }), flush=True)
    dist.barrier()
