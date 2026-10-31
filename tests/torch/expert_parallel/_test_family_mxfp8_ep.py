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
"""A5 EP=2 acceptance through the model builder, planner and Qwen3-MoE recipe."""

import copy
import json

import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F

from hyper_parallel.components.modules.mxfp8_grouped_swiglu import MXFP8GroupedSwiGLU
from hyper_parallel.components.quantization.functional.mxfp8_fused_grouped_swiglu import fused_grouped_swiglu
from hyper_parallel.components.quantization.functional.mxfp8_projection import mxfp8_projection
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.distributed.expert_parallel.experts import _argsort_keys
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.models._transformers.model_builder import apply_model_infrastructure, instantiate_infrastructure
from hyper_parallel.trainer.config.parallelism import normalize_distributed_setup_overrides
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.grouped_experts_family import TinyMoE, family_config, replace_experts
from tests.torch.expert_parallel._test_family_mxfp8_local import CountingMXOps, _device, _metric


_EP_FACTORY = "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn"
_STEPS = ("balanced", "empty_experts", "empty_rank")


def _build_model(source, fused, device):
    """Apply replacement and real planner sharding without manually slicing weights."""
    raw = family_config(use_mxfp8=True, use_fused_mxfp8=fused).to_dict()
    raw["plan_overrides"].append({
        "match": "mlp", "when": "ep", "region_dispatch": False,
        "local_compute_fn": {"_target_": _EP_FACTORY, "use_grouped_gemm": True},
    })
    config = resolve_config(raw)
    mesh = MeshContext(dp_size=2, dp_replicate_size=2, ep_size=2)
    mesh.build_meshs("npu", 2)
    setup = DistributedSetup(mesh_context=mesh)
    normalize_distributed_setup_overrides(setup, config)
    planner, fsdp_manager = instantiate_infrastructure(setup, device)
    return apply_model_infrastructure(
        source, mesh=mesh, sharding_planner=planner, fsdp2_manager=fsdp_manager,
        distributed_setup=setup, device=device,
    )


def _routing(case, device):
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
        values = [[.125 + (token % 3) * .125, .625 + rank * .125] for token in range(40)]
        scores.append(torch.tensor(values, device=device, dtype=torch.bfloat16, requires_grad=True))
    return routes, scores


def _reference_local(inputs, weights, counts, quantizer, mode):
    """Evaluate an owner-sized group using views of the full reference weights."""
    gate_up, down = weights
    if mode == "bf16":
        outputs = []
        for block, first, second in zip(inputs.split(counts.tolist()), gate_up, down):
            gate, up = (block @ first).chunk(2, -1)
            outputs.append((F.silu(gate) * up) @ second)
        return torch.cat(outputs)
    if mode == "fused":
        return fused_grouped_swiglu(inputs, gate_up, down, counts, quantizer)
    projected = mxfp8_projection(inputs, gate_up, counts, quantizer)
    gate, up = projected.chunk(2, -1)
    return mxfp8_projection(F.silu(gate) * up, down, counts, quantizer)


def _reference(experts, inputs, routes, scores, mode):
    """Reproduce EP grouping and reduction order without any communication.

    Each sender gathers input once, then each owner receives sender-rank-ordered
    segments. Sharing only the integer sort helper preserves the kernel's tie
    order, which otherwise changes MXFP8 columnwise quantization blocks.
    Weight views below belong only to this unsharded numerical reference.
    """
    dispatched, orders, token_indices, dispatch_indices, splits = [], [], [], [], []
    for states, indices in zip(inputs, routes):
        flat_indices = indices.flatten()
        order = _argsort_keys(flat_indices // 2 * 4 + flat_indices, bound=8)
        tokens = torch.arange(40, device=states.device).repeat_interleave(2)
        ordered_indices = flat_indices[order]
        orders.append(order)
        token_indices.append(tokens)
        dispatch_indices.append(ordered_indices)
        dispatched.append(states.reshape(-1, 128)[tokens[order]])
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
        order = orders[sender]
        weighted = returned * scores[sender].flatten()[order, None]
        output = torch.zeros_like(inputs[sender].reshape(-1, 128))
        output.index_add_(0, token_indices[sender][order], weighted)
        outputs.append(output.reshape_as(inputs[sender]))
    return outputs


def _exact(actual, expected):
    """Require bitwise-equivalent arithmetic after aligning token and group order."""
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _backward_twice(loss, variables):
    """Verify retained backward reconstructs usable physical saved tensors."""
    loss.backward(retain_graph=True)
    first = [value.grad.detach().clone() for value in variables]
    for value in variables:
        value.grad = None
    loss.backward()
    for value, previous in zip(variables, first):
        _exact(value.grad, previous)


def _compare_math(experts, inputs, routes, scores, expected, fused):
    """Reuse the existing single-device bounds only for quantization differences."""
    for mode in ("bf16", "plain" if fused else "fused"):
        reference_inputs = [value.detach().clone().requires_grad_() for value in inputs]
        reference_scores = [value.detach().clone().requires_grad_() for value in scores]
        outputs = _reference(experts, reference_inputs, routes, reference_scores, mode)
        for rank, (actual, reference) in enumerate(zip(expected, outputs)):
            _metric(f"EP2/fused={fused}/{mode}/rank={rank}/output", actual, reference)


def _check_partition(experts, reference, rank):
    """Inspect physical planner shards and compare their actual initial contents."""
    assert isinstance(experts, MXFP8GroupedSwiGLU)
    assert experts.local_expert_count == 2
    assert experts.gate_up_proj.shape == (2, 128, 512)
    assert experts.down_proj.shape == (2, 256, 128)
    for actual, full in zip(experts.parameters(), reference.parameters()):
        _exact(actual, full[rank * 2:(rank + 1) * 2])


def _run_training(model, reference, ops, device, fused):
    """Exercise cross-rank routing, sparse groups and a fully empty receiver."""
    rank = dist.get_rank()
    expert = model.mlp.experts
    optimizer = torch.optim.SGD(expert.parameters(), lr=.1, foreach=False, fused=False)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.1, foreach=False, fused=False)
    initial = [value.detach().clone() for value in expert.parameters()]
    losses, local_tokens = [], []
    hook = expert.register_forward_pre_hook(lambda _module, args: local_tokens.append(args[0].shape[0]))
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
            _compare_math(reference, inputs, routes, scores, outputs, fused)
        loss = (output.float() * upstream[rank].float()).sum()
        ref_loss = sum((value.float() * grad.float()).sum() for value, grad in zip(outputs, upstream))
        _backward_twice(loss, [actual_input, actual_scores, *expert.parameters()])
        _backward_twice(ref_loss, [*inputs, *scores, *reference.parameters()])
        _exact(actual_input.grad, inputs[rank].grad)
        _exact(actual_scores.grad, scores[rank].grad)
        for actual, full in zip(expert.parameters(), reference.parameters()):
            _exact(actual.grad, full.grad[rank * 2:(rank + 1) * 2])
        if case == "empty_experts":
            assert all(torch.count_nonzero(value.grad[1]) == 0 for value in expert.parameters())
        if case == "empty_rank" and rank == 1:
            assert local_tokens[-1] == 0
            assert all(torch.count_nonzero(value.grad) == 0 for value in expert.parameters())
        optimizer.step()
        ref_optimizer.step()
        _check_partition(expert, reference, rank)
        losses.append(loss.item())
    hook.remove()
    assert all(not torch.equal(before, after) for before, after in zip(initial, expert.parameters()))
    assert len(local_tokens) == 3
    assert len(ops.calls) > 0 and any(call[2] == 2 for call in ops.calls)
    if fused:
        expected_calls = 2 if rank == 1 else 3
        assert ops.fused_forward == expected_calls
        assert ops.fused_backward == expected_calls * 2
    print(json.dumps({"ep": 2, "rank": rank, "fused": fused, "local_tokens": local_tokens,
                      "losses": losses, "gmm_calls": len(ops.calls),
                      "fusion_calls": [ops.fused_forward, ops.fused_backward],
                      "reference": "same sender/expert order; rtol=atol=0"}), flush=True)


@pytest.mark.parametrize("fused", [False, True])
def test_family_mxfp8_ep(fused: bool) -> None:
    """Train both MXFP8 implementations through the EP=2 model builder.

    Args:
        fused: Whether to select the fused MXFP8 Family variant.
    """
    device, backend = _device()
    if not dist.is_initialized():
        dist.init_process_group("hccl")
    assert dist.get_world_size() == 2
    torch.manual_seed(793)
    source = TinyMoE(dtype=torch.bfloat16).to(device)
    unsharded = copy.deepcopy(source)
    replace_experts(unsharded, family_config())
    reference = unsharded.mlp.experts
    model = _build_model(source, fused, device)
    ops = CountingMXOps(backend)
    model.mlp.experts.quantizer = MXFP8Quantizer(npu_ops=ops)
    _check_partition(model.mlp.experts, reference, dist.get_rank())
    _run_training(model, reference, ops, device, fused)
    dist.barrier()
