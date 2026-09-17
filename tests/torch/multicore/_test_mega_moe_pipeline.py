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
"""Two PP stages with noncontiguous EP subgroups and outstanding microbatch graphs."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch_npu
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.functional.grouped_matmul import grouped_matmul
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore import shmem
from tests.torch.multicore._mega_moe_utils import write_evidence

_TOKENS, _HIDDEN, _INTERMEDIATE = 128, 512, 128
_EXPERTS, _TOP_K, _LAYERS, _MICROBATCHES = 4, 2, 2, 4


def _random(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    """Generate identical values without depending on device-local RNG history."""
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(device, torch.bfloat16)


def _route(batch: int, layer: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Vary route order and probabilities between outstanding graphs."""
    positions = torch.arange(2 * _TOKENS * _TOP_K, device=device).reshape(2 * _TOKENS, _TOP_K)
    ids = ((positions + batch + layer) % _EXPERTS).to(torch.int32)
    probabilities = torch.empty((2 * _TOKENS, _TOP_K), dtype=torch.float32, device=device)
    probabilities[:, 0] = 0.25 + 0.1 * batch
    probabilities[:, 1] = 1.0 - probabilities[:, 0]
    return ids, probabilities


def _reference_moe(
    x: torch.Tensor, ids: torch.Tensor, probabilities: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Unsharded native oracle with the same BF16 permutation/reduction boundaries."""
    permuted, mapping = torch_npu.npu_moe_token_permute(x, ids)
    groups = torch.bincount(ids.flatten().long(), minlength=_EXPERTS).cumsum(0)
    gate_up = grouped_matmul(permuted, weights[0], group_list=groups)
    expert_output = grouped_matmul(torch_npu.npu_swiglu(gate_up), weights[1], group_list=groups)
    return torch_npu.npu_moe_token_unpermute(expert_output, mapping, probs=probabilities)


def _oracle(device: torch.device) -> tuple[list, list, list, list, list]:
    """Compute global outputs and gradients without any distributed communication."""
    weights = [
        (
            (_random((_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE), 100 + layer, device) * 0.02).requires_grad_(),
            (_random((_EXPERTS, _INTERMEDIATE, _HIDDEN), 200 + layer, device) * 0.02).requires_grad_(),
        )
        for layer in range(2 * _LAYERS)
    ]
    inputs, outputs, input_grads, probability_grads = [], [], [], []
    for batch in range(_MICROBATCHES):
        x = (_random((2 * _TOKENS, _HIDDEN), 300 + batch, device) * 0.1).requires_grad_()
        stage_inputs, stage_outputs, probabilities = [], [], []
        inputs.append(x.detach().clone())
        for stage in range(2):
            x.retain_grad()
            stage_inputs.append(x)
            for index in range(_LAYERS):
                layer = stage * _LAYERS + index
                ids, probs = _route(batch, layer, device)
                probs.requires_grad_()
                probabilities.append(probs)
                x = x + _reference_moe(x, ids, probs, weights[layer])
            stage_outputs.append(x.detach().clone())
        x.backward(_random(tuple(x.shape), 400 + batch, device))
        outputs.append(stage_outputs)
        input_grads.append([value.grad.detach().clone() for value in stage_inputs])
        probability_grads.append([value.grad.detach().clone() for value in probabilities])
    return weights, inputs, outputs, input_grads, probability_grads


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    """Compare both numerical values and finite gradients on every PP/EP rank."""
    assert torch.isfinite(actual).all(), f"rank={dist.get_rank()} nonfinite {name}"
    torch.testing.assert_close(
        actual, expected, rtol=2e-2, atol=2e-3,
        msg=lambda message: f"rank={dist.get_rank()} {name}: {message}",
    )


def _run_pipeline(use_checkpoint: bool) -> None:
    """Run warmup, FIFO 1F1B, cooldown and group-local teardown with shared resources."""
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch.npu.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("hccl")
    assert dist.get_world_size() == 4
    rank = dist.get_rank()
    stage, ep_rank = rank % 2, rank // 2
    ep_groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
    pp_groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
    ep_group, pp_group = ep_groups[stage], pp_groups[ep_rank]
    peer = rank + 1 if stage == 0 else rank - 1
    rows = slice(ep_rank * _TOKENS, (ep_rank + 1) * _TOKENS)
    expert_slice = slice(ep_rank * 2, (ep_rank + 1) * 2)
    weights, inputs, expected_outputs, expected_dx, expected_dp = _oracle(device)
    layers = [
        MegaMoeExperts(
            local_num_tokens=_TOKENS, hidden_size=_HIDDEN, intermediate_size=_INTERMEDIATE,
            num_experts=_EXPERTS, top_k=_TOP_K, ep_size=2, ep_group=ep_group,
        ).to(device=device, dtype=torch.bfloat16)
        for _ in range(_LAYERS)
    ]
    MegaMoeExperts.share_execution_resources(layers)
    with torch.no_grad():
        for index, layer in enumerate(layers):
            reference = weights[stage * _LAYERS + index]
            layer.gate_up_weight.copy_(reference[0][expert_slice])
            layer.down_weight.copy_(reference[1][expert_slice])
    pending = {}

    def forward(batch: int, value: torch.Tensor) -> torch.Tensor:
        x = value.detach().clone().requires_grad_()
        local_input, probabilities = x, []
        for index, layer in enumerate(layers):
            ids, probs = _route(batch, stage * _LAYERS + index, device)
            local_probs = probs[rows].clone().requires_grad_()
            probabilities.append(local_probs)
            if use_checkpoint:
                output = checkpoint(layer, x, ids[rows], local_probs, use_reentrant=False)
            else:
                output = layer(x, ids[rows], local_probs)
            x = x + output
        pending[batch] = (local_input, x, probabilities)
        _assert_close(x, expected_outputs[batch][stage][rows], f"output batch={batch}")
        return x

    def backward(batch: int, gradient: torch.Tensor) -> torch.Tensor:
        value, output, probabilities = pending.pop(batch)
        output.backward(gradient)
        _assert_close(value.grad, expected_dx[batch][stage][rows], f"dX batch={batch}")
        for index, probs in enumerate(probabilities):
            _assert_close(probs.grad, expected_dp[batch][stage * _LAYERS + index][rows], f"dP batch={batch}")
        return value.grad

    def exchange(send: torch.Tensor, receive: torch.Tensor) -> None:
        # Batch both directions so HCCL P2P launch order cannot deadlock the steady-state pair.
        requests = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send.detach().contiguous(), peer, pp_group),
            dist.P2POp(dist.irecv, receive, peer, pp_group),
        ])
        for request in requests:
            request.wait()

    receive = torch.empty((_TOKENS, _HIDDEN), dtype=torch.bfloat16, device=device)
    if stage == 0:
        # Stage 1 has not initialized SHMEM yet: a WORLD bootstrap barrier would deadlock here.
        output = forward(0, inputs[0][rows])
        dist.send(output.detach().contiguous(), dst=peer, group=pp_group)
        for batch in range(1, _MICROBATCHES):
            output = forward(batch, inputs[batch][rows])
            exchange(output, receive)
            backward(batch - 1, receive)
        dist.recv(receive, src=peer, group=pp_group)
        backward(_MICROBATCHES - 1, receive)
    else:
        dist.recv(receive, src=peer, group=pp_group)
        for batch in range(_MICROBATCHES):
            forward(batch, receive)
            gradient = _random((2 * _TOKENS, _HIDDEN), 400 + batch, device)[rows]
            dx = backward(batch, gradient)
            if batch + 1 < _MICROBATCHES:
                exchange(dx, receive)
            else:
                dist.send(dx.contiguous(), dst=peer, group=pp_group)
    assert not pending
    for index, layer in enumerate(layers):
        reference = weights[stage * _LAYERS + index]
        _assert_close(layer.gate_up_weight.grad, reference[0].grad[expert_slice], f"W1 gradient layer={index}")
        _assert_close(layer.down_weight.grad, reference[1].grad[expert_slice], f"W2 gradient layer={index}")
        layer.close()
    assert shmem.debug_state()["reference_count"] == 0
    write_evidence({
        "rank": rank, "stage": stage, "ep_ranks": dist.get_process_group_ranks(ep_group),
        "checkpoint": use_checkpoint, "microbatches": _MICROBATCHES, "shared_layers": _LAYERS,
        "outputs_and_all_gradients_match": True,
    })
    dist.destroy_process_group()


def test_mega_moe_subgroup_pipeline() -> None:
    """Validate independent subgroup bootstrap and multiple live forward graphs."""
    _run_pipeline(False)


def test_mega_moe_subgroup_pipeline_checkpoint() -> None:
    """Validate checkpoint replay and shared workspace reuse inside FIFO 1F1B."""
    _run_pipeline(True)
