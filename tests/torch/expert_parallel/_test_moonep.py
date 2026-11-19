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

"""Four-rank MoonEP parity with real replica traffic, FSDP and recomputation."""

import os
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import
from torch import nn
from torch.nn import functional

from hyper_parallel import fully_shard, init_device_mesh
from hyper_parallel.core.activation_memory.api import checkpoint
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.distributed.expert_parallel import experts as ep_experts
from hyper_parallel.distributed.expert_parallel import moonep
from hyper_parallel.distributed.expert_parallel.experts import bind_local_expert_forward


def _gate(value):
    gate, up = value.chunk(2, dim=-1)
    return (functional.silu(gate.float()) * up.float()).to(value.dtype)


class _Experts(nn.Module):
    """Input-major expert parameters with the same two projections as Qwen."""

    def __init__(self, gate, down):
        super().__init__()
        self.num_experts = 4
        self.is_transposed = True
        self.gate_up_proj = nn.Parameter(gate.clone())
        self.down_proj = nn.Parameter(down.clone())


class _Stack(nn.Module):
    """Run the root FSDP hooks around both expert layers."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self, value: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor, recompute: bool,
    ) -> torch.Tensor:
        """Evaluate both expert layers with optional checkpoint replay.

        Args:
            value: Source token vectors.
            ids: Selected expert indices.
            scores: Differentiable routing weights.
            recompute: Enable non-reentrant checkpointing per block.
        """
        for block in self.blocks:
            value = (checkpoint(block, value, ids, scores, use_reentrant=False)
                     if recompute else block(value, ids, scores))
        return value


class _Block(nn.Module):
    """Keep the router scores as explicit differentiable test inputs."""

    def __init__(self, gate, down, rank, group, dedup, depth):
        super().__init__()
        self.experts = _Experts(gate.chunk(2)[rank], down.chunk(2)[rank])
        self.group = group
        bind_local_expert_forward(self, 2, use_moonep=True, apply_gate=_gate,
                                  moonep_token_dedup=dedup, moonep_prefetch_depth=depth)

    def forward(self, hidden: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Evaluate the residual expert branch.

        Args:
            hidden: Source token vectors.
            ids: Selected expert indices.
            scores: Differentiable routing weights.
        """
        return hidden + ep_experts._moonep_routed_forward(self.experts, hidden, ids, scores, self.group)


def _reference(hidden, ids, scores, gate, down):
    values = []
    for token in range(hidden.shape[0]):
        terms = []
        for choice in range(ids.shape[1]):
            expert = int(ids[token, choice])
            terms.append(((_gate(hidden[token] @ gate[expert]) @ down[expert]) * scores[token, choice])
                         .to(hidden.dtype))
        values.append(sum(terms))
    return hidden + torch.stack(values)


def _close(actual, expected, label):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    # BF16 grouped reductions and owner accumulation round in a different order from dense GEMMs.
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=4e-3 if label == "output" else 6e-5,
                               msg=lambda message: f"{label}: {message}")


def _build_models(originals, ep_rank, group, dedup, depth, mesh, fsdp):
    """Create identical expert parameters with optional FSDP wrapping."""
    model = _Stack([_Block(gate, down, ep_rank, group, dedup, depth) for gate, down in originals])
    reference = [(gate.clone().requires_grad_(), down.clone().requires_grad_()) for gate, down in originals]
    if fsdp:
        for block in model.blocks:
            fully_shard(block.experts, mesh=mesh["dp"], reshard_after_forward=True)
        fully_shard(model, mesh=mesh["dp"], reshard_after_forward=True)
    return model, reference


def _check_parameter_gradients(model, reference, group, ep_rank, fsdp):
    """Compare accumulated owner gradients against the globally reduced oracle."""
    for block, (gate, down) in zip(model.blocks, reference):
        for parameter, expected in zip(block.experts.parameters(), (gate, down)):
            dist.all_reduce(expected.grad, group=None if fsdp else group)
            gradient = parameter.grad
            if isinstance(gradient, DTensor):
                gradient = gradient.full_tensor()
            target = expected.grad.chunk(2)[ep_rank] / (2 if fsdp else 1)
            _close(gradient, target, "owner parameter gradient")


def test_moonep_distributed():
    """Compare all gradients over two outstanding microbatches and two expert layers."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("dp", "ep"))
    group = mesh["ep"].get_group()
    ep_rank = dist.get_rank(group)
    device = torch.device("npu", local_rank)
    ids = torch.tensor([[0, 2] if index % 4 == 0 else [0, 1] for index in range(16)], device=device)
    for fsdp, recompute, dedup, depth in (
            (False, False, False, 0), (False, False, True, 0),
            (False, True, True, 1), (False, True, True, 2),
            (True, False, True, 1), (True, True, True, 1),
            (True, True, False, 1)):
        torch.manual_seed(113)
        originals = [(torch.randn(4, 64, 128, device=device, dtype=torch.bfloat16) * 0.04,
                      torch.randn(4, 64, 64, device=device, dtype=torch.bfloat16) * 0.04) for _ in range(2)]
        model, reference = _build_models(originals, ep_rank, group, dedup, depth, mesh, fsdp)
        inputs, expected_inputs, scores, expected_scores = [], [], [], []
        actual_outputs, expected_outputs = [], []
        torch.manual_seed(127 + dist.get_rank())
        with (patch.object(ep_experts, "compile_plan", wraps=ep_experts.compile_plan) as planner,
              patch.object(moonep, "_prefetch", wraps=moonep._prefetch) as prefetch):
            for _ in range(2):
                inputs.append(torch.randn(16, 64, device=device, dtype=torch.bfloat16).requires_grad_())
                expected_inputs.append(inputs[-1].detach().clone().requires_grad_())
                score_dtype = torch.float32 if depth == 2 else torch.bfloat16
                scores.append(torch.randn(16, 2, device=device).softmax(-1).to(score_dtype).requires_grad_())
                expected_scores.append(scores[-1].detach().clone().requires_grad_())
                actual = model(inputs[-1], ids, scores[-1], recompute)
                expected = expected_inputs[-1]
                for gate, down in reference:
                    expected = _reference(expected, ids, expected_scores[-1], gate, down)
                actual_outputs.append(actual)
                expected_outputs.append(expected)
            sum(value.float().square().mean() for value in actual_outputs).backward()
            if prefetch.call_count != 8:
                raise AssertionError(f"Expected 4 forward + 4 backward weight exchanges, got {prefetch.call_count}")
            if planner.call_count != 4:
                raise AssertionError(f"Expected 4 original plans, got {planner.call_count}")
        sum(value.float().square().mean() for value in expected_outputs).backward()
        for actual, expected in zip(actual_outputs, expected_outputs):
            _close(actual, expected, "output")
        for actual, expected in zip(inputs + scores, expected_inputs + expected_scores):
            _close(actual.grad, expected.grad, "input/router gradient")
        _check_parameter_gradients(model, reference, group, ep_rank, fsdp)
        torch.npu.synchronize()
        if dist.get_rank() == 0:
            print(f"PASS fsdp={fsdp} recompute={recompute} dedup={dedup} depth={depth}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    test_moonep_distributed()
