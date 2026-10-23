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
"""Distributed regression for the native ragged EP joint graph."""

import json
import os
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.compile import GraphCompiler, PassConfig
from hyper_parallel.compile.tracer.graph_tracer import trace_model_graph
from hyper_parallel.distributed.expert_parallel.experts import bind_local_expert_forward, ep_routed_forward
from hyper_parallel.distributed.mesh import MeshContext
from tests.torch.utils import get_device_type


class _RaggedMoE(nn.Module):
    """Small learned router, stacked experts and shared projection."""

    def __init__(self, group: object) -> None:
        """Keep complete local experts and replicated router/shared weights."""
        super().__init__()
        self.group = group
        self.gate = nn.Linear(64, 4, bias=False)
        self.shared = nn.Linear(64, 64, bias=False)
        self.experts = nn.Module()
        self.experts.num_experts = 4
        self.experts.gate_up_proj = nn.Parameter(torch.randn(2, 128, 64) * 0.01)
        self.experts.down_proj = nn.Parameter(torch.randn(2, 64, 64) * 0.01)
        bind_local_expert_forward(self, 2, use_grouped_gemm=True)

    def forward(self, hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Use per-call destinations while retaining learned router gradients."""
        logits = self.gate(hidden).float().reshape(-1, 4)
        preference = torch.zeros_like(logits).scatter(1, ids, 100.0)
        weights, selected = (logits + preference).softmax(dim=-1).topk(2, dim=-1)

        def router(dummy_module: nn.Module, dummy_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Pass differentiable scores and this call's expert IDs."""
            del dummy_module, dummy_states
            return selected, weights

        return ep_routed_forward(self, hidden, router_fn=router, ep_group=self.group) + self.shared(hidden)


def _loss(model, hidden, ids):
    """Return loss and outputs for eager/graph comparison."""
    output = model(hidden, ids)
    return output.float().square().mean(), {"output": output.detach()}


def _compare_models(reference, candidate, dtype, gradients, expert_mesh=None):
    """Compare every parameter or gradient, including empty expert segments."""
    tolerances = {"rtol": 1e-5, "atol": 1e-6} if dtype == torch.float32 else {"rtol": 1e-2, "atol": 1e-3}
    for (name, left), (_, right) in zip(reference.named_parameters(), candidate.named_parameters()):
        expected, actual = (left.grad, right.grad) if gradients else (left, right)
        if expert_mesh is not None:
            group = expert_mesh.get_group() if name.startswith("experts.") else dist.group.WORLD
            expected = expected.chunk(dist.get_world_size(group), dim=0)[dist.get_rank(group)]
        torch.testing.assert_close(actual, expected, **tolerances,
                                   msg=lambda message, label=name: f"{label}: {message}")


def _average_gradients(model):
    """Average replicated dense gradients when graph FSDP is disabled."""
    for name, parameter in model.named_parameters():
        if name.startswith("experts."):
            continue
        gradient = parameter.grad.contiguous()
        dist.all_reduce(gradient)
        parameter.grad.copy_(gradient.div_(dist.get_world_size()))


def _reference_fsdp_gradient(gradient: torch.Tensor, *, group: object) -> torch.Tensor:
    """Match eager FSDP's per-backward shard reduction before accumulation.

    Reconstructing the full averaged gradient keeps this reference independent
    of compiler parameter sharding. Averaging after microbatch accumulation
    changes BF16 rounding and models a different synchronization schedule.
    """
    degree = dist.get_world_size(group)
    shard = gradient.new_empty((gradient.shape[0] // degree, *gradient.shape[1:]))
    dist.reduce_scatter_tensor(shard, gradient.contiguous(), op=dist.ReduceOp.AVG, group=group)
    result = torch.empty_like(gradient, memory_format=torch.contiguous_format)
    dist.all_gather_into_tensor(result, shard, group=group)
    return result


def _install_reference_sync(model, expert_mesh):
    """Use independent dense and expert groups on every reference backward."""
    handles = []
    for name, parameter in model.named_parameters():
        group = expert_mesh.get_group() if name.startswith("experts.") else dist.group.WORLD
        handles.append(parameter.register_hook(partial(_reference_fsdp_gradient, group=group)))
    return handles


def _run_steps(reference, candidate, compiler, device, dtype, expert_mesh=None):
    """Reuse one captured graph across changing routes and ten SGD updates."""
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.01) for model in (reference, candidate)]
    records = []
    for step in range(10):
        for optimizer in optimizers:
            optimizer.zero_grad()
        for microbatch in range(2):
            torch.manual_seed(71 + step * 2 + microbatch + dist.get_rank())
            hidden = torch.randn(1, 8, 64, device=device, dtype=dtype)
            pair = torch.tensor([0, 1] if (step + microbatch) % 2 == 0 else [2, 3], device=device)
            ids = pair.expand(8, 2).contiguous()
            expected, details = _loss(reference, hidden, ids)
            expected.backward()
            actual, outputs = compiler.forward_backward(hidden=hidden, ids=ids)
            torch.testing.assert_close(actual, expected.detach(), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(outputs["output"], details["output"], rtol=1e-3, atol=1e-4)
            records.append({"step": step, "microbatch": microbatch,
                            "loss": float(actual), "eager_loss": float(expected.detach())})
        if expert_mesh is None:
            _average_gradients(reference)
            _average_gradients(candidate)
        _compare_models(reference, candidate, dtype, gradients=True, expert_mesh=expert_mesh)
        for optimizer in optimizers:
            optimizer.step()
        _compare_models(reference, candidate, dtype, gradients=False, expert_mesh=expert_mesh)
    return records


@pytest.fixture(scope="module", autouse=True)
def _distributed_world():
    """Keep one communicator for all dtypes in this worker process."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    device_type = get_device_type()
    if device_type == "npu":
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="hccl" if device_type == "npu" else "gloo")
    yield
    dist.destroy_process_group()
    torch.use_deterministic_algorithms(deterministic)


def _run_case(dtype, sharded):
    """Capture real collectives and replay changing zero/nonzero receive sizes."""
    device_type = get_device_type()
    device = torch.device(device_type)
    context, expert_mesh = None, None
    group = dist.group.WORLD
    if sharded:
        degree = dist.get_world_size()
        context = MeshContext(dp_size=degree, dp_shard_size=degree, ep_size=2, edp_shard_size=degree // 2)
        context.build_meshs(device_type, degree)
        group = context.fsdp_moe_mesh["ep"].get_group()
        expert_mesh = context.fsdp_moe_mesh["edp_shard"]
    torch.manual_seed(42)
    reference = _RaggedMoE(group).to(device=device, dtype=dtype)
    candidate = _RaggedMoE(group).to(device=device, dtype=dtype)
    candidate.load_state_dict(reference.state_dict())
    torch.testing.assert_close(candidate.state_dict(), reference.state_dict(), rtol=0, atol=0)
    compiler = GraphCompiler(candidate, _loss, PassConfig(fsdp_enabled=sharded, enable_overlap=False),
                             device=device, mesh_context=context)
    captured = []

    def _record_trace(*args, **kwargs):
        """Inspect the captured graph without depending on compiler internals."""
        joint = trace_model_graph(*args, **kwargs)
        captured.append(joint)
        return joint

    handles = _install_reference_sync(reference, expert_mesh) if sharded else []
    try:
        with patch("hyper_parallel.compile.compiler.trace_model_graph", side_effect=_record_trace) as capture:
            records = _run_steps(reference, candidate, compiler, device, dtype, expert_mesh)
    finally:
        for handle in handles:
            handle.remove()
    assert capture.call_count == 1, f"Expected one joint capture, got {capture.call_count}"
    _verify_graph(captured[0].graph_module.code, records, device_type, sharded, dtype)


def _verify_graph(text, records, device_type, sharded, dtype):
    """Require native collective/GMM nodes and optionally persist the evidence."""
    assert "_c10d_functional" in text, f"Missing functional EP communication in graph: {text}"
    assert "npu_grouped_matmul" in text, f"Missing grouped expert computation in graph: {text}"
    assert "alltoall_base_" not in text, f"Unexpected imperative collective in graph: {text}"
    output = os.environ.get("HP_EP_GRAPH_TEST_OUTPUT")
    if output:
        root = Path(output)
        label = f"{device_type}-fsdp{sharded}-{dtype}-rank{dist.get_rank()}"
        (root / f"{label}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        (root / f"{label}-graph.txt").write_text(text, encoding="utf-8")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_ragged_joint_graph(dtype):
    """Verify two-rank EP without data-parallel graph transformations."""
    _run_case(dtype, sharded=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_ragged_fsdp_joint_graph(dtype):
    """Verify distinct dense and expert sharding/reduction groups on four ranks."""
    _run_case(dtype, sharded=True)
