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
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.compile import GraphCompiler, PassConfig
from hyper_parallel.compile.tracer.graph_tracer import trace_model_graph
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.expert_parallel.experts import bind_local_expert_forward, ep_routed_forward
from hyper_parallel.distributed.mesh import MeshContext
from tests.torch.utils import get_device_type


@dataclass(frozen=True)
class _Topology:
    """Dense/expert parallel degrees for one graph EP regression case."""

    dp_size: int
    tp_size: int
    ep_size: int
    edp_shard_size: int

    @property
    def world_size(self) -> int:
        """Return the launched world size (CP=PP=1)."""
        return self.dp_size * self.tp_size

    @property
    def num_experts(self) -> int:
        """Keep two experts per rank so EDP sharding stays non-singleton."""
        return self.ep_size * 2


class _RaggedMoE(nn.Module):
    """Small learned router, stacked experts and shared projection."""

    def __init__(
        self,
        group: object,
        *,
        num_experts: int = 4,
        ep_size: int = 2,
        use_grouped_gemm: bool = True,
    ) -> None:
        """Keep complete local experts and replicated router/shared weights."""
        super().__init__()
        self.group = group
        self.num_experts = num_experts
        local_expert_count = num_experts // ep_size
        self.gate = nn.Linear(64, num_experts, bias=False)
        self.shared = nn.Linear(64, 64, bias=False)
        self.experts = nn.Module()
        self.experts.num_experts = num_experts
        self.experts.gate_up_proj = nn.Parameter(torch.randn(local_expert_count, 128, 64) * 0.01)
        self.experts.down_proj = nn.Parameter(torch.randn(local_expert_count, 64, 64) * 0.01)
        bind_local_expert_forward(self, ep_size, use_grouped_gemm=use_grouped_gemm)

    def forward(self, hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Use per-call destinations while retaining learned router gradients."""
        logits = self.gate(hidden).float().reshape(-1, self.num_experts)
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


def _compare_models(reference, candidate, dtype, gradients, expert_mesh=None, dense_mesh=None):
    """Compare every parameter or gradient, including empty expert segments."""
    tolerances = {"rtol": 1e-5, "atol": 1e-6} if dtype == torch.float32 else {"rtol": 1e-2, "atol": 1e-3}
    for (name, left), (_, right) in zip(reference.named_parameters(), candidate.named_parameters()):
        expected, actual = (left.grad, right.grad) if gradients else (left, right)
        if expert_mesh is not None:
            if name.startswith("experts."):
                group = expert_mesh.get_group()
            elif dense_mesh is not None:
                group = dense_mesh.get_group()
            else:
                group = dist.group.WORLD
            expected = expected.chunk(dist.get_world_size(group), dim=0)[
                dist.get_rank(group)
            ]
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


def _reference_fsdp_gradient(gradient: torch.Tensor, *, group: object, scale: float, reduce_op: object) -> torch.Tensor:
    """Match eager FSDP's per-backward shard reduction before accumulation.

    Reconstructing the full reduced gradient keeps this reference independent
    of compiler parameter sharding. Averaging after microbatch accumulation
    changes BF16 rounding and models a different synchronization schedule.
    """
    degree = dist.get_world_size(group)
    gradient = gradient * scale
    shard = gradient.new_empty((gradient.shape[0] // degree, *gradient.shape[1:]))
    dist.reduce_scatter_tensor(shard, gradient.contiguous(), op=reduce_op, group=group)
    result = torch.empty_like(gradient, memory_format=torch.contiguous_format)
    dist.all_gather_into_tensor(result, shard, group=group)
    return result


def _install_reference_sync(model, expert_mesh, dense_mesh=None):
    """Use independent dense and expert groups on every reference backward.

    Dense parameters reduce over the FSDP shard axis and average out the TP
    loss replicas; expert parameters sum over EDP and divide by the shared
    ``dp*cp*tp`` loss domain, which stays valid when ``tp != ep``.
    """
    handles = []
    for name, parameter in model.named_parameters():
        expert = name.startswith("experts.")
        if expert:
            group = expert_mesh.get_group()
            scale = 1.0 / dist.get_world_size()
            reduce_op = dist.ReduceOp.SUM
        else:
            group = (
                dense_mesh.get_group() if dense_mesh is not None else dist.group.WORLD
            )
            scale = 1.0
            reduce_op = dist.ReduceOp.AVG
        handles.append(parameter.register_hook(
            partial(_reference_fsdp_gradient, group=group, scale=scale, reduce_op=reduce_op)))
    return handles


def _run_steps(reference, candidate, compiler, device, dtype, expert_mesh=None,
               topology=None, dense_mesh=None):
    """Reuse one captured graph across changing routes and ten SGD updates."""
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.01) for model in (reference, candidate)]
    records = []
    tp_size = topology.tp_size if topology is not None else 1
    num_experts = topology.num_experts if topology is not None else 4
    half = num_experts // 2
    route_pairs = [(index, index + half) for index in range(half)]
    for step in range(10):
        for optimizer in optimizers:
            optimizer.zero_grad()
        for microbatch in range(2):
            # TP ranks replicate the loss, so keep their routing inputs equal.
            torch.manual_seed(71 + step * 2 + microbatch + dist.get_rank() // tp_size)
            hidden = torch.randn(1, 8, 64, device=device, dtype=dtype)
            pair = torch.tensor(
                route_pairs[(step + microbatch) % len(route_pairs)], device=device
            )
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
        _compare_models(reference, candidate, dtype, gradients=True, expert_mesh=expert_mesh,
                        dense_mesh=dense_mesh)
        for optimizer in optimizers:
            optimizer.step()
        _compare_models(reference, candidate, dtype, gradients=False, expert_mesh=expert_mesh,
                        dense_mesh=dense_mesh)
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


def _run_case(dtype, sharded, topology=None):
    """Capture real collectives and replay changing zero/nonzero receive sizes."""
    device_type = get_device_type()
    device = torch.device(device_type)
    world_size = dist.get_world_size()
    context, expert_mesh, dense_mesh, source_shard_info = None, None, None, None
    group = dist.group.WORLD
    ep_size = 2
    if sharded:
        if topology is None:
            topology = _Topology(
                dp_size=world_size,
                tp_size=1,
                ep_size=2,
                edp_shard_size=world_size // 2,
            )
        if topology.world_size != world_size:
            raise AssertionError(
                f"topology world size {topology.world_size} != launched {world_size}"
            )
        ep_size = topology.ep_size
        context = MeshContext(
            dp_size=topology.dp_size,
            dp_shard_size=topology.dp_size,
            tp_size=topology.tp_size,
            ep_size=topology.ep_size,
            edp_shard_size=topology.edp_shard_size,
        )
        context.build_meshs(device_type, world_size)
        group = context.fsdp_moe_mesh["ep"].get_group()
        expert_mesh = context.fsdp_moe_mesh["edp_shard"]
        dense_mesh = context.fsdp_non_moe_mesh["fsdp_shard"]
    torch.manual_seed(42)
    num_experts = topology.num_experts if topology is not None else 4
    # Grouped GEMM is Ascend-only; the CPU fallback keeps the regression runnable.
    use_grouped_gemm = device_type != "cpu"
    reference = _RaggedMoE(
        group,
        num_experts=num_experts,
        ep_size=ep_size,
        use_grouped_gemm=use_grouped_gemm,
    ).to(device=device, dtype=dtype)
    candidate = _RaggedMoE(
        group,
        num_experts=num_experts,
        ep_size=ep_size,
        use_grouped_gemm=use_grouped_gemm,
    ).to(device=device, dtype=dtype)
    candidate.load_state_dict(reference.state_dict())
    torch.testing.assert_close(candidate.state_dict(), reference.state_dict(), rtol=0, atol=0)
    if sharded and topology.tp_size > 1:
        tp_mesh = context.fsdp_non_moe_mesh["tp"]
        ep_mesh = context.fsdp_moe_mesh["ep"]
        source_shard_info = {}
        for name, _ in reference.named_parameters():
            if name.startswith("experts."):
                source_shard_info[name] = ((Shard(0),), ep_mesh)
            else:
                source_shard_info[name] = ((Replicate(),), tp_mesh)
    compiler = GraphCompiler(
        candidate,
        _loss,
        PassConfig(fsdp_enabled=sharded, enable_overlap=False),
        device=device,
        mesh_context=context,
        source_shard_info=source_shard_info,
    )
    captured = []

    def _record_trace(*args, **kwargs):
        """Inspect the captured graph without depending on compiler internals."""
        joint = trace_model_graph(*args, **kwargs)
        captured.append(joint)
        return joint

    handles = _install_reference_sync(reference, expert_mesh, dense_mesh) if sharded else []
    try:
        with patch("hyper_parallel.compile.compiler.trace_model_graph", side_effect=_record_trace) as capture:
            records = _run_steps(reference, candidate, compiler, device, dtype, expert_mesh,
                                 topology, dense_mesh)
    finally:
        for handle in handles:
            handle.remove()
    assert capture.call_count == 1, f"Expected one joint capture, got {capture.call_count}"
    _verify_graph(captured[0].graph_module.code, records, device_type, sharded, dtype)


def _verify_graph(text, records, device_type, sharded, dtype):
    """Require native collective/GMM nodes and optionally persist the evidence."""
    assert "_c10d_functional" in text, f"Missing functional EP communication in graph: {text}"
    if device_type == "npu":
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_ragged_tp_ep_fsdp_joint_graph(dtype):
    """Verify DP(4)+TP(2)+EP(4) keeps the shared loss-domain divisor when TP != EP."""
    _run_case(
        dtype,
        sharded=True,
        topology=_Topology(dp_size=4, tp_size=2, ep_size=4, edp_shard_size=2),
    )
