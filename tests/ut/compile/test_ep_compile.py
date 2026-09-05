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
"""Unit tests for capture-first static expert parallelism."""
# Backend selection must happen before importing HyperParallel.
# pylint: disable=protected-access,wrong-import-position

import copy
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import functional

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.distributed.expert_parallel import experts as ep_utils
from hyper_parallel.distributed.expert_parallel import recipes as ep_compute
from hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel import qwen3moe_ep_compute_fn
from hyper_parallel.distributed.expert_parallel.routing import MOE_ROUTER_ADAPTERS
from hyper_parallel.distributed.expert_parallel.experts import (
    resolve_swiglu_weights,
)
from hyper_parallel.compile.ep_capture import (
    _exchange_equal_capacity,
    capture_dynamic_ep,
    static_ep_routed_forward,
)
from hyper_parallel.compile.parallel_config import PassConfig
from hyper_parallel.compile.passes.parallel.ep_pass import ExpertParallelPass
from hyper_parallel.compile.passes.pipeline import PassPipeline


def _graph_module(
    ep_collective_count: int = 0,
    other_collective_count: int = 0,
    expected_collective_count: int = 5,
    include_metadata: bool = True,
) -> torch.fx.GraphModule:
    """Build an FX graph with explicit EP and non-EP All-to-All nodes."""
    graph = torch.fx.Graph()
    tensor = graph.placeholder("tensor")
    splits = [1, 1]
    for _ in range(ep_collective_count):
        tensor = graph.call_function(
            torch.ops._c10d_functional.all_to_all_single.default,
            (tensor, splits, splits, "ep_group"),
        )
    for _ in range(other_collective_count):
        tensor = graph.call_function(
            torch.ops._c10d_functional.all_to_all_single.default,
            (tensor, splits, splits, "other_group"),
        )
    graph.output(tensor)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    if include_metadata:
        graph_module.ep_capture_metadata = {
            "group_names": ("ep_group",),
            "expected_collective_count": expected_collective_count,
        }
    return graph_module


class _FakeExperts(torch.nn.Module):
    """Minimal expert holder exposing the dynamic-EP sharding contract."""

    def __init__(self, local_expert_count: int, global_expert_count: int) -> None:
        """Store the local and global expert counts used by capture validation."""
        super().__init__()
        self.local_expert_count = local_expert_count
        self.num_experts = global_expert_count


class _FakeMoe(torch.nn.Module):
    """Minimal MoE module containing an expert holder."""

    def __init__(
        self, local_expert_count: int = 2, global_expert_count: int = 4
    ) -> None:
        """Build a fake dynamic-EP module."""
        super().__init__()
        self.experts = _FakeExperts(local_expert_count, global_expert_count)





class _FakeEpGroup:
    """Minimal process-group contract used by mocked EP collectives."""

    def __init__(self, size: int = 2) -> None:
        """Store the mocked EP world size."""
        self._size = size
        self.group_name = "ep_group"

    def size(self) -> int:
        """Return the mocked EP world size."""
        return self._size


class _FakeEpMesh:
    """Minimal expert mesh used by the AutoModels EP factories."""

    def __init__(self, group: _FakeEpGroup) -> None:
        """Store the mocked EP process group."""
        self._group = group

    def __getitem__(self, mesh_dim: str) -> "_FakeEpMesh":
        """Return the EP submesh used to query its size."""
        if mesh_dim != "ep":
            raise KeyError(mesh_dim)
        return self

    def get_group(self, mesh_dim: str) -> _FakeEpGroup:
        """Return the EP process group."""
        if mesh_dim != "ep":
            raise ValueError(f"Expected ep mesh dimension, got {mesh_dim}")
        return self._group

    def size(self) -> int:
        """Return the mocked EP world size."""
        return self._group.size()


class _ThreadedAllToAll:
    """Differentiable two-rank All-to-All simulator without a process group."""

    def __init__(self, world_size: int) -> None:
        """Initialize per-thread rank state and collective rendezvous maps."""
        self._world_size = world_size
        self._condition = threading.Condition()
        self._local = threading.local()
        self._inputs = {}
        self._outputs = {}

    @property
    def call_counts(self) -> list[int]:
        """Return the number of collective calls issued by every rank."""
        return [
            sum(rank in rank_inputs for rank_inputs in self._inputs.values())
            for rank in range(self._world_size)
        ]

    def set_rank(self, rank: int) -> None:
        """Bind a logical rank and reset its collective sequence number."""
        self._local.rank = rank
        self._local.call_index = 0

    def get_rank(self, group: Optional[object] = None) -> int:
        """Return the logical rank bound to the current worker thread."""
        del group
        return self._local.rank

    def exchange(
        self,
        tensor: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
        group: object,
    ) -> torch.Tensor:
        """Exchange source chunks after every logical rank reaches this call."""
        del group
        rank = self._local.rank
        call_index = self._local.call_index
        self._local.call_index += 1
        with self._condition:
            rank_inputs = self._inputs.setdefault(call_index, {})
            rank_inputs[rank] = (tensor, tuple(input_splits), tuple(output_splits))
            if len(rank_inputs) == self._world_size:
                outputs = {}
                for destination in range(self._world_size):
                    received_chunks = []
                    for source in range(self._world_size):
                        source_tensor, source_splits, _ = rank_inputs[source]
                        received_chunks.append(
                            source_tensor.split(source_splits, dim=0)[destination]
                        )
                    outputs[destination] = torch.cat(received_chunks, dim=0)
                self._outputs[call_index] = outputs
                self._condition.notify_all()
            completed = self._condition.wait_for(
                lambda: call_index in self._outputs,
                timeout=10,
            )
            if not completed:
                raise RuntimeError(
                    f"Mock All-to-All call {call_index} timed out on rank {rank}"
                )
            return self._outputs[call_index][rank]


class _StaticExperts(torch.nn.Module):
    """Small stacked SwiGLU expert holder for static EP numerics tests."""

    def __init__(self, fused: bool) -> None:
        """Create two deterministic local experts in fused or split layout."""
        super().__init__()
        torch.manual_seed(19 if fused else 23)
        self.local_expert_count = 2
        self.num_experts = 4
        self._ep_act_fn = functional.silu
        if fused:
            self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 6, 4) / 4)
        else:
            self.gate_proj = torch.nn.Parameter(torch.randn(2, 3, 4) / 4)
            self.up_proj = torch.nn.Parameter(torch.randn(2, 3, 4) / 4)
        self.down_proj = torch.nn.Parameter(torch.randn(2, 4, 3) / 4)


class _FixedQwen3Router(torch.nn.Module):
    """Qwen3-style router returning logits, weights, and indices."""

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return deterministic top-k assignments for every local token."""
        token_count = hidden_states.numel() // hidden_states.shape[-1]
        indices = torch.tensor([[0, 1], [1, 0]], device=hidden_states.device)
        weights = torch.tensor(
            [[0.75, 0.25], [0.4, 0.6]],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        logits = hidden_states.new_zeros((token_count, 4))
        return logits, weights[:token_count], indices[:token_count]


class _FixedDeepseekV3Router(torch.nn.Module):
    """Logits provider for the existing DeepSeek sigmoid-group adapter."""

    def forward(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Return logits whose top two experts are local in the mocked exchange."""
        token_count = hidden_states.numel() // hidden_states.shape[-1]
        logits = torch.tensor(
            [[2.0, 1.0, -2.0, -3.0], [1.0, 2.0, -3.0, -2.0]],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return logits[:token_count]


class _StaticMoe(torch.nn.Module):
    """Small MoE exposing the interfaces used by Qwen3 and DeepSeek-V3."""

    def __init__(self, architecture: str) -> None:
        """Build the requested model-family interface."""
        super().__init__()
        if architecture == "qwen3":
            self.gate = _FixedQwen3Router()
            self.experts = _StaticExperts(fused=True)
        elif architecture == "deepseek_v3":
            self.gate = _FixedDeepseekV3Router()
            self.experts = _StaticExperts(fused=False)
            self.shared_experts = torch.nn.Linear(4, 4, bias=False)
        else:
            raise ValueError(f"Unsupported architecture {architecture}")


class _CrossRankExperts(torch.nn.Module):
    """One local SwiGLU expert used by the multi-rank routing simulation."""

    def __init__(self, rank: int) -> None:
        """Create deterministic rank-specific expert parameters."""
        super().__init__()
        generator = torch.Generator().manual_seed(101 + rank)
        self.local_expert_count = 1
        self.num_experts = 2
        self._ep_act_fn = functional.silu
        self.gate_proj = torch.nn.Parameter(
            torch.randn(1, 3, 4, generator=generator)
        )
        self.up_proj = torch.nn.Parameter(
            torch.randn(1, 3, 4, generator=generator)
        )
        self.down_proj = torch.nn.Parameter(
            torch.randn(1, 4, 3, generator=generator)
        )


class _CrossRankMoe(torch.nn.Module):
    """MoE rank with fixed global routes and one local expert."""

    def __init__(self, rank: int, expert_indices: list[int]) -> None:
        """Initialize local weights and the routes for two local tokens."""
        super().__init__()
        self.experts = _CrossRankExperts(rank)
        self.register_buffer(
            "route_indices",
            torch.tensor(expert_indices, dtype=torch.int64).unsqueeze(1),
        )
        self.register_buffer("route_weights", torch.ones(2, 1))


def _cross_rank_router(
    module: _CrossRankMoe, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the rank's fixed routes using the activation dtype."""
    return module.route_indices, module.route_weights.to(hidden_states.dtype)


def _cross_rank_reference(
    modules: list[_CrossRankMoe],
    hidden_states: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Evaluate global expert routes without distributed communication."""
    outputs = []
    for module, rank_states in zip(modules, hidden_states):
        flat_states = rank_states.reshape(-1, rank_states.shape[-1])
        token_outputs = []
        for token_index, token_states in enumerate(flat_states):
            expert_index = int(module.route_indices[token_index, 0])
            experts = modules[expert_index].experts
            gate_states = experts.gate_proj[0] @ token_states
            up_states = experts.up_proj[0] @ token_states
            token_outputs.append(
                experts.down_proj[0] @ (functional.silu(gate_states) * up_states)
            )
        outputs.append(torch.stack(token_outputs).view_as(rank_states))
    return outputs


def _run_cross_rank_static(
    rank: int,
    module: _CrossRankMoe,
    hidden_states: torch.Tensor,
    group: _FakeEpGroup,
    coordinator: _ThreadedAllToAll,
) -> torch.Tensor:
    """Run one logical EP rank through the threaded collective simulator."""
    coordinator.set_rank(rank)
    return static_ep_routed_forward(
        module,
        hidden_states,
        router_fn=_cross_rank_router,
        ep_group=group,
    )


def _build_cross_rank_case():
    """Build skewed routes and matching reference inputs for two ranks."""
    modules = [
        _CrossRankMoe(rank=0, expert_indices=[0, 1]),
        _CrossRankMoe(rank=1, expert_indices=[0, 0]),
    ]
    hidden_states = [
        torch.tensor(
            [[[0.2, -0.4, 0.7, 1.1], [-0.3, 0.8, 0.5, -0.6]]],
            requires_grad=True,
        ),
        torch.tensor(
            [[[0.9, -0.1, -0.5, 0.4], [0.6, 0.3, -0.8, 0.2]]],
            requires_grad=True,
        ),
    ]
    reference_modules = copy.deepcopy(modules)
    reference_states = [
        states.detach().clone().requires_grad_(True) for states in hidden_states
    ]
    return modules, hidden_states, reference_modules, reference_states


def _execute_cross_rank_case(
    modules: list[_CrossRankMoe], hidden_states: list[torch.Tensor]
):
    """Execute the two logical ranks concurrently through three exchanges."""
    coordinator = _ThreadedAllToAll(world_size=2)
    group = _FakeEpGroup(size=2)
    with (
        patch.object(dist, "get_rank", side_effect=coordinator.get_rank),
        patch(
            "hyper_parallel.compile.ep_capture.all_to_all_single",
            side_effect=coordinator.exchange,
        ),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        futures = [
            executor.submit(
                _run_cross_rank_static,
                rank,
                modules[rank],
                hidden_states[rank],
                group,
                coordinator,
            )
            for rank in range(2)
        ]
        outputs = [future.result() for future in futures]
    return outputs, coordinator


def _cross_rank_gradients(
    outputs: list[torch.Tensor],
    hidden_states: list[torch.Tensor],
    modules: list[_CrossRankMoe],
) -> tuple[torch.Tensor, ...]:
    """Differentiate a rank-distinguishing scalar loss for all test tensors."""
    coefficients = [
        torch.tensor([[[1.0, -0.5, 0.25, 0.75], [0.4, 0.3, -0.2, 0.6]]]),
        torch.tensor([[[-0.7, 0.2, 0.8, -0.1], [0.5, -0.4, 0.9, 0.3]]]),
    ]
    loss = sum(
        (output * coefficient).sum()
        for output, coefficient in zip(outputs, coefficients)
    )
    parameters = tuple(
        parameter for module in modules for parameter in module.parameters()
    )
    return torch.autograd.grad(loss, (*hidden_states, *parameters))


def _assert_tensor_sequences_close(
    actual_tensors, expected_tensors, value_name: str
) -> None:
    """Assert two ordered tensor collections are numerically equivalent."""
    for index, (actual, expected) in enumerate(
        zip(actual_tensors, expected_tensors)
    ):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-6,
            msg=(
                f"Cross-rank {value_name} {index} mismatch: "
                f"expected={expected}, got={actual}"
            ),
        )


def _reference_routed_forward(
    module: _StaticMoe,
    hidden_states: torch.Tensor,
    router_fn,
) -> torch.Tensor:
    """Compute the local routed branch without fixed-capacity padding."""
    topk_indices, topk_weights = router_fn(module, hidden_states)
    flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    gate_weight, up_weight, down_weight = resolve_swiglu_weights(module.experts)
    token_outputs = []
    for token_index in range(flat_states.shape[0]):
        routed_outputs = []
        for route_index in range(topk_indices.shape[1]):
            expert_index = int(topk_indices[token_index, route_index])
            if up_weight is None:
                gate_states, up_states = (
                    flat_states[token_index]
                    @ gate_weight[expert_index].transpose(0, 1)
                ).chunk(2, dim=-1)
            else:
                gate_states = (
                    flat_states[token_index]
                    @ gate_weight[expert_index].transpose(0, 1)
                )
                up_states = (
                    flat_states[token_index]
                    @ up_weight[expert_index].transpose(0, 1)
                )
            expert_output = (
                functional.silu(gate_states) * up_states
            ) @ down_weight[expert_index].transpose(0, 1)
            routed_outputs.append(
                expert_output * topk_weights[token_index, route_index]
            )
        token_outputs.append(torch.stack(routed_outputs).sum(dim=0))
    return torch.stack(token_outputs).view_as(hidden_states)


class TestPassConfig(unittest.TestCase):
    """Validate static EP configuration and FSDP composition."""

    def test_ep_config_keeps_explicit_fsdp_enabled(self):
        """EP must compose with the explicit upstream FSDP switch."""
        config = PassConfig(ep_degree=2, fsdp_enabled=True, fsdp_degree=2)

        config.validate()

        self.assertTrue(
            config.ep_enabled, msg=f"Expected EP enabled, got {config.ep_enabled}"
        )
        self.assertTrue(
            config.fsdp_enabled,
            msg=f"Expected FSDP enabled, got {config.fsdp_enabled}",
        )

    def test_ep_config_rejects_tp_combination(self):
        """The current static EP capture must reject TP composition."""
        with self.assertRaisesRegex(NotImplementedError, "tp_size"):
            PassConfig(ep_degree=2, tp_size=2).validate()

    @patch.object(dist, "is_initialized", return_value=False)
    def test_ep_config_does_not_probe_process_group(self, mock_initialized):
        """Configuration validation must stay torch-distributed independent."""
        config = PassConfig(ep_degree=2)

        config.validate()

        mock_initialized.assert_not_called()

    def test_parallel_degrees_must_be_positive_integers(self):
        """All explicit parallel degrees must reject booleans and non-positive values."""
        invalid_configs = (
            {"ep_degree": False},
            {"tp_size": 0},
            {"tp_size": 1.5},
            {"fsdp_degree": 0},
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    PassConfig(**kwargs).validate()


class TestExpertParallelPass(unittest.TestCase):
    """Validate the all-to-all evidence pass and pipeline selection."""

    def test_ep_pass_records_collective_count(self):
        """All expected collectives on the captured EP group are recorded."""
        graph_module = _graph_module(ep_collective_count=5)

        result = ExpertParallelPass().run(
            graph_module,
            PassConfig(ep_degree=2, require_ep_collectives=True),
        )

        self.assertEqual(
            result.ep_collective_count,
            5,
            msg=f"Expected five EP All-to-All nodes, got {result.ep_collective_count}",
        )

    def test_ep_pass_rejects_missing_collective(self):
        """Required EP evidence must reject a graph without all-to-all."""
        graph_module = _graph_module()

        with self.assertRaisesRegex(RuntimeError, "capture is incomplete"):
            ExpertParallelPass().run(
                graph_module,
                PassConfig(ep_degree=2, require_ep_collectives=True),
            )

    def test_ep_pass_rejects_non_ep_all_to_all(self):
        """An unrelated All-to-All group must not satisfy EP provenance."""
        graph_module = _graph_module(other_collective_count=5)

        with self.assertRaisesRegex(RuntimeError, "found 0"):
            ExpertParallelPass().run(
                graph_module,
                PassConfig(ep_degree=2, require_ep_collectives=True),
            )

    def test_ep_pass_rejects_partial_capture(self):
        """A graph missing one routed-EP collective must fail validation."""
        graph_module = _graph_module(ep_collective_count=4)

        with self.assertRaisesRegex(RuntimeError, "expected 5.*found 4"):
            ExpertParallelPass().run(
                graph_module,
                PassConfig(ep_degree=2, require_ep_collectives=True),
            )

    def test_ep_pass_requires_capture_metadata(self):
        """Function names alone must not be accepted without EP provenance."""
        graph_module = _graph_module(
            ep_collective_count=5,
            include_metadata=False,
        )

        with self.assertRaisesRegex(RuntimeError, "no routed-EP capture metadata"):
            ExpertParallelPass().run(
                graph_module,
                PassConfig(ep_degree=2, require_ep_collectives=True),
            )

    def test_ep_pass_allows_missing_collective_when_not_required(self):
        """Diagnostic-only capture may opt out of the collective requirement."""
        graph_module = _graph_module(include_metadata=False)

        result = ExpertParallelPass().run(
            graph_module,
            PassConfig(ep_degree=2, require_ep_collectives=False),
        )

        self.assertEqual(
            result.ep_collective_count,
            0,
            msg=f"Expected no all-to-all nodes, got {result.ep_collective_count}",
        )

    def test_ep_pipeline_selects_fsdp_and_ep_passes(self):
        """A mixed config must run FSDP before EP evidence validation."""
        config = PassConfig(
            ep_degree=2,
            fsdp_enabled=True,
            enable_overlap=False,
        )

        pipeline = PassPipeline.from_config(config)
        pass_names = [graph_pass.name for graph_pass in pipeline.passes]

        self.assertIn(
            "expert_parallel", pass_names, msg=f"EP pass missing from {pass_names}"
        )
        self.assertIn(
            "fsdp_parallel",
            pass_names,
            msg=f"FSDP pass missing from {pass_names}",
        )
        self.assertLess(
            pass_names.index("fsdp_parallel"),
            pass_names.index("expert_parallel"),
            msg=f"Expected FSDP before EP, got {pass_names}",
        )

    def test_ep_pipeline_can_disable_fsdp_explicitly(self):
        """An EP-only config must retain the explicit FSDP opt-out."""
        config = PassConfig(
            ep_degree=2,
            fsdp_enabled=False,
            enable_overlap=False,
        )

        pass_names = [
            graph_pass.name
            for graph_pass in PassPipeline.from_config(config).passes
        ]

        self.assertIn(
            "expert_parallel", pass_names, msg=f"EP pass missing from {pass_names}"
        )
        self.assertNotIn(
            "fsdp_parallel",
            pass_names,
            msg=f"FSDP pass unexpectedly present in {pass_names}",
        )


class TestEpCapture(unittest.TestCase):
    """Validate the fixed-capacity transport and temporary capture adapters."""

    def test_capture_restores_shared_expert_holder(self):
        """Two MoE parents sharing experts must not leak capture state."""
        first, second = _FakeMoe(), _FakeMoe()
        second.experts = first.experts
        model = torch.nn.ModuleList([first, second])
        with capture_dynamic_ep(model, ep_degree=2) as metadata:
            self.assertIs(first.experts._ep_capture_metadata, metadata)
            self.assertIs(second.experts._ep_capture_metadata, metadata)
        self.assertFalse(hasattr(first.experts, "_ep_capture_metadata"))


    def test_capture_rejects_process_group_degree_mismatch(self):
        """Expert metadata alone cannot prove the communication group size."""
        model = _FakeMoe()
        with capture_dynamic_ep(model, ep_degree=2):
            with self.assertRaisesRegex(ValueError, "group size does not match"):
                ep_utils.ep_routed_forward(
                    model, torch.ones(1, 2, 4), router_fn=None, ep_group=_FakeEpGroup(4)
                )

    def test_ep_pass_rejects_compensating_group_counts(self):
        """Extra exchanges in one EP group must not hide missing exchanges in another."""
        graph_module = _graph_module(ep_collective_count=6, other_collective_count=4, expected_collective_count=10)
        graph_module.ep_capture_metadata.update(
            group_names=("ep_group", "other_group"),
            collective_counts_by_group={"ep_group": 5, "other_group": 5},
        )
        with self.assertRaisesRegex(RuntimeError, "EP capture group mismatch"):
            ExpertParallelPass().run(graph_module, PassConfig(ep_degree=2))

    def _assert_static_factory_matches_reference(
        self,
        architecture: str,
        factory,
        router_name: str,
    ) -> None:
        """Compare a model-family factory's static forward and gradients."""
        module = _StaticMoe(architecture)
        group = _FakeEpGroup()
        ep_mesh = _FakeEpMesh(group)
        compute_fn = factory(
            module=module,
            mesh=None,
            tp_mesh=None,
            cp_mesh=None,
            ep_mesh=ep_mesh,
        )
        hidden_states = torch.tensor(
            [[[0.2, -0.5, 0.7, 1.1], [-0.3, 0.9, 0.4, -0.8]]],
            requires_grad=True,
        )
        parameters = tuple(module.parameters())

        with (
            patch.object(dist, "get_rank", return_value=0),
            patch(
                "hyper_parallel.compile.ep_capture.all_to_all_single",
                side_effect=lambda tensor, *_args: tensor.clone(),
            ) as mock_all_to_all,
            capture_dynamic_ep(module, ep_degree=2) as metadata,
        ):
            actual = compute_fn(module, hidden_states)

        expected = _reference_routed_forward(
            module,
            hidden_states,
            MOE_ROUTER_ADAPTERS[router_name],
        )
        if architecture == "deepseek_v3":
            expected = expected + module.shared_experts(hidden_states)

        actual_grads = torch.autograd.grad(actual.sum(), (hidden_states, *parameters))
        expected_grads = torch.autograd.grad(
            expected.sum(), (hidden_states, *parameters)
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
            msg=f"{architecture} output mismatch: expected={expected}, got={actual}",
        )
        for gradient_index, (actual_grad, expected_grad) in enumerate(
            zip(actual_grads, expected_grads)
        ):
            torch.testing.assert_close(
                actual_grad,
                expected_grad,
                rtol=1e-6,
                atol=1e-6,
                msg=(
                    f"{architecture} gradient {gradient_index} mismatch: "
                    f"expected={expected_grad}, got={actual_grad}"
                ),
            )
        self.assertEqual(
            mock_all_to_all.call_count,
            3,
            msg=(
                f"Expected dispatch, expert-index, and combine exchanges, "
                f"got {mock_all_to_all.call_count}"
            ),
        )
        self.assertEqual(
            metadata.group_names,
            {"ep_group"},
            msg=f"Expected EP group provenance, got {metadata.group_names}",
        )
        self.assertEqual(
            metadata.expected_collective_count,
            5,
            msg=(
                f"Expected five joint-graph collectives, got "
                f"{metadata.expected_collective_count}"
            ),
        )

    def test_static_ep_routes_across_ranks_with_gradients(self):
        """Fixed-capacity dispatch must preserve skewed cross-rank routes."""
        case = _build_cross_rank_case()
        modules, hidden_states, reference_modules, reference_states = case
        actual_outputs, coordinator = _execute_cross_rank_case(
            modules, hidden_states
        )
        expected_outputs = _cross_rank_reference(
            reference_modules, reference_states
        )
        actual_gradients = _cross_rank_gradients(
            actual_outputs, hidden_states, modules
        )
        expected_gradients = _cross_rank_gradients(
            expected_outputs, reference_states, reference_modules
        )
        _assert_tensor_sequences_close(actual_outputs, expected_outputs, "output")
        _assert_tensor_sequences_close(
            actual_gradients, expected_gradients, "gradient"
        )
        self.assertEqual(
            coordinator.call_counts,
            [3, 3],
            msg=f"Expected three exchanges per rank, got {coordinator.call_counts}",
        )

    @patch("hyper_parallel.compile.ep_capture.all_to_all_single")
    def test_equal_capacity_all_to_all_uses_differentiable_collective(
        self, mock_all_to_all
    ):
        """Equal-capacity exchange must preserve autograd through the functional op."""
        mock_all_to_all.side_effect = lambda tensor, *_args: tensor.clone()
        tensor = torch.tensor([[1.0, -2.0], [3.0, 4.0]], requires_grad=True)

        output = _exchange_equal_capacity(tensor, 1, 2, object())
        output.square().sum().backward()

        self.assertTrue(
            torch.equal(output, tensor),
            msg=f"Expected identity mocked exchange, got output={output}, tensor={tensor}",
        )
        self.assertTrue(
            torch.equal(tensor.grad, 2 * tensor),
            msg=f"Expected gradient={2 * tensor}, got {tensor.grad}",
        )
        self.assertEqual(
            mock_all_to_all.call_count,
            1,
            msg=f"Expected one functional exchange, got {mock_all_to_all.call_count}",
        )

    @patch("hyper_parallel.compile.ep_capture.all_to_all_single")
    def test_equal_capacity_all_to_all_rejects_invalid_shape(
        self, mock_all_to_all
    ):
        """Invalid fixed-capacity buffers must fail before communication."""
        tensor = torch.randn(3, 2)

        with self.assertRaisesRegex(ValueError, "invalid leading dimension"):
            _exchange_equal_capacity(tensor, rows_per_peer=2, ep_size=2, ep_group=object())

        mock_all_to_all.assert_not_called()

    @patch("hyper_parallel.compile.ep_capture.all_to_all_single")
    def test_equal_capacity_all_to_all_survives_joint_graph_backward(
        self, mock_all_to_all
    ):
        """Joint make_fx capture must retain the functional collective gradient."""
        mock_all_to_all.side_effect = lambda tensor, *_args: tensor.clone()

        def forward_backward(tensor: torch.Tensor) -> torch.Tensor:
            """Return the joint-graph gradient of the mocked EP exchange."""
            output = _exchange_equal_capacity(tensor, 1, 2, object())
            return torch.autograd.grad(output.square().sum(), tensor)[0]

        example = torch.tensor([[1.5, -0.5], [2.0, 3.0]], requires_grad=True)
        graph_module = make_fx(forward_backward)(example)
        actual_grad = graph_module(example)
        expected_grad = 2 * example

        torch.testing.assert_close(
            actual_grad,
            expected_grad,
            rtol=0,
            atol=0,
            msg=(
                f"Joint graph gradient mismatch: expected={expected_grad}, "
                f"got={actual_grad}"
            ),
        )

    def test_nested_ep_captures_restore_outer_metadata(self):
        """Nested capture restores the outer mode and its provenance."""
        model = _FakeMoe()
        with capture_dynamic_ep(model, ep_degree=2) as outer:
            with capture_dynamic_ep(model, ep_degree=2) as inner:
                self.assertIs(model.experts._ep_capture_metadata, inner)
                self.assertIsNot(inner, outer)
            self.assertIs(model.experts._ep_capture_metadata, outer)
        self.assertFalse(hasattr(model.experts, "_ep_capture_metadata"))


    def test_qwen3_static_factory_matches_ragged_reference(self):
        """Qwen3 static EP must match routed forward and parameter gradients."""
        self._assert_static_factory_matches_reference(
            "qwen3",
            qwen3moe_ep_compute_fn,
            "qwen3moe",
        )

    def test_deepseek_v3_static_factory_matches_ragged_reference(self):
        """DeepSeek-V3 static EP must retain router and shared-expert semantics."""
        self._assert_static_factory_matches_reference(
            "deepseek_v3",
            ep_compute.deepseekv3_ep_compute_fn,
            "deepseekv3",
        )

    def test_capture_dynamic_ep_selects_and_restores_static_transport(self):
        """Only the captured model calls static EP, with unchanged arguments."""
        model, other_model = _FakeMoe(), _FakeMoe()
        states, router, group = torch.ones(1, 2, 4), object(), _FakeEpGroup(2)
        with patch("hyper_parallel.compile.ep_capture.static_ep_routed_forward") as static:
            with patch.object(group, "size", side_effect=RuntimeError("dynamic EP entered")) as dynamic_entry:
                with capture_dynamic_ep(model, ep_degree=2) as metadata:
                    self.assertIs(model.experts._ep_capture_metadata, metadata)
                    result = ep_utils.ep_routed_forward(model, states, router_fn=router, ep_group=group)
                    self.assertIs(result, static.return_value)
                    static.assert_called_once_with(model, states, router_fn=router, ep_group=group)
                    dynamic_entry.assert_not_called()
                    with self.assertRaisesRegex(RuntimeError, "dynamic EP entered"):
                        ep_utils.ep_routed_forward(other_model, states, router_fn=router, ep_group=group)
                    dynamic_entry.assert_called_once_with()
                with self.assertRaisesRegex(RuntimeError, "dynamic EP entered"):
                    ep_utils.ep_routed_forward(model, states, router_fn=router, ep_group=group)
                self.assertEqual(dynamic_entry.call_count, 2)
        self.assertFalse(hasattr(model.experts, "_ep_capture_metadata"))
        self.assertFalse(hasattr(other_model.experts, "_ep_capture_metadata"))


    def test_capture_dynamic_ep_restores_mode_after_error(self):
        """Tracing errors must not leave a model in static execution mode."""
        model = _FakeMoe()
        with self.assertRaisesRegex(RuntimeError, "trace failed"):
            with capture_dynamic_ep(model, ep_degree=2):
                raise RuntimeError("trace failed")
        self.assertFalse(hasattr(model.experts, "_ep_capture_metadata"))


    def test_capture_dynamic_ep_requires_applied_sharding_plan(self):
        """Capture must reject a model without dynamic EP expert metadata."""
        with self.assertRaisesRegex(ValueError, "dynamic EP sharding plan"):
            with capture_dynamic_ep(torch.nn.Linear(2, 2), ep_degree=2):
                self.fail("Capture unexpectedly accepted a non-MoE model")

    def test_capture_dynamic_ep_rejects_expert_count_mismatch(self):
        """Capture must reject expert shards that do not match the EP degree."""
        model = _FakeMoe(local_expert_count=1, global_expert_count=4)

        with self.assertRaisesRegex(ValueError, "does not match ep_degree"):
            with capture_dynamic_ep(model, ep_degree=2):
                self.fail("Capture unexpectedly accepted a mismatched expert shard")

    def test_capture_dynamic_ep_rejects_nondivisible_expert_count(self):
        """Floor division must not accept a non-divisible expert topology."""
        model = _FakeMoe(local_expert_count=2, global_expert_count=5)

        with self.assertRaisesRegex(ValueError, "does not match ep_degree"):
            with capture_dynamic_ep(model, ep_degree=2):
                self.fail("Capture unexpectedly accepted non-divisible experts")
