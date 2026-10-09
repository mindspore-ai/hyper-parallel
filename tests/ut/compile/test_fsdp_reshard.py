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
"""Unit tests for FSDP reshard (``fsdp_reshard_after_forward``).

Covers:

1. ``_free_tensor_storage`` releases a tensor's storage in place and is safe
   to call on an already-freed tensor.
2. With reshard enabled, each parameter's forward all_gather is sunk to its
   first use, the replicated parameter is freed after its last forward read,
   and — for parameters the backward still needs — a fresh all_gather plus a
   rematerialization of the saved view chain is inserted before the first
   backward consumer.
3. With reshard disabled the graph keeps the previous shape (top gathers, no
   free, no backward re-gather).
4. End-to-end numerics: a simulated single-rank execution of the transformed
   graph (all_gather == full parameter, reduce_scatter == this rank's chunk,
   free op executed for real) reproduces the reference loss and gradients, so
   the free + rematerialization does not perturb the backward — and a read of
   freed storage fails loudly instead of silently.

No real distributed communication happens: collectives are inserted as FX
nodes and ``torch.distributed`` is mocked.
"""

import copy
import unittest
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Tuple
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from hyper_parallel.compile.graph_parallel_plan import GraphParallelPlan
from hyper_parallel.compile.pass_config import PassConfig
from hyper_parallel.compile.passes.parallel.fsdp_pass import (
    FSDPPass,
    _free_tensor_storage,
)
from hyper_parallel.compile.tracer.graph_tracer import (
    extract_module_state,
    trace_model_graph,
)

_DIST_PATH = "hyper_parallel.compile.passes.parallel.fsdp_pass.dist"
_RESOLVE_PG_PATH = (
    "hyper_parallel.compile.passes.parallel.fsdp_pass._resolve_process_group"
)
_DEGREE = 2


@contextmanager
def _patch_dist(world_size: int = _DEGREE, rank: int = 0) -> Iterator[MagicMock]:
    """Patch ``dist`` / ``_resolve_process_group`` inside ``fsdp_pass``."""
    mock_dist = MagicMock()
    mock_dist.is_initialized.return_value = True
    mock_dist.get_world_size.return_value = world_size
    mock_dist.get_rank.return_value = rank
    with (
        patch(_DIST_PATH, mock_dist),
        patch(_RESOLVE_PG_PATH, return_value=MagicMock()),
    ):
        yield mock_dist


def _count_nodes(gm: torch.fx.GraphModule, needle: str) -> int:
    """Count ``call_function`` nodes whose ``str(target)``/meta matches."""
    return sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function"
        and (needle in str(n.target) or needle == n.meta.get("comm_type"))
    )


def _node_index(gm: torch.fx.GraphModule, pred) -> int:
    """Index of the first graph node satisfying ``pred`` (or ``-1``)."""
    for idx, node in enumerate(gm.graph.nodes):
        if pred(node):
            return idx
    return -1


class _MLP(nn.Module):
    """Stack of ``num_layers`` linear layers with ReLU in between."""

    def __init__(self, num_layers: int) -> None:
        """Build the layer stack (``num_layers >= 1``)."""
        super().__init__()
        dims = [8] + [8] * (num_layers - 1) + [4]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class _ChunkedMLP(nn.Module):
    """Linear layer feeding a fused ``chunk(2)`` weight and a plain tail.

    The earlier layer forces a chained grad_input, so the backward reads both
    chunk pieces through ``getitem`` picks — exercising the multi-output view
    path of the rematerialization.
    """

    def __init__(self) -> None:
        """Create the leading linear layer, fused 8x8 weight and 8x4 tail."""
        super().__init__()
        self.lin = nn.Linear(8, 8, dtype=torch.float64)
        self.w = nn.Parameter(torch.randn(8, 8, dtype=torch.float64))
        self.w2 = nn.Parameter(torch.randn(8, 4, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ReLU-linear, then split ``w`` along dim 0 and sum matmuls."""
        h = torch.relu(self.lin(x))
        a, b = self.w.chunk(2, dim=0)
        return h @ a.t() + h @ b.t() + h @ self.w2


def _trace_and_run(
    model: nn.Module,
    reshard: bool,
    world_size: int = _DEGREE,
    rank: int = 0,
) -> Tuple[torch.fx.GraphModule, nn.Module]:
    """Trace ``model``, run FSDPPass on a sharded copy, return (graph, copy).

    The returned module is the sharded live model whose parameters are the
    graph's runtime static inputs.
    """
    x = torch.randn(4, 8, dtype=torch.float64)
    y = torch.randn(4, 4, dtype=torch.float64)

    def train_fn(
        mdl: nn.Module, *, inp: torch.Tensor, lbl: torch.Tensor
    ) -> torch.Tensor:
        """MSE loss of ``mdl(inp)`` against ``lbl``."""
        return nn.functional.mse_loss(mdl(inp), lbl)

    joint = trace_model_graph(model, train_fn, {"inp": x, "lbl": y})
    graph = joint.graph_module

    live = copy.deepcopy(model)
    config = PassConfig(
        fsdp_enabled=True,
        fsdp_degree=world_size,
        fsdp_reshard_after_forward=reshard,
    )
    pass_obj = FSDPPass(parallel_plan=GraphParallelPlan().fsdp_mark_pattern("*"))
    with _patch_dist(world_size=world_size, rank=rank):
        pass_obj.run(graph, config, model=live, fsdp_group_name="fsdp")
    graph.graph.lint()
    return graph, live


class _SimulatedRank(torch.fx.Interpreter):
    """Execute a reshard graph as one rank with collectives simulated.

    ``all_gather_into_tensor`` returns the full parameter (from ``id_to_full``)
    and ``reduce_scatter_tensor`` returns this rank's chunk of the gradient, so
    the transformed graph can be compared against a plain reference model
    without any real process group. The free op runs for real: any read of
    freed storage after its node raises, so the numerics below also pin the
    free / rematerialization ordering the pass is responsible for.
    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        id_to_full: Dict[int, torch.Tensor],
        degree: int,
    ) -> None:
        """Store the full-parameter lookup and the simulated world size."""
        super().__init__(graph_module)
        self.id_to_full = id_to_full
        self.degree = degree

    def call_function(
        self, target: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """Simulate the c10d collectives and execute the storage free."""
        name = str(target)
        if "all_gather_into_tensor" in name:
            return self.id_to_full[id(args[0])].clone()
        if "reduce_scatter_tensor" in name:
            return args[0].chunk(self.degree, dim=0)[0].clone()
        if "wait_tensor" in name:
            return args[0]
        if "_free_tensor_storage" in name:
            return _free_tensor_storage(*args, **kwargs)
        return super().call_function(target, args, kwargs)


def _run_simulated(graph, live, reference, inputs, labels, degree=_DEGREE):
    """Run the transformed graph on ``live`` and return (loss, grads)."""
    id_to_full = {
        id(shard): full
        for shard, full in zip(live.parameters(), reference.parameters())
    }
    state = extract_module_state(live)
    outputs = _SimulatedRank(graph, id_to_full, degree).run(
        *list(state.values()), inputs, labels
    )
    return outputs[0], list(outputs[1:])


class TestFreeTensorStorage(unittest.TestCase):
    """``_free_tensor_storage`` releases storage and tolerates repeat calls."""

    def test_releases_storage_in_place(self):
        """A freed tensor's storage shrinks to zero bytes."""
        tensor = torch.ones(4, 4)
        self.assertGreater(tensor.untyped_storage().size(), 0)
        _free_tensor_storage(tensor)
        self.assertEqual(tensor.untyped_storage().size(), 0)

    def test_idempotent_and_non_tensor_safe(self):
        """Calling twice, or on a non-tensor, does not raise."""
        tensor = torch.ones(2)
        _free_tensor_storage(tensor)
        _free_tensor_storage(tensor)
        _free_tensor_storage(None)


class TestReshardStructure(unittest.TestCase):
    """Graph shape of the reshard transformation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the 2-layer model used by every structural assertion."""
        torch.manual_seed(0)
        cls.model = _MLP(2).double()

    def test_enabled_inserts_free_and_backward_regather(self):
        """Reshard frees every gathered param and re-gathers backward-needed ones.

        A 2-layer MLP has 4 params: forward gathers = 4, free-after-forward = 4.
        Only the *second* layer's weight is read by the backward (the first
        layer's grad_input is not needed), so exactly one backward re-gather
        and one backward free are inserted.
        """
        graph, _ = _trace_and_run(self.model, reshard=True)

        self.assertEqual(_count_nodes(graph, "fsdp_all_gather"), 4)
        self.assertEqual(_count_nodes(graph, "fsdp_reshard_free"), 4)
        self.assertEqual(_count_nodes(graph, "fsdp_all_gather_backward"), 1)
        self.assertEqual(_count_nodes(graph, "fsdp_reshard_free_backward"), 1)

    def test_forward_gather_is_sunk_to_first_use(self):
        """The last layer's gather lands after earlier compute, not at the top.

        All state placeholders lead the graph, so unless the gather is sunk
        every full parameter is materialized before any compute; the reshard
        free would then reclaim nothing. The layer-2 gather must appear after
        layer 1's forward has run.
        """
        graph, _ = _trace_and_run(self.model, reshard=True)
        first_relu = _node_index(graph, lambda n: "relu" in str(n.target))
        gather_indices = [
            idx
            for idx, n in enumerate(graph.graph.nodes)
            if n.meta.get("comm_type") == "fsdp_all_gather"
        ]
        self.assertEqual(len(gather_indices), 4)
        # The LAST forward gather must be after the first relu (layer 1 done);
        # an unsunk pass would keep every gather next to its placeholder.
        self.assertGreater(max(gather_indices), first_relu)

    def test_disabled_keeps_top_gathers_and_no_free(self):
        """With reshard off there is no free and no backward re-gather."""
        graph, _ = _trace_and_run(self.model, reshard=False)

        self.assertEqual(_count_nodes(graph, "fsdp_all_gather"), 4)
        self.assertEqual(_count_nodes(graph, "_free_tensor_storage"), 0)
        self.assertEqual(_count_nodes(graph, "fsdp_all_gather_backward"), 0)


class TestReshardNumerics(unittest.TestCase):
    """Simulated execution reproduces the reference loss and gradients."""

    def _assert_gradients_match(self, num_layers: int):
        """Compare the reshard graph against a plain full-parameter model.

        Returns the transformed graph for extra structural assertions.
        """
        torch.manual_seed(num_layers)
        reference = _MLP(num_layers).double()
        graph, live = _trace_and_run(reference, reshard=True)

        inputs = torch.randn(4, 8, dtype=torch.float64)
        labels = torch.randn(4, 4, dtype=torch.float64)
        loss, grads = _run_simulated(graph, live, reference, inputs, labels)

        ref = copy.deepcopy(reference)
        ref_loss = nn.functional.mse_loss(ref(inputs), labels)
        ref_grads = torch.autograd.grad(ref_loss, list(ref.parameters()))

        self.assertTrue(torch.allclose(loss, ref_loss, atol=1e-10))
        for grad, ref_grad in zip(grads, ref_grads):
            expected = ref_grad.chunk(_DEGREE, dim=0)[0]
            self.assertTrue(torch.allclose(grad, expected, atol=1e-10))
        return graph

    def test_two_layer_gradients_match(self):
        """2-layer MLP: loss and per-parameter grads match the reference."""
        self._assert_gradients_match(2)

    def test_deep_model_gradients_match(self):
        """3-layer MLP: two layers need backward remat; grads still match."""
        graph = self._assert_gradients_match(3)

        # 3 layers -> 6 params; layers 2 and 3 re-gather for backward.
        self.assertEqual(_count_nodes(graph, "fsdp_all_gather"), 6)
        self.assertEqual(_count_nodes(graph, "fsdp_all_gather_backward"), 2)

    def test_chunked_param_gradients_match(self):
        """Fused chunk(2) weight: getitem views remat; grads match reference.

        ``chunk``/``split`` parameters reach the backward through ``getitem``
        picks that have no op schema of their own; the alias chain and the
        free point must carry through them. The free executes for real (see
        ``_SimulatedRank``), so a misplaced free raises here instead of
        silently corrupting the backward.
        """
        torch.manual_seed(7)
        reference = _ChunkedMLP()
        graph, live = _trace_and_run(reference, reshard=True)

        inputs = torch.randn(4, 8, dtype=torch.float64)
        labels = torch.randn(4, 4, dtype=torch.float64)
        loss, grads = _run_simulated(graph, live, reference, inputs, labels)

        ref = copy.deepcopy(reference)
        ref_loss = nn.functional.mse_loss(ref(inputs), labels)
        ref_grads = torch.autograd.grad(ref_loss, list(ref.parameters()))

        self.assertTrue(torch.allclose(loss, ref_loss, atol=1e-10))
        for grad, ref_grad in zip(grads, ref_grads):
            expected = ref_grad.chunk(_DEGREE, dim=0)[0]
            self.assertTrue(torch.allclose(grad, expected, atol=1e-10))
        # lin.{weight,bias} and the fused weight gather in forward; the fused
        # one re-gathers for backward through its getitem views, as does the
        # tail weight; both directions free after their last reader.
        self.assertEqual(_count_nodes(graph, "fsdp_all_gather"), 4)
        self.assertEqual(_count_nodes(graph, "fsdp_reshard_free"), 4)
        self.assertEqual(_count_nodes(graph, "fsdp_all_gather_backward"), 2)
        self.assertEqual(_count_nodes(graph, "fsdp_reshard_free_backward"), 2)


if __name__ == "__main__":
    unittest.main()
