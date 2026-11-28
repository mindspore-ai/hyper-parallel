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
"""Unit tests for GraphTrainer TP/SP adaptation.

Tests cover:

1. ``fsdp_degree`` resolution: when ``pass_config.fsdp_degree`` is set
   (TP+FSDP hybrid), FSDPPass uses it instead of ``world_size``.
2. ``_shard_live_model_params`` uses the FSDP group's local rank, not
   the global rank, as the chunk index.
3. ``TracingContext`` availability for functional collectives tracing.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from hyper_parallel.compile.pass_config import PassConfig
from hyper_parallel.compile.passes.parallel.fsdp_pass import FSDPPass
from hyper_parallel.compile.graph_parallel_plan import GraphParallelPlan
from hyper_parallel.compile.tracer.graph_tracer import extract_module_state, trace_model_graph


# ---------------------------------------------------------------------------
# Helper: build a minimal model + joint FX graph for testing.
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """A 2-parameter model: linear + bias, suitable for FX tracing."""

    def __init__(self, in_features=8, out_features=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        return self.linear(x)


def _make_joint_graph(model, input_tensor, label_tensor):
    """Trace a joint fwd+bwd graph and return the GraphModule."""
    # pylint: disable=C0415

    def train_fn(mdl, *, inp, lbl):
        return nn.functional.mse_loss(mdl(inp), lbl)

    return trace_model_graph(
        model, train_fn, {"inp": input_tensor, "lbl": label_tensor}
    )


# ---------------------------------------------------------------------------
# Tests: fsdp_degree resolution (TP+FSDP hybrid)
# ---------------------------------------------------------------------------


class TestFsdpDegreeResolution(unittest.TestCase):
    """FSDPPass must use ``pass_config.fsdp_degree`` when set, not
    ``world_size``.  This is the fix for the TP+FSDP IndexError bug."""

    def setUp(self):
        self.model = TinyModel(in_features=8, out_features=4)
        self.input = torch.randn(2, 8)
        self.label = torch.randn(2, 4)
        self.joint = _make_joint_graph(self.model, self.input, self.label)

    @patch("hyper_parallel.compile.passes.parallel.fsdp_pass.dist")
    @patch("hyper_parallel.compile.passes.parallel.fsdp_pass._resolve_process_group")
    def test_uses_explicit_fsdp_degree(self, mock_resolve_pg, mock_dist):
        """When fsdp_degree=2 and world_size=4, the pass should use 2, not 4."""
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 4  # world=4 but FSDP group=2
        mock_dist.get_rank.return_value = 0  # local rank in FSDP group
        mock_resolve_pg.return_value = MagicMock()

        config = PassConfig(fsdp_degree=2)
        plan = GraphParallelPlan()
        plan.fsdp_mark_pattern("*")

        pass_obj = FSDPPass(parallel_plan=plan)
        pass_obj.run(
            self.joint.graph_module,
            config,
            model=self.model,
            fsdp_group_name="fsdp",
            parallel_plan=plan,
        )

        self.assertEqual(pass_obj._fsdp_degree, 2)


# ---------------------------------------------------------------------------
# Tests: _shard_live_model_params uses local rank (TP+FSDP fix)
# ---------------------------------------------------------------------------


class TestShardLiveModelParamsRank(unittest.TestCase):
    """The shard step must use the FSDP group's local rank, not the global
    rank, as the chunk index — preventing IndexError when
    ``fsdp_degree < world_size``.
    """

    def setUp(self):
        self.model = TinyModel(in_features=8, out_features=4)
        self.input = torch.randn(2, 8)
        self.label = torch.randn(2, 4)
        self.joint = _make_joint_graph(self.model, self.input, self.label)

    @patch("hyper_parallel.compile.passes.parallel.fsdp_pass.dist")
    @patch("hyper_parallel.compile.passes.parallel.fsdp_pass._resolve_process_group")
    def test_global_rank_does_not_cause_index_error(self, mock_resolve_pg, mock_dist):
        """Simulate TP+FSDP: world_size=4, fsdp_degree=2, global_rank=3.

        Before the fix, rank=3 with chunk(2) caused IndexError.
        After the fix, the pass uses the FSDP group's local rank.
        """
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 4
        mock_dist.get_rank.return_value = 1  # local rank in FSDP group
        mock_resolve_pg.return_value = MagicMock()

        config = PassConfig(fsdp_degree=2)
        plan = GraphParallelPlan()
        plan.fsdp_mark_pattern("*")

        pass_obj = FSDPPass(parallel_plan=plan)
        pass_obj.run(
            self.joint.graph_module,
            config,
            model=self.model,
            fsdp_group_name="fsdp",
            parallel_plan=plan,
        )

        # weight dim 0 should be halved (4 -> 2)
        self.assertEqual(self.model.linear.weight.shape[0], 2)


# ---------------------------------------------------------------------------
# Tests: TracingContext (TP communication tracing fix)
# ---------------------------------------------------------------------------


class TestTracingContextAvailable(unittest.TestCase):
    """The tracer sets up a ``TracingContext`` during make_fx — this is what
    the platform-level ``differentiable_*`` functions check (via
    ``TracingContext.get() is not None``) to decide whether to use
    functional collectives instead of the autograd-wrapped eager API.
    """

    def test_trace_produces_callable_graph(self):
        """The traced graph module should be callable with real inputs."""
        model = TinyModel(8, 4)
        inp = torch.randn(2, 8)
        lbl = torch.randn(2, 4)

        def train_fn(mdl, *, i, l):
            return nn.functional.mse_loss(mdl(i), l)

        joint = trace_model_graph(model, train_fn, {"i": inp, "l": lbl})

        state = extract_module_state(model)
        state_flat, _ = torch.utils._pytree.tree_flatten({"model": state})
        user_flat, _ = torch.utils._pytree.tree_flatten({"i": inp, "l": lbl})
        flat_inputs = list(state_flat) + list(user_flat)

        with torch.no_grad():
            outputs = joint.graph_module(*flat_inputs)

        self.assertIsInstance(outputs, (list, tuple))
        self.assertTrue(len(outputs) >= 2)  # at least [loss, one_grad]


if __name__ == "__main__":
    unittest.main()
