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
"""Unit tests for ``hyper_parallel.compile.tracer.graph_tracer``.

Covers the tracer end-to-end, which the pass-level tests bypass:

1. ``trace_model_graph`` produces a joint graph with parameters/buffers as
   static leading placeholders (no ``get_attr`` nodes), and attaches
   ``state_fqns`` / ``state_is_param`` / ``num_state_inputs``.
2. ``run_traced_graph`` executes it, and the (loss, grads) match
   ``torch.autograd.grad`` on the live model.
3. Buffers are marked non-param (``state_is_param[i] is False``) so FSDP never
   all-gathers them -- the property the FSDP pass depends on.
4. ``extract_module_state`` merges parameters and buffers, deduplicating tensors
   tied across several FQNs to a single entry.
5. Tied weights (an alias shared by two modules) trace to ONE placeholder and
   ONE gradient — never a baked-in ``get_attr`` constant, and never a
   doubled gradient — because the alias is re-materialized only for tracing.

Running under stock torch (no ``torch.compiler._patch_engine_backward``) still
captures the joint graph; the patcher warning is suppressed during the trace.
"""

import os
import unittest
import warnings


import torch
from torch import nn

from hyper_parallel.compile.tracer.graph_tracer import (
    extract_module_state,
    run_traced_graph,
    trace_model_graph,
)


class _LinearWithBuffer(nn.Module):
    """A model with a trainable Linear and a register_buffer."""

    def __init__(self) -> None:
        """Initialize the Linear and register the zero buffer."""
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("buf", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the linear layer on the input tensor."""
        return self.lin(x)


def _trace_and_run(model, train_fn, x, y):
    """Trace ``model`` and run the joint graph on the same inputs."""
    inputs = {"x": x, "y": y}
    with warnings.catch_warnings():
        # The patched-autograd-engine warning is expected on stock torch; the
        # joint graph still captures correctly in this environment.
        warnings.simplefilter("ignore", RuntimeWarning)
        joint = trace_model_graph(model, train_fn, inputs)
        loss, grads = run_traced_graph(joint, model, inputs)
    return joint, loss, grads


class TestTraceModelGraph(unittest.TestCase):
    """``trace_model_graph`` output structure."""

    def test_state_is_leading_placeholders_not_get_attr(self):
        """Test parameters/buffers are static inputs, never get_attr nodes."""
        model = _LinearWithBuffer()
        joint, _, _ = _trace_and_run(
            model,
            lambda m, x, y: ((m(x) - y) ** 2).mean(),
            torch.randn(2, 4),
            torch.randn(2, 4),
        )
        gm = joint.graph_module
        get_attrs = [n for n in gm.graph.nodes if n.op == "get_attr"]
        self.assertEqual(len(get_attrs), 0, "no parameter get_attr nodes expected")
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        self.assertGreaterEqual(
            len(placeholders),
            gm.num_state_inputs,
            "state placeholders should lead the graph",
        )

    def test_state_fqns_and_param_flag(self):
        """Test state_fqns order and the buffer-vs-param flag."""
        model = _LinearWithBuffer()
        joint, _, _ = _trace_and_run(
            model,
            lambda m, x, y: ((m(x) - y) ** 2).mean(),
            torch.randn(2, 4),
            torch.randn(2, 4),
        )
        gm = joint.graph_module
        self.assertEqual(joint.state_fqns, ["lin.weight", "lin.bias", "buf"])
        self.assertEqual(gm.state_is_param, [True, True, False])
        self.assertEqual(gm.num_state_inputs, 3)

    def test_param_names_only_trainable(self):
        """Test ``param_names`` lists trainable parameter names."""
        model = _LinearWithBuffer()
        joint, _, _ = _trace_and_run(
            model,
            lambda m, x, y: ((m(x) - y) ** 2).mean(),
            torch.randn(2, 4),
            torch.randn(2, 4),
        )
        self.assertEqual(joint.param_names, ["lin.weight", "lin.bias"])


class TestRunTracedGraph(unittest.TestCase):
    """``run_traced_graph`` returns loss+grads matching autograd."""

    def test_loss_and_grads_match_autograd(self):
        """Test the joint graph's (loss, grads) equal autograd's on the live model."""
        model = _LinearWithBuffer()
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)

        joint, loss, grads = _trace_and_run(
            model,
            lambda m, x, y: ((m(x) - y) ** 2).mean(),
            x,
            y,
        )

        # Reference via the live model.
        ref_loss = ((model(x) - y) ** 2).mean()
        ref_grads = torch.autograd.grad(ref_loss, [model.lin.weight, model.lin.bias])

        self.assertTrue(torch.isclose(loss, ref_loss, atol=1e-5).item())
        self.assertEqual(len(grads), len(ref_grads))
        for got, expected in zip(grads, ref_grads):
            self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    def test_run_raises_on_state_mismatch(self):
        """Test ``run_traced_graph`` raises when the model's state changed."""
        model = _LinearWithBuffer()
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        joint, _, _ = _trace_and_run(
            model,
            lambda m, x, y: ((m(x) - y) ** 2).mean(),
            x,
            y,
        )

        # Attach an extra buffer so the state keys diverge from the trace.
        model.register_buffer("extra", torch.zeros(1))
        with self.assertRaises(ValueError) as ctx:
            run_traced_graph(joint, model, {"x": x, "y": y})
        self.assertIn("different parameter/buffer names", str(ctx.exception))


class _TiedEmbedding(nn.Module):
    """Embedding whose weight is tied to the output projection's weight."""

    def __init__(self) -> None:
        """Tie ``lm.weight`` to ``emb.weight`` (one shared tensor)."""
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.lm = nn.Linear(4, 10, bias=False)
        self.lm.weight = self.emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed then project back to vocab logits using the shared weight."""
        return self.lm(self.emb(x))


class TestTiedWeights(unittest.TestCase):
    """Tied parameters trace to one placeholder and one undoubled gradient."""

    def test_tied_weight_one_placeholder_and_full_gradient(self):
        """Test the deduped alias is one static input with the full gradient.

        Without dedup the same tensor is flattened once per FQN, so the graph
        carries two placeholders and FSDP later inserts two collectives; and a
        naive ``remove_duplicate=True`` (without ``tie_weights``) would bake the
        second use in as a ``get_attr`` constant instead. The tracer must land
        in the middle: one placeholder, no ``get_attr``, and the shared weight's
        gradient equal to autograd's full (not doubled) gradient.
        """
        torch.manual_seed(0)
        model = _TiedEmbedding()
        x = torch.randint(0, 10, (2, 3))
        y = torch.randn(2, 3, 10)

        joint, loss, grads = _trace_and_run(
            model, lambda m, x, y: ((m(x) - y) ** 2).mean(), x, y
        )
        gm = joint.graph_module
        self.assertEqual(joint.state_fqns, ["emb.weight"])
        self.assertEqual(gm.num_state_inputs, 1)
        self.assertEqual(
            [n for n in gm.graph.nodes if n.op == "get_attr"],
            [],
            "the tied alias must not be baked in as a get_attr constant",
        )
        self.assertEqual(len(grads), 1, "one shared weight -> one gradient")

        ref_loss = ((model(x) - y) ** 2).mean()
        ref_grad = torch.autograd.grad(ref_loss, [model.emb.weight])[0]
        self.assertTrue(torch.isclose(loss, ref_loss, atol=1e-5).item())
        self.assertTrue(torch.allclose(grads[0], ref_grad, atol=1e-5))


class TestExtractModuleState(unittest.TestCase):
    """``extract_module_state`` merges params and buffers."""

    def test_merges_parameters_and_buffers(self):
        """Test the state dict contains both parameters and buffers."""
        model = _LinearWithBuffer()
        state = extract_module_state(model)
        self.assertIn("lin.weight", state)
        self.assertIn("lin.bias", state)
        self.assertIn("buf", state)
        self.assertEqual(len(state), 3)


if __name__ == "__main__":
    unittest.main()
