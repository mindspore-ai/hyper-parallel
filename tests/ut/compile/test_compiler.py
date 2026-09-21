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
"""Unit tests for ``hyper_parallel.compile.compiler.GraphCompiler``.

Covers the compiler surface extracted from (and now delegated to by)
``GraphTrainer``:

1. Single-process ``forward_backward`` (no ``torch.distributed`` init): the
   graph is compiled lazily on the first call and runs as plain graph mode,
   because ``FSDPPass`` early-returns when distributed is not up.
2. ``compile`` with ``fsdp_enabled=True`` and *no* dist does not raise.
3. Gradient semantics: accumulation across calls (sums into ``param.grad``)
   and the refusal to assign on a graph/live-model trainable mismatch.
4. ``_init_device_mesh`` both branches (fallback 1-D mesh over the world, and
   the external automodel ``MeshContext`` path that back-fills ``fsdp_degree``
   and registers the FSDP sub-group) -- exercised via mocks so no real
   backend is needed.
5. ``to()`` (device move).

Tracing uses a tiny ``nn.Linear`` model and the same joint-graph capture the
tracer tests validate; only the compiler wiring is asserted here.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
from torch import nn

from hyper_parallel.compile.compiler import GraphCompiler
from hyper_parallel.compile.pass_config import PassConfig


def _make_model() -> nn.Linear:
    """A tiny CPU-linear model; tracing is cheap and deterministic."""
    return nn.Linear(4, 4)


def _mse_train_fn(model, *, x, y) -> torch.Tensor:
    """Training function: mean-squared error between prediction and target."""
    return ((model(x) - y) ** 2).mean()


def _fixed_batch():
    """A deterministic ``(input, label)`` batch (same tensors every call)."""
    torch.manual_seed(7)
    return torch.randn(2, 4), torch.randn(2, 4)


class TestGraphCompilerForwardBackward(unittest.TestCase):
    """``compile`` / lazy ``forward_backward`` without distributed."""

    def test_forward_backward_compiles_lazily_without_dist(self):
        """Test the first ``forward_backward`` compiles and the graph runs.

        With ``fsdp_enabled=True`` (the default) but ``dist`` uninitialised,
        the compiler must NOT raise: it skips mesh setup, and ``FSDPPass``
        no-ops, so the step runs as plain graph mode.
        """
        model = _make_model()
        comp = GraphCompiler(
            model=model,
            train_fn=_mse_train_fn,
            pass_config=PassConfig(),
            device=torch.device("cpu"),
        )

        loss = comp.forward_backward(x=torch.randn(2, 4), y=torch.randn(2, 4))

        self.assertIsNotNone(
            comp._joint_graph, "compile should populate the joint graph"
        )
        self.assertIsInstance(loss, torch.Tensor)
        # A real forward/backward ran: the model now holds a non-zero grad.
        self.assertIsNotNone(model.weight.grad)
        self.assertGreater(float(model.weight.grad.abs().sum()), 0.0)

    def test_explicit_compile_then_forward_backward(self):
        """Test an explicit ``compile`` is honoured and gradients land."""
        model = _make_model()
        comp = GraphCompiler(
            model=model,
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )

        comp.compile(x=torch.randn(2, 4), y=torch.randn(2, 4))
        loss = comp.forward_backward(x=torch.randn(2, 4), y=torch.randn(2, 4))

        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsNotNone(model.weight.grad)
        # An explicit compile must NOT be re-triggered by forward_backward.
        joint = comp._joint_graph
        comp.forward_backward(x=torch.randn(2, 4), y=torch.randn(2, 4))
        self.assertIs(comp._joint_graph, joint)

    def test_grads_accumulate_across_forward_backward_calls(self):
        """Test repeated ``forward_backward`` SUMS into ``param.grad``.

        Accumulation (not overwrite) is the micro-batch contract: the caller
        runs several ``forward_backward`` calls before its optimizer step.
        The same batch twice must therefore yield exactly twice the gradient.
        """
        comp = GraphCompiler(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        x, y = _fixed_batch()

        comp.forward_backward(x=x, y=y)
        first = comp.model.weight.grad.clone()
        self.assertIsNotNone(first)

        comp.forward_backward(x=x, y=y)
        self.assertTrue(
            torch.allclose(comp.model.weight.grad, 2 * first),
            "an identical second step must double the accumulated gradient",
        )

    def test_grad_count_mismatch_raises(self):
        """Test the compiler refuses to assign on a trainable-count mismatch.

        Flipping the graph's ``state_is_param`` flag makes the compiler
        expect one fewer gradient than the graph emits; it must raise
        instead of silently misaligning gradients onto parameters.
        """
        comp = GraphCompiler(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        comp.compile(x=torch.randn(2, 4), y=torch.randn(2, 4))

        # Linear has exactly two parameters (weight, bias): hiding one makes
        # the expected trainable count diverge from the graph's grad count.
        comp._joint_graph.graph_module.state_is_param[0] = False
        with self.assertRaises(ValueError):
            comp.forward_backward(x=torch.randn(2, 4), y=torch.randn(2, 4))


class TestGraphCompilerDevice(unittest.TestCase):
    """``to`` (device move)."""

    def test_to_moves_model_and_device(self):
        """Test ``to`` moves the model and records the device."""
        comp = GraphCompiler(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        returned = comp.to(torch.device("cpu"))
        self.assertIs(returned, comp, "to() should be chainable")
        self.assertEqual(comp.device, torch.device("cpu"))


class TestGraphCompilerDeviceMesh(unittest.TestCase):
    """``_init_device_mesh`` fallback and external-mesh branches."""

    def test_init_device_mesh_fallback_registers_fsdp(self):
        """Test the no-mesh fallback builds a 1-D fsdp mesh over the world."""
        comp = GraphCompiler(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=True),
            device=torch.device("cpu"),
        )
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 4
        mock_dist.get_rank.return_value = 0
        mock_register = MagicMock()
        mock_init_mesh = MagicMock()

        # A fake 1-D mesh whose ["fsdp"] sub-mesh has size 4 and a group.
        fake_sub = MagicMock()
        fake_sub.size.return_value = 4
        fake_sub.get_group.return_value = "FAKE_PG"
        fake_mesh = MagicMock()
        fake_mesh.__getitem__.return_value = fake_sub

        mock_init_mesh.return_value = fake_mesh

        with (
            patch("hyper_parallel.compile.compiler.dist", mock_dist),
            patch(
                "hyper_parallel.compile.compiler._register_process_group",
                mock_register,
            ),
            patch("hyper_parallel.compile.compiler.init_device_mesh", mock_init_mesh),
        ):
            comp._init_device_mesh(None)

        mock_init_mesh.assert_called_once()
        self.assertEqual(
            fake_mesh.__getitem__.call_args.args,
            ("fsdp",),
            "the 1-D fallback mesh must be indexed by its 'fsdp' dim",
        )
        self.assertEqual(
            mock_register.call_args.args[0],
            "fsdp",
            "the FSDP group should be registered under the name 'fsdp'",
        )
        self.assertEqual(
            comp.pass_config.fsdp_degree,
            4,
            "fallback should back-fill fsdp_degree from the world size",
        )

    def test_init_device_mesh_external_uses_fsdp_shard_submesh(self):
        """Test the automodel ``MeshContext`` path back-fills ``fsdp_degree``.

        The FSDP group is a proper sub-group of the world (TP+FSDP hybrid), so
        ``fsdp_degree`` must come from the sub-mesh, not ``world_size``.
        """
        comp = GraphCompiler(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=True, fsdp_degree=None),
            device=torch.device("cpu"),
        )

        # Each axis returns a DISTINCT sub-mesh (different size + group) so a
        # wrong-axis selection surfaces as the wrong fsdp_degree / group.
        shard_sub = MagicMock()
        shard_sub.size.return_value = 2
        shard_sub.get_group.return_value = "SHARD_PG"
        repl_sub = MagicMock()
        repl_sub.size.return_value = 99
        repl_sub.get_group.return_value = "REPL_PG"
        tp_sub = MagicMock()
        tp_sub.size.return_value = 77
        tp_sub.get_group.return_value = "TP_PG"
        mock_non_moe = MagicMock()
        mock_non_moe.mesh_dim_names = ("fsdp_replicate", "fsdp_shard", "tp")
        mock_non_moe.__getitem__.side_effect = {
            "fsdp_shard": shard_sub,
            "fsdp_replicate": repl_sub,
            "tp": tp_sub,
        }.__getitem__

        mesh_context = MagicMock()
        mesh_context.fsdp_non_moe_mesh = mock_non_moe
        mesh_context.device_mesh = None

        mock_register = MagicMock()
        with patch(
            "hyper_parallel.compile.compiler._register_process_group", mock_register
        ):
            comp._init_device_mesh(mesh_context)

        self.assertEqual(
            mock_non_moe.__getitem__.call_args.args,
            ("fsdp_shard",),
            "must resolve the fsdp_shard axis of a hybrid mesh",
        )
        self.assertEqual(
            comp.pass_config.fsdp_degree,
            2,
            "external mesh should back-fill fsdp_degree from the fsdp_shard sub-mesh",
        )
        self.assertEqual(mock_register.call_args.args[0], "fsdp")
        self.assertEqual(
            mock_register.call_args.args[1],
            "SHARD_PG",
            "the fsdp_shard sub-mesh's group must be the one registered",
        )

    def test_init_device_mesh_external_falls_back_to_dp_axis(self):
        """Test an automodel mesh with no ``fsdp_shard`` uses the ``dp`` axis."""
        comp = GraphCompiler(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=True, fsdp_degree=None),
            device=torch.device("cpu"),
        )
        # Each axis returns a DISTINCT sub-mesh so picking "dp" (not cp/tp) is
        # the only way to land on fsdp_degree == 8.
        dp_sub = MagicMock()
        dp_sub.size.return_value = 8
        dp_sub.get_group.return_value = "DPPG"
        cp_sub = MagicMock()
        cp_sub.size.return_value = 55
        cp_sub.get_group.return_value = "CP_PG"
        tp_sub = MagicMock()
        tp_sub.size.return_value = 77
        tp_sub.get_group.return_value = "TP_PG"
        mock_mesh = MagicMock()
        mock_mesh.mesh_dim_names = ("dp", "cp", "tp")
        mock_mesh.__getitem__.side_effect = {
            "dp": dp_sub,
            "cp": cp_sub,
            "tp": tp_sub,
        }.__getitem__

        mesh_context = MagicMock()
        mesh_context.fsdp_non_moe_mesh = None
        mesh_context.device_mesh = mock_mesh

        mock_register = MagicMock()
        with patch(
            "hyper_parallel.compile.compiler._register_process_group", mock_register
        ):
            comp._init_device_mesh(mesh_context)

        self.assertEqual(
            mock_mesh.__getitem__.call_args.args,
            ("dp",),
            "must fall back to the dp axis when fsdp_shard is absent",
        )
        self.assertEqual(
            comp.pass_config.fsdp_degree,
            8,
            "a mesh without fsdp_shard should use the dp axis",
        )
        self.assertEqual(mock_register.call_args.args[1], "DPPG")


if __name__ == "__main__":
    unittest.main()
