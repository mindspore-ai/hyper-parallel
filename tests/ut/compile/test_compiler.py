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

from copy import deepcopy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from hyper_parallel.compile.compiler import GraphCompiler
from hyper_parallel.compile.pass_config import PassConfig
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models.build_options import FSDP2Config

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"


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

    def test_terminal_wrapper_preserves_training_metadata(self) -> None:
        """Execute the returned module while preserving loss and gradient output layout."""
        model = _make_model()
        x, y = _fixed_batch()
        expected_loss = _mse_train_fn(model, x=x, y=y)
        expected_grads = torch.autograd.grad(expected_loss, tuple(model.parameters()))
        compiler = GraphCompiler(model, _mse_train_fn, PassConfig(fsdp_enabled=False))
        wrappers = []

        class RuntimeWrapper(nn.Module):
            """Mirror the backend's positional state-and-input calling convention."""

            def __init__(self, inner):
                """Keep the executable without its tracer metadata."""
                super().__init__()
                self.inner = inner

            def forward(self, *args):
                """Forward every flat state and user input positionally."""
                return self.inner(*args)

        def wrap(graph_module, **_kwargs):
            """Return a terminal executable without copying tracer metadata."""
            wrappers.append(RuntimeWrapper(graph_module))
            return wrappers[-1]

        self.assertIsNone(compiler.execution_backend)
        with patch("hyper_parallel.compile.compiler.PassPipeline.run", side_effect=wrap):
            loss, _ = compiler.forward_backward(x=x, y=y)

        self.assertIs(compiler._joint_graph.graph_module, wrappers[0])
        self.assertEqual(compiler.execution_backend, "fx")
        self.assertEqual(wrappers[0].num_state_inputs, len(tuple(model.parameters())))
        self.assertEqual(wrappers[0].state_fqns, compiler._joint_graph.state_fqns)
        self.assertEqual(wrappers[0].gradient_usage, (True, True))
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        for parameter, expected in zip(model.parameters(), expected_grads):
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)

    def test_unit_bindings_reach_passes_with_tied_expert(self) -> None:
        """Claim nested experts before the root and keep every alias on EDP."""
        model = nn.Module()
        model.dense = nn.Linear(4, 4, bias=False)
        model.mlp = nn.Module()
        model.mlp.experts = nn.Linear(4, 4, bias=False)
        model.tied_expert = model.mlp.experts.weight
        dense = init_device_mesh("cpu", (4, 1), mesh_dim_names=("fsdp_shard", "tp"),
                                 rank_list=(0, 1, 2, 3), init_backend=False)
        expert = init_device_mesh("cpu", (2, 2), mesh_dim_names=("edp_shard", "ep"),
                                  rank_list=(0, 1, 2, 3), init_backend=False)
        compiler = GraphCompiler(
            model, _mse_train_fn, PassConfig(fsdp_enabled=True), device=torch.device("cpu"),
            trainer_config=SimpleNamespace(fsdp_config=FSDP2Config(dp_shard_size=4, edp_shard_size=2)),
            mesh_context=MeshContext(fsdp_non_moe_mesh=dense, fsdp_moe_mesh=expert, dp_size=4, ep_size=2),
            source_shard_info={"dense.weight": ((Replicate(),), dense["tp"]),
                               "mlp.experts.weight": ((Shard(0),), expert["ep"])},
        )
        with patch.object(compiler, "_register_mesh_group") as register:
            compiler._bind_parameter_meshes()
            kwargs = compiler._build_pass_kwargs()
        groups = kwargs["parameter_groups"]
        registered = {call.args[0]: call.args[1] for call in register.call_args_list}
        self.assertEqual(len(registered), 2)
        self.assertEqual(groups["tied_expert"], groups["mlp.experts.weight"])
        self.assertEqual(groups["tied_expert"][1], 2)
        self.assertEqual(groups["dense.weight"][1], 4)
        self.assertIs(registered[groups["tied_expert"][0]], expert["edp_shard"])
        self.assertIs(registered[groups["dense.weight"][0]], dense["fsdp_shard"])
        self.assertIs(model.tied_expert, model.mlp.experts.weight)
        self.assertFalse(hasattr(model, "hsdp_scheduler"))
        self.assertFalse(hasattr(model.mlp.experts, "hsdp_scheduler"))

    def test_external_parameters_rejected_before_compile_side_effects(self) -> None:
        """Reject external parameters at explicit and lazy graph entry points."""
        x, y = _fixed_batch()
        for fsdp_enabled in (False, True):
            for entry_point in ("compile", "forward_backward"):
                with self.subTest(fsdp_enabled=fsdp_enabled, entry_point=entry_point):
                    model = _make_model()
                    compiler = GraphCompiler(
                        model, _mse_train_fn, PassConfig(fsdp_enabled=fsdp_enabled, ac_mode="full"),
                        device=torch.device("cpu"),
                    )
                    model._hp_external_state = SimpleNamespace(parameters={"weight": model.weight})
                    preparation = MagicMock()
                    compiler.set_pytree_pre_hook(preparation.pytree)
                    with (
                        patch.object(compiler, "_init_device_mesh", preparation.mesh),
                        patch("hyper_parallel.compile.compiler.prepare_activation_checkpoint", preparation.checkpoint),
                        patch("hyper_parallel.compile.compiler.trace_model_graph", preparation.trace),
                        self.assertRaisesRegex(ValueError, "externally managed parameters"),
                    ):
                        getattr(compiler, entry_point)(x=x, y=y)
                    self.assertEqual(preparation.mock_calls, [])

    def test_empty_external_state_preserves_loss_and_gradients(self) -> None:
        """An empty provider leaves ordinary graph execution numerically unchanged."""
        model = _make_model()
        x, y = _fixed_batch()
        expected_loss = _mse_train_fn(model, x=x, y=y)
        expected_grads = torch.autograd.grad(expected_loss, tuple(model.parameters()))
        compiler = GraphCompiler(model, _mse_train_fn, device=torch.device("cpu"))
        model._hp_external_state = SimpleNamespace(parameters={})

        compiler.compile(x=x, y=y)
        loss, _ = compiler.forward_backward(x=x, y=y)

        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        for parameter, expected in zip(model.parameters(), expected_grads):
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)

    def test_parameter_layouts_are_lazy_and_refresh_replaced_parameters(self):
        """Only final physical storage is described; no gradient references are cached."""
        model = _make_model()
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp_shard",),
                                rank_list=(0,), init_backend=False)
        compiler = GraphCompiler(model, _mse_train_fn, PassConfig(fsdp_enabled=False),
                                 mesh_context=MeshContext(fsdp_non_moe_mesh=mesh))
        self.assertIsNone(compiler.parameter_layouts)
        x, y = _fixed_batch()
        compiler.forward_backward(x=x, y=y)
        first = compiler.parameter_layouts
        self.assertIs(compiler.parameter_layouts, first)
        old = model.weight
        model.weight = nn.Parameter(old.detach().clone())
        updated = compiler.parameter_layouts
        self.assertIsNot(updated, first)
        self.assertIn(model.weight, updated)
        self.assertNotIn(old, updated)
        self.assertEqual(updated[model.weight].tensor_shape, (4, 4))
        self.assertFalse(hasattr(compiler, "clip_grad_norm_"))

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

        loss, _ = comp.forward_backward(x=torch.randn(2, 4), y=torch.randn(2, 4))

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
        loss, _ = comp.forward_backward(x=torch.randn(2, 4), y=torch.randn(2, 4))

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

    def test_gradients_match_parameter_layout(self):
        """Preserve eager's optimizer-facing layout for dense and transposed weights."""
        for transposed in (False, True):
            with self.subTest(transposed=transposed):
                model = _make_model()
                if transposed:
                    model.weight = nn.Parameter(model.weight.T)
                x, y = _fixed_batch()
                expected = torch.autograd.grad(
                    _mse_train_fn(model, x=x, y=y), tuple(model.parameters())
                )
                compiler = GraphCompiler(
                    model,
                    _mse_train_fn,
                    PassConfig(fsdp_enabled=False),
                    device=torch.device("cpu"),
                )
                compiler.forward_backward(x=x, y=y)
                for parameter, gradient in zip(model.parameters(), expected):
                    self.assertEqual(parameter.grad.stride(), parameter.stride())
                    torch.testing.assert_close(
                        parameter.grad, gradient, rtol=1e-5, atol=1e-6
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

    def test_unused_parameters_preserve_adamw_skip_semantics(self):
        """An absent autograd edge must not create weight decay or optimizer state."""
        model = _make_model()
        model.register_parameter("unused", nn.Parameter(torch.ones(4)))
        model.register_parameter("zero_gradient", nn.Parameter(torch.ones(4)))
        reference = deepcopy(model)

        def loss_with_zero_gradient(current, *, x, y):
            """Keep a real zero gradient distinct from an absent autograd edge."""
            return _mse_train_fn(current, x=x, y=y) + current.zero_gradient.sum() * 0

        compiler = GraphCompiler(model, loss_with_zero_gradient, PassConfig(fsdp_enabled=False),
                                 device=torch.device("cpu"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1, foreach=False)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.1, foreach=False)
        x, y = _fixed_batch()
        for _ in range(10):
            expected_loss = loss_with_zero_gradient(reference, x=x, y=y)
            expected_loss.backward()
            loss, _ = compiler.forward_backward(x=x, y=y)
            torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
            self.assertIsNone(model.unused.grad)
            torch.testing.assert_close(model.zero_gradient.grad, torch.zeros(4), rtol=0, atol=0)
            optimizer.step()
            reference_optimizer.step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertNotIn(model.unused, optimizer.state)
            self.assertNotIn(reference.unused, reference_optimizer.state)
            self.assertIn(model.zero_gradient, optimizer.state)
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)


class TestGraphCompilerDevice(unittest.TestCase):
    """``to`` (device move)."""

    def test_pytree_pre_hook_runs_before_lazy_compile(self):
        """The compiler should run tracer setup before its first lazy compile."""
        comp = GraphCompiler(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        calls = []
        comp.set_pytree_pre_hook(lambda: calls.append("hook"))

        comp.forward_backward(x=torch.randn(2, 4), y=torch.randn(2, 4))

        self.assertEqual(calls, ["hook"])

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
            ("fsdp_shard",),
            "the 1-D fallback mesh must be indexed by its 'fsdp_shard' dim",
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
        mesh_context.ep_size = 1
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
        mesh_context.ep_size = 1
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

    def test_init_device_mesh_fallback_ddp_registers_replicate(self):
        """Test ``dp_mode='ddp'`` builds a 1-D ``dp_replicate`` mesh."""
        comp = GraphCompiler(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=True, dp_mode="ddp"),
            device=torch.device("cpu"),
        )
        mock_dist = MagicMock()
        mock_dist.get_world_size.return_value = 4
        fake_sub = MagicMock()
        fake_sub.size.return_value = 4
        fake_sub.get_group.return_value = "REPL_PG"
        fake_mesh = MagicMock()
        fake_mesh.__getitem__.return_value = fake_sub
        mock_init_mesh = MagicMock(return_value=fake_mesh)
        mock_register = MagicMock()

        with (
            patch("hyper_parallel.compile.compiler.dist", mock_dist),
            patch(
                "hyper_parallel.compile.compiler._register_process_group",
                mock_register,
            ),
            patch("hyper_parallel.compile.compiler.init_device_mesh", mock_init_mesh),
        ):
            comp._init_device_mesh(None)

        self.assertEqual(
            fake_mesh.__getitem__.call_args.args,
            ("fsdp_replicate",),
            "the DDP fallback mesh must be indexed by its 'fsdp_replicate' dim",
        )
        self.assertEqual(mock_register.call_args.args[0], "dp_replicate")
        self.assertEqual(mock_register.call_args.args[1], "REPL_PG")
        self.assertEqual(comp.pass_config.dp_replicate_degree, 4)

    def test_init_device_mesh_fallback_hsdp_builds_2d(self):
        """Test ``dp_mode='hsdp'`` builds a 2-D ``(dp_replicate, fsdp)`` mesh."""
        comp = GraphCompiler(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(
                fsdp_enabled=True,
                dp_mode="hsdp",
                fsdp_degree=2,
                dp_replicate_degree=2,
            ),
            device=torch.device("cpu"),
        )
        mock_dist = MagicMock()
        mock_dist.get_world_size.return_value = 4
        repl_sub = MagicMock()
        repl_sub.size.return_value = 2
        repl_sub.get_group.return_value = "REPL_PG"
        shard_sub = MagicMock()
        shard_sub.size.return_value = 2
        shard_sub.get_group.return_value = "SHARD_PG"
        fake_mesh = MagicMock()
        fake_mesh.__getitem__.side_effect = {
            "fsdp_replicate": repl_sub,
            "fsdp_shard": shard_sub,
        }.__getitem__
        mock_init_mesh = MagicMock(return_value=fake_mesh)
        mock_register = MagicMock()

        with (
            patch("hyper_parallel.compile.compiler.dist", mock_dist),
            patch(
                "hyper_parallel.compile.compiler._register_process_group",
                mock_register,
            ),
            patch("hyper_parallel.compile.compiler.init_device_mesh", mock_init_mesh),
        ):
            comp._init_device_mesh(None)

        call = mock_init_mesh.call_args
        self.assertEqual(
            call.args[1],
            (2, 2),
            "HSDP fallback must build a (replicate, shard) 2-D mesh",
        )
        self.assertEqual(
            call.kwargs.get("mesh_dim_names"),
            ("fsdp_replicate", "fsdp_shard"),
            "HSDP mesh axes must match automodel (fsdp_replicate / fsdp_shard)",
        )
        registered = {c.args[0]: c.args[1] for c in mock_register.call_args_list}
        self.assertEqual(
            registered,
            {"dp_replicate": "REPL_PG", "fsdp": "SHARD_PG"},
            "both HSDP groups must be registered",
        )
        self.assertEqual(comp.pass_config.dp_replicate_degree, 2)
        self.assertEqual(comp.pass_config.fsdp_degree, 2)

    def test_init_device_mesh_pp_fsdp_builds_2d(self):
        """Test the PP+FSDP hybrid builds a 2-D ``(pp, fsdp_shard)`` mesh."""
        comp = GraphCompiler(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(
                fsdp_enabled=True, pp_enabled=True, pp_degree=2, fsdp_degree=2
            ),
            device=torch.device("cpu"),
        )
        mock_dist = MagicMock()
        mock_dist.get_world_size.return_value = 4

        def _sub(group: str) -> MagicMock:
            sub = MagicMock()
            sub.size.return_value = 2
            sub.get_group.return_value = group
            return sub

        fake_mesh = MagicMock()
        fake_mesh.__getitem__.side_effect = {
            "pp": _sub("PP_PG"),
            "fsdp_shard": _sub("SHARD_PG"),
        }.__getitem__
        mock_init_mesh = MagicMock(return_value=fake_mesh)
        mock_register = MagicMock()
        with (
            patch("hyper_parallel.compile.compiler.dist", mock_dist),
            patch(
                "hyper_parallel.compile.compiler._register_process_group",
                mock_register,
            ),
            patch("hyper_parallel.compile.compiler.init_device_mesh", mock_init_mesh),
        ):
            comp._init_device_mesh(None)

        call = mock_init_mesh.call_args
        self.assertEqual(call.args[1], (2, 2))
        self.assertEqual(call.kwargs.get("mesh_dim_names"), ("pp", "fsdp_shard"))
        self.assertEqual(call.kwargs.get("rank_list"), tuple(range(4)))
        registered = {c.args[0]: c.args[1] for c in mock_register.call_args_list}
        self.assertEqual(registered, {"pp": "PP_PG", "fsdp": "SHARD_PG"})
        self.assertEqual(comp.pass_config.pp_degree, 2)
        self.assertEqual(comp.pass_config.fsdp_degree, 2)
        self.assertEqual(comp._pp_group_name, "pp")

    def test_build_pass_kwargs_hands_pp_group_to_pp_pass(self):
        """Test ``_build_pass_kwargs`` forwards the registered pp group name."""
        comp = GraphCompiler(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(
                fsdp_enabled=True, pp_enabled=True, pp_degree=2, fsdp_degree=2
            ),
            device=torch.device("cpu"),
        )
        self.assertIsNone(comp._build_pass_kwargs().get("pp_group_name"))
        comp._pp_group_name = "pp"
        self.assertEqual(comp._build_pass_kwargs()["pp_group_name"], "pp")

    def test_build_pass_kwargs_hands_gradient_policy_to_fsdp_pass(self):
        """Forward the DP loss domain independently of the expert group size."""
        context = MagicMock(dp_size=8, cp_size=1, tp_size=1, loss_parallel=False, ep_size=4, edp_shard_size=2)
        comp = GraphCompiler(
            model=_make_model(), train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_degree=4), device=torch.device("cpu"), mesh_context=context,
        )
        self.assertNotIn("gradient_reductions", comp._build_pass_kwargs())
        comp._expert_param_names = {"weight"}
        kwargs = comp._build_pass_kwargs()
        self.assertEqual(set(kwargs["gradient_reductions"]), {"weight"})
        self.assertEqual(kwargs["gradient_reductions"]["weight"].reduce_op, "sum")
        self.assertEqual(kwargs["gradient_reductions"]["weight"].divisor, 8)
        self.assertNotIn("expert_parallel_degree", kwargs)
        self.assertEqual(kwargs["expert_fsdp_group_name"], "fsdp_expert")

    def test_tp_source_policies_and_replicated_gradients_are_forwarded(self):
        """Source-sharded weights and tied aliases avoid replicated-TP gradient sums."""
        model = _make_model()
        model.register_parameter("alias", model.weight)
        replica, shard = MagicMock(), MagicMock()
        replica.is_replicate.return_value = True
        shard.is_replicate.return_value = False
        tp = MagicMock(mesh_dim_names=("tp",), rank_list=(0, 1))
        tp.size.return_value = 2
        tp.__getitem__.return_value = tp
        tp.flatten.return_value = tp
        source = {"weight": ((shard,), tp), "bias": ((replica,), tp)}
        context = MagicMock(dp_size=4, cp_size=1, tp_size=2, loss_parallel=False)
        comp = GraphCompiler(model, _mse_train_fn, PassConfig(fsdp_degree=4),
                             mesh_context=context, source_shard_info=source)
        with patch.object(comp, "_register_mesh_group") as register:
            kwargs = comp._build_pass_kwargs()
        self.assertEqual(set(kwargs["gradient_reductions"]), {"weight", "bias", "alias"})
        self.assertTrue(all(policy.reduce_op == "sum" and policy.divisor == 8
                            for policy in kwargs["gradient_reductions"].values()))
        self.assertEqual(kwargs["source_gradient_groups"], {"bias": "fsdp_source_0"})
        register.assert_called_once_with("fsdp_source_0", tp)
        self.assertIs(comp.source_layout_for_parameter("alias"), source["weight"])

    def test_tp_requires_source_layout_before_capture(self):
        """Localized TP parameters cannot silently lose their gradient-sync semantics."""
        context = MagicMock(tp_size=2, ep_size=1)
        for source in (None, {}):
            comp = GraphCompiler(_make_model(), _mse_train_fn, mesh_context=context, source_shard_info=source)
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "Graph TP requires source_shard_info"):
                comp.compile()

    def test_init_device_mesh_pp_rejects_invalid_topology(self):
        """Test PP rejects an external MeshContext, HSDP, and untiled degrees."""
        cases = (
            ({"fsdp_degree": 2}, MagicMock(), "fsdp", "MeshContext"),
            ({"fsdp_degree": 2, "dp_mode": "hsdp"}, None, "hsdp", "3-D"),
            ({"pp_degree": 3, "fsdp_degree": 2}, None, "fsdp", "tile the world"),
        )
        for overrides, context, dp_mode, match in cases:
            with self.subTest(match=match):
                cfg = {"fsdp_enabled": True, "pp_enabled": True, "pp_degree": 2}
                cfg.update(overrides)
                comp = GraphCompiler(
                    model=_make_model(),
                    train_fn=_mse_train_fn,
                    pass_config=PassConfig(**cfg),
                    device=torch.device("cpu"),
                )
                mock_dist = MagicMock()
                mock_dist.get_world_size.return_value = 4
                with patch("hyper_parallel.compile.compiler.dist", mock_dist):
                    with self.assertRaisesRegex(ValueError, match):
                        comp._init_pp_dp_mesh(dp_mode, context)

    def test_init_device_mesh_context_upgrades_fsdp_to_hsdp(self):
        """Test a replicate axis > 1 upgrades ``dp_mode='fsdp'`` to ``'hsdp'``.

        Eager ``fully_shard`` picks HSDP from ``dp_replicate_size > 1``; the
        graph path must not silently treat such a mesh as plain FSDP.
        """
        comp = GraphCompiler(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=True),  # dp_mode defaults "fsdp"
            device=torch.device("cpu"),
        )
        repl_sub = MagicMock()
        repl_sub.size.return_value = 2
        repl_sub.get_group.return_value = "REPL_PG"
        shard_sub = MagicMock()
        shard_sub.size.return_value = 2
        shard_sub.get_group.return_value = "SHARD_PG"
        mock_non_moe = MagicMock()
        mock_non_moe.mesh_dim_names = ("fsdp_replicate", "fsdp_shard", "tp")
        mock_non_moe.__getitem__.side_effect = {
            "fsdp_replicate": repl_sub,
            "fsdp_shard": shard_sub,
            ("fsdp_replicate", "fsdp_shard"): MagicMock(),
        }.__getitem__
        mesh_context = MagicMock()
        mesh_context.ep_size = 1
        mesh_context.fsdp_non_moe_mesh = mock_non_moe
        mesh_context.device_mesh = None
        mesh_context.dp_replicate_size = 2
        mesh_context.dp_shard_size = 2

        mock_register = MagicMock()
        with patch(
            "hyper_parallel.compile.compiler._register_process_group", mock_register
        ):
            comp._init_device_mesh(mesh_context)

        self.assertEqual(comp.pass_config.dp_mode, "hsdp")
        registered = {c.args[0]: c.args[1] for c in mock_register.call_args_list}
        self.assertEqual(registered, {"dp_replicate": "REPL_PG", "fsdp": "SHARD_PG"})
        self.assertEqual(comp.pass_config.fsdp_degree, 2)
        self.assertEqual(comp.pass_config.dp_replicate_degree, 2)

    def test_init_device_mesh_context_downgrades_hsdp_to_fsdp(self):
        """Test a 1-wide replicate axis downgrades ``'hsdp'`` to ``'fsdp'``."""
        comp = GraphCompiler(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=True, dp_mode="hsdp"),
            device=torch.device("cpu"),
        )
        shard_sub = MagicMock()
        shard_sub.size.return_value = 4
        shard_sub.get_group.return_value = "SHARD_PG"
        mock_non_moe = MagicMock()
        mock_non_moe.mesh_dim_names = ("fsdp_replicate", "fsdp_shard", "tp")
        mock_non_moe.__getitem__.side_effect = {"fsdp_shard": shard_sub}.__getitem__
        mesh_context = MagicMock()
        mesh_context.ep_size = 1
        mesh_context.fsdp_non_moe_mesh = mock_non_moe
        mesh_context.device_mesh = None
        mesh_context.dp_replicate_size = 1
        mesh_context.dp_shard_size = 4

        mock_register = MagicMock()
        with patch(
            "hyper_parallel.compile.compiler._register_process_group", mock_register
        ):
            comp._init_device_mesh(mesh_context)

        self.assertEqual(comp.pass_config.dp_mode, "fsdp")
        self.assertEqual(mock_register.call_args.args[0], "fsdp")
        self.assertEqual(comp.pass_config.fsdp_degree, 4)

    def test_init_device_mesh_context_ddp_rejects_sharding(self):
        """Test ``dp_mode='ddp'`` with a sharding mesh raises (use ``hsdp``)."""
        comp = GraphCompiler(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=True, dp_mode="ddp"),
            device=torch.device("cpu"),
        )
        mesh_context = MagicMock()
        mesh_context.ep_size = 1
        mesh_context.fsdp_non_moe_mesh = None
        mesh_context.device_mesh = MagicMock()
        mesh_context.dp_replicate_size = 1
        mesh_context.dp_shard_size = 2

        with self.assertRaises(ValueError) as ctx:
            comp._init_device_mesh(mesh_context)
        self.assertIn("dp_shard_size", str(ctx.exception))


class TestParameterLayoutMeshFallback(unittest.TestCase):
    """Standalone compilers retain the meshes chosen during initialization."""

    def test_plain_graph_layouts_without_distributed_initialization(self):
        """Lazy metadata access needs neither FSDP wrapping nor process groups."""
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp_shard",),
                                rank_list=(0,), init_backend=False)
        for enabled in (False, True):
            with self.subTest(fsdp_enabled=enabled):
                model = _make_model()
                compiler = GraphCompiler(model, _mse_train_fn, pass_config=PassConfig(fsdp_enabled=enabled),
                                         mesh_context=MeshContext(fsdp_non_moe_mesh=mesh))
                with patch("hyper_parallel.compile.compiler.dist.is_initialized", return_value=False), \
                        patch.object(compiler, "_register_mesh_group") as register:
                    compiler.compile(x=torch.ones(2, 4), y=torch.zeros(2, 4))
                    layouts = compiler.parameter_layouts
                register.assert_not_called()
                self.assertEqual(layouts[model.weight].placements, (Replicate(),))
                self.assertEqual(layouts[model.weight].tensor_shape, tuple(model.weight.shape))

    def test_dense_and_expert_layouts_without_fsdp_config(self):
        """Fallback retains source EP sharding even for a singleton EDP axis."""
        for ep in (4, 8):
            with self.subTest(ep=ep):
                dense = init_device_mesh("cpu", (8,), mesh_dim_names=("fsdp_shard",),
                                         rank_list=tuple(range(8)), init_backend=False)
                expert = init_device_mesh("cpu", (8 // ep, ep), mesh_dim_names=("edp_shard", "ep"),
                                          rank_list=tuple(range(8)), init_backend=False)
                model = nn.Module()
                model.dense = nn.Parameter(torch.ones(2, 4))
                model.experts = nn.Linear(4, 2, bias=False)
                model.experts.local_expert_count = 2
                context = MeshContext(fsdp_non_moe_mesh=dense, fsdp_moe_mesh=expert,
                                      dp_size=8, dp_shard_size=8, ep_size=ep, edp_shard_size=8 // ep)
                compiler = GraphCompiler(model, _mse_train_fn, PassConfig(enable_overlap=False), mesh_context=context)
                with patch.object(compiler, "_register_mesh_group"):
                    compiler._init_device_mesh(context)
                compiler._bind_parameter_meshes()
                self.assertEqual(compiler._parameter_meshes, {})
                compiler.prepare_model()
                layouts = compiler.parameter_layouts
                self.assertEqual(layouts[model.dense].mesh.mesh_dim_names, ("fsdp_shard",))
                expert_layout = layouts[model.experts.weight]
                self.assertEqual(expert_layout.mesh.mesh_dim_names, ("edp_shard", "ep"))
                self.assertEqual(expert_layout.placements, (Replicate(), Shard(0)))
                self.assertEqual(expert_layout.tensor_shape, (2 * ep, 4))

    def test_legacy_named_world_metadata_flattens_dp_and_cp(self):
        """Retain world-only layout queries without a dense mesh or FSDP config."""
        for replicate in (1, 2):
            with self.subTest(replicate=replicate):
                shape = (replicate, 2, 2)
                mesh = init_device_mesh("cpu", shape, mesh_dim_names=("dp_replicate", "dp_shard", "cp"),
                                        rank_list=tuple(range(replicate * 4)), init_backend=False)
                context = MeshContext(device_mesh=mesh, dp_replicate_size=replicate, dp_shard_size=2)
                model = _make_model()
                compiler = GraphCompiler(model, _mse_train_fn, PassConfig(fsdp_enabled=False), mesh_context=context)
                compiler.prepare_model()
                with patch.object(compiler, "_register_mesh_group") as register:
                    layout = compiler.parameter_layouts[model.weight]
                register.assert_not_called()
                expected = ("fsdp_shard",) if replicate == 1 else ("dp_replicate", "fsdp_shard")
                self.assertEqual(layout.mesh.mesh_dim_names, expected)
                self.assertEqual(layout.mesh.mesh_shape[-1], 4)

    def test_hsdp_and_named_world_layout_meshes(self):
        """Use initialized child axes rather than reconstructing mesh selection."""
        for world_only in (False, True):
            with self.subTest(world_only=world_only):
                names = ("dp", "cp", "tp") if world_only else ("fsdp_replicate", "fsdp_shard", "tp")
                mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=names,
                                        rank_list=tuple(range(8)), init_backend=False)
                context = MeshContext(device_mesh=mesh, fsdp_non_moe_mesh=None if world_only else mesh,
                                      dp_replicate_size=1 if world_only else 2, dp_shard_size=2)
                model = _make_model()
                compiler = GraphCompiler(model, _mse_train_fn, mesh_context=context)
                with patch.object(compiler, "_register_mesh_group"):
                    compiler._init_device_mesh(context)
                expected = ("dp",) if world_only else ("fsdp_replicate", "fsdp_shard")
                self.assertEqual(compiler._fsdp_mesh.mesh_dim_names, expected)
                compiler.prepare_model()
                layout_names = expected if world_only else (*expected, "tp")
                self.assertEqual(compiler.parameter_layouts[model.weight].mesh.mesh_dim_names, layout_names)


if __name__ == "__main__":
    unittest.main()
