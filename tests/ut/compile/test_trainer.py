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
"""Unit tests for ``hyper_parallel.compile.trainer.GraphTrainer``.

Covers the parts of the trainer that the pass-level tests cannot reach:

1. Single-process ``train_step`` / ``train`` (no ``torch.distributed`` init):
   the graph is compiled lazily on the first batch and runs as plain graph
   mode, because ``FSDPPass`` early-returns when distributed is not up.
2. ``compile`` with ``fsdp_enabled=True`` and *no* dist does not raise -- the
   old hard guard contradicted the FSDP pass's own ``world_size==1`` no-op.
3. ``_init_device_mesh`` both branches (fallback 1-D mesh over the world, and
   the external automodel ``MeshContext`` path that back-fills ``fsdp_degree``
   and registers the FSDP sub-group) -- exercised via mocks so no real
   backend is needed.
4. ``optimizer_step`` grad-clip path.
5. ``train`` loop bookkeeping: ``log_interval`` printing, ``max_steps``,
   ``log_fn`` callback, and non-iterator iterables.
6. ``to()`` (device move) and ``set_pytree_pre_hook``.

Tracing uses a tiny ``nn.Linear`` model and the same joint-graph capture the
tracer tests validate; only the trainer wiring is asserted here.
"""

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
from torch import nn

from hyper_parallel.compile.parallel_config import PassConfig
from hyper_parallel.compile.trainer import GraphTrainer


def _make_model() -> nn.Linear:
    """A tiny CPU-linear model; tracing is cheap and deterministic."""
    return nn.Linear(4, 4)


def _mse_train_fn(model, x, y) -> torch.Tensor:
    """Training function: mean-squared error between prediction and target."""
    return ((model(x) - y) ** 2).mean()


def _batches(n=2, dim=2):
    """Yield ``n`` ``(input, label)`` batches."""
    x = torch.randn(dim, 4)
    y = torch.randn(dim, 4)
    for _ in range(n):
        yield x, y


class TestGraphTrainerCompile(unittest.TestCase):
    """``compile`` / lazy ``train_step`` without distributed."""

    def test_train_step_compiles_lazily_without_dist(self):
        """Test the first ``train_step`` compiles and the graph runs.

        With ``fsdp_enabled=True`` (the default) but ``dist`` uninitialised,
        the trainer must NOT raise: it skips mesh setup, and ``FSDPPass``
        no-ops, so the step runs as plain graph mode. Before the guard was
        relaxed this raised ``RuntimeError``.
        """
        model = _make_model()
        tr = GraphTrainer(
            model=model,
            train_fn=_mse_train_fn,
            pass_config=PassConfig(),
            device=torch.device("cpu"),
        )

        loss = tr.train_step(torch.randn(2, 4), torch.randn(2, 4))

        self.assertIsNotNone(tr._joint_graph, "compile should populate the joint graph")
        self.assertIsNotNone(tr.optimizer)
        self.assertIsInstance(loss, torch.Tensor)
        # A real forward/backward ran: the model now holds a non-zero grad.
        self.assertIsNotNone(model.weight.grad)
        self.assertGreater(float(model.weight.grad.abs().sum()), 0.0)

    def test_explicit_compile_and_optimizer_step(self):
        """Test ``compile`` then ``optimizer_step`` updates parameters."""
        model = _make_model()
        tr = GraphTrainer(
            model=model,
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        before = model.weight.detach().clone()

        tr.compile(torch.randn(2, 4), torch.randn(2, 4))
        tr.train_step(torch.randn(2, 4), torch.randn(2, 4))
        tr.optimizer_step()

        self.assertFalse(
            torch.equal(before, model.weight),
            "optimizer_step should update parameters from gradients",
        )
        # optimizer.step() then zero_grad() leaves .grad cleared.
        self.assertIsNone(model.weight.grad)

    def test_optimizer_step_grad_clip(self):
        """Test ``optimizer_step`` applies grad-clip when configured."""
        model = _make_model()
        tr = GraphTrainer(
            model=model,
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            optimizer_config={"lr": 1e-3, "grad_clip": 1.0},
            device=torch.device("cpu"),
        )
        tr.train_step(torch.randn(2, 4), torch.randn(2, 4))

        # A very large loss guarantees an un-clipped gradient of norm > 1.
        with torch.no_grad():
            model.weight.mul_(10.0)
        # Reset the graph so the next step recomputes a huge loss.
        tr._joint_graph = None
        tr.train_step(torch.randn(2, 4), torch.randn(2, 4))
        grad_norm = float(model.weight.grad.norm())
        self.assertGreater(grad_norm, 1.0)

        with patch(
            "torch.nn.utils.clip_grad_norm_",
            wraps=torch.nn.utils.clip_grad_norm_,
        ) as mock_clip:
            tr.optimizer_step()
        mock_clip.assert_called_once()
        self.assertEqual(mock_clip.call_args.args[1], 1.0)


class TestGraphTrainerTrainLoop(unittest.TestCase):
    """``train`` drives the loop, honours max_steps/log_interval."""

    def test_train_runs_to_max_steps(self):
        """Test ``train`` stops after ``max_steps`` and returns per-step losses."""
        tr = GraphTrainer(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        losses = tr.train(iter(_batches(5)), max_steps=3)

        self.assertEqual(len(losses), 3, "train should truncate at max_steps")
        self.assertTrue(all(isinstance(t, torch.Tensor) for t in losses))

    def test_train_accepts_reiterable_and_uses_optimizer_step(self):
        """Test ``train`` accepts a non-iterator iterable, and steps the optimizer."""
        tr = GraphTrainer(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        before = tr.model.weight.detach().clone()
        # A list is a non-iterator (re-iterable) iterable: it must be accepted
        # just like a generator, but it can be iterated more than once. Passing
        # a generator here would leave the reiterable regression path uncovered.
        losses = tr.train(list(_batches(2)))
        self.assertEqual(len(losses), 2)
        # The loop advances the optimizer each step, so weights move.
        self.assertFalse(torch.equal(before, tr.model.weight))

    def test_train_log_interval_prints_on_rank0(self):
        """Test ``train`` prints a loss line on the log_interval (rank 0)."""
        tr = GraphTrainer(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            tr.train(_batches(4), log_interval=2)

        out = buf.getvalue()
        # Steps 2 and 4 are printed (rank 0 and step % log_interval == 0).
        self.assertIn("Step 2 | Loss:", out)
        self.assertIn("Step 4 | Loss:", out)

    def test_train_log_fn_callback(self):
        """Test ``train`` calls a supplied ``log_fn`` instead of printing."""
        tr = GraphTrainer(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        seen = []
        buf = io.StringIO()
        with redirect_stdout(buf):
            tr.train(
                _batches(4),
                log_interval=1,
                log_fn=lambda step, loss: seen.append((step, float(loss))),
            )
        self.assertEqual(len(seen), 4)
        self.assertEqual(seen[-1][0], 4)
        # No rank-0 print when a log_fn is supplied.
        self.assertEqual(buf.getvalue(), "")

    def test_train_no_log_interval_is_silent(self):
        """Test ``train`` prints nothing when no log_interval is given."""
        tr = GraphTrainer(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            tr.train(_batches(2))
        self.assertEqual(buf.getvalue(), "")


class TestGraphTrainerDeviceMesh(unittest.TestCase):
    """``_init_device_mesh`` fallback and external-mesh branches."""

    def test_init_device_mesh_fallback_registers_fsdp(self):
        """Test the no-mesh fallback builds a 1-D fsdp mesh over the world."""
        tr = GraphTrainer(
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
            patch("hyper_parallel.compile.trainer.dist", mock_dist),
            patch(
                "hyper_parallel.compile.trainer._register_process_group", mock_register
            ),
            patch("hyper_parallel.compile.trainer.init_device_mesh", mock_init_mesh),
        ):
            tr._init_device_mesh(None)

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
            tr.pass_config.fsdp_degree,
            4,
            "fallback should back-fill fsdp_degree from the world size",
        )

    def test_init_device_mesh_external_uses_fsdp_shard_submesh(self):
        """Test the automodel ``MeshContext`` path back-fills ``fsdp_degree``.

        The FSDP group is a proper sub-group of the world (TP+FSDP hybrid), so
        ``fsdp_degree`` must come from the sub-mesh, not ``world_size``.
        """
        tr = GraphTrainer(
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
            "hyper_parallel.compile.trainer._register_process_group", mock_register
        ):
            tr._init_device_mesh(mesh_context)

        self.assertEqual(
            mock_non_moe.__getitem__.call_args.args,
            ("fsdp_shard",),
            "must resolve the fsdp_shard axis of a hybrid mesh",
        )
        self.assertEqual(
            tr.pass_config.fsdp_degree,
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
        tr = GraphTrainer(
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
            "hyper_parallel.compile.trainer._register_process_group", mock_register
        ):
            tr._init_device_mesh(mesh_context)

        self.assertEqual(
            mock_mesh.__getitem__.call_args.args,
            ("dp",),
            "must fall back to the dp axis when fsdp_shard is absent",
        )
        self.assertEqual(
            tr.pass_config.fsdp_degree,
            8,
            "a mesh without fsdp_shard should use the dp axis",
        )
        self.assertEqual(mock_register.call_args.args[1], "DPPG")


class TestGraphTrainerHelpers(unittest.TestCase):
    """``to``, ``set_pytree_pre_hook``, and device placement."""

    def test_to_moves_model_and_device(self):
        """Test ``to`` moves the model and records the device."""
        tr = GraphTrainer(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        returned = tr.to(torch.device("cpu"))
        self.assertIs(returned, tr, "to() should be chainable")
        self.assertEqual(tr.device, torch.device("cpu"))

    def test_set_pytree_pre_hook_runs_on_compile(self):
        """Test the pre-hook fires exactly once, before the first compile."""
        tr = GraphTrainer(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        calls = []
        returned = tr.set_pytree_pre_hook(lambda: calls.append(True))
        self.assertIs(returned, tr, "set_pytree_pre_hook should be chainable")

        # First train_step triggers lazily and fires the hook.
        tr.train_step(torch.randn(2, 4), torch.randn(2, 4))
        self.assertEqual(len(calls), 1, "pre-hook should fire on the first compile")
        # A second step does NOT recompile, so the hook does not re-fire.
        tr.train_step(torch.randn(2, 4), torch.randn(2, 4))
        self.assertEqual(len(calls), 1)

    def test_place_on_device_moves_tensors_only(self):
        """Test ``_place_on_device`` moves tensors and leaves other objects."""
        tr = GraphTrainer(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        t = torch.randn(2, 4)
        result = tr._place_on_device((t, "not-a-tensor"))
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertEqual(result[1], "not-a-tensor")


if __name__ == "__main__":
    unittest.main()
