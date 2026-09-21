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

Covers the training-policy parts the compiler-level tests cannot reach:

1. Single-process ``train_step`` / ``train`` (no ``torch.distributed`` init):
   the graph is compiled lazily on the first batch and runs as plain graph
   mode, because ``FSDPPass`` early-returns when distributed is not up.
2. ``compile`` with ``fsdp_enabled=True`` and *no* dist does not raise -- the
   old hard guard contradicted the FSDP pass's own ``world_size==1`` no-op.
3. ``optimizer_step`` grad-clip path.
4. ``train`` loop bookkeeping: ``log_interval`` printing, ``max_steps``,
   ``log_fn`` callback, and non-iterator iterables.
5. ``to()`` (device move).

The trainer delegates compilation / execution to ``GraphCompiler``
(``compile.compiler``); the compiler surface (including ``_init_device_mesh``
both branches) is covered by ``test_compiler.py``.

Tracing uses a tiny ``nn.Linear`` model and the same joint-graph capture the
tracer tests validate; only the trainer wiring is asserted here.
"""

import io
import logging
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


import torch
from torch import nn

from hyper_parallel.compile.pass_config import PassConfig
from hyper_parallel.compile.trainer import GraphTrainer


def _make_model() -> nn.Linear:
    """A tiny CPU-linear model; tracing is cheap and deterministic."""
    return nn.Linear(4, 4)


def _mse_train_fn(model, *, x, y) -> torch.Tensor:
    """Training function: mean-squared error between prediction and target."""
    return ((model(x) - y) ** 2).mean()


def _batches(n=2, dim=2):
    """Yield ``n`` model-input dicts (the kwargs ``train_fn`` consumes)."""
    x = torch.randn(dim, 4)
    y = torch.randn(dim, 4)
    for _ in range(n):
        yield {"x": x, "y": y}


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

        loss = tr.train_step(x=torch.randn(2, 4), y=torch.randn(2, 4))

        self.assertIsNotNone(
            tr._compiler._joint_graph, "compile should populate the joint graph"
        )
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
        before = model.weight.detach().clone()  # pylint: disable=not-callable

        tr.compile(x=torch.randn(2, 4), y=torch.randn(2, 4))
        tr.train_step(x=torch.randn(2, 4), y=torch.randn(2, 4))
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
        tr.train_step(x=torch.randn(2, 4), y=torch.randn(2, 4))

        # A very large loss guarantees an un-clipped gradient of norm > 1.
        with torch.no_grad():
            model.weight.mul_(10.0)
        # Reset the graph so the next step recomputes a huge loss.
        tr._compiler._joint_graph = None
        tr.train_step(x=torch.randn(2, 4), y=torch.randn(2, 4))
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
        before = tr._compiler.model.weight.detach().clone()
        # A list is a non-iterator (re-iterable) iterable: it must be accepted
        # just like a generator, but it can be iterated more than once. Passing
        # a generator here would leave the reiterable regression path uncovered.
        losses = tr.train(list(_batches(2)))
        self.assertEqual(len(losses), 2)
        # The loop advances the optimizer each step, so weights move.
        self.assertFalse(torch.equal(before, tr._compiler.model.weight))

    def test_train_log_interval_prints_on_rank0(self):
        """Test ``train`` logs a loss line on the log_interval (rank 0)."""
        tr = GraphTrainer(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        with self.assertLogs(
            "hyper_parallel.compile.trainer", level=logging.INFO
        ) as captured:
            tr.train(_batches(4), log_interval=2)

        out = "\n".join(captured.output)
        # Steps 2 and 4 are logged (rank 0 and step % log_interval == 0).
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


class TestGraphTrainerHelpers(unittest.TestCase):
    """``to`` and device placement."""

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
        self.assertEqual(tr._compiler.device, torch.device("cpu"))

    def test_place_on_device_moves_tensors_only(self):
        """Test ``_place_on_device`` moves tensors and leaves other values."""
        tr = GraphTrainer(
            model=_make_model(),
            train_fn=_mse_train_fn,
            pass_config=PassConfig(fsdp_enabled=False),
            device=torch.device("cpu"),
        )
        t = torch.randn(2, 4)
        result = tr._place_on_device({"x": t, "y": "not-a-tensor"})
        self.assertIsInstance(result["x"], torch.Tensor)
        self.assertEqual(result["y"], "not-a-tensor")


if __name__ == "__main__":
    unittest.main()
