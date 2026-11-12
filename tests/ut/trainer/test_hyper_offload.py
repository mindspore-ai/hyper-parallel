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
"""Native Trainer activation-offload configuration and microstep lifecycle."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config.hyper_offload import HyperOffloadConfig
from hyper_parallel.trainer.config.trainer import TrainerConfig


class TestHyperOffloadConfig(unittest.TestCase):
    """Reject invalid budgets and incompatible execution paths before training."""

    def test_defaults_leave_offload_disabled(self) -> None:
        """Existing native configurations do not allocate an offload session."""
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.config = SimpleNamespace(
            accelerator=SimpleNamespace(loss_parallel=False),
            hyper_offload=HyperOffloadConfig(),
        )
        with patch("hyper_parallel.trainer.base.OffloadSession") as session:
            trainer._build_training_context()
        session.assert_not_called()
        self.assertIsInstance(trainer.hyper_offload_context, nullcontext)

    def test_invalid_budgets(self) -> None:
        """Both capacity limits accept positive integer MiB only."""
        for field in ("max_resident_activation_mb", "max_offload_activation_mb"):
            for invalid in (-1, 0, True, 1.5, "1024"):
                with (
                    self.subTest(field=field, invalid=invalid),
                    self.assertRaisesRegex(ValueError, "positive integer"),
                ):
                    HyperOffloadConfig(**{field: invalid})
        with self.assertRaisesRegex(TypeError, "must be a bool"):
            HyperOffloadConfig(enable="true")

    def test_incompatible_execution_rejected(self) -> None:
        """Pipeline, compilation and activation swapping cannot own the same graph."""
        config = TrainerConfig(model=Mock(), optimizer=SimpleNamespace(fp32_main_params=False))
        self.assertFalse(config.hyper_offload.enable)
        config.hyper_offload.enable = True
        for obj, name, value in (
            (config.accelerator, "pp_size", 2),
            (config.compile, "enabled", True),
            (config, "activation_swap", "attention"),
            (config.activation_checkpoint, "swap_inputs", True),
        ):
            original = getattr(obj, name)
            with self.subTest(field=name):
                setattr(obj, name, value)
                with self.assertRaisesRegex(ValueError, "hyper_offload"):
                    config.__post_init__()
                setattr(obj, name, original)
        config.__post_init__()

    def test_budgets_forwarded_to_core(self) -> None:
        """The native entry builds one session with the requested capacities."""
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.config = SimpleNamespace(
            accelerator=SimpleNamespace(loss_parallel=False),
            hyper_offload=HyperOffloadConfig(True, 4096, 8192),
        )
        with patch("hyper_parallel.trainer.base.OffloadSession") as session:
            trainer._build_training_context()
        runtime = session.call_args.args[0]
        self.assertEqual(runtime.max_resident_activation_mb, 4096)
        self.assertEqual(runtime.max_offload_activation_mb, 8192)
        self.assertIsNone(runtime.planner)
        self.assertIs(trainer.hyper_offload_context, session.return_value)


class TestHyperOffloadMicrostep(unittest.TestCase):
    """Exercise the real Trainer forward/backward context, including cleanup."""

    def test_context_covers_forward_and_backward_and_is_reused(self) -> None:
        """Successive microsteps reuse a session and preserve the original gradient."""
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.model_fwd_context = nullcontext()
        trainer.model_bwd_context = nullcontext()
        trainer.graph_compiler = None
        trainer.model_integration = Mock()
        trainer.train_dataloader = SimpleNamespace()
        trainer.config = SimpleNamespace(training=SimpleNamespace(empty_cache_before_backward=False))
        trainer.current_token_counts = {}
        trainer.step_token_counts = {}
        trainer.preforward = lambda batch: batch
        trainer.loss_fn = SimpleNamespace()
        trainer.postforward = lambda outputs, *args, **kwargs: (outputs.sum(), {})
        parameter = torch.tensor(3.0, requires_grad=True)
        events = []
        active = []
        context = Mock()
        context.__enter__ = Mock(side_effect=lambda: active.append(True))
        context.__exit__ = Mock(side_effect=lambda *args: active.pop() and False)
        trainer.hyper_offload_context = context

        def forward(input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
            """Check context ownership during model computation."""
            self.assertTrue(active)
            self.assertFalse(use_cache)
            events.append("forward")
            return parameter.square() * input_ids

        def backward(gradient: torch.Tensor) -> torch.Tensor:
            """Check context ownership during autograd execution."""
            self.assertTrue(active)
            events.append("backward")
            return gradient

        trainer.model = forward
        parameter.register_hook(backward)
        for _ in range(2):
            trainer.forward_backward_step({"input_ids": torch.tensor(2.0)}, {})
            self.assertFalse(active)
        self.assertEqual(events, ["forward", "backward", "forward", "backward"])
        torch.testing.assert_close(parameter.grad, torch.tensor(24.0))
        self.assertEqual(context.__enter__.call_count, 2)
        trainer.model = Mock(side_effect=RuntimeError("forward failed"))
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            trainer.forward_backward_step({"input_ids": torch.tensor(2.0)}, {})
        self.assertFalse(active)
        self.assertIs(context.__exit__.call_args.args[0], RuntimeError)
