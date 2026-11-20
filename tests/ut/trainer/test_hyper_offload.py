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

from hyper_parallel.auto_parallel.hyper_offload import OffloadConfig, OffloadSession
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


class TestHyperOffloadTrainingLifecycle(unittest.TestCase):
    """Training completion releases the session without waiting for Trainer GC."""

    def setUp(self) -> None:
        """Build an empty training loop with a real, inactive offload session."""
        self.events = []
        trainer = BaseTrainer.__new__(BaseTrainer)
        self.trainer = trainer
        trainer.config = SimpleNamespace(dataloader=SimpleNamespace(use_background_prefetcher=False))
        trainer.local_rank = 0
        trainer.state = SimpleNamespace(global_step=0, epoch=0)
        trainer.train_iters = 0
        trainer.train_epochs = 0
        trainer.train_dataloader = Mock()
        trainer.on_train_begin = Mock()
        trainer.on_train_end = Mock(side_effect=lambda: self.events.append("train_end"))
        trainer.destroy_distributed = Mock(side_effect=lambda: self.events.append("destroy"))
        trainer.hyper_offload_context = OffloadSession(OffloadConfig())
        trainer.hyper_offload_context.close = Mock(side_effect=lambda: self.events.append("close"))
        for name, options in (
            ("memory_profiler", {}),
            ("HyperIter", {}),
            ("synchronize", {"side_effect": lambda: self.events.append("synchronize")}),
        ):
            patcher = patch(f"hyper_parallel.trainer.base.{name}", **options)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_success_closes_session_before_distributed_teardown(self) -> None:
        """Keep the policy during training, then release it before process cleanup."""
        self.trainer.train()
        self.assertEqual(self.events, ["train_end", "close", "synchronize", "destroy"])
        self.trainer.hyper_offload_context.close.assert_called_once_with()

    def test_training_failure_closes_session_and_preserves_the_error(self) -> None:
        """An interrupted run must release its lease even while Trainer remains alive."""
        error = RuntimeError("training failed")
        self.trainer.on_train_begin.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            self.trainer.train()
        self.assertIs(raised.exception, error)
        self.trainer.hyper_offload_context.close.assert_called_once_with()
        self.trainer.on_train_end.assert_not_called()
        self.trainer.destroy_distributed.assert_not_called()

    def test_cleanup_failure_does_not_mask_the_training_error(self) -> None:
        """Report an allocator restoration failure while preserving the causal error."""
        self.trainer.on_train_begin.side_effect = ValueError("training failed")
        self.trainer.hyper_offload_context.close.side_effect = RuntimeError("restore failed")
        with self.assertLogs("hyper_parallel.trainer.base", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "training failed"):
                self.trainer.train()

    def test_cleanup_failure_after_success_is_propagated(self) -> None:
        """Do not report a successful cleanup when native allocator restoration failed."""
        self.trainer.hyper_offload_context.close.side_effect = RuntimeError("restore failed")
        with self.assertRaisesRegex(RuntimeError, "restore failed"):
            self.trainer.train()
