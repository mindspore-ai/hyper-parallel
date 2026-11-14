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
"""Unit tests for the text Trainer step lifecycle."""

import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.trainer.config.data import ModelAssetsConfig
from hyper_parallel.trainer.config.training import DebugConfig
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.components.losses.model_output import ModelOutputLoss
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token


def _load_text_trainer() -> type:
    """Load TextTrainer without requiring the unrelated stateful dataloader dependency."""
    torchdata_module = ModuleType("torchdata")
    stateful_dataloader_module = ModuleType("torchdata.stateful_dataloader")
    stateful_dataloader_module.StatefulDataLoader = torch.utils.data.DataLoader
    with patch.dict(
            sys.modules,
            {
                "torchdata": torchdata_module,
                "torchdata.stateful_dataloader": stateful_dataloader_module,
            },
    ):
        trainer_module = importlib.import_module("hyper_parallel.trainer.text_trainer")
    return trainer_module.TextTrainer


TextTrainer = _load_text_trainer()


class TestTextTrainer(unittest.TestCase):
    """Verify text training streams prepared batches into callbacks."""

    def test_default_loss_uses_aligned_trainer_token_count(self) -> None:
        """Feature: Packed supervision and the default loss use the same token count.
        Description: Exercise first-token-only, unshifted and empty local targets.
        Expectation: Postforward preserves valid CE and makes empty shards a connected zero.
        """
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.loss_fn = ModelOutputLoss()
        trainer.mesh = SimpleNamespace(dp_cp_mesh=None, sequence_parallel=False, dp_size=1, cp_size=1)
        labels = torch.tensor([[2, -100, -100, -100]])
        for shifted, empty in ((True, False), (False, False), (True, True)):
            with self.subTest(shifted=shifted, empty=empty):
                targets = torch.full_like(labels, -100) if empty else labels
                loss_inputs = {"labels": targets}
                if shifted:
                    loss_inputs["shift_labels"] = targets
                aligned = targets if shifted else torch.nn.functional.pad(targets[..., 1:], (0, 1), value=-100)
                logits = torch.randn(1, 4, 8, requires_grad=True)
                raw_loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), aligned.flatten())
                counts = count_loss_token(loss_inputs)
                loss, _ = trainer.postforward(SimpleNamespace(loss=raw_loss), targets, current_token_counts=counts)
                loss.backward()
                if shifted and not empty:
                    torch.testing.assert_close(loss, raw_loss)
                    self.assertGreater(logits.grad.abs().sum().item(), 0)
                else:
                    torch.testing.assert_close(loss, torch.zeros_like(loss))
                    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))

    def test_template_logging_is_independent_of_dataset_debug(self) -> None:
        """Feature: Independent template debug configuration.

        Description: Build text assets with dataset debugging disabled.
        Expectation: The debug template flag reaches the template for both values.
        """
        self.assertFalse(DebugConfig().log_first_rendered_template)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                tokenizer = Mock(pretrained_model_name_or_path="/unused")
                assets = ModelAssetsConfig(chat_template="tokenizer", tokenizer=tokenizer)
                config = SimpleNamespace(
                    dataset=SimpleNamespace(model_assets=assets),
                    debug=DebugConfig(check_dataset=None, log_first_rendered_template=enabled),
                )
                trainer = TextTrainer.__new__(TextTrainer)
                trainer.base = SimpleNamespace(config=config, model_config=object())
                build = Mock()
                with patch.dict(TextTrainer._build_model_assets.__globals__, build_chat_template=build):
                    trainer._build_model_assets()
                build.assert_called_once_with(
                    "tokenizer", tokenizer.build.return_value,
                    chat_template_kwargs={}, log_first_rendered_template=enabled,
                )

    def test_forward_backward_dispatches_metric_hook_before_forward(self) -> None:
        """Send detached metric inputs before executing the model forward.

        Feature: Text Trainer micro-step callbacks.
        Description: Fetch one prepared batch and execute one forward-backward step.
        Expectation: Metric callback receives token metadata immediately before model execution.
        """
        events = []
        labels = torch.tensor([[1, 2, 3, 4], [5, 6, 7, -100]])
        model_inputs = {"input_ids": torch.ones(2, 4)}
        loss_inputs = {"labels": labels, "shift_labels": labels}

        def on_micro_step_begin(metric_inputs: dict[str, object]) -> None:
            """Record and validate the metric callback payload."""
            events.append("hook")
            token_count = int(metric_inputs["token_count"])
            self.assertEqual(token_count, 7, f"expected seven valid tokens, got={token_count}")

        def forward_backward_step(
                actual_model_inputs: dict[str, torch.Tensor],
                actual_loss_inputs: dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """Record model execution and validate forwarded batch identity."""
            events.append("forward")
            self.assertIs(
                actual_model_inputs,
                model_inputs,
                f"model input identity changed, got={actual_model_inputs!r}",
            )
            self.assertIs(
                actual_loss_inputs,
                loss_inputs,
                f"loss input identity changed, got={actual_loss_inputs!r}",
            )
            return torch.tensor(1.0), {"foundation_loss": torch.tensor(1.0)}

        def on_micro_step_end(metrics: dict[str, torch.Tensor]) -> None:
            """Record completion and verify the returned payload is forwarded unchanged."""
            events.append("end")
            self.assertEqual(float(metrics["foundation_loss"]), 1.0)

        base = SimpleNamespace(
            get_batch=Mock(return_value=(model_inputs, loss_inputs)),
            on_micro_step_begin=Mock(side_effect=on_micro_step_begin),
            on_micro_step_end=Mock(side_effect=on_micro_step_end),
            forward_backward_step=Mock(side_effect=forward_backward_step),
        )
        trainer = TextTrainer.__new__(TextTrainer)
        trainer.base = base

        loss, loss_dict = trainer.forward_backward_step(iter(()))

        self.assertEqual(events, ["hook", "forward", "end"], f"expected lifecycle hook order, got={events}")
        base.on_micro_step_end.assert_called_once_with(loss_dict)
        self.assertEqual(float(loss), 1.0, f"expected loss 1.0, got={float(loss)}")
        self.assertEqual(
            float(loss_dict["foundation_loss"]),
            1.0,
            f"expected foundation loss 1.0, got={float(loss_dict['foundation_loss'])}",
        )

    def test_base_micro_step_end_dispatches_state_payload_and_context(self) -> None:
        """The shared dispatcher forwards one unchanged payload to every registered callback."""
        base = BaseTrainer.__new__(BaseTrainer)
        base.state = object()
        callbacks = [Mock(), Mock()]
        base._callbacks = callbacks
        metrics = {"router_aux_loss": torch.tensor(0.25)}
        base.on_micro_step_end(metrics, micro_step=3)
        for callback in callbacks:
            callback.on_micro_step_end.assert_called_once_with(base.state, loss_dict=metrics, micro_step=3)
            self.assertIs(callback.on_micro_step_end.call_args.kwargs["loss_dict"], metrics)

    def test_failed_forward_backward_does_not_dispatch_micro_step_end(self) -> None:
        """A failed microstep propagates its error without emitting a completion event."""
        base = SimpleNamespace(
            get_batch=Mock(return_value=({"input_ids": torch.ones(1, 3)},
                                        {"labels": torch.tensor([[1, 2, 3]])})),
            on_micro_step_begin=Mock(),
            forward_backward_step=Mock(side_effect=RuntimeError("microstep failed")),
            on_micro_step_end=Mock(),
        )
        trainer = TextTrainer.__new__(TextTrainer)
        trainer.base = base
        with self.assertRaisesRegex(RuntimeError, "microstep failed"):
            trainer.forward_backward_step(iter(()))
        base.on_micro_step_begin.assert_called_once()
        base.on_micro_step_end.assert_not_called()




if __name__ == "__main__":
    unittest.main()
