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

from tests.common.mark_utils import arg_mark


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

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
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

        base = SimpleNamespace(
            get_batch=Mock(return_value=(model_inputs, loss_inputs)),
            on_micro_step_begin=Mock(side_effect=on_micro_step_begin),
            forward_backward_step=Mock(side_effect=forward_backward_step),
        )
        trainer = TextTrainer.__new__(TextTrainer)
        trainer.base = base

        loss, loss_dict = trainer.forward_backward_step(iter(()), num_micro_steps=1)

        self.assertEqual(events, ["hook", "forward"], f"expected streaming hook order, got={events}")
        self.assertEqual(float(loss), 1.0, f"expected loss 1.0, got={float(loss)}")
        self.assertEqual(
            float(loss_dict["foundation_loss"]),
            1.0,
            f"expected foundation loss 1.0, got={float(loss_dict['foundation_loss'])}",
        )


if __name__ == "__main__":
    unittest.main()
