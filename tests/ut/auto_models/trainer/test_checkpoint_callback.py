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
"""Unit tests for AutoModels checkpoint restore orchestration."""
# pylint: disable=wrong-import-position

import os
import tempfile
import types
import unittest
from typing import Any
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch  # pylint: disable=wrong-import-position
from torch import nn  # pylint: disable=wrong-import-position

from hyper_parallel.auto_models.trainer.callbacks.checkpoint_callback import (
    CheckpointerCallback,
)
from hyper_parallel.auto_models.trainer.callbacks.base import TrainerState
from tests.common.mark_utils import arg_mark


class _CheckpointModel(nn.Module):  # pylint: disable=abstract-method
    """Model with floating and integer checkpoint state."""

    def __init__(self) -> None:
        """Register a low-precision parameter and persistent buffers."""
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
        self.register_buffer("scale", torch.zeros(2, dtype=torch.bfloat16))
        self.register_buffer("indices", torch.tensor([1, 2], dtype=torch.int64))


class _ModelOnlyCheckpointer:
    """Fill a destination model skeleton with checkpoint values."""

    def load(
            self,
            path: str,
            state: dict[str, Any],
            **kwargs: Any,
    ) -> dict[str, Any]:
        """Copy low-precision checkpoint values into the destination state."""
        del path, kwargs
        state["model"]["weight"].copy_(
            torch.tensor([1.5, 2.5], dtype=torch.bfloat16)
        )
        state["model"]["scale"].copy_(
            torch.tensor([3.5, 4.5], dtype=torch.bfloat16)
        )
        return state

    def maybe_wait_for_async_save(self) -> None:
        """Satisfy the callback's asynchronous-save interface."""


class TestCheckpointModelPrecision(unittest.TestCase):
    """Verify checkpoint restore applies the configured model initialization dtype."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_checkpoint_load_casts_model_to_float32(self):
        """Cast loaded low-precision model state to float32.

        Feature: Trainer-owned checkpoint precision conversion.
        Description: Restore bfloat16 parameters and buffers with float32 configured.
        Expectation: Loaded floating state is float32 and integer state is unchanged.
        """
        model = _CheckpointModel()
        with tempfile.TemporaryDirectory() as checkpoint_path:
            checkpoint_config = types.SimpleNamespace(
                save_ckpt=False,
                checkpoint_dir=checkpoint_path,
                save_steps=0,
                save_epochs=0,
                is_async=False,
                is_peft=False,
                save_optimizer=False,
                save_train_state=False,
                save_extra_state_per_rank=True,
                restore_from=checkpoint_path,
                restore_optimizer=False,
                restore_train_state=False,
            )
            trainer = types.SimpleNamespace(
                config=types.SimpleNamespace(
                    checkpoint=checkpoint_config,
                    model_init_dtype="float32",
                ),
                model=model,
                mesh=None,
                optimizer=None,
                lr_scheduler=None,
                train_dataloader=None,
                data_iterator=None,
                state=TrainerState(),
                start_epoch=0,
                start_step=0,
            )
            with patch(
                    "hyper_parallel.auto_models.trainer.callbacks.checkpoint_callback.build_checkpointer",
                    return_value=_ModelOnlyCheckpointer(),
            ), patch(
                    "hyper_parallel.auto_models.trainer.callbacks.checkpoint_callback.helper.empty_cache",
            ):
                callback = CheckpointerCallback(trainer)
                callback.on_train_begin(trainer.state)

        self.assertEqual(model.weight.dtype, torch.float32)
        self.assertEqual(model.scale.dtype, torch.float32)
        self.assertEqual(model.indices.dtype, torch.int64)
        torch.testing.assert_close(model.weight, torch.tensor([1.5, 2.5]))
        torch.testing.assert_close(model.scale, torch.tensor([3.5, 4.5]))


if __name__ == "__main__":
    unittest.main()
