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
"""Unit tests for the VeRL checkpoint adapter's state handoff."""

from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import load_file

if find_spec("verl") is not None:
    from hyper_parallel.integration.verl.engine import checkpoint
else:
    checkpoint = None


@unittest.skipUnless(checkpoint is not None, "requires VeRL")
class TestCheckpointAdapter(unittest.TestCase):
    """Exercise save and restore with CPU state and a mocked DCP backend."""

    def setUp(self) -> None:
        """Build a small initialized model and optimizer without a process group."""
        self.model = torch.nn.Linear(2, 1)
        self.model.config = Mock()
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        self.model(torch.ones(1, 2)).sum().backward()
        self.optimizer.step()
        self.scheduler.step()
        self.trainer = SimpleNamespace(
            model=self.model,
            model_integration=Mock(),
        )
        self.dcp = Mock()
        self.processing = Mock()
        with (
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch.object(checkpoint, "build_checkpointer", return_value=self.dcp),
        ):
            self.manager = checkpoint.VeRLCheckpointAdapter(
                self.trainer, model=self.model, optimizer=self.optimizer,
                lr_scheduler=self.scheduler, processing_class=self.processing,
            )

    def test_save_and_load_restores_training_state(self) -> None:
        """Check parameters, Adam moments, scheduler and RNG."""
        captured = {}

        def save_state(path, state, **options):
            self.assertEqual(options, {"global_step": 7, "save_async": False})
            self.assertEqual(set(state["extra_state"]), {"lr_scheduler", "rng_state"})
            captured.update(deepcopy(state))
            checkpoint.filesystem_storage.save_checkpoint_file({"weight": state["model"]["weight"]}, path + "/weights")

        def load_state(_path, state, *, extra_state_skeleton):
            self.assertEqual(set(extra_state_skeleton), {"lr_scheduler", "rng_state"})
            state.update(captured)
            # Older per-rank files may still contain training counters.
            state["extra_state"] = {
                **captured["extra_state"],
                "global_step": 4,
                "epoch": 2,
                "verl_global_step": 7,
            }

        self.dcp.save.side_effect = save_state
        self.dcp.load.side_effect = load_state
        with TemporaryDirectory() as directory, patch.object(
            self.manager, "get_rng_state", return_value="saved-rng",
        ), patch.object(self.manager, "load_rng_state") as restore_rng:
            self.manager.save_checkpoint(directory, global_step=7)
            self.model.config.save_pretrained.assert_called_once_with(Path(directory) / "huggingface")
            self.processing.save_pretrained.assert_called_once_with(Path(directory) / "huggingface")
            torch.testing.assert_close(load_file(directory + "/weights")["weight"], self.model.weight)
            with torch.no_grad():
                self.model.weight.zero_()
            self.optimizer.state[self.model.weight]["exp_avg"].zero_()
            self.scheduler.last_epoch = 0
            self.manager.load_checkpoint(directory)

        torch.testing.assert_close(self.model.state_dict(), captured["model"])
        torch.testing.assert_close(self.optimizer.state_dict(), captured["optimizer"])
        self.assertEqual(self.scheduler.state_dict(), captured["extra_state"]["lr_scheduler"])
        restore_rng.assert_called_once_with("saved-rng")

    def test_rejects_unsupported_checkpoint_operations(self) -> None:
        """Reject unsupported paths, formats and roles before interacting with DCP."""
        with TemporaryDirectory() as directory:
            cases = (
                (self.manager.save_checkpoint, {"hdfs_path": "remote"}, NotImplementedError, "remote paths"),
                (self.manager.save_checkpoint, {"max_ckpt_to_keep": 2}, NotImplementedError, "retention"),
                (self.manager.load_checkpoint, {"del_local_after_load": True}, NotImplementedError, "deletion"),
            )
            for method, options, error, message in cases:
                with self.subTest(options=options), self.assertRaisesRegex(error, message):
                    method(directory, **options)

            with patch.object(checkpoint, "initialize_optimizer_state", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "initialize optimizer state"):
                    self.manager.load_checkpoint(directory)

            (Path(directory) / ".metadata").touch()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                self.manager.save_checkpoint(directory)

            (Path(directory) / "hp_checkpoint.json").touch()
            with self.assertRaisesRegex(ValueError, "Legacy"):
                self.manager.load_checkpoint(directory)

        for method in (self.manager.save_checkpoint, self.manager.load_checkpoint):
            method(None)
            with patch.object(self.manager, "optimizer", None), self.assertRaisesRegex(
                NotImplementedError, "trainable actor",
            ):
                method("unused")
        self.dcp.save.assert_not_called()
        self.dcp.load.assert_not_called()

    def test_rejects_external_parameter_provider(self) -> None:
        """Detect an incompatible model before building the checkpointer."""
        with patch.object(checkpoint, "get_model_external_state", return_value=object()), self.assertRaisesRegex(
            NotImplementedError, "external parameter providers",
        ):
            checkpoint.VeRLCheckpointAdapter(self.trainer)
