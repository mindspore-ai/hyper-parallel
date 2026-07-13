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
"""Unit tests for BaseTrainer-owned checkpoint lifecycle."""
# pylint: disable=protected-access
import json
import os
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch  # pylint: disable=wrong-import-position
from torch import nn  # pylint: disable=wrong-import-position

from hyper_parallel.trainer import base as trainer_base  # pylint: disable=wrong-import-position
from hyper_parallel.trainer.base import BaseTrainer, TrainerState  # pylint: disable=wrong-import-position


class _RecordingDataloader:
    """Minimal stateful dataloader stand-in."""

    def __init__(self, position: int = 0) -> None:
        """Initialize dataloader stand-in at the given position."""
        self.position = position

    def state_dict(self) -> dict:
        """Return serializable dataloader position."""
        return {"position": self.position}

    def load_state_dict(self, state: dict) -> None:
        """Restore dataloader position."""
        self.position = state["position"]


def _make_args(tmp_dir, *, save_steps=2, save_async=False, load_path=None):
    """Build args with checkpoint config."""
    checkpoint = types.SimpleNamespace(
        output_dir=tmp_dir,
        save_steps=save_steps,
        save_async=save_async,
        load_path=load_path,
        save_hf_weights=False,
    )
    return types.SimpleNamespace(
        model=types.SimpleNamespace(name="toy"),
        checkpoint=checkpoint,
        train=types.SimpleNamespace(
            max_steps=10,
            num_train_epochs=1,
            checkpoint=checkpoint,
        ),
    )


def _build_trainer(tmp_dir, *, save_steps=2, save_async=False, load_path=None, step=0):
    """Build a BaseTrainer with checkpoint-relevant runtime fields."""
    args = _make_args(
        tmp_dir,
        save_steps=save_steps,
        save_async=save_async,
        load_path=load_path,
    )
    with patch.object(trainer_base, "get_spec", return_value=types.SimpleNamespace(state_dict_adapter=None)):
        trainer = BaseTrainer(args, setup=False)
    trainer.model = nn.Linear(4, 4)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    trainer.train_dataloader = _RecordingDataloader(position=step * 4)
    trainer.on_save = MagicMock()
    trainer.on_resume = MagicMock()
    return trainer


def _fake_dcp():
    """Return dcp save/load fakes that round-trip tensors through JSON."""
    def _save(sd, checkpoint_id, use_collectives=False):  # pylint: disable=unused-argument
        serializable = {key: value.detach().cpu().tolist() for key, value in sd.items()}
        with open(os.path.join(checkpoint_id, "_fake_model_sd.json"), "w", encoding="utf-8") as file:
            json.dump(serializable, file)

    def _load(sd, checkpoint_id, use_collectives=False):  # pylint: disable=unused-argument
        with open(os.path.join(checkpoint_id, "_fake_model_sd.json"), encoding="utf-8") as file:
            payload = json.load(file)
        for key in list(sd.keys()):
            if key in payload:
                sd[key] = torch.tensor(payload[key])

    return _save, _load


class TestCheckpointGating(unittest.TestCase):
    """Save interval and final-save behavior."""

    def test_maybe_save_checkpoint_disabled_when_save_steps_zero(self):
        """save_steps=0 never saves."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _build_trainer(tmp_dir, save_steps=0)
            trainer._dispatch_checkpoint_save = MagicMock()
            for step in range(1, 10):
                trainer.state.global_step = step
                trainer._maybe_save_checkpoint()
            trainer._dispatch_checkpoint_save.assert_not_called()

    def test_maybe_save_checkpoint_fires_on_multiples_and_dedups(self):
        """save_steps=3 fires on multiples once per step."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _build_trainer(tmp_dir, save_steps=3)

            def _bookkeep_save(state):
                trainer._last_saved_step = state.global_step

            trainer._dispatch_checkpoint_save = MagicMock(side_effect=_bookkeep_save)
            fired_at = []
            for step in range(1, 11):
                trainer.state.global_step = step
                trainer._maybe_save_checkpoint()
                if trainer._dispatch_checkpoint_save.call_count > len(fired_at):
                    fired_at.append(step)
                trainer._maybe_save_checkpoint()
            self.assertEqual(fired_at, [3, 6, 9])
            self.assertEqual(trainer._dispatch_checkpoint_save.call_count, 3)

    def test_save_final_checkpoint_force_saves_when_not_already_saved(self):
        """Final checkpoint saves current step when it was not just saved."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _build_trainer(tmp_dir, save_steps=5)
            trainer._save_checkpoint = MagicMock()
            trainer.state.global_step = 7
            trainer._save_final_checkpoint()
            trainer._save_checkpoint.assert_called_once()

    def test_save_final_checkpoint_skips_when_already_saved(self):
        """Final checkpoint does not double-save the same step."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _build_trainer(tmp_dir, save_steps=5)
            trainer._save_checkpoint = MagicMock()
            trainer.state.global_step = 10
            trainer._last_saved_step = 10
            trainer._save_final_checkpoint()
            trainer._save_checkpoint.assert_not_called()


class TestCheckpointRoundTrip(unittest.TestCase):
    """Save/load bucket orchestration."""

    def test_round_trip_restores_all_buckets(self):
        """Save state, mutate trainer, then restore model/state/RNG/dataloader."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _build_trainer(tmp_dir, save_steps=1, step=5)
            save_fn, load_fn = _fake_dcp()
            with patch.object(trainer_base, "dcp_save", save_fn), \
                 patch.object(trainer_base, "dcp_load", load_fn), \
                 patch.object(trainer_base, "platform") as mock_platform:
                mock_platform.get_rank.return_value = 0
                mock_platform.get_rng_state.return_value = torch.tensor([1, 2, 3])
                set_rng_calls = []
                mock_platform.set_rng_state.side_effect = set_rng_calls.append

                with torch.no_grad():
                    trainer.model.weight.fill_(0.42)
                original_weight = trainer.model.weight.detach().clone()

                state = TrainerState(max_steps=20)
                state.global_step = 7
                state.epoch = 1
                state.consumed_tokens = 11
                state.consumed_samples = 22
                trainer._save_checkpoint(state)

                save_dir = os.path.join(tmp_dir, "step_7")
                for file_name in (
                    "_fake_model_sd.json",
                    "optimizer_rank0.pt",
                    "scheduler.pt",
                    "extra_state.json",
                    "rng_rank0.pt",
                    "dataloader_rank0.pt",
                ):
                    self.assertTrue(os.path.isfile(os.path.join(save_dir, file_name)), file_name)

                with torch.no_grad():
                    trainer.model.weight.fill_(0.0)
                trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
                trainer.train_dataloader.position = 0

                trainer._resume_from_checkpoint(save_dir)

                self.assertEqual(trainer.state.global_step, 7)
                self.assertEqual(trainer.state.epoch, 1)
                self.assertEqual(trainer.state.consumed_tokens, 11)
                self.assertEqual(trainer.state.consumed_samples, 22)
                self.assertTrue(torch.allclose(trainer.model.weight, original_weight, atol=1e-6))
                self.assertEqual(trainer.train_dataloader.position, 20)
                self.assertEqual(len(set_rng_calls), 1)
                self.assertTrue(torch.equal(set_rng_calls[0], torch.tensor([1, 2, 3])))
                trainer.on_resume.assert_called_once()

    def test_missing_load_path_is_silent_no_crash(self):
        """Missing checkpoint path warns and returns."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _build_trainer(tmp_dir)
            trainer._resume_from_checkpoint("/no/such/dir")
            trainer.on_resume.assert_not_called()

    def test_none_load_path_short_circuits(self):
        """No load_path leaves model untouched."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _build_trainer(tmp_dir, load_path=None)
            trainer.model.state_dict = MagicMock(side_effect=AssertionError("must not load"))
            trainer._resume_from_checkpoint_if_needed()
            trainer.model.state_dict.assert_not_called()


class TestAsyncCheckpointSave(unittest.TestCase):
    """Async checkpoint dispatch serializes save workers."""

    def test_dispatch_runs_save_in_background_thread(self):
        """Async save runs off caller thread."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _build_trainer(tmp_dir, save_steps=1, save_async=True)
            saved_in = {}
            barrier = threading.Event()

            def _slow_save(state):  # pylint: disable=unused-argument
                saved_in["tid"] = threading.get_ident()
                barrier.wait(timeout=2.0)

            trainer._save_checkpoint = _slow_save
            trainer.state.global_step = 1
            caller_tid = threading.get_ident()
            trainer._dispatch_checkpoint_save(trainer.state)
            self.assertIsNotNone(trainer._checkpoint_save_thread)
            self.assertTrue(trainer._checkpoint_save_thread.is_alive())
            barrier.set()
            trainer._checkpoint_save_thread.join(timeout=2.0)
            self.assertNotEqual(saved_in["tid"], caller_tid)

    def test_second_dispatch_waits_for_first(self):
        """Back-to-back async saves are serialized."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = _build_trainer(tmp_dir, save_steps=1, save_async=True)
            order = []
            release_first = threading.Event()

            def _save_a(state):  # pylint: disable=unused-argument
                order.append(("a_start", time.time()))
                release_first.wait(timeout=2.0)
                order.append(("a_end", time.time()))

            def _save_b(state):  # pylint: disable=unused-argument
                order.append(("b_start", time.time()))
                order.append(("b_end", time.time()))

            trainer._save_checkpoint = _save_a
            trainer.state.global_step = 1
            trainer._dispatch_checkpoint_save(trainer.state)
            trainer._save_checkpoint = _save_b
            trainer.state.global_step = 2

            def _release():
                time.sleep(0.1)
                release_first.set()

            threading.Thread(target=_release, daemon=True).start()
            trainer._dispatch_checkpoint_save(trainer.state)
            trainer._checkpoint_save_thread.join(timeout=2.0)
            self.assertEqual(
                [item[0] for item in order],
                ["a_start", "a_end", "b_start", "b_end"],
            )


if __name__ == "__main__":
    unittest.main()
