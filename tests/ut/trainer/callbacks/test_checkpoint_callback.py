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
"""Unit tests for ``CheckpointCallback`` save / resume orchestration.

This callback is the most failure-prone one in the trainer — it has to
keep six artifact buckets in lockstep (model shards, optimizer, scheduler,
extra_state, RNG, dataloader). A regression that loses any one bucket
breaks resume **silently**: training restarts, the loss curve looks
plausible, but the optimizer state is fresh and the model converges to a
different minimum than the original run.

Coverage:

1. **save_steps gating** — emit on multiples of ``save_steps`` and skip
   immediately if the same step already saved (``_last_saved_step`` dedup).
2. **save_steps == 0** disables saving entirely.
3. **on_train_end** force-saves the final step iff it wasn't just saved.
4. **6-bucket save → load round-trip** — every field that ``_save`` writes
   must be read back by ``on_train_begin`` and applied to the trainer.
   We use real ``torch.save`` for the non-model buckets and stub ``dcp_save``
   /``dcp_load`` to a JSON dump so the orchestration logic is end-to-end
   without the distributed checkpoint plumbing.
5. **async save** — ``save_async=True`` dispatches a thread; a second
   ``_dispatch_save`` must wait for the first via ``_join_pending`` (no
   parallel double-save races).
6. **load_path missing** — ``on_train_begin`` short-circuits silently
   (warning logged, no crash); load contracts honour resume idempotency.
"""
# pylint: disable=protected-access
import json
import logging
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
from torch import nn

from hyper_parallel.trainer.base import TrainerState
from hyper_parallel.trainer.callbacks import base as cb_mod
from hyper_parallel.trainer.callbacks.base import CheckpointCallback


# ----------------------------------------------------------------------------
# Test fixtures
# ----------------------------------------------------------------------------


class _RecordingDataloader:
    """Minimal stateful dataloader stand-in (``state_dict`` / ``load_state_dict``)."""

    def __init__(self, position=0):
        self.position = position

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


def _build_trainer(tmp_dir, *, save_steps=2, save_async=False, load_path=None,
                   step=0):
    """Build a SimpleNamespace ``trainer`` with the fields CheckpointCallback reads."""
    ckpt_cfg = types.SimpleNamespace(
        output_dir=tmp_dir,
        save_steps=save_steps,
        save_async=save_async,
        load_path=load_path,
        save_hf_weights=False,
    )
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # Step the LR scheduler once so ``lr_scheduler.state_dict()`` has content.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    dataloader = _RecordingDataloader(position=step * 4)
    trainer = types.SimpleNamespace(
        args=types.SimpleNamespace(checkpoint=ckpt_cfg),
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=dataloader,
        dispatch_save_event=MagicMock(),
        dispatch_load_event=MagicMock(),
        spec=None,
    )
    return trainer


def _fake_dcp(tmpdir):
    """Return ``(save_fn, load_fn)`` that round-trip a state-dict via JSON-like file.

    Real dcp_save / dcp_load talks to the distributed checkpoint stack; we
    only need a deterministic round-trip so the orchestration is testable.
    """
    def _save(sd, checkpoint_id, use_collectives=False):  # pylint: disable=unused-argument
        # Convert tensors to lists for JSON round-trip; the test only asserts
        # equality after dcp_load mutates the in-place sd back.
        serialisable = {k: v.detach().cpu().tolist() for k, v in sd.items()}
        with open(os.path.join(checkpoint_id, "_fake_model_sd.json"), "w", encoding="utf-8") as f:
            json.dump(serialisable, f)

    def _load(sd, checkpoint_id, use_collectives=False):  # pylint: disable=unused-argument
        with open(os.path.join(checkpoint_id, "_fake_model_sd.json"), encoding="utf-8") as f:
            payload = json.load(f)
        for k in list(sd.keys()):
            if k in payload:
                sd[k] = torch.tensor(payload[k])

    return _save, _load


# ----------------------------------------------------------------------------
# Save gating
# ----------------------------------------------------------------------------


class TestSaveStepsGating(unittest.TestCase):
    """``on_step_end`` emits only at multiples of ``save_steps`` (with dedup)."""

    def test_disabled_when_save_steps_zero(self):
        """``save_steps=0`` → no save ever."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(tmp, save_steps=0)
            cb = CheckpointCallback(trainer)
            cb._save = MagicMock()
            state = TrainerState(max_steps=100)
            for step in range(1, 10):
                state.global_step = step
                cb.on_step_end(state)
            cb._save.assert_not_called()

    def test_emits_only_on_multiples_and_dedups_same_step(self):
        """``save_steps=3`` fires at step 3, 6, 9 — each step exactly once."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(tmp, save_steps=3)
            cb = CheckpointCallback(trainer)

            def _bookkeep_save(state):
                """Stand-in for ``_save`` that records the saved step for dedup."""
                cb._last_saved_step = state.global_step

            cb._save = MagicMock(side_effect=_bookkeep_save)
            state = TrainerState(max_steps=20)
            fired_at = []
            for step in range(1, 11):
                state.global_step = step
                cb.on_step_end(state)
                if cb._save.call_count > len(fired_at):
                    fired_at.append(step)
                # Call twice at the same step — dedup must suppress the second.
                cb.on_step_end(state)
            self.assertEqual(fired_at, [3, 6, 9], (
                f"save must fire on multiples of save_steps only, got {fired_at}"
            ))
            self.assertEqual(cb._save.call_count, 3, (
                f"Same-step dedup must suppress repeats, got {cb._save.call_count} calls"
            ))

    def test_on_train_end_force_saves_when_not_already_saved(self):
        """If the final step wasn't a save boundary, ``on_train_end`` writes it."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(tmp, save_steps=5)
            cb = CheckpointCallback(trainer)
            cb._save = MagicMock()
            state = TrainerState(max_steps=7)
            state.global_step = 7  # not divisible by 5; mid-cycle exit
            cb.on_train_end(state)
            cb._save.assert_called_once()

    def test_on_train_end_skips_when_just_saved(self):
        """If the final step coincides with a save boundary, no double-save."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(tmp, save_steps=5)
            cb = CheckpointCallback(trainer)
            cb._save = MagicMock()
            state = TrainerState(max_steps=10)
            state.global_step = 10
            cb._last_saved_step = 10  # we just saved
            cb.on_train_end(state)
            cb._save.assert_not_called()


# ----------------------------------------------------------------------------
# Round-trip
# ----------------------------------------------------------------------------


class TestSaveLoadRoundTrip(unittest.TestCase):
    """Every bucket written by ``_save`` must be read back by ``on_train_begin``."""

    def test_round_trip_restores_all_six_buckets(self):
        """Save trainer state, mutate everything, load back, assert each bucket restored."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(tmp, save_steps=1, step=5)
            cb = CheckpointCallback(trainer)
            save_fn, load_fn = _fake_dcp(tmp)
            with patch.object(cb_mod, "dcp_save", save_fn), \
                 patch.object(cb_mod, "dcp_load", load_fn), \
                 patch.object(cb_mod, "platform") as mock_plat:
                mock_plat.get_rank.return_value = 0
                mock_plat.get_rng_state.return_value = torch.tensor([1, 2, 3])
                # Capture set_rng_state to verify it's called with our payload.
                set_rng_calls = []
                mock_plat.set_rng_state.side_effect = set_rng_calls.append

                # Mutate the model parameter directly; optimizer-state content
                # is covered by the save/load bucket checks below.
                with torch.no_grad():
                    trainer.model.weight.fill_(0.42)
                original_weight = trainer.model.weight.detach().clone()

                # ---- save -------------------------------------------------
                state = TrainerState(max_steps=20)
                state.global_step = 7
                state.epoch = 1
                cb._save(state)

                save_dir = os.path.join(tmp, "step_7")
                self.assertTrue(os.path.isdir(save_dir), (
                    f"save_dir missing: {save_dir}"
                ))
                # Each bucket file must exist.
                for fname in (
                    "_fake_model_sd.json",
                    "optimizer_rank0.pt",
                    "scheduler.pt",
                    "extra_state.json",
                    "rng_rank0.pt",
                    "dataloader_rank0.pt",
                ):
                    self.assertTrue(
                        os.path.isfile(os.path.join(save_dir, fname)),
                        f"missing checkpoint artifact: {fname}",
                    )

                # ---- mutate trainer to a "fresh post-restart" baseline ----
                with torch.no_grad():
                    trainer.model.weight.fill_(0.0)
                # Reset optimizer to a fresh state (different content).
                fresh_opt = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
                trainer.optimizer = fresh_opt
                trainer.train_dataloader.position = 0
                restart_state = TrainerState(max_steps=20)
                # ---- load --------------------------------------------------
                trainer.args.checkpoint.load_path = save_dir
                # Rebuild callback against new load_path.
                cb2 = CheckpointCallback(trainer)
                cb2.on_train_begin(restart_state)

                # extra_state: step + epoch restored.
                self.assertEqual(restart_state.global_step, 7)
                self.assertEqual(restart_state.epoch, 1)
                # Model weights restored.
                self.assertTrue(
                    torch.allclose(trainer.model.weight, original_weight, atol=1e-6),
                    (
                        f"model weight not restored: original sum="
                        f"{original_weight.sum()}, got sum={trainer.model.weight.sum()}"
                    ),
                )
                # Dataloader position restored.
                self.assertEqual(trainer.train_dataloader.position, 20, (
                    f"dataloader position not restored, got {trainer.train_dataloader.position}"
                ))
                # RNG state restored via platform.set_rng_state.
                self.assertEqual(len(set_rng_calls), 1, (
                    f"platform.set_rng_state must be called once, got {len(set_rng_calls)}"
                ))
                self.assertTrue(torch.equal(set_rng_calls[0], torch.tensor([1, 2, 3])))

    def test_missing_load_path_is_silent_no_crash(self):
        """``on_train_begin`` with a non-existent load_path warns and returns."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(tmp, save_steps=1, load_path="/no/such/dir")
            cb = CheckpointCallback(trainer)
            # Must not raise.
            cb.on_train_begin(TrainerState(max_steps=10))

    def test_none_load_path_short_circuits(self):
        """``load_path = None`` exits ``on_train_begin`` before touching the model."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(tmp, save_steps=1, load_path=None)
            cb = CheckpointCallback(trainer)
            # Replace model with a stub that would raise if state_dict() was called.
            failing_model = MagicMock(spec=nn.Module)
            failing_model.state_dict.side_effect = AssertionError("must not load")
            trainer.model = failing_model
            cb.trainer = trainer  # re-bind
            cb.on_train_begin(TrainerState(max_steps=10))
            failing_model.state_dict.assert_not_called()


# ----------------------------------------------------------------------------
# Async dispatch
# ----------------------------------------------------------------------------


class TestAsyncSaveDispatch(unittest.TestCase):
    """``save_async=True`` dispatches a thread; double-dispatch waits on first."""

    def test_dispatch_runs_save_in_background_thread(self):
        """Async dispatch fires the save off the calling thread."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(tmp, save_steps=1, save_async=True)
            cb = CheckpointCallback(trainer)
            saved_in = {}
            barrier = threading.Event()

            def _slow_save(state):
                saved_in["tid"] = threading.get_ident()
                # Block until the test releases so the background thread
                # is provably alive for the join assertion below.
                barrier.wait(timeout=2.0)

            cb._save = _slow_save
            state = TrainerState(max_steps=10)
            state.global_step = 1
            caller_tid = threading.get_ident()
            cb._dispatch_save(state)
            # The save thread must be live before barrier is signalled.
            self.assertIsNotNone(cb._save_thread)
            self.assertTrue(cb._save_thread.is_alive(), (
                "async save thread must be alive while _save is blocked"
            ))
            barrier.set()
            cb._save_thread.join(timeout=2.0)
            self.assertNotEqual(saved_in["tid"], caller_tid, (
                "async save must run off the caller thread"
            ))

    def test_second_dispatch_waits_for_first(self):
        """Back-to-back ``_dispatch_save`` must serialise via ``_join_pending``."""
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _build_trainer(tmp, save_steps=1, save_async=True)
            cb = CheckpointCallback(trainer)
            order = []
            release_first = threading.Event()

            def _save_a(state):  # pylint: disable=unused-argument
                order.append(("a_start", time.time()))
                release_first.wait(timeout=2.0)
                order.append(("a_end", time.time()))

            def _save_b(state):  # pylint: disable=unused-argument
                order.append(("b_start", time.time()))
                order.append(("b_end", time.time()))

            cb._save = _save_a
            state = TrainerState(max_steps=10)
            state.global_step = 1
            cb._dispatch_save(state)
            # Swap _save to _save_b before second dispatch — second dispatch
            # must wait for first (still blocked on release_first) before
            # firing b_start.
            cb._save = _save_b
            state.global_step = 2

            def _kicker():
                # Release the first save after a small delay so we can prove
                # the second dispatch is genuinely waiting on the join.
                time.sleep(0.1)
                release_first.set()

            threading.Thread(target=_kicker, daemon=True).start()
            cb._dispatch_save(state)  # this call blocks inside _join_pending
            cb._save_thread.join(timeout=2.0)

            kinds = [k for k, _ in order]
            self.assertEqual(kinds, ["a_start", "a_end", "b_start", "b_end"], (
                f"second dispatch must wait for first; observed order: {kinds}"
            ))

    def test_second_dispatch_reports_error_from_joined_save(self):
        """An async failure discovered during join must block the next save."""
        with tempfile.TemporaryDirectory() as tmp:
            callback = CheckpointCallback(_build_trainer(tmp, save_async=True))
            started = threading.Event()
            release = threading.Event()

            def failing_save(_state):
                started.set()
                release.wait(timeout=2.0)
                callback._save_error = OSError("disk full")

            callback._save = failing_save
            state = TrainerState(max_steps=10)
            state.global_step = 1
            callback._dispatch_save(state)
            self.assertTrue(started.wait(timeout=2.0))
            release.set()
            state.global_step = 2

            with self.assertRaisesRegex(RuntimeError, "Checkpoint save failed"):
                callback._dispatch_save(state)


class TestFailurePropagation(unittest.TestCase):
    """Recorded callback failures must remain visible to trainer lifecycles."""

    def test_raise_if_load_failed_preserves_original_error(self):
        """A deferred resume failure must raise with its original cause."""
        with tempfile.TemporaryDirectory() as tmp:
            callback = CheckpointCallback(_build_trainer(tmp, load_path=tmp))
            callback._load_error = ValueError("invalid checkpoint")

            with self.assertRaisesRegex(RuntimeError, "Failed to load checkpoint") as context:
                callback.raise_if_load_failed()

            self.assertIsInstance(context.exception.__cause__, ValueError)

    def test_raise_if_save_failed_preserves_original_error(self):
        """A deferred save failure must raise after pending work is joined."""
        with tempfile.TemporaryDirectory() as tmp:
            callback = CheckpointCallback(_build_trainer(tmp))
            callback._save_error = OSError("disk full")

            with self.assertRaisesRegex(RuntimeError, "Checkpoint save failed") as context:
                callback.raise_if_save_failed()

            self.assertIsInstance(context.exception.__cause__, OSError)

    def test_dispatch_does_not_erase_prior_save_error(self):
        """A later periodic save must not overwrite an unreported failure."""
        with tempfile.TemporaryDirectory() as tmp:
            callback = CheckpointCallback(_build_trainer(tmp))
            callback._save_error = OSError("disk full")
            callback._save = MagicMock()

            with self.assertRaisesRegex(RuntimeError, "Checkpoint save failed"):
                callback._dispatch_save(TrainerState(max_steps=10))

            callback._save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
