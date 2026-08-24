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
"""Tests for ``CheckpointerCallback`` save / resume orchestration.

Resume failures are silent: training restarts, the loss curve still looks
plausible, but one of the state buckets (optimizer moments, data position, RNG)
quietly started from scratch and the run converges somewhere else. These tests
pin each bucket down individually plus the save/restore round trip that ties
them together.
"""

import copy
import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import hyper_parallel.auto_models.components.checkpoint.dcp_checkpointer as dcp_module
import hyper_parallel.auto_models.trainer.callbacks.checkpoint_callback as ckpt_module
from hyper_parallel.auto_models.components.checkpoint.config import CheckpointingConfig
from hyper_parallel.auto_models.trainer.callbacks import TrainerState
from hyper_parallel.auto_models.trainer.config import TrainingConfig


# ----------------------------------------------------------------------
# Fake DCP backend
# ----------------------------------------------------------------------


def _fill_in_place(
    target: Any,
    saved: Any,
    *,
    strict_model: bool = True,
    path: str = "",
    strict: bool = False,
) -> Any:
    """Copy ``saved`` into ``target``, mirroring the real load contract.

    Three behaviours matter here and all come from
    ``_ModelStrictLoadPlanner`` / ``StandardLoadPlanner``:

    * the read plan is built from the state dict the *caller* passes in, so a key
      the caller never materialized is never read from the checkpoint;
    * a key the caller asks for but the checkpoint lacks is skipped under
      ``allow_partial_load``;
    * except in the model subtree, where ``strict_model`` turns it into a hard
      error (``strict_model`` is off for PEFT resumes).
    """
    if isinstance(target, dict):
        for key, value in target.items():
            child_strict = strict or (path == "" and key == "model" and strict_model)
            if key not in saved:
                if child_strict:
                    raise RuntimeError(
                        f"Checkpoint is missing model key(s): {path}{key}"
                    )
                continue
            target[key] = _fill_in_place(
                value,
                saved[key],
                strict_model=strict_model,
                path=f"{path}{key}.",
                strict=child_strict,
            )
        return target
    if isinstance(target, list):
        for index, value in enumerate(target):
            target[index] = _fill_in_place(
                value,
                saved[index],
                strict_model=strict_model,
                path=f"{path}{index}.",
                strict=strict,
            )
        return target
    return copy.deepcopy(saved)


class _FakeDcp:
    """In-memory stand-in for the hyper distributed-checkpoint backend."""

    def __init__(self) -> None:
        self.saved: dict[str, Any] = {}
        self.save_calls = 0
        self.async_calls = 0
        self.load_planners: list[Any] = []

    def save(self, state_dict, *, checkpoint_id):
        """Record the state dict and lay down the DCP metadata file."""
        self.save_calls += 1
        checkpoint_id = str(checkpoint_id)
        os.makedirs(checkpoint_id, exist_ok=True)
        with open(os.path.join(checkpoint_id, ".metadata"), "wb"):
            pass
        self.saved[checkpoint_id] = copy.deepcopy(state_dict)

    def async_save(self, state_dict, *, checkpoint_id):
        """Persist immediately but hand back an unresolved completion future."""
        self.async_calls += 1
        self.save(state_dict, checkpoint_id=checkpoint_id)
        return SimpleNamespace(persist_completion=_ManualFuture())

    def load(self, state_dict, *, checkpoint_id, planner=None):
        """Fill ``state_dict`` in place under the planner's strictness policy."""
        self.load_planners.append(planner)
        _fill_in_place(
            state_dict,
            self.saved[str(checkpoint_id)],
            strict_model=getattr(planner, "strict_model", True),
        )


class _ManualFuture:
    """Future whose ``result()`` must be called explicitly, like a real save."""

    def __init__(self) -> None:
        self.waited = False

    def result(self):
        """Mark the pending save as drained."""
        self.waited = True
        return None


@pytest.fixture(name="fake_dcp")
def _fake_dcp(monkeypatch) -> _FakeDcp:
    """Swap the DCP entry points used by the callback for an in-memory double."""
    backend = _FakeDcp()
    # Patched on the checkpointer, which now owns the DCP calls; the callback
    # still goes through the real DistributedCheckpointer, so its save/load
    # orchestration stays under test.
    monkeypatch.setattr(dcp_module, "dcp_save", backend.save)
    monkeypatch.setattr(dcp_module, "dcp_async_save", backend.async_save)
    monkeypatch.setattr(dcp_module, "dcp_load", backend.load)
    return backend


# ----------------------------------------------------------------------
# Fake trainer
# ----------------------------------------------------------------------


class _StatefulLoader:
    """Minimal stateful dataloader double covering five steps per epoch."""

    def __init__(self, steps_per_epoch: int = 5) -> None:
        self.steps_per_epoch = steps_per_epoch
        self.state = {"epoch": 0, "batches_consumed": 0}

    def __len__(self) -> int:
        """Report the per-epoch step count the callback reads."""
        return self.steps_per_epoch

    def state_dict(self) -> dict:
        """Return the recorded position."""
        return dict(self.state)

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the recorded position."""
        self.state = dict(state_dict)


def _build_trainer(checkpoint_dir: str, **ckpt_kwargs) -> SimpleNamespace:
    """Build a trainer double carrying a real model, optimizer and scheduler."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    config = SimpleNamespace(
        checkpoint=CheckpointingConfig(checkpoint_dir=checkpoint_dir, **ckpt_kwargs),
        training=TrainingConfig(),
    )
    return SimpleNamespace(
        config=config,
        mesh=None,
        world_size=1,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=_StatefulLoader(steps_per_epoch=5),
        state=TrainerState(),
        train_steps=20,
        start_epoch=0,
        start_step=0,
    )


def _train_one_step(trainer: SimpleNamespace) -> None:
    """Run a real optimizer step so the moments become non-trivial."""
    output = trainer.model(torch.ones(2, 4))
    output.sum().backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.lr_scheduler.step()


# ----------------------------------------------------------------------
# Save cadence
# ----------------------------------------------------------------------


def test_saves_on_the_step_cadence_and_skips_duplicates(tmp_path, fake_dcp) -> None:
    """Emit on multiples of save_steps and never save the same step twice."""
    trainer = _build_trainer(str(tmp_path), save_steps=2, save_epochs=1)
    callback = ckpt_module.CheckpointerCallback(trainer)
    state = TrainerState()

    for step in range(1, 5):
        state.global_step = step
        callback.on_step_end(state)

    # Only steps 2 and 4 are on the cadence.
    assert len(fake_dcp.saved) == 2
    assert os.path.isdir(tmp_path / "global_step_2")
    assert os.path.isdir(tmp_path / "global_step_4")

    # epoch_end at an already-saved step must not save again.
    state.epoch = 0
    callback.on_epoch_end(state)
    assert len(fake_dcp.saved) == 2


def test_save_steps_zero_disables_periodic_saving(tmp_path, fake_dcp) -> None:
    """Leave saving entirely to epoch_end / train_end when save_steps is 0."""
    trainer = _build_trainer(str(tmp_path), save_steps=0, save_epochs=0)
    callback = ckpt_module.CheckpointerCallback(trainer)

    for step in range(1, 6):
        callback.on_step_end(TrainerState(global_step=step))

    assert fake_dcp.saved == {}


def test_train_end_saves_the_final_step_once(tmp_path, fake_dcp) -> None:
    """Force-save the last step, but not one that was just written."""
    trainer = _build_trainer(str(tmp_path), save_steps=0, save_epochs=0)
    callback = ckpt_module.CheckpointerCallback(trainer)

    callback.on_train_end(TrainerState(global_step=7))
    assert sorted(fake_dcp.saved) == [str(tmp_path / "global_step_7")]

    callback.on_train_end(TrainerState(global_step=7))
    assert len(fake_dcp.saved) == 1


def test_train_end_does_not_save_a_run_that_never_stepped(tmp_path, fake_dcp) -> None:
    """Skip the final save when no optimizer step ever ran."""
    trainer = _build_trainer(str(tmp_path), save_steps=0, save_epochs=0)
    callback = ckpt_module.CheckpointerCallback(trainer)

    callback.on_train_end(TrainerState(global_step=0))

    assert fake_dcp.saved == {}


@pytest.mark.parametrize(
    ("save_ckpt", "restore_from", "registered", "why"),
    [
        (True, None, True, "saves only"),
        (True, "LATEST", True, "saves and restores"),
        (False, "LATEST", True, "restores only --- still needs the callback"),
        (False, None, False, "does neither"),
    ],
)
def test_registration_is_decided_once_from_config(
    tmp_path, fake_dcp, save_ckpt, restore_from, registered, why
) -> None:
    """Build the callback only when this run actually checkpoints.

    Saving and restoring are independent switches, so the callback is needed if
    *either* is on. Deciding here beats registering unconditionally and
    re-checking an enable flag inside every hook.
    """
    from hyper_parallel.auto_models.trainer.base import BaseTrainer

    trainer = _build_trainer(
        str(tmp_path), save_ckpt=save_ckpt, restore_from=restore_from, save_steps=1
    )

    BaseTrainer._init_callbacks(trainer)

    assert any(
        isinstance(callback, ckpt_module.CheckpointerCallback)
        for callback in trainer._callbacks
    ) is registered, why


def test_save_ckpt_false_restores_without_writing(tmp_path, fake_dcp) -> None:
    """Support "start from this checkpoint and write nothing further"."""
    source = _build_trainer(str(tmp_path), save_steps=1, save_epochs=0)
    ckpt_module.CheckpointerCallback(source).on_step_end(TrainerState(global_step=4))
    saves_after_source = fake_dcp.save_calls

    target = _build_trainer(
        str(tmp_path), save_ckpt=False, restore_from="LATEST", save_steps=1
    )
    callback = ckpt_module.CheckpointerCallback(target)

    callback.on_train_begin(target.state)
    assert target.state.global_step == 4  # the restore still happened

    # Every save trigger must now be inert.
    callback.on_step_end(TrainerState(global_step=5))
    callback.on_epoch_end(TrainerState(global_step=5, epoch=0))
    callback.on_train_end(TrainerState(global_step=5))

    assert fake_dcp.save_calls == saves_after_source


@pytest.mark.parametrize(
    ("save_optimizer", "save_train_state", "expected_buckets"),
    [
        (True, True, {"model", "optimizer", "extra_state"}),
        (False, True, {"model", "extra_state"}),
        (True, False, {"model", "optimizer"}),
        (False, False, {"model"}),
    ],
)
def test_save_flags_select_which_buckets_are_written(
    tmp_path, fake_dcp, save_optimizer, save_train_state, expected_buckets
) -> None:
    """Write only the requested buckets, mirroring the restore-side knobs."""
    trainer = _build_trainer(
        str(tmp_path),
        save_steps=1,
        save_epochs=0,
        # Keep extra_state in the payload so every bucket is observable in one place.
        save_extra_state_per_rank=False,
        save_optimizer=save_optimizer,
        save_train_state=save_train_state,
    )
    callback = ckpt_module.CheckpointerCallback(trainer)

    callback.on_step_end(TrainerState(global_step=1))

    saved = fake_dcp.saved[str(tmp_path / "global_step_1")]
    assert set(saved) == expected_buckets


def test_weights_only_checkpoint_reports_why_it_cannot_resume(tmp_path, fake_dcp) -> None:
    """Say what is missing instead of silently resuming from step zero."""
    source = _build_trainer(
        str(tmp_path),
        save_steps=1,
        save_epochs=0,
        save_optimizer=False,
        save_train_state=False,
    )
    ckpt_module.CheckpointerCallback(source).on_step_end(TrainerState(global_step=4))

    target = _build_trainer(str(tmp_path), restore_from="LATEST")
    restored_callback = ckpt_module.CheckpointerCallback(target)

    with pytest.raises(FileNotFoundError, match="no training state"):
        restored_callback.on_train_begin(target.state)


# ----------------------------------------------------------------------
# Optimizer state priming
# ----------------------------------------------------------------------


def test_optimizer_priming_creates_state_without_moving_parameters() -> None:
    """Materialize Adam moments while leaving the weights untouched.

    Priming runs a real ``step()``, so weight decay would shrink every parameter
    if the hyperparameters were not zeroed for the call.
    """
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.5)
    before = [param.detach().clone() for param in model.parameters()]

    assert not optimizer.state
    assert dcp_module.initialize_optimizer_state(optimizer)

    assert optimizer.state
    for param, original in zip(model.parameters(), before):
        assert torch.equal(param, original)
    # The restored hyperparameters must survive the priming step.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.5)


def test_optimizer_priming_is_a_noop_when_state_exists() -> None:
    """Leave an already-stepped optimizer alone."""
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    moments = copy.deepcopy(optimizer.state_dict()["state"])

    assert dcp_module.initialize_optimizer_state(optimizer)

    assert str(optimizer.state_dict()["state"]) == str(moments)


# ----------------------------------------------------------------------
# Round trip
# ----------------------------------------------------------------------


def test_save_then_restore_recovers_every_state_bucket(tmp_path, fake_dcp) -> None:
    """Restore weights, optimizer moments, progress, scheduler, loader and RNG."""
    source = _build_trainer(str(tmp_path), save_steps=1, save_epochs=0)
    callback = ckpt_module.CheckpointerCallback(source)

    _train_one_step(source)
    source.train_dataloader.state = {"epoch": 1, "batches_consumed": 3}
    torch.manual_seed(1234)
    expected_rng = torch.get_rng_state()
    expected_weights = [param.detach().clone() for param in source.model.parameters()]
    expected_moments = copy.deepcopy(source.optimizer.state_dict()["state"])
    expected_lr = source.lr_scheduler.state_dict()

    state = TrainerState(global_step=8, epoch=1)
    callback.on_step_end(state)

    # A fresh process: new model, new optimizer, nothing stepped yet.
    target = _build_trainer(
        str(tmp_path), save_steps=1, save_epochs=0, restore_from="LATEST"
    )
    with torch.no_grad():
        for param in target.model.parameters():
            param.zero_()
    torch.manual_seed(4321)
    restored_callback = ckpt_module.CheckpointerCallback(target)

    restored_callback.on_train_begin(target.state)

    assert target.state.global_step == 8
    assert target.state.epoch == 1
    # steps_per_epoch is 5, so step 8 sits at step 3 of epoch 1 --- deriving this
    # from the run-total train_steps (20) would wrongly yield epoch 0.
    assert target.start_epoch == 1
    assert target.start_step == 3

    for param, expected in zip(target.model.parameters(), expected_weights):
        assert torch.allclose(param, expected)

    restored_moments = target.optimizer.state_dict()["state"]
    assert set(restored_moments) == set(expected_moments)
    for index, moment in expected_moments.items():
        assert torch.allclose(restored_moments[index]["exp_avg"], moment["exp_avg"])
        assert torch.allclose(
            restored_moments[index]["exp_avg_sq"], moment["exp_avg_sq"]
        )
        # A fresh optimizer would report zeroed moments; make sure the checkpoint
        # actually carried a trained state.
        assert moment["exp_avg"].abs().sum() > 0

    assert target.lr_scheduler.state_dict()["last_epoch"] == expected_lr["last_epoch"]
    assert target.train_dataloader.state_dict() == {"epoch": 1, "batches_consumed": 3}
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("per_rank", [True, False])
def test_extra_state_round_trips_in_both_layouts(tmp_path, fake_dcp, per_rank) -> None:
    """Support per-rank files and the DCP-embedded bundle equally.

    The embedded layout is the one that needs a skeleton on the load side: DCP
    fills only the keys the caller already declared, so without it the restore
    reads nothing back and silently resumes from step zero.
    """
    source = _build_trainer(
        str(tmp_path), save_steps=1, save_epochs=0, save_extra_state_per_rank=per_rank
    )
    callback = ckpt_module.CheckpointerCallback(source)
    _train_one_step(source)
    source.train_dataloader.state = {"epoch": 0, "batches_consumed": 7}
    callback.on_step_end(TrainerState(global_step=7, epoch=0))

    extra_dir = tmp_path / "global_step_7" / "extra_state"
    assert extra_dir.is_dir() is per_rank
    saved_keys = fake_dcp.saved[str(tmp_path / "global_step_7")]
    assert ("extra_state" in saved_keys) is not per_rank

    target = _build_trainer(
        str(tmp_path), save_steps=1, save_epochs=0, restore_from="LATEST"
    )
    ckpt_module.CheckpointerCallback(target).on_train_begin(target.state)

    assert target.state.global_step == 7
    assert target.start_epoch == 1
    assert target.start_step == 2
    assert target.train_dataloader.state_dict() == {"epoch": 0, "batches_consumed": 7}


def test_embedded_extra_state_that_reads_nothing_is_reported(
    tmp_path, fake_dcp
) -> None:
    """Never let an unread extra_state pass as a resume from step zero."""
    source = _build_trainer(
        str(tmp_path), save_steps=1, save_epochs=0, save_extra_state_per_rank=False
    )
    ckpt_module.CheckpointerCallback(source).on_step_end(TrainerState(global_step=3))

    # Simulate a checkpoint that never recorded the bundle at all.
    fake_dcp.saved[str(tmp_path / "global_step_3")].pop("extra_state")

    target = _build_trainer(str(tmp_path), restore_from="LATEST")
    restored_callback = ckpt_module.CheckpointerCallback(target)

    with pytest.raises(FileNotFoundError, match="no training state"):
        restored_callback.on_train_begin(target.state)


def test_restore_tolerates_parameters_without_optimizer_state(
    tmp_path, fake_dcp
) -> None:
    """Resume from a checkpoint whose optimizer never accumulated moments.

    Frozen weights, unrouted MoE experts and a checkpoint taken before the first
    ``step()`` all produce this shape. The primed optimizer asks for entries the
    checkpoint legitimately lacks; those must fall back to their zero init rather
    than abort the resume.
    """
    source = _build_trainer(str(tmp_path), save_steps=1, save_epochs=0)
    callback = ckpt_module.CheckpointerCallback(source)
    assert not source.optimizer.state  # nothing trained yet
    callback.on_step_end(TrainerState(global_step=6))

    target = _build_trainer(
        str(tmp_path), save_steps=1, save_epochs=0, restore_from="LATEST"
    )
    restored_callback = ckpt_module.CheckpointerCallback(target)

    restored_callback.on_train_begin(target.state)

    assert target.state.global_step == 6
    for moment in target.optimizer.state_dict()["state"].values():
        assert torch.count_nonzero(moment["exp_avg"]) == 0


def test_restore_rejects_a_checkpoint_missing_model_weights(tmp_path, fake_dcp) -> None:
    """Fail loudly on a truncated model checkpoint instead of half-loading it."""
    source = _build_trainer(str(tmp_path), save_steps=1, save_epochs=0)
    ckpt_module.CheckpointerCallback(source).on_step_end(TrainerState(global_step=4))

    saved = fake_dcp.saved[str(tmp_path / "global_step_4")]
    saved["model"].pop("weight")

    target = _build_trainer(str(tmp_path), restore_from="LATEST")
    restored_callback = ckpt_module.CheckpointerCallback(target)

    with pytest.raises(RuntimeError, match="missing model key"):
        restored_callback.on_train_begin(target.state)


def test_peft_restore_does_not_demand_full_model_coverage(tmp_path, fake_dcp) -> None:
    """Relax model strictness for adapter checkpoints, which are partial by design."""
    source = _build_trainer(str(tmp_path), save_steps=1, save_epochs=0, is_peft=True)
    ckpt_module.CheckpointerCallback(source).on_step_end(TrainerState(global_step=4))

    # An adapter checkpoint never carries the frozen base weights.
    fake_dcp.saved[str(tmp_path / "global_step_4")]["model"].pop("weight")

    target = _build_trainer(str(tmp_path), is_peft=True, restore_from="LATEST")
    restored_callback = ckpt_module.CheckpointerCallback(target)

    restored_callback.on_train_begin(target.state)

    assert target.state.global_step == 4
    assert fake_dcp.load_planners[-1].strict_model is False


def test_restoring_a_finished_run_does_not_rewrite_its_checkpoint(
    tmp_path, fake_dcp
) -> None:
    """Treat the restored step as already saved so train_end is a no-op."""
    source = _build_trainer(str(tmp_path), save_steps=1, save_epochs=0)
    ckpt_module.CheckpointerCallback(source).on_step_end(TrainerState(global_step=5))
    assert fake_dcp.save_calls == 1

    target = _build_trainer(
        str(tmp_path), save_steps=1, save_epochs=0, restore_from="LATEST"
    )
    callback = ckpt_module.CheckpointerCallback(target)
    callback.on_train_begin(target.state)

    # Nothing was trained, so train_end must not overwrite the loaded checkpoint.
    callback.on_train_end(target.state)

    assert fake_dcp.save_calls == 1


def test_restore_train_state_false_loads_weights_only(tmp_path, fake_dcp) -> None:
    """Support branching a new run off an existing checkpoint's weights."""
    source = _build_trainer(str(tmp_path), save_steps=1, save_epochs=0)
    callback = ckpt_module.CheckpointerCallback(source)
    _train_one_step(source)
    callback.on_step_end(TrainerState(global_step=8, epoch=1))

    target = _build_trainer(
        str(tmp_path),
        save_steps=1,
        save_epochs=0,
        restore_from="LATEST",
        restore_train_state=False,
    )
    restored_callback = ckpt_module.CheckpointerCallback(target)

    restored_callback.on_train_begin(target.state)

    assert target.state.global_step == 0
    assert target.start_epoch == 0
    assert target.train_dataloader.state_dict() == {"epoch": 0, "batches_consumed": 0}


def test_restore_reports_a_missing_extra_state_bundle(tmp_path, fake_dcp) -> None:
    """Fail loudly rather than resume from step zero with restored weights."""
    source = _build_trainer(str(tmp_path), save_steps=1, save_epochs=0)
    callback = ckpt_module.CheckpointerCallback(source)
    callback.on_step_end(TrainerState(global_step=4))

    extra_state_dir = tmp_path / "global_step_4" / "extra_state"
    for entry in extra_state_dir.iterdir():
        entry.unlink()

    target = _build_trainer(str(tmp_path), restore_from="LATEST")
    restored_callback = ckpt_module.CheckpointerCallback(target)

    with pytest.raises(FileNotFoundError, match="extra_state"):
        restored_callback.on_train_begin(target.state)


# ----------------------------------------------------------------------
# Checkpoint discovery
# ----------------------------------------------------------------------


def _find_latest(path) -> str:
    """Shorthand for the checkpointer's discovery entry point."""
    return dcp_module.DistributedCheckpointer().find_latest_checkpoint(str(path))


def test_latest_uses_the_pointer_file_when_present(tmp_path) -> None:
    """Resolve the newest checkpoint from one read instead of a directory scan."""
    for step in (10, 20):
        directory = tmp_path / f"global_step_{step}"
        directory.mkdir()
        (directory / ".metadata").touch()
    (tmp_path / dcp_module.LATEST_CHECKPOINT_FILE).write_text("10\n", encoding="utf-8")

    # The pointer wins over the numerically-newest directory: it records what
    # actually finished saving, which is the question being asked.
    assert _find_latest(tmp_path) == str(tmp_path / "global_step_10")


def test_latest_scans_when_no_pointer_file_exists(tmp_path) -> None:
    """Still find checkpoints written before the pointer file existed."""
    for step in (5, 30):
        directory = tmp_path / f"global_step_{step}"
        directory.mkdir()
        (directory / ".metadata").touch()

    assert _find_latest(tmp_path) == str(tmp_path / "global_step_30")


def test_latest_skips_a_checkpoint_interrupted_mid_save(tmp_path) -> None:
    """Never resume from a directory whose DCP metadata was never written.

    ``.metadata`` lands only after every rank's shards do, so its absence means
    the save was interrupted and the payload is unusable.
    """
    complete = tmp_path / "global_step_10"
    complete.mkdir()
    (complete / ".metadata").touch()
    partial = tmp_path / "global_step_20"
    partial.mkdir()
    (partial / "__0_0.distcp").touch()

    assert _find_latest(tmp_path) == str(complete)


@pytest.mark.parametrize(
    ("pointer", "reason"),
    [
        ("99", "the step it names is gone"),
        ("not a number", "the file is corrupt"),
        ("", "the file is empty"),
    ],
)
def test_latest_falls_back_when_the_pointer_is_unusable(tmp_path, pointer, reason) -> None:
    """Treat the pointer as a cache: fall back to scanning, never fail."""
    directory = tmp_path / "global_step_7"
    directory.mkdir()
    (directory / ".metadata").touch()
    (tmp_path / dcp_module.LATEST_CHECKPOINT_FILE).write_text(pointer, encoding="utf-8")

    assert _find_latest(tmp_path) == str(directory), reason


def test_latest_ignores_unrelated_directories(tmp_path) -> None:
    """Only consider global_step_* directories holding checkpoint data."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "global_step_abc").mkdir()
    (tmp_path / "global_step_3").mkdir()

    assert _find_latest(tmp_path) is None


def test_missing_restore_path_is_reported(tmp_path, fake_dcp) -> None:
    """Reject an explicit restore_from that does not exist."""
    trainer = _build_trainer(
        str(tmp_path), restore_from=str(tmp_path / "global_step_404")
    )
    callback = ckpt_module.CheckpointerCallback(trainer)

    with pytest.raises(FileNotFoundError):
        callback.on_train_begin(trainer.state)


# ----------------------------------------------------------------------
# Async saves
# ----------------------------------------------------------------------


def test_async_save_is_drained_before_the_next_save(tmp_path, fake_dcp) -> None:
    """Never let two saves overlap, and publish a checkpoint as newest only
    once its persistence future has resolved."""
    trainer = _build_trainer(str(tmp_path), save_steps=1, save_epochs=0, is_async=True)
    callback = ckpt_module.CheckpointerCallback(trainer)

    callback.on_step_end(TrainerState(global_step=1))
    pending = callback.checkpointer._async_save_response
    assert pending is not None
    pointer = tmp_path / dcp_module.LATEST_CHECKPOINT_FILE
    assert not pointer.exists()

    callback.on_step_end(TrainerState(global_step=2))

    assert pending.persist_completion.waited
    assert pointer.read_text(encoding="utf-8").strip() == "1"
    assert fake_dcp.async_calls == 2


def test_train_end_drains_the_pending_async_save(tmp_path, fake_dcp) -> None:
    """Finish persisting before the trainer tears the process group down."""
    trainer = _build_trainer(str(tmp_path), save_steps=1, save_epochs=0, is_async=True)
    callback = ckpt_module.CheckpointerCallback(trainer)

    state = TrainerState(global_step=3)
    callback.on_step_end(state)
    pending = callback.checkpointer._async_save_response

    callback.on_train_end(state)

    assert pending.persist_completion.waited
    assert callback.checkpointer._async_save_response is None
    pointer = tmp_path / dcp_module.LATEST_CHECKPOINT_FILE
    assert pointer.read_text(encoding="utf-8").strip() == "3"
