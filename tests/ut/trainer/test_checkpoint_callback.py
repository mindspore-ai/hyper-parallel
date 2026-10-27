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
"""Unit tests for the Trainer checkpoint callback's restore paths."""

import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

from hyper_parallel.components.checkpoint.config import CheckpointingConfig
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback
from hyper_parallel.trainer.state import TrainerState
from tests.common.mark_utils import arg_mark

_CALLBACK_MODULE = "hyper_parallel.trainer.callbacks.checkpoint_callback"
_SUCCESS_LOG = "Checkpoint loaded successfully"


class _FakeStateful:
    """Carry a state dict the way a model, optimizer or scheduler would."""

    def __init__(self, state: Optional[Dict[str, Any]] = None) -> None:
        """Seed the state a save would collect."""
        self._state = dict(state or {})
        self.loaded: Optional[Dict[str, Any]] = None

    def state_dict(self) -> Dict[str, Any]:
        """Return the state a save collects or a load fills in place."""
        return dict(self._state)

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Record what a restore handed back."""
        del strict
        self.loaded = dict(state_dict)


class _FakeDataLoader(_FakeStateful):
    """A stateful loader whose length sets the steps-per-epoch divisor."""

    def __init__(self, length: int) -> None:
        """Fix the epoch length the resume position is computed against."""
        super().__init__({"consumed_samples": 0})
        self._length = length

    def __len__(self) -> int:
        """Return the optimizer steps one epoch contains."""
        return self._length


class _FakeCheckpointer:
    """Replay a fixed payload instead of touching a real DCP checkpoint."""

    def __init__(self, extra_state: Optional[Dict[str, Any]] = None) -> None:
        """Hold the extra_state bundle a restore should read back, if any."""
        self.extra_state = extra_state
        self.skeleton_seen: Optional[Dict[str, Any]] = None
        self.async_failure: Optional[BaseException] = None

    def load(
        self,
        path: str,
        state: Dict[str, Any],
        *,
        strict_model: bool = True,
        extra_state_skeleton: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fill ``state`` only when the caller asked for the extra_state bundle."""
        del path, strict_model
        self.skeleton_seen = extra_state_skeleton
        if extra_state_skeleton is not None:
            state["extra_state"] = dict(self.extra_state or {})
        return state

    def maybe_wait_for_async_save(self) -> None:
        """No asynchronous save is ever in flight in these tests."""

    def raise_for_failed_async_save(self) -> None:
        """Report the failure a test planted, the way the real backend would."""
        if self.async_failure is not None:
            raise self.async_failure


class TestCheckpointerCallbackRestore(unittest.TestCase):
    """Verify what each restore mode reads back into the trainer."""

    @staticmethod
    def _build_trainer(restore_from: str, restore_train_state: bool) -> SimpleNamespace:
        """Build the callback's minimal Trainer dependency surface."""
        checkpoint = CheckpointingConfig(
            save_ckpt=False,
            checkpoint_dir=restore_from,
            restore_from=restore_from,
            restore_optimizer=True,
            restore_train_state=restore_train_state,
        )
        return SimpleNamespace(
            mesh=None,
            config=SimpleNamespace(checkpoint=checkpoint, model_init_dtype=None),
            state=TrainerState(global_step=0, epoch=0),
            model=_FakeStateful({"weight": 1}),
            optimizer=_FakeStateful({"state": {}}),
            lr_scheduler=_FakeStateful({"last_epoch": 0}),
            # Deliberately unequal: train_steps counts optimizer steps per epoch,
            # the loader counts micro-batches, and only the former sets the
            # position. Deriving it from len(loader) would report (0, 7) below.
            train_dataloader=_FakeDataLoader(8),
            train_steps=4,
        )

    @staticmethod
    def _restore(trainer: SimpleNamespace, checkpointer: _FakeCheckpointer) -> Any:
        """Run the callback's restore hook and return the stubbed module logger."""
        with patch(f"{_CALLBACK_MODULE}.build_checkpointer", return_value=checkpointer), \
                patch(f"{_CALLBACK_MODULE}.initialize_optimizer_state", return_value=True), \
                patch(f"{_CALLBACK_MODULE}.logger") as log:
            callback = CheckpointerCallback(trainer)
            callback.on_train_begin(trainer.state)
        return log

    def _reported_position(self, log: Any) -> Tuple[int, int, int]:
        """Return the (global_step, start_epoch, start_step) the success log reports.

        The log is the only consumer of the resume position now, so it is also the
        only place a test can observe it.
        """
        for call in log.info.call_args_list:
            if call.args and str(call.args[0]).startswith(_SUCCESS_LOG):
                return call.args[2], call.args[3], call.args[4]
        emitted = [str(call.args[0])[:40] for call in log.info.call_args_list if call.args]
        raise AssertionError(f"no {_SUCCESS_LOG!r} log was emitted, got={emitted}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_weights_only_restore_reports_a_fresh_position(self) -> None:
        """Load weights without the extra_state bundle and still report a position.

        Feature: Weights-only checkpoint restore.
        Description: Restore with ``restore_train_state=false``, the mode the checkpointer
            itself recommends when a checkpoint carries no training state.
        Expectation: The restore completes, the weights are applied, and it reports step 0
            rather than failing on state that only the extra_state path would produce.
        """
        with tempfile.TemporaryDirectory() as restore_from:
            trainer = self._build_trainer(restore_from, restore_train_state=False)
            checkpointer = _FakeCheckpointer()
            log = self._restore(trainer, checkpointer)

        self.assertIsNone(
            checkpointer.skeleton_seen,
            f"weights-only restore must not request an extra_state skeleton, "
            f"got={checkpointer.skeleton_seen!r}",
        )
        self.assertEqual(
            trainer.model.loaded,
            {"weight": 1},
            f"weights were not applied: expected={{'weight': 1}}, got={trainer.model.loaded!r}",
        )
        self.assertEqual(
            trainer.state.global_step,
            0,
            f"weights-only restore must leave progress fresh: expected=0, "
            f"got={trainer.state.global_step}",
        )
        reported = self._reported_position(log)
        self.assertEqual(
            reported,
            (0, 0, 0),
            f"weights-only restore must report a fresh position: expected=(0, 0, 0), got={reported}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_train_state_restore_maps_the_step_onto_an_epoch_position(self) -> None:
        """Map a restored global step onto its epoch and offset within that epoch.

        Feature: Checkpoint restore of training progress.
        Description: Restore step 7 of epoch 1 with ``restore_train_state=true``, against a
            four-optimizer-step epoch fed by an eight-micro-batch loader.
        Expectation: Progress is restored, the reported position is the loop's own
            ``(state.epoch, global_step - epoch * train_steps)``, and the trainer gains no
            attribute carrying it.
        """
        with tempfile.TemporaryDirectory() as restore_from:
            trainer = self._build_trainer(restore_from, restore_train_state=True)
            checkpointer = _FakeCheckpointer(
                {
                    "global_step": 7,
                    "epoch": 1,
                    "lr_scheduler": {},
                    "train_dataloader": {},
                    "rng_state": {},
                }
            )
            log = self._restore(trainer, checkpointer)

        self.assertEqual(
            trainer.state.global_step,
            7,
            f"restored step mismatch: expected=7, got={trainer.state.global_step}",
        )
        reported = self._reported_position(log)
        self.assertEqual(
            reported,
            (7, 1, 3),
            f"reported resume position mismatch: expected=(7, 1, 3), got={reported}",
        )
        # The loops derive their own position from state, so a restore that also
        # parked it on the trainer would be reintroducing state nobody reads.
        carried = [name for name in ("start_epoch", "start_step") if hasattr(trainer, name)]
        self.assertEqual(
            carried,
            [],
            f"restore must not carry the resume position on the trainer, got={carried}",
        )


class TestCheckpointerCallbackAsyncFailure(unittest.TestCase):
    """Verify when a failed asynchronous save reaches the training loop."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_step_end_reports_a_failed_async_save_before_the_next_save(self) -> None:
        """Raise a failed async save at the next step, not at the next save.

        Feature: Early reporting of a failed asynchronous save.
        Description: End a step that the save cadence does not land on, while an async
            save has already failed.
        Expectation: The step raises that failure, so a long ``save_steps`` interval
            cannot spend thousands of steps on a run whose checkpoint is already lost.
        """
        checkpointer = _FakeCheckpointer()
        trainer = SimpleNamespace(
            mesh=None,
            config=SimpleNamespace(
                checkpoint=CheckpointingConfig(
                    save_ckpt=True,
                    save_steps=1000,
                    checkpoint_dir="unused",
                    is_async=True,
                ),
                model_init_dtype=None,
            ),
            state=TrainerState(global_step=7, epoch=0),
        )
        with patch(f"{_CALLBACK_MODULE}.build_checkpointer", return_value=checkpointer):
            callback = CheckpointerCallback(trainer)

        checkpointer.async_failure = OSError("No space left on device")
        with self.assertRaises(OSError) as ctx:
            callback.on_step_end(trainer.state)
        self.assertIn(
            "No space left on device",
            str(ctx.exception),
            f"the async failure must reach the training loop, got={ctx.exception!r}",
        )

    @patch("hyper_parallel.trainer.callbacks.checkpoint_callback.set_device_rng_state")
    def test_train_state_restore_can_replay_from_configured_data_start(
            self,
            mock_set_device_rng_state: MagicMock,
    ) -> None:
        """Warm-start restore keeps progress and RNG without loading the cursor."""
        dataloader = MagicMock()
        callback = CheckpointerCallback.__new__(CheckpointerCallback)
        callback._restore_dataloader_state = False
        callback.trainer = SimpleNamespace(
            state=TrainerState(),
            train_dataloader=dataloader,
            train_steps=4,
            lr_scheduler=None,
        )

        callback._apply_extra_state({
            "global_step": 1,
            "epoch": 0,
            "lr_scheduler": None,
            "train_dataloader": {"cursor": 7},
            "rng_state": {},
        })

        self.assertEqual(callback.trainer.state.global_step, 1)
        dataloader.load_state_dict.assert_not_called()
        mock_set_device_rng_state.assert_called_once_with(None)


class _FakeExternalState:
    """Minimal host external-state surface the callback interacts with."""

    def __init__(self, parameters: Optional[Dict[str, Any]] = None,
                 persisted_keys: Optional[frozenset] = None) -> None:
        """Seed the external parameter FQNs and the dense optimizer manifest."""
        self.parameters = dict(parameters or {})
        self.persisted_keys = persisted_keys if persisted_keys is not None else frozenset()
        self.saved_runtime: Any = None
        self.load_runtime: Any = None
        self.weights_only_loaded = False
        self.dense_optimizer: Any = None

    def before_checkpoint_save(self, runtime: Any) -> None:
        """Record the save-side runtime the callback publishes."""
        self.saved_runtime = runtime

    def before_checkpoint_load(self, runtime: Any) -> Any:
        """Return the validated load plan carrying the dense key manifest."""
        self.load_runtime = runtime
        return SimpleNamespace(persisted_optimizer_keys=self.persisted_keys)

    def after_checkpoint_load(self, runtime: Any, load_plan: Any) -> None:
        """Record the completed external restore."""
        del runtime, load_plan

    def optimizer_for_dcp(self) -> Any:
        """Expose the dense optimizer used for DCP skeleton initialization."""
        return self.dense_optimizer

    def after_weights_only_load(self) -> None:
        """Record a weights-only external restore."""
        self.weights_only_loaded = True


class _FakeMixedPrecision:
    """Stand-in for MixedPrecisionOptimizer isinstance checks."""

    def __init__(self) -> None:
        """Track main-parameter reloads."""
        self.reloads = 0

    def reload_model_params(self) -> None:
        """Count each main-parameter reload."""
        self.reloads += 1


def _external_trainer(checkpoint: CheckpointingConfig, pp_size: int = 1) -> SimpleNamespace:
    """Build the trainer surface the callback's external paths read."""
    return SimpleNamespace(
        mesh=SimpleNamespace(pp_size=pp_size),
        config=SimpleNamespace(checkpoint=checkpoint, model_init_dtype=None),
        state=TrainerState(global_step=0, epoch=0),
        model=_FakeStateful({"dense": 1, "host": 2}),
        optimizer=_FakeStateful({"state": {}}),
        lr_scheduler=_FakeStateful({"last_epoch": 0}),
        train_dataloader=None,
        data_iterator=None,
        model_integration=None,
        train_steps=4,
    )


def _bare_callback(trainer: SimpleNamespace) -> Any:
    """Create a callback without __init__ for direct method-level checks."""
    callback = CheckpointerCallback.__new__(CheckpointerCallback)
    callback.trainer = trainer
    callback._is_peft = False
    callback._is_async = False
    callback._save_ckpt = True
    callback._save_optimizer = True
    callback._save_train_state = True
    callback._save_extra_state_per_rank = True
    callback._checkpoint_dir = "unused"
    callback._save_total_limit = None
    callback._save_hf_weights = False
    callback._restore_optimizer = True
    callback._restore_train_state = False
    callback.checkpointer = MagicMock()
    return callback


class TestCheckpointerCallbackExternalValidation(unittest.TestCase):
    """Reject checkpoint modes incompatible with host external state."""

    def _build(self, checkpoint: CheckpointingConfig, pp_size: int = 1) -> Any:
        """Construct the callback with external state visible on the model."""
        trainer = _external_trainer(checkpoint, pp_size)
        with patch(f"{_CALLBACK_MODULE}.get_model_external_state",
                   return_value=_FakeExternalState()), \
                patch(f"{_CALLBACK_MODULE}.build_checkpointer"):
            return CheckpointerCallback(trainer)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_async_and_peft_checkpointing_are_rejected(self) -> None:
        """Host external state requires synchronous non-PEFT checkpointing."""
        for flag in ({"is_async": True}, {"is_peft": True}):
            with self.subTest(**flag):
                with self.assertRaisesRegex(ValueError, "synchronous non-PEFT"):
                    self._build(CheckpointingConfig(**flag))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_partial_save_is_rejected(self) -> None:
        """Host checkpoints must save optimizer and train state together."""
        with self.assertRaisesRegex(ValueError, "requires optimizer and train-state saving"):
            self._build(CheckpointingConfig(save_optimizer=False))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_partial_restore_is_rejected(self) -> None:
        """Host restores must include optimizer and train state together."""
        with self.assertRaisesRegex(ValueError, "restore requires optimizer and train state"):
            self._build(CheckpointingConfig(restore_from="x", restore_train_state=False))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pp_requires_per_rank_extra_state(self) -> None:
        """Host PP checkpoints must write extra state per rank."""
        with self.assertRaisesRegex(ValueError, "per-rank extra state"):
            self._build(CheckpointingConfig(save_extra_state_per_rank=False), pp_size=2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_compatible_configuration_passes(self) -> None:
        """A synchronous, complete, per-rank configuration is accepted."""
        callback = self._build(CheckpointingConfig())
        self.assertEqual(callback._last_saved_step, -1)


class TestStateLeafKeys(unittest.TestCase):
    """Dense optimizer manifest flattening."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_nested_optimizer_state_flattens_to_leaf_keys(self) -> None:
        """DCP traversal lists every persisted dense optimizer leaf."""
        from hyper_parallel.trainer.callbacks.checkpoint_callback import _state_leaf_keys

        keys = _state_leaf_keys({"state": {"w": {"exp_avg": 1}}, "param_groups": []})
        self.assertEqual(keys, frozenset({"state.w.exp_avg", "param_groups"}))


class TestModelStateDictExternal(unittest.TestCase):
    """External parameter exclusion from the persisted model state."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_external_parameters_are_stripped(self) -> None:
        """The dense payload drops every external FQN."""
        callback = _bare_callback(_external_trainer(CheckpointingConfig()))
        external = _FakeExternalState({"host": object()})
        with patch(f"{_CALLBACK_MODULE}.get_model_external_state", return_value=external):
            state_dict = callback._model_state_dict()
        self.assertEqual(state_dict, {"dense": 1})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_missing_external_fqn_is_rejected(self) -> None:
        """Every external FQN must exist in the model state dict."""
        callback = _bare_callback(_external_trainer(CheckpointingConfig()))
        external = _FakeExternalState({"host": object(), "other": object()})
        with patch(f"{_CALLBACK_MODULE}.get_model_external_state", return_value=external):
            with self.assertRaisesRegex(RuntimeError, "External model FQNs missing"):
                callback._model_state_dict()


class TestSaveCheckpointExternal(unittest.TestCase):
    """External sidecar publication during checkpoint save."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_save_publishes_persisted_dense_keys(self) -> None:
        """The external save receives the dense optimizer leaf manifest."""
        callback = _bare_callback(_external_trainer(CheckpointingConfig()))
        callback.trainer.optimizer = _FakeStateful({"state": {"w": 1}})
        external = _FakeExternalState({"host": object()})
        with patch(f"{_CALLBACK_MODULE}.get_model_external_state", return_value=external), \
                patch(f"{_CALLBACK_MODULE}.get_device_rng_state", return_value=None):
            callback._save_checkpoint(TrainerState(global_step=3, epoch=0))
        self.assertEqual(external.saved_runtime.step_dir, "unused/global_step_3")
        self.assertEqual(external.saved_runtime.persisted_optimizer_keys,
                         frozenset({"state.w"}))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_save_without_optimizer_publishes_an_empty_manifest(self) -> None:
        """A weights-only save reports no persisted dense optimizer keys."""
        callback = _bare_callback(_external_trainer(CheckpointingConfig()))
        callback._save_optimizer = False
        external = _FakeExternalState({"host": object()})
        with patch(f"{_CALLBACK_MODULE}.get_model_external_state", return_value=external), \
                patch(f"{_CALLBACK_MODULE}.get_device_rng_state", return_value=None):
            callback._save_checkpoint(TrainerState(global_step=3, epoch=0))
        self.assertEqual(external.saved_runtime.persisted_optimizer_keys, frozenset())


class TestLoadOptimizerPayloadExternal(unittest.TestCase):
    """Dense optimizer restore with external-state priming."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unpersisted_dense_state_is_dropped(self) -> None:
        """Primed optimizer state outside the manifest is dropped after load."""
        callback = _bare_callback(_external_trainer(CheckpointingConfig()))
        optimizer = MagicMock()
        callback._load_optimizer_payload(
            {"optimizer": {"state": {}}}, [optimizer], frozenset({"state"}))
        optimizer.load_state_dict.assert_called_once_with({"state": {}})
        optimizer.drop_unpersisted_dense_state.assert_called_once_with(frozenset({"state"}))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_weights_only_reload_restores_main_params(self) -> None:
        """Without optimizer leaves, mixed-precision main params are reloaded."""
        callback = _bare_callback(_external_trainer(CheckpointingConfig()))
        mixed = _FakeMixedPrecision()
        callback.trainer.optimizer = [mixed]
        with patch(f"{_CALLBACK_MODULE}.MixedPrecisionOptimizer", _FakeMixedPrecision):
            callback._load_optimizer_payload({"model": {}}, [], None)
        self.assertEqual(mixed.reloads, 1)


class TestApplyRestoredPayloadExternal(unittest.TestCase):
    """Dense/external key agreement checks after a checkpoint load."""

    @staticmethod
    def _ready_callback() -> Any:
        """Build a callback whose load result is one external key short."""
        callback = _bare_callback(_external_trainer(CheckpointingConfig()))
        callback.trainer.model.load_state_dict = (
            lambda state_dict, strict=True: SimpleNamespace(
                missing_keys=["host"], unexpected_keys=[]))
        return callback

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dense_keys_must_match_external_exclusions(self) -> None:
        """A dense payload that is not exactly the complement raises."""
        callback = self._ready_callback()
        external = _FakeExternalState({"host": object(), "other": object()})
        with self.assertRaisesRegex(RuntimeError, "differ from exact external exclusions"):
            callback._apply_restored_payload(
                {"model": {"dense": 1}}, [], external, None, "path", None)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_load_result_keys_must_match_external_exclusions(self) -> None:
        """Unexpected missing keys from load_state_dict raise."""
        callback = self._ready_callback()
        callback.trainer.model.load_state_dict = (
            lambda state_dict, strict=True: SimpleNamespace(
                missing_keys=["stray"], unexpected_keys=[]))
        external = _FakeExternalState({"host": object()})
        with self.assertRaisesRegex(RuntimeError, "differ from exact external exclusions"):
            callback._apply_restored_payload(
                {"model": {"dense": 1}}, [], external, None, "path", None)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_weights_only_restore_reloads_mixed_precision_main_params(self) -> None:
        """A weights-only load finishes external state without optimizer moments."""
        callback = self._ready_callback()
        callback._restore_optimizer = False
        callback.trainer.optimizer = []
        external = _FakeExternalState({"host": object()})
        external.dense_optimizer = _FakeMixedPrecision()
        with patch(f"{_CALLBACK_MODULE}.MixedPrecisionOptimizer", _FakeMixedPrecision), \
                patch(f"{_CALLBACK_MODULE}.validate_model_init_dtype"):
            callback._apply_restored_payload(
                {"model": {"dense": 1}}, [], external, None, "path", None)
        self.assertEqual(external.dense_optimizer.reloads, 1)
        self.assertTrue(external.weights_only_loaded)


class TestLoadCheckpointExternal(unittest.TestCase):
    """External load plan validation inside the restore flow."""

    def _callback_with_skeleton(self, external: _FakeExternalState,
                                optimizer_keys: frozenset) -> Any:
        """Build a callback whose restore skeleton carries the given keys."""
        callback = _bare_callback(_external_trainer(CheckpointingConfig()))
        callback._resolve_restore_path = lambda: "ckpt_dir"
        optimizer_sd = {key: 1 for key in optimizer_keys}
        callback._build_restore_skeleton = (
            lambda restore_path, ext: ({"model": {"dense": 1}, "optimizer": optimizer_sd},
                                       [callback.trainer.optimizer]))
        callback._apply_restored_payload = MagicMock()
        callback.trainer.model.load_state_dict = (
            lambda state_dict, strict=True: SimpleNamespace(
                missing_keys=["host"], unexpected_keys=[]))
        return callback

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_no_restore_path_returns_early(self) -> None:
        """Without a restore path the load hook is a no-op."""
        callback = _bare_callback(_external_trainer(CheckpointingConfig()))
        callback._resolve_restore_path = lambda: None
        callback._load_checkpoint()
        callback.checkpointer.load.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_matching_dense_manifest_passes(self) -> None:
        """A load plan whose keys survive the skeleton proceeds to apply."""
        external = _FakeExternalState({"host": object()}, frozenset({"w"}))
        callback = self._callback_with_skeleton(external, frozenset({"w"}))
        with patch(f"{_CALLBACK_MODULE}.get_model_external_state", return_value=external), \
                patch(f"{_CALLBACK_MODULE}.logger"):
            callback._load_checkpoint()
        self.assertEqual(external.load_runtime.step_dir, "ckpt_dir")
        callback._apply_restored_payload.assert_called_once()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_lost_dense_optimizer_keys_are_rejected(self) -> None:
        """Persisted keys missing from the loaded skeleton raise."""
        external = _FakeExternalState({"host": object()}, frozenset({"w", "lost"}))
        callback = self._callback_with_skeleton(external, frozenset({"w"}))
        with patch(f"{_CALLBACK_MODULE}.get_model_external_state", return_value=external), \
                patch(f"{_CALLBACK_MODULE}.logger"):
            with self.assertRaisesRegex(RuntimeError, "lost persisted keys"):
                callback._load_checkpoint()


if __name__ == "__main__":
    unittest.main()
