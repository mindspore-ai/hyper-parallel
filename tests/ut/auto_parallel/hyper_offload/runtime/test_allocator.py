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
"""CPU tests for session-owned native allocator configuration."""

from contextlib import ExitStack
import gc
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch
import weakref

import torch

from hyper_parallel.auto_parallel.hyper_offload.runtime.allocator import _AllocatorState, SessionAllocator
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
class TestSessionAllocator(unittest.TestCase):
    """Check exact restoration, overlapping owners and unsupported backends."""

    def setUp(self) -> None:
        """Use mock native configuration without initializing a device."""
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.current = False
        self.setter = Mock(side_effect=self._set_config)
        self.reader = Mock(side_effect=lambda: self.current)
        self.backend = SimpleNamespace(
            memory=SimpleNamespace(_set_allocator_settings=self.setter),
            empty_cache=Mock(), is_initialized=Mock(return_value=True),
        )
        stack.enter_context(patch("torch.accelerator.is_available", return_value=True))
        stack.enter_context(patch("torch.accelerator.current_accelerator", return_value=torch.device("cuda")))
        stack.enter_context(patch("torch.get_device_module", return_value=self.backend))
        stack.enter_context(patch(
            "hyper_parallel.auto_parallel.hyper_offload.runtime.allocator._get_config_reader", return_value=self.reader,
        ))
        stack.enter_context(patch(
            "hyper_parallel.auto_parallel.hyper_offload.runtime.allocator._allocator_state", _AllocatorState(),
        ))
        self.owner = SessionAllocator()
        self.addCleanup(self.owner.close)

    def _set_config(self, value: str) -> None:
        self.current = value == "expandable_segments:True"

    def test_activation_persists_across_iterations_and_restores_on_release(self) -> None:
        """Feature: Session-owned allocator configuration.

        Description: Activate warmup and then enter another training iteration.
        Expectation: Enable only once, reuse cache, and restore on owner release.
        """
        self.owner.activate()
        self.assertTrue(self.current)
        self.owner.activate()
        self.backend.empty_cache.assert_called_once_with()
        self.reader.assert_called_once_with()
        self.setter.assert_called_once_with("expandable_segments:True")
        self.owner.close()
        self.assertFalse(self.current)
        self.assertEqual(self.setter.call_args_list, [
            call("expandable_segments:True"), call("expandable_segments:False"),
        ])

    def test_existing_expandable_setting_is_preserved(self) -> None:
        """Feature: Existing user allocator configuration.

        Description: The caller already enabled expandable segments.
        Expectation: Preserve its enabled value without rewriting settings.
        """
        self.current = True
        self.owner.activate()
        self.owner.close()
        self.assertTrue(self.current)
        self.setter.assert_not_called()
        self.backend.empty_cache.assert_not_called()

    def test_object_destruction_restores_the_original_setting(self) -> None:
        """Feature: Automatic session allocator cleanup.

        Description: Release the final reference to an active allocator owner.
        Expectation: Destroy the owner and restore the exact previous setting.
        """
        owner = SessionAllocator()
        owner.activate()
        reference = weakref.ref(owner)
        del owner
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(self.current)

    def test_failed_cache_retirement_restores_the_setting_and_allows_retry(self) -> None:
        """Feature: Warmup initialization cleanup.

        Description: Cache retirement fails after native configuration was enabled.
        Expectation: Restore the original value and allow a later activation.
        """
        self.backend.empty_cache.side_effect = RuntimeError("cache failed")
        with self.assertRaisesRegex(RuntimeError, "cache failed"):
            self.owner.activate()
        self.assertFalse(self.current)
        self.backend.empty_cache.side_effect = None
        self.owner.activate()
        self.assertTrue(self.current)

    def test_overlapping_owners_restore_only_after_the_last_release(self) -> None:
        """Feature: Shared allocator configuration lifetime.

        Description: Two owners overlap and the earlier one closes first.
        Expectation: Keep configuration enabled until the last owner closes.
        """
        other = SessionAllocator()
        self.addCleanup(other.close)
        self.owner.activate()
        other.activate()
        self.owner.close()
        self.assertTrue(self.current)
        other.close()
        self.assertFalse(self.current)
        self.assertEqual(self.reader.call_count, 2)
        self.backend.empty_cache.assert_called_once_with()
        self.assertEqual(self.setter.call_count, 2)

    def test_unknown_native_configuration_does_not_modify_the_backend(self) -> None:
        """Feature: Unsupported native configuration getters.

        Description: Installed Torch headers cannot expose the prior setting.
        Expectation: Execute unchanged rather than guessing a restore value.
        """
        self.reader.side_effect = None
        self.reader.return_value = None
        self.owner.activate()
        self.owner.close()
        self.setter.assert_not_called()
        self.backend.empty_cache.assert_not_called()

    def test_missing_extension_keeps_the_existing_allocator(self) -> None:
        """Feature: Optional compiler compatibility.

        Description: The optional native helper could not be loaded.
        Expectation: Do not modify the backend or clear its cache.
        """
        with patch(
            "hyper_parallel.auto_parallel.hyper_offload.runtime.allocator._get_config_reader", return_value=None,
        ):
            self.owner.activate()
        self.setter.assert_not_called()
        self.backend.empty_cache.assert_not_called()

    def test_missing_backend_setter_does_not_load_the_extension(self) -> None:
        """Feature: Unsupported accelerator backend.

        Description: A backend has no runtime allocator-setting API.
        Expectation: Execute without loading the optional native helper.
        """
        with patch("torch.get_device_module", return_value=SimpleNamespace(
            memory=SimpleNamespace(), is_initialized=lambda: True,
        )), patch(
            "hyper_parallel.auto_parallel.hyper_offload.runtime.allocator._get_config_reader",
        ) as reader:
            self.owner.activate()
        reader.assert_not_called()

    def test_cpu_activation_does_not_resolve_a_native_allocator(self) -> None:
        """Feature: CPU-only compatibility.

        Description: No accelerator is installed and querying one would fail.
        Expectation: Execute without any native getter or device lookup.
        """
        with patch("torch.accelerator.is_available", return_value=False), patch(
            "torch.accelerator.current_accelerator", side_effect=RuntimeError("no accelerator"),
        ) as current:
            self.owner.activate()
        current.assert_not_called()
        self.reader.assert_not_called()

    def test_rejected_setting_warns_without_registering_a_live_owner(self) -> None:
        """Feature: Failed native setting initialization.

        Description: The backend rejects the setting before changing allocator state.
        Expectation: Warn once, preserve the backend and permit another owner to acquire.
        """
        self.setter.side_effect = RuntimeError("enable failed")
        with self.assertLogs("hyper_parallel.auto_parallel.hyper_offload.runtime.allocator", level="WARNING"):
            self.owner.activate()
        self.owner.activate()
        self.setter.assert_called_once_with("expandable_segments:True")
        self.assertFalse(self.current)
        self.backend.empty_cache.assert_not_called()
        self.setter.side_effect = self._set_config
        other = SessionAllocator()
        self.addCleanup(other.close)
        other.activate()
        self.owner.close()
        self.assertTrue(self.current)
        other.close()
        self.assertFalse(self.current)

    def test_partially_applied_setting_is_restored_before_fallback(self) -> None:
        """Restore an enable that changes native state before reporting failure."""
        def apply_then_fail(value: str) -> None:
            """Simulate a setter that applies the value before rejecting it."""
            self._set_config(value)
            if self.current:
                raise RuntimeError("partially applied")

        self.setter.side_effect = apply_then_fail
        with self.assertLogs("hyper_parallel.auto_parallel.hyper_offload.runtime.allocator", level="WARNING"):
            self.owner.activate()
        self.assertFalse(self.current)
        self.backend.empty_cache.assert_not_called()
        self.owner.close()
        self.setter.side_effect = self._set_config
        self.owner.activate()
        self.assertTrue(self.current)
        self.owner.close()
        self.assertFalse(self.current)

    def test_failed_setting_restoration_is_not_silently_ignored(self) -> None:
        """A failed rollback cannot claim that the original backend was preserved."""
        def fail_setting(value: str) -> None:
            """Leave the setting enabled and reject both update attempts."""
            self._set_config("expandable_segments:True")
            raise RuntimeError(value)

        self.setter.side_effect = fail_setting
        with self.assertRaisesRegex(RuntimeError, "expandable_segments:False"):
            self.owner.activate()
        self.backend.empty_cache.assert_not_called()

    def test_uninitialized_accelerator_can_be_activated_later(self) -> None:
        """Defer hardware work for CPU execution, then acquire after device initialization."""
        self.backend.is_initialized.return_value = False
        self.owner.activate()
        self.reader.assert_not_called()
        self.backend.is_initialized.return_value = True
        self.owner.activate()
        self.assertTrue(self.current)

    def test_close_is_idempotent_and_allows_reactivation(self) -> None:
        """Closing twice must not release another session's lease or corrupt refcounts."""
        other = SessionAllocator()
        self.addCleanup(other.close)
        self.owner.activate()
        other.activate()
        self.owner.close()
        self.owner.close()
        self.assertTrue(self.current)
        other.close()
        self.assertFalse(self.current)
        self.owner.activate()
        self.assertTrue(self.current)

    def test_release_preserves_an_external_disable(self) -> None:
        """Do not rewrite configuration already disabled by the application."""
        self.owner.activate()
        self.current = False
        self.owner.close()
        self.setter.assert_called_once_with("expandable_segments:True")
