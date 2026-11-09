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
"""Neutral diagnostic call lifecycle and single-session dispatch tests."""

from types import SimpleNamespace
import unittest
from unittest import mock

from hyper_parallel.core.multicore.diagnostics import MegaKernelDiagnostics
from hyper_parallel.core.multicore.diagnostics.common.call import DiagnosticCall, LaunchArguments
from hyper_parallel.core.multicore.diagnostics.common.session import (
    _activate_session, _deactivate_session, _reset_active_session_for_test,
)


class TestDiagnosticCall(unittest.TestCase):
    """Validate calls without importing either concrete diagnostic tool."""

    def tearDown(self) -> None:
        """Release diagnostic state after failed assertions."""
        _reset_active_session_for_test()

    def test_call_exposes_complete_neutral_arguments(self):
        """Feature: MegaKernel diagnostics.
        Description: The shared ABI must not be owned by either tool.
        Expectation: Call exposes complete neutral arguments.
        """
        arguments = LaunchArguments(object(), object(), object(), object())
        call = DiagnosticCall(arguments)
        self.assertIs(call.runtime_config, arguments.runtime_config)
        self.assertIs(call.event_counters, arguments.event_counters)
        self.assertIs(call.diagnostic_buffer, arguments.diagnostic_buffer)
        self.assertFalse(hasattr(call, "profile_buffer"))
        self.assertFalse(hasattr(call, "diagnostic_buffer_override"))

    def test_native_failure_cancels_only_the_selected_call(self):
        """Feature: MegaKernel diagnostics.
        Description: Do not complete a failed native launch.
        Expectation: Native failure cancels only the selected call.
        """
        call = DiagnosticCall(LaunchArguments(None, None, None, None))
        with mock.patch.object(call, "cancel") as cancel, mock.patch.object(call, "complete") as complete:
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                with call:
                    raise RuntimeError("native failure")
        cancel.assert_called_once_with()
        complete.assert_not_called()

    def test_capture_failure_cancels_the_selected_call(self):
        """Feature: MegaKernel diagnostics.
        Description: Release tool-owned state when capture completion raises.
        Expectation: Capture failure cancels the selected call.
        """
        call = DiagnosticCall(LaunchArguments(None, None, None, None))
        with (
            mock.patch.object(call, "complete", side_effect=ValueError("capture failure")),
            mock.patch.object(call, "cancel") as cancel,
            self.assertRaisesRegex(ValueError, "capture failure"),
        ):
            with call:
                pass
        cancel.assert_called_once_with()

    def test_dispatch_calls_only_the_active_session(self):
        """Feature: MegaKernel diagnostics.
        Description: A new tool can implement the interface without modifying this adapter.
        Expectation: Dispatch calls only the active session.
        """
        runtime = SimpleNamespace(normal_tensor=object(), parameter_count=1)
        adapter = MegaKernelDiagnostics(runtime)
        session = mock.Mock()
        selected = DiagnosticCall(LaunchArguments(object(), object(), object(), object()))
        session.prepare_call.return_value = selected
        _activate_session("custom_tool", session)
        try:
            self.assertTrue(adapter.is_active)
            call = adapter.launch((object(),), direction="forward", fallback_event_counters=object())
            self.assertIs(call, selected)
            session.prepare_call.assert_called_once()
            self.assertIs(session.prepare_call.call_args.args[0].runtime, runtime)
        finally:
            _deactivate_session(session)

    def test_no_session_never_inspects_diagnostic_metadata(self):
        """Feature: MegaKernel diagnostics.
        Description: Even malformed diagnostic-only metadata cannot affect ordinary execution.
        Expectation: No session never inspects diagnostic metadata.
        """
        runtime = SimpleNamespace(normal_tensor=object())
        adapter = MegaKernelDiagnostics(runtime)
        events = object()
        self.assertFalse(adapter.is_active)
        with adapter.launch((), direction="forward", fallback_event_counters=events) as call:
            self.assertIs(call.runtime_config, runtime.normal_tensor)
            self.assertIs(call.diagnostic_buffer, events)
        self.assertTrue(call.finished)
