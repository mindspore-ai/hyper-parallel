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
"""Process-local ownership for mutually exclusive MegaKernel diagnostics."""

from __future__ import annotations

from threading import Lock
from typing import Any, Protocol

from hyper_parallel.core.multicore.diagnostics.common.call import DiagnosticCall, LaunchContext


class DiagnosticSession(Protocol):
    """Shared contract for profiling, Tensor dumping and Host execution observations."""

    def is_capture_enabled(self) -> bool:
        """Return whether this step needs diagnostic invocation preparation."""
        raise NotImplementedError

    def prepare_call(self, context: LaunchContext) -> DiagnosticCall:
        """Return complete launch arguments and tool-owned invocation state.

        Args:
            context: Selected invocation or operand context supplied by the diagnostic framework.
        """
        raise NotImplementedError


_ACTIVE_LOCK = Lock()
_ACTIVE_KIND: str | None = None
_ACTIVE_SESSION: DiagnosticSession | None = None


def _activate_session(kind: str, session: DiagnosticSession) -> None:
    """Claim the single process-local MegaKernel diagnostic session."""
    # pylint: disable=global-statement
    global _ACTIVE_KIND, _ACTIVE_SESSION
    with _ACTIVE_LOCK:
        if _ACTIVE_SESSION is not None:
            raise RuntimeError(
                "nested MegaKernel diagnostic contexts are not supported: "
                f"active={_ACTIVE_KIND}, requested={kind}"
            )
        _ACTIVE_KIND = kind
        _ACTIVE_SESSION = session


def _deactivate_session(session: DiagnosticSession) -> None:
    """Release ownership when ``session`` is still the active owner."""
    # pylint: disable=global-statement
    global _ACTIVE_KIND, _ACTIVE_SESSION
    with _ACTIVE_LOCK:
        if _ACTIVE_SESSION is session:
            _ACTIVE_KIND = None
            _ACTIVE_SESSION = None


def _get_active_session(kind: str | None = None) -> DiagnosticSession | None:
    """Return the active diagnostic session when its kind matches."""
    with _ACTIVE_LOCK:
        if kind is not None and _ACTIVE_KIND != kind:
            return None
        return _ACTIVE_SESSION


def _get_execution_observer() -> Any | None:
    """Return a selected Host observer without touching inactive execution.

    Activation remains serialized by the shared session lock. Runtime observations
    use existing Host metadata and never request a tensor copy or collective.
    """
    session = _ACTIVE_SESSION if _ACTIVE_KIND == "execution" else None
    return session if session is not None and session.is_capture_enabled() else None


def _reset_active_session_for_test() -> None:
    """Clear leaked diagnostic state after a unit-test failure."""
    # pylint: disable=global-statement
    global _ACTIVE_KIND, _ACTIVE_SESSION
    with _ACTIVE_LOCK:
        _ACTIVE_KIND = None
        _ACTIVE_SESSION = None
