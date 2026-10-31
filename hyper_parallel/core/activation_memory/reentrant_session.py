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
"""Scheduled replay state for reentrant activation checkpointing."""
import contextlib
import contextvars
import threading
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional


@dataclass(frozen=True)
class ReentrantSessionState:
    """Describe the active recompute session on the current thread."""

    session_id: Any
    retain_graph: bool


_CURRENT_HANDLE_COLLECTOR: contextvars.ContextVar[Optional[List[Any]]] = contextvars.ContextVar(
    "hyper_parallel_torch_reentrant_handle_collector",
    default=None,
)
_CURRENT_SESSION: contextvars.ContextVar[Optional[ReentrantSessionState]] = contextvars.ContextVar(
    "hyper_parallel_torch_reentrant_session",
    default=None,
)
_SESSION_LOCK = threading.RLock()
_SESSION_CALLS: Dict[Any, weakref.WeakSet] = defaultdict(weakref.WeakSet)


@contextlib.contextmanager
def reentrant_recompute_handle_collector_ctx() -> Iterator[List[Any]]:
    """Collect reentrant checkpoint handles created by forward execution."""
    handles = []
    token = _CURRENT_HANDLE_COLLECTOR.set(handles)
    try:
        yield handles
    finally:
        _CURRENT_HANDLE_COLLECTOR.reset(token)


@contextlib.contextmanager
def reentrant_recompute_session_ctx(session_id: Any, retain_graph: bool = False) -> Iterator[None]:
    """Bind replay selection and graph retention to the current thread."""
    token = _CURRENT_SESSION.set(ReentrantSessionState(session_id, retain_graph))
    try:
        yield
    finally:
        _CURRENT_SESSION.reset(token)


def get_current_reentrant_session() -> Optional[ReentrantSessionState]:
    """Return the active scheduled replay session, if any."""
    return _CURRENT_SESSION.get()


def collect_reentrant_call(call: Any) -> None:
    """Record one checkpoint call in the active forward collector."""
    collector = _CURRENT_HANDLE_COLLECTOR.get()
    if collector is not None:
        collector.append(call)


def register_reentrant_session_call(session_id: Any, call: Any) -> None:
    """Remember that *call* owns replay state retained by *session_id*."""
    with _SESSION_LOCK:
        _SESSION_CALLS[session_id].add(call)


def clear_reentrant_recompute_session(session_id: Any) -> None:
    """Release all checkpoint graphs retained by *session_id*."""
    with _SESSION_LOCK:
        calls = list(_SESSION_CALLS.pop(session_id, ()))
    for call in calls:
        call.clear_recompute_session(session_id)
