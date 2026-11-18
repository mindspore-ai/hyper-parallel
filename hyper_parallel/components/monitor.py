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
"""Optional statistics interface injected by an owning runtime, without Trainer dependencies."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import torch

_OBSERVER = ContextVar("component_statistics_observer", default=None)


@contextmanager
def statistics_scope(observer: Any) -> Iterator[None]:
    """Limit observation to a real forward or optimizer call, excluding recomputation."""
    token = _OBSERVER.set(observer)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


def statistics_completion(name: str) -> Any:
    """Return a CPU-only future callback capturing the original operation and step."""
    observer = _OBSERVER.get()
    return observer.completion(name) if observer is not None else None


def report_statistics(kind: str, owner: Any, **values: Any) -> None:
    """Pass borrowed values synchronously; the observer must not retain large tensors."""
    if torch.compiler.is_compiling():
        return
    observer = _OBSERVER.get()
    if observer is not None:
        observer.record_component(kind, owner, values)


@contextmanager
def statistics_phase(name: str) -> Iterator[None]:
    """Measure an explicit component boundary only when an observer is installed."""
    observer = _OBSERVER.get()
    if observer is None:
        yield
    else:
        with observer.phase(name):
            yield
