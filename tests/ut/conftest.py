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
"""Pytest hooks for ``tests/ut``."""

from __future__ import annotations

import ctypes
import ctypes.util
import os

import pytest


def _preload_libgomp_early_for_static_tls() -> None:
    """Preload ``libgomp`` so torch then MindSpore do not exhaust static TLS.

    ``pytest_configure`` imports PyTorch first; without an early global load of
    OpenMP, a later ``import mindspore`` can fail with:

        ImportError: libgomp.so.1: cannot allocate memory in static TLS block

    Set ``HYPER_PARALLEL_SKIP_LIBGOMP_PRELOAD=1`` to disable.  Optional
    ``HYPER_PARALLEL_LIBGOMP_PATH`` selects a specific shared object (e.g. CI).
    """
    if os.environ.get("HYPER_PARALLEL_SKIP_LIBGOMP_PRELOAD", ""):
        return
    candidates: list[str] = []
    env_path = os.environ.get("HYPER_PARALLEL_LIBGOMP_PATH", "").strip()
    if env_path:
        candidates.append(env_path)
    found = ctypes.util.find_library("gomp")
    if found:
        candidates.append(found)
    candidates.append("libgomp.so.1")
    for path in candidates:
        if not path:
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


_preload_libgomp_early_for_static_tls()


def pytest_configure(config) -> None:  # pylint: disable=unused-argument
    """Import ``dtensor`` under PyTorch before collection loads tests that set ``mindspore`` at import time."""
    os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
    import hyper_parallel.platform.platform as _pp  # pylint: disable=import-outside-toplevel

    _pp.platform = None
    import hyper_parallel.core.dtensor.dtensor  # noqa: F401 pylint: disable=unused-import


@pytest.fixture(scope="module", autouse=True)
def _restore_torch_platform_between_ut_modules(request):
    """Many ``tests/ut`` modules temporarily set ``HYPER_PARALLEL_PLATFORM=mindspore``; reset for the next module."""
    yield
    try:
        from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # pylint: disable=import-outside-toplevel
            restore_torch_platform_for_ut,
        )

        restore_torch_platform_for_ut()
    # Best-effort: teardown must not abort pytest on import/platform edge cases.
    except (  # pragma: no cover
        ImportError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        AttributeError,
        TypeError,
    ):
        pass
