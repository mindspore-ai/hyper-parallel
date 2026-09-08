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
from unittest import mock

import pytest

# transformers v4 chains ``modeling_utils -> loss_utils -> loss_d_fine ->
# loss_for_object_detection -> image_transforms`` and imports ``tensorflow``
# whenever it is importable. The UT gate image ships a tensorflow build whose
# native preload (self_check) segfaults the ARM executor, so a plain
# ``from transformers import PreTrainedModel`` kills the whole collection.
# ``USE_TF=0`` makes transformers' ``is_tf_available()`` return False and
# skips that chain; nothing in this suite uses tensorflow.
os.environ.setdefault("USE_TF", "0")


class NoAcceleratorGuard:
    """Fail fast when a CPU-gated test accidentally reaches for a device.

    Patches CUDA device selection, NCCL/HCCL process-group initialization and
    ``torch_npu`` entry points to raise immediately, so "accidentally on
    accelerator" can never pass as an ordinary CPU test. Opt in via the
    ``no_accelerator`` fixture or use as a context manager.
    """

    _DEVICE_TARGETS = (
        "torch.cuda.init",
        "torch.cuda.set_device",
    )
    _FORBIDDEN_BACKENDS = {"nccl", "hccl"}

    def __init__(self):
        self._patchers = []

    @staticmethod
    def _fail(name):
        def _blocked(*args, **kwargs):
            raise RuntimeError(
                f"{name} reached from a CPU-gated test; Gate-1 forbids real "
                "accelerator devices and process groups"
            )

        return _blocked

    def _check_init_process_group(self, real_init):
        def _guarded(*args, **kwargs):
            backend = kwargs.get("backend")
            if backend is None and args:
                backend = args[0]
            if isinstance(backend, str) and backend.lower() in self._FORBIDDEN_BACKENDS:
                raise RuntimeError(
                    f"init_process_group(backend={backend!r}) reached from a "
                    "CPU-gated test; Gate-1 forbids real accelerator process groups"
                )
            return real_init(*args, **kwargs)

        return _guarded

    def start(self):
        """Install the guard patches."""
        for target in self._DEVICE_TARGETS:
            patcher = mock.patch(target, self._fail(target))
            patcher.start()
            self._patchers.append(patcher)
        import torch.distributed  # pylint: disable=import-outside-toplevel

        if hasattr(torch.distributed, "init_process_group"):
            patcher = mock.patch(
                "torch.distributed.init_process_group",
                self._check_init_process_group(torch.distributed.init_process_group),
            )
            patcher.start()
            self._patchers.append(patcher)

    def stop(self):
        """Restore every patched entry point."""
        while self._patchers:
            self._patchers.pop().stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


@pytest.fixture
def no_accelerator():
    """Activate :class:`NoAcceleratorGuard` for one test."""
    guard = NoAcceleratorGuard()
    guard.start()
    try:
        yield guard
    finally:
        guard.stop()


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
