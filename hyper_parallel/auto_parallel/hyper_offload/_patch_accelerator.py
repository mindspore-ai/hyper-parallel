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
"""Monkey-patch ``torch.accelerator`` on PyTorch < 2.6.

PyTorch 2.5.x (and earlier) do **not** include the ``torch.accelerator``
module that was introduced in PyTorch 2.6.  This module creates a
compatible namespace and injects it into the ``torch`` module so that
all ``torch.accelerator.xxx`` calls work transparently regardless of
the PyTorch version.

Backend detection priority
  #1. Ascend NPU  — ``torch.npu.is_available()``
  #2. NVIDIA GPU  — ``torch.cuda.is_available()``
  #3. No accelerator — stub (``is_available()`` returns ``False``)

Usage
  Importing this module applies the patch as a side-effect::

      import hyper_parallel.auto_parallel.hyper_offload._patch_accelerator

  Or simply import the parent package (``__init__.py`` already does this)::

      import hyper_parallel.auto_parallel.hyper_offload  # patch applied

Notes
  For NPU backends the APIs ``reset_peak_memory_stats`` and
  ``max_memory_allocated`` are guarded with ``hasattr`` — if the
  installed ``torch_npu`` does not provide them, they fall back to
  no-ops / zero-returning stubs so that callers do not crash.
  ``current_stream`` and ``empty_cache`` are expected on any
  accelerator backend; if absent they will raise ``AttributeError``
  at call time (fail-fast).
"""

from __future__ import annotations

import types

import torch


def _patch() -> None:
    """Inject ``torch.accelerator`` if not already present (PyTorch ≥ 2.6)."""
    if hasattr(torch, "accelerator"):
        return  # PyTorch ≥ 2.6 — native support

    accelerator = types.ModuleType("accelerator")
    accelerator.__doc__ = "Monkey-patched ``torch.accelerator`` for PyTorch < 2.6"

    # ── Backend detection ────────────────────────────────────────────

    if hasattr(torch, "npu") and torch.npu.is_available():
        # ---- Ascend NPU backend ----
        def is_available() -> bool:
            return torch.npu.is_available()

        def current_accelerator() -> torch.device:
            return torch.device("npu", index=torch.npu.current_device())

        accelerator.is_available = is_available
        accelerator.current_accelerator = current_accelerator
        accelerator.current_stream = torch.npu.current_stream
        accelerator.empty_cache = torch.npu.empty_cache

        # These may not exist in older torch_npu versions
        if hasattr(torch.npu, "reset_peak_memory_stats"):
            accelerator.reset_peak_memory_stats = torch.npu.reset_peak_memory_stats
        else:
            def _npu_reset_peak_memory_stats() -> None:
                pass  # no-op fallback
            accelerator.reset_peak_memory_stats = _npu_reset_peak_memory_stats

        if hasattr(torch.npu, "max_memory_allocated"):
            accelerator.max_memory_allocated = torch.npu.max_memory_allocated
        else:
            def _npu_max_memory_allocated() -> int:
                return 0  # fallback
            accelerator.max_memory_allocated = _npu_max_memory_allocated

    elif torch.cuda.is_available():
        # ---- NVIDIA GPU (CUDA) backend ----
        def is_available() -> bool:
            return torch.cuda.is_available()

        def current_accelerator() -> torch.device:
            return torch.device("cuda", index=torch.cuda.current_device())

        accelerator.is_available = is_available
        accelerator.current_accelerator = current_accelerator
        accelerator.current_stream = torch.cuda.current_stream
        accelerator.empty_cache = torch.cuda.empty_cache
        accelerator.reset_peak_memory_stats = torch.cuda.reset_peak_memory_stats
        accelerator.max_memory_allocated = torch.cuda.max_memory_allocated

    else:
        # ---- No accelerator available (CPU-only) ----
        def is_available() -> bool:
            return False

        def current_accelerator() -> torch.device:
            raise RuntimeError(
                "No accelerator available (torch.accelerator not supported on CPU)"
            )

        def current_stream() -> torch.Stream:
            raise RuntimeError(
                "No accelerator available (torch.accelerator not supported on CPU)"
            )

        def empty_cache() -> None:
            pass

        def reset_peak_memory_stats() -> None:
            pass

        def max_memory_allocated() -> int:
            return 0

        accelerator.is_available = is_available
        accelerator.current_accelerator = current_accelerator
        accelerator.current_stream = current_stream
        accelerator.empty_cache = empty_cache
        accelerator.reset_peak_memory_stats = reset_peak_memory_stats
        accelerator.max_memory_allocated = max_memory_allocated

    # ── Inject into torch ────────────────────────────────────────────
    torch.accelerator = accelerator


_patch()
