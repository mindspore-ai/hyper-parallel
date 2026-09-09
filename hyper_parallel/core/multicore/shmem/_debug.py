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
"""Local Runtime snapshots and best-effort SHMEM Allocation diagnostics."""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ._lifecycle import _reference_count
from ._runtime import _load_native, _torch_modules


_HEX_ADDRESS_FIELDS = frozenset({"allocation_base"})
_LOG_LEVEL_ENV = "HYPER_PARALLEL_SHMEM_LOG_LEVEL"
_INTERNAL_FRAME_PREFIX = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _format_debug_value(field: str, value: object) -> object:
    """Render known address fields in hex; everything else keeps its natural repr."""
    if field in _HEX_ADDRESS_FIELDS:
        return hex(value)
    return value


def _render_debug_fields(fields: Mapping[str, object], indent: int = 0) -> list[str]:
    """Render diagnostic fields one per line, nesting mappings and mapping lists."""
    pad = "  " * indent
    lines: list[str] = []
    for key, value in fields.items():
        if isinstance(value, Mapping):
            lines.append(f"{pad}{key}:")
            lines.extend(_render_debug_fields(value, indent + 1))
        elif isinstance(value, list) and value and all(isinstance(item, Mapping) for item in value):
            lines.append(f"{pad}{key}:")
            for item in value:
                rendered = ", ".join(
                    f"{field}={_format_debug_value(field, item[field])}" for field in item
                )
                lines.append(f"{pad}  - {rendered}")
        else:
            lines.append(f"{pad}{key}: {_format_debug_value(key, value)}")
    return lines


def _site_rank() -> int:
    """Return the Torch distributed rank when initialized, else 0 like the Native default."""
    torch, _ = _torch_modules()
    dist = getattr(torch, "distributed", None)
    return int(dist.get_rank()) if dist is not None and dist.is_available() and dist.is_initialized() else 0


def _log_allocation_site(tensor: Any) -> None:
    """Best-effort DEBUG line identifying the direct caller of :func:`empty`."""
    if os.environ.get(_LOG_LEVEL_ENV) != "0":
        return
    try:
        frames = traceback.extract_stack()
        caller = next(
            (
                frame
                for frame in reversed(frames)
                if not os.path.abspath(frame.filename).startswith(_INTERNAL_FRAME_PREFIX)
            ),
            None,
        )
        callsite = "unknown" if caller is None else f"{caller.filename}:{caller.lineno} in {caller.name}"
        now = datetime.now()
        line = (
            f"[HP-SHMEM][rank {_site_rank()}][DEBUG] {now:%Y-%m-%d-%H:%M:%S}.{now.microsecond // 1000:03d} "
            f"[_api.py] op=Empty allocation_base={hex(tensor.data_ptr())} "
            f"bytes={tensor.numel() * tensor.element_size()} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"callsite={callsite}"
        )
        print(line, file=sys.stderr, flush=True)
    except Exception:  # pylint: disable=broad-exception-caught
        # Diagnostics must never turn a successful collective Allocation into an orphan.
        return


class _DebugState(dict):
    """Diagnostic snapshot dict rendered one field per line for human reading."""

    def __repr__(self) -> str:
        """Render diagnostic fields one per line for interactive inspection."""
        return "\n".join(_render_debug_fields(self))


def debug_state() -> dict[str, object]:
    """Return a local, read-only snapshot of current Runtime facts and its latest failure.

    This query does not require an acquired Runtime reference. ``reference_count`` is the number of successful
    process-local ``shmem.acquire()`` calls not yet paired with ``shmem.release()``; ranks may report different values.
    While the Runtime is Ready, ``config`` carries the effective Runtime configuration (``heap_size_bytes``,
    ``timeout_seconds``, ``data_engine``, ``bootstrap_endpoint_base``). Heap usage is projected from requested bytes:
    ``allocated_bytes`` sums the requested sizes of all active Allocations, ``remaining_bytes`` is
    ``heap_size_bytes - allocated_bytes``, and ``active_allocations`` lists every active Allocation as
    ``{"allocation_id", "allocation_base", "allocation_bytes"}`` ordered by identity. Compare a Tensor's
    ``data_ptr()`` against ``allocation_base`` ranges to find its Allocation; the REPL renders addresses in hex while
    stored values remain integers. CANN rounds blocks to 16-byte multiples, so true device occupancy can be higher.

    After the final clean ``shmem.release()``, the state is ``Uninitialized`` and lifecycle-specific Root, device,
    config, Allocation, maximum-active-allocation, and latest-failure fields are ``None`` until the next lifecycle
    starts.
    """
    reference_count = _reference_count()
    native_state = dict(_load_native()._debug_state())  # pylint: disable=protected-access
    state = _DebugState()
    state["state"] = native_state.pop("state")
    state["reference_count"] = reference_count
    state.update(native_state)
    return state


__all__ = ["debug_state"]
