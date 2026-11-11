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
"""MindSpore operator statistics dumper.

This helper wraps ``mindspore.MsDispatchMode`` and dumps per-operator tensor
statistics into step-scoped CSV files so repeated runs can be diffed to locate
the first nondeterministic operator.

Example:
    ```python
    from scripts.mindspore_op_stat_dump import MindSporeOpStatDumpMode

    op_dump = MindSporeOpStatDumpMode(
        output_root="./op_stat_dump",
        record_stack=True,
        verbose=False,
    )

    for step, batch in enumerate(dataset):
        op_dump.set_step(step)
        with op_dump:
            loss = train_step(batch)
    ```
"""

import csv
from dataclasses import dataclass, field
from functools import wraps
import logging
import numbers
import os
from pathlib import Path
import re
from threading import RLock
import traceback
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, Tuple
import zlib

import numpy as np
from mindspore import MsDispatchMode
from mindspore._c_expression import CommHandle as NativeCommHandle, _DisableMsDispatchMode

from hyper_parallel.platform import get_platform

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_ROOT = "./mindspore_op_stat_dump"
DEFAULT_IGNORED_OPS = {"StopGradient"}
CSV_FIELDNAMES = [
    "op_index",
    "op_name",
    "io_type",
    "leaf_path",
    "shape",
    "dtype",
    "numel",
    "nbytes",
    "l2_norm",
    "crc32",
    "completion_state",
    "stack",
]
__all__ = ["MindSporeOpStatDumpMode"]

COMM_INPLACE_OUTPUT_ARG_INDEX = {
    "AllGather": 0,
    "AllGatherIntoTensor": 0,
    "AllGatherIntoTensorUneven": 0,
    "AllReduce": 0,
    "AllToAll": 0,
    "AllToAllSingle": 0,
    "AlltoAll": 0,
    "AlltoAllV": 0,
    "AlltoAllVC": 0,
    "Broadcast": 0,
    "Gather": 0,
    "GatherIntoTensor": 0,
    "Receive": 0,
    "Recv": 0,
    "Reduce": 0,
    "ReduceScatter": 0,
    "ReduceScatterTensor": 0,
    "ReduceScatterTensorUneven": 0,
    "Scatter": 0,
    "ScatterTensor": 0,
}

platform = get_platform()


@dataclass
class _PendingCommRecord:
    """Tensor outputs whose statistics become valid after communication wait."""

    owner: "MindSporeOpStatDumpMode"
    handles: dict[int, NativeCommHandle]
    op_index: int
    op_name: str
    output_tree: Any
    stack_text: str
    csv_path: Path
    _remaining_handle_ids: set[int] = field(init=False)
    _lock: RLock = field(default_factory=RLock, init=False)

    def __post_init__(self) -> None:
        self._remaining_handle_ids = set(self.handles)

    def mark_waited(self, handle: NativeCommHandle) -> None:
        """Dump outputs once every native handle associated with the op has waited."""
        should_complete = False
        with self._lock:
            self._remaining_handle_ids.discard(id(handle))
            if not self._remaining_handle_ids:
                should_complete = True
        if should_complete:
            self.owner._complete_pending_record(self)  # pylint: disable=W0212


class _NativeCommWaitHook:
    """Process-wide native CommHandle.wait hook with per-handle pending records."""

    _lock = RLock()
    _active_users = 0
    _installed = False
    _original_wait: Optional[Callable[..., Any]] = None
    _records_by_handle_id: dict[int, list[_PendingCommRecord]] = {}

    @classmethod
    def acquire(cls) -> None:
        """Install the native wait hook and retain one active dump context."""
        with cls._lock:
            if not cls._installed:
                cls._install()
            cls._active_users += 1

    @classmethod
    def release(cls) -> None:
        """Release one dump context and restore wait when no pending work remains."""
        with cls._lock:
            cls._active_users = max(0, cls._active_users - 1)
            cls._maybe_uninstall()

    @classmethod
    def register(cls, record: _PendingCommRecord) -> None:
        """Associate one pending communication record with all of its handles."""
        with cls._lock:
            if not cls._installed:
                cls._install()
            for handle_id in record.handles:
                cls._records_by_handle_id.setdefault(handle_id, []).append(record)

    @classmethod
    def pending_count(cls, owner: "MindSporeOpStatDumpMode") -> int:
        """Return the number of unique pending records owned by a dump mode."""
        with cls._lock:
            return len({
                id(record)
                for records in cls._records_by_handle_id.values()
                for record in records
                if record.owner is owner
            })

    @classmethod
    def _install(cls) -> None:
        original_wait: Callable[..., Any] = NativeCommHandle.wait
        cls._original_wait = original_wait

        @wraps(original_wait)
        def wait_with_dump(handle: NativeCommHandle, *args: Any, **kwargs: Any) -> Any:
            """Delegate to native wait, then collect outputs registered for this handle."""
            result = original_wait(handle, *args, **kwargs)
            cls._on_wait_completed(handle)
            return result

        NativeCommHandle.wait = wait_with_dump
        cls._installed = True

    @classmethod
    def _on_wait_completed(cls, handle: NativeCommHandle) -> None:
        with cls._lock:
            records = cls._records_by_handle_id.pop(id(handle), [])
        for record in records:
            record.mark_waited(handle)
        with cls._lock:
            cls._maybe_uninstall()

    @classmethod
    def _maybe_uninstall(cls) -> None:
        if cls._installed and cls._active_users == 0 and not cls._records_by_handle_id:
            NativeCommHandle.wait = cls._original_wait
            cls._original_wait = None
            cls._installed = False


class MindSporeOpStatDumpMode(MsDispatchMode):
    """Dump MindSpore operator tensor statistics into CSV files.

    Args:
        output_root: Root directory used to store dumped CSV files.
        record_stack: Whether to record a compact Python call stack per op.
        stack_limit: Maximum number of retained stack frames.
        verbose: Whether to log one line for each dispatched op.
        ignored_ops: Optional iterable of operator names to skip.
        flush_every_op: Whether to flush the CSV file after each op.
    """

    def __init__(
        self,
        output_root: str = DEFAULT_OUTPUT_ROOT,
        *,
        record_stack: bool = False,
        stack_limit: int = 12,
        verbose: bool = False,
        ignored_ops: Optional[Iterable[str]] = None,
        flush_every_op: bool = True,
    ) -> None:
        """Initialize a step-aware MindSpore operator statistics dump mode."""
        super().__init__()
        self.output_root = Path(output_root)
        self.record_stack = record_stack
        self.stack_limit = stack_limit
        self.verbose = verbose
        self.flush_every_op = flush_every_op
        self.ignored_ops = set(DEFAULT_IGNORED_OPS)
        if ignored_ops is not None:
            self.ignored_ops.update(ignored_ops)

        self._rank: Optional[int] = None
        self._step_value: Optional[Any] = None
        self._step_dir_name = "step_unset"
        self._op_index = 0
        self._csv_files: dict[Path, Any] = {}
        self._csv_writers: dict[Path, csv.DictWriter] = {}
        self._csv_path: Optional[Path] = None
        self._script_path = os.path.abspath(__file__)
        self._state_lock = RLock()

    def __enter__(self) -> "MindSporeOpStatDumpMode":
        """Enter dispatch mode and install the native communication wait hook."""
        _NativeCommWaitHook.acquire()
        try:
            return super().__enter__()
        except Exception:
            _NativeCommWaitHook.release()
            raise

    def __exit__(self, exc_type: Any, exc_value: Any, traceback_value: Any) -> bool:
        """Exit dispatch mode while retaining hooks needed by pending communication."""
        try:
            return super().__exit__(exc_type, exc_value, traceback_value)
        finally:
            _NativeCommWaitHook.release()

    def set_step(self, step: Any) -> None:
        """Switch the active dump directory for the given step."""
        next_dir_name = self._format_step_dir(step)
        if next_dir_name == self._step_dir_name and step == self._step_value:
            return
        self._step_value = step
        self._step_dir_name = next_dir_name
        self._op_index = 0

    def step(self, step: Any) -> "MindSporeOpStatDumpMode":
        """Set the active step and return ``self`` for chaining."""
        self.set_step(step)
        return self

    def flush(self) -> None:
        """Flush every open CSV file."""
        with self._state_lock:
            for csv_file in self._csv_files.values():
                csv_file.flush()

    def close(self) -> None:
        """Close CSV files without forcing unfinished communication to wait."""
        pending_count = _NativeCommWaitHook.pending_count(self)
        if pending_count:
            LOGGER.warning(
                "closing MindSpore op dump with %s communication record(s) still waiting; "
                "their outputs will be written if the handles are waited later",
                pending_count,
            )
        self._close_csv_files()

    def __ms_dispatch__(self, func, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        op_name = getattr(func, "name", type(func).__name__)
        if op_name in self.ignored_ops:
            return func(*args, **kwargs)

        csv_path = self._current_csv_path()
        with self._state_lock:
            self._op_index += 1
            op_index = self._op_index
        stack_text = self._capture_stack() if self.record_stack else ""
        input_tree = {"args": args, "kwargs": kwargs}
        input_rows = self._build_rows(
            op_index, op_name, "input", "input", input_tree, stack_text, "ready"
        )
        self._write_rows(input_rows, csv_path)

        out = func(*args, **kwargs)

        handles = self._find_native_comm_handles(out)
        if handles:
            output_tree = self._resolve_comm_output_tree(op_name, args, out)
            record = _PendingCommRecord(
                owner=self,
                handles={id(handle): handle for handle in handles},
                op_index=op_index,
                op_name=op_name,
                output_tree=output_tree,
                stack_text=stack_text,
                csv_path=csv_path,
            )
            _NativeCommWaitHook.register(record)
        else:
            output_rows = self._build_rows(
                op_index, op_name, "output", "output", out, stack_text, "ready"
            )
            self._write_rows(output_rows, csv_path)
        if self.verbose:
            LOGGER.info(
                "dumped op=%s step=%s rank=%s op_index=%s deferred=%s csv=%s",
                op_name,
                self._step_dir_name,
                self._resolve_rank(),
                op_index,
                bool(handles),
                csv_path,
            )
        return out

    def __del__(self):
        self._close_csv_files()

    def _current_csv_path(self) -> Path:
        rank = self._resolve_rank()
        step_dir = self.output_root / self._step_dir_name
        step_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = step_dir / f"rank_{rank}.csv"
        return self._csv_path

    def _ensure_csv_writer(self, csv_path: Path) -> csv.DictWriter:
        writer = self._csv_writers.get(csv_path)
        if writer is not None:
            return writer
        is_new_file = not csv_path.exists() or csv_path.stat().st_size == 0
        csv_file = csv_path.open("a", encoding="utf-8", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        self._csv_files[csv_path] = csv_file
        self._csv_writers[csv_path] = writer
        if is_new_file:
            writer.writeheader()
        return writer

    def _close_csv_files(self) -> None:
        state_lock = getattr(self, "_state_lock", None)
        if state_lock is None:
            return
        with state_lock:
            for csv_file in self._csv_files.values():
                csv_file.close()
            self._csv_files.clear()
            self._csv_writers.clear()

    def _resolve_rank(self) -> int:
        if self._rank is not None:
            return self._rank
        try:
            self._rank = int(platform.get_rank())
            return self._rank
        except Exception:  # pylint: disable=W0718
            pass
        for env_name in ("RANK_ID", "RANK", "OMPI_COMM_WORLD_RANK"):
            env_value = os.environ.get(env_name)
            if env_value is not None:
                self._rank = int(env_value)
                return self._rank
        self._rank = 0
        return self._rank

    def _build_rows(
        self,
        op_index: int,
        op_name: str,
        io_type: str,
        root_name: str,
        tree: Any,
        stack_text: str,
        completion_state: str,
    ) -> list[dict[str, Any]]:
        rows = []
        for leaf_path, leaf in self._iter_named_leaves(tree, root_name):
            if not platform.is_tensor(leaf):
                continue
            tensor_stats = self._tensor_stats(leaf)
            rows.append(
                {
                    "op_index": op_index,
                    "op_name": op_name,
                    "io_type": io_type,
                    "leaf_path": leaf_path,
                    "shape": tensor_stats["shape"],
                    "dtype": tensor_stats["dtype"],
                    "numel": tensor_stats["numel"],
                    "nbytes": tensor_stats["nbytes"],
                    "l2_norm": tensor_stats["l2_norm"],
                    "crc32": tensor_stats["crc32"],
                    "completion_state": completion_state,
                    "stack": stack_text,
                }
            )
        return rows

    def _write_rows(self, rows: Sequence[dict[str, Any]], csv_path: Path) -> None:
        if not rows:
            return
        with self._state_lock:
            writer = self._ensure_csv_writer(csv_path)
            for row in rows:
                writer.writerow(row)
            if self.flush_every_op:
                self._csv_files[csv_path].flush()

    def _complete_pending_record(self, record: _PendingCommRecord) -> None:
        """Collect communication outputs after native wait has established stream ordering."""
        with _DisableMsDispatchMode():
            output_rows = self._build_rows(
                record.op_index,
                record.op_name,
                "output",
                "output",
                record.output_tree,
                record.stack_text,
                "waited",
            )
            self._write_rows(output_rows, record.csv_path)

    @staticmethod
    def _find_native_comm_handles(tree: Any) -> list[NativeCommHandle]:
        handles = []
        seen_handle_ids = set()
        for _, leaf in MindSporeOpStatDumpMode._iter_named_leaves(tree, "output"):
            if isinstance(leaf, NativeCommHandle) and id(leaf) not in seen_handle_ids:
                handles.append(leaf)
                seen_handle_ids.add(id(leaf))
        return handles

    @staticmethod
    def _resolve_comm_output_tree(op_name: str, args: tuple[Any, ...], out: Any) -> Any:
        for _, leaf in MindSporeOpStatDumpMode._iter_named_leaves(out, "output"):
            if platform.is_tensor(leaf):
                return out
        output_arg_index = COMM_INPLACE_OUTPUT_ARG_INDEX.get(op_name)
        if output_arg_index is None:
            # MindSpore's generated primitives use names such as
            # ``DistCommAllReduce`` and ``InnerCommAllReduce``.
            matching_names = (
                name for name in COMM_INPLACE_OUTPUT_ARG_INDEX if op_name.endswith(name)
            )
            matched_name = max(matching_names, key=len, default=None)
            if matched_name is not None:
                output_arg_index = COMM_INPLACE_OUTPUT_ARG_INDEX[matched_name]
        if output_arg_index is None or output_arg_index >= len(args):
            return ()
        return args[output_arg_index]

    def _capture_stack(self) -> str:
        frames = traceback.extract_stack()
        formatted_frames = []
        for frame in frames[:-2]:
            frame_path = os.path.abspath(frame.filename)
            if frame_path == self._script_path:
                continue
            if "/mindspore/" in frame_path:
                continue
            formatted_frames.append(f"{frame.filename}:{frame.lineno}:{frame.name}")
        if self.stack_limit > 0:
            formatted_frames = formatted_frames[-self.stack_limit:]
        return " | ".join(formatted_frames)

    @staticmethod
    def _iter_named_leaves(tree: Any, prefix: str) -> Iterator[Tuple[str, Any]]:
        if isinstance(tree, dict):
            for key, value in tree.items():
                child_prefix = f"{prefix}.{key}"
                yield from MindSporeOpStatDumpMode._iter_named_leaves(value, child_prefix)
            return
        if isinstance(tree, tuple):
            for index, value in enumerate(tree):
                child_prefix = f"{prefix}[{index}]"
                yield from MindSporeOpStatDumpMode._iter_named_leaves(value, child_prefix)
            return
        if isinstance(tree, list):
            for index, value in enumerate(tree):
                child_prefix = f"{prefix}[{index}]"
                yield from MindSporeOpStatDumpMode._iter_named_leaves(value, child_prefix)
            return
        yield prefix, tree

    @staticmethod
    def _tensor_stats(tensor: Any) -> dict[str, Any]:
        source_tensor = MindSporeOpStatDumpMode._tensor_to_numpy_ready(tensor)
        array = np.ascontiguousarray(source_tensor.asnumpy())
        payload = array.tobytes()
        return {
            "shape": str(tuple(array.shape)),
            "dtype": str(tensor.dtype),
            "numel": int(array.size),
            "nbytes": int(array.nbytes),
            "l2_norm": MindSporeOpStatDumpMode._l2_norm(array),
            "crc32": f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}",
        }

    @staticmethod
    def _tensor_to_numpy_ready(tensor: Any) -> Any:
        tensor_dtype = str(tensor.dtype).lower()
        if "bfloat16" in tensor_dtype:
            # MindSpore BF16 tensors cannot always be materialized via asnumpy directly.
            return tensor.float()
        return tensor

    @staticmethod
    def _l2_norm(array: np.ndarray) -> Any:
        if array.size == 0:
            return 0.0
        try:
            if np.issubdtype(array.dtype, np.complexfloating):
                flat = np.abs(array.astype(np.complex128, copy=False).reshape(-1))
            elif np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
                flat = array.astype(np.float64, copy=False).reshape(-1)
            else:
                return ""
        except TypeError:
            return ""
        return float(np.linalg.norm(flat))

    @staticmethod
    def _format_step_dir(step: Any) -> str:
        if isinstance(step, numbers.Integral):
            return f"step_{int(step)}"
        step_text = re.sub(r"[^0-9A-Za-z_.=-]", "_", str(step))
        return f"step_{step_text}"
