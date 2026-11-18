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
"""Unit tests for the MindSpore operator statistics dump wait hook."""

import csv
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from typing import Any, Callable
import unittest
from unittest.mock import patch

import numpy as np


class _FakeMsDispatchMode:
    """Minimal context-manager replacement for MindSpore MsDispatchMode."""

    def __enter__(self) -> "_FakeMsDispatchMode":
        """Enter the fake dispatch mode."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback_value: Any) -> bool:
        """Exit the fake dispatch mode without suppressing exceptions."""
        return False


class _FakeDisableMsDispatchMode:
    """Track whether deferred tensor statistics disable MindSpore dispatch."""

    depth = 0

    def __enter__(self) -> "_FakeDisableMsDispatchMode":
        """Mark MindSpore dispatch as disabled for the current fake scope."""
        type(self).depth += 1
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback_value: Any) -> bool:
        """Restore the fake dispatch state."""
        type(self).depth -= 1
        return False


class _FakeTensor:
    """Numpy-backed Tensor stub with a BF16 conversion guard."""

    float_dispatch_depths = []

    def __init__(self, value: Any, dtype: str = "Float32") -> None:
        """Initialize a numpy-backed fake tensor."""
        self.value = np.asarray(value)
        self.dtype = dtype

    def set_value(self, value: Any) -> None:
        """Replace the fake device value when communication completes."""
        self.value = np.asarray(value)

    def asnumpy(self) -> np.ndarray:
        """Return a host copy of the fake tensor value."""
        return self.value.copy()

    def float(self) -> "_FakeTensor":
        """Convert to Float32 while recording the dispatch-disable depth."""
        type(self).float_dispatch_depths.append(_FakeDisableMsDispatchMode.depth)
        return _FakeTensor(self.value.astype(np.float32), dtype="Float32")


class _FakeNativeCommHandle:
    """Native communication handle stub whose wait materializes an output."""

    def __init__(self, on_wait: Callable[[], None]) -> None:
        """Initialize the completion callback."""
        self._on_wait = on_wait
        self.wait_count = 0

    def wait(self) -> None:
        """Materialize the fake communication output."""
        self.wait_count += 1
        self._on_wait()


class _FakePlatform:
    """Platform methods used by the dump helper."""

    @staticmethod
    def get_rank() -> int:
        """Return the single-process fake rank."""
        return 0

    @staticmethod
    def is_tensor(value: Any) -> bool:
        """Identify fake tensors."""
        return isinstance(value, _FakeTensor)


class _FakeOp:
    """Callable OpFunc replacement with a MindSpore-style name property."""

    def __init__(self, name: str, callback: Callable[..., Any]) -> None:
        """Initialize the fake operator name and callback."""
        self.name = name
        self._callback = callback

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the fake underlying operator."""
        return self._callback(*args, **kwargs)


def _read_rows(output_root: Path, step: int = 0) -> list[dict[str, str]]:
    csv_path = output_root / f"step_{step}" / "rank_0.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _run_collective(dump_module, output_root: Path, wait_immediately: bool) -> list[dict[str, str]]:
    output = _FakeTensor(np.zeros(4, dtype=np.float32))
    handle = _FakeNativeCommHandle(
        lambda: output.set_value(np.asarray([2.0, 4.0, 6.0, 8.0], dtype=np.float32))
    )
    op = _FakeOp("AllGatherIntoTensor", lambda *_args, **_kwargs: (output, handle))
    dump_mode = dump_module.MindSporeOpStatDumpMode(str(output_root))
    dump_mode.set_step(0)

    with dump_mode:
        result = dump_mode.__ms_dispatch__(op, (_FakeTensor([1.0]),), {})
        if not wait_immediately:
            rows_before_wait = _read_rows(output_root)
            assert not [row for row in rows_before_wait if row["io_type"] == "output"]
        result[1].wait()
    dump_mode.close()
    return _read_rows(output_root)


class TestMindSporeOpStatDump(unittest.TestCase):
    """Validate deferred communication collection without a MindSpore runtime."""

    def setUp(self) -> None:
        """Load the dump helper against small MindSpore/platform stubs."""
        self.original_wait = _FakeNativeCommHandle.wait
        _FakeTensor.float_dispatch_depths.clear()
        mindspore_module = ModuleType("mindspore")
        mindspore_module.MsDispatchMode = _FakeMsDispatchMode
        c_expression_module = ModuleType("mindspore._c_expression")
        c_expression_module.CommHandle = _FakeNativeCommHandle
        c_expression_module._DisableMsDispatchMode = _FakeDisableMsDispatchMode
        platform_module = ModuleType("hyper_parallel.platform")
        platform_module.get_platform = _FakePlatform

        module_name = "_test_mindspore_op_stat_dump"
        script_path = Path(__file__).parents[2] / "scripts" / "mindspore_op_stat_dump.py"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        self.module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        with patch.dict(
            sys.modules,
            {
                "mindspore": mindspore_module,
                "mindspore._c_expression": c_expression_module,
                "hyper_parallel.platform": platform_module,
                module_name: self.module,
            },
        ):
            spec.loader.exec_module(self.module)
        self.exit_stack = ExitStack()
        temp_dir = self.exit_stack.enter_context(tempfile.TemporaryDirectory())
        self.output_root = Path(temp_dir)

    def tearDown(self) -> None:
        """Check that every test releases the global wait hook and temporary files."""
        self.assertEqual(self.module._NativeCommWaitHook._active_users, 0)
        self.assertFalse(self.module._NativeCommWaitHook._records_by_handle_id)
        self.assertIs(_FakeNativeCommHandle.wait, self.original_wait)
        self.exit_stack.close()

    def test_async_wait_output_matches_synchronous_wait(self) -> None:
        """Deferred async output has the same statistics as an immediate synchronous wait."""
        sync_rows = _run_collective(self.module, self.output_root / "sync", wait_immediately=True)
        async_rows = _run_collective(self.module, self.output_root / "async", wait_immediately=False)

        sync_output = [row for row in sync_rows if row["io_type"] == "output"]
        async_output = [row for row in async_rows if row["io_type"] == "output"]
        self.assertEqual(len(sync_output), 1)
        self.assertEqual(len(async_output), 1)
        self.assertEqual(sync_output[0]["crc32"], async_output[0]["crc32"])
        self.assertEqual(sync_output[0]["l2_norm"], async_output[0]["l2_norm"])
        self.assertEqual(async_output[0]["completion_state"], "waited")
        self.assertEqual(async_output[0]["op_index"], "1")

    def test_inplace_bfloat16_output_is_collected_after_wait_with_dispatch_disabled(self) -> None:
        """In-place communication resolves args[0] and disables dispatch during BF16 statistics."""
        output = _FakeTensor(np.zeros(2, dtype=np.float32), dtype="BFloat16")
        handle = _FakeNativeCommHandle(
            lambda: output.set_value(np.asarray([3.0, 5.0], dtype=np.float32))
        )
        op = _FakeOp("DistCommAllReduce", lambda *_args, **_kwargs: handle)
        dump_mode = self.module.MindSporeOpStatDumpMode(str(self.output_root))
        dump_mode.set_step(0)

        with dump_mode:
            result = dump_mode.__ms_dispatch__(op, (output,), {})
            self.assertIs(result, handle)
            self.assertFalse([row for row in _read_rows(self.output_root) if row["io_type"] == "output"])
            result.wait()
            result.wait()
        dump_mode.close()

        output_rows = [row for row in _read_rows(self.output_root) if row["io_type"] == "output"]
        self.assertEqual(len(output_rows), 1)
        self.assertEqual(output_rows[0]["dtype"], "BFloat16")
        self.assertEqual(output_rows[0]["l2_norm"], str(np.linalg.norm(np.asarray([3.0, 5.0]))))
        self.assertEqual(handle.wait_count, 2)
        self.assertEqual(_FakeTensor.float_dispatch_depths, [0, 1])
        self.assertEqual(_FakeDisableMsDispatchMode.depth, 0)

    def test_pending_output_keeps_launch_step_after_step_switch(self) -> None:
        """A communication completed after set_step writes to its launch-step CSV."""
        output = _FakeTensor(np.zeros(1, dtype=np.float32))
        handle = _FakeNativeCommHandle(lambda: output.set_value(np.asarray([7.0], dtype=np.float32)))
        op = _FakeOp("ReduceScatterTensor", lambda *_args, **_kwargs: (output, handle))
        dump_mode = self.module.MindSporeOpStatDumpMode(str(self.output_root))
        dump_mode.set_step(0)

        with dump_mode:
            dump_mode.__ms_dispatch__(op, (_FakeTensor([1.0]),), {})
            dump_mode.set_step(1)
            handle.wait()
        dump_mode.close()

        output_rows = [row for row in _read_rows(self.output_root, step=0) if row["io_type"] == "output"]
        self.assertEqual(len(output_rows), 1)
        self.assertEqual(output_rows[0]["completion_state"], "waited")
        self.assertFalse((self.output_root / "step_1" / "rank_0.csv").exists())


if __name__ == "__main__":
    unittest.main()
