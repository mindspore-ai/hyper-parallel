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
"""Run the production AIC/AIV worker loops concurrently with host primitives."""

import ctypes
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hyper_parallel
from hyper_parallel.core.multicore.profiler.profiling import (
    CORE_HEADER,
    CUBE_SLOT_COUNT,
    PROFILE_RECORD,
    _calculate_profile_layout,
)
from hyper_parallel.core.multicore.scheduler.config import (
    EVENT_INVALID_ID,
    TaskAiCoreType,
    TaskDescC,
    TaskSplitValue,
    TaskType,
)
from hyper_parallel.core.multicore.scheduler.mix_task import append_mix_task
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config, serialize_runtime_config


_COMPUTE_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_uint32)
_BARRIER_CALLBACK = ctypes.CFUNCTYPE(None)


def _schedule(core_count: int, kinds: str):
    """Build interleaved core/MIX queues with dependencies after each MIX."""
    cfg = allocate_runtime_config(len(kinds))
    cfg.num_workers = core_count * 2
    values = TaskSplitValue()
    dependency = EVENT_INVALID_ID
    for task_id, kind in enumerate(kinds):
        task = TaskDescC()
        task.task_type = TaskType.TASK_ADD_CUSTOM
        task.task_index = task_id
        task.dependent_event = dependency
        task.trigger_event = task_id
        if kind == "M":
            if dependency == EVENT_INVALID_ID and task_id > 0:
                task.dependent_event = task_id - 1
            values.pre_task_num = task_id
            append_mix_task(cfg, values, task)
            dependency = task_id
        else:
            slot = 0 if kind == "C" else 1
            task.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE if slot == 0 else TaskAiCoreType.TASK_AICORE_VECTOR
            cfg.all_tasks[task_id] = task
            indices = cfg.cube_task_indices if slot == 0 else cfg.vector_task_indices
            indices[cfg.task_index_num[slot]] = task_id
            cfg.task_index_num[slot] += 1
            cfg.all_event_num_triggers[task_id] = 1
    cfg.task_num = len(kinds)
    cfg.atomic_add_values[0] = 1
    return cfg


class TestWorkerKernel(unittest.TestCase):
    """Exercise real queue merging, bounds, events and profiling on CPU threads."""

    @classmethod
    def setUpClass(cls) -> None:
        """Compile the exact production worker for the two device core types."""
        compiler = shutil.which("g++")
        if compiler is None:
            raise unittest.SkipTest("g++ is required for the production worker simulation")
        temporary = tempfile.mkdtemp(prefix="hp-worker-")
        cls.addClassCleanup(shutil.rmtree, temporary)
        cls.libraries = []
        cls.addClassCleanup(cls._unload_libraries)
        include_root = Path(hyper_parallel.__file__).resolve().parent.parent
        source = Path(__file__).with_name("worker_runner.cpp")
        for core_type in (1, 2):
            library_path = Path(temporary) / f"worker_{core_type}.dll"
            command = [
                compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread", "-I", str(include_root),
                "-I", str(source.with_name("worker_stubs")), str(source), "-o", str(library_path),
            ]
            if core_type == 1:
                command.append("-D__DAV_C220_CUBE__")
            if sys.platform == "win32":
                command.extend(["-static-libgcc", "-static-libstdc++"])
            subprocess.run(command, check=True)
            library = ctypes.CDLL(str(library_path))
            library.run_worker.argtypes = [
                ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                _COMPUTE_CALLBACK, _BARRIER_CALLBACK,
            ]
            library.run_worker.restype = ctypes.c_bool
            cls.libraries.append(library)

    @classmethod
    def _unload_libraries(cls) -> None:
        """Release Windows DLL handles before removing their temporary directory."""
        if sys.platform == "win32":
            for library in cls.libraries:
                ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(library._handle))  # pylint: disable=protected-access
        cls.libraries.clear()

    def _run_schedule(self, cfg, core_count: int, profiled: bool, ready: bool = False) -> None:
        """Check execution against queue ownership and all-participant MIX epochs."""
        cfg.cycle_profiling_enabled = profiled
        layout = _calculate_profile_layout(cfg)
        cfg.aic_profile_record_capacity = layout.aic_record_capacity
        cfg.aiv_profile_record_capacity = layout.aiv_record_capacity
        if ready:
            cfg.ready_event = 100
            cfg.all_event_num_triggers[100] = 1
        image = serialize_runtime_config(cfg)
        runtime = ctypes.create_string_buffer(image)
        events = (ctypes.c_int32 * (cfg.event_capacity + 48))()
        if ready:
            events[100] = 1
        profile = ctypes.create_string_buffer(layout.buffer_size)
        tiling = (ctypes.c_int64 * 8)(0, 2 if ready else 1, 0, 0, 0, core_count, len(image), ctypes.sizeof(events))
        cube_ids = list(cfg.cube_task_indices[:cfg.task_index_num[0]])
        vector_ids = list(cfg.vector_task_indices[:cfg.task_index_num[1]])
        mix_ids = list(cfg.mix_task_indices[:cfg.task_index_num[2]])
        participants = core_count * 3
        barrier = threading.Barrier(participants, timeout=5)
        completed = Counter()
        errors = []
        results = {}
        lock = threading.Lock()

        def run(core_type: int, worker_id: int) -> None:
            """Execute one native worker and retain its task and barrier sequence."""
            executed = []
            barrier_count = 0

            @_COMPUTE_CALLBACK
            def compute(task_id: int) -> None:
                """Check phase completion before recording a dispatched compute call."""
                with lock:
                    # A MIX phase cannot start until all earlier core work has drained.
                    predecessors = [index for index in mix_ids if index < task_id]
                    if task_id in mix_ids:
                        predecessors += [index for index in cube_ids + vector_ids if index < task_id]
                    for predecessor in predecessors:
                        expected = participants if predecessor in mix_ids else 1
                        if completed[predecessor] != expected:
                            errors.append((task_id, predecessor, completed[predecessor], expected))
                    completed[task_id] += 1
                executed.append(task_id)

            @_BARRIER_CALLBACK
            def synchronize() -> None:
                """Implement the all-participant hardware barrier with host threads."""
                nonlocal barrier_count
                barrier_count += 1
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    errors.append("MIX barrier timed out")

            success = self.libraries[core_type - 1].run_worker(
                worker_id, runtime, tiling, events, profile, compute, synchronize,
            )
            results[(core_type, worker_id)] = (success, executed, barrier_count)

        with ThreadPoolExecutor(max_workers=participants) as pool:
            futures = [pool.submit(run, 1, worker) for worker in range(core_count)]
            futures += [pool.submit(run, 2, worker) for worker in range(core_count * 2)]
            for future in futures:
                future.result(timeout=15)

        self.assertEqual(errors, [])
        for (core_type, worker_id), (success, executed, barrier_count) in results.items():
            with self.subTest(core_type=core_type, worker_id=worker_id):
                self.assertTrue(success)
                core_ids = cube_ids[worker_id::core_count] if core_type == 1 else (
                    vector_ids[worker_id // 2::core_count] if worker_id % 2 else []
                )
                expected = sorted(core_ids + mix_ids) if mix_ids else core_ids
                self.assertEqual(executed, expected)
                self.assertEqual(barrier_count, 2 * len(mix_ids))
                if profiled:
                    aic_stride = CORE_HEADER.size + layout.aic_record_capacity * PROFILE_RECORD.size
                    aiv_stride = CORE_HEADER.size + layout.aiv_record_capacity * PROFILE_RECORD.size
                    offset = worker_id * aic_stride if core_type == 1 else (
                        CUBE_SLOT_COUNT * aic_stride + worker_id * aiv_stride
                    )
                    header = CORE_HEADER.unpack_from(profile, offset)
                    record_count = sum(
                        1 + (cfg.all_tasks[index].task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL) +
                        (cfg.all_tasks[index].dependent_event != EVENT_INVALID_ID) for index in expected
                    )
                    self.assertEqual(header[1:5], (record_count, 0, core_type, worker_id))
                    task_records = [struct.unpack_from("<I", profile, offset + CORE_HEADER.size +
                                                      index * PROFILE_RECORD.size + 20)[0]
                                    for index in range(record_count)]
                    self.assertEqual(set(task_records), set(expected))
        for task_id in cube_ids + vector_ids + mix_ids:
            expected_count = participants if task_id in mix_ids else 1
            if cfg.all_tasks[task_id].task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                expected_count = 0
            self.assertEqual(events[task_id], expected_count)

    def test_mixed_epochs_run_on_every_aic_and_aiv(self) -> None:
        """Include idle even AIV workers, consecutive MIX tasks and uneven queues.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Run the production C++ loops with uneven queues and consecutive MIX phases.
        Expectation: Every participant executes each MIX exactly once with correct events and profiles.
        """
        for core_count in (1, 3, 20, 24):
            for profiled in (False, True):
                with self.subTest(core_count=core_count, profiled=profiled):
                    self._run_schedule(_schedule(core_count, "CVCCMVVMMCV"), core_count, profiled)

    def test_mix_only_and_empty_core_queues(self) -> None:
        """Keep all workers alive through MIX phases when a core queue is empty.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Run MIX-only schedules and schedules with one empty core queue.
        Expectation: All workers participate and complete every barrier and expected task.
        """
        for kinds in ("MM", "VMV", "CMC"):
            for profiled in (False, True):
                with self.subTest(kinds=kinds, profiled=profiled):
                    self._run_schedule(_schedule(3, kinds), 3, profiled)

    def test_no_mix_keeps_existing_queue_assignment_and_uses_no_barriers(self) -> None:
        """Retain round-robin ownership, reordered queues and inactive even AIVs.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Run reordered Cube/Vector queues with ready metadata and no MIX tasks.
        Expectation: Round-robin ownership is preserved and no gang barriers are invoked.
        """
        for profiled in (False, True):
            cfg = _schedule(3, "CCCCVVVV")
            cfg.cube_task_indices[:4] = [3, 1, 0, 2]
            cfg.vector_task_indices[:4] = [7, 5, 4, 6]
            self._run_schedule(cfg, 3, profiled, ready=True)

    def test_mix_respects_ready_metadata(self) -> None:
        """Read the current runtime header and ready storage before MIX execution.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Run a mixed schedule with the current header and prepublished ready state.
        Expectation: Worker initialization and both execution modes complete with correct accounting.
        """
        for profiled in (False, True):
            self._run_schedule(_schedule(3, "CVMVC"), 3, profiled, ready=True)

    def test_shmem_mix_keeps_signal_owned_by_compute_kernel(self) -> None:
        """Run all MIX participants without adding a local SHMEM trigger or record.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Run a SHMEM MIX task with profiling both enabled and disabled.
        Expectation: Workers execute the task without a local trigger or trigger profile record.
        """
        for profiled in (False, True):
            cfg = _schedule(3, "VM")
            cfg.all_tasks[1].task_type = TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
            cfg.all_event_num_triggers[1] = 0
            self._run_schedule(cfg, 3, profiled)

    def test_mix_rejects_mismatched_launch_topology(self) -> None:
        """Trap before a barrier if serialized workers disagree with the launch.

        Feature: Collective AIC/AIV MIX task scheduling.
        Description: Launch workers with a core count different from the serialized topology.
        Expectation: Both core types trap before entering a gang barrier.
        """
        cfg = _schedule(3, "M")
        cfg.num_workers = 4
        image = serialize_runtime_config(cfg)
        runtime = ctypes.create_string_buffer(image)
        events = (ctypes.c_int32 * cfg.event_capacity)()
        profile = ctypes.create_string_buffer(1)
        tiling = (ctypes.c_int64 * 8)(0, 1, 0, 0, 0, 3, len(image), ctypes.sizeof(events))
        for library in self.libraries:
            self.assertFalse(library.run_worker(
                0, runtime, tiling, events, profile, _COMPUTE_CALLBACK(lambda _: None), _BARRIER_CALLBACK(lambda: None),
            ))
