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
"""Dense runtime serialization and production C++ reader layout."""

import ctypes
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import hyper_parallel
from hyper_parallel.core.multicore.diagnostics.profiler.cycle_trace import CORE_HEADER, PROFILE_RECORD
from hyper_parallel.core.multicore.modules.mega_moe.backward.gen_runtime_data import (
    build_config_for_rank as build_backward_config,
)
from hyper_parallel.core.multicore.modules.mega_moe.backward.storage import can_reuse_backward_dispatch
from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import (
    build_backward_graph,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.gen_runtime_data import (
    build_config_for_rank as build_forward_config,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import (
    build_forward_graph,
)
from hyper_parallel.core.multicore.scheduler.config import (
    NUM_WORKERS_CUBE,
    RuntimeConfigC,
    TaskDescC,
    TaskSplitValue,
    TaskType,
    configure_ready_handshake,
    event_workspace_bytes,
)
from hyper_parallel.core.multicore.scheduler.runtime import (
    RUNTIME_HEADER_BYTES,
    TASK_DESC_SIZE,
    allocate_runtime_config,
    runtime_config_serialized_size,
    runtime_config_task_capacity,
    serialize_runtime_config,
)
from tests.common.mark_utils import arg_mark

FORMER_TASK_LIMIT = 25600


class TestRuntimeSerialization(unittest.TestCase):
    """Cover bounds independently of any accelerator library."""

    def test_host_arrays_follow_capacity_for_every_graph(self) -> None:
        """Small graphs must not retain the former fixed task/queue arrays."""
        for required in (0, 1, 17, 25599, 25600, 25601):
            with self.subTest(required=required):
                cfg = allocate_runtime_config(required)
                capacity = (required + 15) // 16 * 16
                self.assertIsInstance(cfg, RuntimeConfigC)
                self.assertEqual(cfg.task_capacity, capacity)
                self.assertEqual(cfg.event_capacity, 1024)
                for array in (cfg.all_tasks, cfg.cube_task_indices, cfg.vector_task_indices, cfg.mix_task_indices):
                    self.assertEqual(len(array), capacity)
                self.assertEqual(ctypes.sizeof(cfg), runtime_config_serialized_size(capacity))
                self.assertEqual(type(cfg).all_event_num_triggers.offset, RUNTIME_HEADER_BYTES)

    def test_capacity_includes_terminal_sparse_ids_and_queue_lengths(self) -> None:
        """Neither task_num nor a single queue determines the storage bound."""
        cfg = allocate_runtime_config(4000)
        cfg.task_num = 1
        cfg.task_index_num[1] = 2
        cfg.vector_task_indices[1] = 3001
        self.assertEqual(runtime_config_task_capacity(cfg), 3008)
        cfg.vector_task_indices[1] = 0
        cfg.task_index_num[2] = 200
        self.assertEqual(runtime_config_task_capacity(cfg), 208)

    def test_rejects_invalid_queues_before_serialization(self) -> None:
        """Reject negative/out-of-range counts, IDs and descriptor sizes."""
        for count, task_id in ((-1, 0), (17, 0), (1, -1), (1, 16)):
            with self.subTest(count=count, task_id=task_id):
                cfg = allocate_runtime_config(16)
                cfg.task_index_num[0] = count
                cfg.cube_task_indices[0] = task_id
                with self.assertRaisesRegex(ValueError, "invalid runtime"):
                    serialize_runtime_config(cfg)
        for capacity in (-1, True, 1.5, 1 << 32):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                allocate_runtime_config(capacity)
        with self.assertRaises(ValueError):
            allocate_runtime_config(1, 1025)

    def test_exact_task_count_boundary_includes_last_terminal(self) -> None:
        """Test actual task counts, including the terminal above the old limit."""
        for tasks in (25599, 25600, 25601):
            with self.subTest(tasks=tasks):
                cfg = allocate_runtime_config(tasks + 1)
                cfg.task_num = tasks
                cfg.task_index_num[1] = 1
                cfg.vector_task_indices[0] = tasks
                image = serialize_runtime_config(cfg)
                encoded_capacity = struct.unpack_from("<I", image, 8)[0]
                capacity = (tasks + 1 + 15) // 16 * 16
                self.assertEqual(runtime_config_task_capacity(cfg), capacity)
                self.assertEqual(encoded_capacity, capacity)
                self.assertEqual(len(image), runtime_config_serialized_size(capacity))

    def test_real_forward_backward_graphs_exceed_legacy_limit(self) -> None:
        """Build normal unfused graphs with compute descriptors above 25,600."""
        for graph_builder, config_builder in (
            (build_forward_graph, build_forward_config), (build_backward_graph, build_backward_config),
        ):
            with self.subTest(graph=graph_builder.__name__):
                values = TaskSplitValue(tp=1, ep=2, seq_size=327680, all_expert_num=4, top_k=2)
                graph = graph_builder(values, hidden_size=128, intermediate_size=128, num_cube_cores=20)
                graph.propagate_splits(values)
                cfg = config_builder(graph, values, 0, 20)
                compute_tasks = sum(op.task_num for op in graph.topological_sort())
                self.assertGreater(compute_tasks, FORMER_TASK_LIMIT)
                queue_ids = [*cfg.cube_task_indices[:cfg.task_index_num[0]],
                             *cfg.vector_task_indices[:cfg.task_index_num[1]]]
                self.assertGreater(max(queue_ids), FORMER_TASK_LIMIT)
                self.assertEqual(max(queue_ids), compute_tasks)
                self.assertEqual(cfg.all_tasks[compute_tasks].task_type, 0)
                self.assertTrue(serialize_runtime_config(cfg))

    def test_real_graphs_expand_event_storage(self) -> None:
        """Expand events for a valid topology with 16 experts per rank."""
        for graph_builder, config_builder in (
            (build_forward_graph, build_forward_config), (build_backward_graph, build_backward_config),
        ):
            with self.subTest(graph=graph_builder.__name__):
                values = TaskSplitValue(tp=1, ep=64, seq_size=128, all_expert_num=1024, top_k=2)
                graph = graph_builder(values, hidden_size=128, intermediate_size=128, num_cube_cores=20)
                graph.propagate_splits(values)
                cfg = config_builder(graph, values, 63, 20)
                self.assertEqual(len(cfg.all_event_num_triggers), 1104)
                self.assertEqual(len(cfg.all_events), 1104)
                image = serialize_runtime_config(cfg)
                self.assertEqual(struct.unpack_from("<I", image, 12)[0], 1104)

    def test_expert_scratch_bounds_are_checked_before_allocation_or_copy(self) -> None:
        """Reject overflow and metadata whose lists exceed the host allocation."""
        for experts in (-1, True, 1.5, 1 << 32, (1 << 32) - 1):
            with self.subTest(experts=experts), self.assertRaises(ValueError):
                allocate_runtime_config(16, local_experts=experts)
        cfg = allocate_runtime_config(16)
        cfg.dynamic_data.dynamic_group_size = 17
        with self.assertRaisesRegex(ValueError, "group-list scratch"):
            serialize_runtime_config(cfg)

    def test_changed_worker_count_cannot_exceed_allocated_scratch(self) -> None:
        """Reject a widened runtime unless its scratch allocation also covers the workers."""
        cfg = allocate_runtime_config(16, local_experts=2)
        cfg.num_workers = 64
        with self.assertRaisesRegex(ValueError, "group-list scratch"):
            serialize_runtime_config(cfg)


    def test_backward_storage_reuse_requires_all_reader_dependencies(self) -> None:
        """Allow managed plans, but reject changed queues, joins and additional receive readers."""
        for ep, cores, rank in ((4, 20, 0), (8, 24, 7)):
            for mutation in (None, "order", "threshold", "dependency", "reader", "mixed"):
                with self.subTest(ep=ep, cores=cores, mutation=mutation):
                    values = TaskSplitValue(tp=1, ep=ep, seq_size=4096, all_expert_num=48, top_k=8)
                    graph = build_backward_graph(values, hidden_size=5120, intermediate_size=1792,
                                                 num_cube_cores=cores)
                    graph.propagate_splits(values)
                    cfg = build_backward_config(graph, values, rank, cores)
                    if mutation == "order":
                        cfg.cube_task_indices[0], cfg.cube_task_indices[cores] = (
                            cfg.cube_task_indices[cores], cfg.cube_task_indices[0])
                    elif mutation == "threshold":
                        task = cfg.all_tasks[cfg.cube_task_indices[cores]]
                        cfg.all_event_num_triggers[task.trigger_event] = cores - 1
                    elif mutation == "dependency":
                        task = next(t for t in cfg.all_tasks if t.task_type == TaskType.TASK_SWI_GLU_GRAD)
                        task.dependent_event = 0
                    elif mutation == "reader":
                        task = next(t for t in cfg.all_tasks if t.task_type == TaskType.TASK_GROUPED_MATMUL
                                    and t.outputs[0].input_position == 18)
                        task.inputs[0].input_position = 0
                    elif mutation == "mixed":
                        cfg.task_index_num[2] = 1
                    self.assertEqual(can_reuse_backward_dispatch(cfg, values.single_rank_expert_num, cores),
                                     mutation is None)


class TestRuntimeCppReader(unittest.TestCase):
    """Round-trip ctypes records through the exact production header."""

    @classmethod
    def setUpClass(cls) -> None:
        """Compile the device layout/decoder with synchronization-only stubs."""
        compiler = shutil.which("g++")
        if compiler is None:
            raise unittest.SkipTest("g++ is required for the runtime C++ ABI test")
        with tempfile.TemporaryDirectory(prefix="hp-runtime-reader-") as directory:
            library = Path(directory) / "runtime_reader.so"
            (Path(directory) / "kernel_operator.h").write_text(
                "#pragma once\n#include <stdexcept>\nusing GM_ADDR = uint8_t *;\n"
                "namespace AscendC {\nconstexpr int PIPE_ALL = 0;\n"
                "template <int> inline void PipeBarrier() {}\n"
                "inline uint64_t GetSystemCycle() { return 1; }\n"
                "inline void Trap() { throw std::runtime_error(\"Device trap\"); }\n}\n",
                encoding="utf-8",
            )
            # CI unpacks tests separately from the installed package and its headers.
            include_root = Path(hyper_parallel.__file__).resolve().parent.parent
            subprocess.run([
                compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                "-I", directory, "-I", str(include_root), str(Path(__file__).with_name("runtime_reader.cpp")),
                "-o", str(library),
            ], check=True)
            cls.reader = ctypes.CDLL(str(library))
        cls.reader.valid_runtime.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
        cls.reader.valid_runtime.restype = ctypes.c_bool
        cls.reader.valid_ready_runtime.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32]
        cls.reader.valid_ready_runtime.restype = ctypes.c_bool
        cls.reader.valid_expert_runtime.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        cls.reader.valid_expert_runtime.restype = ctypes.c_bool
        cls.reader.read_ready_event.argtypes = [ctypes.c_void_p]
        cls.reader.read_ready_event.restype = ctypes.c_uint32
        cls.reader.read_protocol_profile.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls.reader.read_task.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        cls.reader.read_layout.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls.reader.group_list_offset.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        cls.reader.group_list_offset.restype = ctypes.c_uint32
        cls.reader.append_runtime_bounds.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int64, ctypes.c_int64,
        ]
        cls.reader.append_runtime_bounds.restype = ctypes.c_size_t
        cls.reader.valid_runtime_from_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        cls.reader.valid_runtime_from_launch.restype = ctypes.c_bool
        cls.reader.read_transport_and_kernel_task_types.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        cls.reader.read_transport_and_kernel_task_types.restype = None
        cls.reader.record_profile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        cls.reader.record_profile.restype = None

    def test_transport_and_kernel_task_types_match_cpp(self) -> None:
        """Keep transport, MHC and MegaGate tasks on distinct C++ dispatch IDs."""
        wire_ids = (ctypes.c_uint32 * 4)()
        self.reader.read_transport_and_kernel_task_types(wire_ids)
        expected = (
            int(TaskType.TASK_SHMEM_GET_MEM), int(TaskType.TASK_MHC_POST),
            int(TaskType.TASK_GATE_SOFTPLUS), int(TaskType.TASK_GATE_GRAD_ZEROS),
        )

        self.assertEqual(tuple(wire_ids), expected, msg=f"Task ID mismatch: C++={tuple(wire_ids)}, Python={expected}")
        self.assertEqual(len(set(wire_ids)), len(wire_ids), msg=f"Kernel task IDs overlap: C++={tuple(wire_ids)}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_launch_bounds_preserve_native_tiling_and_use_real_storage(self) -> None:
        """Feature: Graph-owned runtime storage validation.

        Description: Append real buffer sizes behind aligned and unaligned native tiling.
        Expectation: Native bytes are untouched and truncated storage is rejected.
        """
        image = serialize_runtime_config(allocate_runtime_config(17))
        runtime = ctypes.create_string_buffer(image)
        for native_bytes in (64, 67):
            offset = (native_bytes + 7) // 8 * 8
            capacity = offset + 16
            for runtime_bytes, event_bytes, valid in (
                (len(image), 4096, True), (len(image) - 1, 4096, False), (len(image), 4095, False),
            ):
                with self.subTest(native_bytes=native_bytes, runtime_bytes=runtime_bytes, event_bytes=event_bytes):
                    prefix = bytes([0xA5]) * native_bytes
                    tiling = ctypes.create_string_buffer(prefix, capacity)
                    size = self.reader.append_runtime_bounds(tiling, capacity, native_bytes, runtime_bytes, event_bytes)
                    if event_bytes < 4096:
                        self.assertEqual(size, 0)
                        continue
                    self.assertEqual(size, capacity)
                    self.assertEqual(tiling.raw[:native_bytes], prefix)
                    self.assertEqual(struct.unpack_from("<QQ", tiling.raw, offset), (runtime_bytes, event_bytes))
                    self.assertEqual(self.reader.valid_runtime_from_launch(runtime, tiling, native_bytes), valid)
            tiling = ctypes.create_string_buffer(capacity)
            for available, native, runtime_bytes, event_bytes in (
                (capacity - 1, native_bytes, len(image), 4096),
                (capacity, 0, len(image), 4096),
                (capacity, capacity + 1, len(image), 4096),
                (capacity, native_bytes, -1, 4096),
                (capacity, native_bytes, 63, 4096),
                (capacity, native_bytes, 1 << 32, 4096),
                (capacity, native_bytes, len(image), -1),
            ):
                with self.subTest(available=available, native=native, runtime=runtime_bytes, events=event_bytes):
                    self.assertEqual(
                        self.reader.append_runtime_bounds(tiling, available, native, runtime_bytes, event_bytes), 0,
                    )

    def _round_trip(self, cfg) -> None:
        """Compare every TaskDesc field, queues, events, dynamic data and tail."""
        index = int(cfg.task_num) - 1
        record = struct.pack("<144I", *range(144))
        ctypes.memmove(ctypes.addressof(cfg.all_tasks[index]), record, TASK_DESC_SIZE)
        cfg.task_index_num[0] = 1
        cfg.cube_task_indices[0] = index
        cfg.all_event_num_triggers[31] = 7
        cfg.dynamic_data.dynamic_max_seq_len = -17
        group_index = (self.reader.group_list_offset(ctypes.byref(cfg), 1)
                       - type(cfg).grouped_matmul_group_list.offset) // 8
        cfg.grouped_matmul_group_list[group_index] = 123456
        cfg.atomic_add_values[0] = 3
        image = serialize_runtime_config(cfg)
        buffer = ctypes.create_string_buffer(image)
        events = len(cfg.all_event_num_triggers)
        self.assertTrue(self.reader.valid_runtime(buffer, len(image), events * 4))
        observed = TaskDescC()
        self.reader.read_task(buffer, index, ctypes.byref(observed))
        self.assertEqual(bytes(observed), record)
        offsets = (ctypes.c_uint32 * 12)()
        self.reader.read_layout(buffer, offsets)
        capacity = runtime_config_task_capacity(cfg)
        prefix = RUNTIME_HEADER_BYTES
        self.assertEqual(offsets[0], prefix)
        self.assertEqual(offsets[1], prefix + events * 4)
        self.assertEqual(offsets[2], offsets[1] + capacity * 576)
        self.assertEqual(offsets[3], offsets[2] + events * 16)
        self.assertEqual(offsets[5] - offsets[4], capacity * 4)
        self.assertEqual(offsets[6] - offsets[5], capacity * 4)
        self.assertEqual(offsets[7] - offsets[6], capacity * 4)
        self.assertEqual((offsets[10], offsets[11]), (capacity, events))
        self.assertEqual(struct.unpack_from("<i", image, offsets[0] + 31 * 4)[0], 7)
        self.assertEqual(struct.unpack_from("<i", image, offsets[4])[0], index)
        self.assertEqual(struct.unpack_from("<i", image, offsets[7] + 12)[0], -17)
        self.assertEqual(struct.unpack_from("<q", image, offsets[8])[0], 123456)
        self.assertEqual(struct.unpack_from("<i", image, offsets[9])[0], 3)
        self.assertFalse(self.reader.valid_runtime(buffer, len(image) - 1, events * 4))
        self.assertFalse(self.reader.valid_runtime(buffer, len(image), events * 4 - 1))

    def test_dense_runtime_round_trip(self) -> None:
        """Decode small, large and event-expanded graphs using one layout."""
        for experts in (0, 16, 17, 32, 33, 128, 257):
            for tasks, events in ((1, 1024), (17, 1024), (25601, 1024), (17, 2048)):
                with self.subTest(experts=experts, tasks=tasks, events=events):
                    cfg = allocate_runtime_config(tasks, events, experts)
                    cfg.task_num = tasks
                    self._round_trip(cfg)

    def test_group_lists_own_complete_cache_lines_within_existing_storage(self) -> None:
        """Prevent adjacent workers' full-line writebacks from corrupting expert counts."""
        for experts in (1, 16, 17, 31, 32, 33, 64, 128, 257):
            self._check_group_list_isolation(experts)

    def test_worker_sized_group_lists_match_device_offsets(self) -> None:
        """Keep the last Cube list before atomics for both 910 and 950 core counts."""
        for cores in (4, 8, 20, 24, 28, 32, 36):
            for experts in (2, 17, 33):
                with self.subTest(cores=cores, experts=experts):
                    cfg = allocate_runtime_config(16, local_experts=experts, num_workers=2 * cores)
                    image = serialize_runtime_config(cfg)
                    buffer = ctypes.create_string_buffer(image)
                    offsets = (ctypes.c_uint32 * 12)()
                    self.reader.read_layout(buffer, offsets)
                    end = self.reader.group_list_offset(buffer, cores - 1) + experts * 8
                    self.assertLessEqual(end, offsets[9])
                    self.assertEqual(offsets[9] + 32, len(image))
                    self.assertTrue(self.reader.valid_runtime(buffer, len(image), 4096))
                    self.assertFalse(self.reader.valid_runtime(buffer, len(image) - 1, 4096))

    def test_device_recorder_matches_expanded_host_profile_layout(self) -> None:
        """Run the production AIV recorder with SDK-only timing/synchronization stubs."""
        for cores in (4, 8, 20, 24, 28, 32, 36):
            with self.subTest(cores=cores):
                cfg = allocate_runtime_config(16, num_workers=cores * 2)
                cfg.aic_profile_record_capacity = 16
                cfg.aiv_profile_record_capacity = 32
                runtime = ctypes.create_string_buffer(serialize_runtime_config(cfg))
                cube_slots = max(24, cores)
                aic_stride = CORE_HEADER.size + 16 * PROFILE_RECORD.size
                aiv_stride = CORE_HEADER.size + 32 * PROFILE_RECORD.size
                capacity = cube_slots * aic_stride + cube_slots * 2 * aiv_stride
                buffer = ctypes.create_string_buffer(capacity + 128)
                buffer[capacity:] = bytes([0xA5]) * 128
                worker = cores * 2 - 1
                self.reader.record_profile(runtime, buffer, worker)
                offset = cube_slots * aic_stride + worker * aiv_stride
                self.assertEqual(CORE_HEADER.unpack_from(buffer.raw, offset)[:6], (1, 1, 0, 2, worker, 32))
                self.assertEqual(PROFILE_RECORD.unpack_from(buffer.raw, offset + 64), (10, 20, 1, 2, 3, 4))
                self.assertEqual(buffer.raw[capacity:], bytes([0xA5]) * 128)

    def test_real_graphs_allocate_all_950_worker_lists(self) -> None:
        """Thread small and large queried Cube counts through both production graph builders."""
        for graph_builder, config_builder in (
            (build_forward_graph, build_forward_config), (build_backward_graph, build_backward_config),
        ):
            for cores in (4, 8, 20, 24, 28, 32, 36):
                with self.subTest(graph=graph_builder.__name__, cores=cores):
                    values = TaskSplitValue(tp=1, ep=2, seq_size=128, all_expert_num=4, top_k=2)
                    graph = graph_builder(values, hidden_size=512, intermediate_size=128, num_cube_cores=cores)
                    graph.propagate_splits(values)
                    cfg = config_builder(graph, values, 0, cores)
                    image = serialize_runtime_config(cfg)
                    buffer = ctypes.create_string_buffer(image)
                    offsets = (ctypes.c_uint32 * 12)()
                    self.reader.read_layout(buffer, offsets)
                    self.assertEqual(cfg.num_workers, 2 * cores, f"expected={2 * cores} actual={cfg.num_workers}")
                    end = self.reader.group_list_offset(buffer, cores - 1) + 16
                    self.assertLessEqual(end, offsets[9], f"expected end<={offsets[9]} actual={end}")

    def _check_group_list_isolation(self, experts: int) -> None:
        """Bound every worker's list at both cache-line widths and offset residues."""
        for tasks, events in ((1, 1024), (17, 1024), (25601, 1024), (17, 1040), (17, 2048)):
            cfg = allocate_runtime_config(tasks, events, experts)
            cfg.task_num = tasks
            image = serialize_runtime_config(cfg)
            buffer = ctypes.create_string_buffer(image)
            layout = (ctypes.c_uint32 * 12)()
            self.reader.read_layout(buffer, layout)
            scratch_begin = layout[7] + 16
            scratch_end = layout[9]
            for line_bytes in (64, 128):
                with self.subTest(experts=experts, tasks=tasks, events=events, line_bytes=line_bytes):
                    owned_lines = set()
                    for worker in range(NUM_WORKERS_CUBE):
                        begin = self.reader.group_list_offset(buffer, worker)
                        end = begin + experts * 8
                        self.assertEqual(begin % line_bytes, 0)
                        self.assertGreaterEqual(begin, scratch_begin)
                        self.assertLessEqual(end, scratch_end)
                        lines = set(range(begin // line_bytes, (end + line_bytes - 1) // line_bytes))
                        self.assertTrue(owned_lines.isdisjoint(lines))
                        owned_lines.update(lines)
            self.assertEqual(len(image), runtime_config_serialized_size(cfg.task_capacity, events, experts))

    def test_rejects_invalid_storage_bounds(self) -> None:
        """Reject insufficient storage and invalid capacities/queue counts."""
        cfg = allocate_runtime_config(25601, local_experts=33)
        cfg.task_num = 25601
        good = serialize_runtime_config(cfg)
        self.assertTrue(self.reader.valid_expert_runtime(ctypes.create_string_buffer(good), len(good), 4096, 33))
        self.assertFalse(self.reader.valid_expert_runtime(ctypes.create_string_buffer(good), len(good), 4096, 32))
        for size in (0, 8, 15, 16, 31, 63, len(good) - 1):
            with self.subTest(size=size):
                self.assertFalse(self.reader.valid_runtime(ctypes.create_string_buffer(good), size, 4096))
        counts_offset = RUNTIME_HEADER_BYTES + 1024 * 20 + cfg.task_capacity * TASK_DESC_SIZE
        for offset, value in ((8, 17), (8, 0xFFFFFFF0), (12, 1008), (12, 1025), (12, 0xFFFFFFF0),
                              (0, 0xFFFFFFFF), (counts_offset, 0xFFFFFFFF),
                              (counts_offset + 4, cfg.task_capacity + 1),
                              (type(cfg).dynamic_data.offset + 8, 65),
                              (type(cfg).dynamic_data.offset + 8, 0xFFFFFFFF)):
            with self.subTest(offset=offset, value=value):
                invalid = bytearray(good)
                struct.pack_into("<I", invalid, offset, value)
                self.assertFalse(self.reader.valid_runtime(
                    ctypes.create_string_buffer(bytes(invalid)), len(invalid), 4096,
                ))

    def test_header_padding_has_no_format_semantics(self) -> None:
        """The reader uses capacities directly and does not inspect format tags."""
        cfg = allocate_runtime_config(17)
        cfg.task_num = 17
        data = bytearray(serialize_runtime_config(cfg))
        data[40:RUNTIME_HEADER_BYTES] = bytes([255]) * (RUNTIME_HEADER_BYTES - 40)
        self.assertTrue(self.reader.valid_runtime(ctypes.create_string_buffer(bytes(data)), len(data), 4096))
        struct.pack_into("<I", data, 36, 2)
        self.assertFalse(self.reader.valid_runtime(ctypes.create_string_buffer(bytes(data)), len(data), 4096))

    def test_ready_header_and_persistent_storage_bounds(self) -> None:
        """Read ready metadata without tags and bound the persistent tail."""
        for ep_size, experts, events, mode in ((2, 4, 1024, "push"), (2, 4, 1024, "pull"),
                                                (64, 1024, 1104, "push"), (64, 1024, 1104, "pull")):
            with self.subTest(ep_size=ep_size):
                cfg = allocate_runtime_config(17, events)
                cfg.task_num = 17
                values = TaskSplitValue(tp=1, ep=ep_size, seq_size=128, all_expert_num=experts, top_k=2,
                                       dispatch_mode=mode)
                configure_ready_handshake(cfg, values)
                cfg.diagnostic_mode = 1
                cfg.aic_profile_record_capacity, cfg.aiv_profile_record_capacity = 101, 203
                data = serialize_runtime_config(cfg)
                buffer = ctypes.create_string_buffer(data)
                event_bytes = event_workspace_bytes(ep_size, experts)
                fields = (ctypes.c_uint32 * 5)()
                self.reader.read_protocol_profile(buffer, fields)
                self.assertEqual(list(fields), [cfg.ready_event, cfg.completion_event, 1, 101, 203])
                self.assertEqual(cfg.protocol_version, int(mode == "pull"))
                self.assertEqual(cfg.completion_event > 0, mode == "pull")
                self.assertEqual(self.reader.read_ready_event(buffer), cfg.ready_event)
                self.assertTrue(self.reader.valid_ready_runtime(buffer, len(data), event_bytes, ep_size))
                self.assertFalse(self.reader.valid_ready_runtime(buffer, len(data), event_bytes - 1, ep_size))
                self.assertFalse(self.reader.valid_ready_runtime(buffer, len(data), event_bytes, 1))
                for invalid_event in (events - 7, 0xFFFFFFFF):
                    invalid = bytearray(data)
                    struct.pack_into("<I", invalid, 16, invalid_event)
                    self.assertFalse(self.reader.valid_ready_runtime(
                        ctypes.create_string_buffer(bytes(invalid)), len(data), event_bytes, ep_size))
