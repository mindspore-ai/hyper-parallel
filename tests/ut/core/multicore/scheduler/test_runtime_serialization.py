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
from hyper_parallel.core.multicore.modules.mega_moe.backward.gen_runtime_data import (
    build_config_for_rank as build_backward_config,
)
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
    MAX_EXPERT_NUM_PER_RANK,
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_both_transports_support_asymmetric_communication_splits(self) -> None:
        """Feature: both transports support asymmetric communication splits.

        Description: Generate forward and backward graphs with all three communication split pairs.
        Expectation: Keep every GET/PUT scheduled exactly once with mode-specific completion.
        """
        for mode in ("push", "pull"):
            for dispatch_split, combine_split in ((128, 128), (128, 512), (1024, 1024)):
                for graph_builder, config_builder in (
                    (build_forward_graph, build_forward_config), (build_backward_graph, build_backward_config),
                ):
                    with self.subTest(mode=mode, splits=(dispatch_split, combine_split), graph=graph_builder.__name__):
                        values = TaskSplitValue(tp=1, ep=4, seq_size=4096, all_expert_num=48, top_k=8,
                                               dispatch_mode=mode)
                        graph = graph_builder(values, hidden_size=5120, intermediate_size=1792, num_cube_cores=20,
                                              dispatch_sv=dispatch_split, combine_sv=combine_split)
                        graph.propagate_splits(values)
                        cfg = config_builder(graph, values, 3, 20)
                        queues = [*cfg.cube_task_indices[:cfg.task_index_num[0]],
                                  *cfg.vector_task_indices[:cfg.task_index_num[1]]]
                        communication = [index for index in range(cfg.task_num) if cfg.all_tasks[index].task_type in
                                         (TaskType.TASK_SHMEM_GET_MEM, TaskType.TASK_SHMEM_PUT_MEM_SIGNAL)]
                        self.assertTrue(communication)
                        self.assertTrue(all(queues.count(index) == 1 for index in communication))
                        gets = sum(cfg.all_tasks[index].task_type == TaskType.TASK_SHMEM_GET_MEM
                                   for index in communication)
                        self.assertEqual(gets > 0, mode == "pull")
                        self.assertEqual(cfg.protocol_version, int(mode == "pull"))
                        self.assertEqual(cfg.completion_event > 0, mode == "pull")
                        self.assertTrue(serialize_runtime_config(cfg))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_real_graphs_expand_event_storage(self) -> None:
        """Feature: real graphs expand event storage.

        Description: Serialize forward and backward graphs for 16 experts per rank on EP64.
        Expectation: Expand events for a valid topology with 16 experts per rank.
        """
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
            # CI unpacks tests separately from the installed package and its headers.
            include_root = Path(hyper_parallel.__file__).resolve().parent.parent
            subprocess.run([
                compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                "-I", str(include_root), str(Path(__file__).with_name("runtime_reader.cpp")), "-o", str(library),
            ], check=True)
            cls.reader = ctypes.CDLL(str(library))
        cls.reader.valid_runtime.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
        cls.reader.valid_runtime.restype = ctypes.c_bool
        cls.reader.valid_ready_runtime.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32]
        cls.reader.valid_ready_runtime.restype = ctypes.c_bool
        cls.reader.read_ready_event.argtypes = [ctypes.c_void_p]
        cls.reader.read_ready_event.restype = ctypes.c_uint32
        cls.reader.read_protocol_profile.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls.reader.read_task.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        cls.reader.read_layout.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls.reader.group_list_offset.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        cls.reader.group_list_offset.restype = ctypes.c_uint32

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
        for tasks, events in ((1, 1024), (17, 1024), (25601, 1024), (17, 2048)):
            with self.subTest(tasks=tasks, events=events):
                cfg = allocate_runtime_config(tasks, events)
                cfg.task_num = tasks
                self._round_trip(cfg)

    def test_group_lists_own_complete_cache_lines_within_existing_storage(self) -> None:
        """Prevent adjacent workers' full-line writebacks from corrupting expert counts."""
        for tasks, events in ((1, 1024), (17, 1024), (25601, 1024), (17, 1040), (17, 2048)):
            cfg = allocate_runtime_config(tasks, events)
            cfg.task_num = tasks
            image = serialize_runtime_config(cfg)
            buffer = ctypes.create_string_buffer(image)
            layout = (ctypes.c_uint32 * 12)()
            self.reader.read_layout(buffer, layout)
            scratch_begin = layout[7] + 16
            scratch_end = layout[9]
            for line_bytes in (64, 128):
                with self.subTest(tasks=tasks, events=events, line_bytes=line_bytes):
                    owned_lines = set()
                    for worker in range(NUM_WORKERS_CUBE):
                        begin = self.reader.group_list_offset(buffer, worker)
                        end = begin + MAX_EXPERT_NUM_PER_RANK * 8
                        self.assertEqual(begin % line_bytes, 0)
                        self.assertGreaterEqual(begin, scratch_begin)
                        self.assertLessEqual(end, scratch_end)
                        lines = set(range(begin // line_bytes, (end + line_bytes - 1) // line_bytes))
                        self.assertTrue(owned_lines.isdisjoint(lines))
                        owned_lines.update(lines)
            self.assertEqual(len(image), runtime_config_serialized_size(cfg.task_capacity, events))

    def test_rejects_invalid_storage_bounds(self) -> None:
        """Reject insufficient storage and invalid capacities/queue counts."""
        cfg = allocate_runtime_config(25601)
        cfg.task_num = 25601
        good = serialize_runtime_config(cfg)
        for size in (0, 8, 15, 16, 31, 63, len(good) - 1):
            with self.subTest(size=size):
                self.assertFalse(self.reader.valid_runtime(ctypes.create_string_buffer(good), size, 4096))
        counts_offset = RUNTIME_HEADER_BYTES + 1024 * 20 + cfg.task_capacity * TASK_DESC_SIZE
        for offset, value in ((8, 17), (8, 0xFFFFFFF0), (12, 1008), (12, 1025), (12, 0xFFFFFFF0),
                              (0, 0xFFFFFFFF), (counts_offset, 0xFFFFFFFF),
                              (counts_offset + 4, cfg.task_capacity + 1)):
            with self.subTest(offset=offset, value=value):
                invalid = bytearray(good)
                struct.pack_into("<I", invalid, offset, value)
                self.assertFalse(self.reader.valid_runtime(ctypes.create_string_buffer(bytes(invalid)),
                                                         len(invalid), 4096))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_pull_completion_and_profiler_fields_do_not_overlap(self) -> None:
        """Feature: pull completion and profiler fields do not overlap.

        Description: Decode instrumented push and pull headers through the production C++ reader.
        Expectation: Round-trip both protocols through production C++ with profiler enabled.
        """
        for mode in ("push", "pull"):
            values = TaskSplitValue(tp=1, ep=4, seq_size=128, all_expert_num=8, top_k=2, dispatch_mode=mode)
            cfg = allocate_runtime_config(16)
            cfg.num_workers = 40
            configure_ready_handshake(cfg, values)
            cfg.cycle_profiling_enabled = 1
            cfg.aic_profile_record_capacity = 101
            cfg.aiv_profile_record_capacity = 203
            image = serialize_runtime_config(cfg)
            buffer = ctypes.create_string_buffer(image)
            fields = (ctypes.c_uint32 * 5)()
            self.reader.read_protocol_profile(buffer, fields)
            self.assertEqual(list(fields), [cfg.ready_event, cfg.completion_event, 1, 101, 203])
            self.assertTrue(self.reader.valid_ready_runtime(buffer, len(image), event_workspace_bytes(4, 8), 4))
            invalid = bytearray(image)
            struct.pack_into("<I", invalid, 36, 2)
            self.assertFalse(self.reader.valid_ready_runtime(ctypes.create_string_buffer(bytes(invalid)),
                                                           len(image), event_workspace_bytes(4, 8), 4))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_header_padding_has_no_format_semantics(self) -> None:
        """Feature: header padding has no format semantics.

        Description: Fill only unused header padding with nonzero bytes.
        Expectation: The reader uses capacities directly and does not inspect format tags.
        """
        cfg = allocate_runtime_config(17)
        cfg.task_num = 17
        data = bytearray(serialize_runtime_config(cfg))
        data[40:RUNTIME_HEADER_BYTES] = bytes([255]) * (RUNTIME_HEADER_BYTES - 40)
        self.assertTrue(self.reader.valid_runtime(ctypes.create_string_buffer(bytes(data)), len(data), 4096))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_ready_header_and_persistent_storage_bounds(self) -> None:
        """Feature: ready header and persistent storage bounds.

        Description: Decode valid and truncated persistent event arenas for small and large EP groups.
        Expectation: Read ready metadata without tags and bound the persistent tail.
        """
        for ep_size, experts, events in ((2, 4, 1024), (64, 1024, 1104)):
            with self.subTest(ep_size=ep_size):
                cfg = allocate_runtime_config(17, events)
                cfg.task_num = 17
                values = TaskSplitValue(tp=1, ep=ep_size, seq_size=128, all_expert_num=experts, top_k=2)
                configure_ready_handshake(cfg, values)
                data = serialize_runtime_config(cfg)
                buffer = ctypes.create_string_buffer(data)
                event_bytes = event_workspace_bytes(ep_size, experts)
                self.assertEqual(self.reader.read_ready_event(buffer), cfg.ready_event)
                self.assertTrue(self.reader.valid_ready_runtime(buffer, len(data), event_bytes, ep_size))
                self.assertFalse(self.reader.valid_ready_runtime(buffer, len(data), event_bytes - 1, ep_size))
                self.assertFalse(self.reader.valid_ready_runtime(buffer, len(data), event_bytes, 1))
                for invalid_event in (events - 7, 0xFFFFFFFF):
                    invalid = bytearray(data)
                    struct.pack_into("<I", invalid, 16, invalid_event)
                    self.assertFalse(self.reader.valid_ready_runtime(
                        ctypes.create_string_buffer(bytes(invalid)), len(data), event_bytes, ep_size))
