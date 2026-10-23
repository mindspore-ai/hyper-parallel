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
"""Shared graph identity and independent profiler runtime tests."""

from __future__ import annotations

import struct
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

from hyper_parallel.core.multicore.diagnostics.common.graph import DiagnosticSpec, build_graph_metadata
from hyper_parallel.core.multicore.diagnostics.common.runtime import PreparedDiagnosticRuntime
from hyper_parallel.core.multicore.diagnostics.profiler.cycle_trace import (
    GRAPH_STAGE_DESC_BASE, MAX_PROFILE_BUFFER_BYTES, _PreparedProfileRuntime,
    _calculate_profile_layout, _configure_trace_metadata, _resolve_cycle_frequency_mhz,
)
from hyper_parallel.core.multicore.scheduler.config import EVENT_INVALID_ID, RuntimeConfigC, TaskDescC, TaskType
from hyper_parallel.core.multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, SplitSpec
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config
from hyper_parallel.core.multicore.diagnostics import prepare_mega_kernel_diagnostics
from hyper_parallel.core.multicore.modules.mega_gate.diagnostics import (
    MEGA_GATE_ROUTE_STAGE_NAMES, MEGA_GATE_ROUTE_GRAD_STAGE_NAMES, MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES,
    bind_pipeline_graph,
)
from hyper_parallel.core.multicore.modules.mega_gate.plan import (
    MEGA_GATE_AIV_WORKER_CAPACITY, MEGA_GATE_ROUTE_TASK_TYPES, MEGA_GATE_ROUTE_GRAD_TASK_TYPES,
    MEGA_GATE_ROUTE_GRAD_K1_TASK_TYPES, _build_pipeline_runtime_config, build_mega_gate_plan,
)
from tests.common.mark_utils import arg_mark


def _runtime_config(task_capacity: int = 16) -> RuntimeConfigC:
    """Allocate an actual graph-sized Host configuration."""
    return allocate_runtime_config(task_capacity)


def _three_record_task() -> TaskDescC:
    """Create a task with a dependency and completion event."""
    task = TaskDescC()
    task.task_type = TaskType.TASK_GROUPED_MATMUL
    task.dependent_event = 0
    return task


def _graph():
    """Create generic stage identity without any expert semantics."""
    graph = ComputeGraph()
    for name, count in (("first", 1), ("second", 2)):
        operator = OperatorNode(name, OpType.SWIGLU, [], [], [], 1,
                                SplitSpec(None, lambda _: 1), -1, None)
        operator.task_num = count
        graph.add_op(operator)
    return graph


class TestDiagnosticMetadata(unittest.TestCase):
    """Distinguish Host identity from profiling-specific serialized IDs."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_graph_metadata_is_read_only(self):
        """Feature: MegaKernel diagnostics.
        Description: Resolving task identity must never annotate the base TaskDesc.
        Expectation: Graph metadata is read only.
        """
        config = _runtime_config()
        original = bytes(config)
        metadata = build_graph_metadata(_graph(), config, DiagnosticSpec("Custom", "Shard", lambda *_: 7))
        self.assertEqual(metadata.stage_names, {0: "First", 1: "Second"})
        self.assertEqual(metadata.task_stage_ids, {0: 0, 1: 1, 2: 1})
        self.assertEqual(metadata.task_owner_ids, {0: 7, 1: 7, 2: 7})
        self.assertEqual(bytes(config), original)
        with self.assertRaises(TypeError):
            metadata.stage_names[0] = "changed"

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_repeated_names_have_distinct_stage_ids(self):
        """Feature: MegaKernel diagnostics.
        Description: Profiling supports repeated display names without conflating task identity.
        Expectation: Repeated names have distinct stage ids.
        """
        graph = _graph()
        for operator in graph.topological_sort():
            operator.diagnostic_name = "Repeated"
        metadata = build_graph_metadata(graph, _runtime_config(), DiagnosticSpec("Custom"))
        self.assertEqual(metadata.stage_names, {0: "Repeated", 1: "Repeated"})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_optional_and_invalid_owners(self):
        """Feature: MegaKernel diagnostics.
        Description: Allow missing owners but reject IDs outside the serializable uint32 range.
        Expectation: Optional and invalid owners.
        """
        graph, config = _graph(), _runtime_config()
        self.assertIsNone(build_graph_metadata(graph, config, DiagnosticSpec("Custom")).task_owner_ids[0])
        for owner in (-1, True, 0xFFFFFFFF, "owner"):
            with self.subTest(owner=owner), self.assertRaisesRegex(ValueError, "owner_id"):
                spec = DiagnosticSpec("Custom", owner_resolver=lambda *_, value=owner: value)
                build_graph_metadata(graph, config, spec)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_trace_encoding_uses_a_private_configuration(self):
        """Feature: MegaKernel diagnostics.
        Description: Only profiling owns profile_desc_id and profile_owner_id.
        Expectation: Trace encoding uses a private configuration.
        """
        config = _runtime_config()
        config.task_num = 3
        metadata = build_graph_metadata(_graph(), config, DiagnosticSpec("Custom", "Shard", lambda *_: 7))
        original = bytes(config)
        variant = type(config).from_buffer_copy(original)
        trace = _configure_trace_metadata(variant, metadata)
        self.assertEqual(variant.all_tasks[0].profile_desc_id, GRAPH_STAGE_DESC_BASE)
        self.assertEqual(variant.all_tasks[2].profile_desc_id, GRAPH_STAGE_DESC_BASE + 1)
        self.assertEqual(variant.all_tasks[2].profile_owner_id, 7)
        self.assertEqual(trace.task_stage_names[2], "Second")
        self.assertEqual(bytes(config), original)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sparse_termination_preserves_unrelated_slots(self):
        """Feature: MegaKernel diagnostics.
        Description: Queue-only system tasks do not cause annotation of unused task slots.
        Expectation: Sparse termination preserves unrelated slots.
        """
        config = _runtime_config()
        config.task_index_num[0] = 1
        config.cube_task_indices[0] = 7
        config.all_tasks[7].task_type = TaskType.TASK_TERMINATE
        config.all_tasks[1].profile_desc_id = 0xABCDEF
        metadata = build_graph_metadata(ComputeGraph(), config, DiagnosticSpec("Sparse"))
        _configure_trace_metadata(config, metadata)
        self.assertEqual(config.all_tasks[7].profile_desc_id, 0x10006)
        self.assertEqual(config.all_tasks[1].profile_desc_id, 0xABCDEF)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_only_selected_profiler_prepares_a_trace_image(self):
        """Feature: MegaKernel diagnostics.
        Description: Normal preparation produces exactly one normal image and no trace state.
        Expectation: Only selected profiler prepares a trace image.
        """
        config = _runtime_config()
        config.num_workers = 2
        created = []
        def tensor_factory(data: bytes) -> bytes:
            """Count serialized images without creating Device resources.

            Args:
                data: Serialized runtime configuration bytes.
            """
            created.append(data)
            return data
        runtime = PreparedDiagnosticRuntime(ComputeGraph(), config, DiagnosticSpec("Custom"),
                                            parameter_count=0, tensor_factory=tensor_factory, rank=3, device_id=7)
        self.assertEqual(len(created), 1)
        self.assertIsNone(runtime._metadata)
        original = bytes(config)
        profile = _PreparedProfileRuntime(runtime)
        self.assertEqual(len(created), 2)
        offset = RuntimeConfigC.diagnostic_mode.offset
        self.assertEqual(struct.unpack_from("<I", runtime.normal_tensor, offset)[0], 0)
        self.assertEqual(struct.unpack_from("<I", profile.profile_tensor, offset)[0], 1)
        self.assertEqual(bytes(config), original)
        self.assertEqual((profile.rank, profile.device_id), (3, 7))


class TestMegaKernelProfileLayout(unittest.TestCase):
    """Validate exact bounded AIC/AIV buffer sizing."""

    def test_layout_follows_runtime_worker_capacity(self):
        """Size every physical slot without limiting event-driven graphs to 910 cores."""
        for cube_cores in (20, 24, 32, 36):
            with self.subTest(cube_cores=cube_cores):
                config = allocate_runtime_config(128, num_workers=2 * cube_cores)
                config.task_num = 3 * cube_cores + 1
                for task_id in range(config.task_num):
                    config.all_tasks[task_id] = _three_record_task()
                config.task_index_num[0] = 2 * cube_cores + 1
                config.task_index_num[1] = cube_cores
                config.cube_task_indices[:2 * cube_cores + 1] = range(2 * cube_cores + 1)
                config.vector_task_indices[:cube_cores] = range(2 * cube_cores + 1, config.task_num)
                layout = _calculate_profile_layout(config)
                self.assertEqual(layout.aic_required_records, 9)
                self.assertEqual(layout.aiv_required_records, 3)
                slots = max(24, cube_cores)
                self.assertEqual(layout.cube_slot_count, slots)
                self.assertEqual(layout.buffer_size, 3 * slots * (64 + 16 * 32))
                graph = ComputeGraph()
                operator = OperatorNode("gmm", OpType.GMM, [], [], [], 1, SplitSpec(None, lambda _: 1), -1, None)
                operator.task_num = config.task_num
                graph.add_op(operator)
                runtime = _PreparedProfileRuntime(PreparedDiagnosticRuntime(
                    graph, config, DiagnosticSpec("Custom"), parameter_count=0, tensor_factory=bytes,
                    rank=0, device_id=0,
                ))
                self.assertEqual(runtime.buffer_size, layout.buffer_size)

    def test_layout_rounds_busiest_workers_to_sixteen_records(self):
        """Mirror Device round-robin assignment for AIC and AIV independently."""
        runtime_config = _runtime_config(128)
        runtime_config.num_workers = 48
        runtime_config.all_tasks[0] = _three_record_task()
        runtime_config.task_index_num[0] = 121
        runtime_config.task_index_num[1] = 24

        layout = _calculate_profile_layout(runtime_config)

        self.assertEqual(layout.aic_required_records, 18)
        self.assertEqual(layout.aiv_required_records, 3)
        self.assertEqual(layout.aic_record_capacity, 32)
        self.assertEqual(layout.aiv_record_capacity, 16)
        self.assertEqual(layout.buffer_size, 53760)
        self.assertLess(layout.buffer_size, MAX_PROFILE_BUFFER_BYTES)

    def test_layout_preserves_records_beyond_legacy_capacity(self):
        """Allocate the complete trace when a long schedule exceeds 256 records."""
        runtime_config = _runtime_config(2048)
        runtime_config.num_workers = 48
        runtime_config.all_tasks[0] = _three_record_task()
        runtime_config.task_index_num[0] = 2041

        layout = _calculate_profile_layout(runtime_config)

        self.assertEqual(layout.aic_required_records, 258)
        self.assertEqual(layout.aic_record_capacity, 272)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mega_gate_pipeline_runtime_configs(self):
        """Feature: MegaKernel diagnostics.
        Description: Validate forward, ordinary-grad and K=1 descriptor streams.
        Expectation: Mega gate pipeline runtime configs.
        """
        active_task_types = (*MEGA_GATE_ROUTE_TASK_TYPES, *MEGA_GATE_ROUTE_GRAD_TASK_TYPES)
        self.assertEqual([int(task_type) for task_type in active_task_types], list(range(120, 141)))
        cases = (
            (MEGA_GATE_ROUTE_TASK_TYPES, MEGA_GATE_ROUTE_STAGE_NAMES, "HyperMegaGateRoute"),
            (MEGA_GATE_ROUTE_GRAD_TASK_TYPES, MEGA_GATE_ROUTE_GRAD_STAGE_NAMES, "HyperMegaGateRouteGrad"),
            (MEGA_GATE_ROUTE_GRAD_K1_TASK_TYPES, MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES,
             "HyperMegaGateRouteGrad"),
        )
        for task_types, stage_names, kernel_name in cases:
            expected_count = len(task_types)
            with self.subTest(kernel_name=kernel_name, stage_count=expected_count):
                runtime_config = _build_pipeline_runtime_config(
                    task_types,
                    stage_names,
                    kernel_name=kernel_name,
                )
                adapter = prepare_mega_kernel_diagnostics(
                    None, runtime_config, kernel_name=kernel_name, parameter_count=11, device="cpu",
                    execution_mode="aiv_pipeline", vector_worker_stride=1,
                    graph_adapter=lambda graph, cfg, kernel_name=kernel_name: bind_pipeline_graph(
                        graph, cfg, backward=kernel_name.endswith("Grad")),
                )
                profile = _PreparedProfileRuntime(adapter._runtime)
                layout = profile._layout
                tasks = [runtime_config.all_tasks[task_id] for task_id in range(expected_count)]
                metadata = adapter.metadata

                self.assertEqual(runtime_config.task_num, expected_count)
                self.assertEqual(runtime_config.task_index_num[:2], [0, expected_count])
                self.assertEqual(runtime_config.num_workers, MEGA_GATE_AIV_WORKER_CAPACITY)
                self.assertEqual(layout.aic_required_records, 0)
                self.assertEqual(layout.aic_record_capacity, 0)
                self.assertEqual(layout.aiv_required_records, expected_count)
                self.assertEqual(layout.aiv_record_capacity, 16)
                self.assertEqual([task.task_type for task in tasks], list(task_types))
                self.assertEqual([task.tiling_data_offset for task in tasks], list(range(expected_count)))
                self.assertTrue(all(task.trigger_event == EVENT_INVALID_ID for task in tasks))
                self.assertTrue(all(task.dependent_event == EVENT_INVALID_ID for task in tasks))
                self.assertEqual(
                    [metadata.task_stage_names[task_id] for task_id in range(expected_count)],
                    list(stage_names),
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mega_gate_plan_uses_the_distributed_rank(self):
        """Feature: MegaKernel diagnostics.
        Description: Use the global process rank for cycle-trace ownership.
        Expectation: Mega gate plan uses the distributed rank.
        """
        prepared_ranks = []
        for initialized, expected_rank in ((False, 0), (True, 3)):
            with self.subTest(initialized=initialized):
                prepared_ranks.clear()

                def prepare_runtime(*args: Any, **kwargs: Any) -> object:
                    """Record the rank passed to each prepared runtime."""
                    del args
                    prepared_ranks.append(kwargs["rank"])
                    return object()

                with (
                    patch(
                        "hyper_parallel.core.multicore.modules.mega_gate.plan.dist.is_initialized",
                        return_value=initialized,
                    ),
                    patch(
                        "hyper_parallel.core.multicore.modules.mega_gate.plan.dist.get_rank",
                        return_value=3,
                    ) as get_rank,
                    patch(
                        "hyper_parallel.core.multicore.modules.mega_gate.plan.prepare_mega_kernel_diagnostics",
                        side_effect=prepare_runtime,
                    ),
                    patch(
                        "hyper_parallel.core.multicore.modules.mega_gate.plan.torch.zeros",
                        return_value=object(),
                    ),
                ):
                    build_mega_gate_plan(SimpleNamespace(type="npu", index=0))

                self.assertEqual(prepared_ranks, [expected_rank] * 3)
                if initialized:
                    get_rank.assert_called_once_with()
                else:
                    get_rank.assert_not_called()

    def test_soc_aliases_use_the_documented_counter_frequency(self):
        """Recognize A2/A3 and both A5 families using their documented system clocks."""
        cases = (("Ascend910B4", 50.0), ("Ascend910C1", 50.0), ("Ascend910_9372", 50.0),
                 ("Ascend950PR_9589", 1000.0), ("Ascend950PR_9599", 1000.0),
                 ("Ascend950DT", 1000.0), ("950PR_9589", 1000.0), (" ascend950dt ", 1000.0))
        for soc_name, expected in cases:
            with self.subTest(soc_name=soc_name):
                actual = _resolve_cycle_frequency_mhz(soc_name)
                self.assertEqual(actual, expected, f"soc={soc_name} expected={expected} actual={actual}")

    def test_950_submodels_share_the_family_counter_frequency(self) -> None:
        """PR/DT suffixes must not require individual entries in the profiler table."""
        families = (("Ascend950PR", ("_950z", "_9579", "_957bx", "_9589", "_9599")),
                    ("Ascend950DT", ("_950x", "_9571", "_9581x", "_9588", "_9596", "_95A2")))
        for family, suffixes in families:
            for suffix in suffixes:
                soc_name = family + suffix
                with self.subTest(soc_name=soc_name):
                    actual = _resolve_cycle_frequency_mhz(soc_name)
                    self.assertEqual(actual, 1000.0, f"soc={soc_name} expected=1000.0 actual={actual}")

    def test_unknown_soc_never_inherits_a_supported_counter_frequency(self):
        """Unknown hardware must not silently produce a wrongly scaled trace."""
        with self.assertRaisesRegex(RuntimeError, "Unsupported Ascend SoC"):
            _resolve_cycle_frequency_mhz("Ascend960")


if __name__ == "__main__":
    unittest.main()
