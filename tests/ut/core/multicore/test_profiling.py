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
"""Unit tests for MegaKernel Host profiling metadata and buffer planning."""

import struct
import unittest

from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import build_backward_graph
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import build_forward_graph
from hyper_parallel.core.multicore.modules.mega_moe.profiling import (
    MEGA_MOE_PROFILE_OWNER_LABEL,
    _configure_mega_moe_profile_metadata,
)
from hyper_parallel.core.multicore.profiler.profiling import (
    GRAPH_STAGE_DESC_BASE,
    MAX_PROFILE_BUFFER_BYTES,
    _ProfileSpec,
    _apply_mega_kernel_profile_graph,
    _calculate_profile_layout,
    _get_mega_kernel_profile_metadata,
    _prepare_mega_kernel_runtime_config,
    _resolve_cycle_frequency_mhz,
    _set_mega_kernel_profile_metadata,
)
from hyper_parallel.core.multicore.scheduler.config import (
    INVALID_PROFILE_OWNER_ID,
    RuntimeConfigC,
    TaskDescC,
    TaskSplitValue,
    TaskType,
)
from hyper_parallel.core.multicore.scheduler.graph import (
    ComputeGraph,
    OperatorNode,
    OpType,
    SplitSpec,
)
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config


def _runtime_config(task_capacity: int = 16) -> RuntimeConfigC:
    """Allocate a graph-sized RuntimeConfig for profiling unit tests."""
    return allocate_runtime_config(task_capacity)


def _three_record_task() -> TaskDescC:
    task = TaskDescC()
    task.task_type = TaskType.TASK_GROUPED_MATMUL
    task.dependent_event = 0
    return task


def _add_profile_op(
    graph: ComputeGraph,
    name: str,
    *,
    task_num: int,
    diagnostic_name: str | None = None,
) -> OperatorNode:
    """Append a minimal graph node for profile metadata tests."""
    operator = OperatorNode(
        name=name,
        op_type=OpType.GMM,
        inputs=[],
        outputs=[],
        param_positions=[],
        split_value=1,
        split_spec=SplitSpec(split_inputs=None, task_num_fn=lambda _: task_num),
        tiling_position=-1,
        fill_config=None,
        diagnostic_name=diagnostic_name,
    )
    operator.task_num = task_num
    graph.add_op(operator)
    return operator


class TestMegaKernelProfilingMetadata(unittest.TestCase):
    """Validate graph-driven stage and owner resolution."""

    def test_graph_serializes_stage_names_and_owner_ids(self):
        """Use graph task ranges and prefer an explicit diagnostic name."""
        graph = ComputeGraph()
        _add_profile_op(graph, "first_stage", task_num=1, diagnostic_name="First")
        _add_profile_op(graph, "second_stage", task_num=2)
        runtime_config = _runtime_config()
        runtime_config.task_num = 3
        for task_id in range(3):
            runtime_config.all_tasks[task_id].task_index = task_id

        def resolve_owner(_, task_desc, context):
            return context + task_desc.task_index

        _apply_mega_kernel_profile_graph(
            runtime_config,
            graph,
            _ProfileSpec("GraphKernel", "Shard", resolve_owner),
            context=10,
        )

        self.assertEqual(runtime_config.all_tasks[0].profile_desc_id, GRAPH_STAGE_DESC_BASE)
        self.assertEqual(runtime_config.all_tasks[1].profile_desc_id, GRAPH_STAGE_DESC_BASE + 1)
        self.assertEqual(runtime_config.all_tasks[2].profile_desc_id, GRAPH_STAGE_DESC_BASE + 1)
        self.assertEqual(runtime_config.all_tasks[2].profile_owner_id, 12)
        metadata = _get_mega_kernel_profile_metadata(runtime_config)
        self.assertEqual(metadata.stage_names[GRAPH_STAGE_DESC_BASE], "First")
        self.assertEqual(metadata.stage_names[GRAPH_STAGE_DESC_BASE + 1], "SecondStage")
        self.assertEqual(metadata.task_stage_names, {0: "First", 1: "SecondStage", 2: "SecondStage"})

    def test_mega_moe_uses_graph_names_and_expert_owners(self):
        """Keep MegaMoe display semantics without a second task-matching table."""
        topology = TaskSplitValue()
        topology.rank_id = 1
        graph = build_forward_graph(topology)
        for operator in graph.topological_sort():
            operator.task_num = 1
        runtime_config = _runtime_config()
        runtime_config.task_num = 5

        dispatch = TaskDescC()
        dispatch.task_type = TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
        dispatch.task_split_num = 64
        dispatch.task_index = 4
        runtime_config.all_tasks[0] = dispatch

        gmm1 = TaskDescC()
        gmm1.task_type = TaskType.TASK_GROUPED_MATMUL
        gmm1.task_index = 48
        runtime_config.all_tasks[1] = gmm1

        swiglu = TaskDescC()
        swiglu.task_type = TaskType.TASK_SWI_GLU
        swiglu.task_split_value = 128
        swiglu.task_index = 128
        runtime_config.all_tasks[2] = swiglu

        gmm2 = TaskDescC()
        gmm2.task_type = TaskType.TASK_GROUPED_MATMUL
        gmm2.task_index = 72
        runtime_config.all_tasks[3] = gmm2

        combine = TaskDescC()
        combine.task_type = TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
        combine.task_split_num = 64
        combine.task_index = 4
        runtime_config.all_tasks[4] = combine

        _configure_mega_moe_profile_metadata(
            runtime_config,
            graph,
            topology,
            num_cube_cores=24,
            is_backward=False,
        )

        metadata = _get_mega_kernel_profile_metadata(runtime_config)
        self.assertEqual(
            metadata.task_stage_names,
            {0: "Dispatch", 1: "GMM1", 2: "SwiGLU", 3: "GMM2", 4: "Combine"},
        )
        self.assertEqual(runtime_config.all_tasks[0].profile_owner_id, 2)
        self.assertEqual(runtime_config.all_tasks[1].profile_owner_id, 10)
        self.assertEqual(runtime_config.all_tasks[2].profile_owner_id, 10)
        self.assertEqual(runtime_config.all_tasks[3].profile_owner_id, 11)
        self.assertEqual(runtime_config.all_tasks[4].profile_owner_id, 10)

    def test_forward_and_backward_metadata_are_selected_internally(self):
        """Bind direction-specific graph names without user-provided mappings."""
        topology = TaskSplitValue()
        forward_graph = build_forward_graph(topology)
        backward_graph = build_backward_graph(topology)
        forward = _runtime_config()
        backward = _runtime_config()
        _configure_mega_moe_profile_metadata(
            forward, forward_graph, topology, num_cube_cores=24, is_backward=False
        )
        _configure_mega_moe_profile_metadata(
            backward, backward_graph, topology, num_cube_cores=24, is_backward=True
        )

        forward_runtime = _prepare_mega_kernel_runtime_config(
            forward,
            tensor_factory=bytes,
            profile_tensor_factory=lambda tensor: tensor,
            rank=0,
            device_id=0,
        )
        backward_runtime = _prepare_mega_kernel_runtime_config(
            backward,
            tensor_factory=bytes,
            profile_tensor_factory=lambda tensor: tensor,
            rank=0,
            device_id=0,
        )

        self.assertEqual(forward_runtime.kernel_name, "MegaMoe")
        self.assertEqual(backward_runtime.kernel_name, "MegaMoeGrad")
        self.assertEqual(MEGA_MOE_PROFILE_OWNER_LABEL, "Expert")
        forward_names = _get_mega_kernel_profile_metadata(forward).stage_names.values()
        backward_names = _get_mega_kernel_profile_metadata(backward).stage_names.values()
        self.assertIn("GMM1", forward_names)
        self.assertIn("ActGrad", backward_names)

    def test_duplicate_graph_names_use_distinct_graph_stage_ids(self):
        """Allow repeated display names because graph nodes still have distinct stage IDs."""
        graph = ComputeGraph()
        _add_profile_op(graph, "first", task_num=1, diagnostic_name="Duplicate")
        _add_profile_op(graph, "second", task_num=1, diagnostic_name="Duplicate")
        runtime_config = _runtime_config()
        runtime_config.task_num = 2
        runtime_config.all_tasks[0] = _three_record_task()
        runtime_config.all_tasks[1] = _three_record_task()

        _apply_mega_kernel_profile_graph(
            runtime_config,
            graph,
            _ProfileSpec("RepeatedStageKernel", "Owner"),
        )

        metadata = _get_mega_kernel_profile_metadata(runtime_config)
        self.assertEqual(metadata.stage_names[GRAPH_STAGE_DESC_BASE], "Duplicate")
        self.assertEqual(metadata.stage_names[GRAPH_STAGE_DESC_BASE + 1], "Duplicate")
        self.assertEqual(runtime_config.all_tasks[0].profile_desc_id, GRAPH_STAGE_DESC_BASE)
        self.assertEqual(runtime_config.all_tasks[1].profile_desc_id, GRAPH_STAGE_DESC_BASE + 1)

    def test_owner_resolver_can_leave_a_stage_without_owner(self):
        """Keep owner optional for kernels whose graph has both owned and unowned stages."""
        graph = ComputeGraph()
        _add_profile_op(graph, "owned", task_num=1)
        _add_profile_op(graph, "unowned", task_num=1)
        runtime_config = _runtime_config()
        runtime_config.task_num = 2

        _apply_mega_kernel_profile_graph(
            runtime_config,
            graph,
            _ProfileSpec(
                "MixedOwnerKernel",
                "Shard",
                lambda operator, *_: 3 if operator.name == "owned" else None,
            ),
        )

        self.assertEqual(runtime_config.all_tasks[0].profile_owner_id, 3)
        self.assertEqual(runtime_config.all_tasks[1].profile_owner_id, INVALID_PROFILE_OWNER_ID)

    def test_sparse_scheduled_terminate_is_configured_without_touching_gaps(self):
        """Recognize a queue-only terminate task while preserving unrelated slots."""
        runtime_config = _runtime_config()
        runtime_config.num_workers = 48
        runtime_config.all_tasks[1].profile_desc_id = 0xABCDEF
        terminate = TaskDescC()
        terminate.task_type = TaskType.TASK_TERMINATE
        runtime_config.all_tasks[7] = terminate
        runtime_config.task_index_num[0] = 1
        runtime_config.cube_task_indices[0] = 7

        _apply_mega_kernel_profile_graph(
            runtime_config,
            ComputeGraph(),
            _ProfileSpec("SparseKernel", "Owner"),
        )

        self.assertEqual(runtime_config.all_tasks[7].profile_desc_id, 0x10006)
        self.assertEqual(runtime_config.all_tasks[1].profile_desc_id, 0xABCDEF)

    def test_display_metadata_does_not_change_serialized_runtime_config(self):
        """Keep names on Host while serializing only numeric IDs."""
        runtime_config = _runtime_config()
        serialized_before = bytes(runtime_config)

        _set_mega_kernel_profile_metadata(
            runtime_config,
            kernel_name="MegaMoe",
            owner_label="Expert",
            stage_names={GRAPH_STAGE_DESC_BASE: "GMM1"},
        )

        self.assertEqual(bytes(runtime_config), serialized_before)


class TestMegaKernelProfileLayout(unittest.TestCase):
    """Validate exact bounded AIC/AIV buffer sizing."""

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

    def test_layout_caps_each_worker_at_256_records(self):
        """Bound Device memory when a long schedule needs more records."""
        runtime_config = _runtime_config(2048)
        runtime_config.num_workers = 48
        runtime_config.all_tasks[0] = _three_record_task()
        runtime_config.task_index_num[0] = 2041

        layout = _calculate_profile_layout(runtime_config)

        self.assertEqual(layout.aic_required_records, 258)
        self.assertEqual(layout.aic_record_capacity, 256)

    def test_prepared_runtime_serializes_disabled_config_and_lazily_profiles(self):
        """Keep the normal tensor disabled and create the enabled tensor on demand."""
        runtime_config = _runtime_config()
        runtime_config.num_workers = 48
        created_profile_tensors = []

        def profile_tensor_factory(tensor):
            enabled = bytearray(tensor)
            struct.pack_into(
                "<I",
                enabled,
                RuntimeConfigC.cycle_profiling_enabled.offset,
                1,
            )
            created_profile_tensors.append(bytes(enabled))
            return created_profile_tensors[-1]

        runtime = _prepare_mega_kernel_runtime_config(
            runtime_config,
            tensor_factory=bytes,
            profile_tensor_factory=profile_tensor_factory,
            rank=3,
            device_id=7,
        )

        disabled = struct.unpack_from(
            "<I",
            runtime.normal_tensor,
            RuntimeConfigC.cycle_profiling_enabled.offset,
        )[0]
        self.assertEqual(disabled, 0)
        self.assertEqual(created_profile_tensors, [])
        enabled = struct.unpack_from(
            "<I",
            runtime.profile_tensor,
            RuntimeConfigC.cycle_profiling_enabled.offset,
        )[0]
        self.assertEqual(enabled, 1)
        self.assertEqual(len(created_profile_tensors), 1)
        self.assertIs(runtime.profile_tensor, created_profile_tensors[0])
        self.assertEqual(runtime.rank, 3)
        self.assertEqual(runtime.device_id, 7)

    def test_soc_aliases_use_the_documented_counter_frequency(self):
        """Recognize current 910B and 910C/A3 Torch NPU names."""
        for soc_name in ("Ascend910B4", "Ascend910C1", "Ascend910_9372"):
            with self.subTest(soc_name=soc_name):
                self.assertEqual(_resolve_cycle_frequency_mhz(soc_name), 50.0)


if __name__ == "__main__":
    unittest.main()
