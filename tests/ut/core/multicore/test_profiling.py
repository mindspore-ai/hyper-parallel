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

from hyper_parallel.core.multicore.modules.mega_moe.profiling import (
    MEGA_MOE_GRAD_PROFILE_STAGE_NAMES,
    MEGA_MOE_PROFILE_OWNER_LABEL,
    MEGA_MOE_PROFILE_STAGE_NAMES,
    _configure_mega_moe_profile_metadata,
)
from hyper_parallel.core.multicore.profiling import (
    MAX_PROFILE_BUFFER_BYTES,
    _ProfileSpec,
    _ProfileStage,
    _apply_mega_kernel_profile_spec,
    _calculate_profile_layout,
    _get_mega_kernel_profile_metadata,
    _prepare_mega_kernel_runtime_config,
    _resolve_cycle_frequency_mhz,
    _set_mega_kernel_profile_metadata,
)
from hyper_parallel.core.multicore.scheduler.config import (
    INVALID_PROFILE_DESC_ID,
    INVALID_PROFILE_OWNER_ID,
    RuntimeConfigC,
    TaskDescC,
    TaskSplitValue,
    TaskType,
)


def _three_record_task() -> TaskDescC:
    task = TaskDescC()
    task.task_type = TaskType.TASK_GROUPED_MATMUL
    task.dependent_event = 0
    return task


class TestMegaKernelProfilingMetadata(unittest.TestCase):
    """Validate centralized stage and owner resolution."""

    def test_mega_moe_profile_spec_serializes_stage_and_owner_ids(self):
        """Resolve MegaMoe task semantics from its single Host declaration."""
        topology = TaskSplitValue()
        topology.rank_id = 1
        runtime_config = RuntimeConfigC()
        runtime_config.task_num = 5

        gmm = TaskDescC()
        gmm.task_type = TaskType.TASK_GROUPED_MATMUL
        gmm.tiling_data_position = 17
        gmm.task_index = 48
        runtime_config.all_tasks[0] = gmm

        swiglu = TaskDescC()
        swiglu.task_type = TaskType.TASK_SWI_GLU
        swiglu.task_split_value = 128
        swiglu.task_index = 128
        runtime_config.all_tasks[1] = swiglu

        dispatch = TaskDescC()
        dispatch.task_type = TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
        dispatch.inputs[0].input_position = 1
        dispatch.task_split_num = 64
        dispatch.task_index = 4
        runtime_config.all_tasks[2] = dispatch

        terminate = TaskDescC()
        terminate.task_type = TaskType.TASK_TERMINATE
        runtime_config.all_tasks[3] = terminate

        unknown = TaskDescC()
        unknown.task_type = TaskType.TASK_MATMUL
        runtime_config.all_tasks[4] = unknown

        _configure_mega_moe_profile_metadata(
            runtime_config,
            topology,
            num_cube_cores=24,
            is_backward=False,
        )

        self.assertEqual(runtime_config.all_tasks[0].profile_desc_id, 0x10002)
        self.assertEqual(runtime_config.all_tasks[0].profile_owner_id, 10)
        self.assertEqual(runtime_config.all_tasks[1].profile_desc_id, 0x10003)
        self.assertEqual(runtime_config.all_tasks[1].profile_owner_id, 10)
        self.assertEqual(runtime_config.all_tasks[2].profile_desc_id, 0x10001)
        self.assertEqual(runtime_config.all_tasks[2].profile_owner_id, 2)
        self.assertEqual(runtime_config.all_tasks[3].profile_desc_id, 0x10006)
        self.assertEqual(
            runtime_config.all_tasks[3].profile_owner_id,
            INVALID_PROFILE_OWNER_ID,
        )
        self.assertEqual(
            runtime_config.all_tasks[4].profile_desc_id,
            INVALID_PROFILE_DESC_ID,
        )
        metadata = _get_mega_kernel_profile_metadata(runtime_config)
        self.assertEqual(
            metadata.task_stage_names,
            {0: "GMM1", 1: "SwiGLU", 2: "Dispatch", 3: "TerminateTask"},
        )

    def test_forward_and_backward_metadata_are_selected_internally(self):
        """Bind the correct display metadata without user-provided names."""
        topology = TaskSplitValue()
        forward = RuntimeConfigC()
        backward = RuntimeConfigC()
        _configure_mega_moe_profile_metadata(
            forward, topology, num_cube_cores=24, is_backward=False
        )
        _configure_mega_moe_profile_metadata(
            backward, topology, num_cube_cores=24, is_backward=True
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
        self.assertEqual(MEGA_MOE_PROFILE_STAGE_NAMES[0x10002], "GMM1")
        self.assertEqual(MEGA_MOE_GRAD_PROFILE_STAGE_NAMES[0x11002], "ActGrad")

    def test_profile_spec_rejects_ambiguous_rules_without_partial_writes(self):
        """Reject a conflicting declaration before mutating task metadata."""
        runtime_config = RuntimeConfigC()
        runtime_config.task_num = 1
        runtime_config.all_tasks[0] = _three_record_task()
        spec = _ProfileSpec(
            kernel_name="ConflictingKernel",
            owner_label="Owner",
            stages=(
                _ProfileStage(0x12001, "First", task_type=TaskType.TASK_GROUPED_MATMUL),
                _ProfileStage(
                    0x12002, "Second", task_type=TaskType.TASK_GROUPED_MATMUL
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "matches multiple profile stages"):
            _apply_mega_kernel_profile_spec(runtime_config, spec)

        self.assertEqual(
            runtime_config.all_tasks[0].profile_desc_id,
            INVALID_PROFILE_DESC_ID,
        )
        self.assertEqual(
            runtime_config.all_tasks[0].profile_owner_id,
            INVALID_PROFILE_OWNER_ID,
        )

    def test_sparse_scheduled_task_id_is_configured_without_touching_gaps(self):
        """Use actual queue IDs instead of inferring a contiguous queue length."""
        runtime_config = RuntimeConfigC()
        runtime_config.task_num = 1
        runtime_config.num_workers = 48
        runtime_config.all_tasks[1].profile_desc_id = 0xABCDEF
        task = _three_record_task()
        task.tiling_data_position = 17
        runtime_config.all_tasks[7] = task
        runtime_config.task_index_num[0] = 1
        runtime_config.cube_task_indices[0] = 7
        spec = _ProfileSpec(
            kernel_name="SparseKernel",
            owner_label="Owner",
            stages=(
                _ProfileStage(
                    0x12001,
                    "Compute",
                    task_type=TaskType.TASK_GROUPED_MATMUL,
                    tiling_data_position=17,
                ),
            ),
        )

        _apply_mega_kernel_profile_spec(runtime_config, spec)

        self.assertEqual(runtime_config.all_tasks[7].profile_desc_id, 0x12001)
        self.assertEqual(runtime_config.all_tasks[1].profile_desc_id, 0xABCDEF)

    def test_display_metadata_does_not_change_serialized_runtime_config(self):
        """Keep names on Host while serializing only numeric IDs."""
        runtime_config = RuntimeConfigC()
        serialized_before = bytes(runtime_config)

        _set_mega_kernel_profile_metadata(
            runtime_config,
            kernel_name="MegaMoe",
            owner_label="Expert",
            stage_names=MEGA_MOE_PROFILE_STAGE_NAMES,
        )

        self.assertEqual(bytes(runtime_config), serialized_before)


class TestMegaKernelProfileLayout(unittest.TestCase):
    """Validate exact bounded AIC/AIV buffer sizing."""

    def test_layout_rounds_busiest_workers_to_sixteen_records(self):
        """Mirror Device round-robin assignment for AIC and AIV independently."""
        runtime_config = RuntimeConfigC()
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
        runtime_config = RuntimeConfigC()
        runtime_config.num_workers = 48
        runtime_config.all_tasks[0] = _three_record_task()
        runtime_config.task_index_num[0] = 2041

        layout = _calculate_profile_layout(runtime_config)

        self.assertEqual(layout.aic_required_records, 258)
        self.assertEqual(layout.aic_record_capacity, 256)

    def test_prepared_runtime_serializes_disabled_config_and_lazily_profiles(self):
        """Keep the normal tensor disabled and create the enabled tensor on demand."""
        runtime_config = RuntimeConfigC()
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
