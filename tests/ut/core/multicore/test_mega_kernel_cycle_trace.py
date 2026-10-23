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
"""Unit tests for MegaKernel cycle-buffer decoding."""

import unittest
from unittest.mock import patch

from hyper_parallel.core.multicore.diagnostics.common.graph import DiagnosticSpec
from hyper_parallel.core.multicore.diagnostics.common.runtime import PreparedDiagnosticRuntime
from hyper_parallel.core.multicore.diagnostics.profiler.cycle_trace import (
    CORE_HEADER,
    CUBE_SLOT_COUNT,
    INVALID_OWNER_ID,
    PROFILE_RECORD,
    TASK_TYPE_NAMES,
    _CycleTraceConfig,
    _parse_cycle_buffer,
    _PreparedProfileRuntime,
    _profile_buffer_bytes_for_capacities,
)
from hyper_parallel.core.multicore.scheduler.config import TaskType
from hyper_parallel.core.multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, SplitSpec
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config


_AIC_CAPACITY = 16
_AIV_CAPACITY = 16


def _slot_offset(core_type: int, block_id: int, cube_slot_count: int = CUBE_SLOT_COUNT) -> int:
    aic_stride = CORE_HEADER.size + _AIC_CAPACITY * PROFILE_RECORD.size
    if core_type == 1:
        return block_id * aic_stride
    aiv_stride = CORE_HEADER.size + _AIV_CAPACITY * PROFILE_RECORD.size
    return cube_slot_count * aic_stride + block_id * aiv_stride


def _profile_buffer(
    *,
    core_type: int = 1,
    block_id: int = 0,
    entry_cycle: int = 100,
    dropped_count: int = 0,
    desc_id: int = 0x10002,
    owner_id: int = INVALID_OWNER_ID,
    cube_slot_count: int = CUBE_SLOT_COUNT,
) -> bytearray:
    """Build a synthetic profile buffer with one initialized core slot."""
    buffer = bytearray(
        _profile_buffer_bytes_for_capacities(_AIC_CAPACITY, _AIV_CAPACITY, cube_slot_count)
    )
    offset = _slot_offset(core_type, block_id, cube_slot_count)
    capacity = _AIC_CAPACITY if core_type == 1 else _AIV_CAPACITY
    CORE_HEADER.pack_into(
        buffer,
        offset,
        entry_cycle,
        1,
        dropped_count,
        core_type,
        block_id,
        capacity,
        0,
    )
    PROFILE_RECORD.pack_into(
        buffer,
        offset + CORE_HEADER.size,
        110,
        160,
        desc_id,
        9,
        2,
        owner_id,
    )
    return buffer


def _parse(
    buffer: bytes, *, detailed_task_names: bool = False, aiv_capacity: int = _AIV_CAPACITY,
    cube_slot_count: int = CUBE_SLOT_COUNT,
) -> dict:
    return _parse_cycle_buffer(
        buffer,
        _CycleTraceConfig(
            rank=3,
            device_id=7,
            cycle_frequency_mhz=50.0,
            detailed_task_names=detailed_task_names,
            kernel_name="MegaMoe",
            owner_label="Expert",
            stage_names={0x10002: "GMM1"},
            soc_name="Ascend910B3",
            task_stage_names={9: "GMM1"},
            cube_slot_count=cube_slot_count,
        ),
        aic_record_capacity=_AIC_CAPACITY,
        aiv_record_capacity=aiv_capacity,
    )


class TestMegaKernelCycleTrace(unittest.TestCase):
    """Validate schema, timing, naming, fallback, and corruption checks."""

    def test_prepared_runtime_decodes_950_ticks_as_nanoseconds(self):
        """Decode PR/DT traces at every current Cube-count class without shrinking legacy slots."""
        for cores in (4, 8, 28, 32, 36):
            cube_slots = max(24, cores)
            config = allocate_runtime_config(1, num_workers=2 * cores)
            config.task_num = 1
            config.all_tasks[0].task_type = TaskType.TASK_GROUPED_MATMUL
            config.task_index_num[0] = 1
            graph = ComputeGraph()
            operator = OperatorNode("gmm", OpType.GMM, [], [], [], 1, SplitSpec(None, lambda _: 1), -1, None)
            operator.task_num = 1
            graph.add_op(operator)
            runtime = _PreparedProfileRuntime(PreparedDiagnosticRuntime(
                graph, config, DiagnosticSpec("Custom"), parameter_count=0, tensor_factory=bytes,
                rank=0, device_id=0,
            ))
            buffer = _profile_buffer(block_id=cores - 1, cube_slot_count=cube_slots)
            for soc_name in ("Ascend950PR_9589", "Ascend950DT"):
                with (self.subTest(cube_slots=cube_slots, soc_name=soc_name),
                      patch("hyper_parallel.core.multicore.diagnostics.profiler.cycle_trace._get_soc_name",
                            return_value=soc_name)):
                    trace = runtime.parse(buffer, detailed_task_names=False)
                    event = next(item for item in trace["traceEvents"] if item["ph"] == "X")
                    metadata = trace["megaKernelCycleTrace"]
                    self.assertEqual(event["ts"], 0.01, f"expected=0.01 us actual={event['ts']}")
                    self.assertEqual(event["dur"], 0.05, f"expected=0.05 us actual={event['dur']}")
                    self.assertEqual(metadata["socName"], soc_name,
                                     f"expected={soc_name} actual={metadata['socName']}")
                    self.assertEqual(metadata["cycleFrequencyMHz"], 1000.0,
                                     f"expected=1000.0 MHz actual={metadata['cycleFrequencyMHz']}")

    def test_parse_converts_cycles_and_preserves_raw_identifiers(self):
        """Convert 50 MHz cycles to microseconds and keep raw record fields."""
        trace = _parse(_profile_buffer())
        event = next(event for event in trace["traceEvents"] if event["ph"] == "X")
        metadata = trace["megaKernelCycleTrace"]

        self.assertEqual(event["name"], "GMM1")
        self.assertEqual(event["ts"], 0.2)
        self.assertEqual(event["dur"], 1.0)
        self.assertEqual(event["args"]["task_id"], 9)
        self.assertEqual(event["args"]["stage_task_index"], 2)
        self.assertEqual(metadata["schemaVersion"], 1)
        self.assertEqual(metadata["recordCount"], 1)
        self.assertEqual(metadata["profileBufferBytes"], len(_profile_buffer()))
        self.assertEqual(metadata["recordCapacityPerCore"], {"AIC": 16, "AIV": 16})

    def test_detailed_name_uses_internal_owner_label(self):
        """Add the configured owner and one-based task number only on request."""
        trace = _parse(_profile_buffer(owner_id=5), detailed_task_names=True)
        event = next(event for event in trace["traceEvents"] if event["ph"] == "X")

        self.assertEqual(event["name"], "Expert5_GMM1_task3")
        self.assertEqual(event["args"]["owner_id"], 5)

    def test_wait_and_trigger_names_include_their_compute_stage(self):
        """Distinguish synchronization records belonging to different compute stages."""
        expected_names = {
            0x10000: "Expert5_GMM1_WaitDependency_task3",
            0x10007: "Expert5_GMM1_TriggerEvent_task3",
        }
        for desc_id, expected_name in expected_names.items():
            with self.subTest(desc_id=desc_id):
                trace = _parse(
                    _profile_buffer(desc_id=desc_id, owner_id=5),
                    detailed_task_names=True,
                )
                event = next(
                    event for event in trace["traceEvents"] if event["ph"] == "X"
                )

                self.assertEqual(event["name"], expected_name)
                self.assertEqual(event["args"]["task_stage"], "GMM1")

    def test_unknown_task_type_uses_generic_fallback(self):
        """Keep a record readable when its task type is not registered."""
        unknown_task_type = 0xFFFF
        self.assertNotIn(unknown_task_type, TASK_TYPE_NAMES)
        trace = _parse(_profile_buffer(desc_id=0x20000 + unknown_task_type))
        event = next(event for event in trace["traceEvents"] if event["ph"] == "X")

        self.assertEqual(event["name"], f"TaskType_{unknown_task_type}")
        self.assertEqual(event["args"]["task_type"], unknown_task_type)

    def test_registered_mhc_post_task_type_preserves_its_name(self):
        """Decode registered MHC task 107 instead of treating it as unknown."""
        trace = _parse(_profile_buffer(desc_id=0x20000 + 107))
        event = next(event for event in trace["traceEvents"] if event["ph"] == "X")

        self.assertEqual(event["name"], "TaskType_MhcPost")
        self.assertEqual(event["args"]["task_type"], 107)

    def test_get_task_name_does_not_alias_mega_gate(self):
        """Decode GET using its dispatch ID without labeling MegaGate as SHMEM."""
        cases = (
            (TaskType.TASK_SHMEM_GET_MEM, "TaskType_ShmemGetMem"),
            (TaskType.TASK_GATE_SOFTPLUS, f"TaskType_{int(TaskType.TASK_GATE_SOFTPLUS)}"),
        )
        for task_type, expected_name in cases:
            with self.subTest(task_type=task_type):
                trace = _parse(_profile_buffer(desc_id=0x20000 + int(task_type)))
                event = next(event for event in trace["traceEvents"] if event["ph"] == "X")

                self.assertEqual(event["name"], expected_name)
                self.assertEqual(event["args"]["task_type"], int(task_type))

    def test_dropped_count_is_reported_without_overwriting_records(self):
        """Surface Device overflow in metadata and warnings."""
        trace = _parse(_profile_buffer(dropped_count=4))
        metadata = trace["megaKernelCycleTrace"]

        self.assertEqual(metadata["droppedRecordCount"], 4)
        self.assertEqual(metadata["warnings"], ["Device dropped 4 cycle trace records"])

    def test_aiv_slot_uses_its_own_region_and_numeric_thread_id(self):
        """Decode a sparse AIV block with the Host-known dynamic stride."""
        trace = _parse(_profile_buffer(core_type=2, block_id=3))
        event = next(event for event in trace["traceEvents"] if event["ph"] == "X")

        self.assertEqual(event["tid"], 2003)
        self.assertEqual(event["args"]["core_type"], "AIV")
        self.assertEqual(event["args"]["block_id"], 3)

    def test_decoder_reads_last_worker_in_expanded_regions(self):
        """Decode AIC/AIV tails after the baseline 24/48-slot region."""
        for cube_slots in (24, 32, 36):
            for core_type, block_id in ((1, cube_slots - 1), (2, 2 * cube_slots - 1)):
                with self.subTest(cube_slots=cube_slots, core_type=core_type):
                    buffer = _profile_buffer(
                        core_type=core_type, block_id=block_id, cube_slot_count=cube_slots,
                    )
                    trace = _parse(buffer, cube_slot_count=cube_slots)
                    event = next(item for item in trace["traceEvents"] if item["ph"] == "X")
                    self.assertEqual(event["args"]["block_id"], block_id)
                    self.assertEqual(trace["megaKernelCycleTrace"]["profileBufferBytes"], len(buffer))

    def test_parser_rejects_header_identity_mismatch(self):
        """Reject a slot whose header does not match its physical region."""
        buffer = _profile_buffer()
        offset = _slot_offset(1, 0)
        CORE_HEADER.pack_into(
            buffer,
            offset,
            100,
            1,
            0,
            2,
            0,
            _AIC_CAPACITY,
            0,
        )

        with self.assertRaisesRegex(ValueError, "slot identity mismatch"):
            _parse(buffer)

    def test_parser_rejects_buffer_without_records(self):
        """Fail clearly when profiling ran without producing Device records."""
        empty = bytes(
            _profile_buffer_bytes_for_capacities(_AIC_CAPACITY, _AIV_CAPACITY)
        )

        with self.assertRaisesRegex(ValueError, "contains no records"):
            _parse(empty)


if __name__ == "__main__":
    unittest.main()
