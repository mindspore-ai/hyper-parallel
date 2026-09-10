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

from hyper_parallel.core.multicore.profiling import (
    CORE_HEADER,
    CUBE_SLOT_COUNT,
    INVALID_OWNER_ID,
    PROFILE_RECORD,
    _parse_cycle_buffer,
    _profile_buffer_bytes_for_capacities,
)


_AIC_CAPACITY = 16
_AIV_CAPACITY = 16


def _slot_offset(core_type: int, block_id: int) -> int:
    aic_stride = CORE_HEADER.size + _AIC_CAPACITY * PROFILE_RECORD.size
    if core_type == 1:
        return block_id * aic_stride
    aiv_stride = CORE_HEADER.size + _AIV_CAPACITY * PROFILE_RECORD.size
    return CUBE_SLOT_COUNT * aic_stride + block_id * aiv_stride


def _profile_buffer(
    *,
    core_type: int = 1,
    block_id: int = 0,
    entry_cycle: int = 100,
    dropped_count: int = 0,
    desc_id: int = 0x10002,
    owner_id: int = INVALID_OWNER_ID,
) -> bytearray:
    buffer = bytearray(
        _profile_buffer_bytes_for_capacities(_AIC_CAPACITY, _AIV_CAPACITY)
    )
    offset = _slot_offset(core_type, block_id)
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


def _parse(buffer: bytes, *, detailed_task_names: bool = False) -> dict:
    return _parse_cycle_buffer(
        buffer=buffer,
        rank=3,
        device_id=7,
        cycle_frequency_mhz=50.0,
        detailed_task_names=detailed_task_names,
        kernel_name="MegaMoe",
        owner_label="Expert",
        stage_names={0x10002: "GMM1"},
        soc_name="Ascend910B3",
        task_stage_names={9: "GMM1"},
        aic_record_capacity=_AIC_CAPACITY,
        aiv_record_capacity=_AIV_CAPACITY,
    )


class TestMegaKernelCycleTrace(unittest.TestCase):
    """Validate schema, timing, naming, fallback, and corruption checks."""

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
        """Keep a record readable when a concrete Kernel has no stage rule."""
        trace = _parse(_profile_buffer(desc_id=0x20000 + 107))
        event = next(event for event in trace["traceEvents"] if event["ph"] == "X")

        self.assertEqual(event["name"], "TaskType_107")
        self.assertEqual(event["args"]["task_type"], 107)

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
