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
"""Unit tests for task-level Torch MegaKernel Tensor dump protocol."""

import json
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from torch import (
    bfloat16 as torch_bfloat16,
    float32 as torch_float32,
    frombuffer as torch_frombuffer,
    load as torch_load,
    ones as torch_ones,
    tensor as torch_tensor,
    uint8 as torch_uint8,
    zeros as torch_zeros,
)
from torch.testing import assert_close

from hyper_parallel.core.multicore.diagnostics.dump import api as tensor_dump
from hyper_parallel.core.multicore.diagnostics.dump import l1 as tensor_dump_l1
from hyper_parallel.core.multicore.diagnostics.common.session import _reset_active_session_for_test
from hyper_parallel.core.multicore.scheduler.config import TaskAiCoreType, TaskType
from hyper_parallel.core.multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, SplitSpec, TensorSpec
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config, serialize_runtime_config


def _captured_l1_dump(values, dtype=torch_float32):
    """Build one Device-like captured L1 record for CPU-only tests."""
    payload = torch_tensor(values, dtype=dtype)
    payload_bytes = payload.numel() * payload.element_size()
    payload_offset = tensor_dump_l1._align_up(
        tensor_dump_l1.L1_DUMP_RECORDS_OFFSET
        + tensor_dump_l1.L1_DUMP_MAX_RECORDS * tensor_dump_l1.L1_DUMP_RECORD.size
    )
    record = tensor_dump_l1._L1RecordPlan(
        task_id=7,
        task_index=2,
        stage="SwiGLU",
        owner_id=3,
        owner_label="Expert",
        core_type="AIV",
        core_id=1,
        io="output",
        index=0,
        phase="after",
        parameter_position=8,
        tensor_name="swiglu_out",
        dtype=dtype,
        source_offset_bytes=32,
        actual_shape=tuple(payload.shape),
        payload_offset=payload_offset,
        payload_bytes=payload_bytes,
        full_bytes=payload_bytes,
        truncated=False,
        note="task-local test tile",
        record_offset=tensor_dump_l1.L1_DUMP_RECORDS_OFFSET,
    )
    raw_buffer = tensor_dump_l1._pack_l1_buffer((record,))
    raw_buffer[payload_offset:payload_offset + payload_bytes] = bytes(payload.view(torch_uint8).tolist())
    struct.pack_into(
        "<I",
        raw_buffer,
        record.record_offset + 7 * 4,
        tensor_dump_l1.L1_DUMP_STATUS_CAPTURED,
    )
    prepared = tensor_dump_l1._PreparedL1Dump(
        runtime_config=torch_ones(1, dtype=torch_uint8),
        buffer=torch_frombuffer(raw_buffer, dtype=torch_uint8).clone(),
        records=(record,),
        total_bytes=len(raw_buffer),
    )
    return prepared


class TestL1DumpProtocol(unittest.TestCase):
    """Validate the fixed L1 record ABI and Torch Tensor decoding."""

    def test_decodes_captured_record(self):
        """Feature: MegaKernel diagnostics.
        Description: Decodes captured record.
        Expectation: Decodes captured record.
        """
        prepared = _captured_l1_dump([1.5, -2.0])

        snapshots = tensor_dump_l1._decode_l1_dump(prepared)

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].record.task_id, 7)
        self.assertEqual(snapshots[0].stored_shape, (2,))
        assert_close(snapshots[0].data, torch_tensor([1.5, -2.0]))

    def test_decodes_bfloat16_without_conversion(self):
        """Feature: MegaKernel diagnostics.
        Description: Keep the original bfloat16 dtype when decoding Device payload bytes.
        Expectation: Decodes bfloat16 without conversion.
        """
        prepared = _captured_l1_dump([1.5, -2.0], dtype=torch_bfloat16)

        snapshot = tensor_dump_l1._decode_l1_dump(prepared)[0]

        self.assertEqual(snapshot.original_dtype, "bfloat16")
        self.assertEqual(snapshot.stored_dtype, "bfloat16")
        self.assertEqual(snapshot.data.dtype, torch_bfloat16)
        assert_close(snapshot.data, torch_tensor([1.5, -2.0], dtype=torch_bfloat16))

    def test_rejects_record_not_captured_by_device(self):
        """Feature: MegaKernel diagnostics.
        Description: Rejects record not captured by device.
        Expectation: Rejects record not captured by device.
        """
        prepared = _captured_l1_dump([1.0])
        prepared.buffer[
            tensor_dump_l1.L1_DUMP_RECORDS_OFFSET + 7 * 4
        ] = tensor_dump_l1.L1_DUMP_STATUS_PENDING

        with self.assertRaisesRegex(RuntimeError, "was not captured"):
            tensor_dump_l1._decode_l1_dump(prepared)

    def test_plans_records_from_graph_sized_runtime(self):
        """Feature: MegaKernel diagnostics.
        Description: Retain dynamic RuntimeConfig arrays while planning task-local views.
        Expectation: Plans records from graph sized runtime.
        """
        runtime_config = allocate_runtime_config(1)
        runtime_config.task_num = 1
        runtime_config.num_workers = 2
        runtime_config.task_index_num[1] = 1
        runtime_config.vector_task_indices[0] = 0
        task = runtime_config.all_tasks[0]
        task.task_type = TaskType.TASK_SWI_GLU
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
        task.num_inputs = 1
        task.num_outputs = 1
        task.task_index = 0
        task.task_split_num = 1
        task.task_split_value = 2
        task.profile_owner_id = 999  # Deliberately unrelated to the common owner metadata.
        for parameter_position, tensor_desc in enumerate((task.inputs[0], task.outputs[0])):
            tensor_desc.num_dims = 1
            tensor_desc.dim[0] = 2
            tensor_desc.data_type = 4
            tensor_desc.input_position = parameter_position

        graph_input = TensorSpec("input", [2], param_position=0, dtype_size=4)
        graph_output = TensorSpec("output", [2], param_position=1, dtype_size=4)
        operator = OperatorNode(
            "swiglu",
            OpType.SWIGLU,
            [graph_input],
            [graph_output],
            [0, 1],
            2,
            SplitSpec([(0, 0)], lambda _: 1, [0]),
            -1,
            None,
            diagnostic_name="SwiGLU",
        )
        graph = ComputeGraph().add_op(operator)
        runtime_bytes = torch_frombuffer(
            bytearray(serialize_runtime_config(runtime_config)),
            dtype=torch_uint8,
        ).clone()
        runtime = SimpleNamespace(
            host_config=runtime_config,
            task_stage_names={0: "SwiGLU"},
            owner_label="Expert",
            metadata=SimpleNamespace(task_owner_ids={0: 4}),
            normal_tensor=runtime_bytes,
        )
        config = tensor_dump.TensorDumpConfig("dump", level="L1", stages="SwiGLU")

        prepared = tensor_dump_l1._prepare_l1_dump(
            runtime,
            graph,
            (torch_ones(2), torch_zeros(2)),
            config,
            rank=0,
            device="cpu",
        )

        self.assertEqual([(record.io, record.phase) for record in prepared.records], [
            ("input", "before"),
            ("output", "after"),
        ])
        self.assertEqual({record.core_type for record in prepared.records}, {"AIV"})
        self.assertEqual({record.owner_id for record in prepared.records}, {4})
        self.assertEqual(
            prepared.runtime_config[
                tensor_dump_l1.RuntimeConfigC.diagnostic_mode.offset
            ].item(),
            tensor_dump_l1.L1_DIAGNOSTIC_MODE,
        )


class TestL1DumpExport(unittest.TestCase):
    """Validate L1 complete launch arguments and task-centric export layout."""

    def tearDown(self) -> None:
        """Reset the process-local diagnostic session after each test."""
        _reset_active_session_for_test()

    def test_exports_task_files_and_metadata(self):
        """Feature: MegaKernel diagnostics.
        Description: Exports task files and metadata.
        Expectation: Exports task files and metadata.
        """
        prepared = _captured_l1_dump([4.0, 6.0])
        with tempfile.TemporaryDirectory() as output_dir:
            config = tensor_dump.TensorDumpConfig(
                output_dir,
                level="L1",
                stages="SwiGLU",
                directions="forward",
            )
            with mock.patch.object(
                tensor_dump,
                "_prepare_l1_dump",
                return_value=prepared,
            ):
                with tensor_dump.tensor_dump(config) as dumper:
                    events = torch_zeros(16)
                    call = dumper._acquire_call(
                        "graph", ("parameter",), kernel_name="test_kernel", direction="forward",
                        rank=0, device="cpu", runtime=SimpleNamespace(normal_tensor="normal"),
                        event_counters=events,
                    )
                    self.assertIs(call.runtime_config, prepared.runtime_config)
                    self.assertIs(call.diagnostic_buffer, prepared.buffer)
                    self.assertIs(call.event_counters, events)
                    call.complete()

            manifest_path = dumper.last_manifest_paths[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            invocation = manifest["invocations"][0]
            task = invocation["tasks"][0]
            tensor = task["tensors"][0]
            task_path = manifest_path.parent / task["metadataFile"]
            task_document = json.loads(task_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["level"], "L1")
            self.assertEqual(manifest["schemaVersion"], 4)
            self.assertEqual(manifest["tensorSerialization"], "torch.save")
            self.assertEqual(manifest["tensorFileLayout"], "stage_only")
            self.assertEqual(manifest["capturePhaseSemantics"], "task_boundary")
            self.assertEqual(manifest["stageViewStorage"], "per_task_files")
            self.assertEqual(invocation["taskCount"], 1)
            self.assertEqual(task["taskId"], 7)
            self.assertEqual(task["stage"], "SwiGLU")
            self.assertEqual(task_document["owner"], {"label": "Expert", "id": 3})
            self.assertEqual(tensor["capturePhase"], "after")
            self.assertEqual(tensor["sourceOffsetBytes"], 32)
            assert_close(
                torch_load(
                    manifest_path.parent / tensor["file"],
                    map_location="cpu",
                    weights_only=True,
                ),
                torch_tensor([4.0, 6.0]),
            )
