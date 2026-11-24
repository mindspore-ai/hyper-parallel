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
"""CPU-only contract and end-to-end public diagnostics tests for a non-MoE graph."""

import json
from functools import partial
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from torch import (
    arange as torch_arange,
    bfloat16 as torch_bfloat16,
    bool as torch_bool,
    device as torch_device,
    empty as torch_empty,
    float32 as torch_float32,
    frombuffer as torch_frombuffer,
    int32 as torch_int32,
    int64 as torch_int64,
    load as torch_load,
    ones as torch_ones,
    tensor as torch_tensor,
    uint8 as torch_uint8,
    zeros as torch_zeros,
)
from torch.testing import assert_close

from hyper_parallel.core.multicore import prepare_mega_kernel_diagnostics
from hyper_parallel.core.multicore.diagnostics.dump import api as tensor_dump
from hyper_parallel.core.multicore import validate_mega_kernel_diagnostic_contract
from hyper_parallel.core.multicore import diagnostics as diagnostics_module
from hyper_parallel.core.multicore.diagnostics import profiler as multicore_profiler
from hyper_parallel.core.multicore.diagnostics.dump import l1 as tensor_dump_l1
from hyper_parallel.core.multicore.modules.mega_moe.plan import build_mega_moe_plan
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.modules.mega_mhc.plan import build_mega_mhc_plan
from hyper_parallel.core.multicore.modules.mega_mhc_grad.plan import build_mega_mhc_grad_plan
from hyper_parallel.core.multicore.modules.mega_gate.diagnostics import bind_pipeline_graph
from hyper_parallel.core.multicore.modules.mega_gate.plan import (
    _build_pipeline_runtime_config, MEGA_GATE_ROUTE_TASK_TYPES, MEGA_GATE_ROUTE_STAGE_NAMES,
    MEGA_GATE_ROUTE_GRAD_K1_TASK_TYPES, MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES,
)
from hyper_parallel.core.multicore.diagnostics.profiler.cycle_trace import _PreparedProfileRuntime
from hyper_parallel.core.multicore.scheduler.config import RuntimeConfigC, TaskAiCoreType, TaskType
from hyper_parallel.core.multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, SplitSpec, TensorSpec
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config


def _two_stage_graph():
    """Create Copy/Scale stages without expert, GMM or MegaMoe routing metadata."""
    source, temporary, output = [TensorSpec(name, [2], index, dtype_size=4)
                                 for index, name in enumerate(("source", "temporary", "output"))]
    copy = OperatorNode("copy", OpType.SWIGLU, [source], [temporary], [0, 1], 2,
                        SplitSpec(None, lambda _: 1, [-1]), -1, None)
    scale = OperatorNode("scale", OpType.SWIGLU, [temporary], [output], [1, 2], 2,
                         SplitSpec(None, lambda _: 1, [-1]), -1, None)
    copy.task_num = scale.task_num = 1
    graph = ComputeGraph().add_op(copy).add_op(scale).add_edge(copy, scale)
    config = allocate_runtime_config(3)
    config.task_num = 2
    config.num_workers = 2
    config.task_index_num[1] = 3
    for task_id, operator in enumerate((copy, scale)):
        config.vector_task_indices[task_id] = task_id
        task = config.all_tasks[task_id]
        task.task_type = TaskType.TASK_SWI_GLU
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
        task.num_inputs = task.num_outputs = 1
        task.task_split_num = 1
        task.task_split_value = 2
        for descriptor, spec in ((task.inputs[0], operator.inputs[0]), (task.outputs[0], operator.outputs[0])):
            descriptor.input_position = spec.param_position
            descriptor.data_type = spec.dtype_size
    config.vector_task_indices[2] = 2
    config.all_tasks[2].task_type = TaskType.TASK_TERMINATE
    return graph, config


def _prepare(graph, config, **options):
    return prepare_mega_kernel_diagnostics(graph, config, kernel_name="copy_scale",
                                           parameter_count=3, device="cpu", **options)


class TestMegaKernelDiagnosticContract(unittest.TestCase):
    """Reject structural errors before any runtime image reaches a Device."""

    def test_read_only_validation_accepts_termination_outside_task_num(self):
        """Feature: MegaKernel diagnostics.
        Description: Read only validation accepts termination outside task num.
        Expectation: Read only validation accepts termination outside task num.
        """
        graph, config = _two_stage_graph()
        original = bytes(config)
        validate_mega_kernel_diagnostic_contract(graph, config, 3)
        self.assertEqual(bytes(config), original)
        config.task_num = 3
        validate_mega_kernel_diagnostic_contract(graph, config, 3)

    def test_rejects_bad_parameter_count_and_header_only_runtime(self):
        """Feature: MegaKernel diagnostics.
        Description: Rejects bad parameter count and header only runtime.
        Expectation: Rejects bad parameter count and header only runtime.
        """
        graph, config = _two_stage_graph()
        for count in (True, -1, 1.5, 2):
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_mega_kernel_diagnostic_contract(graph, config, count)
        with self.assertRaisesRegex(TypeError, "graph-sized"):
            validate_mega_kernel_diagnostic_contract(graph, RuntimeConfigC(), 3)

    def test_rejects_graph_port_and_task_port_mismatches(self):
        """Feature: MegaKernel diagnostics.
        Description: Rejects graph port and task port mismatches.
        Expectation: Rejects graph port and task port mismatches.
        """
        for field in ("param_positions", "num_inputs", "input_position", "data_type"):
            graph, config = _two_stage_graph()
            if field == "param_positions":
                graph.get_op("copy").param_positions[0] = 2
            elif field == "num_inputs":
                config.all_tasks[0].num_inputs = 0
            else:
                setattr(config.all_tasks[0].inputs[0], field, 2)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "disagrees|does not match"):
                validate_mega_kernel_diagnostic_contract(graph, config, 3)

    def test_stage_directory_constraints_belong_only_to_dump(self):
        """Feature: MegaKernel diagnostics.
        Description: Stage directory constraints belong only to dump.
        Expectation: Stage directory constraints belong only to dump.
        """
        for first, second in (("Same", "Same"), ("A/B", "A?B"), ("Copy", "..")):
            graph, config = _two_stage_graph()
            graph.get_op("copy").diagnostic_name = first
            graph.get_op("scale").diagnostic_name = second
            validate_mega_kernel_diagnostic_contract(graph, config, 3)
            with self.subTest(names=(first, second)), self.assertRaisesRegex(ValueError, "stage"):
                tensor_dump._stage_names(graph.topological_sort())

    def test_rejects_missing_duplicate_and_out_of_capacity_schedule_entries(self):
        """Feature: MegaKernel diagnostics.
        Description: Rejects missing duplicate and out of capacity schedule entries.
        Expectation: Rejects missing duplicate and out of capacity schedule entries.
        """
        for task_id in (0, 2, -1, 100):
            graph, config = _two_stage_graph()
            config.vector_task_indices[1] = task_id
            with self.subTest(task_id=task_id), self.assertRaisesRegex(ValueError, "schedule|scheduled"):
                validate_mega_kernel_diagnostic_contract(graph, config, 3)

    def test_rejects_graph_task_count_and_unmapped_compute_tasks(self):
        """Feature: MegaKernel diagnostics.
        Description: Rejects graph task count and unmapped compute tasks.
        Expectation: Rejects graph task count and unmapped compute tasks.
        """
        graph, config = _two_stage_graph()
        graph.get_op("copy").task_num = 3
        with self.assertRaisesRegex(ValueError, "exceed"):
            validate_mega_kernel_diagnostic_contract(graph, config, 3)
        graph, config = _two_stage_graph()
        config.all_tasks[2].task_type = TaskType.TASK_MATMUL
        with self.assertRaisesRegex(ValueError, "no graph stage"):
            validate_mega_kernel_diagnostic_contract(graph, config, 3)

    def test_dynamic_view_constraints_belong_only_to_l1_dump(self):
        """Feature: MegaKernel diagnostics.
        Description: Dynamic view constraints belong only to l1 dump.
        Expectation: Dynamic view constraints belong only to l1 dump.
        """
        graph, config = _two_stage_graph()
        config.all_tasks[0].inputs[0].dynamic_shape = 1
        for resolver in (None, mock.Mock(), 123):
            graph.get_op("copy").diagnostic_view_resolver = resolver
            validate_mega_kernel_diagnostic_contract(graph, config, 3)
            if callable(resolver):
                resolver.assert_not_called()


class TestMegaKernelDiagnostics(unittest.TestCase):
    """Exercise the same public launch API for ordinary, L0, L1 and profiling modes."""

    def test_normal_preparation_ignores_unused_diagnostic_semantics(self):
        """Feature: MegaKernel diagnostics.
        Description: A broken optional diagnostic name or callback cannot break ordinary execution.
        Expectation: Normal preparation ignores unused diagnostic semantics.
        """
        graph, config = _two_stage_graph()
        graph.get_op("copy").diagnostic_name = 123
        graph.get_op("copy").diagnostic_view_resolver = 123
        owner = mock.Mock(side_effect=AssertionError("ordinary execution must not resolve owners"))
        adapter = _prepare(graph, config, owner_resolver=owner)
        events = torch_zeros(16, dtype=torch_int32)
        with adapter.launch([torch_ones(2)] * 3, direction="forward", fallback_event_counters=events) as call:
            self.assertIs(call.runtime_config, adapter.runtime_config)
        self.assertIsNone(adapter._runtime._metadata)
        owner.assert_not_called()

    def test_ordinary_launch_and_metadata_need_no_business_resolver(self):
        """Feature: MegaKernel diagnostics.
        Description: Ordinary launch and metadata need no business resolver.
        Expectation: Ordinary launch and metadata need no business resolver.
        """
        graph, config = _two_stage_graph()
        adapter = _prepare(graph, config)
        metadata = adapter.metadata
        self.assertEqual(metadata.kernel_name, "copy_scale")
        self.assertEqual(metadata.task_stage_names[0], "Copy")
        self.assertEqual(metadata.owner_label, "Owner")
        events = torch_zeros(16, dtype=torch_int32)
        with adapter.launch([torch_ones(2)] * 3, direction="forward", fallback_event_counters=events) as call:
            self.assertIs(call.runtime_config, adapter.runtime_config)
            self.assertIs(call.event_counters, events)
            self.assertIs(call.diagnostic_buffer, events)
        with multicore_profiler.mega_kernel_profile():
            with self.assertRaisesRegex(ValueError, "expects 3"):
                adapter.launch([], direction="forward", fallback_event_counters=events)
            with self.assertRaisesRegex(ValueError, "direction"):
                adapter.launch([torch_ones(2)] * 3, direction="sideways", fallback_event_counters=events)

    def test_owner_callback_and_profiling_use_public_adapter(self):
        """Feature: MegaKernel diagnostics.
        Description: Owner callback and profiling use public adapter.
        Expectation: Owner callback and profiling use public adapter.
        """
        graph, config = _two_stage_graph()
        context = object()
        owner = mock.Mock(return_value=7)
        adapter = _prepare(graph, config, owner_label="Block", owner_resolver=owner, owner_context=context)
        original = bytes(config)
        owner.assert_not_called()
        events = torch_zeros(16, dtype=torch_int32)
        with multicore_profiler.mega_kernel_profile() as profiler:
            with adapter.launch([torch_ones(2)] * 3, direction="forward", fallback_event_counters=events) as call:
                self.assertEqual(adapter.metadata.task_owner_ids[0], 7)
                self.assertIs(owner.call_args.args[2], context)
                self.assertEqual(bytes(config), original)
                self.assertEqual(call.runtime_config[RuntimeConfigC.diagnostic_mode.offset].item(), 1)
                self.assertIsNot(call.diagnostic_buffer, events)
                call.cancel()
            self.assertEqual(len(profiler._pending), 0)

    def test_tools_prepare_independently_and_skipped_calls_remain_lazy(self):
        """Feature: MegaKernel diagnostics.
        Description: Tools prepare independently and skipped calls remain lazy.
        Expectation: Tools prepare independently and skipped calls remain lazy.
        """
        graph, config = _two_stage_graph()
        owner = mock.Mock(return_value=3)
        resolver = mock.Mock(side_effect=AssertionError("must not resolve views"))
        graph.get_op("copy").diagnostic_view_resolver = resolver
        adapter = _prepare(graph, config, owner_resolver=owner)
        events = torch_zeros(16, dtype=torch_int32)
        self.assertFalse(adapter.is_active)
        owner.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            selection = tensor_dump.TensorDumpConfig(
                directory, kernel_names="another_kernel", level="L1", stages="Copy",
            )
            with tensor_dump.tensor_dump(selection):
                with adapter.launch([torch_ones(2)] * 3, direction="forward",
                                    fallback_event_counters=events) as call:
                    self.assertIs(call.runtime_config, adapter.runtime_config)
                    self.assertIs(call.diagnostic_buffer, events)
                owner.assert_not_called()
            with mock.patch(
                "hyper_parallel.core.multicore.diagnostics.profiler.api._PreparedProfileRuntime",
                side_effect=AssertionError("dump must not prepare profiling"),
            ):
                with tensor_dump.tensor_dump(tensor_dump.TensorDumpConfig(directory)):
                    with adapter.launch([torch_ones(2)] * 3, direction="forward", fallback_event_counters=events):
                        pass
            with mock.patch.object(tensor_dump, "_prepare_l1_dump",
                                   side_effect=AssertionError("profiling must not prepare dump")):
                with multicore_profiler.mega_kernel_profile():
                    with adapter.launch([torch_ones(2)] * 3, direction="forward",
                                        fallback_event_counters=events) as call:
                        call.cancel()
            resolver.assert_not_called()
            self.assertEqual(adapter.runtime_config[RuntimeConfigC.diagnostic_mode.offset].item(), 0)
            self.assertEqual(config.diagnostic_mode, 0)

    def test_inactive_dump_schedule_does_not_activate_launch(self):
        """Feature: MegaKernel diagnostics.
        Description: Inactive dump schedule does not activate launch.
        Expectation: Inactive dump schedule does not activate launch.
        """
        graph, config = _two_stage_graph()
        owner = mock.Mock(return_value=3)
        adapter = _prepare(graph, config, owner_resolver=owner)
        with tempfile.TemporaryDirectory() as directory:
            with tensor_dump.tensor_dump(tensor_dump.TensorDumpConfig(directory),
                                         schedule=diagnostics_module.schedule(wait=1, warmup=1, active=1)) as dumper:
                self.assertFalse(adapter.is_active)
                with adapter.launch([], direction="unused", fallback_event_counters=torch_zeros(16)) as call:
                    self.assertIs(call.runtime_config, adapter.runtime_config)
                dumper.step()
                self.assertFalse(adapter.is_active)
                owner.assert_not_called()
                dumper.step()
                self.assertTrue(adapter.is_active)

    def test_l0_two_stage_dump_without_moe_semantics_or_port_customization(self):
        """Feature: MegaKernel diagnostics.
        Description: L0 two stage dump without moe semantics or port customization.
        Expectation: L0 two stage dump without moe semantics or port customization.
        """
        graph, config = _two_stage_graph()
        adapter = _prepare(graph, config)
        parameters = (torch_tensor([1.0, 2.0]), torch_zeros(2), torch_zeros(2))
        with tempfile.TemporaryDirectory() as directory:
            with tensor_dump.tensor_dump(tensor_dump.TensorDumpConfig(directory, kernel_names="copy_scale")) as dumper:
                with adapter.launch(parameters, direction="forward", fallback_event_counters=torch_zeros(16)):
                    parameters[1].copy_(parameters[0] + 1)
                    parameters[2].copy_(parameters[1] * 2)
                    parameters[0].zero_()
            manifest_path = dumper.last_manifest_paths[0]
            invocation = json.loads(manifest_path.read_text(encoding="utf-8"))["invocations"][0]
            records = {record["name"]: record for record in invocation["tensors"]}
            assert_close(torch_load(manifest_path.parent / records["source"]["file"], weights_only=True),
                                       torch_tensor([1.0, 2.0]))
            assert_close(torch_load(manifest_path.parent / records["output"]["file"], weights_only=True),
                                       torch_tensor([4.0, 6.0]))
            ports = records["temporary"]["ports"]
            self.assertTrue((manifest_path.parent / ports[0]["file"]).samefile(manifest_path.parent / ports[1]["file"]))
            self.assertFalse(list(Path(directory).rglob("before")))
            self.assertFalse(list(Path(directory).rglob("after")))

    def test_l1_public_launch_decodes_simulated_device_records(self):
        """Feature: MegaKernel diagnostics.
        Description: L1 public launch decodes simulated device records.
        Expectation: L1 public launch decodes simulated device records.
        """
        graph, config = _two_stage_graph()
        adapter = _prepare(graph, config)
        parameters = (torch_tensor([3.0, 4.0]), torch_zeros(2), torch_zeros(2))
        with tempfile.TemporaryDirectory() as directory:
            selection = tensor_dump.TensorDumpConfig(directory, level="L1", stages="Copy")
            with tensor_dump.tensor_dump(selection) as dumper:
                with adapter.launch(parameters, direction="forward", fallback_event_counters=torch_zeros(16)) as call:
                    self.assertEqual(call.runtime_config[RuntimeConfigC.diagnostic_mode.offset].item(), 2)
                    raw = bytearray(call.diagnostic_buffer.tolist())
                    count = tensor_dump_l1.L1_DUMP_HEADER.unpack_from(raw)[3]
                    self.assertEqual(count, 2)
                    for index in range(count):
                        offset = (tensor_dump_l1.L1_DUMP_RECORDS_OFFSET
                                  + index * tensor_dump_l1.L1_DUMP_RECORD.size)
                        record = tensor_dump_l1.L1_DUMP_RECORD.unpack_from(raw, offset)
                        value = parameters[0] if index == 0 else parameters[0] + 1
                        raw[record[5]:record[5] + record[6]] = bytes(value.view(torch_uint8).tolist())
                        struct.pack_into("<I", raw, offset + 7 * 4, tensor_dump_l1.L1_DUMP_STATUS_CAPTURED)
                    call.diagnostic_buffer.copy_(torch_frombuffer(raw, dtype=torch_uint8))
            manifest_path = dumper.last_manifest_paths[0]
            task = json.loads(manifest_path.read_text(encoding="utf-8"))["invocations"][0]["tasks"][0]
            self.assertEqual(task["stage"], "Copy")
            record = next(record for record in task["tensors"] if record["io"] == "output")
            assert_close(torch_load(manifest_path.parent / record["file"], weights_only=True),
                                       torch_tensor([4.0, 5.0]))

    def test_dynamic_bounds_failure_and_failed_launch_leave_no_dump(self):
        """Feature: MegaKernel diagnostics.
        Description: Dynamic bounds failure and failed launch leave no dump.
        Expectation: Dynamic bounds failure and failed launch leave no dump.
        """
        graph, config = _two_stage_graph()
        config.all_tasks[0].inputs[0].dynamic_shape = 1
        resolver = mock.Mock(return_value=tensor_dump_l1.L1TensorView(100, (2,)))
        graph.get_op("copy").diagnostic_view_resolver = resolver
        adapter = _prepare(graph, config)
        resolver.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            selection = tensor_dump.TensorDumpConfig(directory, level="L1", stages="Copy")
            with tensor_dump.tensor_dump(selection):
                with self.assertRaisesRegex(ValueError, "exceeds its ABI"):
                    adapter.launch([torch_ones(2)] * 3, direction="forward", fallback_event_counters=torch_zeros(16))
            with tensor_dump.tensor_dump(tensor_dump.TensorDumpConfig(directory)):
                with self.assertRaisesRegex(RuntimeError, "native failure"):
                    with adapter.launch([torch_ones(2)] * 3, direction="forward",
                                        fallback_event_counters=torch_zeros(16)):
                        raise RuntimeError("native failure")
            self.assertFalse(list(Path(directory).rglob("*.pt")))

    def test_l1_rejects_invalid_views_noncontiguous_parameters_and_wrong_element_size(self):
        """Feature: MegaKernel diagnostics.
        Description: L1 rejects invalid views noncontiguous parameters and wrong element size.
        Expectation: L1 rejects invalid views noncontiguous parameters and wrong element size.
        """
        for view, parameter, error in (
            (tensor_dump_l1.L1TensorView(-4, (2,)), torch_ones(2), "non-negative"),
            (tensor_dump_l1.L1TensorView(1, (1,)), torch_ones(2), "align"),
            (tensor_dump_l1.L1TensorView(0, (-1,)), torch_ones(2), "non-negative"),
            (tensor_dump_l1.L1TensorView(0, (2,)), torch_ones(2, 2).t(), "contiguous"),
            (tensor_dump_l1.L1TensorView(0, (2,)), torch_ones(2, dtype=torch_bfloat16), "byte size"),
        ):
            graph, config = _two_stage_graph()
            graph.get_op("copy").diagnostic_view_resolver = mock.Mock(return_value=view)
            adapter = _prepare(graph, config)
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                selection = tensor_dump.TensorDumpConfig(directory, level="L1", stages="Copy", io="input")
                with tensor_dump.tensor_dump(selection), self.assertRaisesRegex(ValueError, error):
                    adapter.launch((parameter, torch_zeros(2), torch_zeros(2)), direction="forward",
                                   fallback_event_counters=torch_zeros(16))
        graph, config = _two_stage_graph()
        operator = graph.get_op("copy")
        operator.inputs[0].shape = [10, 2]
        operator.split_spec.split_inputs = [(0, 0)]
        task = config.all_tasks[0]
        task.task_split_value = 4
        task.inputs[0].base_ptr_offset = 16
        view = tensor_dump_l1._default_tensor_view(operator, task, task.inputs[0], operator.inputs[0], "input", 0)
        self.assertEqual((view.source_offset_bytes, view.shape), (64, (2, 2)))
        operator.inputs[0].shape = [2, 8]
        operator.split_spec.split_inputs = [(0, 1)]
        task.inputs[0].base_ptr_offset = 0
        with self.assertRaisesRegex(ValueError, "non-contiguous split"):
            tensor_dump_l1._default_tensor_view(operator, task, task.inputs[0], operator.inputs[0], "input", 0)

    def test_mega_moe_forward_backward_plans_pass_public_contract_on_cpu(self):
        """Feature: MegaKernel diagnostics.
        Description: Mega moe forward backward plans pass public contract on cpu.
        Expectation: Mega moe forward backward plans pass public contract on cpu.
        """
        spec = MegaMoeSpec(128, 32, 32, 2, 1, None, 128, 1, None, 0, 24)
        with mock.patch("torch.distributed.is_initialized", return_value=True), \
                mock.patch("torch.distributed.get_rank", return_value=7):
            plan = build_mega_moe_plan(spec, torch_device("cpu"))
        self.assertIsInstance(plan.fwd_diagnostics, diagnostics_module.MegaKernelDiagnostics)
        self.assertIsInstance(plan.bwd_diagnostics, diagnostics_module.MegaKernelDiagnostics)
        self.assertIs(plan.fwd_runtime_config, plan.fwd_diagnostics.runtime_config)
        self.assertEqual(plan.fwd_diagnostics._runtime.rank, 7)
        self.assertEqual(plan.fwd_diagnostics._runtime.spec.owner_context.topology.rank_id, 0)
        with tempfile.TemporaryDirectory() as directory:
            parameters = [torch_empty(0)] * 16
            parameters[6] = torch_tensor([64, 128], dtype=torch_int64)
            parameters[7] = torch_zeros(128, 64, dtype=torch_bfloat16)
            selection = tensor_dump.TensorDumpConfig(directory, level="L1", stages="SwiGLU", io="input", ranks=7)
            with tensor_dump.tensor_dump(selection):
                call = plan.fwd_diagnostics.launch(
                    parameters, direction="forward", fallback_event_counters=torch_zeros(8),
                )
                call.cancel()

    def test_mhc_native_ports_are_bound_lazily_without_changing_normal_runtime(self):
        """Feature: Native hard-coded ABI. Description: Bind both mHC graphs on CPU.

        Expectation: Real ports validate, while original scheduling storage is unchanged.
        Description: Mhc native ports are bound lazily without changing normal runtime.
        """
        with mock.patch("torch.distributed.is_initialized", return_value=True), \
                mock.patch("torch.distributed.get_rank", return_value=7):
            plans = (build_mega_mhc_plan(64, 32, "cpu", 2),
                     build_mega_mhc_grad_plan(64, 32, "cpu", 2, 4))
        for plan in plans:
            runtime = plan.runtime._runtime
            self.assertEqual(runtime.rank, 7)
            original = runtime.normal_tensor.clone()
            original_graph = runtime._graph
            self.assertIsNone(runtime._metadata)
            self.assertTrue(plan.runtime.metadata.task_stage_names)
            self.assertIsNot(runtime.graph, original_graph)
            self.assertTrue(all(op.diagnostic_host_ports for op in runtime.graph.topological_sort()))
            _PreparedProfileRuntime(runtime)
            assert_close(runtime.normal_tensor, original)
        with tempfile.TemporaryDirectory() as directory:
            forward = [torch_empty(shape, dtype=dtype) for shape, dtype in (
                ((1, 64, 32), torch_bfloat16), ((1, 64, 4, 32), torch_bfloat16),
                ((1, 64, 4), torch_float32), ((1, 64, 4), torch_float32), ((1, 64, 4, 4), torch_float32),
                ((24, 128), torch_float32), ((3,), torch_float32), ((24,), torch_float32), ((32,), torch_bfloat16),
                ((8,), torch_uint8), ((8,), torch_uint8), ((8,), torch_uint8),
                ((1, 64, 4, 32), torch_bfloat16), ((1, 64, 4), torch_float32), ((1, 64, 4), torch_float32),
                ((1, 64, 16), torch_float32), ((1, 64, 32), torch_bfloat16), ((1, 64, 24), torch_float32),
                ((1, 64, 1), torch_float32), ((40, 1, 64, 4), torch_float32),
                ((40, 1, 64, 4, 4), torch_float32), ((1, 64, 32), torch_bfloat16), ((1, 64, 1), torch_float32),
            )]
            all_stages = tuple(plans[0].runtime.metadata.stage_names.values())
            with tensor_dump.tensor_dump(tensor_dump.TensorDumpConfig(directory, level="L1", stages=all_stages)):
                plans[0].runtime.launch(forward, direction="forward", fallback_event_counters=torch_zeros(8)).cancel()
            parameters = [torch_empty(0)] * 33
            parameters[3] = torch_empty(1, 64, 4, 32, dtype=torch_bfloat16)
            for position, shape, dtype in (
                (24, (1, 64, 4, 32), torch_bfloat16), (28, (1, 64, 32), torch_bfloat16),
                (30, (1, 64, 4), torch_float32), (31, (1, 64, 4, 4), torch_float32),
            ):
                parameters[position] = torch_empty(shape, dtype=dtype)
            selection = tensor_dump.TensorDumpConfig(
                directory, level="L1", stages="MhcGradPrevXAndPost", io="output",
            )
            with tensor_dump.tensor_dump(selection):
                call = plans[1].runtime.launch(parameters, direction="backward", fallback_event_counters=torch_zeros(8))
                call.cancel()  # Planning only: CPU tests do not execute a Device task.
            unsupported = tensor_dump.TensorDumpConfig(directory, level="L1", stages="RmsNormGrad", io="output")
            with tensor_dump.tensor_dump(unsupported), self.assertRaisesRegex(ValueError, "multi-core reductions"):
                plans[1].runtime.launch(parameters, direction="backward", fallback_event_counters=torch_zeros(8))

    def test_gate_broadcast_snapshots_keep_worker_payloads_and_files_separate(self):
        """Feature: Broadcast L1. Description: Simulate four workers on one task ID.

        Expectation: Each worker receives a cache-line-aligned record and a unique file.
        Description: Gate broadcast snapshots keep worker payloads and files separate.
        """
        config = _build_pipeline_runtime_config(MEGA_GATE_ROUTE_TASK_TYPES, MEGA_GATE_ROUTE_STAGE_NAMES,
                                                kernel_name="HyperMegaGateRoute")
        single_worker = type(config).from_buffer_copy(bytes(config))
        single_worker.num_workers = 1
        self.assertEqual(len(tensor_dump_l1._task_locations(single_worker, broadcast_tasks=True)), config.task_num)
        adapter = prepare_mega_kernel_diagnostics(
            None, config, kernel_name="HyperMegaGateRoute", parameter_count=11, device="cpu",
            execution_mode="aiv_pipeline", vector_worker_stride=1, graph_adapter=bind_pipeline_graph,
        )
        scores = torch_arange(32, dtype=torch_float32).reshape(4, 8)
        parameters = (scores.clone(), torch_zeros(8), torch_zeros(8), torch_zeros(1, dtype=torch_bool),
                      adapter.runtime_config, adapter.runtime_config, torch_zeros(4, 2),
                      torch_zeros(4, 2, dtype=torch_int64), scores, torch_zeros(4, 2), torch_zeros(4, 1))
        with tempfile.TemporaryDirectory() as directory:
            selection = tensor_dump.TensorDumpConfig(directory, level="L1", stages="Sqrt", io="output")
            with tensor_dump.tensor_dump(selection) as dumper:
                with adapter.launch(parameters, direction="forward") as call:
                    prepared = getattr(call, "_prepared_l1", None)
                    self.assertIsNotNone(prepared)
                    plans = prepared.records
                    self.assertEqual([record.worker_id for record in plans], [0, 1, 2, 3])
                    raw = bytearray(call.diagnostic_buffer.tolist())
                    for record in plans:
                        self.assertEqual(record.payload_offset % 64, 0)
                        self.assertEqual(record.record_offset % 64, 0)
                        data = scores[record.core_id:record.core_id + 1].reshape(-1).view(torch_uint8)
                        raw[record.payload_offset:record.payload_offset + record.payload_bytes] = bytes(data.tolist())
                        struct.pack_into("<I", raw, record.record_offset + 28, tensor_dump_l1.L1_DUMP_STATUS_CAPTURED)
                    call.diagnostic_buffer.copy_(torch_frombuffer(raw, dtype=torch_uint8))
            manifest = dumper.last_manifest_paths[0]
            tasks = json.loads(manifest.read_text(encoding="utf-8"))["invocations"][0]["tasks"]
            self.assertEqual(len({task["tensors"][0]["file"] for task in tasks}), 4)
            for task in tasks:
                payload = torch_load(manifest.parent / task["tensors"][0]["file"], weights_only=True)
                assert_close(payload, scores[task["coreId"]:task["coreId"] + 1])

    def test_gate_acl_tail_output_is_captured_only_after_native_completion(self):
        """Feature: ACLNN tail L0. Description: Supply the result after a native call.

        Expectation: L0 keeps the finished gradient, not an internal workspace alias.
        Description: Gate acl tail output is captured only after native completion.
        """
        config = _build_pipeline_runtime_config(MEGA_GATE_ROUTE_GRAD_K1_TASK_TYPES,
                                                MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES, kernel_name="RouteGrad")
        adapter = prepare_mega_kernel_diagnostics(
            None, config, kernel_name="RouteGrad", parameter_count=11, device="cpu",
            execution_mode="aiv_pipeline", vector_worker_stride=1,
            graph_adapter=partial(bind_pipeline_graph, backward=True),
        )
        logits = torch_ones(3, 8)
        parameters = [torch_ones(3, 1), torch_ones(3, 1), torch_ones(3, 1), logits,
                      torch_zeros(3, 1, dtype=torch_int64), adapter.runtime_config, adapter.runtime_config,
                      None, None, None, logits]
        with tempfile.TemporaryDirectory() as directory:
            with tensor_dump.tensor_dump(tensor_dump.TensorDumpConfig(directory)) as dumper:
                with adapter.launch(parameters, direction="backward"):
                    parameters[9] = logits * 3
                parameters[9].zero_()
            manifest = dumper.last_manifest_paths[0]
            tensors = json.loads(manifest.read_text(encoding="utf-8"))["invocations"][0]["tensors"]
            gradient = next(tensor for tensor in tensors if tensor["name"] == "grad_logits")
            payload = torch_load(manifest.parent / gradient["file"], weights_only=True)
            assert_close(payload, logits * 3)
