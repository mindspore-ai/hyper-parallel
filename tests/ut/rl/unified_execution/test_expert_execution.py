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
"""Scheduler bounds and graph-safe inference transport transitions."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from rl.unified_execution.runtimes.vllm.moe import InferenceExpertExecutor, execute_inference_ep
from rl.unified_execution.runtimes.vllm.parallel import build_expert_runtime, inference_row_bounds
from rl.unified_execution.core.execution_plan import UNIFIED_EP_TARGET


def _context(counts=None, graph_rows=None) -> SimpleNamespace:
    metadata = None if counts is None else SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor(counts))
    descriptor = None if graph_rows is None else SimpleNamespace(num_tokens=graph_rows)
    return SimpleNamespace(dp_metadata=metadata, batch_descriptor=descriptor, no_compile_layers={})


class TestInferenceExpertExecutor(unittest.TestCase):
    """Exercise group agreement, actual capture padding, and repeated compiled calls."""

    def test_scheduler_and_graph_bounds_include_rounding(self) -> None:
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=300),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[128, 301], max_cudagraph_capture_size=301),
            parallel_config=SimpleNamespace(tensor_parallel_size=4), num_speculative_tokens=2)
        payload = {"parallelism": {"sequence_parallel": True, "cp": 4}}
        self.assertEqual(inference_row_bounds(config, payload), (304, 4))
        config.scheduler_config.max_num_batched_tokens = 1025
        self.assertEqual(inference_row_bounds(config, payload), (1028, 4))
        payload["parallelism"]["sequence_parallel"] = False
        self.assertEqual(inference_row_bounds(config, payload), (1025, 1))

    def test_startup_rejects_different_group_limits(self) -> None:
        """Different scheduler budgets must fail before any dedicated group is created."""
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[32], max_cudagraph_capture_size=32),
            parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=2),
            device_config=SimpleNamespace(device="cpu"),
            model_config=SimpleNamespace(dtype=torch.bfloat16, hf_config=object()))
        payload = {"plan_overrides": [{"match": "*.mlp.forward", "when": "ep", "replace": {
            "kind": "method", "implementation": {"_target_": UNIFIED_EP_TARGET, "mc2_capacity": 32}}}]}
        group = Mock()
        group.size.return_value = 2
        native = SimpleNamespace()
        scope = "rl.unified_execution.runtimes.vllm.parallel"

        for field in (2, 5):
            def disagree(output, selected, **kwargs):
                del kwargs
                output.copy_(selected.repeat(2))
                output[selected.numel() + field] += 1

            with self.subTest(field=field), patch.dict("sys.modules", {"torch_npu": native}), \
                    patch.object(torch, "npu", Mock(), create=True), \
                    patch(scope + ".select_ep_backend", return_value="allgather"), \
                    patch(scope + ".dist") as distributed:
                distributed.all_gather_into_tensor.side_effect = disagree
                with self.assertRaisesRegex(ValueError, "scheduler/graph bounds"):
                    build_expert_runtime(config, payload, group, [0, 1])
                distributed.new_group.assert_not_called()

    def test_all_ranks_choose_fallback_from_common_schedule(self) -> None:
        dispatch = InferenceExpertExecutor(None, {}, "mc2", "dedicated", 256, 1024, 2, 2)
        context = _context([3, 301])
        for local_rows in (0, 1, 4, 302):
            with self.subTest(local_rows=local_rows):
                self.assertEqual(dispatch.select_backend_and_bucket(local_rows, context), ("allgather", 302))
        self.assertEqual(dispatch.select_backend_and_bucket(256, _context([1, 256])), ("mc2", 256))
        self.assertEqual(dispatch.select_backend_and_bucket(0, _context([0, 0])), ("mc2", 256))

    def test_allgather_and_actual_graph_sizes_ignore_mc2_capacity(self) -> None:
        dispatch = InferenceExpertExecutor(None, {}, "allgather", "", 32, 1024, 4, 1)
        self.assertEqual(dispatch.select_backend_and_bucket(300, _context(graph_rows=320)), ("allgather", 320))
        self.assertEqual(dispatch.select_backend_and_bucket(1000, _context()), ("allgather", 1000))
        self.assertEqual(dispatch.select_backend_and_bucket(0, _context()), ("allgather", 1))

    def test_invalid_metadata_fails_before_any_collective(self) -> None:
        dispatch = InferenceExpertExecutor(None, {}, "mc2", "dedicated", 32, 512, 1, 2)
        with self.assertRaisesRegex(ValueError, "synchronized DP"):
            dispatch.select_backend_and_bucket(1, _context())
        for rows, counts in ((513, [1, 513]), (12, [1, 10])):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "schedule/layout mismatch"):
                dispatch.select_backend_and_bucket(rows, _context(counts))

    def test_compiled_op_selects_each_schedule_without_recompilation(self) -> None:
        context = _context([2, 2])
        dispatch = InferenceExpertExecutor(None, {"rank": 0, "world_size": 2, "local_experts": 64},
                                  "mc2", "dedicated", 32, 512, 1, 2)
        context.no_compile_layers["fixture"] = dispatch
        fake_vllm = SimpleNamespace(get_forward_context=lambda: context)
        captured_graphs = []

        def compiler(graph, example_inputs):
            del example_inputs
            captured_graphs.append(graph)
            return graph.forward

        compiled = torch.compile(execute_inference_ep, backend=compiler, fullgraph=True, dynamic=True)
        states = torch.ones(2, 4)
        ids, weights = torch.zeros(2, 2, dtype=torch.int64), torch.ones(2, 2)
        scope = "rl.unified_execution.runtimes.vllm.moe"
        with patch.dict("sys.modules", {"vllm.forward_context": fake_vllm}), \
                patch(scope + ".mc2_ep", side_effect=lambda x, *args, **kwargs: x * 2) as mc2, \
                patch(scope + ".allgather_ep", side_effect=lambda x, *args, **kwargs: x * 3) as allgather:
            for remote_rows in (2, 300, 32, 512, 2):
                context.dp_metadata.num_tokens_across_dp_cpu = torch.tensor([2, remote_rows])
                actual = compiled(states, ids, weights, "fixture")
                factor = 2 if remote_rows <= 32 else 3
                torch.testing.assert_close(actual, states * factor, rtol=0, atol=0)
            self.assertEqual(mc2.call_count, 3)
            self.assertEqual(allgather.call_count, 2)
            self.assertEqual(len(captured_graphs), 1)
            self.assertEqual(allgather.call_args.kwargs["bucket_rows"], 512)

    def test_transport_failure_is_not_retried(self) -> None:
        dispatch = InferenceExpertExecutor(None, {"rank": 0, "world_size": 2, "local_experts": 64},
                                  "mc2", "dedicated", 32, 512, 1, 1)
        scope = "rl.unified_execution.runtimes.vllm.moe"
        with patch(scope + ".mc2_ep", side_effect=RuntimeError("communication failed")), \
                patch(scope + ".allgather_ep") as allgather, \
                self.assertRaisesRegex(RuntimeError, "communication failed"):
            dispatch.forward(torch.ones(2, 4), torch.zeros(2, 2), torch.ones(2, 2), _context())
        allgather.assert_not_called()
