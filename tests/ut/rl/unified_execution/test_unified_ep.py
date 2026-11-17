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
"""Role binding, selection and gradient checks independent of NPU hardware."""

import inspect
import subprocess
import sys
import unittest
from copy import deepcopy
from functools import partial
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from rl.unified_execution.core.contract import build_unified_payload, parse_unified_payload
from rl.unified_execution.core.execution_plan import UNIFIED_EP_TARGET, unified_ep_options
from rl.unified_execution.ops.moe.mc2_ep import mc2_ep
from rl.unified_execution.ops.moe.training_ep import training_ep
from rl.unified_execution.runtimes.hyper_parallel.builder import build_inference_model
from rl.unified_execution.runtimes.hyper_parallel.moe import bind_training_ep, validate_training_ep
from rl.unified_execution.runtimes.vllm.moe import bind_inference_ep, validate_inference_ep
from rl.unified_execution.runtimes.vllm.parallel import select_ep_backend


class TestUnifiedEP(unittest.TestCase):
    """Reject transport drift and compare dynamic routing with independent derivatives."""

    def test_backend_selection_is_static_and_explicit(self) -> None:
        """Operator constraints decide admission, not historical benchmark shapes."""
        arguments = {"device_name": "Ascend910B4", "world_size": 4, "mc2_capacity": 256,
                     "dtype": torch.bfloat16, "available": True, "hccl_buffer_mb": 512,
                     "config": SimpleNamespace(hidden_size=2048, moe_intermediate_size=768,
                                               num_experts=128, num_experts_per_tok=8)}
        self.assertEqual(select_ep_backend("auto", **arguments), "mc2")
        self.assertEqual(select_ep_backend("allgather", **arguments), "allgather")
        for key, value in (("world_size", 16), ("world_size", 32), ("mc2_capacity", 1),
                           ("mc2_capacity", 64), ("mc2_capacity", 255), ("dtype", torch.float16),
                           ("config", SimpleNamespace(hidden_size=4096, moe_intermediate_size=1024,
                                                      num_experts=64, num_experts_per_tok=4))):
            with self.subTest(key=key, value=value):
                supported = dict(arguments, **{key: value})
                self.assertEqual(select_ep_backend("auto", **supported), "mc2")
                self.assertEqual(select_ep_backend("mc2", **supported), "mc2")
        for key, value in (("world_size", 12), ("mc2_capacity", 257), ("dtype", torch.float32),
                           ("device_name", "Ascend910_95"), ("available", False),
                           ("hccl_buffer_mb", 1)):
            with self.subTest(key=key):
                unsupported = dict(arguments, **{key: value})
                self.assertEqual(select_ep_backend("auto", **unsupported), "allgather")
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    select_ep_backend("mc2", **unsupported)

    def test_mc2_shape_and_window_boundaries(self) -> None:
        """Protect real shape/window limits while allowing previously unmeasured shapes."""
        config = SimpleNamespace(hidden_size=2048, num_experts=128, num_experts_per_tok=8)
        arguments = {"device_name": "Ascend910B4", "world_size": 16, "mc2_capacity": 256,
                     "dtype": torch.bfloat16, "available": True, "config": config}
        self.assertEqual(select_ep_backend("auto", hccl_buffer_mb=260, **arguments), "mc2")
        self.assertEqual(select_ep_backend("auto", hccl_buffer_mb=259, **arguments), "allgather")
        with self.assertRaisesRegex(ValueError, "HCCL_BUFFSIZE >= 260"):
            select_ep_backend("mc2", hccl_buffer_mb=259, **arguments)
        for key, value in (("hidden_size", 2049), ("hidden_size", 10272),
                           ("num_experts", 127), ("num_experts", 2048), ("num_experts_per_tok", 17)):
            with self.subTest(key=key):
                changed = deepcopy(config)
                setattr(changed, key, value)
                unsupported = arguments.copy()
                unsupported["config"] = changed
                self.assertEqual(select_ep_backend("auto", **unsupported), "allgather")

    def test_recipe_preserves_role_options_and_rejects_unknown_fields(self) -> None:
        """Keep public configuration distinct from internal worker bindings."""
        entry = {"match": "*.mlp.forward", "when": "ep", "replace": {"kind": "method",
                 "implementation": {"_target_": UNIFIED_EP_TARGET, "mc2_capacity": 32}}}
        self.assertEqual(unified_ep_options([entry]), {"mc2_capacity": 32, "inference_backend": "auto"})
        for key, value in (("backend", "mc2"), ("inference_backend", "invalid"),
                           ("mc2_capacity", True), ("capacity", 32)):
            changed = deepcopy(entry)
            changed["replace"]["implementation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                unified_ep_options([changed])

    def test_inference_builder_binds_role_without_mutating_shared_plan(self) -> None:
        """Worker adaptation must not change the serialized Actor plan or EP1 model."""
        entry = {"match": "*.mlp.forward", "when": "ep", "replace": {"kind": "method",
                 "implementation": {"_target_": UNIFIED_EP_TARGET, "mc2_capacity": 32}}}
        model_class = Mock()
        runtime = SimpleNamespace(backend="mc2", comm_name="dedicated", max_token_rows=1024,
                                  row_alignment=2, dp_size=2)
        context = SimpleNamespace(ep_size=2, mesh_context=object(), expert_runtime=runtime)
        scope = "rl.unified_execution.runtimes.hyper_parallel.builder"
        with patch(scope + ".resolve_runtime_symbol", return_value=model_class), \
                patch(scope + ".normalized_compute_setup") as setup:
            build_inference_model(SimpleNamespace(model_type="qwen3_moe"), [entry], parallel_context=context)
        implementation = setup.call_args.args[0][0]["replace"]["implementation"]
        self.assertTrue(implementation["_target_"].endswith(".inference_ep_compute_fn"))
        self.assertEqual(implementation["backend"], "mc2")
        self.assertEqual(implementation["comm_name"], "dedicated")
        self.assertEqual(entry["replace"]["implementation"]["_target_"], UNIFIED_EP_TARGET)
        context.ep_size, context.expert_runtime = 1, None
        with patch(scope + ".resolve_runtime_symbol", return_value=model_class), \
                patch(scope + ".normalized_compute_setup") as setup:
            build_inference_model(SimpleNamespace(model_type="qwen3_moe"), [entry], parallel_context=context)
        self.assertEqual(setup.call_args.args[0], [entry])


    def test_training_import_does_not_load_inference_runtime(self) -> None:
        """Training/config imports must not register vLLM custom ops or compile entrypoints."""
        subprocess.run([sys.executable, "-c", (
            "import sys; import rl.unified_execution.runtimes.hyper_parallel.moe; "
            "assert not any(n.startswith('rl.unified_execution.runtimes.vllm') for n in sys.modules); "
            "assert 'vllm' not in sys.modules"
        )], check=True, timeout=60)

    def test_inference_audit_rejects_registry_and_ownership_drift(self) -> None:
        """The object reached by the graph must retain the audited experts and limits."""
        group = Mock()
        group.size.return_value, group.rank.return_value = 2, 0
        experts = SimpleNamespace(local_expert_count=2, _ep_use_grouped_gemm=True)
        experts.forward_expert_major = MethodType(GroupedExperts.forward_expert_major, experts)
        module = SimpleNamespace(experts=experts)
        options = {"group": group, "rank": 0, "world_size": 2, "local_experts": 2}
        router = Mock()
        registry = {}
        config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=registry))
        fake_config = SimpleNamespace(get_current_vllm_config=lambda: config)
        runtime = SimpleNamespace(backend="allgather", comm_name="", max_token_rows=512,
                                  row_alignment=2, dp_size=1)
        recipe = {"mc2_capacity": 32, "inference_backend": "auto"}
        scope = "rl.unified_execution.runtimes.vllm.moe"
        with patch.dict("sys.modules", {"vllm.config": fake_config}), \
                patch(scope + ".bind_expert_ownership", return_value=(router, options)):
            compute = bind_inference_ep(module, object(), mc2_capacity=32, backend="allgather",
                                        comm_name="", max_token_rows=512, row_alignment=2, dp_size=1)
            validate_inference_ep(module, partial(compute, module), 2, "mlp", router, recipe, runtime=runtime)
            key = next(iter(registry))
            registry[key] = object()
            with self.assertRaisesRegex(ValueError, "executor differs"):
                validate_inference_ep(module, partial(compute, module), 2, "mlp", router, recipe, runtime=runtime)
            registry[key] = compute.expert_executor
            options["rank"] = 1
            with self.assertRaisesRegex(ValueError, "expert ownership differs"):
                validate_inference_ep(module, partial(compute, module), 2, "mlp", router, recipe, runtime=runtime)

    def test_dynamic_inverse_permutation_and_gradients(self) -> None:
        """A deliberately shuffled return must restore token slots before FP32 weighting."""
        states = torch.tensor([[2., 3.], [4., 5.]], requires_grad=True)
        ids = torch.tensor([[0, 1], [1, 0]])
        weights = torch.tensor([[.123456, .876544], [.234567, .765433]], requires_grad=True)
        order = torch.tensor([3, 0, 2, 1])
        routed = states.repeat_interleave(2, 0) * (ids.flatten()[:, None] + 1)
        dispatch = SimpleNamespace(states=states, expert_indices=ids, send_counts=[2, 2],
                                   receive_counts=[2, 2], dispatch_order=order)
        with patch("rl.unified_execution.ops.moe.training_ep._prepare_ep_dispatch", return_value=dispatch), \
                patch("rl.unified_execution.ops.moe.training_ep._run_ep_local_experts", return_value=routed[order]):
            actual = training_ep(object(), states, ids, weights, group=object(), rank=0,
                                world_size=2, local_experts=1)
        expected = states * ((ids + 1) * weights).sum(-1, keepdim=True)
        torch.testing.assert_close(actual, expected)
        grads = torch.autograd.grad(actual.sum(), (states, weights), retain_graph=True)
        reference = torch.autograd.grad(expected.sum(), (states, weights))
        for value, target in zip(grads, reference):
            torch.testing.assert_close(value, target)

    def test_training_role_and_audit_survive_eval_mode(self) -> None:
        """Actor eval keeps dynamic EP and the audit rejects a substituted transport."""
        experts = SimpleNamespace(local_expert_count=64, _ep_use_grouped_gemm=True)
        module = SimpleNamespace(experts=experts, training=False)
        group = Mock()
        group.size.return_value, group.rank.return_value = 2, 0
        mesh = Mock()
        mesh.size.return_value, mesh.get_local_rank.return_value = 2, 0
        mesh.get_group.return_value = group
        router = Mock()
        with patch("rl.unified_execution.models.bindings.component_family",
                   return_value=SimpleNamespace(router_adapter="fixture")), \
                patch("rl.unified_execution.models.bindings.bind_local_expert_forward"), \
                patch.dict("rl.unified_execution.models.bindings.MOE_ROUTER_ADAPTERS", {"fixture": router}):
            compute = bind_training_ep(module, {"ep": mesh})
        closure = inspect.getclosurevars(compute).nonlocals
        self.assertIs(closure["transport"], training_ep)
        recipe = {"mc2_capacity": 32, "inference_backend": "auto"}
        validate_training_ep(module, partial(compute, module), 2, "mlp", router, recipe)
        for name, cell in zip(compute.__code__.co_freevars, compute.__closure__):
            if name == "transport":
                cell.cell_contents = Mock()
        with self.assertRaisesRegex(ValueError, "role, router or expert ownership"):
            validate_training_ep(module, partial(compute, module), 2, "mlp", router, recipe)

    def test_explicit_plan_and_invalid_capacity(self) -> None:
        entry = {"match": "*.mlp.forward", "when": "ep", "replace": {"kind": "method",
                 "implementation": {"_target_": UNIFIED_EP_TARGET, "mc2_capacity": 32}}}
        self.assertEqual(unified_ep_options([entry]), {"mc2_capacity": 32, "inference_backend": "auto"})
        self.assertIsNone(unified_ep_options([]))
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            unified_ep_options([entry, entry])
        entry["replace"]["implementation"]["mc2_capacity"] = True
        with self.assertRaisesRegex(ValueError, "positive integer"):
            unified_ep_options([entry])

    def test_shared_recipe_survives_worker_contract(self) -> None:
        """Retain the shared expert recipe across worker contract serialization."""
        entry = {"match": "*.mlp.forward", "when": "ep", "replace": {"kind": "method",
                 "implementation": {"_target_": UNIFIED_EP_TARGET, "mc2_capacity": 32}}}
        config = {
            "model": {"name": "qwen3_moe"},
            "consistency": {"schema_version": 1, "mode": "batch_invariant",
                            "attention_backend": "flash_attention_npu_v4",
                            "batch_invariant": {"provider": "torch_npu_level3", "fill_uninitialized_memory": True}},
            "train": {"plan_overrides": [entry], "accelerator": {"tp": 2, "ep": 2}},
            "rollout": {"vllm": {"enforce_eager": False}},
        }
        payload = parse_unified_payload(build_unified_payload(config))
        self.assertEqual(unified_ep_options(payload["plan_overrides"]),
                         {"mc2_capacity": 32, "inference_backend": "auto"})
        self.assertEqual(payload["expert_parallelism"], {"training": 2, "inference": 2})
        config["train"]["plan_overrides"] = []
        with self.assertRaisesRegex(ValueError, "shared graph EP recipe"):
            build_unified_payload(config)



class TestMC2Metadata(unittest.TestCase):
    """Verify the native pair receives legal padding and unchanged opaque metadata."""

    def test_padding_and_opaque_metadata_are_preserved(self) -> None:
        """Padding IDs remain unique and combine retains original FP32 probabilities."""
        x = torch.ones(1, 4, dtype=torch.bfloat16)
        ids = torch.tensor([[2, 0]])
        weights = torch.tensor([[.123456, .876544]], dtype=torch.float32)
        assist, received, scales = object(), object(), object()
        packed, counts = torch.ones(4, 4, dtype=torch.bfloat16), torch.tensor([2, 2])
        native = SimpleNamespace(
            npu_moe_distribute_dispatch_v2=Mock(return_value=(packed, None, assist, counts, received, None, scales)),
            npu_moe_distribute_combine_v2=Mock(return_value=torch.ones(4, 4, dtype=torch.bfloat16)),
        )
        experts = SimpleNamespace(forward_expert_major=Mock(return_value=packed))
        with patch.dict("sys.modules", {"torch_npu": native}):
            result = mc2_ep(x, ids, weights, experts=experts, comm_name="dedicated", rank=0,
                            world_size=2, local_experts=2, mc2_capacity=4)
        sent = native.npu_moe_distribute_dispatch_v2.call_args.kwargs
        merged = native.npu_moe_distribute_combine_v2.call_args.kwargs
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(sent["expert_ids"].tolist(), [[2, 0], [0, 1], [0, 1], [0, 1]])
        self.assertEqual(sent["x_active_mask"].tolist(), [True, False, False, False])
        self.assertIs(merged["assist_info_for_combine"], assist)
        self.assertIs(merged["ep_send_counts"], received)
        self.assertIs(merged["expand_scales"], scales)
        self.assertIs(merged["x_active_mask"], sent["x_active_mask"])
        self.assertEqual(merged["expert_scales"].dtype, torch.float32)
        torch.testing.assert_close(merged["expert_scales"][:1], weights, rtol=0, atol=0)
        torch.testing.assert_close(merged["expert_scales"][1:], torch.zeros(3, 2), rtol=0, atol=0)
        experts.forward_expert_major.assert_called_once_with(packed, counts)

    def test_capacity_is_checked_before_dispatch(self) -> None:
        """Oversized inference rejects before entering any communication call."""
        native = SimpleNamespace(npu_moe_distribute_dispatch_v2=Mock())
        with patch.dict("sys.modules", {"torch_npu": native}), self.assertRaisesRegex(ValueError, "MC2 capacity"):
            mc2_ep(torch.zeros(5, 4), torch.zeros(5, 2), torch.zeros(5, 2), experts=object(),
                    comm_name="dedicated", rank=0, world_size=2, local_experts=2, mc2_capacity=4)
        native.npu_moe_distribute_dispatch_v2.assert_not_called()
