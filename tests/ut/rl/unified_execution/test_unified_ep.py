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
"""Dynamic training dispatch and graph-safe rollout expert execution."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rl.unified_execution.execution_plan import EXPERTS_TARGET, ROLLOUT_EXPERTS_TARGET
from rl.unified_execution.vllm.model_builder import build_inference_model
from rl.unified_execution.training.grouped_experts import bind_native_unpermute
from rl.unified_execution.vllm.moe.grouped_experts import (
    RolloutEPGroupedExperts, build_rollout_experts, register_expert_backend,
)
from rl.unified_execution.vllm.moe.fused_experts import AscendFusedExpertsBackend


class TestUnifiedEP(unittest.TestCase):
    """Reject transport drift and compare dynamic routing with independent derivatives."""

    def test_inference_builder_binds_role_without_mutating_shared_plan(self) -> None:
        """Worker adaptation must not change the serialized Actor plan or EP1 model."""
        entry = {"match": "*.mlp.experts", "replace": {"kind": "module",
                 "implementation": {"_target_": EXPERTS_TARGET}}}
        model_class = Mock()
        context = SimpleNamespace(ep_size=2, sequence_parallel=True, mesh_context=object())
        scope = "rl.unified_execution.vllm.model_builder"
        with patch(scope + ".resolve_runtime_symbol", return_value=model_class), \
                patch(scope + ".normalized_compute_setup") as setup:
            build_inference_model(SimpleNamespace(model_type="qwen3_moe"), [entry], parallel_context=context)
        self.assertEqual(len(setup.call_args.args[0]), 1)
        implementation = setup.call_args.args[0][0]["replace"]["implementation"]
        self.assertEqual(implementation["_target_"], ROLLOUT_EXPERTS_TARGET)
        self.assertEqual(set(implementation), {"_target_"})
        self.assertEqual(entry["replace"]["implementation"]["_target_"], EXPERTS_TARGET)
        context.sequence_parallel = False
        mixed_config = SimpleNamespace(model_type="qwen3_moe", num_hidden_layers=3,
                                       mlp_only_layers=[1], decoder_sparse_step=1)
        with patch(scope + ".resolve_runtime_symbol", return_value=model_class), \
                patch(scope + ".normalized_compute_setup") as setup:
            build_inference_model(mixed_config, [entry], parallel_context=context)
        overrides = setup.call_args.args[0][1:]
        self.assertEqual([entry["match"] for entry in overrides], ["model.layers.0.mlp", "model.layers.2.mlp"])
        override = overrides[-1]
        self.assertEqual(override["in_dst"]["hidden_states"]["tp"], "replicate")
        self.assertEqual(override["out_src"]["output"]["tp"], "replicate")
        self.assertEqual(entry["replace"]["implementation"]["_target_"], EXPERTS_TARGET)
        from hyper_parallel.components.modules.module_family import FamilyRequest, resolve_family
        from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import GroupedExpertsModule

        request = FamilyRequest(build_rollout_experts, {})
        for degree, expected in ((1, GroupedExpertsModule), (2, RolloutEPGroupedExperts)):
            with self.subTest(ep=degree):
                self.assertIs(resolve_family(request, context={"ep_size": degree}).constructor, expected)

    def test_empty_source_preserves_activation_dtype_and_router_gradient(self):
        """Empty FP32 scores retain their gradient without promoting BF16 expert outputs."""
        from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
            GroupedExpertsConfig, GroupedExpertsModule,
        )

        config = GroupedExpertsConfig(2, 4, 3, routing_weight_mode="output", num_layers=1)
        experts = GroupedExpertsModule(config=config)
        bind_native_unpermute(experts)
        values = torch.empty(0, 4, dtype=torch.bfloat16, requires_grad=True)
        weights = torch.empty(0, 2, dtype=torch.float32, requires_grad=True)
        result = experts.unpermute(values, torch.empty(0, dtype=torch.int32), weights, (0, 4))
        self.assertEqual(result.dtype, values.dtype)
        result.sum().backward()
        self.assertEqual(values.grad.dtype, values.dtype)
        self.assertEqual(weights.grad.dtype, weights.dtype)

    def test_graph_reads_each_layers_live_weights(self) -> None:
        """One backend serves both layers; a replay reads the latest published values."""
        from torch.fx.experimental.proxy_tensor import make_fx
        from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import GroupedExpertsConfig

        registry = {}
        config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=registry))
        context = SimpleNamespace(get_forward_context=lambda: SimpleNamespace(no_compile_layers=registry))
        topology = SimpleNamespace(mesh_context=SimpleNamespace(fsdp_moe_mesh={"ep": Mock()}))
        expert_config = GroupedExpertsConfig(2, 4, 3, num_layers=1)
        layers = torch.nn.ModuleList([RolloutEPGroupedExperts(config=expert_config) for _ in range(2)])
        states = torch.zeros(1, 4)
        ids, weights = torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1)

        def compute(states, counts, gate_up, down):
            self.assertTrue(gate_up.is_contiguous() and down.is_contiguous())
            return states + gate_up[0, 0, 0]

        def unpermute(slots, inverse, weights):
            routes = slots[inverse.long()].reshape(*weights.shape, -1)
            return (routes * weights.unsqueeze(-1)).sum(dim=1)

        backend = Mock(local_count=2)
        backend.token_dispatch.side_effect = lambda x, ids: (x, None, ids, None, len(x))
        backend.compute_experts.side_effect = compute
        backend.token_combine.side_effect = lambda x, *_: x.unsqueeze(1)
        with patch.dict("sys.modules", {"vllm.forward_context": context}), patch(
            "rl.unified_execution.vllm.moe.grouped_experts.AscendFusedExpertsBackend", return_value=backend
        ), patch("rl.unified_execution.vllm.moe.grouped_experts.moe_token_unpermute",
                 side_effect=unpermute), torch.no_grad():
            mesh = Mock()
            for layer in layers:
                layer.bind_parallel(SimpleNamespace(mesh_binding=mesh))
            register_expert_backend(layers, config, topology, "test")
            for i, layer in enumerate(layers):
                layer.gate_up_proj.fill_(i)
                graph = make_fx(layer)(states, ids, weights)
                layer.gate_up_proj.add_(2)
                torch.testing.assert_close(graph(states, ids, weights), states + i + 2)


class TestNativeExpertLayout(unittest.TestCase):
    """Cover sparse shard ownership and the distinction between rows and replicas."""

    def test_source_exchange_preserves_bits_and_pads_after_sp(self):
        """Source-row packets preserve integer IDs and zero padding on partial and empty ranks."""
        native = Mock(spec=AscendFusedExpertsBackend)
        native.common_source_rows = False
        native.dp_size, native.sp_size, native.source_count = 2, 2, 2
        native.source_group = object()
        native.config = SimpleNamespace(num_experts=513)
        context = SimpleNamespace(dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([3, 5])))
        states = torch.tensor([[-0., 1.], [2., -3.]], dtype=torch.bfloat16)
        ids = torch.tensor([[511, 1], [256, 17]], dtype=torch.int32)

        def gather(destination, packet, *, group):
            self.assertIs(group, native.source_group)
            destination.copy_(packet.repeat(2, 1))

        with patch("torch.distributed.all_gather_into_tensor", side_effect=gather):
            for rows in (2, 0):
                values, routes, bucket = AscendFusedExpertsBackend._gather_source_rows(
                    native, states[:rows], ids[:rows], context,
                )
                self.assertEqual(bucket, 3)
                for offset in (0, bucket):
                    self.assertTrue(torch.equal(values[offset:offset+rows].view(torch.int16),
                                                states[:rows].view(torch.int16)))
                    self.assertTrue(torch.equal(routes[offset:offset+rows], ids[:rows]))
                    self.assertTrue(routes[offset+rows:offset+bucket].eq(513).all())
