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
"""TP/SP and CP contracts for the EP-owned DSV4.1 MegaMoe adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock

from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.distributed import ShardingPlanner
from hyper_parallel.models._transformers.model_builder import _build_replacement_context
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import DeepseekV41MegaMoeExperts
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_megamoe_compute_fn,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41TopKRouter
from tests.ut.auto_models.models.deepseek_v41.test_megamoe import _native_reference, _source


class _Mesh:
    """Metadata-only mesh; unit tests never create process groups."""

    def __init__(self, shape: tuple[int, ...], names: tuple[str, ...], rank: int = 0) -> None:
        """Record topology metadata without initializing a distributed backend."""
        self.mesh_shape = shape
        self.mesh_dim_names = names
        self.ndim = len(shape)
        self.rank = rank

    def size(self, mesh_dim: str | None = None) -> int:
        """Return a requested axis or the total mesh size."""
        if mesh_dim is not None:
            return self.mesh_shape[self.mesh_dim_names.index(mesh_dim)]
        result = 1
        for size in self.mesh_shape:
            result *= size
        return result

    def __getitem__(self, axis: str | tuple[str, ...]) -> Any:
        """Return a metadata-only view of the selected axes."""
        names = (axis,) if isinstance(axis, str) else axis
        return _Mesh(tuple(self.size(name) for name in names), names, self.rank)

    def get_local_rank(self, axis: str | None = None) -> int:
        """Return the selected rank for mask alignment tests."""
        del axis
        return self.rank

    def get_group(self, axis: str | None = None) -> None:
        """Keep native resources inactive during metadata-only tests."""
        del axis


class TestMegaMoeParallel(unittest.TestCase):
    """Check ownership and real router/shared-expert outputs and gradients."""

    def test_replacement_context_exposes_sequence_parallel(self) -> None:
        """Replacement guards must read the actual configured SP flag."""
        for enabled in (False, True):
            setup = SimpleNamespace(mesh_context=SimpleNamespace(
                tp_size=2, cp_size=2, ep_size=4, pp_size=1, sequence_parallel=enabled,
            ))
            context = _build_replacement_context(setup, None)
            self.assertIs(context['sequence_parallel'], enabled)
            self.assertTrue(context['tp'])
            self.assertTrue(context['cp'])
        context = _build_replacement_context(SimpleNamespace(), None)
        for axis in ('tp', 'cp', 'ep', 'pp', 'sequence_parallel'):
            self.assertFalse(context[axis])

    def test_guard_unsupported_parallel_layouts(self) -> None:
        """Fail before execution for PP, non-SP TP, and expert tensor parallelism."""
        cases = (
            ({'pp': True}, 'PP=1'),
            ({'tp': True, 'ep': True, 'sequence_parallel': False}, 'sequence_parallel'),
            ({'tp': True, 'ep': False, 'sequence_parallel': True}, 'EP>1'),
        )
        for context, message in cases:
            with self.subTest(context=context), self.assertRaisesRegex(ValueError, message):
                DeepseekV41MegaMoeExperts(module=_source(), context=context)

    def test_tp_cp_local_rows_router_shared_outputs_and_gradients(self) -> None:
        """CP is already local; align the VLM mask to each TP/SP token interval."""
        for tp, cp in ((1, 2), (2, 1), (2, 2)):
            for mode in ('push', 'pull'):
                for vision in (False, True):
                    for batch, count in ((1, 0), (1, 3), (1, 17), (2, 17)):
                        with self.subTest(tp=tp, cp=cp, mode=mode, vision=vision, rows=count, batch=batch):
                            self._check_local_compute(tp, cp, mode, vision, count, batch)

    def _check_local_compute(self, tp: int, cp: int, mode: str, vision: bool, count: int, batch: int) -> None:
        """Check local token intervals against independent expert outputs and gradients."""
        torch.manual_seed(83)
        context = {'tp': tp > 1, 'cp': cp > 1, 'ep': True, 'sequence_parallel': tp > 1}
        experts = DeepseekV41MegaMoeExperts(module=_source(), context=context,
                                          local_num_tokens=128, dispatch_mode=mode)
        self.addCleanup(experts.close_runtime)
        gate = DeepseekV41TopKRouter(SimpleNamespace(
            hidden_size=8, num_local_experts=4, num_experts_per_tok=2,
            scoring_func='sigmoid', routed_scaling_factor=1.0, v41_vision_enabled=vision,
        ))
        with torch.no_grad():
            gate.weight.normal_(std=0.1)
            gate.bias.copy_(torch.tensor([2., 2., 0., 0.]))
            if vision:
                gate.bias_vl.copy_(torch.tensor([0., 0., 2., 2.]))
        shared = nn.Linear(8, 8)
        shared.limit = 10.0
        signature = (lambda hidden_states, input_ids=None, image_mask=None: hidden_states) if vision else (
            lambda hidden_states, input_ids=None: hidden_states)
        module = SimpleNamespace(experts=experts, gate=gate, shared_experts=shared,
                                 is_hash=False, forward=signature)
        tp_rank = tp - 1
        with patch('hyper_parallel.core.multicore.modules.mega_moe.module.shmem.root_members', return_value=(0, 1)):
            compute = deepseek_v41_megamoe_compute_fn(
                module=module, mesh=None, tp_mesh=_Mesh((tp,), ('tp',), tp_rank) if tp > 1 else None,
                cp_mesh=_Mesh((cp,), ('cp',)) if cp > 1 else None, ep_mesh=_Mesh((2,), ('ep',)),
            )
        hidden = torch.randn(batch, count, 8, requires_grad=True)
        mask = (torch.arange(batch * count * tp).reshape(batch, count * tp) % 3 == 0) if vision else None
        expected_mask = mask[:, tp_rank * count:(tp_rank + 1) * count] if vision else None
        _, weights, indices = gate(hidden, image_mask=expected_mask)
        expected = _native_reference(experts, hidden, indices, weights,
                                     expert_weights=(experts.gate_up_proj, experts.down_proj)) + shared(hidden)
        kwargs = {'image_mask': mask} if vision else {}
        if vision:
            handle = gate.register_forward_pre_hook(
                lambda _module, _args, gate_kwargs: self.assertTrue(gate_kwargs['image_mask'].is_contiguous()),
                with_kwargs=True,
            )
            self.addCleanup(handle.remove)
        with patch.object(MegaMoeExperts, 'forward', _native_reference):
            actual = compute(module, hidden, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        targets = (hidden, gate.weight, experts.gate_up_proj, experts.down_proj, shared.weight, shared.bias)
        derivative = torch.randn_like(actual)
        observed = torch.autograd.grad(actual, targets, derivative)
        reference = torch.autograd.grad(expected, targets, derivative)
        for actual_grad, expected_grad in zip(observed, reference):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)

    def test_planner_keeps_expert_matrices_ep_only(self) -> None:
        """Use the production planner: experts stay full-width, shared experts use TP."""
        for tp, cp in ((1, 2), (2, 1), (2, 2)):
            with self.subTest(tp=tp, cp=cp):
                config = DeepseekV4Config.from_dict({
                    'hidden_size': 8, 'moe_intermediate_size': 4, 'num_hidden_layers': 1,
                    'num_local_experts': 4, 'num_experts_per_tok': 2, 'mlp_layer_types': ['moe'],
                    'swiglu_limit': 10.0, 'scoring_func': 'sigmoid', 'routed_scaling_factor': 1.0,
                })
                model = nn.Module()
                model.config = config
                model.model = nn.Module()
                layer = nn.Module()
                layer.mlp = DeepseekV4SparseMoeBlock(config, 0)
                layer.mlp.experts = DeepseekV41MegaMoeExperts(
                    module=layer.mlp.experts, context={'tp': tp > 1, 'cp': cp > 1, 'ep': True,
                                                     'sequence_parallel': tp > 1},
                )
                model.model.layers = nn.ModuleList([layer])
                plan = ShardingPlanner().plan(
                    model, _Mesh((4 // (tp * cp), cp, tp), ('dp', 'cp', 'tp')),
                    tp_size=tp, cp_size=cp, ep_size=4, sequence_parallel=tp > 1,
                )
                spec = plan.modules['model.layers.0.mlp']
                for name in ('experts.gate_up_proj', 'experts.down_proj'):
                    self.assertEqual(spec.params[name]['ep'], Shard(0))
                    self.assertNotIn('tp', spec.params[name])
                for layout in (*spec.in_dst.values(), *spec.out_src.values(), *spec.out_dst.values()):
                    if tp > 1:
                        self.assertEqual(layout['tp'], Shard(1))
                    if cp > 1:
                        self.assertEqual(layout['cp'], Shard(1))
                shared = plan.modules['model.layers.0.mlp.shared_experts']
                if tp > 1:
                    self.assertEqual(shared.params['gate_proj.weight']['tp'], Shard(0))
                    self.assertEqual(shared.params['down_proj.weight']['tp'], Shard(1))
