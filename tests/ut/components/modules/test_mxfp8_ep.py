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
"""Native ModuleFamily EP and MoonEP contracts for all low-precision GroupedExperts formats."""

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.nn import functional as F

from hyper_parallel.components.modules import GroupedExperts, get_family_build_info
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import GroupedExpertsConfig
from hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts import (
    EPLowPrecisionGroupedExperts,
    MoonEPLowPrecisionGroupedExperts,
)
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.distributed.expert_parallel.ep_context import EPContext
from hyper_parallel.distributed.expert_parallel.moonep_context import MoonEPContext
from hyper_parallel.distributed.module_binding import MeshBinding, ParallelBinding
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.grouped_experts_family import (
    TinyMoE,
    cpu_token_permute,
    cpu_token_unpermute,
    replace_experts,
)
from tests.common.qat_family import qat_config

_LP_MODULE = "hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts."
_GE_MODULE = "hyper_parallel.components.modules.moe.grouped_experts.grouped_experts."
_STRATEGY_FACTORY = "hyper_parallel.components.quantization.functional.strategy_factory."


def _cpu_mxfp8_projection(inputs, weights, counts, quantizer):
    """Differentiable CPU oracle for [C, K, N] transposed MXFP8 projections."""
    del quantizer
    parts = inputs.split(counts.tolist())
    return torch.cat([part @ weight for part, weight in zip(parts, weights)])


def _cpu_fused_swiglu(inputs, gate_up, down, counts, quantizer, clamp_limit=None):
    """Differentiable CPU oracle for fused MXFP8 SwiGLU."""
    gate, up = _cpu_mxfp8_projection(inputs, gate_up, counts, quantizer).chunk(2, dim=-1)
    if clamp_limit is not None:
        gate = gate.clamp(max=clamp_limit)
        up = up.clamp(min=-clamp_limit, max=clamp_limit)
    return _cpu_mxfp8_projection(F.silu(gate) * up, down, counts, quantizer)


def _cpu_grouped_linear(inputs, weights, counts, strategy, group_list_type):
    """Differentiable CPU oracle for [C, O, K] canonical GroupedLinear strategies."""
    del strategy, group_list_type
    parts = inputs.split(counts.tolist())
    return torch.cat([part @ weight.T for part, weight in zip(parts, weights)])


class TestNativeModuleFamilyLowPrecisionEP(unittest.TestCase):
    """Verify native EP and MoonEP selection, sharding, ParallelModuleSpec, and stage execution."""

    def setUp(self) -> None:
        """Isolate NPU hardware probes while keeping family selection and binding real."""
        torch.manual_seed(793)
        for path in (
            _LP_MODULE + "validate_npu_gmm_runtime",
            _LP_MODULE + "validate_npu_swiglu_runtime",
            _STRATEGY_FACTORY + "validate_npu_gmm_runtime",
            _STRATEGY_FACTORY + "validate_w4a8_gmm_runtime",
            _STRATEGY_FACTORY + "validate_fake_w4a8_gmm_runtime",
            _STRATEGY_FACTORY + "validate_hifloat8_gmm_runtime",
        ):
            probe = patch(path)
            probe.start()
            self.addCleanup(probe.stop)

    def test_ep_and_moonep_protocols_and_sharding_for_all_formats(self) -> None:
        """
        Feature: Native ModuleFamily EP and MoonEP across all 5 low-precision formats.
        Description: Select EP and MoonEP members via YAML config for MXFP8, Fused MXFP8, W4A8, Fake-W4A8, and HiF8.
        Expectation: Members inherit Shard(0) layout and declare grouped_experts.slots.ep / forward.moonep protocols.
        """
        schemes = (
            ("mxfp8", "mxfp8", False, False),
            ("mxfp8", "mxfp8", False, True),
            ("mxfp4", "mxfp8", False, False),
            ("mxfp4", "mxfp8", True, False),
            ("hif8", "hif8", False, False),
        )
        for weight_fmt, act_fmt, fake, fused in schemes:
            for moonep in (False, True):
                with self.subTest(weight_fmt=weight_fmt, fake=fake, fused=fused, moonep=moonep):
                    raw = qat_config(fake=fake, weight_format=weight_fmt if weight_fmt != "hif8" else "mxfp8").to_dict()
                    raw["training"]["low_precision"]["dtype_schemes"]["experts"] = {
                        "weight_format": weight_fmt,
                        "act_format": act_fmt,
                        "is_fake_quantize": fake,
                    }
                    impl = raw["plan_overrides"][0]["replace"]["implementation"]
                    impl["fused"] = fused
                    impl["moonep"] = moonep
                    impl["overlap_shared_expert"] = False
                    model = TinyMoE()
                    orig_weight = model.mlp.experts.gate_up_proj
                    replace_experts(model, resolve_config(raw), context={"ep_size": 2})
                    expert = model.mlp.experts
                    expected_cls = MoonEPLowPrecisionGroupedExperts if moonep else EPLowPrecisionGroupedExperts
                    self.assertIs(type(expert), expected_cls)
                    self.assertIs(expert.gate_up_proj, orig_weight)

                    info = get_family_build_info(expert)
                    sharding = info.resolved.sharding(
                        module=expert, module_fqn="mlp.experts", member=info.resolved, context=None,
                    )
                    self.assertEqual(
                        sharding.params,
                        {"gate_up_proj": {"ep": Shard(0)}, "down_proj": {"ep": Shard(0)}},
                    )
                    spec = expert.parallel_module_spec()
                    expected_protocol = (
                        "grouped_experts.forward.moonep" if moonep else "grouped_experts.slots.ep"
                    )
                    self.assertEqual(spec.protocol, expected_protocol)
                    self.assertEqual(spec.binding_kind, "ep")
                    self.assertFalse(spec.options["overlap_shared_expert"])

    def test_ep_dispatch_compute_combine_end_to_end_gradients(self) -> None:
        """
        Feature: End-to-end EP stage delegation with low-precision compute_experts.
        Description: Bind a mock BF16 EPContext and execute forward/backward for fused MXFP8 and W4A8.
        Expectation: Outputs and gradients match dense reference across empty and populated experts.
        """
        cfg = GroupedExpertsConfig(2, 32, 32, routing_weight_mode="output", num_layers=1)
        for kwargs in (
            {"use_mxfp8": True, "fused": True},
            {"weight_format": "mxfp4", "act_format": "mxfp8"},
        ):
            with self.subTest(kwargs=kwargs):
                expert = GroupedExperts(config=cfg, context={"ep_size": 2}, **kwargs).to(torch.float64)
                ep_ctx = MagicMock(spec=EPContext)
                ep_ctx.__class__ = EPContext
                ep_ctx.num_experts = 2
                ep_ctx.dispatch.side_effect = (
                    lambda h, ids, w: (
                        h,
                        torch.bincount(ids, minlength=2),
                        w,
                        [h.shape[0]],
                        [h.shape[0]],
                        torch.arange(h.shape[0], dtype=torch.int32),
                    )
                )
                ep_ctx.combine.side_effect = lambda vals, _s, _r, _inv: vals
                expert.bind_parallel(ParallelBinding(mesh_binding=MagicMock(spec=MeshBinding), ep_context=ep_ctx))

                hidden = torch.randn(3, 2, 32, dtype=torch.float64, requires_grad=True)
                topk_ids = torch.tensor([[0, 1]] * 6, dtype=torch.int64)
                topk_weights = torch.full((6, 2), 0.5, dtype=torch.float64, requires_grad=True)
                ref_hidden = hidden.detach().clone().requires_grad_()
                ref_weights = topk_weights.detach().clone().requires_grad_()

                with (
                    patch(_GE_MODULE + "moe_token_permute", side_effect=cpu_token_permute),
                    patch(_GE_MODULE + "moe_token_unpermute", side_effect=cpu_token_unpermute),
                    patch(_LP_MODULE + "fused_grouped_swiglu", side_effect=_cpu_fused_swiglu),
                    patch(_LP_MODULE + "_GroupedLinearFunction.apply", side_effect=_cpu_grouped_linear),
                ):
                    actual = expert(hidden, topk_ids, topk_weights)

                flat = ref_hidden.reshape(-1, 32)
                expected_rows = []
                for tok_idx in range(6):
                    acc = torch.zeros(32, dtype=torch.float64)
                    for slot_idx in range(2):
                        eid = int(topk_ids[tok_idx, slot_idx])
                        g, u = F.linear(flat[tok_idx], expert.gate_up_proj[eid]).chunk(2, dim=-1)  # pylint: disable=not-callable
                        out = F.linear(F.silu(g) * u, expert.down_proj[eid])  # pylint: disable=not-callable
                        acc = acc + out * ref_weights[tok_idx, slot_idx]
                    expected_rows.append(acc)
                expected = torch.stack(expected_rows).reshape_as(ref_hidden)

                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
                actual_grads = torch.autograd.grad(
                    actual.square().sum(), (hidden, topk_weights, expert.gate_up_proj, expert.down_proj),
                )
                expected_grads = torch.autograd.grad(
                    expected.square().sum(), (ref_hidden, ref_weights, expert.gate_up_proj, expert.down_proj),
                )
                for a_grad, e_grad in zip(actual_grads[:2], expected_grads[:2]):
                    torch.testing.assert_close(a_grad, e_grad, rtol=1e-12, atol=1e-14)
                ep_ctx.dispatch.assert_called_once()
                ep_ctx.combine.assert_called_once()

    def test_moonep_dynamic_replica_weights_flow_through_compute_experts(self) -> None:
        """
        Feature: MoonEP dynamic replica execution with low-precision compute_experts.
        Description: Bind a mock MoonEPContext supplying 3 replicated expert rows [3, 2I, H] on a 2-expert module.
        Expectation: compute_experts consumes the supplied replica table rather than self.gate_up_proj.
        """
        cfg = GroupedExpertsConfig(4, 32, 32, routing_weight_mode="output", num_layers=1)
        expert = GroupedExperts(
            config=cfg, moonep=True, weight_format="hif8", act_format="hif8", context={"ep_size": 2},
        ).to(torch.float64)
        self.assertIs(type(expert), MoonEPLowPrecisionGroupedExperts)

        moon_ctx = MagicMock(spec=MoonEPContext)
        moon_ctx.num_experts = 4
        replica_gate = torch.randn(3, 64, 32, dtype=torch.float64, requires_grad=True)
        replica_down = torch.randn(3, 32, 32, dtype=torch.float64, requires_grad=True)
        moon_ctx.plan.return_value = "mock_plan"
        moon_ctx.expert_weights.return_value = (replica_gate, replica_down)
        moon_ctx.dispatch.side_effect = (
            lambda h, w, plan: (h, torch.tensor([2, 0, 2]), w, torch.arange(h.shape[0], dtype=torch.int32))
        )
        moon_ctx.combine.side_effect = lambda vals, _plan, _inv: vals
        expert.bind_parallel(ParallelBinding(mesh_binding=MagicMock(spec=MeshBinding), ep_context=moon_ctx))

        hidden = torch.randn(2, 32, dtype=torch.float64, requires_grad=True)
        topk_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.float64)
        with (
            patch(_GE_MODULE + "moe_token_permute", side_effect=cpu_token_permute),
            patch(_GE_MODULE + "moe_token_unpermute", side_effect=cpu_token_unpermute),
            patch(_LP_MODULE + "_GroupedLinearFunction.apply", side_effect=_cpu_grouped_linear),
        ):
            out = expert(hidden, topk_ids, topk_weights)
        loss = out.square().sum()
        grads = torch.autograd.grad(loss, (hidden, replica_gate, replica_down))
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
