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
"""MXFP8 Family selection and local EP contracts without device operators."""

import copy
import unittest
from itertools import product
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.components.modules.mxfp8_grouped_swiglu import (
    FusedMXFP8GroupedSwiGLU,
    MXFP8GroupedSwiGLU,
)
from hyper_parallel.distributed.expert_parallel.experts import (
    _run_ep_local_experts,
    bind_local_expert_forward,
)
from hyper_parallel.trainer.config import entries_to_plan_overrides
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.grouped_experts_family import TinyMoE, family_config, replace_experts
from tests.common.mark_utils import arg_mark


_LOCAL_MATH = "hyper_parallel.components.modules.mxfp8_grouped_swiglu."
_RECIPES = (
    "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn",
    "hyper_parallel.distributed.expert_parallel.recipes.deepseekv3_ep_compute_fn",
)


class _ExpertMesh:
    """Only the read-only mesh metadata needed to build an EP recipe."""

    def __getitem__(self, name: str) -> "_ExpertMesh":
        if name != "ep":
            raise KeyError(name)
        return self

    @staticmethod
    def size() -> int:
        """Return the tested EP degree."""
        return 2

    @staticmethod
    def get_group(name: str) -> str:
        """Return a placeholder; construction does not execute collectives."""
        return name


def _projection(inputs, weights, counts, quantizer):
    """Differentiable CPU oracle replacing only the device projection operator."""
    del quantizer
    parts = inputs.split(counts.tolist())
    return torch.cat([part @ weight for part, weight in zip(parts, weights)])


def _fused_swiglu(inputs, gate_up, down, counts, quantizer):
    """CPU oracle for the fused operator with the same expert-major boundary."""
    gate, up = _projection(inputs, gate_up, counts, quantizer).chunk(2, dim=-1)
    return _projection(F.silu(gate) * up, down, counts, quantizer)


class TestMXFP8ExpertParallel(unittest.TestCase):
    """Exercise real replacement, recipe and binder selection on CPU."""

    def setUp(self) -> None:
        """Isolate hardware checks while retaining production construction."""
        torch.manual_seed(793)
        for name in ("validate_npu_gmm_runtime", "validate_npu_swiglu_runtime"):
            probe = patch(_LOCAL_MATH + name)
            probe.start()
            self.addCleanup(probe.stop)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_ep_replacement_preserves_family_and_weights(self) -> None:
        """
        Feature: MXFP8 Family with expert parallelism.
        Description: Execute both replacements with an active EP context.
        Expectation: The selected variant retains the canonical persistent parameters.
        """
        for fused, expected in ((False, MXFP8GroupedSwiGLU), (True, FusedMXFP8GroupedSwiGLU)):
            with self.subTest(fused=fused):
                model = TinyMoE(canonical=True)
                weight = model.mlp.experts.gate_up_proj
                replace_experts(model, family_config(use_mxfp8=True, use_fused_mxfp8=fused), context={"ep": True})
                self.assertIs(type(model.mlp.experts), expected)
                self.assertIs(model.mlp.experts.gate_up_proj, weight)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_configured_recipes_enable_grouped_mxfp8(self) -> None:
        """
        Feature: Configured EP grouped selection.
        Description: Build Qwen3 and DeepSeek recipes through the real configuration parser.
        Expectation: Explicit grouped configuration binds either MXFP8 variant.
        """
        for recipe, fused in product(_RECIPES, (False, True)):
            with self.subTest(recipe=recipe, fused=fused):
                raw = family_config(use_mxfp8=True, use_fused_mxfp8=fused).to_dict()
                target = {"_target_": recipe, "use_grouped_gemm": True}
                raw["plan_overrides"].append({"match": "mlp", "when": "ep", "local_compute_fn": target})
                config = resolve_config(raw)
                model = TinyMoE()
                model.mlp.shared_experts = nn.Identity()
                replace_experts(model, config, context={"ep": True})
                factory = entries_to_plan_overrides(config.plan_overrides, ep_size=2)["mlp"].local_compute_fn
                kwargs = {"module": model.mlp, "mesh": None, "tp_mesh": None,
                          "cp_mesh": None, "ep_mesh": _ExpertMesh()}
                self.assertTrue(callable(factory.build(**kwargs)))
                self.assertTrue(model.mlp.experts._ep_use_grouped_gemm)  # pylint: disable=protected-access
                self.assertEqual(model.mlp.experts.local_expert_count, 2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_dispatch_calls_experts_and_restores_token_order(self) -> None:
        """
        Feature: Local grouped EP call protocol.
        Description: Run real binding and local dispatch with CPU device-operator oracles.
        Expectation: Hooks, counts, output order and gradients preserve outer routing weights.
        """
        for fused in (False, True):
            with self.subTest(fused=fused):
                source = TinyMoE()
                model = copy.deepcopy(source)
                replace_experts(model, family_config(use_mxfp8=True, use_fused_mxfp8=fused), context={"ep": True})
                bind_local_expert_forward(model.mlp, 1, use_grouped_gemm=True)
                experts = model.mlp.experts
                events = []
                experts.register_forward_pre_hook(lambda *_, events=events: events.append("enter"))
                experts.register_forward_hook(lambda *_, events=events: events.append("exit"))
                indices = torch.tensor([3, 0, 3, 2, 0])
                inputs = torch.randn(5, 128, requires_grad=True)
                reference_inputs = inputs.detach().clone().requires_grad_()
                scores = torch.tensor([0.2, 0.7, 0.3, 0.6, 0.4], requires_grad=True)
                reference_scores = scores.detach().clone().requires_grad_()
                with patch(_LOCAL_MATH + "mxfp8_projection", side_effect=_projection) as projection, \
                     patch(_LOCAL_MATH + "fused_grouped_swiglu", side_effect=_fused_swiglu) as fusion, \
                     patch("hyper_parallel.distributed.expert_parallel.experts.ep_all_to_all",
                           side_effect=lambda values, *_: values):
                    output = _run_ep_local_experts(model.mlp, inputs, indices[:, None], [5], [5], None, 0)
                self.assertEqual(events, ["enter", "exit"])
                self.assertEqual(projection.call_count, 0 if fused else 2)
                self.assertEqual(fusion.call_count, 1 if fused else 0)
                call = fusion.call_args if fused else projection.call_args_list[0]
                counts = call.args[3 if fused else 2]
                torch.testing.assert_close(counts, torch.tensor([2, 0, 1, 2]))
                actual = output * scores[:, None]
                expected = source.mlp.experts(reference_inputs, indices[:, None], reference_scores[:, None])
                torch.testing.assert_close(actual, expected)
                actual.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(inputs.grad, reference_inputs.grad)
                torch.testing.assert_close(scores.grad, reference_scores.grad)
                for name in ("gate_up_proj", "down_proj"):
                    torch.testing.assert_close(getattr(experts, name).grad,
                                               getattr(source.mlp.experts, name).grad.transpose(-2, -1))
