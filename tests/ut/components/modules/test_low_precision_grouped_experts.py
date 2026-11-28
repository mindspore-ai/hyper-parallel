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
"""CPU contracts for unified Policy/ModuleFamily LowPrecisionGroupedExperts."""

import copy
import os
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.components.modules import GroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import GroupedExpertsConfig
from hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts import (
    LowPrecisionGroupedExperts,
    replace_fake_w4a8_grouped_experts,
    replace_mxfp8_grouped_experts,
    replace_w4a8_grouped_experts,
)
from hyper_parallel.components.quantization.config import LowPrecisionDtypeScheme
from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import FakeW4A8GroupedLinear
from hyper_parallel.components.quantization.functional.hifloat8_gmm_func import HiFloat8GroupedLinear
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import MXFP8GroupedLinear
from hyper_parallel.components.quantization.functional.w4a8_gmm_func import W4A8GroupedLinear
from hyper_parallel.components.quantization.modules import GroupedExperts as PolicyGroupedExperts
from hyper_parallel.components.quantization.modules.replacement import replace_grouped_experts
from hyper_parallel.components.quantization.ops.npu_mxfp8 import LowPrecisionCapabilityError
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.grouped_experts_family import TinyMoE
from tests.common.qat_family import install_family, qat_config

_MODULE = "hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts."
_STRATEGIES = "hyper_parallel.components.quantization.functional.strategy_factory."


def _cpu_projection(inputs, weight, counts, strategy, group_list_type):
    """Keep autograd and replace only the device projection with dense CPU matmuls."""
    del strategy, group_list_type
    return torch.cat([part @ matrix.T for part, matrix in zip(inputs.split(counts.tolist()), weight)])


def _cpu_mxfp8_proj(inputs, weight, counts, quantizer):
    """Dense CPU oracle for transposed [C, K, N] MXFP8 projections."""
    del quantizer
    return torch.cat([part @ matrix for part, matrix in zip(inputs.split(counts.tolist()), weight)])


def _hif8_config():
    """Select HiF8 through the real named-scheme resolver and Family rule."""
    raw = qat_config().to_dict()
    raw["training"]["low_precision"]["dtype_schemes"]["experts"] = {
        "weight_format": "hif8", "act_format": "hif8",
    }
    return resolve_config(raw)


def _unaligned_model() -> TinyMoE:
    """Use expert dimensions that native HiF8 supports without an MX block-32 gate."""
    model = TinyMoE()
    model.config.hidden_size = 31
    model.config.moe_intermediate_size = 7
    model.mlp.config = model.config
    source = model.mlp.experts
    source.hidden_size, source.intermediate_size = 31, 7
    source.gate_up_proj = nn.Parameter(torch.randn(4, 14, 31, dtype=torch.float64) * 0.1)
    source.down_proj = nn.Parameter(torch.randn(4, 31, 7, dtype=torch.float64) * 0.1)
    source.act_fn = nn.SiLU()
    return model


class TestUnifiedLowPrecisionGroupedExperts(unittest.TestCase):
    """Exercise public Policy conversion, Family selection, before_fc2 mode, and error guards."""

    def setUp(self) -> None:
        """Keep randomness isolated and replace hardware probes only."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        for path in (
            _STRATEGIES + "validate_npu_gmm_runtime",
            _STRATEGIES + "validate_w4a8_gmm_runtime",
            _STRATEGIES + "validate_fake_w4a8_gmm_runtime",
            _STRATEGIES + "validate_hifloat8_gmm_runtime",
            _MODULE + "validate_npu_gmm_runtime",
            _MODULE + "validate_npu_swiglu_runtime",
        ):
            probe = patch(path)
            probe.start()
            self.addCleanup(probe.stop)

    def test_policy_and_hif8_family_share_canonical_layout_and_compute(self) -> None:
        """
        Feature: Unified Policy and HiF8 Family construction.
        Description: Resolve non-MX-aligned experts through Policy and ModuleFamily entry points.
        Expectation: Both retain [E, 2I, H] layout without copying and produce identical gradients.
        """
        model = _unaligned_model()
        policy_source = copy.deepcopy(model.mlp.experts)
        policy_source.register_buffer("persistent_state", torch.tensor(3))
        policy_source.register_buffer("scratch_state", torch.tensor(5), persistent=False)
        policy_source.extra = nn.Linear(2, 2)
        policy_source.eval()
        rng = torch.get_rng_state()
        policy = replace_grouped_experts(
            module=policy_source,
            module_fqn="mlp.experts",
            context={"low_precision": LowPrecisionDtypeScheme(weight_format="hif8", act_format="hif8")},
        )
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertIs(PolicyGroupedExperts, LowPrecisionGroupedExperts)
        self.assertIs(type(policy), LowPrecisionGroupedExperts)
        self.assertIs(policy.gate_up_proj, policy_source.gate_up_proj)
        self.assertIs(policy.down_proj, policy_source.down_proj)
        self.assertIs(policy.persistent_state, policy_source.persistent_state)
        self.assertIs(policy.scratch_state, policy_source.scratch_state)
        self.assertNotIn("scratch_state", policy.state_dict())
        self.assertFalse(policy.training)
        self.assertEqual(policy.make_transforms(), [])

        orig_gate_up = model.mlp.experts.gate_up_proj
        family = install_family(model, _hif8_config()).mlp.experts
        self.assertIs(type(family), LowPrecisionGroupedExperts)
        self.assertIs(type(family.grouped_linear), HiFloat8GroupedLinear)
        self.assertIs(family.gate_up_proj, orig_gate_up)
        self.assertEqual(tuple(family.gate_up_proj.shape), (4, 14, 31))
        self.assertEqual(len(family.make_transforms()), 2)

        inputs = torch.randn(5, 31, dtype=torch.float64, requires_grad=True)
        family_inputs = inputs.detach().clone().requires_grad_()
        counts = torch.tensor([2, 0, 1, 2])
        with patch(_MODULE + "_GroupedLinearFunction.apply", side_effect=_cpu_projection):
            policy_out = policy.forward_expert_major(inputs, counts)
            family_out = family.forward_expert_major(family_inputs, counts)
        torch.testing.assert_close(policy_out, family_out, rtol=1e-12, atol=1e-14)
        policy_grads = torch.autograd.grad(
            policy_out.square().sum(), (inputs, policy.gate_up_proj, policy.down_proj),
        )
        family_grads = torch.autograd.grad(
            family_out.square().sum(), (family_inputs, family.gate_up_proj, family.down_proj),
        )
        for p_grad, f_grad in zip(policy_grads, family_grads):
            torch.testing.assert_close(p_grad, f_grad, rtol=1e-12, atol=1e-14)

    def test_before_fc2_and_unfused_mxfp8_compute_modes(self) -> None:
        """
        Feature: before_fc2 routing-weight mode and unfused MXFP8 compute_experts.
        Description: Exercise before_fc2 weighting on both strategy and unfused mxfp8 paths.
        Expectation: Slot probabilities scale intermediate activations before FC2; missing weights raise.
        """
        cfg = GroupedExpertsConfig(2, 32, 32, routing_weight_mode="before_fc2", num_layers=1)
        mxfp8_expert = GroupedExperts(config=cfg, use_mxfp8=True).to(torch.float64)
        hif8_expert = GroupedExperts(
            config=cfg, weight_format="hif8", act_format="hif8",
        ).to(torch.float64)
        hif8_expert.load_state_dict(mxfp8_expert.state_dict())

        x = torch.randn(3, 32, dtype=torch.float64, requires_grad=True)
        x2 = x.detach().clone().requires_grad_()
        counts = torch.tensor([2, 1])
        weights = torch.tensor([0.2, 0.8, 0.5], dtype=torch.float64, requires_grad=True)
        w2 = weights.detach().clone().requires_grad_()
        with (
            patch(_MODULE + "mxfp8_projection", side_effect=_cpu_mxfp8_proj),
            patch(_MODULE + "_GroupedLinearFunction.apply", side_effect=_cpu_projection),
        ):
            out1 = mxfp8_expert.compute_experts(
                x, counts, mxfp8_expert.gate_up_proj, mxfp8_expert.down_proj, weights,
            )
            out2 = hif8_expert.compute_experts(
                x2, counts, hif8_expert.gate_up_proj, hif8_expert.down_proj, w2,
            )
            with self.assertRaisesRegex(ValueError, "before_fc2 requires"):
                mxfp8_expert.compute_experts(
                    x, counts, mxfp8_expert.gate_up_proj, mxfp8_expert.down_proj, None,
                )
        torch.testing.assert_close(out1, out2, rtol=1e-12, atol=1e-14)

    def test_explicit_factories_dimension_constructor_and_validation_errors(self) -> None:
        """
        Feature: Explicit format factories, dimension constructor, and input validation.
        Description: Exercise standalone factories, Policy constructor, and invalid count/shape guards.
        Expectation: Factories preserve caller context; invalid arguments raise clear exceptions.
        """
        context = {"low_precision": LowPrecisionDtypeScheme(weight_format="hif8", act_format="hif8")}
        for factory, expected in (
            (replace_mxfp8_grouped_experts, MXFP8GroupedLinear),
            (replace_w4a8_grouped_experts, W4A8GroupedLinear),
            (replace_fake_w4a8_grouped_experts, FakeW4A8GroupedLinear),
        ):
            with self.subTest(factory=factory.__name__):
                expert = factory(module=TinyMoE().mlp.experts, module_fqn="experts", context=context)
                self.assertIs(type(expert), LowPrecisionGroupedExperts)
                self.assertIs(type(expert.grouped_linear), expected)
                self.assertEqual(context["low_precision"].weight_format, "hif8")

        dim_expert = PolicyGroupedExperts(2, 4, 6, grouped_linear=HiFloat8GroupedLinear(), fqn="experts")
        self.assertEqual(tuple(dim_expert.gate_up_proj.shape), (2, 12, 4))
        self.assertEqual(tuple(dim_expert.down_proj.shape), (2, 4, 6))
        self.assertEqual(dim_expert.fqn, "experts")

        with self.assertRaises(TypeError):
            PolicyGroupedExperts(2, 4, grouped_linear=HiFloat8GroupedLinear())
        with self.assertRaises(TypeError):
            PolicyGroupedExperts(2, 4, 6, grouped_linear=nn.Linear(4, 4))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "counts must contain one count"):
            dim_expert.forward_expert_major(torch.ones(2, 4), torch.tensor([2]))
        with self.assertRaisesRegex(ValueError, "counts must be integers"):
            dim_expert.forward_expert_major(torch.ones(2, 4), torch.tensor([1.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "non-negative and sum"):
            dim_expert.forward_expert_major(torch.ones(2, 4), torch.tensor([1, 2]))
        with patch(
            _MODULE + "_GroupedLinearFunction.apply",
            side_effect=LowPrecisionCapabilityError("kernel unavailable"),
        ):
            with self.assertRaisesRegex(LowPrecisionCapabilityError, "'experts' cannot run"):
                dim_expert.forward_expert_major(torch.ones(2, 4), torch.tensor([1, 1]))

    def test_public_policy_import_does_not_load_optional_npu_backend(self) -> None:
        """
        Feature: Optional NPU dependency isolation.
        Description: Import and construct the public Policy adapter with torch_npu autoload disabled.
        Expectation: Canonical conversion preserves parameters without importing torch_npu.
        """
        script = textwrap.dedent("""
            import sys
            sys.modules["torch_npu"] = None
            import torch
            from torch import nn
            from hyper_parallel.components.quantization import GroupedExperts
            from hyper_parallel.components.quantization.functional.hifloat8_gmm_func import HiFloat8GroupedLinear
            from hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts import (
                LowPrecisionGroupedExperts,
            )

            source = nn.Module()
            source.gate_up_proj = nn.Parameter(torch.ones(2, 6, 4))
            source.down_proj = nn.Parameter(torch.ones(2, 4, 3))
            converted = GroupedExperts.from_module(source, fqn="experts", grouped_linear=HiFloat8GroupedLinear())
            assert type(converted) is LowPrecisionGroupedExperts
            assert converted.gate_up_proj is source.gate_up_proj
            assert sys.modules["torch_npu"] is None
        """)
        env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
        result = subprocess.run(
            [sys.executable, "-c", script], check=False, capture_output=True, text=True, env=env,
        )
        self.assertEqual(result.returncode, 0, f"exit={result.returncode}, stderr={result.stderr}")
