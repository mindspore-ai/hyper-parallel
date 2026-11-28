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
"""QAT GroupedExperts Family contracts across MXFP8, W4A8, Fake-W4A8, HiF8, and DeepSeek V4.1."""

import gc
import unittest
from unittest.mock import patch
import weakref

import torch
import yaml
from torch import nn

from hyper_parallel.components.modules.linear import LowPrecisionLinear
from hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts import (
    EPLowPrecisionGroupedExperts,
    LowPrecisionGroupedExperts,
    MoonEPLowPrecisionGroupedExperts,
)
from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import FakeW4A8GroupedLinear
from hyper_parallel.components.quantization.functional.hifloat8_gmm_func import HiFloat8GroupedLinear
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import MXFP8GroupedLinear
from hyper_parallel.components.quantization.functional.mxfp8_linear_func import MXFP8LinearStrategy
from hyper_parallel.components.quantization.functional.w4a8_gmm_func import W4A8GroupedLinear
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config.parallelism import entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import ConfigResolutionError, replace_override_path, resolve_config
from tests.common.grouped_experts_family import TinyMoE
from tests.common.mark_utils import arg_mark
from tests.common.qat_family import deepseek_experts_model, deepseek_qat_config, install_family, qat_config

_LOW_PRECISION = "hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts."
_STRATEGY_FACTORY = "hyper_parallel.components.quantization.functional.strategy_factory."


class TestQATGroupedExperts(unittest.TestCase):
    """Mock device capability checks only; registration and strategy selection stay real."""

    def setUp(self) -> None:
        """Isolate NPU availability without replacing the selected computation strategy."""
        for name in (
            "validate_npu_gmm_runtime",
            "validate_w4a8_gmm_runtime",
            "validate_fake_w4a8_gmm_runtime",
            "validate_hifloat8_gmm_runtime",
        ):
            probe = patch(_STRATEGY_FACTORY + name)
            probe.start()
            self.addCleanup(probe.stop)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_scheme_selection_local_ep_and_moonep(self) -> None:
        """
        Feature: Scheme-selected GroupedExperts across local, EP, and MoonEP.
        Description: Resolve W8A8, W4A8, Fake-W4A8, and HiF8 schemes under EP=1 and EP=2.
        Expectation: Canonical [E, 2I, H] parameters are retained and bound to the exact strategy.
        """
        cases = (
            (False, "mxfp8", MXFP8GroupedLinear),
            (False, "mxfp4", W4A8GroupedLinear),
            (True, "mxfp4", FakeW4A8GroupedLinear),
        )
        for fake, fmt, expected_strategy in cases:
            with self.subTest(fake=fake, fmt=fmt):
                model = TinyMoE()
                weight = model.mlp.experts.gate_up_proj
                weight.requires_grad_(False)
                install_family(model, qat_config(fake=fake, weight_format=fmt))
                expert = model.mlp.experts
                self.assertIs(type(expert), LowPrecisionGroupedExperts)
                self.assertIs(type(expert.grouped_linear), expected_strategy)
                self.assertIs(expert.gate_up_proj, weight)
                self.assertFalse(weight.requires_grad)
                with patch(
                    _LOW_PRECISION + "_GroupedLinearFunction.apply",
                    side_effect=lambda x, w, *_: x @ w[0].T,
                ):
                    out = expert.forward_expert_major(
                        torch.randn(4, 128), torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    )
                self.assertEqual(tuple(out.shape), (4, 128))

        raw_hif8 = qat_config().to_dict()
        raw_hif8["training"]["low_precision"]["dtype_schemes"]["experts"] = {
            "weight_format": "hif8", "act_format": "hif8",
        }
        raw_hif8["plan_overrides"][0]["replace"]["implementation"]["moonep"] = True
        cfg_hif8 = resolve_config(raw_hif8)
        rules = entries_to_module_replacements(
            cfg_hif8.plan_overrides, low_precision_config=cfg_hif8.training.low_precision,
        )
        ep_model = TinyMoE()
        apply_module_replacements(
            ep_model, compile_module_replacements(ep_model, rules), weights_mapping=[], context={"ep_size": 2},
        )
        self.assertIs(type(ep_model.mlp.experts), MoonEPLowPrecisionGroupedExperts)
        self.assertIs(type(ep_model.mlp.experts.grouped_linear), HiFloat8GroupedLinear)

        raw_w4 = qat_config(fake=False, weight_format="mxfp4").to_dict()
        cfg_w4 = resolve_config(raw_w4)
        rules_w4 = entries_to_module_replacements(
            cfg_w4.plan_overrides, low_precision_config=cfg_w4.training.low_precision,
        )
        ep_w4_model = TinyMoE()
        apply_module_replacements(
            ep_w4_model, compile_module_replacements(ep_w4_model, rules_w4),
            weights_mapping=[], context={"ep_size": 2},
        )
        self.assertIs(type(ep_w4_model.mlp.experts), EPLowPrecisionGroupedExperts)
        self.assertIs(type(ep_w4_model.mlp.experts.grouped_linear), W4A8GroupedLinear)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_source_activation_contract(self) -> None:
        """
        Feature: SwiGLU source validation.
        Description: Exercise supported activation selectors on packed expert sources.
        Expectation: Non-SwiGLU sources are rejected before computation.
        """
        for fake, fmt in ((False, "mxfp8"), (False, "mxfp4"), (True, "mxfp4")):
            for selector in ("hidden_act", "act_fn"):
                for valid in (False, True):
                    with self.subTest(fake=fake, fmt=fmt, selector=selector, valid=valid):
                        model = TinyMoE()
                        source = model.mlp.experts
                        if selector == "act_fn":
                            source.act_fn = nn.SiLU() if valid else nn.GELU()
                        else:
                            source.config.hidden_act = "silu" if valid else "gelu"
                            model.config.hidden_act = "silu" if valid else "gelu"
                            model.mlp.config.hidden_act = "silu" if valid else "gelu"
                        config = qat_config(fake=fake, weight_format=fmt)
                        if valid:
                            install_family(model, config)
                            projected = torch.randn(3, 512)
                            gate, up = projected.chunk(2, dim=-1)
                            torch.testing.assert_close(
                                model.mlp.experts.activation_func(projected), nn.SiLU()(gate) * up,
                            )
                        else:
                            with self.assertRaisesRegex(ValueError, "non-SwiGLU"):
                                install_family(model, config)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_scheme_conflicts_and_unknown_arguments(self) -> None:
        """
        Feature: Scheme-selected Module Family.
        Description: Invalid choices fail before any device computation.
        Expectation: Conflicts and unknown options raise ValueError or TypeError.
        """
        for args in (
            {"use_mxfp8": True},
            {"use_fused_mxfp8": True},
            {"fused": True},
            {"unknown_option": 1},
            {"weight_format": "mxfp8"},
            {"is_fake_quantize": True},
        ):
            with self.subTest(args=args):
                raw = qat_config().to_dict()
                raw["plan_overrides"][0]["replace"]["implementation"].update(args)
                config = resolve_config(raw)
                with self.assertRaises((ValueError, TypeError)):
                    install_family(TinyMoE(), config)

        raw = qat_config().to_dict()
        raw["plan_overrides"][0]["replace"]["implementation"]["low_precision_dtype_scheme"] = " "
        with self.assertRaisesRegex(ConfigResolutionError, "non-empty catalog name"):
            resolve_config(raw)
        with self.assertRaisesRegex(ValueError, "Bind low_precision_dtype_scheme"):
            qat_config().plan_overrides[0].replace.implementation.build()
        raw = qat_config().to_dict()
        raw["plan_overrides"][0]["when"] = "low_precision"
        raw["plan_overrides"][0]["low_precision_dtype_scheme"] = "experts"
        with self.assertRaisesRegex(ValueError, "specify dtype scheme only inside replace.implementation"):
            entries_to_module_replacements(
                resolve_config(raw).plan_overrides, low_precision_enabled=True,
                low_precision_config=qat_config().training.low_precision,
            )
        with self.assertRaisesRegex(ValueError, "dtype scheme requires training.low_precision"):
            entries_to_module_replacements(qat_config().plan_overrides, low_precision_config=None)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_named_schemes_keep_upstream_block_constraints(self) -> None:
        """
        Feature: Scheme-selected quantization constraints.
        Description: Use the upstream block-size and fake W4A8 tile checks.
        Expectation: Block128 and unsupported CANN converter shapes fail before computation.
        """
        for fake in (False, True):
            with self.subTest(fake=fake), self.assertRaisesRegex(ValueError, "block_size"):
                qat_config(fake=fake, block_size=128)
        model = TinyMoE()
        model.config.hidden_size = 96
        model.config.moe_intermediate_size = 96
        model.mlp.config = model.config
        model.mlp.experts.hidden_size = 96
        model.mlp.experts.intermediate_size = 96
        model.mlp.experts.gate_up_proj = nn.Parameter(torch.zeros(4, 192, 96))
        model.mlp.experts.down_proj = nn.Parameter(torch.zeros(4, 96, 96))
        with self.assertRaisesRegex(ValueError, "at least one matrix dimension divisible by 64"):
            install_family(model, qat_config(fake=True))

    @patch(_LOW_PRECISION + "validate_npu_swiglu_runtime")
    @patch(_LOW_PRECISION + "validate_npu_gmm_runtime")
    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fused_yaml_and_legacy_alias_overrides(self, gmm_probe, swiglu_probe) -> None:
        """
        Feature: Format-independent fusion selector.
        Description: YAML and CLI-style overrides accept both spellings and serialize the canonical name.
        Expectation: Fusion selects fused_mxfp8; disabling it returns to the scheme strategy.
        """
        for spelling in ("fused", "use_fused_mxfp8"):
            with self.subTest(spelling=spelling):
                raw = qat_config(weight_format="mxfp8").to_dict()
                raw["plan_overrides"][0]["replace"]["implementation"][spelling] = True
                config = resolve_config(yaml.safe_load(yaml.safe_dump(raw)))
                target = config.plan_overrides[0].replace.implementation
                self.assertTrue(target.to_dict()["fused"])
                self.assertNotIn("use_fused_mxfp8", target.to_dict())
                model = install_family(TinyMoE(), config)
                self.assertIs(type(model.mlp.experts), LowPrecisionGroupedExperts)
                self.assertEqual(model.mlp.experts._lp_mode, "fused_mxfp8")  # pylint: disable=protected-access
                for override in ("fused", "use_fused_mxfp8"):
                    config.plan_overrides[0] = replace_override_path(
                        config.plan_overrides[0], ["replace", "implementation", override],
                        False, path="plan_overrides[0]",
                    )
                    model = install_family(TinyMoE(), resolve_config(config.to_dict()))
                    self.assertIs(type(model.mlp.experts.grouped_linear), MXFP8GroupedLinear)
        self.assertEqual(gmm_probe.call_count, 2)
        self.assertEqual(swiglu_probe.call_count, 2)

    @patch("hyper_parallel.components.quantization.functional.linear_strategy_factory.validate_npu_runtime")
    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mixed_dense_and_expert_schemes(self, probe) -> None:
        """
        Feature: Independent Dense and Expert low-precision schemes.
        Description: Bind dense MXFP8 Linear and W4A8 GroupedExperts in one model plan.
        Expectation: Each boundary receives its distinct strategy without cross-talk.
        """
        del probe
        model = TinyMoE()
        model.proj = nn.Linear(128, 128)
        raw = qat_config(fake=False, weight_format="mxfp4").to_dict()
        raw["training"]["low_precision"]["dtype_schemes"]["dense"] = {
            "weight_format": "mxfp8", "act_format": "mxfp8",
        }
        raw["plan_overrides"].append({
            "match": "proj",
            "replace": {
                "kind": "module",
                "source_type": "torch.nn.Linear",
                "exact_type": True,
                "implementation": {
                    "_target_": "hyper_parallel.components.modules.Linear",
                    "low_precision_dtype_scheme": "dense",
                },
            },
        })
        install_family(model, resolve_config(raw))
        self.assertIs(type(model.proj), LowPrecisionLinear)
        self.assertIs(type(model.proj.strategy), MXFP8LinearStrategy)
        self.assertIs(type(model.mlp.experts.grouped_linear), W4A8GroupedLinear)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_deepseek_clamp_and_custom_gate_guard(self) -> None:
        """
        Feature: DeepSeek V4.1 clamped activation and custom _apply_gate rejection.
        Description: Preserve clamp values/gradients and reject arbitrary custom _apply_gate hooks.
        Expectation: Native clamp matches bit-for-bit; non-DeepSeek custom _apply_gate raises ValueError.
        """
        for fake, fmt in ((False, "mxfp8"), (False, "mxfp4"), (True, "mxfp4")):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(fake=fake, fmt=fmt, dtype=dtype):
                    model = deepseek_experts_model()
                    source = model.experts
                    source_ref = weakref.ref(source)
                    install_family(model, deepseek_qat_config(fake=fake, weight_format=fmt))
                    projected = torch.tensor(
                        [[-20., -10., -5., 0., 5., 10., 20., -20., -10., -5., 0., 5., 10., 20.]],
                        dtype=dtype, requires_grad=True,
                    )
                    reference = projected.detach().clone().requires_grad_()
                    actual = model.experts.activation_func(projected)
                    expected = source._apply_gate(reference)  # pylint: disable=protected-access
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, dtype)
                    actual.sum().backward()
                    expected.sum().backward()
                    torch.testing.assert_close(projected.grad, reference.grad, rtol=0, atol=0)
                    self.assertEqual(projected.grad[0, 6].item(), 0)
                    self.assertEqual(projected.grad[0, 13].item(), 0)
                    self.assertEqual(projected.grad[0, 7].item(), 0)
                    del source
                    gc.collect()
                    self.assertIsNone(source_ref())

        bad_model = TinyMoE()
        bad_model.mlp.experts._apply_gate = lambda x: x  # pylint: disable=protected-access
        with self.assertRaisesRegex(ValueError, "custom _apply_gate"):
            install_family(bad_model, qat_config())

    @patch(_LOW_PRECISION + "validate_npu_swiglu_runtime")
    @patch(_LOW_PRECISION + "validate_npu_gmm_runtime")
    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_deepseek_fusion_preserves_clamp_and_canonical_parameters(self, gmm_probe, fusion_probe) -> None:
        """
        Feature: V4.1 clipped MXFP8 fusion through the named Family scheme.
        Description: Resolve the production configuration and retain canonical [E, 2I, H] weights and clip limit.
        Expectation: Setup validates clipped kernel and compute_experts forwards transposed views and limit.
        """
        del gmm_probe
        model = deepseek_experts_model()
        source = model.experts
        source_ref = weakref.ref(source)
        gate_up = source.gate_up_proj
        install_family(model, deepseek_qat_config(fused=True))
        expert = model.experts
        self.assertIs(type(expert), LowPrecisionGroupedExperts)
        self.assertEqual(expert.clamp_limit, source.limit)
        fusion_probe.assert_called_once_with(clamp_limit=source.limit)
        self.assertIs(expert.gate_up_proj, gate_up)
        inputs = torch.randn(4, 128, dtype=torch.bfloat16)
        counts = torch.tensor([1, 0, 1, 2])
        with patch(_LOW_PRECISION + "fused_grouped_swiglu", return_value=inputs) as fused:
            expert.forward_expert_major(inputs, counts)
        self.assertEqual(fused.call_args.kwargs, {"clamp_limit": source.limit})
        self.assertEqual(tuple(fused.call_args.args[1].shape), (4, 128, 512))
        self.assertEqual(tuple(fused.call_args.args[2].shape), (4, 256, 128))
        self.assertEqual(set(dict(expert.named_parameters())), {"gate_up_proj", "down_proj"})
        del source
        gc.collect()
        self.assertIsNone(source_ref())
