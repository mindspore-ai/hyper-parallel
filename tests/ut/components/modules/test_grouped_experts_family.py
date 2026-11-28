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
"""Real Family configuration, selection, and canonical layout contracts for GroupedExperts."""

import copy
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.modules import GroupedExperts, get_family_build_info
from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import EPGroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig,
    GroupedExpertsModule,
)
from hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts import (
    EPLowPrecisionGroupedExperts,
    LowPrecisionGroupedExperts,
    MoonEPLowPrecisionGroupedExperts,
)
from hyper_parallel.trainer.config.resolver import ConfigResolutionError
from tests.common.grouped_experts_family import TinyMoE, family_config, replace_experts

_LP_MODULE = "hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts."


class TestGroupedExpertsFamily(unittest.TestCase):
    """Exercise registered production targets through configuration and replacement."""

    def setUp(self) -> None:
        """Isolate device availability without bypassing selection or constructors."""
        for name in ("validate_npu_gmm_runtime", "validate_npu_swiglu_runtime"):
            device_check = patch(_LP_MODULE + name)
            device_check.start()
            self.addCleanup(device_check.stop)

    def test_selection_and_canonical_layout(self) -> None:
        """
        Feature: Unified GroupedExperts Family.
        Description: Parse legal combinations across local, EP, and MoonEP and execute replacement.
        Expectation: Selected member reuses canonical [E, 2I, H] / [E, H, I] parameters without copying.
        """
        source = TinyMoE()
        cases = (
            ({}, None, GroupedExpertsModule, None),
            ({"use_mxfp8": True}, None, LowPrecisionGroupedExperts, "mxfp8"),
            ({"use_mxfp8": True, "use_fused_mxfp8": True}, None, LowPrecisionGroupedExperts, "fused_mxfp8"),
            ({"use_mxfp8": True, "fused": True}, None, LowPrecisionGroupedExperts, "fused_mxfp8"),
            ({}, {"ep_size": 2}, EPGroupedExperts, None),
            ({"use_mxfp8": True}, {"ep_size": 2}, EPLowPrecisionGroupedExperts, "mxfp8"),
            ({"use_mxfp8": True, "fused": True}, {"ep_size": 2}, EPLowPrecisionGroupedExperts, "fused_mxfp8"),
            (
                {"use_mxfp8": True, "fused": True, "moonep": True},
                {"ep_size": 2},
                MoonEPLowPrecisionGroupedExperts,
                "fused_mxfp8",
            ),
        )
        for selectors, context, expected_cls, expected_mode in cases:
            with self.subTest(selectors=selectors, context=context):
                model = copy.deepcopy(source)
                orig_gate_up = model.mlp.experts.gate_up_proj
                orig_down = model.mlp.experts.down_proj
                replace_experts(model, family_config(**selectors), context=context)
                expert = model.mlp.experts
                self.assertIs(type(expert), expected_cls)
                self.assertIsInstance(expert, GroupedExpertsModule)
                self.assertEqual(tuple(expert.gate_up_proj.shape), (4, 512, 128))
                self.assertEqual(tuple(expert.down_proj.shape), (4, 128, 256))
                self.assertIs(expert.gate_up_proj, orig_gate_up)
                self.assertIs(expert.down_proj, orig_down)
                self.assertEqual(set(dict(expert.named_parameters())), {"gate_up_proj", "down_proj"})
                if expected_mode is not None:
                    self.assertEqual(expert._lp_mode, expected_mode)  # pylint: disable=protected-access

    def test_invalid_selectors_and_megakernel_exclusion(self) -> None:
        """
        Feature: GroupedExperts Family selector validation.
        Description: Reject fusion without MXFP8, nonboolean selectors, typos, and MegaKernel+low-precision.
        Expectation: Invalid types/conflicts fail at config resolution; MegaKernel+LP fails at selection.
        """
        for selectors in (
            {"fused": 1},
            {"use_mxfp8": True, "fused": False, "use_fused_mxfp8": True},
            {"use_mxfp8": 1},
            {"unknown": True},
            {"use_mxfp8": True, "unknown": True},
        ):
            with self.subTest(selectors=selectors), self.assertRaises(ConfigResolutionError):
                family_config(**selectors)

        for selectors in (
            {"use_fused_mxfp8": True},
            {"fused": True},
            {"megakernel": True, "use_mxfp8": True},
            {"megakernel": True, "weight_format": "hif8", "act_format": "hif8"},
        ):
            with self.subTest(selectors=selectors):
                with self.assertRaisesRegex(ValueError, "unsupported selector combination"):
                    replace_experts(TinyMoE(), family_config(**selectors), context={"ep_size": 2})

    def test_direct_config_and_meta_construction(self) -> None:
        """
        Feature: Direct GroupedExpertsConfig construction for low-precision variants.
        Description: Build local and EP low-precision members directly on meta and materialize them.
        Expectation: Provenance is direct, transforms remain registered, and fused mode enforces output routing.
        """
        cfg = GroupedExpertsConfig(4, 128, 256, routing_weight_mode="output", num_layers=2)
        with torch.device("meta"):
            ep_lp = GroupedExperts(
                config=cfg, use_mxfp8=True, fused=True, context={"ep_size": 2},
            )
        self.assertIs(type(ep_lp), EPLowPrecisionGroupedExperts)
        self.assertEqual(get_family_build_info(ep_lp).origin, "direct")
        self.assertTrue(ep_lp.gate_up_proj.is_meta)
        ep_lp.to_empty(device="cpu")
        ep_lp.reset_parameters()
        self.assertEqual(len(ep_lp.make_transforms()), 2)
        copied = copy.deepcopy(ep_lp)
        self.assertIs(get_family_build_info(copied), get_family_build_info(ep_lp))
        self.assertIsNot(copied.gate_up_proj, ep_lp.gate_up_proj)

        before_fc2_cfg = GroupedExpertsConfig(4, 128, 256, routing_weight_mode="before_fc2", num_layers=1)
        with self.assertRaisesRegex(ValueError, "routing_weight_mode='output'"):
            GroupedExperts(config=before_fc2_cfg, use_mxfp8=True, fused=True)
