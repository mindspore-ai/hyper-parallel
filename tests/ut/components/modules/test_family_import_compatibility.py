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
"""Canonical family imports and compatibility aliases register exactly once."""
import importlib
import inspect
import unittest

import torch

from examples.torch.module_family.config import TokenChoiceTopKRouterConfig

from hyper_parallel.components.modules.module_family import get_module_family


class TestFamilyImportCompatibility(unittest.TestCase):
    """Keep old import identities without duplicate implementation or registration."""

    def test_default_member_names_are_local(self):
        """The public ordinary member has one stable name across the migrated families."""
        entries = (
            ("common.rms_norm", ("RMSNorm", "OffsetRMSNorm")),
            ("swiglu_mlp_family", ("SwiGLUMLP",)),
            ("shared_expert_family", ("SharedExpert",)),
            ("common.rotary_embedding", ("RotaryEmbedding",)),
            ("common.spatial_merge_projector", ("SpatialMergeProjector",)),
            ("router_family", ("TokenChoiceTopKRouter",)),
            ("mhc.mhc_family", ("MHCPre", "MHCPost", "PipelinedMhc", "MegaMhc")),
            ("engram.engram_family", ("Engram",)),
            ("gate_family", ("Gate",)),
        )
        for module_name, factories in entries:
            module = importlib.import_module(f"hyper_parallel.components.modules.{module_name}")
            for name in factories:
                with self.subTest(family=name):
                    family = get_module_family(getattr(module, name))
                    self.assertEqual(family.default_variant, "local")
                    self.assertIn("local", family.variants)


    def test_family_exports(self):
        """Public exports use the canonical family classes and builders."""
        public = importlib.import_module("hyper_parallel.components.modules")
        for name, cls, factory in (
            ("common.rms_norm", "RMSNormModule", "RMSNorm"),
            ("swiglu_mlp_family", "SwiGLUMLPModule", "SwiGLUMLP"),
            ("shared_expert_family", "SharedExpertModule", "SharedExpert"),
        ):
            with self.subTest(name=name):
                canonical = importlib.import_module(f"hyper_parallel.components.modules.{name}")
                self.assertIs(getattr(public, cls), getattr(canonical, cls))
                self.assertIs(getattr(public, factory), getattr(canonical, factory))
                self.assertTrue(inspect.isfunction(getattr(canonical, factory)))
                self.assertFalse(getattr(getattr(canonical, cls), "_hp_module_replacement", False))
                self.assertIsNone(get_module_family(getattr(canonical, cls)))
                self.assertIs(get_module_family(getattr(public, factory)),
                              get_module_family(getattr(canonical, factory)))

    def test_package_exports(self):
        """Old mHC/Engram paths expose the same objects as the implementation files."""
        for package, implementation, cls, factory in (
            ("mhc", "mhc_pre", "MhcPreModule", "MHCPre"),
            ("mhc", "mhc_post", "MhcPostModule", "MHCPost"),
            ("mhc", "pipelined_mhc", "PipelinedMhcModule", "PipelinedMhc"),
            ("engram", "engram", "EngramModule", "Engram"),
        ):
            with self.subTest(cls=cls):
                public = importlib.import_module(f"hyper_parallel.components.modules.{package}")
                concrete = importlib.import_module(f"hyper_parallel.components.modules.{package}.{implementation}")
                family = importlib.import_module(f"hyper_parallel.components.modules.{package}.{package}_family")
                self.assertIs(getattr(public, cls), getattr(concrete, cls))
                self.assertIs(getattr(public, factory), getattr(family, factory))
                self.assertTrue(inspect.isfunction(getattr(family, factory)))
                self.assertFalse(getattr(getattr(concrete, cls), "_hp_module_replacement", False))
                self.assertIsNone(get_module_family(getattr(concrete, cls)))
                self.assertTrue(getattr(getattr(family, factory), "_hp_module_replacement", False))
                self.assertIsNotNone(get_module_family(getattr(family, factory)))

    def test_public_router_is_the_canonical_family(self):
        """Root imports support config construction and source replacement through the same family."""
        public = importlib.import_module("hyper_parallel.components.modules")
        canonical = importlib.import_module("hyper_parallel.components.modules.router_family")
        self.assertIs(public.TokenChoiceTopKRouter, canonical.TokenChoiceTopKRouter)
        self.assertEqual(get_module_family(public.TokenChoiceTopKRouter).name, "token_choice_topk_router")
        direct = public.TokenChoiceTopKRouter(config=TokenChoiceTopKRouterConfig(4, 6, 2))
        self.assertIs(type(direct), canonical.TokenChoiceTopKRouterModule)
        source = canonical.TokenChoiceTopKRouterModule(config=TokenChoiceTopKRouterConfig(4, 6, 2)).eval()
        adopted = public.TokenChoiceTopKRouter(module=source)
        self.assertIs(adopted.gate, source.gate)
        self.assertFalse(adopted.training)
        values = torch.randn(3, 4)
        for actual, expected in zip(adopted(values), source(values)):
            torch.testing.assert_close(actual, expected)
