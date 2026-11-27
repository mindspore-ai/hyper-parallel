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

from examples.torch.module_family.config import GateConfig

from hyper_parallel.components.modules.module_family import get_module_family
from tests.common.mark_utils import arg_mark


class TestFamilyImportCompatibility(unittest.TestCase):
    """Keep old import identities without duplicate implementation or registration."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_default_member_names_are_local(self) -> None:
        """Feature: Module family member names.
        Description: Inspect migrated canonical registrations.
        Expectation: Every family retains a local default member.
        """
        entries = (
            ("common.rms_norm", ("RMSNorm", "OffsetRMSNorm")),
            ("mlp.mlp_family", ("SwiGLUMLP",)),
            ("moe.shared_expert", ("SharedExpert",)),
            ("common.rotary_embedding", ("RotaryEmbedding",)),
            ("common.spatial_merge_projector", ("SpatialMergeProjector",)),
            ("moe.router", ("Gate",)),
            ("mhc.mhc_family", ("MHCPre", "MHCPost", "PipelinedMhc", "MegaMhc")),
            ("engram.engram_family", ("Engram",)),
        )
        for module_name, factories in entries:
            module = importlib.import_module(f"hyper_parallel.components.modules.{module_name}")
            for name in factories:
                with self.subTest(family=name):
                    family = get_module_family(getattr(module, name))
                    self.assertEqual(family.default_variant, "local")
                    self.assertIn("local", family.variants)


    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_family_exports(self) -> None:
        """Feature: Canonical public exports.
        Description: Compare package exports with the canonical implementations.
        Expectation: Each factory registers once and keeps its public identity.
        """
        public = importlib.import_module("hyper_parallel.components.modules")
        for name, cls, factory in (
            ("common.rms_norm", "RMSNormModule", "RMSNorm"),
            ("mlp", "SwiGLUMLPModule", "SwiGLUMLP"),
            ("moe.shared_expert", "SharedExpertModule", "SharedExpert"),
            ("moe.router", "GateModule", "Gate"),
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_package_exports(self):
        """Feature: Family import compatibility.
        Description: Old mHC/Engram paths expose the same objects as the implementation files.
        Expectation: Package exports.
        """
        for package, implementation, cls, factory in (
            ("mhc", "mhc_pre", "MhcPreModule", "MHCPre"),
            ("mhc", "mhc_post", "MhcPostModule", "MHCPost"),
            ("mhc", "pipelined_mhc", "PipelinedMhcModule", "PipelinedMhc"),
            ("engram", "engram", "EngramModule", "Engram"),
            ("mlp", "swiglu_mlp", "SwiGLUMLPModule", "SwiGLUMLP"),
            ("mlp", "swiglu_mlp", "W1W2SwiGLUMLPModule", "SwiGLUMLP"),
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_public_router_is_the_canonical_family(self) -> None:
        """Feature: Public router factory.
        Description: Build directly and replace a source through public imports.
        Expectation: Both paths use the canonical family and preserve source state.
        """
        public = importlib.import_module("hyper_parallel.components.modules")
        canonical = importlib.import_module("hyper_parallel.components.modules.moe.router")
        self.assertIs(public.Gate, canonical.Gate)
        self.assertEqual(get_module_family(public.Gate).name, "gate")
        direct = public.Gate(config=GateConfig(4, 6, 2))
        self.assertIs(type(direct), canonical.GateModule)
        source = canonical.GateModule(config=GateConfig(4, 6, 2)).eval()
        adopted = public.Gate(module=source)
        self.assertIs(adopted.weight, source.weight)
        self.assertFalse(adopted.training)
        values = torch.randn(3, 4)
        for actual, expected in zip(adopted(values), source(values)):
            torch.testing.assert_close(actual, expected)
