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
"""Config-resolver characterization: custom models and the HF fallback.

Renamed from ``test_registry.py`` in S5f (05 stage-5 item 4): the former
``_transformers/registry.py`` facade was split into
``_transformers/config_resolver.py`` (HF config helpers, tested here) and
``models/registry.py`` (family registry, M1). An unknown architecture resolves
to None, so ``get_is_hf_model`` selects the HF native implementation; a broken lazy entry also falls back to HF
instead of raising. No Hub/network access is needed: ``MODEL_ARCH_MAPPING``
entries are registered or injected locally and ``AutoConfig.from_pretrained``
is never called.
"""
# pylint: disable=wrong-import-position

import os
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.models._transformers import config_resolver
from hyper_parallel.models import registry
from tests.common.mark_utils import arg_mark


class TestEmptyMappingFallback(unittest.TestCase):
    """Custom-architecture resolution and HF fallback semantics."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_deepseek_v41_registers_lazily(self):
        """Family discovery registers the V4.1 custom architecture."""
        self.assertIsInstance(registry.MODEL_ARCH_MAPPING, OrderedDict)
        spec = registry.get_model_adapter("deepseek_v41")
        self.assertIsNotNone(spec)
        self.assertEqual(
            registry.MODEL_ARCH_MAPPING["DeepseekV41ForCausalLM"],
            (
                "hyper_parallel.models.deepseek_v41.modeling_deepseek_v41",
                "DeepseekV41CroppedForCausalLM",
            ),
        )
        model_cls = registry._resolve_custom_model_cls("DeepseekV41ForCausalLM")
        self.assertEqual(model_cls.__name__, "DeepseekV41CroppedForCausalLM")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unknown_arch_resolves_to_none(self):
        """An unregistered architecture (e.g. Qwen3MoeForCausalLM) falls back."""
        self.assertIsNone(registry._resolve_custom_model_cls("Qwen3MoeForCausalLM"))
        self.assertIsNone(registry._resolve_custom_model_cls(""))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_get_is_hf_model(self):
        """Unknown and missing architectures select the HF implementation."""
        config = SimpleNamespace(architectures=["Qwen3MoeForCausalLM"])
        self.assertTrue(config_resolver.get_is_hf_model(config))
        self.assertTrue(config_resolver.get_is_hf_model(SimpleNamespace(architectures=[])))
        self.assertTrue(config_resolver.get_is_hf_model(SimpleNamespace(architectures=None)))
        self.assertTrue(config_resolver.get_is_hf_model(SimpleNamespace()))
        # force_hf short-circuits regardless of architectures
        self.assertTrue(config_resolver.get_is_hf_model(config, force_hf=True))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_custom_model_resolution_triggers_family_discovery(self):
        """AutoModel path selection should not require registry preheating."""
        config = SimpleNamespace(
            model_type="lazy_family",
            architectures=["LazyFamilyForCausalLM"],
        )
        with (
                patch.object(config_resolver, "get_model_adapter") as discover,
                patch.object(
                    config_resolver,
                    "_resolve_custom_model_cls",
                    return_value=dict,
                ) as resolve,
        ):
            is_hf_model = config_resolver.get_is_hf_model(config)

        self.assertFalse(is_hf_model)
        discover.assert_called_once_with("lazy_family")
        resolve.assert_called_once_with("LazyFamilyForCausalLM")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_lazy_load_success_and_failure(self):
        """A resolvable entry returns the class; a broken entry falls back to None."""
        registry.MODEL_ARCH_MAPPING["_M0GoodArch"] = ("builtins", "dict")
        registry.MODEL_ARCH_MAPPING["_M0BadArch"] = (
            "no_such_module_m0_xyz",
            "NoSuchClass",
        )
        registry.MODEL_ARCH_MAPPING["_M0BadAttrArch"] = ("builtins", "no_such_attr")
        try:
            self.assertIs(registry._resolve_custom_model_cls("_M0GoodArch"), dict)
            self.assertIsNone(registry._resolve_custom_model_cls("_M0BadArch"))
            self.assertIsNone(registry._resolve_custom_model_cls("_M0BadAttrArch"))
            config = SimpleNamespace(architectures=["_M0GoodArch"])
            self.assertFalse(config_resolver.get_is_hf_model(config))
        finally:
            for name in ("_M0GoodArch", "_M0BadArch", "_M0BadAttrArch"):
                registry.MODEL_ARCH_MAPPING.pop(name, None)
            registry._resolve_custom_model_cls.cache_clear()


if __name__ == "__main__":
    unittest.main()
