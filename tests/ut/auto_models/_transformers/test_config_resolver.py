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
"""Config-resolver characterization: the empty ``MODEL_ARCH_MAPPING`` HF fallback.

Renamed from ``test_registry.py`` in S5f (05 stage-5 item 4): the former
``_transformers/registry.py`` facade was split into
``_transformers/config_resolver.py`` (HF config helpers, tested here) and
``models/registry.py`` (family registry, M1). With an empty mapping every
architecture resolves to None, so ``get_is_hf_model`` always selects the
HF native implementation; a broken lazy entry also falls back to HF
instead of raising. No Hub/network access is needed: ``MODEL_ARCH_MAPPING``
entries are injected locally and ``AutoConfig.from_pretrained`` is never
called.
"""
# pylint: disable=wrong-import-position

import os
import unittest
from collections import OrderedDict
from types import SimpleNamespace

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.models._transformers import config_resolver
from hyper_parallel.models import registry
from tests.common.mark_utils import arg_mark


class TestEmptyMappingFallback(unittest.TestCase):
    """HF fallback semantics of the (currently empty) arch registry."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mapping_is_empty_ordered_dict(self):
        """Current master ships no custom architectures.

        The registry is filled lazily by ``models/registry.py``; update this
        snapshot in the same commit that registers the first model.
        """
        self.assertIsInstance(registry.MODEL_ARCH_MAPPING, OrderedDict)
        self.assertEqual(len(registry.MODEL_ARCH_MAPPING), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unknown_arch_resolves_to_none(self):
        """An unregistered architecture (e.g. Qwen3MoeForCausalLM) falls back."""
        self.assertIsNone(registry._resolve_custom_model_cls("Qwen3MoeForCausalLM"))
        self.assertIsNone(registry._resolve_custom_model_cls(""))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_get_is_hf_model(self):
        """Every config shape selects HF native while the mapping is empty."""
        config = SimpleNamespace(architectures=["Qwen3MoeForCausalLM"])
        self.assertTrue(config_resolver.get_is_hf_model(config))
        self.assertTrue(config_resolver.get_is_hf_model(SimpleNamespace(architectures=[])))
        self.assertTrue(config_resolver.get_is_hf_model(SimpleNamespace(architectures=None)))
        self.assertTrue(config_resolver.get_is_hf_model(SimpleNamespace()))
        # force_hf short-circuits regardless of architectures
        self.assertTrue(config_resolver.get_is_hf_model(config, force_hf=True))

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
