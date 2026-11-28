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
"""Unit tests for nested model source composition."""

import ast
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hyper_parallel.codegen.emit.modeling import rewrite_relative_imports
from hyper_parallel.codegen.source.composition import compose_nested_model_sources
from hyper_parallel.codegen.plan.observation import observe_emission_plan


_PRIMARY_SOURCE = """\
from transformers.models.auto import AutoModel

def rotate_half(value):
    return value

class KimiModel:
    def __init__(self, config):
        self.language_model = AutoModel.from_config(config.text_config)
"""

_NESTED_SOURCE = """\
from __future__ import annotations
from .configuration_deepseek_v3 import DeepseekV3Config

def rotate_half(value):
    return -value

class DeepseekV3MoE:
    def forward(self, hidden_states):
        return rotate_half(hidden_states)

class DeepseekV3Model:
    def __init__(self, config: DeepseekV3Config):
        self.config = config
"""


class _ObservedBoundary:
    """Represent a component-owned boundary after native replacement."""

    __module__ = "hyper_parallel.components.fixture"


class _ObservedTower:
    """Keep the enclosing model's source owner after replacing a child."""

    def __init__(self):
        self.block = _ObservedBoundary()


class _AutoFactory:
    """Stand in for the native registry without modifying Transformers globals."""

    @staticmethod
    def from_config(_config: object) -> _ObservedTower:
        """Construct the native tower selected by the fixture registry."""
        return _ObservedTower()


AutoModel = _AutoFactory


class _ObservedHost:
    """Use both concrete and registry construction in the same source file."""

    def __init__(self, config):
        self.direct = _ObservedTower()
        self.dynamic = AutoModel.from_config(config)

    def get_submodule(self, path: str) -> object:
        """Resolve a recorded module path without Torch or process groups."""
        module = self
        for name in path.split("."):
            module = getattr(module, name)
        return module


class TestSourceComposition(unittest.TestCase):
    """Validate one-file composition for a nested AutoModel text tower."""

    @patch(
        "hyper_parallel.codegen.source.composition._read_module_source",
        return_value=_NESTED_SOURCE,
    )
    def test_nested_model_is_merged_and_constructed_directly(self, _mock_read):
        """The generated primary source owns and constructs the nested model."""
        meta = SimpleNamespace(nested_components=[{
            "fqn": "model.language_model",
            "type": "transformers.models.deepseek_v3.modeling_deepseek_v3:DeepseekV3Model",
            "owner_module": "transformers.models.deepseek_v3.modeling_deepseek_v3",
        }])

        rendered = compose_nested_model_sources(_PRIMARY_SOURCE, meta)

        ast.parse(rendered)
        self.assertIn("self.language_model = DeepseekV3Model(config.text_config)", rendered)
        self.assertIn("class DeepseekV3MoE", rendered)
        self.assertIn("class DeepseekV3Model", rendered)
        self.assertIn("def _hp_nested_rotate_half", rendered)
        self.assertNotIn("self.language_model = AutoModel.from_config", rendered)

    @patch(
        "hyper_parallel.codegen.source.composition._read_module_source",
        return_value=_NESTED_SOURCE,
    )
    def test_non_auto_model_component_is_not_composed(self, mock_read):
        """A different source module alone does not imply nested AutoModel."""
        meta = SimpleNamespace(nested_components=[{
            "fqn": "model.text_tower",
            "type": "transformers.models.deepseek_v3.modeling_deepseek_v3:DeepseekV3Model",
            "owner_module": "transformers.models.deepseek_v3.modeling_deepseek_v3",
        }])

        rendered = compose_nested_model_sources(_PRIMARY_SOURCE, meta)

        self.assertEqual(rendered, _PRIMARY_SOURCE)
        mock_read.assert_not_called()

    @patch(
        "hyper_parallel.codegen.source.composition._read_module_source",
        return_value=_NESTED_SOURCE,
    )
    def test_dynamic_revision_package_uses_parseable_imports(self, _mock_read):
        """HF revision hashes that begin with digits remain valid generated syntax."""
        meta = SimpleNamespace(nested_components=[{
            "fqn": "model.language_model",
            "type": (
                "transformers_modules.kimi.7dc8cececbd11535.modeling_linear:"
                "DeepseekV3Model"
            ),
            "owner_module": "transformers_modules.kimi.7dc8cececbd11535.modeling_linear",
        }])

        rendered = compose_nested_model_sources(_PRIMARY_SOURCE, meta)

        ast.parse(rendered)
        self.assertIn("__import__('transformers_modules.kimi.7dc8cececbd11535", rendered)
        self.assertNotIn("\nfrom transformers_modules.kimi.7dc8cececbd11535", rendered)

    def test_primary_dynamic_revision_import_is_rewritten_without_invalid_syntax(self):
        """Primary trusted source also handles numeric HF revision components."""
        source = (
            "from .configuration_kimi import KimiConfig\n"
            "from .layers import First as Alias, Second\n"
        )
        module_name = "transformers_modules.kimi.7dc8cececbd11535.modeling_kimi"

        rendered = rewrite_relative_imports(source, module_name)

        ast.parse(rendered)
        self.assertIn("__import__('transformers_modules.kimi.7dc8cececbd11535", rendered)
        self.assertIn("Alias =", rendered)
        self.assertNotIn("\nfrom transformers_modules.kimi.7dc8cececbd11535", rendered)

    def test_freeze_records_nested_owner_and_full_boundary_type(self):
        """The snapshot preserves enough identity to find external source."""
        root_type = type("KimiModel", (), {"__module__": "transformers.models.kimi.modeling_kimi"})
        text_type = type(
            "DeepseekV3Model",
            (),
            {"__module__": "transformers.models.deepseek_v3.modeling_deepseek_v3"},
        )
        layer_type = type(
            "DeepseekV3DecoderLayer",
            (),
            {"__module__": "transformers.models.deepseek_v3.modeling_deepseek_v3"},
        )
        moe_type = type(
            "DeepseekV3MoE",
            (),
            {"__module__": "transformers.models.deepseek_v3.modeling_deepseek_v3"},
        )
        modules = {
            "model": object(),
            "model.language_model": text_type(),
            "model.language_model.layers": object(),
            "model.language_model.layers.1": layer_type(),
            "model.language_model.layers.1.mlp": moe_type(),
        }
        model = root_type()
        model.get_submodule = modules.__getitem__
        plan = SimpleNamespace(
            modules={"model.language_model.layers.1.mlp": SimpleNamespace()},
            mesh_dim_names=("tp", "ep"),
        )

        frozen = observe_emission_plan(plan, model).to_dict()

        self.assertEqual(
            frozen["boundary_types"]["model.language_model.layers.1.mlp"],
            "transformers.models.deepseek_v3.modeling_deepseek_v3:DeepseekV3MoE",
        )
        self.assertEqual(frozen["nested_components"], [{
            "fqn": "model.language_model",
            "type": "transformers.models.deepseek_v3.modeling_deepseek_v3:DeepseekV3Model",
            "owner_module": "transformers.models.deepseek_v3.modeling_deepseek_v3",
        }])

    def test_same_source_factory_is_observed_below_a_replaced_boundary(self):
        """Component ownership must not hide the native registry-built parent."""
        model = _ObservedHost(object())
        plan = SimpleNamespace(modules={
            "direct.block": SimpleNamespace(), "dynamic.block": SimpleNamespace(),
        })

        frozen = observe_emission_plan(plan, model).to_dict()

        self.assertEqual(frozen["nested_components"], [{
            "fqn": "dynamic", "type": f"{__name__}:_ObservedTower", "owner_module": __name__,
        }])

    @patch("hyper_parallel.codegen.source.composition._read_module_source")
    def test_same_source_towers_use_emitted_classes_without_duplicate_source(self, mock_read):
        """Instantiated child methods come from the artifact, not the global registry."""
        source = (
            "class TextTower:\n"
            "    def __init__(self, config):\n        self.config = config\n"
            "class VisionTower(TextTower):\n    pass\n"
            "class Host:\n"
            "    def __init__(self, config):\n"
            "        self.language = AutoModel.from_config(config)\n"
            "        self.visual = AutoModel.from_config(config)\n"
        )
        meta = SimpleNamespace(source={"module_name": "fixture.modeling"}, nested_components=[
            {"fqn": name, "type": f"fixture.modeling:{kind}"}
            for name, kind in (("language", "TextTower"), ("visual", "VisionTower"))
        ])
        registry = SimpleNamespace(from_config=lambda _config: self.fail("native registry was executed"))

        rendered = compose_nested_model_sources(source, meta)
        namespace = {"AutoModel": registry}
        exec(compile(rendered, "<emitted-towers>", "exec"), namespace)  # pylint: disable=exec-used
        model = namespace["Host"](object())

        self.assertIs(type(model.language), namespace["TextTower"])
        self.assertIs(type(model.visual), namespace["VisionTower"])
        self.assertEqual(rendered.count("class TextTower:"), 1)
        self.assertEqual(model.language.__init__.__func__.__code__.co_filename, "<emitted-towers>")
        mock_read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
