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

from hyper_parallel.codegen.source.composition import compose_nested_model_sources
from hyper_parallel.codegen.plan.freeze import freeze_plan


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

        frozen = freeze_plan(plan, model).to_dict()

        self.assertEqual(
            frozen["boundary_types"]["model.language_model.layers.1.mlp"],
            "transformers.models.deepseek_v3.modeling_deepseek_v3:DeepseekV3MoE",
        )
        self.assertEqual(frozen["nested_components"], [{
            "fqn": "model.language_model",
            "type": "transformers.models.deepseek_v3.modeling_deepseek_v3:DeepseekV3Model",
            "owner_module": "transformers.models.deepseek_v3.modeling_deepseek_v3",
        }])


if __name__ == "__main__":
    unittest.main()
