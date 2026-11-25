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
"""Shipped recipes resolve public targets and keep model-family selection in specifications."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from torch import nn
from transformers import Qwen3_5Config, Qwen3_5MoeConfig

import hyper_parallel
from hyper_parallel.components.modules import CausalLMHead
from hyper_parallel.components.modules.module_family import get_family_build_info
from hyper_parallel.models._transformers import auto_model
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.module_build import (
    build_module_context,
    prepare_model_components,
    resolve_effective_model_spec,
)
from hyper_parallel.models.replacement import ModuleReplacementRequest
from hyper_parallel.trainer.config.resolver import resolve_config

MODELS = Path(hyper_parallel.__file__).resolve().parent / "models"


class _DiTProvider:
    """Exercise the real generic facade without an optional Diffusers installation."""

    @staticmethod
    def build_model(config: dict) -> nn.Module:
        """Return a source tree whose config has no HF model identity."""
        model = nn.Module()
        model.config = config
        model.lm_head = nn.Linear(8, 16, bias=False)
        return model

    @staticmethod
    def resolve_pretrained(*_args: object, **_kwargs: object) -> tuple[dict, None]:
        """Select the no-checkpoint construction branch without model downloads."""
        return {}, None


def _dit_replacements() -> tuple[ModuleReplacementRequest, ...]:
    """Declare a real public family for a source module in the generic-provider fixture."""
    return (ModuleReplacementRequest(match=("lm_head",), target=CausalLMHead),)


def _prepare_dit(model: nn.Module, **_kwargs: object) -> nn.Module:
    """Execute real family preparation where the facade would apply distributed infrastructure."""
    context = getattr(model, "_hp_module_build_context", build_module_context())
    return prepare_model_components(model, context)[0]


class TestModelRecipes(unittest.TestCase):
    """Validate real recipe entry points without downloading models or allocating their weights."""

    def test_training_recipes_resolve_without_legacy_adapter_or_overrides(self) -> None:
        """Every shipped training YAML resolves its targets with replacements supplied by its spec."""
        for path in sorted(MODELS.rglob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            with self.subTest(recipe=str(path.relative_to(MODELS))):
                self.assertNotIn("plan_overrides", raw)
                text = path.read_text(encoding="utf-8")
                self.assertNotIn(".adapter.", text)
                self.assertNotIn("hyper_parallel.components.modules.", text)
                if "_target_" not in raw.get("model", {}):
                    continue
                config = resolve_config(raw)
                self.assertEqual(config.model_init_dtype, "float32")
                self.assertEqual(config.plan_overrides, [])
                self.assertTrue(config.optimizer.fp32_main_params)

    def test_qwen_recipe_specs_preserve_attention_execution_choices(self) -> None:
        """Pure Ulysses and state-CP recipes select the corresponding public families in text and VLM builds."""
        for family, config_class in (("qwen3_8", Qwen3_5Config), ("qwen3_6_moe", Qwen3_5MoeConfig)):
            for path in sorted((MODELS / family / "recipes").glob("*.yaml")):
                with self.subTest(recipe=str(path.relative_to(MODELS))):
                    config = resolve_config(yaml.safe_load(path.read_text(encoding="utf-8")))
                    spec = config.model.model_spec.build()
                    full_config = config_class()
                    self.assertIs(resolve_effective_model_spec(full_config, spec), spec)
                    model_config = full_config if "vlm" in path.stem else full_config.text_config
                    rules = spec.replacements(model=SimpleNamespace(config=model_config), context={})
                    selected = {request.target.__name__: request for request in rules}
                    degree = 1 if "p2p" in path.stem else None
                    self.assertEqual(selected["GatedDeltaNet"].kwargs["backend"], "triton")
                    self.assertEqual(selected["GatedDeltaNet"].kwargs["conv_backend"],
                                     "eager" if degree == 1 else "triton")
                    self.assertEqual(selected["GatedDeltaNet"].kwargs["ulysses_degree"], degree)
                    self.assertEqual(selected["GatedGroupedQueryAttention"].kwargs["ulysses_degree"], degree)
                    if "vlm" in path.stem:
                        self.assertIn("AxialVisionTransformer", selected)
                    if family == "qwen3_6_moe":
                        self.assertTrue(selected["GroupedExperts"].kwargs["use_grouped_gemm"])

    def test_dit_facades_preserve_selected_spec_and_provider_only_behavior(self) -> None:
        """Both DiT entry points apply spec families; a provider without a model type keeps its source tree."""
        spec = ModelAdapterSpec("FixtureDiT", "fixture_dit", replacements=_dit_replacements)
        setup = SimpleNamespace(mesh_context=SimpleNamespace())
        for entry in ("from_config", "from_pretrained"):
            for identity in (None, "fixture_dit"):
                with self.subTest(entry=entry, identity=identity), \
                        patch.object(auto_model, "_current_device", return_value=torch.device("cpu")), \
                        patch.object(auto_model, "instantiate_infrastructure", return_value=(None, None)), \
                        patch.object(auto_model, "get_model_adapter", return_value=spec), \
                        patch.object(auto_model, "apply_model_infrastructure", side_effect=_prepare_dit):
                    argument = {} if entry == "from_config" else "unused-local-fixture"
                    model = getattr(auto_model.HyperAutoModelForDiT, entry)(
                        argument, model_provider=_DiTProvider, model_type=identity, distributed_setup=setup,
                    )
                    self.assertEqual(get_family_build_info(model.lm_head) is not None, identity is not None)
