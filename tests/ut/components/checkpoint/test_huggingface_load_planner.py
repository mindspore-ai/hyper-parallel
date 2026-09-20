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
"""UT for :mod:`hyper_parallel.components.checkpoint.huggingface_load_planner`, against the legacy loader."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch import nn

from hyper_parallel.components.checkpoint.conversion_ops import AddScalar, InterleaveQKV
from hyper_parallel.components.checkpoint.weight_conversion import (
    ConversionOps,
    WeightConverter,
    WeightRenaming,
)
from hyper_parallel.components.checkpoint.huggingface_checkpointer import (
    HuggingFaceCheckpointer,
    resolve_hf_loader,
)
from hyper_parallel.components.checkpoint.huggingface_load_planner import (
    HFLoadPlanner,
    load_hf_checkpoint,
)

_QKV_SOURCES = ["q_proj.weight", "k_proj.weight", "v_proj.weight"]


def _interleave() -> InterleaveQKV:
    """Two KV heads, two queries per KV head, head dimension two: 8 + 4 + 4 rows grouped into 16."""
    return InterleaveQKV(2, 2, 2, 2, source_is_fused=False)


class _Product(ConversionOps):
    """Multiplies two checkpoint tensors elementwise, which no region can describe."""

    def convert(self, input_dict: dict[str, Any], source_patterns: list[str], target_patterns: list[str],
                **kwargs: Any) -> dict[str, torch.Tensor]:
        """Multiply the tensors of the two source patterns."""
        del kwargs
        first, second = (input_dict[pattern][0] for pattern in source_patterns)
        return {target_patterns[0]: first * second}


class _Layer(nn.Module):
    """A layer holding a fused, grouped QKV projection and a norm weight."""

    def __init__(self, attention_scope: bool) -> None:
        super().__init__()
        holder = nn.Module()
        holder.linear_qkv = nn.Linear(4, 16, bias=False)
        if attention_scope:
            self.self_attn = holder
        else:
            self.linear_qkv = holder.linear_qkv
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.zeros(4))


class _TinyModel(nn.Module):
    """Embedding tied to the head, one layer, a non-persistent buffer, and optionally a tensor nobody loads."""

    def __init__(self, attention_scope: bool = False, extra: bool = False) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([_Layer(attention_scope)])
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.embed.weight
        self.register_buffer("scale", torch.ones(1), persistent=False)
        if extra:
            self.extra = nn.Parameter(torch.zeros(3))
        for parameter in self.parameters():
            nn.init.zeros_(parameter)


def _checkpoint_tensors(attention_prefix: str = "") -> dict[str, torch.Tensor]:
    """The tensors of a checkpoint for :class:`_TinyModel`, laid out the way Hugging Face stores them."""
    generator = torch.Generator().manual_seed(0)
    return {
        "embed.weight": torch.randn(8, 4, generator=generator).to(torch.bfloat16),
        f"layers.0.{attention_prefix}q_proj.weight": torch.randn(8, 4, generator=generator),
        f"layers.0.{attention_prefix}k_proj.weight": torch.randn(4, 4, generator=generator),
        f"layers.0.{attention_prefix}v_proj.weight": torch.randn(4, 4, generator=generator),
        "layers.0.norm.weight": torch.randn(4, generator=generator),
        "layers.0.norm_scale.weight": torch.randn(4, generator=generator),
        "unused.bias": torch.randn(2, generator=generator),
    }


def _mapping() -> list[Any]:
    """Rules fusing Q, K and V into the grouped projection and shifting the norm by one."""
    return [
        WeightConverter(source_patterns=list(_QKV_SOURCES), target_patterns="linear_qkv.weight",
                        operations=[_interleave()]),
        WeightConverter(source_patterns="norm.weight", target_patterns="norm.weight", operations=[AddScalar(1.0)]),
    ]


def _replacement_mapping() -> list[Any]:
    """The same conversions, scoped to the modules a replacement installs, as replacements scope them."""
    qkv = WeightConverter(source_patterns=list(_QKV_SOURCES), target_patterns="linear_qkv.weight",
                          operations=[_interleave()])
    qkv.scope_prefix = "layers.0.self_attn"
    norm = WeightConverter(source_patterns="weight", target_patterns="weight", operations=[AddScalar(1.0)])
    norm.scope_prefix = "layers.0.norm"
    return [qkv, norm]


class TestHFLoadPlanner(unittest.TestCase):
    """The DCP loader has to load what the legacy loader loads."""

    def _checkpoint_dir(self, tensors: dict[str, torch.Tensor], file_name: str = "model.safetensors") -> Path:
        """A checkpoint directory holding ``tensors`` in one file, removed once the test is done."""
        path = Path(tempfile.mkdtemp(prefix="test_huggingface_load_planner_"))
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        save_file(tensors, str(path / file_name))
        return path

    def _assert_same_models(self, legacy: nn.Module, dcp: nn.Module) -> None:
        """Assert that two models hold the same tensors under the same names."""
        legacy_state, dcp_state = legacy.state_dict(), dcp.state_dict()
        self.assertEqual(legacy_state.keys(), dcp_state.keys(), "state dict keys differ")
        for name, value in legacy_state.items():
            self.assertTrue(torch.equal(value, dcp_state[name]),
                            f"{name} mismatch: legacy={value}, dcp={dcp_state[name]}")

    def _load_both(self, path: str, make_model: Any, make_mapping: Optional[Any]) -> tuple[Any, Any, Any, Any]:
        """Load ``path`` into two fresh models, one per loader, each with rules of its own."""
        models, reports = [], []
        for loader in ("legacy", "dcp"):
            model = make_model()
            mapping = make_mapping(model) if make_mapping is not None else None
            state = {"model": model}
            HuggingFaceCheckpointer(loader=loader, weights_mapping=mapping).load(path, state)
            reports.append(state["load_report"])
            models.append(model)
        return models[0], models[1], reports[0], reports[1]

    def test_dcp_loader_matches_the_legacy_loader(self):
        """
        Feature: HuggingFaceCheckpointer.load with loader="dcp".
        Description: Load a checkpoint whose Q, K and V are grouped into one projection, whose norm is
            shifted by one and whose embedding is bfloat16 and tied to the head, once per loader.
        Expectation: Both models hold the same tensors, the reports agree, the tied head counts as
            loaded, the tensor nobody owns is unexpected, and the conversions used are recorded.
        """
        path = str(self._checkpoint_dir(_checkpoint_tensors()))

        legacy, dcp, legacy_report, dcp_report = self._load_both(path, _TinyModel, lambda model: _mapping())

        self._assert_same_models(legacy, dcp)
        self.assertEqual(dcp_report, legacy_report, f"reports differ: legacy={legacy_report}, dcp={dcp_report}")
        self.assertIn("lm_head.weight", dcp_report.loaded_keys, f"tied head not loaded: {dcp_report}")
        self.assertIn("unused.bias", dcp_report.unexpected_keys, f"unused tensor not reported: {dcp_report}")
        used = [len(model._weight_conversions) for model in (legacy, dcp)]  # pylint: disable=protected-access
        self.assertEqual(used, [2, 2], f"expected both rules recorded by both loaders, got {used}")

    def test_replacement_conversions_load_in_two_stages(self):
        """
        Feature: loader="dcp" on a model whose replacement modules convert the normalized weights again.
        Description: Scope the QKV and norm conversions to the modules a replacement installs, record the
            pre-replacement shapes on the model as replacements do, and load once per loader.
        Expectation: Both models hold the same tensors and both record the two replacement conversions.
        """
        tensors = _checkpoint_tensors(attention_prefix="self_attn.")
        path = str(self._checkpoint_dir(tensors))
        shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items() if not name.startswith("unused")}
        shapes["lm_head.weight"] = (8, 4)

        def make_model() -> nn.Module:
            model = _TinyModel(attention_scope=True)
            model._hp_checkpoint_source_shapes = dict(shapes)  # pylint: disable=protected-access
            return model

        def make_mapping(model: nn.Module) -> list[Any]:
            replacements = _replacement_mapping()
            model._hp_replacement_weight_conversions = replacements  # pylint: disable=protected-access
            return list(replacements)

        legacy, dcp, legacy_report, dcp_report = self._load_both(path, make_model, make_mapping)

        self._assert_same_models(legacy, dcp)
        self.assertEqual(dcp_report, legacy_report, f"reports differ: legacy={legacy_report}, dcp={dcp_report}")
        counts = [len(model._hp_used_replacement_weight_conversions)  # pylint: disable=protected-access
                  for model in (legacy, dcp)]
        self.assertEqual(counts, [2, 2], f"expected two replacement conversions recorded, got {counts}")

    def test_untraceable_conversion_runs_on_real_tensors(self):
        """
        Feature: HFLoadPlanner fallback for a conversion no region can describe.
        Description: Compute the norm as the elementwise product of two checkpoint tensors, and load
            once per loader, keeping the planner of the DCP load.
        Expectation: The norm is planned as a whole-tensor read rather than a region read, and both
            models hold the same tensors.
        """
        path = str(self._checkpoint_dir(_checkpoint_tensors()))

        def rules() -> list[Any]:
            product = WeightConverter(source_patterns=["norm.weight", "norm_scale.weight"],
                                      target_patterns="norm.weight", operations=[_Product()])
            return [_mapping()[0], product]

        legacy = _TinyModel()
        HuggingFaceCheckpointer(loader="legacy", weights_mapping=rules()).load(path, {"model": legacy})
        dcp = _TinyModel()
        planner = HFLoadPlanner(dcp, weights_mapping=rules())
        load_hf_checkpoint(dcp, path, planner=planner)

        self._assert_same_models(legacy, dcp)
        self.assertNotIn("layers.0.norm.weight", planner.table, "the product was planned as region reads")
        self.assertEqual(len(planner.deferred), 1, f"expected one whole-tensor read, got {planner.deferred}")

    def test_rules_compose_in_order_and_hooks_apply(self):
        """
        Feature: HFLoadPlanner rule composition and overridable hooks.
        Description: Pass extra rules repeating one of the rules passed in, and key_mapping. Then load
            through a subclass that renames an old norm key, skips a key and records converter choices.
        Expectation: Extra rules come first, then the key_mapping renames, then the rules passed in, each
            rule once. The renamed key loads the norm, the skipped key is not reported, and every
            converter choice goes through the hook.
        """
        first, second = _mapping()
        extra = WeightRenaming(source_patterns="old_embed.weight", target_patterns="embed.weight")
        planner = HFLoadPlanner(_TinyModel(), weights_mapping=[first, second], extra_weights_mapping=[extra, first],
                                key_mapping={"^legacy\\.": ""})
        resolved = planner._resolved_mapping()  # pylint: disable=protected-access
        self.assertEqual([type(rule).__name__ for rule in resolved],
                         ["WeightRenaming", "WeightConverter", "WeightRenaming", "WeightConverter"],
                         f"unexpected rule order: {resolved}")
        self.assertIs(resolved[0], extra, "extra rules do not come first")
        self.assertIs(resolved[1], first, "a repeated rule was not kept at its first position")
        self.assertIs(resolved[3], second, "the rules passed in do not come last")

        tensors = _checkpoint_tensors()
        tensors["old.norm.weight"] = tensors.pop("layers.0.norm.weight")
        path = str(self._checkpoint_dir(tensors))
        choices = []

        class _Hooked(HFLoadPlanner):
            """Renames the old norm key, skips the unused tensor, and records converter choices."""

            def map_checkpoint_key(self, key: str) -> Optional[str]:
                """Skip ``unused.bias`` and give the old norm key its current name."""
                if key == "unused.bias":
                    return None
                return "layers.0.norm.weight" if key == "old.norm.weight" else key

            def select_converter(self, source_pattern: str, target_name: str, candidates: Any) -> Any:
                """Record the target, then pick as the default does."""
                choices.append(target_name)
                return super().select_converter(source_pattern, target_name, candidates)

        model = _TinyModel()
        report = load_hf_checkpoint(model, path, planner=_Hooked(model, weights_mapping=_mapping()))

        expected_norm = tensors["old.norm.weight"] + 1.0
        self.assertTrue(torch.equal(model.layers[0].norm.weight.detach(), expected_norm),
                        f"norm mismatch: expected={expected_norm}, got={model.layers[0].norm.weight}")
        self.assertNotIn("unused.bias", report.unexpected_keys, f"skipped key reported: {report}")
        self.assertIn("layers.0.norm.weight", choices, f"converter choices did not go through the hook: {choices}")

    def test_strict_load_fails_before_reading(self):
        """
        Feature: HFLoadPlanner strict mode.
        Description: Load into a model with a tensor the checkpoint does not have, with strict=True.
        Expectation: RuntimeError naming the tensor, raised while planning, so nothing is read.
        """
        path = str(self._checkpoint_dir(_checkpoint_tensors()))
        model = _TinyModel(extra=True)

        with self.assertRaises(RuntimeError) as ctx:
            load_hf_checkpoint(model, path, weights_mapping=_mapping(), strict=True)

        self.assertIn("extra", str(ctx.exception), f"missing tensor not named in: {ctx.exception}")
        self.assertFalse(bool(model.embed.weight.detach().any()), "the embedding was read before planning failed")

    def test_single_file_checkpoint(self):
        """
        Feature: loader="dcp" on a checkpoint that is one safetensors file of any name.
        Description: Save the checkpoint as weights.safetensors and load the file path once per loader.
        Expectation: Both models hold the same tensors.
        """
        path = self._checkpoint_dir(_checkpoint_tensors(), file_name="weights.safetensors") / "weights.safetensors"

        legacy, dcp, _, _ = self._load_both(str(path), _TinyModel, lambda model: _mapping())

        self._assert_same_models(legacy, dcp)

    def test_loader_is_chosen_by_argument_then_environment(self):
        """
        Feature: resolve_hf_loader.
        Description: Resolve with nothing set, with HYPER_PARALLEL_HF_LOADER=dcp, with an argument
            overriding the environment, and with a name that is no loader.
        Expectation: legacy, dcp, the argument, and ValueError.
        """
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HYPER_PARALLEL_HF_LOADER", None)
            self.assertEqual(resolve_hf_loader(), "legacy", "default loader is not legacy")
        with patch.dict(os.environ, {"HYPER_PARALLEL_HF_LOADER": "DCP"}):
            self.assertEqual(resolve_hf_loader(), "dcp", "environment not read")
            self.assertEqual(resolve_hf_loader("legacy"), "legacy", "argument does not override environment")
        with self.assertRaises(ValueError):
            resolve_hf_loader("fastest")


if __name__ == "__main__":
    unittest.main()
