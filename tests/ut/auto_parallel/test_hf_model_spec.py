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
"""Unit tests for the shared AutoModels model-spec resolver.

How to run this:
    pytest tests/ut/auto_parallel/test_hf_model_spec.py -v
"""
import json
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.auto_parallel import _hf_model_spec as spec_mod
from hyper_parallel.auto_parallel._hf_model_spec import (
    is_auto_models_schema,
    resolve_hf_model_spec,
)

_REGISTRY = "hyper_parallel.models._transformers.config_resolver"


class _Recorder:
    """Stand-in for registry.get_hf_config that records its arguments."""

    def __init__(self, result: Any) -> None:
        """Store the config to hand back."""
        self.result = result
        self.calls = []

    def __call__(self, path: str, attn_implementation: str,
                 torch_dtype: Any, **kwargs: Any) -> Any:
        """Record one call and return the canned config."""
        self.calls.append((path, attn_implementation, torch_dtype, kwargs))
        return self.result


def _stub_registry(recorder):
    """Patch the registry module so the local import needs no transformers.

    ``_get_hf_config`` imports the registry inside the function body, so
    seeding ``sys.modules`` covers that import without requiring the
    optional transformers dependency to be installed.
    """
    module = types.ModuleType(_REGISTRY)
    module.get_hf_config = recorder
    return patch.dict(sys.modules, {_REGISTRY: module})


class TestSchemaProbe(unittest.TestCase):
    """is_auto_models_schema must accept mappings, objects and neither."""

    def test_mapping_with_a_root_section(self) -> None:
        """A parsed YAML dict is recognised by its root sections."""
        self.assertTrue(is_auto_models_schema({"training": {}}))
        self.assertTrue(is_auto_models_schema({"fsdp_config": {}}))

    def test_legacy_mapping_is_rejected(self) -> None:
        """The legacy schema nests everything under ``train``."""
        self.assertFalse(is_auto_models_schema({"train": {}}))

    def test_object_with_attributes(self) -> None:
        """A Config-like object is probed through its ``__dict__``."""
        self.assertTrue(is_auto_models_schema(SimpleNamespace(accelerator={})))
        self.assertFalse(is_auto_models_schema(SimpleNamespace(train={})))

    def test_value_without_a_dict_is_not_a_schema(self) -> None:
        """Scalars carry no sections and must not raise."""
        self.assertFalse(is_auto_models_schema(42))
        self.assertFalse(is_auto_models_schema("training"))


class TestSharedExpertDerivation(unittest.TestCase):
    """One wide shared expert is encoded as several narrow ones."""

    def test_width_is_converted_to_a_count(self) -> None:
        """Qwen2-MoE declares a width instead of a shared-expert count."""
        recorder = _Recorder(SimpleNamespace(
            model_type="qwen2_moe", hidden_size=2048, num_hidden_layers=24,
            num_attention_heads=16, intermediate_size=5632, vocab_size=151936,
            max_position_embeddings=32768, num_experts=60,
            moe_intermediate_size=1408, shared_expert_intermediate_size=5632,
        ))
        with _stub_registry(recorder):
            result = resolve_hf_model_spec({"pretrained_model_name_or_path": "x"})
        # 5632 / 1408 == 4 narrow experts of the routed width.
        self.assertEqual(result["num_shared_experts"], 4)

    def test_declared_count_is_left_alone(self) -> None:
        """A config that states the count keeps it."""
        recorder = _Recorder(SimpleNamespace(
            model_type="deepseek_v3", hidden_size=7168, num_hidden_layers=61,
            num_attention_heads=128, intermediate_size=18432, vocab_size=129280,
            max_position_embeddings=4096, n_routed_experts=256,
            moe_intermediate_size=2048, shared_expert_intermediate_size=8192,
            n_shared_experts=1,
        ))
        with _stub_registry(recorder):
            result = resolve_hf_model_spec({"pretrained_model_name_or_path": "x"})
        self.assertEqual(result["num_shared_experts"], 1)

    def test_a_model_stated_by_hand_is_derived_as_a_resolved_one(self) -> None:
        """
        Feature: resolve_hf_model_spec on a model its section states by hand (I1).
        Description: An all-MoE model, experts 256 wide and one shared
            expert 512 wide, stated in config_overrides with no checkpoint,
            and the same model whose checkpoint cannot be resolved.
        Expectation: Both counts the shared expert as two of the routed
            width and give the model the shared expert's width as its dense
            one, as the resolved path does; without that, intermediate_size
            stayed unstated and the shared expert was priced at nothing.
        """
        overrides = {
            "hidden_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 8, "vocab_size": 32000,
            "num_experts": 16, "num_experts_per_tok": 4, "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 512,
        }
        stated = resolve_hf_model_spec({"name": "qwen3_5_moe", "config_overrides": dict(overrides)})
        with patch.object(spec_mod, "_get_hf_config", side_effect=OSError("offline")):
            fallback = resolve_hf_model_spec({"pretrained_model_name_or_path": "x",
                                              "config_overrides": dict(overrides)})
        for spec in (stated, fallback):
            self.assertEqual((spec["num_shared_experts"], spec["intermediate_size"]), (2, 512))

    def test_the_checkpoints_dtype_is_read_by_name(self) -> None:
        """
        Feature: resolve_hf_model_spec, the checkpoint's dtype (I12).
        Description: A checkpoint whose config holds its dtype as torch's
            bfloat16, as newer Transformers state it under dtype, resolved
            with no torch_dtype in the section, with auto, and with float32.
        Expectation: bfloat16 by name for the first two, auto stating
            nothing over the checkpoint; the section's float32 for the last.
        """
        recorder = _Recorder(SimpleNamespace(model_type="llama", hidden_size=1024, num_hidden_layers=4,
                                             num_attention_heads=8, vocab_size=32000, dtype=torch.bfloat16))
        got = []
        for stated in ({}, {"torch_dtype": "auto"}, {"torch_dtype": "float32"}):
            with _stub_registry(recorder):
                got.append(resolve_hf_model_spec({"pretrained_model_name_or_path": "x", **stated})["torch_dtype"])
        self.assertEqual(got, ["bfloat16", "bfloat16", "float32"])


class TestVisualSequenceLength(unittest.TestCase):
    """The encoder sequence length is derived, overridden or defaulted."""

    @staticmethod
    def _vl(**vision):
        """Return a composite config whose vision tower carries *vision*."""
        base = {"hidden_size": 1152, "depth": 27, "num_heads": 16,
                "intermediate_size": 4304, "spatial_merge_size": 2}
        base.update(vision)
        return SimpleNamespace(
            model_type="qwen3_vl_moe",
            text_config=SimpleNamespace(
                hidden_size=2048, num_hidden_layers=24, num_attention_heads=16,
                intermediate_size=5632, vocab_size=151936,
                max_position_embeddings=128000,
            ),
            vision_config=SimpleNamespace(**base),
        )

    def test_derived_from_the_positional_grid(self) -> None:
        """The grid over the spatial merge is the default bound."""
        recorder = _Recorder(self._vl(num_position_embeddings=2304))
        with _stub_registry(recorder):
            result = resolve_hf_model_spec({"pretrained_model_name_or_path": "x"})
        self.assertEqual(result["vision"]["max_position_embeddings"], 576)
        self.assertEqual(result["vision"]["layers"], [{"kind": "encoder", "count": 27}])

    def test_override_wins(self) -> None:
        """context.visual_seq_len replaces the derived bound."""
        recorder = _Recorder(self._vl(num_position_embeddings=2304))
        with _stub_registry(recorder):
            result = resolve_hf_model_spec(
                {"pretrained_model_name_or_path": "x"}, visual_seq_len=2304)
        self.assertEqual(result["vision"]["max_position_embeddings"], 2304)

    def test_missing_grid_warns_and_defaults(self) -> None:
        """A tower with no positional grid cannot be derived from."""
        recorder = _Recorder(self._vl())
        with _stub_registry(recorder), self.assertLogs(
                spec_mod.__name__, level="WARNING") as captured:
            result = resolve_hf_model_spec({"pretrained_model_name_or_path": "x"})
        self.assertEqual(result["vision"]["max_position_embeddings"],
                         spec_mod._DEFAULT_VISUAL_SEQ_LEN)  # pylint: disable=protected-access
        self.assertTrue(any("num_position_embeddings" in m for m in captured.output))


class TestTransformersEntryPoint(unittest.TestCase):
    """The registry call carries the Trainer's model arguments."""

    def test_arguments_are_forwarded(self) -> None:
        """attn_implementation, dtype and loader kwargs reach the registry."""
        recorder = _Recorder(SimpleNamespace(
            model_type="qwen3_moe", hidden_size=2048, num_hidden_layers=48,
            num_attention_heads=32, intermediate_size=6144, vocab_size=151936,
            max_position_embeddings=40960,
        ))
        model_raw = {
            "pretrained_model_name_or_path": "local/model",
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "bfloat16",
            "local_files_only": True,
            "revision": "main",
            "cache_dir": None,
        }
        with _stub_registry(recorder):
            result = resolve_hf_model_spec(model_raw)

        self.assertEqual(len(recorder.calls), 1)
        path, attn, dtype, kwargs = recorder.calls[0]
        self.assertEqual(path, "local/model")
        self.assertEqual(attn, "flash_attention_2")
        self.assertEqual(dtype, "bfloat16")
        self.assertEqual(kwargs, {"local_files_only": True, "revision": "main"})
        self.assertEqual(result["name"], "qwen3_moe")

    def test_explicit_overrides_win_over_the_resolved_config(self) -> None:
        """config_overrides is the last word on any field."""
        recorder = _Recorder(SimpleNamespace(
            model_type="qwen3_moe", hidden_size=2048, num_hidden_layers=48,
            num_attention_heads=32, intermediate_size=6144, vocab_size=151936,
            max_position_embeddings=40960,
        ))
        with _stub_registry(recorder):
            result = resolve_hf_model_spec({
                "pretrained_model_name_or_path": "local/model",
                "config_overrides": {"num_hidden_layers": 4},
            })
        self.assertEqual(result["num_hidden_layers"], 4)
        self.assertEqual(result["hidden_size"], 2048)


class TestFromConfigRecipe(unittest.TestCase):
    """A from_config recipe names its model directory and its own shape."""

    def test_the_recipe_states_its_directory_and_its_crop(self) -> None:
        """
        Feature: resolve_hf_model_spec on a recipe whose factory builds its model.
        Description: The released Qwen3.5-MoE shape under config_path, cropped by the
            factory's arguments, then with config_overrides restating a count.
        Expectation: config_path names the directory the config is read from; the counts
            the recipe builds beat the released ones, and what it does not restate is the
            released config's; config_overrides stay the last word.
        """
        recorder = _Recorder(SimpleNamespace(
            model_type="qwen3_5_moe", hidden_size=2048, num_hidden_layers=40,
            num_attention_heads=16, num_key_value_heads=2, head_dim=256,
            num_experts=256, num_experts_per_tok=8, moe_intermediate_size=512,
            vocab_size=248320, max_position_embeddings=262144,
        ))
        recipe = {"_target_": "examples.training_demo.cropped_qwen3_5_moe.build_cropped_qwen3_5_moe",
                  "config_path": "/home/tt/models/Qwen3.5-35B-A3B-Base", "num_hidden_layers": 8, "num_experts": 64}
        with _stub_registry(recorder):
            cropped = resolve_hf_model_spec(recipe)
            overridden = resolve_hf_model_spec(dict(recipe, config_overrides={"num_hidden_layers": 2}))
        self.assertEqual(recorder.calls[0][0], "/home/tt/models/Qwen3.5-35B-A3B-Base")
        self.assertEqual([cropped[key] for key in ("num_hidden_layers", "num_experts", "hidden_size")], [8, 64, 2048])
        self.assertEqual(overridden["num_hidden_layers"], 2)


class TestKeptCensus(unittest.TestCase):
    """A census is kept on disk and read back while nothing it ran on has changed."""

    _LAYERS = ({"kind": "decoder", "count": 2},)

    def setUp(self) -> None:
        """An empty census folder of the test's own, and no census run in this process yet."""
        folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        for patcher in (patch.dict(os.environ, {"ND_CENSUS_CACHE": self.folder}),
                        patch.dict(spec_mod._CENSUSES, clear=True)):  # pylint: disable=protected-access
            patcher.start()
            self.addCleanup(patcher.stop)
        self.config = SimpleNamespace(to_json_string=lambda: '{"hidden_size": 64}')

    def _run(self, saved: float = 1.0) -> tuple:
        """The spec's census as a new process gets it, the census itself stubbed, and how often it ran."""
        spec_mod._CENSUSES.clear()  # pylint: disable=protected-access
        record = SimpleNamespace(to_dict=lambda: {"saved": saved, "seq_length": 256})
        with patch.object(spec_mod, "census_activations", return_value={"decoder": record}) as kinds:
            with patch.object(spec_mod, "census_output_activations", return_value=record):
                census = spec_mod._census(self.config, self._LAYERS, 256)  # pylint: disable=protected-access
        return census, kinds.call_count

    def test_a_later_run_reads_the_kept_census_until_its_code_changes(self) -> None:
        """
        Feature: the kept census.
        Description: Run the census of one model twice, as two runs of a search do; then
            change the digest the kept file holds of the census's own code, as an edit to
            that file would, and run it twice more.
        Expectation: The first run measures and keeps it, the second reads it back without
            measuring; after the change the next run measures again and keeps the new
            census, which the run after reads back.
        """
        first, measured = self._run()
        self.assertEqual((measured, first["activations"]["decoder"]["saved"]), (1, 1.0))
        self.assertEqual(self._run(saved=2.0), (first, 0))
        name, = os.listdir(self.folder)
        path = os.path.join(self.folder, name)
        with open(path, encoding="utf-8") as handle:
            kept = json.load(handle)
        kept["sources"]["auto_parallel/_layer_census.py"] = "0" * 64
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(kept, handle)
        census, measured = self._run(saved=2.0)
        self.assertEqual((measured, census["activations"]["decoder"]["saved"]), (1, 2.0))
        self.assertEqual(self._run(saved=3.0), (census, 0))


if __name__ == "__main__":
    unittest.main()
