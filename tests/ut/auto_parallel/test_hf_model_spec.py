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
    exec_overrides,
    is_auto_models_schema,
    resolve_hf_model_spec,
)
from hyper_parallel.auto_parallel._model_spec import ModelSpecError

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
        Feature: resolve_hf_model_spec on a model its section states by hand.
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
        Feature: resolve_hf_model_spec, the checkpoint's dtype.
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


class TestOverridesOfTheRun(unittest.TestCase):
    """config_overrides states the model, and a few keys of the run."""

    _MODEL = {"hidden_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 8,
              "vocab_size": 32000}

    def test_the_run_keys_leave_the_model_spec(self) -> None:
        """The offset, the recompute and the capacity factor are not the model's."""
        run = {"offset": [1, -1], "full_rec": [2, 1], "sel_rec": False, "capacity_factor": 1.5}
        model_raw = {"name": "toy", "config_overrides": dict(self._MODEL, **run)}
        resolved = resolve_hf_model_spec(model_raw)
        self.assertEqual(sorted(set(run) & set(resolved)), [], f"resolved={resolved}")
        self.assertEqual(exec_overrides(model_raw), run)

    def test_an_unknown_key_is_refused(self) -> None:
        """A misspelt model field fails by name instead of pricing nothing."""
        model_raw = {"name": "toy", "config_overrides": dict(self._MODEL, hidden_sise=2048)}
        with self.assertRaisesRegex(ModelSpecError, "hidden_sise"):
            resolve_hf_model_spec(model_raw)



class TestQkNorm(unittest.TestCase):
    """Whether a model's attention normalizes its queries and keys is settled with its spec."""

    _MODEL = {"hidden_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 8, "vocab_size": 32000}

    def _qk_norm(self, model_type: str, **stated: Any) -> Any:
        """The QK-norm the spec of a *model_type* config stating *stated* settles on."""
        with _stub_registry(_Recorder(SimpleNamespace(model_type=model_type, **self._MODEL, **stated))):
            return resolve_hf_model_spec({"pretrained_model_name_or_path": "local/model"})["qk_norm"]

    def test_the_qwen3_generation_normalizes_queries_and_keys(self) -> None:
        """
        Feature: the QK-norm of a Transformers config, which Qwen3's does not state.
        Description: Qwen3, Qwen3-MoE, Qwen3.5-MoE, Qwen3-VL-MoE, Llama and Qwen2 configs.
        Expectation: The Qwen3 generation has one, Llama and Qwen2 none.
        """
        types_ = ("qwen3", "qwen3_moe", "qwen3_5_moe", "qwen3_vl_moe", "llama", "qwen2")
        self.assertEqual([self._qk_norm(model_type) for model_type in types_],
                         [True, True, True, True, False, False])

    def test_a_stated_qk_norm_wins(self) -> None:
        """
        Feature: the QK-norm of a Transformers config that states it.
        Description: A Qwen3 config stating use_qk_norm False, and a GLM-4.5
            one stating it True.
        Expectation: Each is taken at its word.
        """
        self.assertEqual([self._qk_norm("qwen3", use_qk_norm=False), self._qk_norm("glm4_moe", use_qk_norm=True)],
                         [False, True])


class TestVectors(unittest.TestCase):
    """The biases and norms a Transformers model's layers hold are settled with its spec."""

    _MODEL = {"hidden_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 8, "vocab_size": 32000}
    _FACTS = ("qkv_bias", "o_bias", "mlp_bias", "norm_bias", "layer_norms", "shared_expert_gate")

    def _facts(self, model_type: str, **stated: Any) -> Any:
        """The vector facts the spec of a *model_type* config stating *stated* settles on."""
        with _stub_registry(_Recorder(SimpleNamespace(model_type=model_type, **self._MODEL, **stated))):
            spec = resolve_hf_model_spec({"pretrained_model_name_or_path": "local/model"})
        return tuple(spec[name] for name in self._FACTS)

    def test_each_family_holds_its_own(self):
        """
        Feature: the vector facts of a Transformers config.
        Description: A Llama config stating attention_bias and mlp_bias, a
            Qwen2 one stating neither, a Qwen2-MoE and a Qwen3.5-MoE one,
            and a Mixtral one.
        Expectation: Llama biases all four projections and its
            feed-forward; Qwen2's generation its query, key and value
            projections; Qwen's MoE generations gate their shared expert;
            each layer holds two RMSNorms, and nothing else is biased.
        """
        self.assertEqual(self._facts("llama", attention_bias=True, mlp_bias=True), (True, True, True, False, 2, False))
        self.assertEqual(self._facts("qwen2"), (True, False, False, False, 2, False))
        self.assertEqual(self._facts("qwen2_moe")[0::5], (True, True))
        self.assertEqual(self._facts("qwen3_5_moe")[0::5], (False, True))
        self.assertEqual(self._facts("mixtral"), (False, False, False, False, 2, False))


class TestFromConfigRecipe(unittest.TestCase):
    """A from_config recipe names its model directory and its own shape."""

    @staticmethod
    def _released_config() -> SimpleNamespace:
        """The released Qwen3.5-MoE shape, before any crop."""
        return SimpleNamespace(
            model_type="qwen3_5_moe", hidden_size=2048, num_hidden_layers=40,
            num_attention_heads=16, num_key_value_heads=2, head_dim=256,
            num_experts=256, num_experts_per_tok=8, moe_intermediate_size=512,
            vocab_size=248320, max_position_embeddings=262144,
        )

    def test_config_path_names_the_model_directory(self) -> None:
        """A factory recipe carries config_path instead of a pretrained path."""
        recorder = _Recorder(self._released_config())
        with _stub_registry(recorder):
            result = resolve_hf_model_spec({
                "_target_": "examples.training_demo.cropped_qwen3_5_moe.build_cropped_qwen3_5_moe",
                "config_path": "/home/tt/models/Qwen3.5-35B-A3B-Base",
            })
        self.assertEqual(recorder.calls[0][0], "/home/tt/models/Qwen3.5-35B-A3B-Base")
        self.assertEqual(result["name"], "qwen3_5_moe")
        self.assertEqual(result["num_hidden_layers"], 40)

    def test_factory_arguments_crop_the_released_config(self) -> None:
        """The layer and expert counts the recipe builds beat the released ones."""
        recorder = _Recorder(self._released_config())
        with _stub_registry(recorder):
            result = resolve_hf_model_spec({
                "_target_": "examples.training_demo.cropped_qwen3_5_moe.build_cropped_qwen3_5_moe",
                "config_path": "/home/tt/models/Qwen3.5-35B-A3B-Base",
                "num_hidden_layers": 8,
                "num_experts": 64,
                "torch_dtype": "bfloat16",
            })
        self.assertEqual(result["num_hidden_layers"], 8)
        self.assertEqual(result["num_experts"], 64)
        # Everything the recipe does not restate still comes from the released config.
        self.assertEqual(result["hidden_size"], 2048)
        self.assertEqual(result["num_experts_per_tok"], 8)

    def test_config_overrides_still_beat_factory_arguments(self) -> None:
        """The declared override form stays the last word."""
        recorder = _Recorder(self._released_config())
        with _stub_registry(recorder):
            result = resolve_hf_model_spec({
                "config_path": "/models/qwen",
                "num_hidden_layers": 8,
                "config_overrides": {"num_hidden_layers": 2},
            })
        self.assertEqual(result["num_hidden_layers"], 2)

    def test_error_names_every_accepted_key(self) -> None:
        """A model section with no shape at all says what would satisfy it."""
        with self.assertRaises(ValueError) as raised:
            resolve_hf_model_spec({"_target_": "some.factory", "torch_dtype": "bfloat16"})
        message = str(raised.exception)
        for key in ("pretrained_model_name_or_path", "config_path", "config_overrides"):
            self.assertIn(key, message)


class _Record:
    """A census record as the census returns one."""

    def __init__(self, saved: float) -> None:
        """Keep *saved* bytes per token."""
        self.saved = saved

    def to_dict(self) -> dict:
        """The record's plain form."""
        return {"saved": self.saved, "seq_length": 256}


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
        with patch.object(spec_mod, "census_activations", return_value={"decoder": _Record(saved)}) as kinds, \
                patch.object(spec_mod, "census_output_activations", return_value=_Record(saved / 2)):
            census = spec_mod._census(self.config, self._LAYERS, 256)  # pylint: disable=protected-access
        return census, kinds.call_count

    def _kept(self) -> list:
        """The files the folder holds."""
        return [os.path.join(self.folder, name) for name in os.listdir(self.folder)]

    def test_a_later_run_reads_the_kept_census(self) -> None:
        """
        Feature: the kept census.
        Description: Run the census of one model twice, as two runs of a search do.
        Expectation: The first run measures and keeps it, the second reads the same census back
            without measuring.
        """
        first, measured = self._run()
        second, measured_again = self._run(saved=2.0)
        self.assertEqual((measured, measured_again), (1, 0))
        self.assertEqual(second, first)
        self.assertEqual(first["activations"]["decoder"]["saved"], 1.0)

    def test_a_changed_file_measures_again(self) -> None:
        """
        Feature: the kept census.
        Description: Keep a census, then change the digest it holds of the census's own code, as
            an edit to that file would.
        Expectation: The next run measures again and keeps the new census, which the run after
            reads back.
        """
        self._run()
        path, = self._kept()
        with open(path, encoding="utf-8") as handle:
            kept = json.load(handle)
        kept["sources"]["auto_parallel/_layer_census.py"] = "0" * 64
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(kept, handle)
        census, measured = self._run(saved=2.0)
        self.assertEqual(measured, 1)
        self.assertEqual(census["activations"]["decoder"]["saved"], 2.0)
        self.assertEqual(self._run(saved=3.0), (census, 0))

    def test_the_kept_census_names_the_code_it_ran_on(self) -> None:
        """
        Feature: the kept census.
        Description: Keep a census and read the files it is checked against.
        Expectation: The census's own code and the spec resolver are among them, and none of the
            search's code, whose changes cannot change a census.
        """
        self._run()
        path, = self._kept()
        with open(path, encoding="utf-8") as handle:
            sources = json.load(handle)["sources"]
        self.assertIn("auto_parallel/_layer_census.py", sources)
        self.assertIn("auto_parallel/_hf_model_spec.py", sources)
        self.assertFalse([source for source in sources if source.startswith("auto_parallel/sapp_nd/")])

    def test_line_endings_do_not_change_a_file(self) -> None:
        """
        Feature: the kept census's file digests.
        Description: The same code with Windows and with Unix line endings, as two checkouts of one
            commit can hold it, and one line changed.
        Expectation: The first two are one file to the census, the third another.
        """
        digests = []
        for name, text in (("crlf.py", b"x = 1\r\ny = 2\r\n"), ("lf.py", b"x = 1\ny = 2\n"),
                           ("other.py", b"x = 1\ny = 3\n")):
            path = os.path.join(self.folder, name)
            with open(path, "wb") as handle:
                handle.write(text)
            digests.append(spec_mod._source_digest(path))  # pylint: disable=protected-access
        self.assertEqual(digests[0], digests[1])
        self.assertNotEqual(digests[1], digests[2])

    def test_off_keeps_none(self) -> None:
        """
        Feature: ND_CENSUS_CACHE=off.
        Description: Run the census twice with the cache off.
        Expectation: Both runs measure, and nothing is written.
        """
        with patch.dict(os.environ, {"ND_CENSUS_CACHE": "off"}):
            measured = [self._run()[1] for _ in range(2)]
        self.assertEqual(measured, [1, 1])
        self.assertEqual(self._kept(), [])

    def test_a_damaged_file_measures_again(self) -> None:
        """
        Feature: the kept census.
        Description: Keep a census, then cut its file short, as a run stopped mid-write would
            leave it were it written in place.
        Expectation: The next run measures again without an error.
        """
        self._run()
        path, = self._kept()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"sources": {')
        self.assertEqual(self._run(saved=2.0)[1], 1)

    def test_a_refused_census_is_not_kept(self) -> None:
        """
        Feature: the kept census.
        Description: A census this host cannot run, as one that needs a native kernel.
        Expectation: Nothing is kept, so a host that can run it measures it.
        """
        spec_mod._CENSUSES.clear()  # pylint: disable=protected-access
        with patch.object(spec_mod, "census_activations", side_effect=spec_mod.CensusUnavailable("no kernel")):
            self.assertEqual(spec_mod._census(self.config, self._LAYERS, 256), {})  # pylint: disable=protected-access
        self.assertEqual(self._kept(), [])


if __name__ == "__main__":
    unittest.main()
