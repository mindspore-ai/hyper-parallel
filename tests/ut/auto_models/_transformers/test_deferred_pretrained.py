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
"""Avoid Transformers checkpoint reads before the finalized HF loading stage."""
# Tests inspect build-pipeline helpers and the recipe registry they write.
# pylint: disable=protected-access
import gc
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, BertConfig, BertForSequenceClassification, LlamaConfig, LlamaForCausalLM

from hyper_parallel.models._transformers.auto_model import (
    HyperAutoModelForCausalLM, HyperAutoModelForImageTextToText, HyperAutoModelForSequenceClassification,
)
from hyper_parallel.models._transformers.model_builder import _declare_constructed_buffers
from hyper_parallel.models.materialization import (
    MaterializationContext, rebuild_materialized_state, register_rebuildable_buffer,
)


_TARGET = "hyper_parallel.models._transformers.auto_model"
_REAL_TO_EMPTY = torch.nn.Module.to_empty


def _poisoned_to_empty(self, *, device, recurse=True):
    """to_empty whose fresh storage holds a sentinel instead of allocator leftovers.

    Freed memory is often handed straight back for a same-sized tensor, so
    uninitialized storage can hold the right bytes by chance; poisoning makes
    "was this restored" a deterministic question.
    """
    module = _REAL_TO_EMPTY(self, device=device, recurse=recurse)
    with torch.no_grad():
        for tensor in (*module.parameters(), *module.buffers()):
            tensor.fill_(float("nan") if tensor.is_floating_point() else -1)
    return module


def _make_check_callbacks(test_case, events, model, failed):
    """Create loader and verifier callbacks bound to one test scenario."""
    def build(*_args, **kwargs):
        test_case.assertNotIn("verify_hf_weights", kwargs)
        events.append("loaded")
        return model

    def check(actual, source, output, *, phase):
        test_case.assertIs(actual, model)
        test_case.assertEqual(events, ["loaded"])
        test_case.assertEqual((source, output, phase),
                              ("/weights", "/report", "after_load_before_first_forward"))
        events.append("checked")
        if failed:
            raise RuntimeError("weight mismatch")

    return build, check


class TestDeferredPretrained(unittest.TestCase):
    """Construct unquantized models from config for both finalized HF loaders."""

    def test_from_config_binds_standard_hf_auto_model_parent(self):
        """Feature: remote-code lookup uses the standard AutoModel identity.

        Description: Delegate a HyperAutoModel config-only construction to Hugging Face.
        Expectation: The standard parent receives the config and remote-code switch.
        """
        config = object()
        model = object()
        with patch.object(AutoModelForCausalLM, "from_config", return_value=model) as parent:
            result = HyperAutoModelForCausalLM._from_config_parent_class(
                config,
                trust_remote_code=True,
            )

        self.assertIs(result, model)
        parent.assert_called_once_with(config, trust_remote_code=True)

    def test_loader_selects_initial_construction_and_preserves_final_load(self):
        """Feature: automatic deferred HF loading.

        Description: Exercise HF, custom and generated models with DCP and legacy loaders.
        Expectation: Both loaders construct on meta without a checkpoint path and load after sharding.
        """
        for loader in ("dcp", "legacy"):
            for is_hf_model in (True, False):
                with self.subTest(loader=loader, is_hf_model=is_hf_model):
                    model = torch.nn.Linear(2, 2)
                    with patch.dict(os.environ, {"HYPER_PARALLEL_HF_LOADER": loader}), \
                            patch(_TARGET + "._init_model", return_value=(None, model)) as initialize, \
                            patch(_TARGET + ".apply_model_infrastructure", return_value=model) as infrastructure, \
                            patch(_TARGET + "._current_device", return_value=torch.device("cpu")), \
                            patch("torch.distributed.is_initialized", return_value=False):
                        HyperAutoModelForImageTextToText._build_model(
                            "/weights", is_hf_model=is_hf_model, hf_config=None, mesh=None,
                            sharding_planner=None, fsdp2_manager=None, peft_config=None,
                            torch_dtype=torch.float32, attn_implementation="eager", validate_placement=False,
                            load_base_model=True, local_files_only=True, ignore_mismatched_sizes=True)
                    self.assertEqual(initialize.call_args.args[1], None)
                    self.assertEqual("local_files_only" in initialize.call_args.kwargs, False)
                    self.assertNotIn("ignore_mismatched_sizes", initialize.call_args.kwargs)
                    self.assertEqual(infrastructure.call_args.kwargs["pretrained_path"], "/weights")
                    self.assertEqual(infrastructure.call_args.kwargs["is_meta_device"], True)
                    self.assertTrue(infrastructure.call_args.kwargs["load_base_model"])
                    self.assertNotIn("init_from_config", initialize.call_args.kwargs)

    def test_from_config_does_not_enable_checkpoint_loading(self):
        """Feature: random initialization remains independent of HF loader selection.

        Description: Build without base weights while the DCP loader is selected.
        Expectation: No checkpoint path is passed and final loading stays disabled.
        """
        model = torch.nn.Linear(2, 2)
        with patch.dict(os.environ, {"HYPER_PARALLEL_HF_LOADER": "dcp"}), \
                patch(_TARGET + "._init_model", return_value=(None, model)) as initialize, \
                patch(_TARGET + ".apply_model_infrastructure", return_value=model) as infrastructure, \
                patch(_TARGET + "._current_device", return_value=torch.device("cpu")):
            HyperAutoModelForImageTextToText._build_model(
                None, is_hf_model=False, hf_config=None, mesh=None, sharding_planner=None,
                fsdp2_manager=None, peft_config=None, torch_dtype=torch.float32,
                attn_implementation="eager", validate_placement=False, load_base_model=False)
        self.assertIsNone(initialize.call_args.args[1])
        self.assertFalse(infrastructure.call_args.kwargs["load_base_model"])

    def test_quantized_dcp_fails_before_initial_construction(self):
        """Feature: unsupported quantized DCP initialization.

        Description: Request quantized construction with DCP selected.
        Expectation: Fail explicitly before creating a partially initialized model.
        """
        with patch.dict(os.environ, {"HYPER_PARALLEL_HF_LOADER": "dcp"}), \
                patch(_TARGET + "._init_model") as initialize:
            with self.assertRaisesRegex(ValueError, "quantized initialization"):
                HyperAutoModelForImageTextToText._build_model(
                    "/weights", is_hf_model=True, hf_config=None, mesh=None, sharding_planner=None,
                    fsdp2_manager=None, peft_config=None, torch_dtype=torch.float32,
                    attn_implementation="eager", validate_placement=False, load_base_model=True,
                    quantization_config=object())
        initialize.assert_not_called()

    def test_legacy_quantized_construction_retains_transformers_loading(self):
        """Feature: Transformers-owned quantized initialization.

        Description: Build a quantized HF model using the legacy path.
        Expectation: Transformers receives the checkpoint and non-meta construction is preserved.
        """
        model = torch.nn.Linear(2, 2)
        quantization = object()
        with patch.dict(os.environ, {"HYPER_PARALLEL_HF_LOADER": "legacy"}), \
                patch(_TARGET + "._init_model", return_value=(None, model)) as initialize, \
                patch(_TARGET + ".apply_model_infrastructure", return_value=model) as infrastructure, \
                patch(_TARGET + "._current_device", return_value=torch.device("cpu")):
            HyperAutoModelForImageTextToText._build_model(
                "/weights", is_hf_model=True, hf_config=None, mesh=None, sharding_planner=None,
                fsdp2_manager=None, peft_config=None, torch_dtype=torch.float32,
                attn_implementation="eager", validate_placement=False, load_base_model=True,
                quantization_config=quantization)
        self.assertEqual(initialize.call_args.args[1], "/weights")
        self.assertIs(initialize.call_args.kwargs["quantization_config"], quantization)
        self.assertFalse(infrastructure.call_args.kwargs["is_meta_device"])

    def test_real_single_rank_model_loads_exact_weights_without_transformers_preload(self):
        """Feature: finalized single-rank pretrained loading.

        Description: Load a tiny local BERT through the actual build and checkpoint pipeline.
        Expectation: Both loaders reproduce all source tensors without Transformers weight reads.
        """
        config = BertConfig(vocab_size=16, hidden_size=8, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=16)
        source = BertForSequenceClassification(config)
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            for loader in ("legacy", "dcp"):
                with self.subTest(loader=loader):
                    with patch.dict(os.environ, {"HYPER_PARALLEL_HF_LOADER": loader}), \
                            patch("transformers.core_model_loading._materialize_copy",
                                  side_effect=AssertionError("Unexpected Transformers weight read")), \
                            patch(_TARGET + "._current_device", return_value=torch.device("cpu")), \
                            patch.object(torch.nn.Module, "to_empty", _poisoned_to_empty):
                        model = HyperAutoModelForSequenceClassification._build_model(
                            directory, is_hf_model=True, hf_config=config, mesh=None,
                            sharding_planner=None, fsdp2_manager=None, peft_config=None,
                            torch_dtype=torch.float32, attn_implementation="eager", validate_placement=False,
                            load_base_model=True, local_files_only=True, ignore_mismatched_sizes=True)
                    self.assertEqual(model._hp_checkpoint_load_report.missing_keys, ())
                    for name, expected in source.state_dict().items():
                        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
                    # position_ids / token_type_ids are non-persistent: no checkpoint carries them.
                    loaded_buffers = dict(model.named_buffers())
                    for name, expected in source.named_buffers():
                        torch.testing.assert_close(loaded_buffers[name], expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pretrained_rope_buffers_and_logits_match_transformers(self):
        """Feature: derived buffers survive deferred pretrained loading.

        Description: Load a tiny Llama through the meta build path and through Transformers.
        Expectation: RoPE buffers, parameters and logits agree bitwise for both loaders.
        """
        config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32)
        torch.manual_seed(0)
        source = LlamaForCausalLM(config)
        tokens = torch.randint(0, 64, (2, 12), generator=torch.Generator().manual_seed(1))
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory)
            reference = LlamaForCausalLM.from_pretrained(directory, dtype=torch.float32,
                                                         attn_implementation="eager").eval()
            expected_buffers = {name: buffer.clone() for name, buffer in reference.named_buffers()}
            with torch.no_grad():
                expected_logits = reference(input_ids=tokens).logits
            # The reference maps the safetensors file; Windows refuses to delete a mapped file.
            del reference
            gc.collect()
            for loader in ("legacy", "dcp"):
                with self.subTest(loader=loader):
                    with patch.dict(os.environ, {"HYPER_PARALLEL_HF_LOADER": loader}), \
                            patch(_TARGET + "._current_device", return_value=torch.device("cpu")), \
                            patch.object(torch.nn.Module, "to_empty", _poisoned_to_empty):
                        model = HyperAutoModelForCausalLM._build_model(
                            directory, is_hf_model=True, hf_config=config, mesh=None,
                            sharding_planner=None, fsdp2_manager=None, peft_config=None,
                            torch_dtype=torch.float32, attn_implementation="eager", validate_placement=False,
                            load_base_model=True, local_files_only=True).eval()
                    self.assertEqual(set(dict(model.named_buffers())), set(expected_buffers))
                    for name, buffer in model.named_buffers():
                        torch.testing.assert_close(buffer, expected_buffers[name], rtol=0, atol=0)
                    with torch.no_grad():
                        torch.testing.assert_close(model(input_ids=tokens).logits, expected_logits,
                                                   rtol=0, atol=0)

    def test_public_pretrained_checks_finalized_model_and_propagates_failure(self):
        """Feature: generic weight checking.

        Description: Request verification through the public loader for two model families.
        Expectation: Only the finalized model is checked and failures prevent returning it.
        """
        for auto_class in (HyperAutoModelForImageTextToText, HyperAutoModelForSequenceClassification):
            for enabled, failed in ((False, False), (True, False), (True, True)):
                with self.subTest(auto_class=auto_class, enabled=enabled, failed=failed):
                    events = []
                    model = torch.nn.Linear(2, 2)
                    build_callback, check_callback = _make_check_callbacks(self, events, model, failed)

                    with patch(_TARGET + ".instantiate_infrastructure", return_value=(None, None)), \
                            patch(_TARGET + "._current_device", return_value=torch.device("cpu")), \
                            patch(_TARGET + ".get_hf_config", return_value=BertConfig()), \
                            patch(_TARGET + ".get_is_hf_model", return_value=True), \
                            patch.object(auto_class, "_build_model", side_effect=build_callback), \
                            patch(_TARGET + ".check_hf_weights", side_effect=check_callback) as verifier:
                        options = {"distributed_setup": SimpleNamespace(mesh_context=None),
                                   "verify_hf_weights": "/report" if enabled else None}
                        if failed:
                            with self.assertRaisesRegex(RuntimeError, "weight mismatch"):
                                auto_class.from_pretrained("/weights", **options)
                        else:
                            self.assertIs(auto_class.from_pretrained("/weights", **options), model)
                        self.assertEqual(verifier.call_count, int(enabled))

    def test_public_pretrained_rejects_check_without_loading(self):
        """Feature: weight-check preconditions.

        Description: Disable loading while requesting weight verification.
        Expectation: Fail before constructing model infrastructure.
        """
        with patch(_TARGET + ".instantiate_infrastructure") as infrastructure:
            with self.assertRaisesRegex(ValueError, "load_base_model=True"):
                HyperAutoModelForSequenceClassification.from_pretrained(
                    "/weights", load_base_model=False, verify_hf_weights="/report")
        infrastructure.assert_not_called()


class TestConstructedBufferDeclaration(unittest.TestCase):
    """Constructed non-persistent buffers get a rebuild recipe before materialization."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_constructed_value_is_restored_after_to_empty(self):
        """Feature: generic recipe for native derived buffers.

        Description: Declare, wipe storage with to_empty, then rebuild for a checkpoint load.
        Expectation: The constructed value returns and persistent state is not given a recipe.
        """
        model = torch.nn.Module()
        model.register_buffer("derived", torch.arange(4, dtype=torch.float32), persistent=False)
        model.register_buffer("persistent_state", torch.ones(2))
        _declare_constructed_buffers(model)
        self.assertEqual(set(model._hp_rebuildable_buffer_specs), {"derived"})
        model.to_empty(device="cpu")
        with torch.no_grad():
            model.derived.fill_(-1)
        rebuild_materialized_state(model, MaterializationContext("checkpoint_load", torch.device("cpu")))
        torch.testing.assert_close(model.derived, torch.arange(4, dtype=torch.float32), rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_adapter_recipe_wins_and_meta_buffer_is_reported(self):
        """Feature: declared recipes take precedence over the generic one.

        Description: One buffer already has an adapter recipe, another was built on meta.
        Expectation: The adapter recipe is kept and the meta buffer is named in a warning.
        """
        model = torch.nn.Module()
        model.register_buffer("declared", torch.zeros(2), persistent=False)
        register_rebuildable_buffer(model, "declared", value=torch.full((2,), 7.0))
        model.register_buffer("meta_only", torch.empty(2, device="meta"), persistent=False)
        with self.assertLogs("hyper_parallel.models._transformers.model_builder", level="WARNING") as logs:
            _declare_constructed_buffers(model)
        self.assertEqual(set(model._hp_rebuildable_buffer_specs), {"declared"})
        torch.testing.assert_close(model._hp_rebuildable_buffer_specs["declared"].source,
                                   torch.full((2,), 7.0), rtol=0, atol=0)
        self.assertIn("meta_only", "\n".join(logs.output))
