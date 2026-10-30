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
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import BertConfig, BertForSequenceClassification

from hyper_parallel.models._transformers.auto_model import (
    HyperAutoModelForImageTextToText, HyperAutoModelForSequenceClassification,
)
from tests.common.mark_utils import arg_mark


_TARGET = "hyper_parallel.models._transformers.auto_model"


class TestDeferredPretrained(unittest.TestCase):
    """Construct unquantized models from config for both finalized HF loaders."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_loader_selects_initial_construction_and_preserves_final_load(self):
        """Feature: automatic deferred HF loading.

        Description: Exercise HF, custom and generated models with DCP and legacy loaders.
        Expectation: Both loaders construct on meta without a checkpoint path and load after sharding.
        """
        for loader in ("dcp", "legacy"):
            for is_hf_model, backend in ((True, None), (False, None)):
                with self.subTest(loader=loader, is_hf_model=is_hf_model, backend=backend):
                    model = torch.nn.Linear(2, 2)
                    with patch.dict(os.environ, {"HYPER_PARALLEL_HF_LOADER": loader}), \
                            patch(_TARGET + "._init_model", return_value=(None, model)) as initialize, \
                            patch(_TARGET + ".apply_model_infrastructure", return_value=model) as infrastructure, \
                            patch(_TARGET + "._current_device", return_value=torch.device("cpu")), \
                            patch("torch.distributed.is_initialized", return_value=False):
                        HyperAutoModelForImageTextToText._build_model(
                            "/weights", is_hf_model=is_hf_model, hf_config=None, mesh=None,
                            sharding_planner=None, fsdp2_manager=None, backend=backend, peft_config=None,
                            torch_dtype=torch.float32, attn_implementation="eager", validate_placement=False,
                            load_base_model=True, local_files_only=True, ignore_mismatched_sizes=True)
                    self.assertEqual(initialize.call_args.args[1], None)
                    self.assertEqual("local_files_only" in initialize.call_args.kwargs, False)
                    self.assertNotIn("ignore_mismatched_sizes", initialize.call_args.kwargs)
                    self.assertEqual(infrastructure.call_args.kwargs["pretrained_path"], "/weights")
                    self.assertEqual(infrastructure.call_args.kwargs["is_meta_device"], True)
                    self.assertTrue(infrastructure.call_args.kwargs["load_base_model"])
                    self.assertNotIn("init_from_config", initialize.call_args.kwargs)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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
                fsdp2_manager=None, backend=None, peft_config=None, torch_dtype=torch.float32,
                attn_implementation="eager", validate_placement=False, load_base_model=False)
        self.assertIsNone(initialize.call_args.args[1])
        self.assertFalse(infrastructure.call_args.kwargs["load_base_model"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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
                    fsdp2_manager=None, backend=None, peft_config=None, torch_dtype=torch.float32,
                    attn_implementation="eager", validate_placement=False, load_base_model=True,
                    quantization_config=object())
        initialize.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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
                fsdp2_manager=None, backend=None, peft_config=None, torch_dtype=torch.float32,
                attn_implementation="eager", validate_placement=False, load_base_model=True,
                quantization_config=quantization)
        self.assertEqual(initialize.call_args.args[1], "/weights")
        self.assertIs(initialize.call_args.kwargs["quantization_config"], quantization)
        self.assertFalse(infrastructure.call_args.kwargs["is_meta_device"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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
                            patch(_TARGET + "._current_device", return_value=torch.device("cpu")):
                        model = HyperAutoModelForSequenceClassification._build_model(
                            directory, is_hf_model=True, hf_config=config, mesh=None,
                            sharding_planner=None, fsdp2_manager=None, backend=None, peft_config=None,
                            torch_dtype=torch.float32, attn_implementation="eager", validate_placement=False,
                            load_base_model=True, local_files_only=True, ignore_mismatched_sizes=True)
                    self.assertEqual(model._hp_checkpoint_load_report.missing_keys, ())
                    for name, expected in source.state_dict().items():
                        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
