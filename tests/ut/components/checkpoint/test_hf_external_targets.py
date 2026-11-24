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
"""External host parameters stay outside ordinary HF checkpoint reads."""
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from hyper_parallel.components.checkpoint.huggingface_checkpointer import HuggingFaceCheckpointer


class TestExternalHFTargets(unittest.TestCase):
    """Keep externally loaded tables intact in both pretrained loaders."""

    def test_external_parameters_and_source_keys_are_excluded(self):
        """Feature: external host table materialization.

        Description: Exclude an externally loaded table whose checkpoint has a different global shape.
        Expectation: Dense weights load strictly, external values remain unchanged and no keys are unexpected.
        """
        with tempfile.TemporaryDirectory() as directory:
            save_file({"weight": torch.tensor([2., 3.]), "external.weight": torch.ones(16, 2)},
                      str(Path(directory) / "model.safetensors"))
            for loader in ("legacy", "dcp"):
                with self.subTest(loader=loader):
                    model = nn.Module()
                    model.weight = nn.Parameter(torch.zeros(2))
                    model.external = nn.Embedding(4, 2)
                    with torch.no_grad():
                        model.external.weight.fill_(7)
                    report = HuggingFaceCheckpointer(loader=loader, weights_mapping=[]).load(
                        directory, {"model": model}, external_target_fqns=frozenset({"external.weight"}),
                        claimed_source_keys=frozenset({"external.weight", "manifest_only_key"}),
                    )["load_report"]
                    self.assertEqual(report.loaded_keys, ("weight",))
                    self.assertEqual(report.missing_keys, ())
                    self.assertEqual(report.unexpected_keys, ())
                    torch.testing.assert_close(model.weight, torch.tensor([2., 3.]), rtol=0, atol=0)
                    torch.testing.assert_close(model.external.weight, torch.full((4, 2), 7.), rtol=0, atol=0)

    def test_external_target_must_exist(self):
        """Feature: external target ownership validation.

        Description: Exclude a target absent from the actual model.
        Expectation: Both loaders reject it before loading ordinary parameters.
        """
        with tempfile.TemporaryDirectory() as directory:
            save_file({"weight": torch.ones(2)}, str(Path(directory) / "model.safetensors"))
            for loader in ("legacy", "dcp"):
                with self.subTest(loader=loader):
                    model = nn.Module()
                    model.weight = nn.Parameter(torch.zeros(2))
                    with self.assertRaisesRegex(ValueError, "External target is absent"):
                        HuggingFaceCheckpointer(loader=loader, weights_mapping=[]).load(
                            directory, {"model": model}, external_target_fqns=frozenset({"missing"}))
