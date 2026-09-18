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
"""CPU regression tests for streaming replacement checkpoint loading."""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from hyper_parallel.components.checkpoint.weight_conversion import WeightRenaming
from hyper_parallel.models._transformers.checkpoint_loader import (
    CheckpointManager,
    LoadReport,
    _CheckpointIndex,
    _finalize_model_loading,
)


class TestReplacementCheckpointLoading(unittest.TestCase):
    """Exercise actual tensor conversion and copying without distributed setup."""

    def _load(self, *, shared=False, extra=False, missing=False, strict=True, replacement=True):
        model = nn.Module()
        model.register_parameter("target", nn.Parameter(torch.zeros(2)))
        targets = {"target": model.target}
        if missing:
            model.register_parameter("missing", nn.Parameter(torch.zeros(2)))
            targets["missing"] = model.missing
        if shared:
            model.register_parameter("alias", model.target)
            targets["alias"] = model.alias
        transform = WeightRenaming(source_patterns="source", target_patterns="target")
        tensors = {"source": torch.tensor([2.0, 3.0])}
        if extra:
            tensors["extra"] = torch.ones(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(tensors, str(path))
            if replacement:
                report = CheckpointManager(model)._load_with_replacement_conversions(
                    _CheckpointIndex({key: path for key in tensors}), targets,
                    [transform], [transform], {"source": (2,)}, directory, strict,
                )
            else:
                report = CheckpointManager(model).load_checkpoint(directory, weights_mapping=[transform], strict=strict)
        return model, report

    def test_renaming_and_shared_target(self):
        """Feature: replacement loading.

        Description: Load a renamed tensor with and without a shared target alias.
        Expectation: Preserve target identity, load values and report every alias.
        """
        for shared in (False, True):
            with self.subTest(shared=shared):
                model, report = self._load(shared=shared)
                torch.testing.assert_close(model.target, torch.tensor([2.0, 3.0]))
                self.assertEqual(report.loaded_keys, ("alias", "target") if shared else ("target",))
                self.assertEqual(report.missing_keys, ())
                self.assertEqual(report.unexpected_keys, ())
                self.assertEqual(len(model._hp_used_replacement_weight_conversions), 1)
                if shared:
                    self.assertIs(model.alias, model.target)

    def test_unexpected_source_reporting(self):
        """Feature: replacement load reporting.

        Description: Load a checkpoint with an unknown source in non-strict mode.
        Expectation: Load valid tensors and report the unknown source.
        """
        _, report = self._load(extra=True, strict=False)
        self.assertEqual(report.unexpected_keys, ("extra",))

    def test_strict_missing_target(self):
        """Feature: strict replacement loading.

        Description: Leave an owned target unloaded with strict validation enabled.
        Expectation: Reject the missing model tensor.
        """
        with self.assertRaisesRegex(RuntimeError, "owned model tensors"):
            self._load(missing=True)

    def test_standard_loading(self):
        """Feature: ordinary checkpoint loading.

        Description: Load a renamed shared parameter with an unexpected source.
        Expectation: Preserve aliases and report unused checkpoint keys.
        """
        model, report = self._load(shared=True, extra=True, replacement=False)
        torch.testing.assert_close(model.target, torch.tensor([2.0, 3.0]))
        self.assertIs(model.alias, model.target)
        self.assertEqual(report.loaded_keys, ("alias", "target"))
        self.assertEqual(report.unexpected_keys, ("extra",))
        self.assertEqual(len(model._weight_conversions), 1)

    def test_finalize_loaded_parameter(self):
        """Feature: deferred loading finalization.

        Description: Finalize a fully loaded ordinary parameter.
        Expectation: Keep the parameter object and values unchanged.
        """
        model = nn.Module()
        model.register_parameter("target", nn.Parameter(torch.tensor([2.0, 3.0])))
        target = model.target
        report = LoadReport(("target",), (), ())
        self.assertEqual(_finalize_model_loading(model, report, strict=True), report)
        self.assertIs(model.target, target)
        torch.testing.assert_close(model.target, torch.tensor([2.0, 3.0]))

    def test_finalize_strict_missing_parameter(self):
        """Feature: strict deferred finalization.

        Description: Finalize a report containing a missing owned parameter.
        Expectation: Reject it before initialization.
        """
        model = nn.Module()
        model.register_parameter("target", nn.Parameter(torch.zeros(2)))
        with self.assertRaisesRegex(RuntimeError, "after finalization"):
            _finalize_model_loading(model, LoadReport((), ("target",), ()), strict=True)
