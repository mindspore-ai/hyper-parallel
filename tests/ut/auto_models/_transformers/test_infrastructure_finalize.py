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
"""Tests for finalizing deferred pretrained loading."""

import unittest

import torch
from torch import nn

from hyper_parallel.auto_models._transformers.checkpoint_loader import LoadReport
from hyper_parallel.auto_models._transformers.infrastructure import _finalize_model_loading, _move_model_to_device


class _MissingState(nn.Module):
    """Owns non-persistent state that is absent from checkpoint files."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("rope", torch.empty(3), persistent=False)


class _PretrainedLikeModel(nn.Module):
    """Small module whose initialization follows Hugging Face module flags."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.missing = _MissingState()
        for module in self.modules():
            module._is_hf_initialized = False  # pylint: disable=protected-access

    def initialize_weights(self) -> None:
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module) -> None:
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        if isinstance(module, _MissingState):
            module.rope.fill_(7.0)
        module._is_hf_initialized = True  # pylint: disable=protected-access


class TestFinalizeModelLoading(unittest.TestCase):
    """Coverage for pretrained tensors surviving missing-state initialization."""

    def test_materialization_preserves_nonpersistent_buffers(self):
        model = nn.Linear(2, 2, device="meta")
        model.register_buffer("rope", torch.full((4,), 5.0, dtype=torch.float32), persistent=False)

        _move_model_to_device(model, is_meta_device=True, device=torch.device("cpu"))

        self.assertFalse(model.weight.is_meta)
        self.assertEqual(model.rope.dtype, torch.float32)
        self.assertTrue(torch.equal(model.rope, torch.full((4,), 5.0)))

    def test_missing_nonpersistent_state_does_not_reinitialize_loaded_modules(self):
        model = _PretrainedLikeModel()
        with torch.no_grad():
            model.proj.weight.fill_(3.0)
            model.proj.bias.fill_(4.0)

        _finalize_model_loading(
            model,
            LoadReport(
                loaded_keys=("proj.weight", "proj.bias"),
                missing_keys=(),
                unexpected_keys=(),
            ),
            strict=True,
        )

        self.assertTrue(torch.equal(model.proj.weight, torch.full((2, 2), 3.0)))
        self.assertTrue(torch.equal(model.proj.bias, torch.full((2,), 4.0)))
        self.assertTrue(torch.equal(model.missing.rope, torch.full((3,), 7.0)))


if __name__ == "__main__":
    unittest.main()
