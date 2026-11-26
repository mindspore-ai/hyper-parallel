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
"""Preserve model-owned validation while sharing the Host Engram lifecycle."""

import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.components.modules.engram.host_engram import HostEngramModule
from hyper_parallel.components.modules.engram.host_state import HostEngramState
from hyper_parallel.models.deepseek_v41.adapter.engram.host_state import (
    DeepseekV41HostState,
    build_deepseek_v41_external_state,
)
from hyper_parallel.models.deepseek_v41.deepseek_v41_spec import MODEL_SPEC
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Engram
from hyper_parallel.models.external_state import ExternalBuildContext


def _model() -> torch.nn.Module:
    """Construct the actual legacy Engram architecture with tiny hash assets."""
    config = SimpleNamespace(engram_storage_backend="device", hidden_size=4, hc_mult=2, rms_norm_eps=1e-6)
    assets = {
        "layer_ids": [0], "max_ngram_size": 2, "num_heads": 2,
        "token_map": list(range(6)), "pad_token_id": 0,
        "primes": [[[5, 7]]], "multipliers": [[1, 3]],
        "num_embeddings": [12], "padded_num_embeddings": [16], "head_dim": 2,
    }
    with torch.device("meta"):
        model = torch.nn.Module()
        model.config = config
        model.engram = DeepseekV41Engram(config, 0, assets)
    return model


class TestHostState(unittest.TestCase):
    """Exercise mixed old/new module detection through the registered provider."""

    def setUp(self) -> None:
        """Use the production meta build policy without process groups."""
        self.context = ExternalBuildContext(None, {}, torch.device("meta"), torch.float32, False)

    def test_legacy_device_module_rejected_in_host_model(self) -> None:
        """An unreplaced native device table cannot escape Host mode validation."""
        model = _model()
        model.config.engram_storage_backend = "host"
        model.host = HostEngramModule(module=model.engram)
        with self.assertRaisesRegex(ValueError, "still contains a device replacement"):
            MODEL_SPEC.external_state(model, self.context)

    def test_host_module_uses_shared_state_and_preserves_alias(self) -> None:
        """A fully replaced model builds the same state class used by recovery code."""
        model = _model()
        model.config.engram_storage_backend = "host"
        model.engram = HostEngramModule(module=model.engram)
        state = MODEL_SPEC.external_state(model, self.context)
        self.assertIs(DeepseekV41HostState, HostEngramState)
        self.assertIsInstance(state, DeepseekV41HostState)
        self.assertIs(state.tables["engram.embed"], model.engram.embed)

    def test_device_backend_keeps_native_module(self) -> None:
        """Device-only construction continues without external Host ownership."""
        self.assertIsNone(build_deepseek_v41_external_state(_model(), self.context))
