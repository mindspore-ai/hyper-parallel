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
"""Transformers-facing fakes for ``tests/ut/auto_models/_transformers``.

``FakeHFConfig`` and ``FakeAutoModel`` isolate the build pipeline from the
Transformers package, the HF hub and the network; construction arguments are
recorded so tests can assert exactly what the builder asked for.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn


class FakeHFConfig(SimpleNamespace):
    """A stand-in PretrainedConfig with the fields the builders read."""

    def __init__(self, **overrides):
        values = {
            "model_type": "fake",
            "architectures": ["FakeAutoModel"],
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "tie_word_embeddings": False,
            "torch_dtype": torch.float32,
        }
        values.update(overrides)
        super().__init__(**values)

    def to_dict(self):
        return dict(vars(self))


class FakeAutoModel(nn.Module):
    """A stand-in HF model recording how it was constructed."""

    from_config_calls = []

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.embed_tokens.weight

    @classmethod
    def from_config(cls, config, **kwargs):
        """Record construction and return an instance (no hub access)."""
        cls.from_config_calls.append({"config": config, "kwargs": kwargs})
        return cls(config)

    @classmethod
    def reset_calls(cls):
        cls.from_config_calls = []

    def forward(self, input_ids):
        return self.lm_head(self.embed_tokens(input_ids))


@pytest.fixture
def fake_hf_config():
    """A fresh default FakeHFConfig."""
    return FakeHFConfig()


@pytest.fixture
def fake_auto_model_cls():
    """The FakeAutoModel class with a clean construction record."""
    FakeAutoModel.reset_calls()
    yield FakeAutoModel
    FakeAutoModel.reset_calls()
