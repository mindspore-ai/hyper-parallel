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
"""Native Qwen rotary buffer recovery after meta initialization."""

import unittest

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextConfig, Qwen3_5TextRotaryEmbedding, Qwen3_5VisionRotaryEmbedding,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTextConfig, Qwen3_5MoeTextRotaryEmbedding, Qwen3_5MoeVisionRotaryEmbedding,
)

from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state
from hyper_parallel.models.registry import get_model_adapter
from tests.common.mark_utils import arg_mark


class TestRotaryMaterialization(unittest.TestCase):
    """Exercise real HF buffers without constructing model weights."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_native_rotary_buffers_survive_empty_materialization(self):
        """All four adapters restore native values without touching loaded parameters."""
        cases = (
            ("qwen3_5", Qwen3_5TextConfig, Qwen3_5TextRotaryEmbedding, Qwen3_5VisionRotaryEmbedding),
            ("qwen3_5_text", Qwen3_5TextConfig, Qwen3_5TextRotaryEmbedding, Qwen3_5VisionRotaryEmbedding),
            ("qwen3_5_moe", Qwen3_5MoeTextConfig, Qwen3_5MoeTextRotaryEmbedding, Qwen3_5MoeVisionRotaryEmbedding),
            ("qwen3_5_moe_text", Qwen3_5MoeTextConfig, Qwen3_5MoeTextRotaryEmbedding, Qwen3_5MoeVisionRotaryEmbedding),
        )
        for model_type, config_cls, text_cls, vision_cls in cases:
            with self.subTest(model_type=model_type):
                config = config_cls()
                reference = text_cls(config, device="cpu")
                vision_reference = vision_cls(32)
                with torch.device("meta"):
                    model = torch.nn.Module()
                    model.text = text_cls(config)
                    model.vision = vision_cls(32)
                    model.weight = torch.nn.Parameter(torch.empty(2))
                get_model_adapter(model_type=model_type).materialization(model)
                model.to_empty(device="cpu")
                with torch.no_grad():
                    model.weight.fill_(7)
                    for buffer in model.buffers():
                        buffer.fill_(float("nan"))
                rebuild_materialized_state(model, MaterializationContext("checkpoint_load", torch.device("cpu")))
                for name, expected in reference.named_buffers():
                    self.assertTrue(torch.equal(getattr(model.text, name), expected), name)
                self.assertTrue(torch.equal(model.vision.inv_freq, vision_reference.inv_freq))
                self.assertTrue(torch.equal(model.weight, torch.full((2,), 7.0)))
                self.assertEqual(set(model.state_dict()), {"weight"})
                positions = torch.arange(8).reshape(1, 1, 8).expand(3, 1, 8)
                hidden = torch.zeros(1, 8, 16)
                for actual, expected in zip(model.text(hidden, positions), reference(hidden, positions)):
                    self.assertTrue(torch.equal(actual, expected))
