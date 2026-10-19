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
"""Qwen text chunk loss protocol and local-shard numerical checks."""

import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import load_balancing_loss_func

from hyper_parallel.components.losses import ChunkedCausalLMLoss
from hyper_parallel.models.qwen3_5.adapter.chunk_loss import bind_chunk_loss


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(13, 5)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_5_text")
        self.model = _Decoder()
        self.lm_head = nn.Linear(5, 13, bias=False)

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids).last_hidden_state)


class TestQwenChunkLoss(unittest.TestCase):
    """Exercise binding without a real accelerator or Transformers decoder."""

    model_type = "qwen3_5_text"

    def _make_model(self):
        model = _Model()
        model.config.model_type = self.model_type
        model.config.output_router_logits = False
        return model

    def test_loss_and_gradients_match_full_logits(self):
        """Uneven chunks preserve masks, target alignment and parameter gradients."""
        torch.manual_seed(42)
        model = self._make_model()
        reference = copy.deepcopy(model)
        ids = torch.randint(0, 13, (2, 7))
        targets = torch.randint(0, 13, (2, 7))
        targets[0, 1] = -100
        mask = torch.ones_like(targets)
        mask[1, 4:] = 0
        objective = ChunkedCausalLMLoss(chunk_size=3)
        objective.bind_model(model)
        inputs = objective.prepare_model_inputs({"input_ids": ids}, {"shift_labels": targets, "loss_mask": mask})
        output = model(**inputs)
        actual = objective(model_output=output, labels=None)
        (actual * 0.37).backward()
        logits = reference(ids)
        expected = F.cross_entropy(logits.flatten(0, 1).float(), targets.masked_fill(mask == 0, -100).flatten())
        (expected * 0.37).backward()
        self.assertIsNone(output.logits)
        torch.testing.assert_close(actual, expected)
        for param, ref_param in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(param.grad, ref_param.grad, atol=1e-7, rtol=1e-5)

    def test_inference_and_repeat_binding_unchanged(self):
        """The opt-in adapter leaves ordinary forwards untouched."""
        model = self._make_model()
        ids = torch.tensor([[1, 2, 3]])
        expected = model(ids)
        bind_chunk_loss(model)
        bind_chunk_loss(model)
        torch.testing.assert_close(model(ids), expected)

    def test_cache_rejected(self):
        """Cached decoding is not silently accepted by the training path."""
        model = self._make_model()
        objective = ChunkedCausalLMLoss(chunk_size=2)
        objective.bind_model(model)
        inputs = objective.prepare_model_inputs(
            {"input_ids": torch.tensor([[1, 2]]), "use_cache": True},
            {"shift_labels": torch.tensor([[2, 3]])},
        )
        with self.assertRaisesRegex(ValueError, "KV cache"):
            model(**inputs)


class _RouterDecoder(_Decoder):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(5, 3, bias=False)

    def forward(self, input_ids, output_router_logits=False, **kwargs):
        hidden = self.embed(input_ids)
        router_logits = (self.router(hidden).flatten(0, 1),) if output_router_logits else None
        return SimpleNamespace(last_hidden_state=hidden, router_logits=router_logits)


class TestQwenMoeChunkLoss(TestQwenChunkLoss):
    """Run the same numerical and inference contract through MoE registration."""

    model_type = "qwen3_5_moe_text"

    def test_router_auxiliary_loss_and_gradients(self):
        """Preserve native router loss, including explicit flag overrides."""
        for enabled in (None, True, False):
            with self.subTest(output_router_logits=enabled):
                torch.manual_seed(42)
                model = self._make_model()
                model.model = _RouterDecoder()
                model.config.output_router_logits = True
                model.num_experts = 3
                model.num_experts_per_tok = 2
                model.router_aux_loss_coef = 0.01
                reference = copy.deepcopy(model)
                ids = torch.tensor([[1, 2, 3, 4, 5]])
                targets = torch.tensor([[2, 3, -100, 5, 6]])
                attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
                objective = ChunkedCausalLMLoss(chunk_size=2)
                objective.bind_model(model)
                inputs = objective.prepare_model_inputs(
                    {"input_ids": ids, "attention_mask": attention_mask, "output_router_logits": enabled},
                    {"shift_labels": targets},
                )
                output = model(**inputs)
                actual = objective(model_output=output, labels=None)
                decoder = reference.model(ids, output_router_logits=enabled is not False)
                expected = F.cross_entropy(reference.lm_head(decoder.last_hidden_state).flatten(0, 1),
                                           targets.flatten())
                if enabled is not False:
                    aux = load_balancing_loss_func(decoder.router_logits, 3, 2, attention_mask)
                    expected = expected + 0.01 * aux
                    torch.testing.assert_close(output.aux_loss, aux)
                else:
                    self.assertIsNone(output.aux_loss)
                torch.testing.assert_close(actual, expected)
                (actual * 0.37).backward()
                (expected * 0.37).backward()
                for param, ref_param in zip(model.parameters(), reference.parameters()):
                    if ref_param.grad is None:
                        self.assertIsNone(param.grad)
                    else:
                        torch.testing.assert_close(param.grad, ref_param.grad, atol=1e-7, rtol=1e-5)
