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
"""Shared generation contracts, independent of a Trainer or accelerator."""

from types import SimpleNamespace
import unittest

import torch

from hyper_parallel.infer import GenerationConfig, GenerationOptions, GenerationResult, GenerationRuntime, generate
from hyper_parallel.infer.config import resolve_generation_options
from hyper_parallel.infer.sampler import top_k_sample, top_p_sample


class IncrementModel(torch.nn.Module):
    """Emit input+1 with no cache; each row can reach EOS at a different step."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.generation_config = SimpleNamespace(eos_token_id=6, pad_token_id=6)
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        """Return deterministic logits for each input token's successor."""
        del kwargs
        self.calls += 1
        logits = torch.full((*input_ids.shape, 10), -20.0, device=input_ids.device)
        logits.scatter_(-1, ((input_ids + 1) % 10).unsqueeze(-1), 20.0)
        return SimpleNamespace(logits=logits)


class TestGeneration(unittest.TestCase):
    """Metadata comes from the loop and legacy callers keep a Tensor result."""

    def test_mixed_eos_lengths_and_left_padding(self):
        model = IncrementModel()
        ids = torch.tensor([[6, 4], [1, 2]])
        result = generate(model, ids, attention_mask=torch.tensor([[0, 1], [1, 1]]),
                          options=GenerationOptions(max_new_tokens=4), return_dict=True)
        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.sequences.tolist(), [[4, 5, 6, 6, 6, 6], [1, 2, 3, 4, 5, 6]])
        self.assertEqual(result.prompt_lengths.tolist(), [1, 2])
        self.assertEqual(result.generated_lengths.tolist(), [2, 4])
        self.assertEqual(result.finish_reasons, ["eos", "eos"])
        self.assertTrue(model.training)
        self.assertIsNone(model.weight.grad)

    def test_multiple_eos_ids(self):
        result = generate(IncrementModel(), torch.tensor([[3], [5]]),
                          options=GenerationOptions(max_new_tokens=5, eos_token_id=[4, 6]), return_dict=True)
        self.assertEqual(result.generated_lengths.tolist(), [1, 1])
        self.assertEqual(result.finish_reasons, ["eos", "eos"])

    def test_zero_tokens_and_legacy_return(self):
        model = IncrementModel()
        result = generate(model, torch.tensor([[6, 3]]), attention_mask=torch.tensor([[0, 1]]),
                          options=GenerationOptions(max_new_tokens=0), return_dict=True)
        self.assertEqual(result.sequences.tolist(), [[3]])
        self.assertEqual(result.generated_lengths.tolist(), [0])
        self.assertEqual(result.finish_reasons, ["length"])
        self.assertEqual(model.calls, 0)
        result = generate(model, torch.tensor([[3]]), GenerationConfig(max_new_tokens=1))
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.tolist(), [[3, 4]])

    def test_stop_reason_is_distinct_from_length_and_eos(self):
        config = GenerationConfig(max_new_tokens=5, eos_token_id=6, stopping_criteria=[lambda *_: True])
        result = generate(IncrementModel(), torch.tensor([[2], [5]]), config, return_dict=True)
        self.assertEqual(result.finish_reasons, ["stop", "eos"])
        length = generate(IncrementModel(), torch.tensor([[1]]),
                          options=GenerationOptions(max_new_tokens=1), return_dict=True)
        self.assertEqual(length.finish_reasons, ["length"])

    def test_sampling_rng_and_top_k_legacy_helper(self):
        logits = torch.zeros(2, 10)
        one = top_p_sample(logits, 0.8, generator=torch.Generator().manual_seed(7))
        two = top_p_sample(logits, 0.8, generator=torch.Generator().manual_seed(7))
        torch.testing.assert_close(one, two)
        self.assertEqual(top_k_sample(logits, 3).shape, (2, 1))
        options = GenerationOptions(max_new_tokens=3, do_sample=True, top_k=0, top_p=0.9)
        results = [generate(IncrementModel(), torch.tensor([[1]]), options=options,
                            runtime=GenerationRuntime(rng=torch.Generator().manual_seed(4))) for _ in range(2)]
        torch.testing.assert_close(*results)

    def test_invalid_masks_and_mixed_apis(self):
        for mask in ([[1, 0]], [[0, 0]], [[1, 2]]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                generate(IncrementModel(), torch.tensor([[1, 2]]), attention_mask=torch.tensor(mask))
        with self.assertRaises(ValueError):
            generate(IncrementModel(), torch.tensor([[1]]), GenerationConfig(), options=GenerationOptions())

    def test_option_validation_and_token_precedence(self):
        for values in ({"max_new_tokens": -1}, {"max_new_tokens": True}, {"temperature": float("nan")},
                       {"top_p": 2}, {"eos_token_id": []}, {"pad_token_id": -1}, {"use_cache": "yes"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                GenerationOptions(**values)
        tokenizer = SimpleNamespace(eos_token_id=8, pad_token_id=9)
        options = GenerationOptions(eos_token_id=[4, 5])
        resolved = resolve_generation_options(options, IncrementModel(), tokenizer)
        self.assertEqual(resolved.eos_token_id, [4, 5])
        self.assertEqual(resolved.pad_token_id, 6)
        self.assertIsNone(options.pad_token_id)

    def test_repetition_penalty_ignores_left_padding(self):
        options = GenerationOptions(max_new_tokens=2, repetition_penalty=2.0)
        single = generate(IncrementModel(), torch.tensor([[3]]), options=options)
        padded = generate(IncrementModel(), torch.tensor([[6, 3]]), options=options,
                          attention_mask=torch.tensor([[0, 1]]))
        torch.testing.assert_close(single, padded)
