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
"""Compare shared mHC/Engram replacements with independent V4.1 reference forwards."""

import unittest
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from hyper_parallel.components.modules import Engram, PipelinedMhc
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41Engram,
    DeepseekV41PipelinedHyperConnection,
)


class TestModuleFamilyParity(unittest.TestCase):
    """Check all input and trainable-parameter gradients after actual source adoption."""

    device = torch.device("cpu")

    def _compare(self, reference, candidate, shape, dtype, *extra) -> None:
        """Use independent weights and the same objective for both implementations."""
        source_params = dict(reference.named_parameters())
        target_params = dict(candidate.named_parameters())
        self.assertEqual(set(source_params), set(target_params))
        for name, parameter in source_params.items():
            self.assertIsNot(parameter, target_params[name])
            torch.testing.assert_close(parameter, target_params[name], rtol=0, atol=0)
        values = torch.linspace(-0.7, 1.3, torch.Size(shape).numel(), device=self.device).reshape(shape).to(dtype)
        x, y = values.clone().requires_grad_(), values.clone().requires_grad_()
        expected, actual = reference(x, *extra), candidate(y, *extra)
        expected = expected if isinstance(expected, tuple) else (expected,)
        actual = actual if isinstance(actual, tuple) else (actual,)
        for target, output in zip(expected, actual):
            torch.testing.assert_close(output, target, rtol=1e-4, atol=1e-5)
        sum(value.float().square().mean() for value in expected).backward()
        sum(value.float().square().mean() for value in actual).backward()
        torch.testing.assert_close(y.grad, x.grad, rtol=1e-4, atol=1e-5)
        for name, parameter in source_params.items():
            self.assertIsNotNone(parameter.grad, name)
            self.assertIsNotNone(target_params[name].grad, name)
            torch.testing.assert_close(target_params[name].grad, parameter.grad, rtol=1e-4, atol=1e-5)

    def test_mhc_source_forward_and_all_gradients(self) -> None:
        """Reference V4.1 coefficient math matches the family in FP32 and BF16 inputs."""
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(42)
                source = nn.Module()
                source.input_norm = nn.RMSNorm(64, eps=1e-6, elementwise_affine=False)
                source.fn = nn.Parameter(torch.randn(24, 64) * 0.02)
                source.base = nn.Parameter(torch.randn(24) * 0.02)
                source.scale = nn.Parameter(torch.ones(3))
                source.hc_mult, source.hc_sinkhorn_iters, source.hc_eps = 4, 20, 1e-6
                reference = DeepseekV41PipelinedHyperConnection(source).to(self.device)
                candidate = PipelinedMhc(module=deepcopy(reference)).to(self.device)
                self._compare(reference, candidate, (1, 8, 4, 16), dtype)

    def test_engram_source_forward_and_all_gradients(self) -> None:
        """Native hashing, masking, fusion and gradients match the family after adoption."""
        assets = {
            "layer_ids": [0], "max_ngram_size": 2, "num_heads": 2,
            "token_map": list(range(6)), "pad_token_id": 0,
            "primes": [[[5, 7]]], "multipliers": [[1, 3]],
            "num_embeddings": [12], "padded_num_embeddings": [16], "head_dim": 2,
        }
        config = SimpleNamespace(engram_storage_backend="device", hidden_size=4, hc_mult=2, rms_norm_eps=1e-6)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(43)
                reference = DeepseekV41Engram(config, 0, assets).to(device=self.device, dtype=dtype)
                candidate = Engram(module=deepcopy(reference)).to(device=self.device, dtype=dtype)
                ids = torch.tensor([[1, 2, 3, 4, 0, 1, 4, 3]], device=self.device)
                starts = torch.tensor([[1, 0, 0, 1, 0, 0, 0, 0]], device=self.device)
                mask = torch.tensor([[1, 1, 1, 1, 0, 1, 1, 1]], device=self.device)
                self._compare(reference, candidate, (1, 8, 2, 4), dtype, ids, starts, mask)
