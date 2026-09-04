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
"""Shared fakes for ``tests/ut/auto_models`` (Gate-1).

``FakeNpuBackend`` injects a recording ``torch_npu`` stub so NPU-only modules
import and run on CPU; ``TorchReferenceKernels`` provides pure-Torch reference
implementations of the fused ops (RMSNorm / RoPE / SDPA / grouped GEMM /
token permute) so module orchestration and gradients can be checked without
real backend kernels.
"""

import sys
import types
import unittest.mock

import torch
import torch.nn.functional as F


class FakeNpuBackend:
    """Recording ``torch_npu`` stub injected into ``sys.modules``.

    Any attribute access on the stub returns a recording callable; recorded
    calls are ``[(api_name, args, kwargs), ...]``. Use as a context manager::

        with FakeNpuBackend() as npu:
            module_under_test = importlib.import_module(...)
        assert [call[0] for call in npu.calls] == [...]
    """

    def __init__(self):
        self.calls = []
        self._patcher = None

    def _record(self, name):
        def _api(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None

        return _api

    def __enter__(self):
        stub = types.ModuleType("torch_npu")
        stub.__getattr__ = self._record  # module-level __getattr__ (PEP 562)
        self._patcher = unittest.mock.patch.dict(sys.modules, {"torch_npu": stub})
        self._patcher.start()
        return self

    def __exit__(self, *exc):
        self._patcher.stop()
        return False


class TorchReferenceKernels:
    """Pure-Torch references for fused ops; the oracle for module tests."""

    @staticmethod
    def rms_norm(x, weight, eps=1e-6):
        """Reference RMSNorm in float32 accumulation."""
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normed = x.float() * torch.rsqrt(variance + eps)
        return (normed * weight.float()).to(x.dtype)

    @staticmethod
    def apply_rotary(x, cos, sin):
        """Rotate-half RoPE with broadcastable cos/sin."""
        half = x.shape[-1] // 2
        first, second = x[..., :half], x[..., half:]
        rotated = torch.cat([-second, first], dim=-1)
        return x * cos + rotated * sin

    @staticmethod
    def sdpa(query, key, value, attention_mask=None, dropout=0.0, scale=None):
        """Reference scaled dot-product attention (SDPA math path)."""
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scale,
        )

    @staticmethod
    def grouped_gemm(x, expert_weights, expert_ids):
        """Reference grouped GEMM: one matmul per expert, gathered by id."""
        outputs = torch.zeros_like(x)
        for expert_index, weight in enumerate(expert_weights):
            mask = expert_ids == expert_index
            if mask.any():
                outputs[mask] = x[mask] @ weight
        return outputs

    @staticmethod
    def token_permute(x, expert_ids, num_experts):
        """Reference token permute: sort tokens by expert id."""
        order = torch.argsort(expert_ids, stable=True)
        counts = torch.bincount(expert_ids, minlength=num_experts)
        return x[order], counts

    @staticmethod
    def token_unpermute(permuted, order_inverse):
        """Reference token unpermute: inverse of ``token_permute`` ordering."""
        return permuted[order_inverse]
