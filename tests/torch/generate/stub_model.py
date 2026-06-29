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
"""Stub causal language models for generate tests."""
import torch
from torch import nn


class CacheLengthLM(nn.Module):
    """Predicts the current total sequence length modulo vocab size."""

    def __init__(self, vocab_size: int = 32, eos_at_total_len: int | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_at_total_len = eos_at_total_len
        self.calls = []

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        **kwargs,
    ):
        """Run deterministic forward with an optional tuple KV cache."""
        del position_ids, attention_mask, kwargs
        past_len = 0
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[-2]
        batch_size, seq_len = input_ids.shape
        total_len = past_len + seq_len
        token_id = total_len % self.vocab_size
        if self.eos_at_total_len is not None and total_len >= self.eos_at_total_len:
            token_id = 2
        logits = input_ids.new_full(
            (batch_size, seq_len, self.vocab_size),
            -1000,
            dtype=torch.float32,
        )
        logits[:, -1, token_id] = 1000
        self.calls.append({
            "seq_len": seq_len,
            "past_len": past_len,
            "use_cache": use_cache,
        })
        past = None
        if use_cache:
            key = torch.zeros(batch_size, 1, total_len, 4)
            value = torch.zeros(batch_size, 1, total_len, 4)
            past = [(key, value)]
        return {"logits": logits, "past_key_values": past}


class NoCacheLengthLM(CacheLengthLM):
    """Same logits as CacheLengthLM but never returns past_key_values."""

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output["past_key_values"] = None
        return output
