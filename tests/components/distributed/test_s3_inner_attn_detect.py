# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S3.2: _find_inner_attention + HF/NeMo 风格判定（单进程 mock）。"""

import torch.nn as nn

from hyper_parallel.components.distributed.sharding_applier import (
    _find_inner_attention,
    _is_flex_attention,
    _is_hf_style_attention,
    _is_sdpa_attention,
)


class _Cfg:
    _attn_implementation = "sdpa"


class HFSdpaAttention(nn.Module):
    """HF 风格：持有 q/k/v_proj，forward(hidden_states)。"""

    def __init__(self):
        super().__init__()
        self.config = _Cfg()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)

    def forward(self, hidden_states):
        return hidden_states


class NeMoAttention(nn.Module):
    """NeMo 风格：inner_attention 子模块 forward(q,k,v)。"""

    class Inner(nn.Module):
        def forward(self, q, k, v):
            return q

    def __init__(self):
        super().__init__()
        self.inner_attention = self.Inner()

    def forward(self, hidden_states):
        return hidden_states


class FlexHFAattention(HFSdpaAttention):
    class _FlexCfg:
        _attn_implementation = "flex_attention"

    def __init__(self):
        super().__init__()
        self.config = self._FlexCfg()


class TestFindInnerAttention:
    def test_explicit_inner_attention_attr(self):
        m = NeMoAttention()
        assert _find_inner_attention(m) is m.inner_attention

    def test_hf_classname_self(self):
        m = HFSdpaAttention()
        assert _find_inner_attention(m) is m

    def test_structural_qkv_fallback(self):
        class Plain(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.k_proj = nn.Linear(8, 8)
                self.v_proj = nn.Linear(8, 8)

            def forward(self, hidden_states):
                return hidden_states
        m = Plain()
        assert _find_inner_attention(m) is m

    def test_not_found_returns_none(self):
        class Bare(nn.Module):
            def forward(self, x):
                return x
        assert _find_inner_attention(Bare()) is None


class TestStyleDetection:
    def test_hf_style_by_signature(self):
        assert _is_hf_style_attention(HFSdpaAttention()) is True

    def test_nemo_style_not_hf(self):
        inner = NeMoAttention().inner_attention
        assert _is_hf_style_attention(inner) is False

    def test_sdpa_detection(self):
        assert _is_sdpa_attention(HFSdpaAttention()) is True
        assert _is_flex_attention(HFSdpaAttention()) is False

    def test_flex_detection(self):
        assert _is_flex_attention(FlexHFAattention()) is True
        assert _is_sdpa_attention(FlexHFAattention()) is False
