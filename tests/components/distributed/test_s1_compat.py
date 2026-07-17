# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.11: validate_model_compatibility。"""

import pytest

from hyper_models.components.distributed.sharding_planner import (
    validate_model_compatibility,
)
from tests.components.distributed.conftest import TinyConfig, TinyLlamaForCausalLM


def _model(**kw):
    return TinyLlamaForCausalLM(TinyConfig(**kw))


class TestCompat:
    def test_heads_not_divisible(self):
        with pytest.raises(ValueError, match="num_attention_heads"):
            validate_model_compatibility(
                _model(num_attention_heads=3), tp_size=2)

    def test_kv_heads_not_divisible(self):
        with pytest.raises(ValueError, match="num_key_value_heads"):
            validate_model_compatibility(
                _model(num_attention_heads=4, num_key_value_heads=3), tp_size=2)

    def test_seq_len_not_divisible_2cp(self):
        with pytest.raises(ValueError, match=r"2\*cp"):
            validate_model_compatibility(_model(), cp_size=2, seq_len=10)

    def test_seq_len_ok(self):
        validate_model_compatibility(_model(), cp_size=2, seq_len=8)

    def test_num_experts_not_divisible(self):
        with pytest.raises(ValueError, match="num_experts"):
            validate_model_compatibility(_model(num_experts=3), ep_size=2)

    def test_ep_requires_moe(self):
        with pytest.raises(ValueError, match="MoE"):
            validate_model_compatibility(_model(num_experts=0), ep_size=2)

    def test_moe_inter_dim_not_divisible_tp(self):
        with pytest.raises(ValueError, match="moe_intermediate_size"):
            validate_model_compatibility(
                _model(num_experts=4, moe_intermediate_size=7), tp_size=2, ep_size=2)

    def test_all_pass(self):
        validate_model_compatibility(
            _model(num_experts=4, moe_intermediate_size=8),
            tp_size=2, cp_size=2, ep_size=2, seq_len=16)
