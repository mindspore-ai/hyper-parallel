# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.14: head_count — TP 本地头数改写（D-17）。

覆盖点：
- 检测：q/k/v 投影在 TP 维 colwise Shard(0) → head-sharded；mlp、无 TP 轴、
  rowwise/replicate 均不命中；MLA 的 q_b_proj 命中；
- 改写：num_heads/num_key_value_heads 整除改写，num_key_value_groups
  （比值不变量）与 head_dim（不切维）不动，config 不动；
- 属性清单：transformers 全库调研的命名变体（n_heads/num_kv_heads 等）；
- 幂等：重复调用不二次除法；非整除只告警不改写；tp_size<=1 直接跳过。
"""

import logging

import torch.nn as nn

from hyper_models.components.distributed.head_count import (
    _is_head_sharded,
    update_module_head_counts,
)
from hyper_models.components.distributed.sharding_config import (
    CP,
    TP,
    ModuleShardingSpec,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.components.distributed.conftest import TinyConfig, TinyLlamaAttention


def _spec(**params):
    return ModuleShardingSpec(params=params)


class TestIsHeadSharded:
    def test_qkv_colwise_detected(self):
        spec = _spec(**{
            "q_proj.weight": {TP: Shard(0)},
            "k_proj.weight": {TP: Shard(0)},
            "v_proj.weight": {TP: Shard(0)},
            "o_proj.weight": {TP: Shard(1)},
        })
        assert _is_head_sharded(spec, ("tp",)) is True

    def test_mla_q_b_proj_detected(self):
        """D-14 MLA：q_b_proj 上投影按头维 colwise → 命中。"""
        spec = _spec(**{
            "q_a_proj.weight": {TP: Replicate()},
            "q_b_proj.weight": {TP: Shard(0)},
            "kv_b_proj.weight": {TP: Shard(0)},
        })
        assert _is_head_sharded(spec, ("tp",)) is True

    def test_mlp_not_detected(self):
        spec = _spec(**{
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        })
        assert _is_head_sharded(spec, ("tp",)) is False

    def test_no_tp_axis_not_detected(self):
        spec = _spec(**{"q_proj.weight": {TP: Shard(0), CP: Replicate()}})
        assert _is_head_sharded(spec, ("cp",)) is False

    def test_rowwise_only_not_detected(self):
        """q/k/v 均为 Replicate（如 fused QKV rowwise 方案）→ 头数未切。"""
        spec = _spec(**{
            "q_proj.weight": {TP: Replicate()},
            "o_proj.weight": {TP: Shard(1)},
        })
        assert _is_head_sharded(spec, ("tp",)) is False


class _NamedAttrAttention(nn.Module):
    """transformers 命名变体覆盖：n_heads / num_kv_heads（falcon 风格）。"""

    def __init__(self):
        super().__init__()
        self.n_heads = 8
        self.num_kv_heads = 2
        self.num_key_value_groups = 4
        self.head_dim = 16
        self.config = TinyConfig(num_attention_heads=8)


class TestUpdateModuleHeadCounts:
    def test_divide_and_preserve_invariants(self):
        attn = TinyLlamaAttention(TinyConfig())   # num_heads=4, head_dim=4
        attn.num_key_value_heads = 4
        n = update_module_head_counts(attn, 2, "self_attn")
        assert n == 2
        assert attn.num_heads == 2
        assert attn.num_key_value_heads == 2
        assert attn.head_dim == 4                     # 头维不切
        assert attn.config.num_attention_heads == 4   # config 不改写
        assert attn._hp_full_head_counts == {
            "num_heads": 4, "num_key_value_heads": 4}

    def test_name_variants(self):
        attn = _NamedAttrAttention()
        n = update_module_head_counts(attn, 4, "self_attn")
        assert n == 1                     # num_kv_heads=2 对 tp=4 不整除 → 仅告警
        assert attn.n_heads == 2
        assert attn.num_kv_heads == 2     # 不整除 → 保持原值

    def test_num_key_value_groups_untouched(self):
        attn = _NamedAttrAttention()
        update_module_head_counts(attn, 2, "self_attn")
        assert attn.n_heads == 4
        assert attn.num_kv_heads == 1
        assert attn.num_key_value_groups == 4   # 比值不变量，绝不动

    def test_idempotent(self):
        attn = TinyLlamaAttention(TinyConfig())
        assert update_module_head_counts(attn, 2) == 1
        assert update_module_head_counts(attn, 2) == 0   # 不二次除法
        assert attn.num_heads == 2

    def test_non_divisible_warns_and_keeps(self, caplog):
        attn = TinyLlamaAttention(TinyConfig())   # num_heads=4
        with caplog.at_level(logging.WARNING):
            n = update_module_head_counts(attn, 3, "self_attn")
        assert n == 0
        assert attn.num_heads == 4
        assert "not divisible" in caplog.text
        # 重复调用不重复告警
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            update_module_head_counts(attn, 3, "self_attn")
        assert "not divisible" not in caplog.text

    def test_tp1_noop(self):
        attn = TinyLlamaAttention(TinyConfig())
        assert update_module_head_counts(attn, 1) == 0
        assert attn.num_heads == 4
        assert not hasattr(attn, "_hp_full_head_counts")
