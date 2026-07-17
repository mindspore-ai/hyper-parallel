# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.5: ShardingTemplate + TEMPLATES 7 模板字段完整性。"""

from hyper_models.components.distributed.sharding_config import (
    CP,
    TEMPLATES,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard

EXPECTED_TEMPLATES = {
    "attention", "mlp", "norm", "embed", "lm_head", "moe_gate", "moe_mlp",
}


def test_seven_templates_enumerated():
    assert set(TEMPLATES) == EXPECTED_TEMPLATES


def test_field_completeness_sp_and_nosp():
    """每模板 SP/non-SP I/O 字段全填。"""
    for name, t in TEMPLATES.items():
        for prefix in ("sp", "nosp"):
            assert getattr(t, f"{prefix}_in_src"), f"{name}.{prefix}_in_src"
            assert getattr(t, f"{prefix}_in_dst"), f"{name}.{prefix}_in_dst"
            assert getattr(t, f"{prefix}_out_src") is not None, f"{name}.{prefix}_out_src"
            assert getattr(t, f"{prefix}_out_dst") is not None, f"{name}.{prefix}_out_dst"


def test_cp_dim_present_in_io_contracts(self=None):
    """I/O 契约声明 CP 维（激活），参数侧 CP 恒 Replicate 由 _multi_dim 保证。"""
    for name, t in TEMPLATES.items():
        for key, named in t.sp_in_dst.items():
            assert CP in named, f"{name}.sp_in_dst[{key}] 缺 CP 维"


def test_attention_needs_cp_attn_and_keeps_cp_shard():
    t = TEMPLATES["attention"]
    assert t.needs_cp_attn is True
    # §6.3.2 非对称职责：attention sp_in_dst 的 CP 维保持 Shard(1)（boundary 不 gather）
    assert t.sp_in_dst["hidden_states"][CP] == Shard(1)
    assert t.sp_out_dst[CP] == Shard(1)


def test_moe_mlp_use_local_map():
    assert TEMPLATES["moe_mlp"].use_local_map is True
    assert TEMPLATES["moe_mlp"].moe_expert_placement == Shard(0)


def test_moe_gate_out_dst_ep_shard():
    t = TEMPLATES["moe_gate"]
    from hyper_models.components.distributed.sharding_config import EP
    assert t.sp_out_dst[EP] == Shard(0)
    assert t.nosp_out_dst[EP] == Shard(0)


def test_lm_head_out_src_shard_last_dim():
    # 标量简写：{TP: Shard(-1), ...}
    from hyper_models.components.distributed.sharding_config import TP
    assert TEMPLATES["lm_head"].sp_out_src[TP] == Shard(-1)


def test_norm_template_all_replicate_params():
    assert TEMPLATES["norm"].norm_placement == Replicate()
