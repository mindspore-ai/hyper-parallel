# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S2.10（2 进程）: Phase D tied weights（detect/broadcast/replicate）。"""

import torch

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_models.components.distributed.sharding_applier import (
    detect_tied_weights,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaForCausalLM,
    run_dist,
)


def _worker_tied(rank, world_size):
    torch.manual_seed(1234)
    cfg = TinyConfig(tie_word_embeddings=True)
    model = TinyLlamaForCausalLM(cfg)

    tied = detect_tied_weights(model)
    assert tied == [("model.embed_tokens.weight", "lm_head.weight")]

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    assert plan.tied_pairs == [("model.embed_tokens.weight", "lm_head.weight")]
    model, _ = apply_sharding_plan(model, plan, mesh)

    # N9：rank1 是接收端（src=0）——每个 rank 上两端 local 存储逐元素一致
    emb_w = model.model.embed_tokens.weight
    lm_w = model.lm_head.weight
    torch.testing.assert_close(emb_w.data, lm_w.data)
    # 且与全量切片的本地段一致（tied 内容 == 原始 embed 权重切片）
    torch.manual_seed(1234)
    ref = TinyLlamaForCausalLM(cfg)
    chunk = cfg.vocab_size // world_size
    torch.testing.assert_close(
        emb_w.data,
        ref.model.embed_tokens.weight.data[rank * chunk:(rank + 1) * chunk],
    )


def _worker_not_tied(rank, world_size):
    torch.manual_seed(1234)
    model = TinyLlamaForCausalLM(TinyConfig(tie_word_embeddings=False))
    assert detect_tied_weights(model) == []
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    assert plan.tied_pairs == []


def test_tied_weights_2proc():
    run_dist(2, _worker_tied)


def test_not_tied_2proc():
    run_dist(2, _worker_not_tied)
