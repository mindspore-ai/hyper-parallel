# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S6（D-09/D-10，05 §6.4.7/§6.4.8）: EP 通信原语分布式用例。

- pad 路径 _ep_all_to_all fwd/bwd 对拍（2 进程 gloo；uneven 路径 collective
  在 NCCL/HCCL 环境验证）。

HF 原生 MoE 的端到端用例见 test_dist_s6_ep_extend.py（D-10 TP-extend-EP：
ep>1 → 扩展 EP 组含 TP rank，expert 权重仅 expert 维切分）。
"""

import torch

from hyper_models.components.distributed.ep_utils import _ep_all_to_all
from tests.components.distributed.conftest import run_dist


def _worker_padded_a2a(rank, world_size):
    """pad-to-max a2a（gloo 路径）：fwd 数值 + bwd 梯度（a2a 是跨 rank 置换）。"""
    assert world_size == 2
    group = None  # world group（gloo → pad 路径）
    h = 4
    if rank == 0:
        send_counts, recv_counts = [2, 1], [2, 0]
        x = torch.tensor([[0.], [1.], [2.]]).repeat(1, h)
        expected = torch.tensor([[0.], [1.]]).repeat(1, h)
    else:
        send_counts, recv_counts = [0, 3], [1, 3]
        x = torch.tensor([[13.], [14.], [15.]]).repeat(1, h)
        expected = torch.tensor([[2.], [13.], [14.], [15.]]).repeat(1, h)
    x.requires_grad_(True)
    out = _ep_all_to_all(x, send_counts, recv_counts, group)
    torch.testing.assert_close(out, expected)
    out.sum().backward()
    # a2a 是跨 rank 置换：每行输入恰好流向一行输出 → grad 全 1
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_ep_all_to_all_padded_2proc():
    run_dist(2, _worker_padded_a2a)
