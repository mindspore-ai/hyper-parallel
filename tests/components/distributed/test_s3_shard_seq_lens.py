# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S3.7: _shard_seq_lens_for_cp（pack 完全在内/跨界/在外/哨兵/防空）。"""

import torch

from hyper_models.components.distributed.cp_utils import _shard_seq_lens_for_cp

SENTINEL = -1000


def _run(seq_lens, seq_lens_padded, cp_rank, chunk):
    return _shard_seq_lens_for_cp(seq_lens, seq_lens_padded,
                                  cp_rank=cp_rank, chunk=chunk)


class TestShardSeqLens:
    def test_pack_fully_inside(self):
        """pack [0,4) 完全落在 rank0 [0,4) 内 → 原样保留。"""
        sl = torch.tensor([[4, -1000]])
        slp = torch.tensor([[4, -1000]])
        out_lens, out_pad = _run(sl, slp, cp_rank=0, chunk=4)
        assert out_lens[0, 0].item() == 4
        assert out_pad[0, 0].item() == 4

    def test_pack_crosses_lo_boundary(self):
        """pack [0,6) 跨 rank1 的 lo=4 边界 → 截断为 [4,6) 长度 2。"""
        sl = torch.tensor([[6, -1000]])
        slp = torch.tensor([[6, -1000]])
        out_lens, out_pad = _run(sl, slp, cp_rank=1, chunk=4)
        assert out_lens[0, 0].item() == 2
        assert out_pad[0, 0].item() == 2

    def test_pack_crosses_hi_boundary(self):
        """pack [2,8) 跨 rank0 的 hi=4 → 截断为 [2,4) 长度 2。"""
        sl = torch.tensor([[6, -1000]])
        slp = torch.tensor([[6, -1000]])
        out_lens, out_pad = _run(sl, slp, cp_rank=0, chunk=4)
        # pack_start=2 → 需要另一个前置 pack 构造 offset
        sl = torch.tensor([[2, 6, -1000]])
        slp = torch.tensor([[2, 6, -1000]])
        out_lens, out_pad = _run(sl, slp, cp_rank=0, chunk=4)
        assert out_lens[0, 0].item() == 2   # 第一个 pack 完整
        assert out_lens[0, 1].item() == 2   # 第二个 pack 截断到 hi=4
        # 哨兵填充
        assert out_lens.shape[1] == 2

    def test_pack_fully_outside(self):
        """pack [0,4) 完全在 rank1 [4,8) 外 → 跳过（防空 → 哨兵）。"""
        sl = torch.tensor([[4, -1000]])
        slp = torch.tensor([[4, -1000]])
        out_lens, out_pad = _run(sl, slp, cp_rank=1, chunk=4)
        # max_local_packs=0→1 防空
        assert out_lens.shape == (1, 1)
        assert out_lens[0, 0].item() == SENTINEL

    def test_sentinel_terminates(self):
        """哨兵之后的 pack 不处理。"""
        sl = torch.tensor([[4, -1000, 4]])
        slp = torch.tensor([[4, -1000, 4]])
        out_lens, _ = _run(sl, slp, cp_rank=0, chunk=8)
        assert out_lens.shape[1] == 1
        assert out_lens[0, 0].item() == 4

    def test_padded_covers_separator(self):
        """seq_lens_padded 含 separator：pack 实际 3 + pad 1，跨界按 padded 截断。"""
        sl = torch.tensor([[3, -1000]])
        slp = torch.tensor([[4, -1000]])
        out_lens, out_pad = _run(sl, slp, cp_rank=0, chunk=2)
        # pack [0,4) 跨 hi=2：实际 token 截断 [0,2) → 2；pad 区间 [0,2) → 2
        assert out_lens[0, 0].item() == 2
        assert out_pad[0, 0].item() == 2
        # rank1 [2,4)：实际 token [2,3) → 1；pad [2,4) → 2
        out_lens1, out_pad1 = _run(sl, slp, cp_rank=1, chunk=2)
        assert out_lens1[0, 0].item() == 1
        assert out_pad1[0, 0].item() == 2

    def test_per_rank_asymmetry(self):
        """N5：rank0 与 rank1 的 pack 交集不同 → 重算结果不同（逐 rank 断言）。"""
        sl = torch.tensor([[5, 3, -1000]])
        slp = torch.tensor([[5, 3, -1000]])
        out0, _ = _run(sl, slp, cp_rank=0, chunk=4)
        out1, _ = _run(sl, slp, cp_rank=1, chunk=4)
        # rank0 [0,4)：pack1 [0,5) 截断 → 4；pack2 在外 → 共 1 项
        assert out0[0, 0].item() == 4
        # rank1 [4,8)：pack1 [0,5) 截断 → 1；pack2 [5,8) 完整 → 3
        assert out1[0, 0].item() == 1
        assert out1[0, 1].item() == 3
