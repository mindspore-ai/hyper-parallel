# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S3.6: shard_batch_for_cp（FakeMesh 单进程，逐 rank 参数化断言）。"""

import torch

from hyper_models.components.distributed.cp_utils import shard_batch_for_cp


class FakeCpMesh:
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


def _batch(S=10):
    return {
        "input_ids": torch.arange(S).unsqueeze(0),
        "labels": torch.arange(100, 100 + S).unsqueeze(0),
        "position_ids": torch.arange(S).unsqueeze(0),
        "qkv_format": "thd",
    }


class TestShardBatch:
    def test_cp_size1_passthrough(self):
        b = _batch()
        out = shard_batch_for_cp(b, FakeCpMesh(1, 0))
        assert out is b

    def test_equal_split_no_pad(self):
        b = _batch(S=8)
        for rank, slc in ((0, slice(0, 4)), (1, slice(4, 8))):
            out = shard_batch_for_cp(b, FakeCpMesh(2, rank))
            torch.testing.assert_close(
                out["input_ids"], b["input_ids"][:, slc])
            torch.testing.assert_close(out["labels"], b["labels"][:, slc])
            assert out["qkv_format"] == "thd"

    def test_pad_to_2cp_multiple(self):
        """S=10 pad 到 12（2*cp=4 的倍数）→ chunk=6；最后 rank 的 chunk 含 pad 区。"""
        b = _batch(S=10)
        out0 = shard_batch_for_cp(b, FakeCpMesh(2, 0))
        out1 = shard_batch_for_cp(b, FakeCpMesh(2, 1))
        assert out0["input_ids"].shape[1] == 6
        assert out1["input_ids"].shape[1] == 6
        # rank0: 原始前 6 个
        torch.testing.assert_close(out0["input_ids"], b["input_ids"][:, :6])
        # N6：rank1 的 pad 区 label=-100、input_ids=0、position_ids 连续递增
        torch.testing.assert_close(out1["labels"][0, -2:],
                                   torch.tensor([-100, -100]))
        torch.testing.assert_close(out1["input_ids"][0, -2:],
                                   torch.tensor([0, 0]))
        torch.testing.assert_close(out1["position_ids"][0, -2:],
                                   torch.tensor([10, 11]))
        # rank1 的有效区 == 原始 [6:10]
        torch.testing.assert_close(out1["input_ids"][0, :4],
                                   b["input_ids"][0, 6:10])

    def test_non_tensor_passthrough(self):
        b = _batch(S=8)
        b["meta"] = "info"
        out = shard_batch_for_cp(b, FakeCpMesh(2, 0))
        assert out["meta"] == "info"

    def test_seq_lens_recomputed(self):
        b = _batch(S=8)
        b["seq_lens"] = torch.tensor([[8, -1000]])
        b["seq_lens_padded"] = torch.tensor([[8, -1000]])
        out = shard_batch_for_cp(b, FakeCpMesh(2, 1))
        # rank1 区间 [4,8)：一个 pack 截断为 4
        assert out["seq_lens"][0, 0].item() == 4
