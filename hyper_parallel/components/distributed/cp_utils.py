# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""cp_utils: Context Parallel 工具（05 §4.4.2 / §6.3.4 canonical）。

- ``flex_cp_allgather``：CP 维 K/V all-gather（复用 cp_mesh.get_group()，禁 new_group）；
- ``shard_batch_for_cp``：数据管道 CP 切分（与 02 collater 的 THD 契约对齐）；
- ``_shard_seq_lens_for_cp``：seq_lens/seq_lens_padded 按 CP rank 重算。

说明（G5）：seq_len % (2*cp) 的 padding 约束源自 zigzag/ring 负载均衡方案；
本设计采用 all-gather K/V + contiguous chunk（D-01'' 已否决 ring），各 rank
Q chunk 等长、FLOPs 天然均衡，约束冗余但无害——保留实现、文档注明。
"""

import torch
import torch.distributed as dist


class _AllGatherAlongDim(torch.autograd.Function):
    """all-gather along cp_dim + backward reduce-scatter 语义（求和后取本 rank chunk）。"""

    @staticmethod
    def forward(ctx, t, cp_dim, group, cp_size):
        ctx.cp_dim = cp_dim
        ctx.group = group
        ctx.cp_size = cp_size
        world_t = [torch.empty_like(t) for _ in range(cp_size)]
        dist.all_gather(world_t, t.contiguous(), group=group)
        # 按 cp_rank 顺序 cat：[chunk_rank0, chunk_rank1, ...]
        return torch.cat(world_t, dim=cp_dim)

    @staticmethod
    def backward(ctx, grad_output):
        # reduce-scatter：跨 rank 求和梯度，取本 rank 对应的 chunk
        grad = grad_output.contiguous().clone()
        dist.all_reduce(grad, group=ctx.group)
        rank = dist.get_rank(ctx.group)
        local = torch.chunk(grad, ctx.cp_size, dim=ctx.cp_dim)[rank]
        return local.contiguous(), None, None, None


def flex_cp_allgather(k, v, cp_dim: int, cp_mesh):
    """All-gather K/V along CP dimension for context parallel attention.

    Forward: all-gather K/V 沿 cp_dim（各 rank 持有全量 K/V）。
    Backward: reduce-scatter 语义（梯度跨 rank 求和后取本 rank chunk，
      由 _AllGatherAlongDim autograd.Function 显式实现——plain
      ``dist.all_gather`` 没有 autograd 核）。

    Args:
        k, v: [B, N, S_local, H]（cp_dim=2 时为序列维）。
        cp_dim: gather 维度。
        cp_mesh: CP 维 DeviceMesh。通信组取 ``cp_mesh.get_group()``——
            DeviceMesh 构建时已创建并缓存，**此处不得再调 dist.new_group**
            （否则每次 forward 泄露一个 process group，且语义错位）。
    """
    cp_size = cp_mesh.size()
    if cp_size <= 1:
        return k, v
    group = cp_mesh.get_group()
    return (_AllGatherAlongDim.apply(k, cp_dim, group, cp_size),
            _AllGatherAlongDim.apply(v, cp_dim, group, cp_size))


def shard_batch_for_cp(batch: dict, cp_mesh) -> dict:
    """将 batch 中的序列维度 tensors 沿 CP mesh 切分（05 §6.3.4 canonical）。

    契约（与 02 collater 产出对齐）：
      - input_ids/labels/position_ids: [B, S] int64
      - seq_lens / seq_lens_padded: [B, max_num_packs] int64，-1000 哨兵填充
      - qkv_format: "thd"（透传）

    切分策略：pad 到 2*cp 倍数后按 token 区间 [cp_rank*chunk, (cp_rank+1)*chunk)
    切片；seq_lens 系列单独重算（_shard_seq_lens_for_cp）。
    """
    cp_size = cp_mesh.size()
    if cp_size <= 1:
        return batch

    cp_rank = cp_mesh.get_local_rank()
    seq_len = batch["input_ids"].shape[1]
    pad_len = (-seq_len) % (cp_size * 2)
    chunk = (seq_len + pad_len) // cp_size
    lo = cp_rank * chunk
    hi = lo + chunk
    slc = slice(lo, hi)

    _PAD_VALUE = {"labels": -100, "input_ids": 0, "attention_mask": 0}
    padded = dict(batch)
    if pad_len > 0:
        for k, v in batch.items():
            if k == "qkv_format" or not isinstance(v, torch.Tensor) or v.ndim < 1:
                continue
            if k in ("seq_lens", "seq_lens_padded"):
                continue  # 单独重算，不 pad
            if k == "position_ids":
                # position_ids 递增 pad：接续末值继续递增
                last = v[..., -1:].to(torch.long)
                inc = torch.arange(1, pad_len + 1, device=v.device,
                                   dtype=v.dtype)
                inc = inc.reshape(*([1] * (v.ndim - 1)), pad_len)
                pad_block = inc.expand(*v.shape[:-1], pad_len) + last
            else:
                shape = list(v.shape)
                shape[-1] = pad_len
                pad_block = torch.full(shape, _PAD_VALUE.get(k, 0),
                                       dtype=v.dtype, device=v.device)
            padded[k] = torch.cat([v, pad_block], dim=-1)

    out = {}
    for k, v in padded.items():
        if k in ("seq_lens", "seq_lens_padded"):
            continue
        if k == "qkv_format":
            out[k] = v
        elif isinstance(v, torch.Tensor) and v.ndim >= 1:
            out[k] = v[..., slc]
        else:
            out[k] = v

    if "seq_lens" in batch and "seq_lens_padded" in batch:
        out["seq_lens"], out["seq_lens_padded"] = _shard_seq_lens_for_cp(
            batch["seq_lens"], batch["seq_lens_padded"],
            cp_rank=cp_rank, chunk=chunk,
        )
    return out


def _shard_seq_lens_for_cp(seq_lens, seq_lens_padded, *, cp_rank: int, chunk: int):
    """seq_lens/seq_lens_padded 按 CP 分片重算（保留 -1000 哨兵语义）。

    遍历每个样本的 pack 累计偏移（按 seq_lens_padded 累加），对每个 pack：
    - 完全在 [lo, hi) 内：原样保留；
    - 跨界：截断到 [lo, hi)，按截断后实际/含 padding 长度重算；
    - 完全在外：跳过。
    输出平移到本地坐标系；max_local_packs=0 时置 1 防空 tensor。
    """
    B, _K = seq_lens.shape
    lo = cp_rank * chunk
    hi = lo + chunk
    device = seq_lens.device
    SENTINEL = -1000

    local_lens_b, local_lens_padded_b = [], []
    max_local_packs = 0
    for b in range(B):
        row_lens = seq_lens[b].tolist()
        row_padded = seq_lens_padded[b].tolist()
        local_lens, local_padded = [], []
        offset = 0
        for raw_len, raw_pad in zip(row_lens, row_padded):
            if raw_len == SENTINEL:
                break
            pack_start = offset
            pack_end = offset + raw_pad
            offset = pack_end
            inter_start = max(pack_start, lo)
            inter_end = min(pack_end, hi)
            if inter_start >= inter_end:
                continue
            actual_start = max(pack_start, lo)
            actual_end = min(pack_start + raw_len, hi)
            local_actual = max(actual_end - actual_start, 0)
            local_pad = inter_end - inter_start
            if local_actual > 0 or local_pad > 0:
                local_lens.append(local_actual)
                local_padded.append(local_pad)
        local_lens_b.append(local_lens)
        local_lens_padded_b.append(local_padded)
        max_local_packs = max(max_local_packs, len(local_lens))

    if max_local_packs == 0:
        max_local_packs = 1

    out_lens = torch.full((B, max_local_packs), SENTINEL,
                          dtype=seq_lens.dtype, device=device)
    out_padded = torch.full((B, max_local_packs), SENTINEL,
                            dtype=seq_lens_padded.dtype, device=device)
    for b in range(B):
        n = len(local_lens_b[b])
        if n > 0:
            out_lens[b, :n] = torch.tensor(
                local_lens_b[b], dtype=seq_lens.dtype, device=device)
            out_padded[b, :n] = torch.tensor(
                local_lens_padded_b[b], dtype=seq_lens_padded.dtype, device=device)
    return out_lens, out_padded


def _cp_offset_causal_mask(q_len: int, kv_len: int, lo: int,
                           device, dtype=torch.bool):
    """D-04：offset-aware causal mask（本 rank Q chunk 全局偏移 lo）。

    允许 attend 的位置：j <= lo + i（i 为本地 Q 行号）。
    替代 is_causal=True——SDPA 的 is_causal 在 q_len ≠ kv_len 时按右下对齐，
    对 rank>0 的 chunk 会错误掩码（G4）。
    """
    i = torch.arange(q_len, device=device).view(-1, 1)
    j = torch.arange(kv_len, device=device).view(1, -1)
    return (j <= (lo + i)).to(dtype)
