# Chunk KDA：总接口与显式阶段接口

这些接口位于 `hyper_parallel.custom_ops.experimental`，目前仅实现 MindSpore Ascend 后端。需要配套的 FLA-NPU OPP；本文按 FLA-NPU `70eb6ba8` 的 ACLNN ABI 编写。不同 OPP 的支持范围不能混用。

## 两层接口

- 普通训练使用 `npu_chunk_kda`：HP 管理自动微分和缓存。Ulysses CP 在外部通过 `ContextParallel` 先将输入变为“完整序列、本地 heads”，再调用总接口。
- 沿序列切分、交换递归状态的 CP 使用下面九个显式阶段原语。它们只处理本地 Tensor，**没有隐式自动微分或通信**。调用方负责缓存、边界状态、AllGather/PTP、反向编排及共享参数梯度归约。

不能把这些阶段直接串到普通 loss 后，期待框架自动生成整个 CP 的 backward；应由调用方提供相应的自定义反向或显式训练流程。

| 接口后缀（均以 `npu_chunk_kda_` 开头） | ACLNN 名称（均以 `aclnn` 开头） | 固定返回值 |
|---|---|---|
| `fwd_prepare` | `ChunkKdaFwdPrepare` | 13 项 cache，见下文 |
| `pre_process_fwd` | `PreProcessFwdKernelMerged` | `(summary,)` |
| `fwd_h` | `ChunkFwdH` | `(h, v_new, final_state)` |
| `fwd_finalize` | `ChunkKdaFwdFinalize` | output Tensor |
| `bwd_prepare` | `ChunkKdaBwdPrepare` | `(dAqk, dv_local, dq_raw)` |
| `pre_process_bwd` | `PreProcessBwdKernelMerged` | summary Tensor |
| `bwd_dhu` | `ChunkGatedDeltaRuleBwdDhu` | `(dh, dh0, dv_scan)` |
| `bwd_finalize` | `ChunkKdaBwdFinalize` | `(dq, dk, dv, dbeta, dg, dA_log, ddt_bias)` |
| `merge_fwd_bwd` | `MergeFwdBwdKernel` | 更新后的 state Tensor（原位写回） |

## 布局、缓存与 dtype

`fwd_prepare(layout=...)` 将 layout 直接传给 FLA，不在 Python 中转轴：

| layout | Q/K | V | gate | beta |
|---|---|---|---|---|
| BNSD | `[B,Hk,T,K]` | `[B,Hv,T,V]` | `[B,Hv,T,K]` | `[B,Hv,T]` |
| BSND | `[B,T,Hk,K]` | `[B,T,Hv,V]` | `[B,T,Hv,K]` | `[B,T,Hv]` |
| NTD | `[Hk,T,K]` | `[Hv,T,V]` | `[Hv,T,K]` | `[Hv,T]` |
| TND | `[T,Hk,K]` | `[T,Hv,V]` | `[T,Hv,K]` | `[T,Hv]` |

prepare 的输出始终是 head-major（rank-4 BNSD 或 rank-3 NTD），不是输入 layout：

```text
0 gk          FP32，末维 K
1 Aqk         Q dtype，末维 chunk_size
2 Akk         Q dtype，末维 chunk_size
3 w           Q dtype，末维 K
4 u           Q dtype，末维 V
5 qg          Q dtype，末维 K
6 kg          Q dtype，末维 K
7 qg_scaled   Q dtype，末维 K
8 q_hat       Q dtype，Q 的 heads/末维
9 k_hat       Q dtype，K 的 heads/末维
10 q_rstd     FP32，无 feature 维，Hk heads
11 k_rstd     FP32，无 feature 维，Hk heads
12 beta_eff   FP32，无 feature 维，Hv heads
```

前八项使用 value heads。`backward_mode` 只决定输出缓存，不选择自动微分或自动重算路径：

| 模式 | 非 None 的槽位 |
|---|---|
| `none` | 0,1,3,4,6,7 |
| `forward` | 上述 + 2 |
| `recompute` | 上述 + 8..12 |
| `save`（默认） | 全部 13 项（含 5） |

未请求的输出固定为 `None`，不改变 tuple 长度。未开启归一化时 norm 缓存为 identity；beta_eff 保存有效 beta。

状态/反向接口另有独立契约：

- FwdH 和 Dhu 当前接收 rank-4 BNSD relays。rank-3 prepare 的 cache 不能直接传给这两个接口；需调用方显式增加 batch 维并匹配状态形状。
- FwdH 的 `h` 是 `[B,Nc,Hv,K,V]`，dtype 跟随输入；`v_new` 跟随 u。边界 state 是 `[N,Hv,K,V]`，N 为 batch 数或 packed 序列数，Nc 按每条序列分别计数并包含 tail。
- FwdH 可传 BF16/FP32 initial_state；final_state 跟随 initial_state dtype，无初态时为 FP32。`state_v_first=True` 将 h 和边界 state 的最后两维改为 V,K。
- Dhu 的 h0/dht/dh0 dtype 必须跟随 Q，而不是总接口前向默认的 FP32 state。**dh 始终为 K-first**，即使边界 state 是 V-first。只有传 h0 才返回 dh0。
- BwdPrepare 的 `Aqk/v_new/d_o` 必须都符合其 layout；dAqk/dq_raw 为 FP32，dv_local 跟随 value dtype。
- 当前配套 BwdFinalize 要求 BF16 Q/K/V 及 relays、chunk=64、K-first、`safe_gate/use_gate_in_kernel/use_exp2=True`，并要求 a_log 和 dt_bias。dg/dbeta 跟随 raw_g/beta dtype，两个参数梯度为 FP32。能力检查交由 FLA 执行。
- 开启 in-kernel QK norm 时，BwdFinalize 传 `q_hat/k_hat` 和配对的 FP32 rstd；开启 beta sigmoid 时，传 FP32 `beta_eff` 而非 raw logits，并保持 `allow_neg_eigval` 一致。
- FwdFinalize 的 output_layout 显式选择 BNSD/BSND 或 NTD/TND；BwdFinalize 同样显式选择梯度 layout。不要假定所有阶段的输出保持原始输入 layout。现有总接口对 BNSD 输入也返回 token-major output，比较时需要显式转轴。

`cu_seqlens/chunk_indices` 是 host 整数 metadata，可用 list 或 Tensor。未提供 chunk_indices 时，支持该参数的阶段按每条序列的 tail 自动生成规范索引。pre-process 接口的 cu_seqlens 可以选择子区间，不会被当作完整 chunk plan 重写。各项 shape/dtype/SoC/GQA 能力以目标 FLA stage 为准，不能从一个阶段的能力推断其他阶段。

## 显式前反向示例

以下是 BF16、BNSD、K=V=128、chunk=64、safe raw gate 的 CP=1 链。a_log、dt_bias 为 FP32，q/k/v/g/beta 已由调用方准备。scale 通常取 K**-0.5。

```python
from hyper_parallel.custom_ops.experimental import (
    npu_chunk_kda_fwd_prepare, npu_chunk_kda_fwd_h,
    npu_chunk_kda_fwd_finalize, npu_chunk_kda_bwd_prepare,
    npu_chunk_kda_bwd_dhu, npu_chunk_kda_bwd_finalize,
)

c = npu_chunk_kda_fwd_prepare(
    q, k, v, g, beta, scale, safe_gate=True,
    use_gate_in_kernel=True, a_log=a_log, dt_bias=dt_bias,
)
gk, aqk, akk, w, u, qg, kg, qs = c[:8]

# Sequence-split CP 在此先生成 summary、通信并合并，得到本 rank 的 h0。
h, vn, final = npu_chunk_kda_fwd_h(
    kg, w, u, gk=gk, initial_state=h0, output_final_state=True,
)
out = npu_chunk_kda_fwd_finalize(qs, aqk, vn, h, output_layout="BNSD")

# d_o 由上层反向提供，并与此处的 BNSD output 对齐。
da, dv_local, dq_raw = npu_chunk_kda_bwd_prepare(aqk, vn, d_o, h, scale=scale)
# Sequence-split CP 在此执行反向 summary/通信，得到本 rank 的 dht。
# h0_relay/dht dtype 必须跟随 q；h0 为 None 时 h0_relay 也为 None。
h0_relay = None if h0 is None else h0.astype(q.dtype)
dh, dh0, dv_scan = npu_chunk_kda_bwd_dhu(
    qg, kg, w, d_o, dv_local, scale, gk=gk, h0=h0_relay, dht=dht,
)
dq, dk, dv, dbeta, dg, da_log, dbias = npu_chunk_kda_bwd_finalize(
    q, k, v, gk, g, beta, a_log, dt_bias, akk, vn, h, dh,
    dv_scan, da, dq_raw, scale=scale,
)
```

CP=1 且无边界状态时，示例中的 h0/dht 均为 None。总接口现在也可用相同阶段链处理 initial-state/final-state 自动微分；支持范围和精度缺口见下一节。

## 总接口的状态与重计算

配套 safe raw gate（`safe_gate/use_gate_in_kernel/use_exp2=True`、提供 a_log）使用本地阶段链，普通 attention loss 也走该路径。保存模式保留 relays、h 和 v_new；`disable_recompute=False` 的前向只保留原输入和边界状态，反向重新执行 prepare 和 FwdH，实际重建缓存。它不等于 `FwdPrepare(backward_mode="recompute")`，后者只是显式阶段的输出选择。

`initial_state` 可微，终态上游梯度传给 Dhu，`dh0` 返回初态；`state_v_first=True` 在阶段链外转为 K-first，并由框架反转输出及输入梯度的转置。支持 BNSD/BSND/NTD/TND、GQA 的 repeat/reduce、dense tail 和已验证 packed 边界。总接口保持既有 token-major output 契约。底层 fused backward 对 state 的拒绝并未删除，新增能力通过公开 stage 实现。

该状态/重计算路径要求 BF16、K=V=128、chunk=64 和六个本地阶段 API。缺少阶段会按需报依赖错误；不会静默退回忽略状态的旧反向。总接口默认 safe_gate 仍为 False；调用方需显式开启它才能使用新增路径。

**精度尚未完整验收**：强 gate 衰减、T=65、常量终态梯度且 loss 仅依赖 final_state 时，dk relative L2 为 6.8738%，超过 4% 阈值；该测试保留为失败，不跳过、不放宽门槛。主输出与终态的混合 loss 测试通过，不能据此宣称任意 state loss 精度通过。

末尾 token 的 v_new 对 CPU reference 的 relative L2 约 0.204%，但与常量 dht 点积时因正负抵消放大到 8.727%；Dhu 输出的末段 dh 与 dht 一致。用实际 NPU v_new 在 CPU 计算对应 dk 项，与 NPU 末尾 dk 差约 0.130%。这支持 BF16 relay 量化/抵消为主要误差路径，尚不足以证明所有 kernel 阶段无误；需要继续评估高精度 relay 或专门的数值改进。

## Summary 和通信责任

`pre_process_fwd(kg,w,u,gk=...,cu_seqlens=...)` 返回 FP32 `[N,Hv,K,V+K]`；`pre_process_bwd(qg,kg,w,d_o,dv_local,scale,gk=...)` 返回单段的 FP32 `[Hv,K,V+K]`。summary 拼接 `[E | M]`，表示 `H_out=M@H_in+E`。

调用方 AllGather 后可调用 `merge_fwd_bwd(state,summaries,count,rank,...)`，或按 FLA 约定做 FP32 compose/PTP。merge 接收 rank-3 state 和 rank-4 gathered summaries，dtype 必须匹配，更新并返回原 state；非连续 state 会报错，不能静默写回副本。count/rank 和正反向 summary 顺序必须遵循 FLA MergeFwdBwdKernel 契约。

HP 不创建通信组、不调用 collective、不管理 packed 序列跨 rank reset，也不归约 a_log/dt_bias 梯度。上层需正确处理这些事项以及卷积 halo、loss weighting、异步 buffer 生命周期。

## 可选依赖和验证

九个 bridge 使用运行时符号解析，不依赖 FLA 头文件或静态链接 FLA。缺某个 stage 不触发整包重新编译，也不阻断其他 custom ops 的导入；仅在调用缺失阶段时报出对应 ACLNN 名称。已装的 HP extension 太旧、没有 bridge 时，需要用户显式重建。

验证入口：

```bash
# 激活项目环境；硬件测试前加载 CANN 和配套 FLA OPP 环境。
python -m pytest tests/ut/custom_ops/experimental/test_chunk_kda_stages.py -q
HYPER_PARALLEL_PLATFORM=mindspore python -m pytest \
  tests/mindspore/st/custom_ops/test_chunk_kda_stages.py -q -rs
```

UT 检查路由、metadata、固定返回值、缺 bridge、原位状态和 Torch 明确报错。ST 使用独立 CPU FP32 递推和 autograd，检查 output、final state、七项梯度、dh0，并比较 stage chain 与总接口。要求 finite，relative L2 < 4% 或 RMS absolute < 3e-8，参考向量范数 > 1e-12 时还要求 cosine > 0.995；layout/cache 对比使用更严格的 1% relative L2。另覆盖融合 QK norm、beta sigmoid 和 allow_neg_eigval 的梯度。

缺少 FLA API 时 ST 按阶段跳过；真实 kernel 失败和数值误差不被吞掉。单卡通过不等于多卡通过；以下分别记录已测范围。

2026-11-04 开发验证：相关 UT 回归 41 passed（另 4 subtests passed）；完整单卡 ST 41 passed、1 failed（上述 state-only dk 精度）。完整 OPP 已构建并装到独立验证目录，原来因缺两个 PreProcess 和 Merge API 跳过的 5 项已实际运行通过。

MF 调用方已完成 Ulysses CP2/CP4、TP2×CP2+FSDP+LoRA+bias 和 CP2/CP4 重计算的两步训练对照。另有 recurrence-only state scan：CP2/tail65/保存模式、CP4/tail65/重计算+V-first，以及 CP2/aligned64/重计算+V-first，均对照独立 CPU FP32 recurrence/autograd。通信、参数梯度归约在 MF；当前 state scan 仅限 B=1、每 rank 一个非空按序连续 dense shard，不含 ShortConv halo、packed CP 或 PTP。它是实验性调用方组件，不是生产模型 Ulysses 策略的自动替代。
