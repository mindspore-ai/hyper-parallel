# 显存拆解维度与分类规则

规则版本 **v2**（按 torch_npu 2.9.0.post2 实采字段与帧形态重写）。字段出处与
实测证据见 [snapshot-format.md](snapshot-format.md)。

## 总量三层

| 层 | 来源 | 含义 |
|---|---|---|
| Reserved | `Σ segment.total_size` | 分配器向设备保留的总量 |
| Allocated | `Σ segment.allocated_size` | 取样瞬间在用 |
| 碎片 | `Reserved − Allocated` | 保留但未在用；占 Reserved 超 10% 先查分配器配置 |

段总量**直接取段字段**，不重新累加 block——累加值另算，专用于校验（见下）。

峰值 `max_memory_allocated` 不在快照里，由 `mem_hook` 写进
`<snapshot>.stats.json` 旁车。**快照 ≠ 峰值**，差距作为校验项输出。

## 类别判定规则（按记录栈，顺序判定）

| 顺序 | 类别 | 判据 | 实测验证 |
|---|---|---|---|
| 1 | 优化器状态 | Python 帧路径含 `/optim/` | AdamW 走 `torch/optim/adam.py:177/181 _init_group`（AdamW 复用 adam.py，不要按 `adamw` 匹配） |
| 2 | 参数 | Python 帧含 `_module.py` 且函数名为 `convert`/`to` | `torch_npu/utils/_module.py:76 convert`，由 `module.to(device)` 触发 |
| 3 | 梯度 | **native** 帧含 `AccumulateGrad` | 梯度无任何 Python 帧，这是唯一信号 |
| 4 | 激活 | 最内层自有代码帧的函数名含 `forward` | 仅在 backward 期间的快照里出现 |
| 5 | 其他用户张量 | 有自有代码帧但不匹配上述 | 如输入 batch |
| 6 | unclassified | 无可用帧 | 如实标注，不猜 |

### 实测对账（一次已知构成的运行）

模型：3 × `Linear(2048, 2048, bias=False)`，bf16；AdamW；输入 512×2048 bf16。

| 类别 | 工具给出 | 地面真值 | 核对 |
|---|---|---|---|
| 参数 | 24.0 MiB | 3 × 2048² × 2B = 24.0 MiB | ✅ |
| 梯度 | 24.0 MiB | 同参数（bf16）= 24.0 MiB | ✅ |
| 优化器状态 | 48.0 MiB | 2 状态 × 3 × 8 MiB = 48.0 MiB | ✅ |
| 激活（backward 中途） | 12.0 MiB | — | 步后快照降至 2.0 MiB |
| 其他用户张量 | 2.0 MiB | 512 × 2048 × 2B = 2.0 MiB（输入） | ✅ |
| 合计 | 112.0 MiB | = `allocated_size` 112.01 MiB | ✅ |

> 注意 AdamW 的 `exp_avg`/`exp_avg_sq` 走 `zeros_like(param)`，**参数是 bf16 时
> 状态也是 bf16**（8 MiB/个），不是 fp32 的 16 MiB/个。按 fp32 估算会得出错误的
> 预期值。

## 归属（调用点）规则

见 [snapshot-format.md](snapshot-format.md) § frames 的真实形态。要点：

1. 记录栈最内层优先，最内几帧是快照机制自身（`unwind` / `CapturedTraceback` /
   `??`）——**取 `frames[0]` 会让所有块归到同一处**。
2. 取最内层「不在 site-packages 内」的 `.py` 帧 → 调用方自己的代码位置。
3. 退到最内层框架 `.py` 帧。
4. 再退到 native 符号，但**先跳过 `at::` / `c10::` 分配器管道**，否则拿到
   `at::_ops::empty_strided::redispatch` 这类无用帧。

## 一致性校验（正面生效标记）

| 校验 | 依据 | 离线可查 |
|---|---|---|
| 逐段 Σ块 = `total_size`，Σ活跃块 = `allocated_size` | 分配器自洽 | 是 |
| Σ活跃块 = Σ段 `allocated_size` | 同上，全局 | 是 |
| Σ段 `total_size` = `memory_reserved()` | 旁车计数器 | 需旁车 |
| Σ段 `allocated_size` = `memory_allocated()` | 旁车计数器 | 需旁车 |
| 快照是否为峰值（容差 2%） | 旁车 `max_memory_allocated` | 需旁车 |

前四条是解析正确的正面证据；第五条防止把取样瞬间当峰值读。任一不通过
→ 退出码 2，报告里标红。

## 采集纪律

- **默认采 step 1–3**（`--steps`）；跑长不代表构成会变，但峰值可能在后面出现。
- **要看激活就必须在 backward 期间取样**：`mem_hook.dump(tag)` 可从 autograd
  hook 内调用。步后快照里激活为 0。
- **长跑要能中途取样**：性能/显存异常时当场 dump 当场定位，不要等跑完
  （见仓库纪律：性能是贯穿整个实验的判据）。
- 省显存的手段（重计算、offload、分配器配置）见 `perf-playbook`；本 skill 只
  定位占用来源，**换手段必须同时量化性能影响**，不能只报显存降了。

## 待对齐

- `device_traces` 回放峰值构成的口径 `[待对齐]`（当前只用快照瞬间 + 旁车峰值
  标注差距）。
- Module/Layer 粒度归并：需要按模块名正则把调用点聚合，规则随实践补充。
