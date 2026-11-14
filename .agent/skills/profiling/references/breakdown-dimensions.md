# 拆解维度与分类规则

规则版本 **v2**（按 CANN 9.0.0 + torch_npu 2.9.0 的实采字段重写；v1 按通配猜的
`op_summary`/`Task Type` 在该版本不存在）。字段出处与实测证据见
[output-files.md](output-files.md)。

## 步时拆解：三项加和 + 一项子集

工具链已经把步时切好了，`step_trace_time.csv` 的实测恒等式是：

```
Stage = Computing + Communication(Not Overlapped) + Free
```

| 维度 | 来源 | 判定规则 | 加和 |
|---|---|---|---|
| **计算** | `step_trace_time.csv:Computing` | 工具链给定 | 是 |
| **通信（未掩盖）** | `step_trace_time.csv:Communication(Not Overlapped)` | 工具链给定 | 是 |
| **Free** | `step_trace_time.csv:Free` | 工具链给定 | 是 |
| **优化器** | `operator_details.csv` 的 `Optimizer.step#*` 区间 `Device Total` | 每步一条 | **否，是 Computing 的子集** |

> 路线图把拆解写成「四维」，但实测的加和维度只有三项——**优化器落在
> Computing 内部**。单列优化器是为了看它占步时多少，若把它当第四个加和项，
> 总和会超过 100%。报告里据此标注。

占比以各维度耗时 / `Stage`。采集本身有开销，拆解取相对占比，不把带 profiling
开销的绝对步时当真实步时。

## 计算维度分类（按 Core Type，非按算子名）

`op_statistic.csv:Core Type`（= `kernel_details.csv:Accelerator Core`）直接给出
执行单元，不需要按算子名猜：

| 取值 | 归类 |
|---|---|
| `AI_CORE` | Cube |
| `AI_VECTOR_CORE` | Vector |
| `MIX_AIV` | 混合 |
| `AI_CPU` | AI CPU |
| `COMMUNICATION` | 通信 kernel |

未登记的取值**原样保留**，不折进相邻类别。

FA（FlashAttention）这类融合算子在 `Core Type` 上仍是 `AI_CORE`/`MIX_AIV`，要
单独看需按 `OP Type` 再筛一层；当前按执行单元出一级分类，FA 的二级细分标
`[待对齐]`（取决于所用融合算子的注册名）。

### AI_CPU 里必须先剔通信

HCCL 每个集合通信都附带一个 AI CPU 侧 launch kernel（`*AicpuKernel`，如
`allreduceAicpuKernel`）。它在 AI CPU 上执行，但属于通信，**不是需要消除的算子
回退**。实测一次双卡采集中它占 AI_CPU 总时间 17%（2841us / 16497us）。

规则：计入 `AI_CPU` 类别总量，但排除在「回退热点」表之外。回退热点只收真回退
（典型：整型 ArgSort 落 AI CPU、scatter_reduce 回退）。

## 通信维度

| 项 | 来源 | 读法 |
|---|---|---|
| 每步通信总量 | `communication.json` 各 step 的 `Total Op Info` | 聚合键，枚举单算子时跳过 |
| 等待占比 | `Wait Time(ms) / Elapse Time(ms)` | 高→瓶颈在同步等待，不是带宽 |
| 链路带宽 | `Communication Bandwidth Info` 各链路 | `MB/ms` 即 `GB/s`；`SDMA` 与 `SIO` 同源不相加 |
| 掩盖率 | `Overlapped / Communication`（step_trace） | 低且未掩盖占比高，才是真通信瓶颈 |

### 通信域 → 并行轴

`profiler_metadata.json` 的 `parallel_group_info` 落盘了 `group_name` 与
`global_ranks`，这就是映射来源，不需要外部并行配置 dump。

- 只有一个通信组时直接对应。
- 多组时需把组名与 `communication.json` 键里的 `@<hash>` 对上；该 join key 在
  metadata 中暂未暴露，故多组场景的轴归属标 `[待对齐]`，报告里给出 hash 与
  已声明的组列表，由人工对齐。

## 视角汇总

| 视角 | 含义 | 来源 |
|---|---|---|
| **算子类型** | 按 `OP Type` 聚合耗时 Top-N | `op_statistic.csv` |
| **单 kernel** | 逐 kernel 明细，含 `aic_*`/`aiv_*` 流水线比与 `cube_utilization(%)` | `kernel_details.csv` |
| **框架区间** | `aten::*`、autograd、`Optimizer.step#*` | `operator_details.csv` |
| **Module / Layer** | 需采集时开 `with_modules=True` | `[待对齐]`：本次实采为 `with_modules: false`，未验证 |

> `operator_details.csv` **不能朴素求和**：父区间与子区间都在表里（重复计），
> 且同步原语会拿到伪 device 归属——实测 `wait_event` 的 `Device Self` 累计
> 24882us，`c10d::allreduce_` 的 `Device Total` 被 `wait_event` 污染到 5370us，
> 而真实每步通信仅 1272us。通信耗时取 `step_trace_time.csv` 或
> `communication.json`，不要取 `c10d::*` 区间。

## 一致性校验（正面生效标记）

`breakdown.py` 把下列不变量作为校验项输出，不通过时对应数字不可采信
（退出码 2）：

| 校验 | 依据 |
|---|---|
| `Stage = 计算 + 通信(未掩盖) + Free` | step_trace 自身恒等式 |
| `communication.json` 的 Elapse = step_trace 的 `Communication` | 两份独立文件互证，实测差 0.000us |
| 采集跳过了 warmup | `profiler_info` 的 `schedule.warmup > 0` |
| 稳态步数 ≥ 2 | `schedule.active >= 2` |
| 优化器区间数 = 采集步数 | 每步应恰好一条 `Optimizer.step#` |

前两条是解析正确性的正面证据；后三条把本 skill 的采集硬规则变成机器可查。

## 复现纪律

- 分类规则有版本：报告里写明规则版本，换规则即换口径，要重算。
- 同一份 trace 必须产出同一份拆解；拆解是规则结果不是判断。
- AI_CPU 真回退、未掩盖通信、Free 气泡是三类最常见可动瓶颈，优先标出，交
  `perf-playbook` 选手段。
- 报告带采集 provenance（CANN/torch_npu 版本、schedule、并行组）；无出处的拆解
  不算证据。
