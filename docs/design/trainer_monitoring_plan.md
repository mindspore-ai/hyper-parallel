# 通用 Trainer 训练监控与日志方案

范围：Base/Text/VLM Trainer；需求来自 [Issue #1174](https://github.com/mindspore-ai/hyper-parallel/issues/1174)。
本文件只保留架构、配置、模块边界和验收要求；具体字段见 [使用说明](../trainer_monitor_usage.md)，
指标的问题定位方法见 [诊断说明](../trainer_monitor_diagnostics.md)，设备实测见 [验收报告](../trainer_monitor_npu_acceptance.md)。
设计要求不等于已经通过设备验收。

## 1. 架构与配置

蓝色表示生命周期，绿色表示现场采集，紫色表示汇总，橙色表示输出，红色表示移除项。

```mermaid
%%{init: {"themeVariables": {"fontSize": "22px"}, "flowchart": {"nodeSpacing": 25, "rankSpacing": 35}}}%%
flowchart TB
    subgraph CONFIG["蓝色 · 生命周期"]
        C["monitor YAML → MonitorSession<br/>初始化、采样、续训、清理"]
    end
    subgraph COLLECT["绿色 · 显式采集，不安装 tensor hook"]
        T["启动 / 取数 / forward / backward / 保存计时"]
        R["Local loss / local grad norm<br/>ExpertLoadRecorder / Muon NS / datasets"]
        F["初始化注册标准 FLOPs 块<br/>step 输入几何 → 运算量"]
    end
    subgraph METER["紫色 · 汇总"]
        S["小摘要异步 D2H<br/>就绪后形成带原始 step 的诊断事件"]
        E["EnvironMeterCallback<br/>loss、吞吐、累计量、耗时、MFU<br/>设备显存沿用原实现"]
    end
    subgraph OUTPUT["橙色 · 唯一输出入口"]
        L["LoggingCallback<br/>外部频次触发 train_step，global rank 0 打印"]
        W["OutputWriter<br/>JSONL 完整记录<br/>可选 TensorBoard / W&B"]
    end
    C --> T & R & F
    R --> S --> W
    T --> E
    T -- "启动 / checkpoint 事件" --> W
    F --> E --> L --> W
    style CONFIG fill:#e8f1ff,stroke:#3975bd,color:#162d4a
    style COLLECT fill:#e8f5eb,stroke:#3b8751,color:#173d25
    style METER fill:#f1eafb,stroke:#8354ad,color:#39204f
    style OUTPUT fill:#fff1df,stroke:#bb792b,color:#573510
```

```mermaid
%%{init: {"themeVariables": {"fontSize": "22px"}}}%%
flowchart TB
    subgraph REMOVE["红色 · 移除与不建设"]
        X["删除 TqdmCallback、进度条 writer 和注册<br/>不迁移 ETA、进度比例、额外 total_steps"]
        Y["删除顶层 debug / wandb 配置<br/>数据日志与 FSDP 顺序检查统一入口到 monitor"]
    end
    style REMOVE fill:#fdebed,stroke:#bd4a57,color:#65242c
```

`EnvironMeterCallback` 汇总基础 step 指标；`MonitorSession` 管理采集生命周期；
`LoggingCallback` 决定何时输出基础训练日志；`OutputWriter` 负责落盘与可选 SDK。
启动、checkpoint 和异步诊断事件直接复用 writer，不伪装成另一条 `train_step`。

### YAML

以下是完整默认配置。只暴露功能开关、采样频次和输出设置，不提供内部 recorder 的配置项。

```yaml
monitor:
  enabled: true
  sinks: [console, jsonl]  # 可选 tensorboard、wandb；不默认上传
  output_dir: ./outputs/metrics
  timing: true
  check_fsdp_runtime: false
  mfu: false
  wandb:
    project: hyper-parallel
    mode: online
  local_loss:
    enabled: false
    sampler_steps: 100
  local_grad_norm:
    enabled: false
    sampler_steps: 100
  moe:
    enabled: false
    sampler_steps: 100
    record_distribution: false
  optimizer:
    ns_norm:
      enabled: false
      sampler_steps: 100
  datasets:
    sampler_steps: 100
    token_distribution: true
    top_n_tokens: 20  # 0 关闭
    packing_stats: true
    packing_boundary_check: true
    log_first_rendered_template: false
    log_level: null  # debug / info / warn；null 跟随全局日志级别
```

- `sampler_steps` 统一表示采集间隔；基础日志沿用外部 `training.logging_steps`，monitor 不重复定义输出频次。
- `local_loss` 是整体 loss；`local_grad_norm` 每个 rank 只有一个值，均不配置 module。
- 持有数据的 rank 都参与采集，不增加 `ranks` 或 `include_children` 开关；不持有 loss 的 PP stage 不制造 loss。
- `training.log_tflops` 控制已有 TFLOP/s 输出，`monitor.mfu` 控制 MFU；两者复用标准块估算。
- W&B 只有显式加入 sinks 才启动；entity/run ID 等实验管理信息复用外部入口。

## 2. 模块职责、问题定位与验收

| 模块 | 监控内容 / 可定位问题 | 实现边界 | 验收依据 |
| --- | --- | --- | --- |
| 基础日志 | total/foundation/router AUX loss、lr、global grad norm；观察收敛与异常 | 保留既有全局聚合语义，LoggingCallback 不重新算指标 | 同一 step 无重复基础日志；全局 AUX 不被 rank-local 值覆盖 |
| 分块耗时 | 组网、模型/数据构建、权重加载/保存、checkpoint、训练各阶段；定位等待与保存阻塞 | 主机墙钟，嵌套阶段不相加；不新增逐块设备同步 | 成功/失败都有记录；异步完成仍属于提交 step |
| Local/micro loss | 规约前 micro loss、token 加权 step loss；定位脏数据与 rank/micro 差异 | loss 计算后、全局规约前显式采集，不改变 backward loss，不新增通信 | 不等长 micro 加权正确；零分母不可用；监控开关前后参数一致 |
| Local grad norm | 当前 rank 全部有梯度的可训练参数的本地分片 L2；定位梯度异常 | Partial 规约后、clip 前；优先 main_grad，复用裁剪视图，包含本地 expert 参数 | 不修改梯度；不增加 collective；没有规约后视图时明确不可用 |
| datasets | 本 step token 长度分布、Top-N、packing 与边界；定位样本异常、低填充与跨样本监督错误 | CPU 源 batch，CP/H2D 前；合并全部 micro 后取 Top-N | 对齐 shift/IGNORE_INDEX/loss_mask/EOD；不分配 S×S mask；缺 metadata 不报成功 |
| EP | expert assignment、max/mean、max_vio、空 expert、drop、可选分布；定位路由倾斜 | ExpertLoadRecorder 固定层序；只记录真实 forward，按布局规约 | top-k 计数守恒；重计算不重复；未执行层不可用；本地之和与全局一致 |
| Muon NS | NS 后、LR/RMS 缩放前的逐矩阵 Frobenius norm；定位优化器更新尺度异常 | 显式插在优化器内部，保留 parameter/transform/call 归属 | norm 与独立复算相符，不保留完整更新 tensor，不改变权重更新 |
| MFU | 标准模块 FLOPs、每设备 TFLOP/s、step/window MFU；定位计算利用率变化 | 初始化建立块清单，step 只代入几何；模型完整覆盖且峰值可信才报 MFU | 独立公式核算；无重复块；缺模型块或峰值不输出误导性数值 |
| 设备显存 | 既有 allocated/reserved 峰值 | 保持既有实现、字段与统计口径 | 与基线输出一致，检查 recorder 是否造成额外峰值或长期增长 |

诊断步骤与前提只维护在 [指标诊断说明](../trainer_monitor_diagnostics.md)，
相关验证 skill 链接该文档，不复制一套公式或诊断表。

### 显式插入和 Hook

| 位置 | 接入方式 |
| --- | --- |
| Trainer 启动、取数、forward、梯度准备、optimizer、checkpoint | `monitor_phase(..., observe=True)` 合并计时与现场采集作用域 |
| backward、scheduler 等仅计时阶段 | `monitor_phase(...)`，不打开 tensor 观察，避免重计算重复统计 |
| loss、clip-grad、MoE dispatch、Muon NS、CPU batch | 原始计算点显式提交已有 tensor / 统计；组件接口不导入 Trainer |
| MFU | 初始化读取模型 metadata，不用 forward hook 逐次计数 |

当前不新增 forward/backward hook。若新模型只能借助 hook 适配，必须单独验证 hook 调用次数、
重计算去重、编译兼容、耗时和显存；不能把 hook 的成本当作零。

### 共同实现边界

- 不改变 loss 归一化、梯度、路由、expert bias 或优化器算法。
- 关闭或未采样时不创建诊断 tensor；只借用现场 tensor，保留有界小摘要，不持有完整 activation、梯度或路由 logits。
- 摘要在训练线程生成，D2H 异步传输；event 就绪后才能读 CPU。异步不代表没有现场 kernel 或 CPU 成本。
- 设备摘要最多 32 槽、每份最多 65,536 元素。EP 所有层打包后提交；溢出显式计数，不无限积压。
- collective 在训练线程上按固定层序执行，不按打印 rank 或路由结果决定是否参加。后台线程只处理 CPU 输出。
- recorder 不应造成可观测的训练性能回退或显存增长；必须通过关闭/开启对照验收，不能只靠代码结构承诺。
- compiled 区域不为 Python 采集插入 graph break；缺失采集点或 external gradient owner 无适配时明确 unavailable。

### 计时和 checkpoint

启动覆盖默认进程组、子组、模型构建、预训练权重加载、tokenizer/processor、dataset/dataloader 构建。
step 覆盖取数等待、H2D 提交、forward/backward、梯度准备/裁剪、optimizer、scheduler。
`get_batch` 包含内部等待和传输，不得将嵌套值相加冒充 step 时间；worker transform/collate 无独立 metadata 时不单独归因。

| 保存阶段 | 含义 |
| --- | --- |
| `weights_save` | 完整 HF 权重与资产导出，包括完成 barrier |
| `checkpoint_save` | 同步 DCP 保存 |
| `checkpoint_submit` | 异步 DCP 提交返回 |
| `checkpoint_persist` | 提交到 persist future 完成，保留原 step 与成功/失败状态 |
| `checkpoint_wait` | 调用方等待未完成保存 |
| `checkpoint_restore` | 续训恢复 |

checkpoint extra state 保存累计 token/sample 和 run ID；恢复后再启动远端 writer。
旧 checkpoint 缺少监控状态时允许加载，但不能恢复旧作业未保存的累计量。

### 数据与 EP 口径

`data/*` 表示数量/分布，`performance/*` 表示数量除以时间的速率。
有效监督 tokens 与参与计算的 input tokens 分开，MFU 使用后者及 segment 长度平方和。
不凭 pad/eos ID 猜测填充率；packing 只校验已有 offsets、位置和监督边界，不声称验证 attention kernel。

EP load 是 top-k assignment 次数，一个 token 可计入多个 expert。
逐层 `max_vio = max(counts) / mean(counts) - 1`；汇总取当前 stage 所持有层的最大值，不能平均掩盖异常。
DP/CP 独立 token 域求和，sequence-parallel TP 再求和，TP 副本不重复计数。
当前 adapter 要求全局 expert ID 顺序的完整向量；其他 dispatcher 或分片向量需专门适配。
原生 EP 的 `drop = attempted - accepted`，共享 dropless MoE 的 drop 为零；物理 padding 槽位无数据时标记 unavailable。

### MFU 标准模块方案

按 Linear、attention、GDN、Dense/routed/shared FFN、router、LM head 分块求和。
模型可声明 `monitor_flops_blocks`，块名唯一表示逻辑调用；冻结/LoRA 等需声明真实梯度倍率。
纯文本 Qwen3.6 的 GDN、带 output gate 的全注意力、routed/shared MoE 必须完整覆盖。
详细块公式及 20/40 层独立核算值只维护在 [MFU 完整性](../trainer_monitor_usage.md#mfu-完整性)。

```text
T = sum(input_sequence_lengths)
Q = sum(input_sequence_lengths²)
TFLOP/s/device = F_step / (step_seconds × device_count × 10¹²)
MFU_percent = 100 × F_step / (step_seconds × device_count × dense_peak_FLOPs_per_second)
window_MFU_percent = 100 × sum(F_step) / (sum(step_seconds) × device_count × dense_peak)
```

标准训练块使用 forward×3、causal attention 三角近似；不将重计算、通信或 optimizer 工作加入 MFU 分子。
峰值由部署提供真实 SKU/dtype/dense 来源；缺峰值只输出完整模型 TFLOP/s 和 `unavailable_peak`，
缺模型块则不输出部分模型的 TFLOP/s/MFU。异构设备和未知混合精度不套一个猜测峰值。

下表保留 2026-11-05 调研结论，不作为当前上游版本的新核验：

| 框架 / 原调研来源 | 算法与借鉴 | 限制 |
| --- | --- | --- |
| [Megatron-LM，31fbdf03](https://github.com/NVIDIA/Megatron-LM/blob/31fbdf03643f8dc7aaa0671b3587cca558bb48ad/megatron/training/training.py) | attention、FFN、logits 分项；训练矩阵乘 forward×3；按窗口时间和设备数归一化 | TFLOP/s 本身不是 MFU；GDN/MLA/MTP 需专用项 |
| [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/training/utils/flop_utils.py) | 运行时长度和、平方和；支持模型 estimator；GDN 标准估算 | 不能用平均长度平方替代平方和 |
| [TorchTitan，历史索引](https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/metrics.py) | 模型 FLOPs/token × 每设备吞吐 / 峰值 | 原调研旧路径 404，仅保留方法参考；量化不能盲套 BF16 峰值 |
| [VeOmni，617cc0fb](https://github.com/ByteDance-Seed/VeOmni/blob/617cc0fb7d7af70c07f4507fc31d871f4eef55d0/veomni/utils/count_flops.py) | 模型族结构公式 + 实际输入长度；区分 routed/shared expert | 不沿用未知设备 infinity 或不支持模型返回零的缺失值语义 |

## 3. 输出前后与渠道边界

改前：tqdm 展示摘要，逐步日志再打印同一组值；重定向或多 rank 时容易混杂。

```text
Training: 15%|...| 3/20 [... loss=10.5977, tokens/s=993.369] | step=3 epoch=0 ...
```

改后：基础日志只有一个 `train_step`，诊断事件以 event 类型区分。下例省略 run ID 等公共字段：

```json
{"event":"train_step","rank":0,"step":3,"epoch":0,"metrics":{"training/total_loss":10.5977,"performance/tokens_per_second":993.369}}
{"event":"local_micro_loss","rank":0,"step":3,"micro_step":0,"name":"foundation_loss","values":[10.6,1600]}
{"event":"timing","rank":0,"step":3,"phase":"checkpoint_persist","clock":"host_wall","duration_seconds":2.1,"outcome":"ok"}
```

| 渠道 | 输出约定 |
| --- | --- |
| Console | global rank 0 打印基础 step / 阶段计时；无进度条 |
| JSONL | `<output_dir>/<run_id>/rank_<rank>.jsonl` 保存完整标量、分布与诊断原因 |
| W&B | 显式启用，原生标量；仅 token Top-N 和 expert 分布转 Table，同一事件一次 `run.log`；普通诊断明细留 JSONL |
| TensorBoard | 原生 scalar，原始 step；使用与 W&B 一致的标量 |

所有记录带 schema/run/rank/event/step；异步到达仍保留采样 step，不拿上一采样值冒充当前值。
远端仅 rank 0 写其已有数据；不隐式汇集其他 rank 的局部细节。local key 标明 rank/module。
W&B Table 保留 step/rank/module/expert 或 token ID，不能用无身份的直方图代替。
W&B offline SDK run 可 `wandb sync`；JSONL 不是 SDK 离线文件，不能直接当作 run 上传。

输出线程队列有界（256 条），可选 sink 独立容错，首次故障告警，满队列显式计数，退出等待最多 5 秒。
完整记录先落 JSONL；不能因 SDK 故障改变训练。逐层 Table 仍有开销，精简后需要重新测量。

## 4. 验收与当前边界

- **单测**：配置约束、原始 step、计数/权重、采样、不修改 tensor、重计算去重、缺失值、队列就绪、输出格式及恢复。
- **训练回归**：真实 TextTrainer tiny MoE / GDN hybrid 各 20 steps；开关前后参数一致，输出频次和事件数量符合配置。
- **设备验收**：同模型、数据和并行配置，对照关闭 / JSONL / SDK；统计预热后 P50/P95、峰值显存和长跑趋势。
- **分布式验收**：所有 rank 参与顺序一致；计数不重复，CP/TP/EP 范围正确；多机只有 global rank 0 控制台输出。
- **checkpoint**：同步/异步 DCP、HF 权重导出及恢复；完成事件对应原提交 step，累计量与 run ID 连续。

已有单机 8 NPU 20-step 结果及 SDK 开销见 [设备验收报告](../trainer_monitor_npu_acceptance.md)。
该报告属于此前实现，不能替代本轮精简后的设备性能验收。
完整 40 层、多机、其他 TP/PP 拓扑、GPU、长跑和在线网络故障恢复仍需对应实测。
