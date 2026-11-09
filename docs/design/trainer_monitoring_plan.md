# 通用 Trainer 训练监控与日志改造方案

状态：设计草案，尚未实现。新增接口、文件名和配置字段为建议。
范围：`hyper_parallel/trainer`；需求来源为 [Issue #1174](https://github.com/mindspore-ai/hyper-parallel/issues/1174)。
基线：`upstream/master@c75f5f9f`，2026-11-05 的 Qwen3.6 MoE 纯文本 20-step 实跑。

## 1. 方案架构：增加什么、去掉什么

颜色表示职责分区，框内的【新增 / 改造】表示变更类型；各图使用相同配色。
**蓝色：配置与生命周期；绿色：现场采集；紫色：汇总与快照；橙色：输出；红色：移除项。**

**总架构：从采集到输出**

```mermaid
%%{init: {"themeVariables": {"fontSize": "22px"}, "flowchart": {"nodeSpacing": 25, "rankSpacing": 35}}}%%
flowchart TB
    subgraph CONFIG["配置与生命周期｜控制何时、在哪采集"]
        C["monitor 配置【新增】<br/>开关 / 采样频次 / module / sinks"]
    end
    subgraph COLLECT["现场采集｜只提供原始统计，详见下面分块"]
        H["计时 · loss / grad norm · 数据量<br/>ExpertLoadRecorder · Muon NS · FLOPs<br/>设备显存沿用现状"]
    end
    subgraph AGG["汇总层｜统一口径，不负责模型内部采集"]
        M["EnvironMeterCallback【改造】<br/>消费本步统计 / 累计量 / 吞吐与 MFU"]
        S["统一快照【新增】<br/>异步完成后就绪，保留原采样 step<br/>rank / module / 单位 / 数值 / 采样时机"]
        M --> S
    end
    subgraph OUTPUT["输出层｜消费同一份数据，不重复计算"]
        L["LoggingCallback【改造】<br/>响应外部日志触发 / 去重 / 渠道分发"]
        W["共享输出适配器【新增】<br/>有界异步队列 / 外部日志触发<br/>Console / JSONL / TensorBoard / WandB"]
        L --> W
    end
    C --> H --> M
    S --> L
    H -- "启动与 ckpt 阶段事件<br/>直接走共享输出" --> W
    style CONFIG fill:#e8f1ff,stroke:#3975bd,color:#162d4a
    style COLLECT fill:#e8f5eb,stroke:#3b8751,color:#173d25
    style AGG fill:#f1eafb,stroke:#8354ad,color:#39204f
    style OUTPUT fill:#fff1df,stroke:#bb792b,color:#573510
```

**分块 A：启动与阶段计时——在 callback 注册前也能记录**

```mermaid
%%{init: {"themeVariables": {"fontSize": "22px"}}}%%
flowchart TB
    subgraph A["蓝色 · 生命周期接入位置"]
        A1["组网 / 模型构建 / 数据集处理"]
        A2["预训练权重加载 / resume 恢复<br/>权重保存 / 完整 ckpt 保存"]
    end
    subgraph B["绿色 · 阶段计时器【新增】"]
        B1["阶段开始与结束配对<br/>阶段名 / 耗时 / rank / 成功或失败"]
    end
    subgraph D["橙色 · 阶段事件输出"]
        D1["共享输出适配器<br/>不伪装成 train_step，不依赖 callback 注册"]
    end
    A1 --> B1
    A2 --> B1 --> D1
    style A fill:#e8f1ff,stroke:#3975bd,color:#162d4a
    style B fill:#e8f5eb,stroke:#3b8751,color:#173d25
    style D fill:#fff1df,stroke:#bb792b,color:#573510
```

**分块 B：训练 step——现场采集与通用汇总分开**

```mermaid
%%{init: {"themeVariables": {"fontSize": "22px"}, "flowchart": {"rankSpacing": 30}}}%%
flowchart TB
    subgraph TRAIN["绿色 · 训练生命周期采集【增强】"]
        T1["取 batch / collate / H2D<br/>耗时、样本数、各口径 token 数"]
        T2["forward / backward / micro-step<br/>耗时、total / foundation / aux / local loss"]
        T3["规约完成后、裁剪前<br/>local shard / module grad norm"]
        T4["optimizer / step 结束<br/>更新耗时、lr<br/>设备显存沿用现状"]
        T1 --> T2 --> T3 --> T4
    end
    subgraph METER["紫色 · step 汇总【改造】"]
        M1["EnvironMeterCallback<br/>汇合本步采集结果与 recorder 结果"]
        M2["累计 samples / tokens、吞吐<br/>标准模块 FLOPs 分项求和 + 设备峰值 → MFU"]
        M3["冻结本步快照 → LoggingCallback<br/>未采样指标省略，不沿用旧值"]
        M1 --> M2 --> M3
    end
    T4 -- "传递各阶段累计的统计" --> M1
    style TRAIN fill:#e8f5eb,stroke:#3b8751,color:#173d25
    style METER fill:#f1eafb,stroke:#8354ad,color:#39204f
```

图中梯度位置对应默认 `post_reduce_pre_clip`；可选 `pre_reduce` 必须另接规约前位置并标明口径。
local loss 直接复用训练流程产生的 loss，不要求用户指定 module 路径。

**分块 C：模型与优化器内部——通过 recorder 暴露统计**

```mermaid
%%{init: {"themeVariables": {"fontSize": "22px"}}}%%
flowchart TB
    subgraph SOURCE["蓝色 · 内部接入位置"]
        E["MoE 路由 / dispatch<br/>已注册 module 与并行布局"]
        N["Muon 优化器<br/>NS 输出位置"]
    end
    subgraph REC["绿色 · 专用采集【新增】"]
        ER["ExpertLoadRecorder<br/>expert count / drop / 分布<br/>采样、重计算去重、布局感知归约"]
        NR["NS norm 采集<br/>按频次与参数范围统计"]
    end
    subgraph RESULT["紫色 · 本步汇总接口"]
        R["finalize 后交给 EnvironMeterCallback<br/>逐层 max_vio / 汇总负载 / norm<br/>消费后清理采样缓冲"]
    end
    E --> ER --> R
    N --> NR --> R
    style SOURCE fill:#e8f1ff,stroke:#3975bd,color:#162d4a
    style REC fill:#e8f5eb,stroke:#3b8751,color:#173d25
    style RESULT fill:#f1eafb,stroke:#8354ad,color:#39204f
```

Recorder 不负责打印，不持有完整 activation、router logits 或梯度副本。
local 统计不新增通信；global EP 统计的参与进程按并行布局确定，不能按打印 rank 过滤。

```mermaid
%%{init: {"themeVariables": {"fontSize": "22px"}}}%%
flowchart TB
    subgraph REMOVE["红色 · 本次移除与不建设范围"]
        X["删除 TqdmCallback / _LogProgressBar / 拼接输出<br/>不迁移 ETA、进度比例等字段"]
        Y["不建设独立 Debug 子系统<br/>既有安全检查保持原职责"]
    end
    style REMOVE fill:#fdebed,stroke:#bd4a57,color:#65242c
```

### 1.1 增删清单

| 动作 | 对象 | 本次方案的变化 |
| --- | --- | --- |
| 删除 | `TqdmCallback`、`_LogProgressBar` | 移除 Trainer 注册、导出、write 调用和对应测试；不影响其他组件使用 tqdm |
| 不迁移 | elapsed、ETA、完成比例、平均速度、额外 total_steps | 保留既有 step、epoch，不建设 progress 指标分类 |
| 改造 | `EnvironMeterCallback` | 统一 step 指标出口，接入 recorder/计时结果；不承载所有现场采集逻辑 |
| 改造 | `LoggingCallback` | 直接分发结构化记录；router aux loss 移到共享快照，不再由日志独享 |
| 增加 | 启动/分块计时器 | 分布式组网、模型构建、ckpt 加载、数据处理、训练块与保存耗时 |
| 增加 | local loss / module grad norm 采集 | 独立开关、采样频次、全 rank 采集；module 筛选仅用于 grad norm |
| 增加 | `ExpertLoadRecorder` | 逐层 expert load、max_vio、可用的 drop 与分布 |
| 增加 | Muon NS 采集 | 在 NS 输出位置记录 norm，不改变更新算法 |
| 扩展 | 数据、FLOPs 统计 | 计数语义、续训累计量、模型适配和可信 MFU |
| 沿用 | 设备显存 | 保持现有采集、字段和统计口径 |
| 增加 | JSONL、TensorBoard、WandB 输出适配 | 消费同一份指标，按 run/step 管理生命周期 |
| 移出本方案 | Debug 子系统及新异常处置流程 | 不删除原有数据合法性、NaN/Inf、FSDP 等安全检查；首步渲染保留为 `monitor.datasets` 可选数据输出 |

### 1.2 共同实现边界

- 不改 loss 归一化、并行算法、MoE 路由、expert bias 更新或优化器数学语义。
- 不建设完整 profiler、checkpoint 或高可用系统；仅复用 checkpoint 扩展保存监控累计状态。
- local 指标默认不增加通信；global 指标说明通信组并优先复用已有结果。
- 关闭新增监控后，不增加其采集 hook、缓冲、统计通信或设备同步。
- **训练流程中的 Recorder 不影响训练性能及显存：**作为实现与验收约束，关闭或未采样时不执行额外 tensor 计算、拷贝、统计通信或设备同步；启用采样后优先复用已有统计，采用有界缓冲，不滞留计算图或大 tensor。以同配置对照实测确认吞吐、step 耗时及峰值显存无超出测量波动的退化，且长跑显存不持续增长；未达标的采集项必须优化或降采样后重新验收，不能仅凭 Recorder 封装宣称零开销。
- 不保留 batch、大 activation、router logits、完整梯度副本或计算图。
- 字段和配置兼容迁移必须显式，缺少指标不伪造零值。
- **Hook 必须计入开销预算：**优先复用已有生命周期/计数，确需 hook 时只注册必要采集点。未采样时的 Python 回调分派、GIL、图编译 graph break 和重计算重复触发也需要对照测量，不能把“没有 tensor 运算”当作零开销。
- **允许异步采集与输出：**在 tensor 有效期间生成有界摘要；异步 D2H、CPU 汇总/落盘可以与训练重叠，但必须满足 stream/event 与缓冲生命周期约束，不改变训练同步语义。方案与降级边界见第 2.1 节。

### 1.3 最终 YAML 配置形态（设计草案）

以下展示改造后训练 YAML 中的 `monitor` 配置段，其他模型、数据和并行配置保持各自入口。
这是拟实现的配置接口，尚不能直接用于当前 master；可选的细粒度采集默认关闭。
不再提供 tqdm 或独立 debug 配置。

```yaml
monitor:
  enabled: true
  sinks: [console, jsonl]  # 可选 tensorboard、wandb；不默认上传
  output_dir: ./outputs/metrics
  timing: true
  mfu: false  # 标准模块分块估算；已有 log_tflops 保持原入口

  wandb:  # 仅 sinks 包含 wandb 时生效
    project: hyper-parallel
    mode: online

  local_loss:
    enabled: false
    sampler_steps: 100

  local_grad_norm:
    enabled: false
    sampler_steps: 100
    modules: []  # 空列表统计整体；可选规范 module 路径，包含其子模块参数

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
```

精简原则：只暴露功能开关、采样间隔和必要的筛选/输出参数，不把内部生命周期作为配置。
开启的指标由所有实际持有相应数据的 rank 采集，不重复配置 `ranks`；PP 不持有 loss 的 stage 不制造 loss。
local loss 固定记录采样 step 的全部 micro-batch 及整步汇总；local norm 固定 L2、规约后裁剪前。
module 子树参数按身份去重，包含该子树中的 expert 参数；不提供 `include_children`、`include_experts`。
MoE 固定采集全部已注册层的本地计数，按布局形成逐层全局结果，生命周期自动清理缓冲；
不暴露 recorder 类型、scope、per_layer、flush 或子模块递归开关。输出记录仍保留 scope/rank 元数据。
WandB 的 entity/run ID/resume 等复用既有外部实验管理入口，不在 monitor 重复定义。

## 2. 分模块设计

每个模块按“监控功能 → 接入与边界 → 验收”描述。
文件定位以基线代码为准；新增模块路径是候选，不表示文件已经存在。

### 2.1 统一采集、汇总与配置

**监控功能**

统一采样计划、指标元数据和 step 快照，供所有输出渠道使用。
每项定义单位、分母、local/global、通信组、采集时机及可用性。
各 Recorder 的 `sampler_steps` 控制可降采样统计；日志输出时机由外部 Trainer 的既有配置与生命周期控制。`monitor` 不再提供顶层采样频次或日志频次开关；累计消费量不能因降采样遗漏。

**接入与边界**

- 配置入口：`trainer/config/`；Trainer 生命周期：`trainer/base.py`、`text_trainer.py`、`vlm_trainer.py`。
- step 汇总：`trainer/callbacks/environ_meter_callback.py`；复用现有字典或引入只读快照数据契约。
- recorder 在相应执行位置采集，必须先 finalize，再由 meter 汇总，最后 logging 消费。
- 新数据契约、采样计划可置于候选 `trainer/runtime/monitor/`，不把模型实现导入通用日志模块。
- 启动事件不伪装成 step 0；未采样值不沿用上一轮结果冒充当前值。
- 所有持有指标的 rank 参与采集；module 选择不改变必要通信的参与集合。global rank 0 是汇总输出角色，不是唯一采集者。

**验收**

- 测试 Recorder 开关、采样与外部日志触发的组合；重复触发同一记录时不重复输出。
- 快照消费者不互相修改数据，重复 finalize 不重复累计或通信。
- 关闭统计后无新增采集资源；旧配置映射和冲突报错有覆盖。
- step 计数和累计状态在续训后连续。

#### 训练内部 tensor 采集：统一 Recorder 机制

所有为监控而进入 forward、backward、路由、优化器或数据处理流程读取 tensor 的逻辑，
统一遵循 Recorder 的采样、生命周期与资源约束。`ExpertLoadRecorder` 是其中的 EP 专用实现，
不扩展成包含所有统计公式的单一类。共享采样计划、注册管理与快照协议；具体 recorder 保留各自采集时机和计算语义。

| 采集对象 | 专用 recorder（候选名称） | 现场与保留数据 |
| --- | --- | --- |
| local / micro loss | `LossRecorder` | loss 产生处；只保留脱离计算图的标量统计 |
| local / module grad norm | `GradNormRecorder` | 本期规约后裁剪前显式采集；复用已有 norm 或按需计算，不复制完整梯度 |
| EP expert load | `ExpertLoadRecorder` | router/dispatch；只保留 expert 计数小向量 |
| Muon NS norm | `OptimizerStatsRecorder` | NS 输出现场；计算统计后立即释放中间矩阵引用 |
| token 长度、重复度、Top-N、模板样本 | `DatasetStatsRecorder` | 数据处理/batch 现场；有界摘要或限长样本，不缓存完整 batch |

普通 CPU 阶段计时、设备内存 API 查询、配置驱动的 FLOPs 估算不必强行套 tensor recorder；
它们复用统一快照出口。当前计时方案仅提供 host 墙钟耗时，不加入设备 event 采集配置。

**共用执行边界**

1. 初始化时解析 module 选择并注册必要采集点；所有拥有该指标的 rank 按一致计划采集，关闭时不注册额外 hook 或分配统计 buffer。
2. step 开始按 `sampler_steps` 决定是否采集，在训练现场执行新增 tensor 运算之前检查采样开关；未采样时不计算后丢弃。
3. 优先复用训练已有统计；必要时在现场计算小型摘要。`detach()` 仅断开计算图，仍可能共享大 tensor 的存储，不能据此允许长期保存切片或原 tensor。
4. 热路径不逐 micro-step/layer 执行 `.item()`、`.cpu()` 或打印设备 tensor。按既定完成边界批量取得摘要；如需跨流读取，明确 event/wait，不能读未完成结果。
5. global 统计明确通信组、顺序与参与者；输出 rank 筛选不能改变集合通信参与条件。local 统计不额外通信。
6. finalize 只生成一次快照；交给输出层的是 CPU 数值或有界结构，不把设备 tensor 放进日志/上传队列。消费、异常退出和 close 均释放引用。

统一 Recorder 能让开销可控、可验证，不能保证开启后零开销。norm、Top-N、全局归约和设备到主机读取仍可能增加计算、临时显存或同步等待。
采集点需逐个接入真实生命周期；不通过全模型通用 hook 自动推断所有 tensor 的含义，也不默认抓取 activation。

**共同验收**

- 对比关闭、开启但未命中采样、命中采样三种状态；分别测稳态 step 耗时、采样步耗时、峰值显存和长跑显存趋势。
- 检查关闭/未采样时没有新增 tensor 归约、主机拷贝、统计通信或设备同步；开启后的临时缓冲有明确大小上限。
- 覆盖重计算、梯度累积、PP micro-batch、重复 finalize、异常中止与续训；无重复采集、计算图滞留或失效 tensor 引用。
- 先独立验收每个 recorder，再验证组合开关；以相同训练配置对照 loss/梯度/更新结果，允许范围沿用训练数值精度标准。

#### 采集接入清单：显式调用与 Hook 的分工

本期主路径采用显式采集调用，复用已有 Trainer callback 生命周期做汇总与发布。
这里的“显式”指在明确语义位置调用 recorder/计时器；`on_micro_step_end` 等已有 callback 不是新增的 PyTorch tensor/module hook。
不会为了支持 module 筛选，就给每层都装 forward/backward hook。

| 监控部分 | 本期接入方式 | 具体位置与边界 |
| --- | --- | --- |
| 启动组网、模型构建、数据初始化 | **显式计时** | Trainer `_setup()` 的对应构建调用前后；独立阶段事件可早于 callback 注册，不用 module hook 反推启动耗时 |
| 权重加载/保存、resume、完整 checkpoint | **显式计时** | 真实 loader/saver 调用边界；异步保存从已有完成回调/future 取得完成时刻，不把提交返回视为完成 |
| batch_fetch、transform、collate、H2D | **显式计时/metadata 上报** | 主进程取 batch/H2D 及实际执行的 transform/collate；worker 侧回传小型阶段摘要，不在模型 forward hook 中猜测数据耗时 |
| forward、backward、clip、optimizer、scheduler 耗时 | **显式计时** | `text_trainer.py`/`base.py` 的对应调用前后；无法拆开的 PP/编译联合路径记录组合范围，不通过逐层 hook 强行拆分 |
| local / micro loss | **显式 recorder 调用** | `BaseTrainer.postforward()` 中 `local_loss = self.loss_fn(...)` 之后、`mean_global_loss()` 加权/规约之前；携带有效分母与 micro-step，不在 `outputs` 上装 hook，也不重算 CE |
| global loss、lr、训练使用的 grad norm | **复用已有结果，显式汇总** | loss 聚合返回值、optimizer/scheduler 状态与 `prepare_optimizer_step()` 返回的裁剪前 norm；不新增 tensor hook，不重复规约同语义指标 |
| local / module grad norm | **显式 recorder 调用** | 所有 micro-batch backward 及必要梯度通信完成后、实际裁剪/更新前；初始化解析 module 参数集合，运行时只读这些梯度，不注册逐参数梯度 hook |
| EP 负载 | **显式 recorder 调用** | model/router adapter 在路由计数或 dispatch 接收计数产生后立即上报；源端 attempted/drop、接收端 accepted/padding 分别标注。普通 MoE 层输出 hook 无法可靠取得内部计数，因此不作为主路径 |
| Muon NS norm | **显式 recorder 调用** | `core/optimizer/muon.py` 的 NS 输出已生成、尚未按学习率等缩放或应用更新的位置；兼顾默认与自定义 `zeropower_fn`，保留参数/批处理映射。optimizer step 后的 hook 已无法取得这个语义时点 |
| 数据长度/Top-N/packing/边界校验/模板 | **显式 recorder 调用** | CPU transform/collate 或 batch 适配处读取已有输入与 packing metadata；不为统计给模型装 hook，也不拷回完整设备 batch |
| MFU 标准模块清单与动态几何 | **显式注册与更新为主；限定的 forward-pre-hook 为可选补充** | adapter 初始化声明块组成/静态维度，运行时由 batch metadata 更新 token 几何。仅对无法从接口获得、且确需核对调用形状的白名单标准模块使用下述临时 hook |
| 设备显存 | **保持现状** | 沿用原有采集逻辑，不新增 hook 或采集入口 |
| 异步拷贝、汇总与输出 | **显式队列提交** | 上述位置生成摘要后交给 event/有界队列；后台 worker 只接收已就绪的摘要，不安装 hook 来延迟读取活跃 tensor |

**本期唯一预留的新增 hook：MFU 形状核对用的 `register_forward_pre_hook`。**

- 它是受限的兼容补充，不是所有模型默认安装。只有 adapter 无法提供必要 metadata，且已验证该模块通过正常 module 调用入口执行时才使用；编译/fused 路径不保证 Python hook 可见。
- 只读取 Python 可见的 shape、dtype、调用身份等 metadata，不访问 tensor 数值、不 `.item()`、不保留输入引用，不在 hook 中做通信或 FLOPs tensor 运算。
- 在真实训练调用中核对，不额外运行一遍 forward。固定结构确认后移除；重计算调用单独标记，不能计入标准 MFU 分子。
- 临时核对不能代表后续动态执行：若 shape/分支/调用次数会变化，必须由运行时 adapter 每步提供准确 metadata。无法提供时相关块标记 incomplete，不能沿用首步值或只按采样步推算完整 MFU。
- 安装/移除会触发重编译或该路径不支持 hook 时，不使用此补充；要求显式 adapter 接入或标记不支持。未通过空 hook 与实际 hook 的性能对照，不进入默认方案。

**梯度接入的额外边界：**普通 `.register_hook()` 的触发时点并不保证是“整步累积完成且所有规约结束”，因此不用它实现本期 local norm。
FSDP/main-grad/external optimizer 路径须从实际用于更新的梯度存储中读取，并由该路径提供明确的完成边界；不能假定统一读取 `.grad` 就覆盖所有模型。
未来若实现 pre-reduce norm，另在实际规约入口设计显式接入，本期不额外注册 backward/full-backward hook。
既有 `bind_aux_loss_scale()` 的梯度 hook 服务于训练数学语义，保持原行为，不复用成监控 hook，也不把它算作本次新增采集。

**组件依赖边界：**Trainer 持有 recorder；模型 adapter、EP 和优化器仅接受可选的轻量统计接口，由构建阶段注入。
核心组件不反向导入 `trainer`，也不直接调用 LoggingCallback/WandB；采样判断在新增 norm/统计计算之前完成，未启用时无额外统计 buffer。
接口名和插入位置是待实现设计；需按目标 revision 核对具体函数，不能据此宣称已接入。

**接入验收：**逐项提供实际插入位置、调用次数、采样条件与资源持有时间；显式路径和 hook 路径分别核对数值、额外 kernel/通信、CPU 开销和显存。
特别覆盖多 micro-batch、activation checkpoint、共享模块多次调用、graph compiler 及 FSDP main-grad；关闭功能后无残留 hook 或 recorder 资源。

#### Hook 与异步采集的实现边界

具体方式以“采集接入清单”为准：本期只有 MFU 形状核对预留临时 forward-pre-hook，其余均显式接入；不引入通用的“拿不到 tensor 就加 hook”回退。
hook 本体只校验采样状态并产生必要摘要，不打印、序列化、调用网络或执行同步设备取值。
能在 step 边界安全启停时，未采样步不安装 hook；需常驻的 hook 必须单独测空回调成本，且关闭功能时彻底移除。
forward/backward hook 的调用次数、autograd 语义、重计算身份和 torch.compile/graph capture 兼容性均需验证；不支持时明确报告，不能静默切回 eager。

候选异步链路为：

```text
训练流产生 tensor → 在有效期内提取/写入小型摘要 → 记录 ready event
→ 采集流等待 event → 异步 D2H 到有界 pinned buffer → 记录 copy-complete event
→ 完成后后台 CPU worker 汇总/序列化 → 外部输出触发后发布
```

- 异步不是延迟读取任意 tensor。梯度可能被 optimizer 修改、activation 可能被释放；必须在语义时点完成摘要读取或按流依赖固定一个小快照。跨流直接读取训练 tensor 只有在可证明其生命周期与后续写入安全时使用，不默认保留大 tensor。
- D2H 是否真正异步取决于设备后端与 pinned memory 支持；接口用 `record_event`/wait/query 与可用异步拷贝能力组合，不能在 CPU 读完成前的数据。CPU-only 统计直接进入 worker，不引入设备 event。
- buffer 固定槽位/字节上限，每个槽位携带 run、step、micro-step、module、rank；在途槽位不得清零或复用。仅保存有界统计，设备与 pinned-host 内存都纳入预算。
- global 统计的 collective 按统一计划和固定顺序发起，持有异步 work 直到完成；不能由各 rank 的后台线程按本地队列是否空闲随意发起通信。容量不足时，仅可在统一决定后跳过可选的全局采样；已发起通信必须完成。
- 输出晚于训练时保留真实采样 step，使用独立采样事件，不伪装为最新 step，也不重复常规 train_step。输出触发时未 ready 的可选记录延后，不为等指标阻塞主训练路径。
- 队列满时跳过后续可选详细采样并记录 dropped/pending 计数；基础 loss、token/sample 累计量和训练正确性检查不可因背压丢失。异步 worker 不承载训练必须立即执行的错误检查。
- 关闭/退出采用有界 drain，未完成记录明确标记 incomplete；不把 pending 记录当作零。设备不支持异步时报告该路径不可用，可采用已满足开销预算的低频同步摘要，不宣称异步。

验收至少比较：无 hook、常驻空 hook、同步摘要、异步摘要四种模式；记录 hook 调用次数与 CPU 开销、graph break、训练 step P50/P95、设备/pinned-host 峰值、队列峰值与采样丢弃率。
覆盖慢消费者、异步输出乱序、缓冲复用、异常退出、重计算和多 rank collective 顺序；所有指标值与同步参考一致。

### 2.2 日志输出与 WandB

**监控功能**

Console、JSONL、TensorBoard、WandB 使用相同字段与数值。
常规汇总仅 global rank 0 输出；所有 rank 的 local/module 记录按 rank 独立保存，避免每个 rank 都打印到共享终端。
交互和非交互终端均采用结构化 step 行，不保留进度条。

**接入与边界**

- 改造 `trainer/callbacks/logging_callback.py`；删除 `tqdm_callback.py` 及 callback exports/注册/调用。
- 输出适配器放在独立 runtime 模块，供启动期和 LoggingCallback 共用。
- JSONL 使用 schema_version、run_id、event、rank；step/epoch/module 仅在适用时携带。
- 同一 step 指标合并上报一次。WandB 直接接收指标字典，不使用日志正则解析。
- 远端输出只有一个 run；支持显式在线/离线、恢复 run_id、有界重试和退出 flush/finish。
- 上传失败告警并保留本地记录，不无限阻塞训练或增长内存。详细 rank 数据上传如需通信必须显式启用。
- 对请求但不支持的指标只解释一次；不开启的指标不逐项刷缺失告警。
- 具体输出协议与改前/改后日志见第 3 节；远端细则见附录 E。

**验收**

- 外部每步触发输出时，20 steps 恰好 20 条 train_step，无 tqdm、回车刷新和重复摘要。
- 外部每 5 步触发输出时，输出第 5、10、15、20 步；最终记录是否触发遵循外部生命周期，monitor 不另设补发策略。
- 两节点以上确认仅 global rank 0 输出汇总；不是每台 local rank 0 各输出一份。
- Console、JSONL、TensorBoard、WandB 同名指标一致，展示舍入不影响原始记录。
- 验证断网、离线、恢复、重复 step、队列上限和正常退出；可选依赖未安装时行为明确。

### 2.3 分块耗时

统一由 `monitor.timing: true/false` 控制，不再分别设置 startup、step、checkpoint 开关。
启用后记录实际执行的各阶段；未执行的阶段不产生耗时指标。

**监控功能**

| 阶段 | 监控内容 |
| --- | --- |
| 启动组网 | rendezvous/默认进程组、DeviceMesh 与子组初始化 |
| 模型初始化 | 构建模块树、预训练权重读取/转换/加载、并行替换与包装 |
| 数据初始化 | tokenizer/assets、数据集读取/映射、dataloader 构建 |
| 续训恢复 | checkpoint 的模型、optimizer、scheduler、RNG、dataloader 状态加载 |
| Step 数据 | 获取 batch、预处理、collate、H2D |
| Step 计算 | 前向/反向、梯度裁剪、optimizer、scheduler |
| 权重保存 | 模型权重导出/保存时长，包括必要的权重收集、转换、序列化与写入完成 |
| 完整 checkpoint 保存 | 模型及 optimizer、scheduler 等续训状态的整体保存时长；区分同步保存与异步提交/完成 |

**接入与边界**

- 在 Trainer 初始化入口创建轻量计时器，早于 `_setup()` 和 `_init_callbacks()`。
- 启动阶段接入 `base.py` 的构建调用和模型 checkpoint loader；恢复/保存接入 `callbacks/checkpoint_callback.py` 的真实调用边界。
- 数据时间在 `data/` 的 transform、collate、batch 获取/H2D 实际执行位置采集。
- step 计算时间围绕 text/VLM 训练路径及优化器调用采集，不假设所有 PP 路径都能拆开 forward/backward。
- CPU 墙钟使用单调时钟；统计 host 调用耗时；不将异步设备任务的提交耗时称为设备执行时间。
- 不为每块增加全设备同步；嵌套、重叠和不同 rank 的 max 不能相加解释成关键路径。
- 权重加载与续训恢复分开，未执行不填零；无法拆分的 from_pretrained 路径只报可测父阶段。
- 权重保存计时接在实际权重导出/保存入口，覆盖定期保存与训练结束导出；复用 `monitor.timing` 开关。完整 checkpoint 内可拆分的权重保存作为子阶段，不与父阶段重复相加；无法拆分则只报整体保存耗时。
- worker 的预处理耗时和主进程等待时间分开。异步保存用 checkpoint_id 关联提交和完成；完成时长从保存请求开始计到保存接口确认写入完成，不把提交返回当作保存完成。
- 详细字段及计时边界见附录 D。

**验收**

- 受控延时验证计时边界、嵌套关系、次数与本地累计结果。
- 启动 phase_start/end 能配对；初始化失败时保留已获得阶段信息。
- 验证不加载、预训练加载、续训恢复三条路径及同步/异步保存。
- 覆盖仅保存权重、完整 checkpoint、训练结束导出；验证权重保存边界包含实际写入，异步提交与完成可关联，失败记录状态且不误报成功。
- 多 worker 下不把 batch_fetch 冒充 preprocess；计时不引入额外设备同步。
- 实跑比较监控前后稳态耗时，不以 host 提交时间宣称设备执行时间。

### 2.4 Loss、梯度与 module 监控

**监控功能**

- 总 loss、分项 loss、学习率、训练链路使用的 global grad norm。
- 可配置 local loss、micro loss 和 module local grad norm。
- local loss/norm 独立启用，各自设置采样频次；所有持有对应数据的 rank 采集，module 路径仅用于 grad norm。

**接入与边界**

- global loss/norm 复用训练循环和裁剪结果；meter 不再为同一语义重复求均值或重算 norm。
- local loss 在 loss provider 跨 rank 规约前取得，保留分子、分母及 scale。
- local loss 由训练 loss 产生位置直接采集，不提供 module 路径配置，也不构造逐层 loss。
- local step loss 使用全部 micro-batch 分子分母，不以采样 micro loss 均值代替。
- local norm 固定在规约后、裁剪前读取本地梯度分片；pre_reduce 不属于本次精简配置，只保留语义区别。
- 后续若扩展 pre_reduce，需在实际通信前接入并另行验证；本次不修改通信算法，也不复制完整梯度强行实现。
- 规范 module 名由模型映射到包装后的执行对象；PP stage 不拥有目标模块时不制造记录。
- 共享参数去重，明确 unscale、expert 参数和递归子模块语义；micro norm 不能相加充当 step norm。
- 完整字段、筛选规则和配置语义见附录 C。

**验收**

- 独立开关和采样频次有效，验证全 rank 覆盖；grad norm 的 module 选择生效且全局无匹配时明确报告，local loss 无需配置模块路径。
- local step loss 与全部真实 micro-batch 的独立分子分母统计一致。
- 验证 post-reduce 的采集时点、混精 unscale、共享参数、expert 参数、PP 所有权，不能标成 pre-reduce。
- global norm 与实际裁剪使用值一致；采集不改变梯度和更新结果。

### 2.5 数据量、数据质量与吞吐

数据相关的监控配置统一归入 `monitor.datasets`；配置分组不改变 `data/`、`performance/` 指标分类。

**监控功能**

- 每步与累计 input/valid token、padding/ignored 统计、全局逻辑样本。
- 长度分布、packing 边界/拼接数、token Top-N/重复度、可选首步渲染。
- input tokens/s 与 valid tokens/s 分开输出；前者对应输入规模，后者对应有效监督量。

| 配置 | 监控内容与输出 |
| --- | --- |
| `token_distribution` | 本采样 step 全部 micro-batch 中，每条输入序列的非 padding 长度与有效监督长度分别输出 min/max/mean、P50/P90/P99 和分桶计数；另报 padding、ignored 比例及其分母 |
| `top_n_tokens` | 统计非 padding 输入中每个 token ID 的出现次数，输出前 N 项的 ID、count、占非 padding input tokens 的比例；频次相同按 token ID 排序，默认不解码文本 |
| `packing_stats` | 输出 packing 是否启用、packed 行数、原始样本数、每行 segment 数分布、segment 长度分布和填充率；metadata 可用时增加截断样本数/丢弃 token 数 |
| `packing_boundary_check` | 按实际 packing 策略检查 segment 偏移、长度、position IDs、attention 分段与 shifted labels/监督 mask 的边界一致性；输出检查数、失败数、检查项和有界定位信息 |
| `log_first_rendered_template` | 仅 global rank 0 限长输出首步实际渲染结果；一次性输出，不受周期采样漏过首步 |

`sampler_steps` 控制这些详细统计与边界抽查，不影响基础 token/sample 累计量逐步更新。
“本 step”只指命中采样的 optimizer step，不将前 100 步混成该步分布；每份记录附 step、rank、scope 和采样样本数。
长度分布按实际输入行统计；packing 前的 segment 长度单列，避免把 packed 行当成原始样本。
Top-N 使用 token ID 频次，和序列长度分布是两类统计；padding 按有效位置 mask 排除，不能仅按 pad token ID 排除。

**接入与边界**

- 数据 metadata 在 `data/` 生成；Trainer batch 接口携带小规模计数，meter 累计和计算吞吐。
- 有效监督计数复用 loss 实际使用的 shifted labels/分母；原始输入计数不能直接按 label mask 得出。
- input、padding、ignored、valid 不天然组成互斥集合，必须按字段定义。
- CP 切片求和与副本去重按数据布局处理，不能所有字段都统一除 CP size。
- packing 后的 batch 行数不等于原始样本数，缺少 metadata 则原始样本计数不可用。
- 只提交完成的训练消费；预取但未训练的 batch 不推进累计计数。
- 通过既有 checkpoint 扩展保存/恢复累计量，不新建 checkpoint 系统。
- Top-N/重复度限频统计；首步渲染通过 `monitor.datasets.log_first_rendered_template` 显式开启，固定 global rank 0、限制长度和次数；文本预览作为单份展示，不限制各 rank 的数值统计。
- 详细数据统计由 `DatasetStatsRecorder` 接入，优先消费 collate 后、H2D 前的 CPU 数据与 packing metadata，不为统计把完整设备 batch 拷回主机。
- Top-N 汇总采样 step 内全部 token 计数后再选取，不能只合并各 micro-batch 的 Top-N。计数结构按词表大小或显式容量约束，临时工作空间和样本缓冲需有上限；分位数采用有界直方图近似时必须标明近似口径。
- 所有数据持有 rank 输出各自本地分布，不标成全局分布；CP 切片前后范围需标识。未来全局 Top-N 需要完整计数归约或有误差说明的算法，不能拼接各 rank Top-N 冒充精确全局结果。
- packing 边界检查以实际策略为准：isolated 策略验证禁止跨 segment attention/监督；允许拼接的策略按既定语义验证，不强制 position 重置或一律隔离。检查 offsets 单调、起止合法、segment 覆盖与有效输入一致，并核对 loss shift 是否意外跨界。
- 不构造新的完整 S×S attention mask；通过 offsets/cu_seqlens、position IDs 和实际 mask 构建参数检查。只能验证 metadata 时标明覆盖范围，不能宣称已验证 kernel 行为。
- packing 未启用时标为 disabled；缺少 metadata 的检查标为 unavailable，不能当作通过或填零。抽查失败输出有界定位，不新增独立 Debug 流程，也不替代或关闭原有逐 batch 合法性检查。

**验收**

- 小样本手工复算 token、sample、shift、mask、packing 和 CP 去重。
- 外部每 5 步输出与每步输出时的累计量一致；恢复前后与 dataloader 游标一致。
- 覆盖全 ignored label、padding、重复 token、特殊 token、空输入的边界。
- 首步渲染只在 global rank 0 输出一次，反映模型实际输入，不默认输出全文。
- 用已知 token ID 频次和序列长度的多 micro-batch 数据复算分布与 Top-N；覆盖频次并列、N 大于唯一 token 数、pad/eos 同 ID 和空有效输入。
- 构造正确 packing 及 offsets 越界/重叠/缺失、错误 position、attention 配置不一致、shifted label 意外跨界等样本；确认按策略检测且报告检查覆盖范围。
- 分别验证 packing 关闭、metadata 缺失、截断和多 segment 情形，缺少的信息不伪造；详细统计关闭或未采样时无新增扫描、设备拷贝和通信。
- JSONL 保留分桶和 token ID/count 对应关系；WandB 显式启用时分布/Top-N 转 Table 或 Histogram，标量与本地记录一致。

### 2.6 EP 负载均衡：ExpertLoadRecorder

**监控功能**

逐层有效 expert load、max_vio、空 expert 数；模型可提供时增加 drop 和物理 padding 槽位。
按需输出完整分布，常规日志默认仅汇总标量。

**接入与边界**

- 新增 Trainer 持有的 `ExpertLoadRecorder`，候选位置 `trainer/runtime/monitor/`；遵循第 2.1 节的统一 Recorder 机制，专注 EP 统计。
- model/router adapter 注册规范 module、expert 全局 ID、所有权及计数副本/分片关系。
- Router/dispatch 提供已有小向量；一份 recorder 管理多个层，不为每层注册 callback。
- 生命周期：register → begin_step → record → finalize_step → meter 合并 → logging → reset。
- Top-k 负载默认是 assignment 次数，不是唯一 token 数；drop assignment 不能冒充 token drop。
- 所有层的持有 rank 采集本地计数；逐层全局结果按注册布局拼接/求和并去除路由副本，不能机械 all-reduce。不再让用户选择 local/global 模式。
- activation checkpoint 重计算不重复记录，共享模块多次真实调用不能漏计。
- 不介入既有 MoEMonitorCallback 的 expert bias 更新，不保存 logits 或 token→expert 大索引。
- finalize 幂等；缓冲固定规模并消费后清理。接口、统计公式和边界详见附录 F。

**验收**

- 用显式路由参考复算 load、max_vio、drop；覆盖 top-k、空 batch/expert、padding。
- EP 分片、重复路由副本和 DP/CP/TP/PP 组合统计正确。
- 重计算、共享模块调用、重复 finalize 不漏计、不重复累计或通信。
- 长训练缓冲不增长；关闭 recorder 无额外采集，开启后训练结果不变。

### 2.7 Muon NS norm

**监控功能**

NS 迭代输出矩阵 norm；可选迭代前 norm 与相对变化。按采样频次采集所有相关参数，保留 module/参数与 rank 身份。

**接入与边界**

- 在 `components/optim/` 下实际 Muon/NS 执行路径输出产生后、更新应用前采集。
- 模块参数选择在初始化时解析，传递稳定参数身份；不能靠逐步扫描模型树关联。
- 只记录 detached 标量，meter 合并；不保存 NS 中间矩阵，不改变迭代次数、精度和算法。
- 不新增 Debug 或自动异常处置流程；既有安全检查保持原有职责。
- 非 Muon 优化器启用该指标时明确不可用。

**验收**

- 小矩阵与独立 norm 参考一致，零输入、前后 norm 和相对变化分母规则明确。
- 采样频次及全 rank 覆盖有效；未启用时不执行新增 norm 计算。
- 开启前后参数更新一致，无矩阵引用残留和额外全局通信。

### 2.8 设备显存：沿用现状

设备显存不在本次改造范围内。保留现有采集逻辑、字段名、单位和统计口径，既有结果继续进入统一输出。
不新增 current 指标、跨 rank 聚合或窗口峰值，也不调整 peak 重置行为。
监控自身的显存开销仍作为 Recorder 和整体方案的验收项，见第 4.2 节。

### 2.9 FLOPs、TFLOP/s 与 MFU：按标准模块分块计算

`monitor.mfu` 只控制新增 MFU，默认关闭；已有 `training.log_tflops` 保持原入口。
基线 master 已有 step FLOPs 估算与 TFLOP/s 输出，但 estimator 仅支持 qwen3_moe。
本次确定采用“标准模块计算块 → 模块分项 → 整步求和”的统一 estimator，模型 adapter 只声明块组成、维度与特殊语义，不在通用 Trainer 持续增加 model_type 分支。
两个开关任一启用时共用一次估算；TFLOP/s 和 MFU 分别按各自开关输出，不重复计算。
使用既有整步时间，不依赖分块计时 `monitor.timing`。

**监控功能与计算口径**

每个标准块记录 block_type、规范 module、调用身份、输入几何、公式版本、forward/backward FLOPs 和 coverage。
分块是理论模型运算量的组成，不是把各模块实测耗时相加。`model_flops_per_step` 可作为诊断分项输出。

| 标准计算块 | 计数方法与边界 |
| --- | --- |
| Linear / LM head / router projection | GEMM forward 按 `2 × M × K × N`；全量训练且输入、权重梯度均需计算时 forward+backward 为 `6 × M × K × N`。冻结/LoRA 按真实梯度路径拆分，不统一乘 3 |
| Attention 核心 QKᵀ / AV | 按 Q/K 长度、query heads、head_dim 和有效 query-key 对数 A 计算，两次矩阵乘 forward 为 `4 × A × query_heads × head_dim`，backward 按梯度路径计入；causal-triangular/full-square 明确版本。QKV/O 投影由 Linear 块计数，禁止重复 |
| Dense FFN / SwiGLU | 作为组合节点，展开 gate/up/down（或实际两层）Linear；不另加一份父 FFN 总量 |
| Routed / shared expert FFN | routed 标准模型工作量按有效输入 token × top-k 分配到 expert 块；共享 expert 按实际声明计入。dispatch padding、drop、路由实际负载另报差异，不静默改变标准 MFU 分子 |
| GDN / MLA / 其他非标准计算 | adapter 注册可复用的计算块公式与几何；投影部分复用 Linear，其余核单独核算。未覆盖时标记 incomplete，不冒充普通 Attention |
| 重计算 / 通信 / optimizer / 轻量算子 | 重计算、通信、optimizer 不进入本方案标准 MFU 分子；norm、激活、RoPE 等是否忽略在公式版本中明确列出，不将未知大块归为“轻量忽略” |

```text
F_step = Σ（本步各逻辑调用的标准块 forward FLOPs + 必要 backward FLOPs）
TFLOP/s/device = F_step / (dt × device_count × 10^12)
MFU_percent = 100 × F_step / (dt × device_count × peak_flops_per_device)
```

**接入与边界**

- 在 `trainer/runtime/flops.py` 及候选子模块实现标准块 registry/聚合器；模型 adapter 生成计算清单，数据层提供实际输入几何。复用既有模型估算作为迁移对照，不另行双算。
- 初始化时建立块清单，固定形状项缓存；逐步仅更新 token 总量、各 segment 长度平方和等小型 metadata。优先无 hook，不能每步全模型遍历或为公式留存 tensor。
- 变长/packing Attention 使用与 mask 语义一致的 query-key 对数（分段平方和或明确的因果对数），不能用平均长度平方代替；被 loss mask 的 prompt 仍参与模型计算。
- 统计逻辑完整模型工作量。DP 汇总独立样本，TP/CP/EP 副本和分片按块所有权还原，PP 汇合不同层；同一块不因副本、重计算或父子递归重复计数。共享权重的多次真实调用分别计数，权重共享不等于计算只发生一次。
- fused 替换前后使用同一逻辑计算块定义；adapter 明确 composite 与子块互斥的计数所有权，不能 fused 父块与内部 Linear 同时记账。
- MFU 分子默认标准模型有用运算量；MoE padding/drop 等实际执行差异作为单独诊断数据，跨框架对比先统一此口径。
- meter 只做求和、时间和设备数归一化，输出层不算 MFU。记录公式版本、块覆盖率、忽略项以及峰值的 SKU/dtype/dense-sparse/来源。
- 必要块未覆盖时可输出已知分项，但整体 MFU/TFLOP/s 标记不可用；峰值缺失时仅 MFU 不可用。不能把部分和伪装成完整模型工作量，也不输出假零。
- 窗口指标用 sum(F)/sum(dt)，不纳入组网、加载/保存或以各分块耗时相加代替 dt。混合精度/异构峰值无可信统一约定时不报 MFU。
- 框架算法调研保留在附录 B，作为分块公式核对来源；实施方案固定为本节的标准模块组合。

**验收**

- 独立手算 Linear、causal/full Attention、Dense/SwiGLU、routed/shared expert 分项，并验证组合总量等于唯一计数块之和。
- 覆盖可变长度/packing、共享模块重复调用、fused 替换、重计算、冻结/LoRA；公式不遗漏也不重复计数。
- 对同一逻辑 batch、模型和公式版本切换 DP/TP/CP/EP/PP，F_step 应一致；设备数归一化只做一次。
- Qwen 混合 Attention/GDN 核对逐层清单，未知块明确 incomplete；与旧 qwen3_moe estimator 在相同公式约定下交叉复算。
- 验证未知峰值、混合精度和不完整覆盖；启用 estimator 的 hook/遍历/metadata 开销纳入第 4.2 节验收。

### 2.10 指标诊断文档与 Skill

各项指标不仅定义计算方法，还必须说明“能观察什么问题、应联查什么证据、哪些结论不能单独推出”。
完整映射维护在 [训练监控指标诊断说明](../trainer_monitor_diagnostics.md)，覆盖 loss、norm、数据、EP、Muon、分块耗时、既有显存和 MFU。
该说明是诊断语义的单一来源；新增/改名指标时同步更新字段定义与该映射，不把指标升降直接判成训练故障。

[模型精度验证 skill](../../.agent/skills/hf-model-precision-validation/SKILL.md) 引用该文档指导训练异常定位；
skill 先核对实现支持、统计范围和采样完整性，再按对齐的 step/rank/module 关联证据。
本次仍是设计阶段，文档中的新指标不代表已可运行，不能凭配置草案开启未实现采集。

验收：每项设计指标都能在诊断表找到用途与边界；用 loss 突升、EP 倾斜、packing 错位、MFU 下降四类案例复核诊断路径；缺失/异步丢弃指标不能被判成健康或零值。

## 3. 输出效果：改前与改后

### 3.1 改前

以下是实际第 3 步日志的节选，为阅读方便换行：

```text
Training: 15%|...| 3/20 [00:22<02:08, 7.54s/step,
loss=10.5977, grad_norm=51.1115, lr=9.510e-06,
foundation_loss=10.5977, tokens/s=993.369, step_time=3.297s]
| step=3 epoch=0 data/consumed_samples=24 data/consumed_tokens=7802
performance/step_time=3.29686 performance/tokens_per_second=993.369
training/total_loss=10.5977 ...
```

`7.54s/step` 是进度条启动以来的累计平均耗时，受首步预热影响；
`3.29686s` 是当前第 3 步耗时。两者不是同一指标。

### 3.2 改后：终端和文件统一结构化记录

交互与非交互终端采用相同规则，不创建动态进度条。
以下使用实跑值和建议字段名；为方便阅读换行，实际每条记录占一行：

```text
event=train_step step=3 epoch=0 rank=0
training/total_loss=10.5977 training/foundation_loss=10.5977
training/grad_norm=51.1115 training/lr=9.50953e-06
performance/step_time_s=3.29686 performance/valid_tokens_per_second=993.369
data/step_valid_tokens=3275 data/consumed_valid_tokens=7802
data/step_samples=8 data/consumed_samples=24
memory/device_max_allocated_gb=51.0571 memory/device_max_reserved_gb=53.7422
```

删除 tqdm 后不迁移其 elapsed、ETA、完成比例、平均速度或额外的 total_steps 字段。
沿用 LoggingCallback 的 step、epoch 和训练指标，不新增 progress 指标分类。
当前步耗时及启动/训练分块耗时继续保留，用于性能分析。

| 场景 | 规则 |
| --- | --- |
| 交互终端 / 重定向 | global rank 0 输出结构化 step 行，不输出 tqdm |
| Console 和文件同时启用 | 各写入一次同一记录，不通过抓取 stdout 生成文件 |
| 多机 | 仅 global rank 0 输出汇总；不按各节点 local rank 0 输出 |
| 全 rank 详细统计 | 各 rank 独立文件，携带真实 global rank；共享终端只打印汇总 |
| 最后一步未命中外部日志间隔 | 遵循外部是否触发最终记录的决定；monitor 仅去重和分发 |

### 3.3 分块耗时记录

阶段开始时立即记录 `phase_start`，完成时记录 `phase_end`，便于判断训练卡在哪个阶段。
以下是格式模板，占位符不代表实测值：

```text
event=phase_start phase=checkpoint_load rank=0 phase_id=<id>
event=phase_end phase=checkpoint_load rank=0 phase_id=<id> duration_s=<seconds> status=completed
event=phase_end phase=distributed_init rank=0 phase_id=<id> duration_s=<seconds> status=completed
```

嵌套阶段携带 `parent_phase_id`；失败时保存已获得的耗时和失败状态，不新增异常处置逻辑。
训练 step 的分段耗时使用 `performance/step/*_s`，启动阶段使用 `performance/startup/*_s`。

### 3.4 启动说明与不可用指标

启动时记录一次采样、输出和统计口径说明，例如：

```text
 event=monitor_config output_rank=0
 loss_scope=dp_cp token_basis=valid_labels
 sample_count=global_cp_deduplicated memory_unit=GiB
```

具体模型仍需提供准确的 loss 归一化、时间窗口和并行组定义，不能只依赖上述简写。
对用户明确请求、却无法提供的指标只解释一次：

```text
 event=metric_unavailable metric=performance/tflops_per_device
 reason=unsupported_model_type model_type=qwen3_5_moe_text
```

不伪造 `0`，不每步打印 `N/A`；未启用指标无需逐项报缺失。
JSONL 可保存结构化的不可用事件，TensorBoard 和 WandB 不写伪造数值。

## 4. 实施顺序与交付验收

| 批次 | 实现模块 | 交付证据 |
| --- | --- | --- |
| 1 | 2.1 统一契约、2.2 结构化日志及输出适配 | 删除 tqdm 的代码检查，改前/改后日志，Console/JSONL/WandB 一致性，多机 rank 输出验证 |
| 2 | 2.3 分块耗时、2.5 数据 | 启动/加载/训练阶段日志，数据计数复算与续训恢复 |
| 3 | 2.4 local/module、2.6 EP recorder、2.7 Muon | 独立参考结果，过滤/频次用例，EP 布局/重计算边界和数值不变验证 |
| 4 | 2.9 标准模块 FLOPs/MFU | 块注册与唯一计数清单、逐块及组合公式复算、并行布局不变性、缺失块/峰值示例 |

每批次对应模块的验收项必须落地，不以全量 UT 通过代替指标口径验证。
参考源码入口见附录 I；后续实现如调整模块位置，保持各模块职责边界不变。

### 4.1 集成验证

- 复用附录 A 的 20-step 配置，保存 command、revision、环境、退出码与原始指标文件。
- 仅输出改造时比较 loss、norm、lr、数据消费，确认训练语义不变。
- 增加真实多机验证，单机八卡不能证明各节点不重复输出。
- 验证保存/恢复、外部最终日志触发、远端故障、详细采样组合，不因 rank 分支导致死锁。

### 4.2 性能与显存验收

- 同配置比较关闭、基础监控、详细采样三种模式，排除预热后报告 step time 分布和峰值显存。
- 检查关闭功能无其额外 hook、buffer、统计通信和设备同步。
- 检查详细采样频率、批量规约和有界输出队列，长训练无 buffer 增长。
- 不保留完整梯度、activation、router logits 或计算图。
- 分别测无 hook、空 hook、同步采集与异步采集；检查 graph break、采样丢弃率、CPU worker 与 pinned-host 内存，不能只看设备峰值。
- 全 rank 采集需在多机上测聚合通信与文件/上传压力；每 rank 有界不等于全任务开销固定。
- 以无超出测量波动的性能/显存退化为目标，记录测量方法和波动区间；若固定缓冲等开销可测，明确报出并继续优化，不能宣称数学意义的零开销。

### 4.3 模块级待确认项

| 模块 | 实现前需固定 | 不允许的回退 |
| --- | --- | --- |
| Local norm | 各后端 pre_reduce 的可用粒度和采集点 | 用规约后的 .grad 冒充规约前数据 |
| 数据 | packed 行、原始样本、已完成消费的映射 | 缺 metadata 时伪造原始样本数 |
| 计时 | 每条路径的起止边界、host/device 与嵌套关系 | 用提交耗时冒充设备完成时间 |
| 输出 | schema 迁移、WandB resume 去重和有界重试 | 同名字段静默改变语义 |
| MFU | 标准块公式、组合所有权、并行去重与设备峰值版本 | 未覆盖块用 Dense 公式补齐或把部分和当完整值 |

## 附录 A：当前实现与实跑证据

### A.1 验证配置

| 项目 | 本次基线 |
| --- | --- |
| 设备 | 单机 8 × Ascend 910B3 |
| 模型 | Qwen3.6-35B-A3B，沿用 20 层模型视图 |
| 模型类型 | `qwen3_5_moe_text` |
| 训练 | 纯文本，20 个 optimizer steps |
| 并行 | CP=2、EP=2、TP=1、PP=1，沿用 recipe 的 FSDP 配置 |
| 数据 | `decif-30k.jsonl`，最大序列长度 4096，全局 batch size 8 |
| 环境 | Torch 2.9.0+cpu、torch_npu 2.9.0.post6、Transformers 5.13.0、CANN 9.1 |
| 指标开关 | `logging_steps=1`、`log_tflops=true` |
| 结果 | 20 步完成，退出码 0；每个 step 的结构化记录各出现一次 |

此次是单机验证，不能替代多机 global rank 输出验证。
完整服务器产物位于 `/home/chaoran/youchu_valid/trainer-metrics-master-c75f5f9f-20steps/`，
包括 `train.log`、`command.txt`、`environment.txt`、`revision.txt` 和 `exit_code.txt`。

### A.2 已输出指标

以下 12 个指标均在 20/20 步出现，解析结果均为有限数值。

| 指标 | 第 20 步 |
| --- | ---: |
| `training/total_loss` | 5.05664 |
| `training/foundation_loss` | 5.05664 |
| `training/grad_norm` | 10.3072 |
| `training/lr` | 1e-6 |
| `performance/step_time` | 3.66267 秒 |
| `performance/tokens_per_second` | 727.885 |
| `data/step_tokens` | 2666 |
| `data/consumed_tokens` | 49641 |
| `data/step_samples` | 8 |
| `data/consumed_samples` | 160 |
| `memory/device_max_allocated_gb` | 51.0602 GiB |
| `memory/device_max_reserved_gb` | 54.2344 GiB |

本次未输出：

- `performance/tflops_per_device`：已启用开关，但估算器只支持 `qwen3_moe`，不支持本次模型类型。
- `training/router_aux_loss`：本次模型路径未提供监控所读取的 `last_aux_loss`。

没有输出某指标，不等于整个仓库没有该能力；核心库具备能力，也不等于通用 Trainer 已接入。

### A.3 当前调用链

```text
TextTrainer.train_step()
  └─ TextTrainer.on_step_end()
      └─ BaseTrainer.on_step_end()
          ├─ EnvironMeterCallback：生成指标字典
          ├─ LoggingCallback：组装完整日志
          │   └─ TqdmCallback.write(message)
          └─ TqdmCallback.on_step_end()
              ├─ set_postfix(..., refresh=False)
              └─ update(...)
```

两个 callback 各注册一次、每步各调用一次，并有 step 去重保护。
本次非交互路径中，`write()` 只暂存消息；随后 `_LogProgressBar.update()` 拼接摘要和完整日志并输出一次。
成功交给 tqdm 后不会再调用 `logger.info()` 输出同一条消息。
因此当前问题是内容重复展示，而非 callback 重复执行。

`EnvironMeterCallback` 写入的是内存中的 `step_train_metrics` 和 `step_env_metrics`，不负责落盘。
本次文件来自 shell 重定向。

## 附录 B：主流框架 MFU 算法调研

先列算法与口径差异，再决定 HyperParallel 的实现。核对日期为 2026-11-05。
Megatron-LM 与 VeOmni 使用本地干净源码的固定提交；其余来源注明为官方文档或搜索索引快照，
不将历史路径的公式视为最新主线实现。

**统一符号与比较基准：**

- `F_step`：整个模型处理一个全局 optimizer step 的理论训练 FLOPs。
- `dt`：与该 FLOPs 统计范围对应的墙钟时间，单位秒。
- `W`：参与该模型训练的物理加速设备数，不能只取 DP size。
- `P_peak`：单设备、指定精度及稀疏性口径的理论峰值 FLOP/s。
- `N_mm`：参与矩阵乘的参数量；MoE 使用激活路径，而非全部 expert 参数。
- `L`：层数，`H`：hidden size，`S`：序列长度，`B`：全局序列数，`V`：词表大小。
- `T = sum(S_i)`，`Q = sum(S_i²)`：全局输入序列的 token 总量和长度平方和。

```text
TFLOP/s/device = F_step / (dt × W × 10^12)
MFU_ratio     = F_step / (dt × W × P_peak)
MFU_percent   = 100 × MFU_ratio
```

如果已有 per-device tokens/s，则 `MFU = F_token × tps_device / P_peak`，不能再除一次 W。
FLOPs 是运算量，FLOP/s 是速率，二者在接口和日志中分开。
MFU 的分子通常不含 activation checkpoint 额外重计算；重计算增加耗时会降低 MFU。
将重计算加入分子得到的是不同的硬件工作量估计，应另称 HFU，不能用它抬高同名 MFU。

**算法对照：**

| 框架与核对范围 | FLOPs 分子 | 利用率计算与特点 | 需要注意 |
| --- | --- | --- | --- |
| Megatron-LM，`31fbdf03` | 按 attention、Dense/MoE FFN、logits 等结构求和；训练矩阵乘通常是 forward ×3 | 训练日志先输出 `F_step / dt / W / 1e12`，再除明确设备峰值才得到 MFU | TFLOP/s/GPU 本身不是 MFU；支持的 GDN/MLA/MTP 等必须走对应分支 |
| Megatron Bridge，官方 FLOPs 接口 | 接受序列长度和、平方和、视觉 patch 等运行时统计，也允许模型提供 estimator | 先获得模型 FLOPs，再与训练耗时和设备数结合 | 特别适合变长/packing；不能只用平均长度平方代替平方和 |
| TorchTitan，官方搜索索引快照 | 模型提供 `num_flops_per_token` | 日志窗口内 `MFU%=100×F_token×tps_device/P_peak` | 此快照量化启用时不输出 BF16 基准 MFU；旧文件路径当前返回 404，需锁定可访问版本后再移植 |
| VeOmni，`617cc0fb` | 根据 model_type 使用结构公式，输入实际长度列表；涵盖 Qwen MoE/混合注意力等 | estimator 返回当前输入范围的 TFLOP/s 与单设备峰值，调用方仍需正确处理分布式范围 | 数据范围不能从函数名猜测；不能把 supervised-token 数当输入长度 |

**A. Megatron-LM：结构展开法**

在核对版本的标准 GQA/SwiGLU 路径中，记 query 投影宽度为 `Dq`，KV 投影宽度为 `Dkv`，
则无 output gate 的每层 attention 训练量可整理为：

```text
F_attn_layer = 6 × B × S × [H × (Dq + 2Dkv) + Dq × H + Dq × S]
```

最后一项使用因果三角注意力约定。SwiGLU 的 Dense FFN 为 `18 × B × S × H × I`；
MoE routed FFN 为 `18 × B × S × H × I_expert × top_k`，共享 expert 另计。
Logits 项是 `6 × B × S × H × V`。各类层分别累计；MLA、GDN、MTP 等有专门路径，
不能将全部层直接套标准 attention 公式。
其日志使用一个日志窗口的平均 iteration 耗时计算 TFLOP/s/GPU，
另有 job/cumulative throughput，时间窗口不同不能混比。

来源：[固定提交 training.py](https://github.com/NVIDIA/Megatron-LM/blob/31fbdf03643f8dc7aaa0671b3587cca558bb48ad/megatron/training/training.py)。

**B. Megatron Bridge：运行时长度与模型 estimator**

官方接口接受 `seqlen_sum`、`seqlen_squared_sum`，并为视觉、交叉注意力提供独立规模输入；
支持模型提供运行时 FLOPs estimator。对变长训练，可将线性项写成 `a×T`，
标准全注意力项写成 `b×Q`，而不是对所有样本都使用固定最大长度。
这是一种实现方向，不表示任意混合架构都能缩成同一组 a/b 系数。

来源：[Bridge FLOPs 实现](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/training/utils/flop_utils.py)。

**C. TorchTitan：模型 FLOPs/token × 日志窗口吞吐**

官方索引快照中，先按 non-data-parallel size 将本地 token 速率换算为 per-device 吞吐，
然后计算 TFLOP/s 和 MFU%。模型 FLOPs/token 由模型侧提供，指标层只负责窗口计时及归一化。
该快照明确在 FP8/MX 等量化场景跳过基于 BF16 峰值的 MFU。
本次旧路径原文访问为 404，因此仅作为已索引设计参考，不宣称已核实当前 main 的模型系数或 MoE 公式。
实施前需补齐可访问提交，不能用第三方转述代替模型实现。

来源：[官方索引对应 metrics.py 路径](https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/metrics.py)。

**D. VeOmni：按模型族和实际长度估算**

核对版本以 `sum(batch_seqlens)` 和模型 config 计算，训练线性层一般为 `6×N_mm×T`，
注意力等额外项由各模型函数实现。Qwen3.5 family 有 Dense/MoE、混合注意力和视觉分支；
MoE 区分 top-k 激活 expert、共享 expert、router、shared gate，embedding lookup 不按 LM head 矩阵乘计费。
LoRA/frozen 路径对 forward、输入梯度、权重梯度使用不同系数，不能一律乘 6。
设备峰值来自表；该版本未知设备返回 infinity、不支持模型可返回零，这种缺失值处理不用于本方案。
设备名称匹配也不能替代具体 SKU、精度与稀疏性核验。

来源：[固定提交 count_flops.py](https://github.com/ByteDance-Seed/VeOmni/blob/617cc0fb7d7af70c07f4507fc31d871f4eef55d0/veomni/utils/count_flops.py)。

**对 HyperParallel 的设计结论（建议，尚未实现）：**

1. 采用第 2.9 节确定的标准模块分块方案：adapter 声明块组成、通用块公式求和、运行时提供几何、指标层计时/归一化。
2. 保持 `training.log_tflops` 与 `monitor.mfu` 分别控制输出；分块明细可用于诊断，完整覆盖且峰值可信时输出 `mfu_percent`。
3. Qwen3.6 应分别核算标准 attention、GDN、top-k routed/shared FFN、router、LM head，
   不能使用总参数量或直接复用现有仅支持 qwen3_moe 的 estimator。
4. 输入 prompt 即使被 loss mask 也参与模型计算；同时保留 input/valid tokens/s，但 MFU 不用监督 token 吞吐替代计算规模。
5. 明确 causal-triangular/full-square 约定，记录 estimator 名称与版本，跨框架对比先统一公式。
6. MoE recorder 的有效 assignment 可辅助观察实际路由负载；padding、drop 或容量变化导致的实际额外工作量
   与标准模型工作量分开记录，不无说明地改变 MFU 分子。
7. 设备峰值支持显式配置，记录 SKU、dtype、dense/sparse 和来源。FP8/BF16 混合计算不套单一未知峰值。
8. 每步及窗口 MFU 分开命名；窗口使用 `sum(F_step)/sum(dt)`，不是简单平均各步 MFU。
   checkpoint 加载、组网等启动耗时另报，不能混入稳态 step MFU；端到端利用率如提供需独立命名。
9. 未支持模型或未确认设备峰值时输出不可用原因，不报零、不回退到任意设备数值。

## 附录 C：Local loss 与 module norm 详细规则

复用训练链路的 step loss 和实际用于裁剪的 global norm，不由 callback 再计算一次。
local loss 与 local grad norm 均支持独立开关和采样频次，覆盖所有持有对应数据的 rank；仅 local grad norm 支持 module 筛选。
这些属于 monitor，不引入 Debug 配置或 callback。

### C.1 Module 选择与输出频次

- 以下 module 选择规则仅用于 local grad norm；local loss 不配置 `modules`。
- `modules` 使用模型规范路径列表精确选择子树；空列表表示整体。不额外暴露 exact/glob/regex 模式开关。
- 初始化时解析模块选择并记录命中结果；全局无匹配时明确报错，某个 PP stage 不持有目标模块则正常跳过。
- 以并行包装前的规范模块名标识，保留到实际执行模块的映射；某个 PP stage 不拥有目标模块时正常跳过。
- 不提供 `ranks` 配置；所有数据持有 rank 按采样计划采集，grad norm 额外按 module 选择。local 采集本身不额外全局规约。
- `sampler_steps` 表示每 N 个 optimizer step 采集一次；输出由外部日志触发控制，记录保留实际采样 step，不将旧采样值标成当前值。
- 采样 step 覆盖全部真实 micro-batch，编号从 0 开始；不暴露 micro_steps 选择开关。
- 每个 local 事件携带 step、micro_step（如适用）、rank、PP stage 和采集阶段；module 仅在 grad norm 等适用事件中携带。
- JSONL 按 rank 保存，console 输出明确标识 rank；远端汇总只在开启相应输出时有界收集，不能冒充零通信。

### C.2 Local / micro loss

- 在跨 rank 规约前的 loss 产生位置采集；记录原始分子、有效分母及 reduction/scale 语义。
- 固定记录 micro 与 step 两种粒度，不再配置 granularity。step 级 local loss 使用该 rank 全部 micro-batch
  的分子和分母汇总，不能仅对采样出的 micro loss 直接取均值。
- step local loss 需取得该 step 的完整统计，不通过部分 micro-batch 代替整步。
- 直接从训练流程取得 local/micro loss，不需要 module 路径或用户注册 loss provider；普通 decoder/block 不生成逐层 loss。
- 仅在实际计算 loss 的 PP stage 记录该 loss；其他 stage 不制造同名 loss。

### C.3 Gradient local norm

Issue 的“规约前”定义与“规约后”采集时机存在冲突，语义必须区分。当前精简配置只提供规约后裁剪前；规约前保留为未来能力，不加入本次 YAML：

| 采集阶段 | 含义 | 输出键 |
| --- | --- | --- |
| `pre_reduce` | 训练梯度通信前，当前 rank 可观测的梯度范数 | `training/grad_norm_local_pre_reduce` |
| `post_reduce_pre_clip` | 训练通信完成后、裁剪前的本地梯度/分片范数 | `training/grad_norm_local_post_reduce_pre_clip` |

module 选择统计该 module 子树的全部参数，包含 expert；去掉 include_children/include_experts 等重复开关。
共享参数按参数身份去重；逐 module 输出与跨 module 总计的去重规则分别说明。
norm 固定为 L2，不提供 norm_type 开关；混精时以 unscaled 梯度为统计目标，并说明累计和缩放状态。

默认采用 `post_reduce_pre_clip`，不把它称为规约前 local norm，也不把本地分片 norm 称为 global norm。
FSDP/DDP 可能在 backward 中已执行通信，因此 `pre_reduce` 需要在实际通信前接入采集点；
不能简单在 backward 结束后读取 `.grad` 冒充规约前结果。
如果当前并行路径无法低开销取得指定粒度的规约前累计梯度，应报告不支持，不能保存完整梯度副本强行实现。
不同 micro-batch norm 不能直接相加得到累计梯度 norm，因其缺少梯度间的交叉项。

分项 loss 需说明是否包含 MoE auxiliary、MTP 等项，不能假定各模型总 loss 都有相同组成。

## 附录 D：分块耗时字段与计时规则

### D.1 启动阶段

“组网”分别统计分布式通信初始化和模型网络构建，避免两者同名。

| 阶段 | 建议字段（均为秒） | 起止范围 |
| --- | --- | --- |
| 分布式初始化 | `performance/startup/distributed_init_s` | rendezvous / 默认进程组初始化；记录可测部分 |
| 并行组构建 | `performance/startup/parallel_groups_s` | DeviceMesh 及 DP/CP/TP/PP/EP 子组构建 |
| 模型构建 | `performance/startup/model_build_s` | config 解析后至模块树构建结束 |
| 预训练权重加载 | `performance/startup/pretrained_load_s` | 权重读取、转换和加载，可细分子阶段 |
| 并行包装 | `performance/startup/parallel_wrap_s` | 替换、分片、FSDP 等并行配置应用 |
| 数据初始化 | `performance/startup/data_init_s` | tokenizer/assets、数据集读取/映射、dataloader 构建 |
| 优化器初始化 | `performance/startup/optimizer_init_s` | optimizer 与 scheduler 构建 |
| 续训 checkpoint 加载 | `performance/startup/checkpoint_restore_s` | 模型、优化器、scheduler、RNG、dataloader 状态恢复 |
| 总启动时间 | `performance/startup/total_s` | Trainer 初始化入口至可开始首个训练 step |

预训练权重加载和续训 checkpoint 恢复分别记录，未执行的阶段标记为未执行，不能编造加载耗时。
某些 from_pretrained 路径将构建、读取、转换交织执行，应保留真实嵌套关系；
无法准确拆分时先记录父阶段，不靠总耗时相减猜测。
启动计时器必须在 `_setup()` 等初始化之前可用，不能等 `_init_callbacks()` 后才开始计时。
进程组尚不可用时先本地记录，之后按需汇总，不为计时提前初始化通信组。

### D.2 Step 内分块

| 阶段 | 建议字段 | 统计方式 |
| --- | --- | --- |
| 获取 batch / 等待 | `performance/step/batch_fetch_s` | 主进程获取本步各 micro-batch 的耗时之和 |
| 样本处理 | `performance/step/preprocess_s` | transform/tokenize 等实际执行位置采集 |
| Collate | `performance/step/collate_s` | batch 组装耗时，多 worker 时携带执行范围 |
| H2D | `performance/step/h2d_host_s` | host 提交/阻塞时间，不冒充设备传输完成耗时 |
| 前向与反向 | `performance/step/forward_backward_host_s` | 调用路径 host 墙钟时间；能准确拆分时再细分 |
| 梯度裁剪 | `performance/step/grad_clip_host_s` | 实际裁剪调用范围 |
| 优化器更新 | `performance/step/optimizer_host_s` | optimizer.step 调用范围 |
| 调度器 | `performance/step/scheduler_host_s` | scheduler.step 调用范围 |
| 权重保存 | `performance/checkpoint/weights_save_s` | 同步权重保存，从请求开始至保存接口确认写入完成，包含必要的收集、转换和序列化 |
| 异步权重保存 | `performance/checkpoint/weights_save_submit_s`、`weights_save_complete_s` | 分别为提交调用耗时、请求开始至完成确认的时长，通过 checkpoint_id 关联 |
| 完整 checkpoint 保存 | `performance/checkpoint/save_submit_s`、`save_complete_s` | 整体保存；同步时记录完成时长，异步时区分提交和完成。包含权重子阶段时不可重复相加 |

设备 event 计时作为后续扩展，本次不提供相关配置或 `_device_s` 字段。
不为每个块增加全设备同步，也不把 host 提交时间解释成 kernel 耗时。
通信与计算重叠时，不根据同步 API 调用耗时推断真实通信占比；完整重叠分析属于 profiler。

### D.3 时间聚合原则

- CPU 墙钟采用单调时钟；同一阶段明确 inclusive/exclusive 口径。
- 嵌套或重叠块不能直接相加等于 step 时间；数据 worker 时间不能当成主进程等待时间。
- micro-batch 同类耗时在本地 step 内累计，附带次数；按需提供均值和最大值。
- 跨 rank 聚合先保留 local 值，再以明确组给出 max/min/mean；不同块的 max 可能来自不同 rank。
- 全局 max 的和不等于实际关键路径，输出和文档不能这样解释。
- 未观测到的 worker 处理时间标记不可用，不用主进程 batch_fetch 代替。
- 保留既有 step_time 基线口径；如另设端到端时间，单独命名，不能静默替换吞吐分母。

## 附录 E：输出协议与 WandB 细则

WandB 直接消费统一指标字典和元数据，**不解析 console 文本，也不要求用户按日志正则上传**。
Console 面向阅读，JSONL 是机器可读记录，TensorBoard/WandB 分别实现输出适配器。

**格式适配与直接上报：**

默认 `sinks: [console, jsonl]`，不初始化 WandB SDK，也不上传。
用户显式加入 `wandb`（如 `sinks: [console, jsonl, wandb]`）后，由适配器调用 SDK 直接上传；无需先生成或转换日志文件。
在线模式需要安装 SDK、配置凭据并能访问服务；离线模式写入 W&B 本地 run 数据，之后同步。
JSONL 是独立的本地记录，不会被 W&B 自动识别为训练曲线；补传 JSONL 需要显式转换工具。

| 数据类型 | W&B 表示 | 适配边界 |
| --- | --- | --- |
| loss、norm、lr、耗时、吞吐、显存、MFU | `run.log()` 的扁平数值字典 | 保留 `training/total_loss` 等字段名，值为 CPU 端 Python int/float，不传格式化字符串或设备 tensor |
| 训练步数 | 自定义 `global_step` 横轴 | 用 `define_metric` 绑定训练指标前缀；W&B 内部 history step 不等同于训练 step |
| EP 逐 expert 计数 | `wandb.Table` | 列含 step、rank/scope、module、expert_id、count；保留 expert 身份，不默认生成每 expert 一条曲线 |
| 负载数值分布 | 可选 `wandb.Histogram` | 只表示负载分布，不能替代表格中的 expert ID 对应关系 |
| 启动与 ckpt 事件 | 事件 Table；一次性启动标量可写 summary | 不伪造训练 step；反复保存的事件不能只用 summary 覆盖 |
| 模型、并行配置、版本 | run config / 元数据 | 初始化时写入，过滤凭据等敏感配置 |

以下为适配器调用示意，初始化和最终 `finish()` 由生命周期管理；仅 global rank 0 调用：

```python
run.define_metric("global_step")
for prefix in ("training", "performance", "data", "memory", "moe", "optimizer"):
    run.define_metric(f"{prefix}/*", step_metric="global_step")

# 每个输出步汇总后调用一次；metrics 为已转换的数值字典。
run.log({"global_step": step, **metrics})
```

- 采样指标只在实际采集的 step 上报；不补零、不复用旧值。同一步各标量先合并再 log。
- 多 rank 细节默认留在各 rank 本地文件；需要上传时显式汇集到 global rank 0，再转 Table。
- resume 恢复原 run ID 与发送进度，适配器处理去重；自定义横轴本身不保证去重。
- 验收增加：20-step 在线 run 曲线与 JSONL 对齐、EP 表格 ID/计数一致、离线同步后结果一致、续训不重复记录。

SDK 格式依据：[W&B 自定义横轴与字典上报文档](https://docs.wandb.ai/models/track/log/customize-logging-axes)。

- 同一训练 run 仅由 global rank 0 初始化一次远端输出，记录代码版本、模型、并行配置和 schema 版本。
- 同一采样 step 的常规指标合并上报，保持统一 global_step 轴，避免多 callback 分别重复发送。
- 启动耗时与 checkpoint 生命周期事件使用独立事件/汇总字段，不作为虚假的训练 step。
- rank/层细节按稳定命名或表格/分布表示，避免默认展开所有 expert 导致指标数量失控。
- JSONL 公共字段包含 schema_version、run_id、event、rank；step/epoch 仅在适用时出现。
- 正常 step 标量使用数值类型，单位通过字段或元数据固定；缺失值不填零。
- WandB 为显式启用的可选输出，支持在线/离线模式；凭据不写入 YAML 或日志。
- 初始化、flush、finish 与训练生命周期一致；恢复时明确 run_id、resume 策略和已发送 step。
- 上传失败默认告警并保留 JSONL，远端重试采用有界缓冲，不无限阻塞训练或增长内存。
- 原有 wandb 配置映射到新输出入口，冲突时明确报错；SDK 接口在实现时核对官方文档。

详细监控记录保留 `micro_step`、`moe_layer`、`optimizer_metrics`，携带必要的 rank/层/参数信息。
不设置 Debug callback、debug 配置段或独立调试子系统。

## 附录 F：ExpertLoadRecorder 接口与统计语义

增加独立的 `ExpertLoadRecorder`，由 Trainer 持有，model/router adapter 注册采集点。
一个 recorder 管理全部已注册的 MoE 层，不为每层创建 callback；不直接打印或调用 WandB。
现有 `MoEMonitorCallback` 的 expert bias 更新职责不迁入 recorder，避免监控改变训练算法。

```text
Trainer step_begin → recorder.begin_step(step)
Router / Dispatch → recorder.record(layer, micro_step, invocation, counters)
Trainer step_end → recorder.finalize_step()
                 → 等待异步结果就绪（如适用，保留原采样 step）
                 → EnvironMeterCallback 合并标量与分布快照
                 → LoggingCallback → Console / JSONL / TensorBoard / WandB
                 → recorder.reset_step()
```

Recorder 在组网完成、模型完成并行包装后初始化，注册规范 module 名、expert 全局 ID、
本地 expert 所有权、相关通信组及计数副本/切片关系。模型不暴露可靠计数时标记不支持。

**接口草案：**

| 接口 | 职责 |
| --- | --- |
| `register_layer(spec)` | 注册 module 路径、expert 数量、ID 映射、分布式布局、计数语义和能力 |
| `begin_step(step)` | 按统一采样计划启用当前 step，取得空闲槽位；不清空上一轮在途摘要 |
| `record(layer_id, micro_step, invocation_id, counters, phase)` | 收集当前实际 forward 的轻量计数，忽略重计算 phase |
| `finalize_step()` | 按预定组和顺序提交聚合，完成后生成一次不可变快照；异步未完成时保留 pending，重复调用不重复通信或累计 |
| `reset_step()` | 已完成且消费结束后归还槽位；中止时记录 incomplete，完成在途访问后再回收 |
| `close()` | 解除采集注册、释放资源；不负责销毁训练通信组 |

接口名称是设计草案。常规路由只复用现有小规模计数，不额外保存 router logits、
完整 token→expert 索引、hidden states 或梯度。

**原始数据约定：**

| 数据 | 含义 |
| --- | --- |
| `accepted_assignments_per_expert` | 每个 expert 实际接收的有效路由分配次数，排除物理 padding |
| `attempted_assignments_per_expert` | 容量裁剪前的路由分配次数，仅源端提供时采集 |
| `dropped_assignments_per_expert` | 被丢弃的路由分配次数，仅源端明确提供时采集 |
| `physical_slots_per_expert` | 含对齐/padding 的物理处理槽位，可选，和有效负载分开 |
| `valid_input_tokens` | 对应路由输入中的有效 token 数，不是监督 label token 数 |

Top-k 路由中一个 token 可以贡献多次 assignment，因此 expert load 默认以 assignment 计数。
`dropped_assignments / attempted_assignments` 只能称分配丢弃率；
若要输出唯一 token 丢弃率，必须由 router 提供明确定义的 token 去重统计，不能直接换名。
计数 buffer 使用 detached 整数向量，首次采集复制到自有 buffer，避免源张量复用后改变记录。

**分布式归约语义：**

- 优先采用 dispatch 接收端的有效计数，此时各 EP rank 贡献自己拥有的 expert，按全局 expert ID 拼接。
- 若源端计数来自互不重叠的输入 token 分片，则按 expert ID 求和；若多个 rank 看到同一份路由数据，
  只计一次，不能机械 all-reduce 后重复累加。
- CP/TP、expert replication、DP/EDP 的副本和分片关系由 adapter 的 layer spec 明确给出，
  不能仅根据 world size 推断；跨数据副本汇总也要注明通信组和观察窗口。
- PP 各 stage 只拥有部分层；逐层统计在所属 stage 完成，再按显式流程向汇总 rank 传递。
- 采样与通信顺序由参与组一致决定，不能因为只有 rank 0 打印，就让其他 rank 跳过通信。
- 本地采集不增加通信；逐层全局汇总的通信仅在采样步发生，尽量批量处理多层计数。scope 是记录元数据，不再是用户开关。

**派生指标与输出：**

每层输出 assignment 总量、max、mean、空 expert 数和 `max_vio`，可选保存完整 expert 计数向量。
`max_vio = max(load) / mean(load) - 1`，总量为零时定义 `max_vio=0` 并标记 `empty_batch=true`；
expert 数量为零属于注册错误。所有 expert 负载应参与均值，包括零负载 expert。
汇总先计算各层 max_vio，再输出跨层最大值；不能把不同层的 expert 数和负载直接混合计算。
完整 expert 向量在 JSONL 保留 ID 对应关系；远端图表可使用表格/分布，不默认创建每 expert 一条曲线。

```text
event=moe_layer step=<step> module=<canonical_module> scope=<scope>
count_unit=assignment accepted_total=<count> load_max=<count> load_mean=<count>
max_vio=<value> empty_experts=<count> empty_batch=<bool>
```

每次快照附带采样 step、micro-batch 次数、聚合组/范围和计数语义。
采样步统计整步全部实际 micro-batch，按外部日志触发发布已就绪记录，保留原采样 step，不与前后未采样步混合。
未采样步不沿用上一轮结果；只有明确启用跨步窗口时才累计并输出窗口范围。

**重计算、共享模块与资源管理：**

- activation checkpoint 的反向重计算不重复记录；不能仅凭 grad_enabled 区分，因为重计算模式不同。
- 使用实际 forward/recompute 上下文以及调用身份区分；同一 module 在同一 micro-step 合法调用多次时都应统计。
- 每个槽位状态为 idle → collecting/skipped → pending（异步时）→ finalized → consumed → reset，异常中止按在途完成状态安全回收。
- 采样快照消费后自动释放或归还缓冲，不暴露 flush 开关；异步在途摘要须等待完成后才能复用，不表示每层同步设备或立即网络上传。
- 缓冲按“已注册层数 × expert 数 × 有界在途槽数”固定规模，不随 token 数或训练步数增长。
- 关闭时不创建 recorder 统计 buffer 或额外 hook，不改变 router 的输出、expert bias 和梯度。

**验收：**

- 以小规模显式路由结果独立复算 expert load、max_vio 和 assignment drop。
- 覆盖 top-k、空 batch、空 expert、padding、重复路由副本、EP 分片及 DP/CP/TP/PP 组合。
- 验证重计算不重复计数，共享 module 多次真实调用不漏计，finalize 重入不重复通信。
- 关闭和开启时的训练数值一致；采样间隔与全部已注册层的覆盖生效，buffer 长时间运行不增长。
- 未提供 drop 统计的 router 不输出伪造的零 drop；local/global 输出名称与 scope 明确区分。

## 附录 G：配置语义与兼容规则

完整 YAML 见第 1.3 节，以下补充采样、输出和兼容语义。

详细规则：

- `sampler_steps` 控制可降采样的统计，累计消费量仍逐步维护或由可靠的累计源提供。
- 输出频率由外部 Trainer 控制，`monitor` 不定义 `logging_steps`；当前步指标不自动解释成间隔平均值。
- 未在某步采集的可选指标不得沿用旧值冒充当前值；需要携带采样 step 或省略。
- 详细指标全 rank 采集，grad norm 固定 L2；仅保留 module 选择、Top-N 等必要配置。
- 沿用外部训练配置的日志策略，不在 `monitor` 新增日志间隔、兼容别名或优先级规则。
- 主输出固定 global rank 0；详细指标覆盖全部持有数据的 rank，无重复 rank 配置。
- 文件命名包含 run 标识，详细指标文件再包含 global rank，避免多进程竞争写入。

## 附录 H：字段兼容与迁移

第一阶段展示变更不得静默改变数值口径。
新字段名最终确定后提供 schema 版本和完整迁移表，以下为建议映射：

| 旧字段 | 建议字段 | 条件 |
| --- | --- | --- |
| `performance/step_time` | `performance/step_time_s` | 时间窗口保持一致 |
| `performance/tokens_per_second` | `performance/valid_tokens_per_second` | 确认来源是有效监督 token；不能对所有回退路径机械重命名 |
| `data/step_tokens` | `data/step_valid_tokens` | 同上 |
| `data/consumed_tokens` | `data/consumed_valid_tokens` | 同上，恢复状态同步迁移 |

可提供可选旧格式，不默认同时输出两套名称。

## 附录 I：参考实现

以下链接固定到本次分析和实跑版本：

- [Callback 注册与分发](https://github.com/mindspore-ai/hyper-parallel/blob/c75f5f9f/hyper_parallel/trainer/base.py)
- [纯文本训练循环](https://github.com/mindspore-ai/hyper-parallel/blob/c75f5f9f/hyper_parallel/trainer/text_trainer.py)
- [EnvironMeterCallback](https://github.com/mindspore-ai/hyper-parallel/blob/c75f5f9f/hyper_parallel/trainer/callbacks/environ_meter_callback.py)
- [LoggingCallback](https://github.com/mindspore-ai/hyper-parallel/blob/c75f5f9f/hyper_parallel/trainer/callbacks/logging_callback.py)
- [TqdmCallback](https://github.com/mindspore-ai/hyper-parallel/blob/c75f5f9f/hyper_parallel/trainer/callbacks/tqdm_callback.py)
- [模型 FLOPs 估算](https://github.com/mindspore-ai/hyper-parallel/blob/c75f5f9f/hyper_parallel/trainer/runtime/flops.py)
