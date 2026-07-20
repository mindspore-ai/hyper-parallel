# HyperDryRun：面向大模型分布式训练的无卡显存分析

## 1. 背景与目标

大模型参数规模、序列长度和并行规模持续增长后，显存是否足够通常只能通过申请真实 NPU/GPU 集群、启动完整训练任务来验证。一次配置调整需要经历资源申请、排队、环境启动、运行失败和重新配置，失败任务还会造成整组设备空转。

静态参数量公式只能给出参数、梯度和优化器状态的理论常驻量，难以覆盖训练过程中的以下动态峰值：

- HyperParallel FSDP/HSDP 参数 all-gather、reduce-scatter 和融合通信缓冲区；
- 前向激活、反向临时张量、梯度累积和优化器首次建 state；
- `reshard_after_forward`、激活重计算以及 `resize_(0)` 引起的生命周期变化；
- TP、CP、EP 与 FSDP 组合后每个 rank 不同的本地 shape；
- 峰值所在的模型层和前向/反向阶段。

HyperDryRun 的目标是在不初始化真实加速卡、不执行数值计算的前提下，复用 Trainer 的模型构建和模型自有 `parallelize_fn`，模拟一个完整训练 step，输出目标 rank 的逻辑张量显存峰值、构成和模块定位信息。

首版目标：

1. 通过现有 `scripts/train_lm.py` 和训练 YAML 一键启用；
2. 单进程模拟目标 world size 和 rank，不要求其他进程在线；
3. 首个可验收基线覆盖 Torch Trainer 的 FSDP；runner 复用模型完整 `parallelize_fn`，HSDP、TP/CP/EP 和激活重计算组合按第 9 节兼容矩阵逐项纳入；
4. 覆盖参数、buffer、激活、梯度、反向临时张量、优化器状态和通信张量；
5. 输出稳定、可机读的 JSON 报告，并给出 OOM 风险判断；
6. 对不支持的模型算子、数据依赖 shape 和配置给出明确错误，不回退到真实设备执行。

## 2. 非目标与术语

### 2.1 首版非目标

- MindSpore 后端：配置入口保留，但启用时明确报错；待 MindSpore 提供等价 fake execution 和内存跟踪能力后扩展。
- Pipeline Parallel：PP schedule 涉及跨 stage P2P 和仅首 stage 读数据，首版不输出估算结果，启用 `pp > 1` 时 fail-fast。
- 加速卡 allocator 的 reserved memory、碎片、框架上下文、kernel workspace 和第三方 fused kernel 内部临时显存。
- 通信耗时、计算耗时和吞吐量预测。
- 数据依赖动态 shape（例如从 Tensor 值决定 token 数或分支）的自动求解。
- 加载 checkpoint 的数值内容。Dry-run 只需要架构、dtype 和 shape；`weights_path` 仍可供模型读取配置，但不会加载权重 Tensor。

### 2.2 统计口径

报告中的“显存”是目标设备上、可由 TorchDispatch 观察到的**逻辑 live tensor bytes**，不是 `nvidia-smi`/NPU SMI 的 reserved bytes。分类沿用 PyTorch `MemTracker`：

| 分类 | 含义 |
| --- | --- |
| Parameter | 当前 rank 的参数 shard，以及 FSDP unshard 后仍存活的参数存储 |
| Buffer | module buffer |
| Activation | 前向产生且仍存活的张量 |
| Gradient | 参数梯度 |
| Temp | 反向临时张量 |
| Optstate | AdamW 首次 step 创建的优化器状态 |
| Other | 显式纳入跟踪但无法归入上述类别的输入等张量 |

通信算子由 PyTorch fake collectives 执行 meta kernel，产生的通信输出仍由 `MemTracker` 按其生命周期归入 Activation/Temp；报告额外记录所用并行拓扑和 `comm_fusion`，避免把通信张量误解为独立常驻项。

## 3. 用户接口

### 3.1 YAML 配置

在严格三层配置的 `train` 下新增：

```yaml
train:
  dry_run:
    enabled: true
    world_size: 8        # 可省略；仅当并行度能够完全推导时自动计算
    rank: 0              # 需要分析的逻辑 rank
    device_type: npu     # 可选 npu/cuda；CPU-only 环境建议显式填写
    output_dir: outputs/dry_run
    module_depth: 3      # JSON 中展开的 module FQN 深度，0 表示不限制
    top_modules: 20      # 按局部峰值输出的热点模块数，0 表示全部
    device_memory_gib: 64.0  # 可选；用于 OOM 风险判断
```

等价 CLI：

```bash
python scripts/train_lm.py train.yaml \
  --train.dry_run.enabled=true \
  --train.dry_run.world_size=8 \
  --train.dry_run.rank=0
```

运行成功后不进入正常 `trainer.train()`，只执行一次 dry-run，并写入：

```text
<output_dir>/rank_<rank>.json
<output_dir>/rank_<rank>_memory.csv
```

### 3.2 配置校验

- `backend` 必须是 `torch`；
- PyTorch 必须为 2.7 或更高版本。2.6 的 `MemTracker` 尚未集成 fake collectives/DTensor dispatch，不能可靠统计 HyperParallel 通信；
- `rank >= 0`、`world_size > 0`、`rank < world_size`；
- `device_type` 只能是 `npu`、`cuda` 或空；为空时按当前 Torch 扩展推断，CPU-only 环境应显式配置以避免目标歧义；
- `module_depth >= 0`、`top_modules >= 0`、`device_memory_gib > 0`（若配置）；
- `world_size` 与 `dp_replicate * dp_shard * ep * cp * tp * pp` 必须一致；当 `dp_shard` 为自动值时必须显式提供 `world_size`；
- `pp` 必须为 1；
- Phase 1 正式支持 `dp_shard` FSDP 和 `activation checkpoint=full`；`dp_replicate/tp/cp/ep/etp > 1`、CPU offload 或 selective activation checkpoint 当前 fail-fast，待对应 CPU fake 集成用例和内存不变量通过后再放开；
- 为保证无卡，dry-run 内部强制使用 meta 初始化；用户的 `init_device` 只记录到报告，不触发真实分配；
- `weights_path` 不加载权重数据，但仍允许模型 factory 从中读取纯配置文件。若 factory 自身无条件加载 Tensor，错误会被包装为不支持提示。

## 4. 总体架构

```text
parse_args / discover_model_spec
            |
            v
  dry_run.enabled ? ---------------------- no --> LLMTrainer.train()
            |
           yes
            v
 TorchDryRunRunner
   1. 校验配置并复制 args（init_device 强制 meta）
   2. 注册 fake ProcessGroup backend
   3. 单进程创建目标 world_size/rank 的 DeviceMesh
   4. 在 FakeTensorMode 外构造 meta model 并执行 parallelize
   5. 进入 FakeTensorMode，复用 BaseTrainer materialize fake shard
   6. 构造固定 shape 的 fake LM micro-batch
   7. MemTracker：forward -> backward -> AdamW step -> zero_grad
   8. 归一化 snapshot/module stats，写原子 JSON
   9. 销毁 fake ProcessGroup，恢复全局 hook/context
```

### 4.1 为什么在 Trainer 构造前分流

现有 `LLMTrainer.__init__` 会立即执行 `_setup()`、设置真实 device、构造 dataloader 并 materialize 参数。若只在 `train()` 外层套 FakeTensorMode，真实设备已经被初始化，违反“无卡”目标。因此 `scripts/train_lm.py` 必须在构造 `LLMTrainer` 之前判断 `dry_run.enabled`，改由专用 runner 驱动 `BaseTrainer` 的必要 build steps。

### 4.2 通信组网模拟

Torch 的 `FakeProcessGroup` 可以让单进程声明任意 `world_size/rank`，collective 不等待其他 rank。实现通过 platform 抽象新增 dry-run process-group 初始化接口：

- Torch：注册 `fake` backend，使用 `HashStore` 和 `FakeProcessGroup`；
- MindSpore/基类：抛出描述性 `NotImplementedError`；
- BaseTrainer 在 dry-run 时跳过 `device_handle.set_device()` 和设备 RNG 初始化，但仍构造真实的逻辑 `ParallelDims`/`DeviceMesh`，从而复用现有模型 parallelize 逻辑。

fake backend 只模拟 shape 和生命周期，collective 输出数值没有语义；dry-run 禁止任何依赖 collective 数值的控制流。

### 4.3 FakeTensor 与模型构建

DeviceMesh 必须在 FakeTensorMode 外创建，因为 mesh 初始化包含真实 CPU rank Tensor 和 Python 控制数据。模型构建及 parallelize 也在 mode 外使用零存储的 meta 参数完成，只有 materialize 和训练 step 进入 FakeTensorMode：

1. `init_empty_weights()` 在 meta 上建立完整模型；
2. 模型自己的 `parallelize_fn` 对 meta 参数应用 Phase 1 已校验的 FSDP；
3. 进入 FakeTensorMode 后，`_post_parallelize()` 调用 `to_empty(device=<simulation>)` 得到 FakeTensor，不分配设备内存；
4. dry-run 跳过 `_load_weights()`，保留 dtype 转换、FSDP lazy init 和训练模式设置；
5. `MemTracker.track_external()` 注册已经创建的模型、优化器和输入。

当当前 Torch wheel 编译了目标后端时，`simulation` 就是逻辑目标设备（`cuda` 或 `npu`）。CPU-only wheel 无法为加速卡 Tensor 建立 autograd device guard，此时 runner 自动使用 CPU FakeTensor 传播完全相同的 shape/dtype/lifecycle；报告同时记录 `device_type`（逻辑目标）和 `simulation_device_type=cpu`。这一回退不改变逻辑 tensor bytes，但仍不代表 allocator、kernel workspace 或真实后端自定义算子已经校准。

### 4.4 代表性训练 step

首版使用固定 shape 的 LM batch：

- `input_ids`: `[micro_batch_size, max_seq_len]`, `int64`；
- `labels`: 同 shape，`int64`；
- token 数按 `micro_batch_size * max(max_seq_len - 1, 1)` 直接计算，避免对 FakeTensor 调用 `.item()`；
- 调用 model spec 的 `prepare_batch_fn`（若有）并复用 Trainer CP 分片逻辑；
- 复用 `BaseTrainer.forward_backward_step()` 完成模型 forward、loss 缩放和 backward；
- 直接执行一次 `optimizer.step()` 以创建 AdamW state，再 `zero_grad(set_to_none=True)`。

不构造 dataset/dataloader、tokenizer、callback、checkpoint 和 logger sink，避免真实数据 I/O 与无关内存进入统计。

梯度累积的峰值通常由最后一个 micro-batch 决定；首版只执行一个 micro-batch，并在报告中标记 `gradient_accumulation_simulated=false`。后续可在确认 MemTracker 多次调用语义后扩展到完整 grad accumulation。

## 5. 报告协议

JSON 顶层字段：

```json
{
  "schema_version": 2,
  "status": "ok",
  "metadata": {
    "model": "qwen3_5",
    "torch_version": "2.9.1",
    "rank": 0,
    "world_size": 8,
    "device_type": "npu",
    "simulation_device_type": "cpu",
    "original_init_device": "meta",
    "weights_loaded": false,
    "gradient_accumulation_simulated": false,
    "parallel": {}
  },
  "summary": {
    "peak_bytes": 0,
    "current_bytes": 0,
    "peak_gib": 0.0,
    "capacity_bytes": null,
    "headroom_bytes": null,
    "oom_risk": null,
    "peak_breakdown_bytes": {},
    "current_breakdown_bytes": {}
  },
  "devices": {},
  "modules": [],
  "memory_blocks": [],
  "limitations": []
}
```

规则：

- 目标 rank 正常只出现一个 simulation device；`summary` 按 `simulation_device_type` 汇总，`devices` 保留 tracker 观察到的全部设备；逻辑目标仍以 `metadata.device_type` 为准；
- 同一类别和 Total 均以 byte 整数保存；显示值由消费者换算，避免精度损失；
- `peak_breakdown_bytes` 是全局峰值时刻的构成；优化器 state 可能在该时刻尚未创建，因此 `current_breakdown_bytes` 同时展示 step 结束后的常驻构成；
- module 条目包含 FQN、参数/buffer/input/output bytes、局部 peak bytes 和各阶段 snapshots；
- `top_modules` 在完成深度过滤后按 `local_peak_bytes` 降序截断；
- 写文件使用临时文件 + `os.replace()`，防止中断后留下半份 JSON；
- JSON 的 `memory_blocks` 与 CSV 同源：每个不与输入 alias 的新算子输出 storage 产生一条逻辑内存块；
  对已有 storage 的 size 变更会关闭旧块，非零的新 size 作为该 resize 算子的输出块重新记录；
- CSV 不包含 summary、rank、world size、容量或 OOM 字段，列名对齐 `ms_memory_block.csv`；
- `start_time_stamp/end_time_stamp` 是从 0 开始的逻辑任务索引，不采集或暗示真实时间；使用该 storage
  的后续任务写入 `user_tasks` 和 `last_user_task`，step 结束仍存活的 storage 使用
  `9223372036854775807` 并设置 `is_persistent=1`；生命周期按 `[start_time_stamp, end_time_stamp)`
  半开区间解释；
- `device_addr` 来自从 `0xf00000000000` 开始、按 512 bytes 对齐的确定性虚拟地址空间，
  `pool_type=FakeTensorLogicalMemoryPool` 明确其不是真实 HBM 地址；
- `actual_used_memory/actual_peak_memory` 来自生产算子返回后的 MemTracker 当前/运行峰值；`size` 是
  新输出 storage 的逻辑 bytes；`python_stack`、叶子 `file_name/line_num` 在算子 dispatch 时实时采集；
- storage 的释放优先由 Python storage weakref 回调识别；对 FakeTensor 缓存仍持有 wrapper 的情况，
  以 MemTracker 已删除该 storage 或其 size 归零作为释放信号。finalize 只将 MemTracker 中仍有正 size
  的 storage 标记为 persistent，并采用其最终类别，避免把 FSDP 已 reshard 的全量 buffer 或 AdamW
  state 错记为常驻 Activation；
- C++ autograd engine 直接进入 dispatch 时可能不存在用户 Python frame，此时 backward/Temp 行的
  `python_stack`、`file_name` 允许为空；`node_name`、`task_name=backward` 和 `graph_name` 仍保留算子、
  phase 与 module 上下文，不生成伪造的 Python frame；
- 根目录 `memory_analysis.html` 同时识别 MindSpore 的 `DefaultEnhancedAscendMemoryPool` 和 dry-run 的
  `FakeTensorLogicalMemoryPool`，并优先读取标准列 `stream_id`（兼容旧列 `stream`）；生命周期视图额外
  依赖在线加载固定版本的 PyTorch `MemoryViz.js`，离线时折线图仍可使用；
- `task_name`（phase）按运行状态识别：optimizer hook 的 `_in_opt` 优先，其次
  `recompute_state.is_recomputing()` 为 recompute，再次 `ModTracker.is_bw` 为 backward，其余为
  forward；`graph_name`（FQN）取 `ModTracker.parents` 中嵌套最深的活跃 module，表示调用上下文
  而不是算子身份，算子身份由 `node_name` 保存；
- 若执行失败，不伪造成功报告。命令返回非零并输出异常类型、触发阶段和改进建议；能确定输出目录时额外写 `status=error` 的最小报告。

## 6. 生命周期与资源清理

FakeTensorMode、MemTracker、FakeProcessGroup 都修改或依赖进程级状态，必须严格按栈清理：

1. tracker 先退出，恢复 `UntypedStorage.resize_`、DTensor dispatch 和 optimizer global hooks；
2. FakeTensorMode 退出；
3. fake process group 在 `finally` 中销毁；
4. 报告写入失败不能掩盖原始执行异常；
5. 同一进程只允许串行运行一个 dry-run，首版不承诺线程安全。

## 7. 动态 shape 与不支持算子

FakeTensor 可以传播由输入 shape 决定的静态 shape，但不能为 `.item()`、`nonzero()` 后参与 Python 分支等数据依赖表达式提供真实值。首版策略：

- 统一包装 PyTorch 的数据依赖 shape 和 unsupported fake operator 错误，保留原始异常类型、文本和最后执行阶段；
- 不自动填充值绕过分支，因为这会让峰值路径不可验证；
- 后续版本可增加 `dynamic_shape_overrides`，由用户显式选择分支或给出动态维上界，并在报告中记录假设。

## 8. 误差来源与校准

### 8.1 首版可比较口径

验收时真实 eager 基线必须使用“allocated tensor memory”而非 allocator reserved memory，并关闭 checkpoint/save/profile 等 dry-run 未执行功能。比较：

```text
relative_error = abs(dry_peak_bytes - eager_peak_allocated_bytes)
                 / max(eager_peak_allocated_bytes, 1)
```

### 8.2 5% 验收条件

在以下校准矩阵内，目标是峰值误差不超过 5%：

- Torch 2.7/2.9；
- dense decoder-only LM；
- FSDP world size 1/2/4/8；
- `reshard_after_forward` true/false；
- fp32/bf16；
- activation checkpoint off；
- AdamW，`foreach=false` 作为确定性基线。

`activation checkpoint=full` 已支持算子级 dry-run 和独立 recompute phase 记录，但尚不属于 5%
误差验收矩阵；加入硬门槛前仍需补齐同配置的真实 eager allocated-memory 校准。`selective` 模式
尚未进入 dry-run 支持范围并保持 fail-fast。

不在 5% 硬门槛内的项目必须单独展示，不混入通过结论：allocator 碎片/reserved、fused kernel workspace、数据依赖 MoE、第三方自定义算子、PP、真实 checkpoint/callback 内存。报告中的 `limitations` 始终列出这些差异。

## 9. 测试与验收计划

### 9.1 单元测试（无设备）

1. 配置解析：YAML/CLI、默认值、非法 rank/world size/depth/capacity；
2. world size 推导和并行度不一致错误；
3. 报告序列化：enum key、多个 device、模块深度/top-N、OOM/headroom；
4. 原子写入与输出目录创建；
5. `_post_parallelize()` 在 dry-run 时不调用 `_load_weights()`；
6. runner 生命周期：异常时 tracker/mode/process group 均被清理；
7. Torch 2.6、MindSpore、PP 的 fail-fast 信息。
8. 算子 storage 生命周期峰值不超过 MemTracker 峰值，persistent 输出不超过 current，且 current
   Activation 为 0 时不得残留 persistent Activation。
9. 可视化行为合同直接执行 HTML 中的 `parseCsv`、`analyzeRows`、`buildSnapshot` 和时刻存活查询，
   覆盖 fake pool、`stream_id`、INT64 sentinel、CSV 引号转义、free 事件和 `[start, end)` 边界。

### 9.2 CPU 集成测试

在安装 Torch >= 2.7 的 CPU 环境，用最小两层 LM 和 fake `world_size=2` 运行：

- 确认无 CUDA/NPU 可用时仍能完成；
- 确认输出包含 Parameter/Activation/Gradient/Optstate；
- 确认 FSDP rank shard 小于未分片参数量，且 forward all-gather 使峰值高于常驻量；
- 确认 AdamW step 后有 optimizer state；
- 确认 `rank=0/1` 均能独立运行。

### 9.3 真实设备校准测试

同一 YAML 分别运行 dry-run 和单步真实 eager，采集 `memory_allocated/max_memory_allocated`，按 8.2 的矩阵生成误差表。该测试需要真实 NPU/GPU，不作为普通 UT 前置条件，但在发布首个稳定版本前必须完成。

### 9.4 当前无卡验证基线

2026-07-18 在 PyTorch 2.13 CPU-only 环境，以仓库 `examples/qwen3_5_0_8b_base/train.yaml`（4 层、bf16、FSDP4、`comm_fusion=true`）从 `scripts/train_lm.py` 完整执行成功：逻辑峰值为 2,530,471,526 bytes，报告含 Parameter、Activation、Gradient、Temp 和 step 后 337,299,644 bytes Optstate。相同最小 FSDP 集成用例已在 PyTorch 2.7.1、2.9.1 和 2.13.0 CPU wheel 通过。该结果验证了入口、版本兼容和逻辑生命周期，不替代 9.3 的真实 NPU/GPU 误差校准。

同日在 8 × Ascend 910B3、`torch_npu 2.13.0` 环境，以相同 Qwen3.5 FSDP4 配置执行 NPU fake backend
成功，`simulation_device_type=npu`，逻辑峰值仍为 2,530,471,526 bytes，生成 4,957 条算子输出
storage 内存块 CSV（无 summary 行）。算子块生命周期峰值为 1,853,259,254 bytes，低于 MemTracker
峰值；110 条 persistent Optstate 合计 337,299,424 bytes，与 current Optstate 一致，未残留 persistent
Activation。该用例覆盖 torch_npu 注册、NPU device guard 与 NPU fake/meta kernel；它仍不执行数值
计算，也不替代 9.3 的真实 eager `max_memory_allocated` 校准。

2026-07-19 在相同 NPU 环境以 Qwen3.5 8 layers、FSDP4、bf16、micro batch 4、sequence 1024
运行全层 `activation_checkpoint=full` 成功。报告包含 20,417 条算子输出 storage，其中 forward
4,341 条、backward 11,458 条、recompute 4,186 条、optimizer 432 条；MemTracker 逻辑峰值为
15,067,528,806 bytes，算子生命周期峰值为 14,348,742,662 bytes。5 秒间隔 `npu-smi` 监控显示
NPU 0 无进程基线 HBM 为 3,423–3,424 MB，运行中为 3,487–3,488 MB，Python 进程口径为
118 MB，退出后恢复到 3,423 MB；因此 FakeTensor 的十几 GB 逻辑 tensor 未在真实 HBM 分配，
实际设备增量仅为 torch_npu runtime/context 带来的约 64–65 MB。

## 10. 分阶段交付

### Phase 1（本文实现）

- Trainer 配置与 `train_lm.py` 一键入口；
- Torch FakeProcessGroup/FakeTensorMode/MemTracker runner；
- FSDP 无卡基线与 full activation checkpoint；HSDP/TP/CP/EP/selective AC 当前 fail-fast，按模型
  逐项增加兼容性用例后转为正式支持；
- JSON 峰值、分类、模块快照和 OOM 判断；
- 无设备 UT/CPU fake integration test；
- PP、MindSpore、动态 shape 明确 fail-fast。

### Phase 2

- 完整 gradient accumulation；
- activation checkpoint full 的真实设备误差校准，以及 selective checkpoint 的 fake 集成与
  生命周期验证；
- PP rank/stage 分析与多 rank 批量报告聚合；
- 动态 shape 上界/分支 override；
- 自定义算子 fake/meta registration 清单。

### Phase 3

- allocator/碎片和 kernel workspace 校准系数；
- 与 auto-parallel 搜索器联动，批量筛除 OOM 策略；
- MindSpore 等价实现。

## 11. 完成定义

Phase 1 被认为可开发完成，需要同时满足：

- 配置开启后不调用真实 device `set_device`，无可用 NPU/GPU 的 Torch 环境可运行最小集成用例；
- 成功报告符合 schema，包含峰值、构成、模块热点、拓扑、限制和可选 OOM 判断；
- 不支持路径返回描述性错误且不产生伪成功数据；
- 所有新增 UT 通过，原有 Trainer 配置测试不回归；
- 至少完成一组真实设备校准；若当前环境无设备，功能实现可进入评审，但必须把校准明确记录为稳定版发布阻塞项，不能宣称已满足 5% 精度验收。
