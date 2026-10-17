# HyperParallel 静态图 FSDP 参数预取：接入、差异与验证

记录日期：2026-10-16。工作环境为服务器 `113.46.17.205:153` 的 `whh_titan` 容器，项目目录为 `/home/whh/github_graph_automodel_adapt`，分支为 `graph_automodel_adapt`。

本次将 FSDP 参数 all-gather 预取接入现有 joint graph pass 流程，完成了 TP2 + FSDP2 四卡 FP32 参数训练对照。确定性模式下，关闭和开启预取各训练 10 步，首个 micro-batch 梯度及最终参数的指纹全部一致。BF16 只完成前反向对照，完整优化器更新仍受到已有的 FP32 主参数延迟切分问题阻塞。

本文依据当前源码及 [验证记录目录](work/fsdp_prefetch_20261016/) 编写；没有将短时性能变化解释为已经测得设备通信与计算的物理重叠，也没有将 BF16 前反向通过解释为原生 FSDP 混合精度已适配。

## 1. 预取 pass 如何接入、如何使用

### 1.1 接入位置与调用链

现有 `PassPipeline` 已预留 `AutoOverlapPass`，此前其调度主体为空。本次让它调用新增的 `overlap_param_gather()`，并补齐训练配置到 `PassConfig` 的传递。

```text
TextTrainer 的首次 forward_backward
  └─ GraphCompiler 延迟 trace，生成 forward + backward joint FX graph
      └─ PassPipeline
          ├─ DeadCodeEliminationPass
          ├─ CanonicalizeGraphPass
          ├─ FSDPPass                    [fsdp_enabled=true]
          │   └─ 参数切分、all-gather/wait、梯度 reduce-scatter、reshard
          ├─ PpPass                      [pp_enabled=true]
          └─ AutoOverlapPass             [enable_overlap=true]
              └─ overlap_param_gather(graph_module, pass_config)
```

预取位于 FSDP pass 之后，因为此时图里才有可以调度的 FSDP 通信节点及其元数据。当前验证范围为 TP2 + FSDP2、PP=1；PP 切图后的 stage stub 不暴露内部 wait 节点，本次没有增加 PP stage 内部的预取调度。

| 文件 | 职责 |
| --- | --- |
| [pipeline.py](hyper_parallel/compile/passes/pipeline.py) | 原有 pass 编排，根据开关添加 `AutoOverlapPass` |
| [schedule.py](hyper_parallel/compile/passes/overlap/schedule.py) | `AutoOverlapPass.run()` 调用预取实现 |
| [overlap_param_gather.py](hyper_parallel/compile/passes/overlap/overlap_param_gather.py) | 新增的识别、锚点选择、预算判断、提前发起和 wait/view 延后逻辑 |
| [fsdp_pass.py](hyper_parallel/compile/passes/parallel/fsdp_pass.py) | 为参数 all-gather 提供通信类型、参数身份、前反向标记及完整参数字节数 |
| [pass_config.py](hyper_parallel/compile/pass_config.py) | pass 配置、校验及 Trainer 配置映射 |
| [build_options.py](hyper_parallel/models/build_options.py) | `CompileConfig` 的 overlap 开关与预算 |
| [train_lm_graph_tp2_fsdp2.yaml](hyper_parallel/compile/examples/automodel_text_graph/train_lm_graph_tp2_fsdp2.yaml) | 已开启预取的示例配置 |

### 1.2 实际调度了什么

预取 pass 只改变现有节点的顺序，不新增 all-gather、reduce-scatter 或 dtype 转换。目标是提前发起后续参数的 all-gather，并让 wait 尽量靠近参数真正被计算使用的位置。

```text
示意：未预取
  gather(A) → wait(A) → compute(A) → gather(B) → wait(B) → compute(B)

示意：预取后，且数据依赖、通信顺序和预算允许
  gather(A) → gather(B) → wait(A) → compute(A) → wait(B) → compute(B)
```

实现分为以下步骤：

1. **准确识别参数通信。** 只处理 `comm_type` 为 `fsdp_all_gather` 或 `fsdp_all_gather_backward` 的 functional all-gather，要求其唯一直接使用者为 `wait_tensor`。同时兼容算子 packet 和 `.default`，不会仅凭输入是 placeholder 就把 TP activation all-gather 当作参数通信。
2. **寻找首次真实消费。** 从 wait 遍历完整的 view/alias 分支 DAG，包括 split/getitem；按最早的非 view 使用者分组。output、storage free 等实际使用者也会约束移动位置。
3. **选择提前发起的锚点。** 同时计算“按消费组向前看”的锚点和 attention 区间内的重计算量算子锚点，再选更早的候选。这里的重计算量算子指 mm、addmm、bmm、matmul、linear 和 attention，不是激活重计算。
4. **应用约束和预算。** 前向、反向分别调度；遵守输入就绪、阶段边界、已有 functional collective 发起顺序、原地修改/未知副作用以及同一参数的 storage free 边界。不同参数的无关 free 可以跨越。保留所有已识别 functional collective 的全局发起顺序较为保守，也可能限制与 TP 通信并存时的提前空间。
5. **延后 wait/view。** 对完整 realization DAG 逆拓扑移动，每个节点仍在所有使用者之前，并尊重副作用和阶段边界。最终执行 `graph.lint()` 和 `recompile()`。

若 gather 前已有仅供其使用的 shard cast，且 cast 的输入都是 placeholder，可以一起移动 `prims.convert_element_type` 或 `aten._to_copy`。这只是移动已有转换，**不代表当前静态 FSDP 已实现 `param_dtype`/`reduce_dtype` 混合精度转换**。

### 1.3 配置与开关语义

Trainer YAML 可使用：

```yaml
fsdp_config:
  dp_shard_size: 2
  forward_prefetch_depth: 1
  backward_prefetch_depth: 1
compile:
  enabled: true
  use_joint_graph: true
  enable_overlap: true
  fsdp_prefetch_budget_bytes: 268435456  # 256 MiB
```

| 配置 | 默认值与含义 |
| --- | --- |
| `compile.enabled`、`compile.use_joint_graph` | 均为 `true` 时选择这里的 joint graph 训练路径 |
| `compile.enable_overlap` | Trainer 路径默认 `false`；指定示例 YAML 已设置为 `true` |
| `fsdp_config.forward_prefetch_depth` | 默认 1，映射到 `PassConfig.fsdp_forward_prefetch_depth` |
| `fsdp_config.backward_prefetch_depth` | 默认 1，映射到 `PassConfig.fsdp_backward_prefetch_depth` |
| `compile.fsdp_prefetch_budget_bytes` | 默认 256 MiB，映射到同名 pass 字段 |

直接使用编译 API 时，`PassConfig(enable_overlap=True, fsdp_forward_prefetch_depth=1, fsdp_backward_prefetch_depth=1, fsdp_prefetch_budget_bytes=256 << 20)` 即可配置。注意 `PassConfig.enable_overlap` 自身默认是 `true`，与 Trainer 的 `CompileConfig` 默认值不同；`build_pass_config_from_trainer_config()` 默认读取 Trainer 配置，显式传入的 `enable_overlap` 参数优先。

深度单位是静态图里的**首次消费组**，不是 eager FSDP 的模块数。attention 区间策略可以选择更早的锚点，因此 depth 也不是所有移动距离的硬上限。

- `enable_overlap=false`：关闭整个预取 pass，包括 wait/view 延后。
- 某阶段 depth 为 0：关闭该阶段所有提前发起，包括 attention 区间策略；wait/view 延后仍可执行。
- budget 为 0：关闭所有提前发起；wait/view 延后仍可执行。
- 深度和预算必须为非负整数，不接受 bool 或小数。

预算按阶段检查“已提前发起、但尚未首次消费”的完整参数副本字节数，并在不同锚点及两种选点策略之间共享。FSDP pass 在切分前记录 `fsdp_unsharded_nbytes`：在 TP+FSDP 场景中，它是 **TP 本地、尚未做 FSDP 切分的参数大小**。大小未知的参数保守地不预取。

这个预算不是总显存上限：参数首次使用之后仍可能存活，激活、梯度、常驻参数和通信运行时缓冲也不属于这个统计窗口。`reshard_after_forward=false` 时已有完整参数会保留更久，本次没有改变其生命周期语义。

### 1.4 当前环境中的实际运行方式

原示例 YAML 设置 `model.torch_dtype=bfloat16`，但默认没有开启 FP32 主参数，会触发第 4 节的训练入口限制。因此已通过的完整训练使用 `model_init_dtype=float32`，最终参数及计算均走 FP32；它不是“FP32 存储、BF16 计算”的混合精度验证。

在容器内执行：

```bash
cd /home/whh/github_graph_automodel_adapt
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto

torchrun --nproc_per_node=4 --master_port=29743 \
  scripts/train_lm.py \
  hyper_parallel/compile/examples/automodel_text_graph/train_lm_graph_tp2_fsdp2.yaml \
  --model.pretrained_model_name_or_path=/home/whh/used/hyper_dev/.assets/models/Qwen3-0.6B \
  --dataset.data_path=/home/whh/github_graph_automodel_adapt/hyper_parallel/compile/examples/automodel_text_graph/output/data/indexed_text_document \
  --model_init_dtype=float32 \
  --training.enable_full_determinism=true \
  --compile.enable_overlap=true
```

将最后一项改为 `false` 即可运行关闭预取的基线。HCCL 端口环境变量用于解决该服务器的默认端口绑定冲突。

编译时应能看到类似日志：

```text
FSDP parameter prefetch: forward=309 backward=309 sunk=12 budget_bytes=268435456
```

编译后的 `GraphModule.meta["fsdp_prefetch"]` 也会保存统计。`forward`/`backward` 是该次 pass 实际提前移动的 gather 数；`sunk` 是移动的 wait/view **节点数**，不是参数数或 wait 链数。统计不保证每个移动的 gather 都跨过计算节点，具体证据见第 3 节。

## 2. 与参考 PR 的相同点和差异

参考为 [Ascend/pytorch PR #38283](https://gitcode.com/Ascend/pytorch/pull/38283/diffs)，本次下载并比较的提交为 `c35fe933c8d42c892659ddc6dbdfcd5ffe8b1036`。下表针对这份固定快照，不代表 PR 后续版本。

服务器保留了 [原始预取实现](work/fsdp_prefetch_20261016/reference/overlap_param_gather.py)、[原始测试](work/fsdp_prefetch_20261016/reference/test_overlap_param_gather.py) 和 [初始化文件 diff](work/fsdp_prefetch_20261016/reference/fx_passes_init.diff)。原始路径为 `torch_npu/_inductor/fx_passes/overlap_param_gather.py`。

两者的核心思路一致：按参数首次真实消费分组，提前发起 all-gather，结合 attention 区间分配计算锚点，再把 wait/view 放到接近使用的位置；支持将私有 shard cast 随 gather 移动。当前实现是针对 HyperParallel joint graph 的适配，存在以下行为差异。

| 维度 | 参考 PR 固定快照 | 当前 HyperParallel 实现 |
| --- | --- | --- |
| 集成位置 | torch_npu Inductor FX pass | `FSDPPass` 之后的 `AutoOverlapPass`，接入 Trainer YAML |
| 通信识别 | 根据 gather 输入是否为 placeholder 或其 cast 推断参数通信 | 必须带 FSDP `comm_type`，避免误调度 TP activation gather |
| 前反向处理 | 默认 `forward_only=true`；检测到 `tangents_*` placeholder 时跳过，认为可能是 backward/joint graph | 使用 `autograd_backward` 元数据，在同一 joint graph 中分别调度前向和反向，深度独立 |
| 锚点策略 | 先执行 interleaved 移动，再执行 attention 区间移动，同一 gather 可能移动两次 | 先合并两类候选，再按共同约束与预算移动一次 |
| depth 大于 1 时的早期组 | interleaved 从第 `lookahead` 个组开始 | 除首组外，早期组可选择 `max(0, index-depth)` 对应的组 |
| 预算默认值和覆盖范围 | 默认 4 GiB；仅约束 attention 区间策略，每个计算锚点重新计数 | 默认 256 MiB；两类策略共同受约束，检查阶段内跨锚点的未首次消费副本总量 |
| 大小未知 | 可以移动，但不计入已知字节预算 | 不提前发起该参数 |
| 关闭语义 | `lookahead=0` 仅关闭 interleaved；attention 策略仍可运行，其预算不约束 interleaved | 阶段 depth=0 关闭该阶段所有提前发起；budget=0 关闭全部提前发起 |
| wait/view 分支 | 遍历分支后选取通向最早真实使用者的一条 wait/view 链，外部分支使用者可能阻止整链移动 | 收集完整分支 DAG，逆拓扑逐节点延后，支持 split/getitem 和 schema alias |
| 私有 cast | 支持 `prims.convert_element_type` | 还支持 `aten._to_copy`；两者均只移动已有 cast |
| 顺序和生命周期约束 | 检查输入依赖、wait 链外部使用者等；没有当前这些显式边界检查 | 额外保留 functional collective 发起顺序、阶段边界、副作用以及同一参数的 free 边界 |
| 依赖和图校验 | 使用 Inductor 内部的识别 helper、`stable_topological_sort` | 不依赖这些 Inductor 内部 helper；基于现有拓扑安全移动并执行 `graph.lint()` |
| 可观测统计 | 日志分别统计 interleaved、interval 的移动次数和 wait 链数 | 日志及 `gm.meta` 统计前/反向 gather 移动数、wait/view 节点数，并标记 gather 的候选 anchor |

这些差异使配置不能机械地一一换算。例如，PR 的 4 GiB 是某个锚点的 attention 策略预算，不能等同于当前整个阶段的 4 GiB 未消费预取窗口；两份代码的“changed”和“sunk”计数口径也不同。

## 3. 如何验证预取 pass 生效

验证分为调度行为、数值正确性、真实四卡训练三层。证据均保存在 [work/fsdp_prefetch_20261016](work/fsdp_prefetch_20261016/)，没有只根据开关或日志判断成功。

### 3.1 单测和 CPU 数值回归

```bash
python -m pytest tests/ut/compile tests/ut/auto_models/test_build_options.py -q
python -m pytest tests/ut/compile/test_overlap_param_gather.py tests/ut/compile/test_auto_overlap.py -q
```

第一组为 230 passed，最终 overlap 专项复核为 20 passed。记录见 [compile_ut.log](work/fsdp_prefetch_20261016/compile_ut.log) 和 [final_overlap_ut.log](work/fsdp_prefetch_20261016/final_overlap_ut.log)。

测试覆盖 FSDP 元数据过滤、前反向独立开关、预算为零/不足/未知、跨锚点预算、分支 view 和 split/getitem、输入就绪、原地修改边界、TP collective 顺序、私有 cast、重复运行以及配置校验。

CPU 数值回归使用模拟 collective，但执行真实的参数 storage 释放；对多层 MLP 和多输出 view 场景分别检查 `reshard_after_forward=true/false`，与未切分参考比较 loss 和每个参数梯度，容差为 `1e-12`。这部分检查调度是否破坏数据依赖、反向重建或 storage 生命周期。

### 3.2 检查真实训练图的节点移动

FP32 四卡训练中，每个 rank 的统计为：

| 项目 | 结果 |
| --- | ---: |
| joint graph 节点总数 | 10007 |
| 前向 FSDP all-gather 总数 / 实际提前移动数 | 311 / 309 |
| 反向 FSDP all-gather 总数 / 实际提前移动数 | 310 / 309 |
| 延后的 wait/view 节点数 | 12 |

进一步对比保存的关闭/开启预取图，而不只读取 pass 自报统计，得到：

- 节点名称集合相同，没有新增或删除通信/计算节点。
- 1050 个 collective 的发起顺序保持一致。
- 308 个重计算量算子的相对顺序保持一致。
- 前向 28 个、反向 168 个 gather 的移动实际跨过了重计算量算子。

详见 [graph_comparison.json](work/fsdp_prefetch_20261016/graph_comparison.json)、[基线图](work/fsdp_prefetch_20261016/baseline/joint_graph.py) 和 [预取图](work/fsdp_prefetch_20261016/prefetch/joint_graph.py)。这证明提前发起已经作用于真实图，并创建了可与计算重叠的调度窗口；309 次移动不等于 309 次都跨过了计算。

本次没有采集设备 profiler 时间线，因此没有直接测量 HCCL kernel 与计算 kernel 的物理重叠比例。

### 3.3 TP2 + FSDP2 四卡训练与数值对照

测试使用 4 张 Ascend NPU，Qwen3-0.6B 结构、28 层，seq_length=128，global_batch_size=4，micro_batch_size=1，seed=42，AdamW 学习率 `1e-4`，训练 10 步。

模型资产目录只有配置和 tokenizer，没有预训练权重；因此是随机初始化模型，数据为服务器上的真实本地 indexed 文本数据。本次验证训练流程及相同初始化下的数值一致性，不代表预训练模型质量评估。

普通模式下，关闭/开启预取均完成 10 步，退出码均为 0；日志中的 loss 和 grad norm 曲线相同，但张量指纹未全部一致。重复运行关闭预取的基线也出现 rank 0 embedding 首梯度的位级差异，说明需要控制运行本身的非确定性。这里没有把未量化的差异直接归为可接受误差。

随后设置 `training.enable_full_determinism=true` 重新做严格对照：

| 检查 | 结果 |
| --- | --- |
| 关闭/开启预取 | 各完成 10 步，退出码均为 0 |
| 首个 micro-batch 梯度 | 4 个 rank，各 311 个张量，SHA-256 指纹逐一一致 |
| 第 10 步优化器更新后的模型参数 | 4 个 rank，各 311 个张量，SHA-256 指纹逐一一致 |
| 有限性 | 所检查的梯度、参数均有限 |
| loss 和 grad norm | 10 步日志曲线一致 |

这里采集的是**首个 micro-batch**梯度，以及 10 步后的全部参数，没有保存并逐一比较每个训练 micro-batch 的梯度。证据见 [deterministic_summary.json](work/fsdp_prefetch_20261016/deterministic_summary.json)、[确定性基线记录](work/fsdp_prefetch_20261016/deterministic_baseline/) 和 [确定性预取记录](work/fsdp_prefetch_20261016/deterministic_prefetch/)。

确定性模式单次短训练的 rank 0 日志如下，单步时间取第 2–10 步中位数，排除包含 trace 和首批指纹采集的第 1 步：

| 指标 | 关闭预取 | 开启预取 |
| --- | ---: | ---: |
| 单步时间中位数 | 0.555940 s | 0.534696 s |
| 日志峰值 allocated GB | 4.59269 | 4.60884 |
| 日志峰值 reserved GB | 5.89258 | 5.95898 |
| 第 1 步 loss | 12.0963 | 12.0963 |
| 第 10 步 loss | 10.4612 | 10.4612 |

本次短跑单步时间约减少 3.8%，同时显存有所增加。样本仅为单次运行的 9 个稳态步骤，尚不能据此认定稳定性能收益。

复现采集与严格比较可执行：

```bash
bash work/fsdp_prefetch_20261016/run_deterministic_validation.sh
PREFETCH_COMPARISON_PREFIX=deterministic_ \
  python work/fsdp_prefetch_20261016/compare_results.py
```

脚本会复用同名输出目录；需要保留原始证据时，先复制脚本并修改输出目录。采集脚本使用相同的 TextTrainer 训练流程，额外保存图、首批梯度和最终参数指纹。

### 3.4 BF16 验证的范围

BF16 基线在优化器报错前已经成功执行前反向，可取得首个 micro-batch 梯度。开启预取后，验证脚本使用 `PREFETCH_VALIDATE_BACKWARD_ONLY=1` 显式跳过优化器更新，完成一个训练迭代的前反向。

4 个 rank 各 311 个首批梯度指纹与该基线一致，均有限；预取统计为 forward=310、backward=309、sunk=12。证据见 [bf16_backward_comparison.json](work/fsdp_prefetch_20261016/bf16_backward_comparison.json)、[BF16 前反向日志](work/fsdp_prefetch_20261016/bf16_prefetch_backward/train.log) 和 [validate_train.py](work/fsdp_prefetch_20261016/validate_train.py)。

跳过 step 仅发生在 work 目录的验证脚本中，不是生产训练修复。BF16 的完整 10 步训练及优化器更新没有通过。

## 4. FP32 主参数延迟切分导致的 shape 报错

### 4.1 首先遇到的是训练入口限制

原 YAML 构建 BF16 模型，`optimizer.fp32_main_params` 默认关闭。[model_builder.py](hyper_parallel/models/_transformers/model_builder.py) 的 `_validate_optimize_dtype()` 会拒绝这种低精度参数直接优化的组合。这是当前项目训练入口的约束，不能推导为 PyTorch 的优化器普遍不支持 BF16 参数。

打开 `optimizer.fp32_main_params=true` 后，[TrainerConfig](hyper_parallel/trainer/config/trainer.py) 又要求 `fsdp_config.mix_precision.reduce_dtype='float32'`。满足该校验后，训练才继续到后面的 shape 错误。

需要区分配置校验和图实现：当前静态 `PassConfig` 没有承接 `mix_precision.param_dtype/reduce_dtype`，FSDP pass 也没有据此插入参数及梯度 cast。因此，设置 `reduce_dtype=float32` 在这次失败实验中满足了入口校验，**不能据此认定静态图的 reduce-scatter 已按 FP32 执行**。

### 4.2 为什么会同时存在 BF16 模型参数和 FP32 主参数

[mixed_precision_optimizer.py](hyper_parallel/components/optim/mixed_precision_optimizer.py) 中的 `Float16OptimizerWithFloat16Params` 对低精度模型参数创建独立的 FP32 主参数：

```python
main_param = nn.Parameter(
    model_param.detach().to(dtype=torch.float32),
    requires_grad=model_param.requires_grad,
)
```

优化器 parameter group 改为引用 `main_param`，模型仍引用原来的 BF16 `model_param`。训练时先将模型梯度转换并交给主参数，优化器更新 FP32 主参数，然后将更新结果复制回 BF16 模型参数。这是当前优化器包装层的持久双份参数设计。

对本来就是 FP32 的模型参数，包装层采用 `model_param.main_param = model_param`，不会再创建独立的 FP32 参数副本。另需区分 AdamW 的动量状态：它们是优化器状态，不是这里讨论的模型参数与主参数两份权重。

### 4.3 shape 如何发生分叉

[TextTrainer](hyper_parallel/trainer/text_trainer.py) 先构建模型和优化器，再构建 GraphCompiler。而 [GraphCompiler](hyper_parallel/compile/compiler.py) 真正 trace 和运行 FSDP pass 的时机，是首次 `forward_backward()`。仅仅提前构造 GraphCompiler 对象，并不会提前完成物理切分。

以本次 embedding 为例：词表 151936、hidden size 1024，TP=2 后每个 TP 分片为 `(75968, 1024)`；再做 FSDP=2 的 dim 0 切分，每个本地 shard 应为 `(37984, 1024)`。

| 时机 | BF16 模型参数 | 独立 FP32 主参数 | 梯度 |
| --- | --- | --- | --- |
| TP 已生效，构建优化器包装层 | `(75968, 1024)` | 此时克隆为 `(75968, 1024)` | 尚未生成 |
| 首次运行静态 FSDP pass | `.data` 切为 `(37984, 1024)` | 仍为 `(75968, 1024)` | 尚未生成 |
| backward + reduce-scatter 后 | `(37984, 1024)` | 仍为 `(75968, 1024)` | `(37984, 1024)` |
| 向主参数交付梯度 | 本地 shard | 期待完整 TP 分片大小 | shape 校验失败 |

FSDP pass 的 `_shard_live_model_params()` 修改的是模型参数的 `.data`：

```python
param.data = param.detach().chunk(self._fsdp_degree, dim=0)[rank].clone()
```

独立的 FP32 主参数在此之前已经复制完成，不会跟着另一个 Parameter 的 `.data` 改变。最后 `_copy_model_grads_to_main_grads()` 调用 `_gradient_for_param()`，发现梯度与优化器参数的本地形状不同，抛出：

```text
Optimizer gradient local shape must match its parameter shard:
got (37984, 1024) and (75968, 1024)
```

这不是 dtype cast 能修复的问题：cast 只能改变数据类型，不能使一个 FSDP shard 变成切分前的参数。错误发生在准备优化器梯度阶段，尚未正常执行该步优化器更新。

报错在 **`compile.enable_overlap=false`** 的基线已经复现，见 [baseline_master_param_shape_failure/train.log](work/fsdp_prefetch_20261016/baseline_master_param_shape_failure/train.log)。因此当前证据指向“主参数创建早于静态 FSDP 物理切分”的生命周期问题，不是本次预取节点重排引入的问题。

### 4.4 为什么 FP32 基线可以完成训练

已通过的实验使用 `model_init_dtype=float32`，且没有开启低精度参数的独立 FP32 主参数路径。优化器引用模型中的同一个 FP32 Parameter，FSDP 对其 `.data` 做切分后，优化器看到的也是切分后的形状；AdamW 状态在首次 step 时才按这个 shard 初始化。因此没有独立主参数留在旧 shape 的问题。

这解释了为何 FP32 完整训练可以用来验证预取，又不能用来宣称 BF16 混合精度训练问题已经解决。

### 4.5 与原生 FSDP2 按需 dtype 转换的区别

用户提出的“FSDP 应在需要参数时转换 dtype”对应的是原生 FSDP2 的典型用法：模型持久参数先保持 FP32，设置 `MixedPrecisionPolicy(param_dtype=bfloat16, reduce_dtype=float32)`。原始 sharded 参数保留 FP32，优化器直接更新它；FSDP 为通信和计算准备临时低精度参数，梯度按配置使用 FP32 做归约。参见 [PyTorch MixedPrecisionPolicy 文档](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.MixedPrecisionPolicy)。

```text
典型原生 FSDP2 路径（原始参数为 FP32）
  常驻 FP32 shard ── cast ──> BF16 shard ── all-gather ──> BF16 完整计算参数
        ↑                                                    │
        └── optimizer 更新 ← FP32 shard 梯度 ← FP32 归约 ← backward
```

这种路径无需为了优化器再长期保存一份独立的 FP32 权重：常驻 FP32 shard 本身就是优化器参数。执行过程中，常驻 FP32 shard 与临时 BF16 通信/计算缓冲仍会同时存在，后者按 reshard 生命周期释放。`param_dtype` 决定计算/通信时的参数 dtype，不会自动把原本以 BF16 初始化的持久参数提升为 FP32。

本次失败使用的则是“常驻 BF16 模型参数 + 外部优化器包装层复制的常驻 FP32 主参数”。当前静态 FSDP pass 尚未实现上面的完整混合精度路径，所以不能把两者当成同一个机制。

### 4.6 后续修复范围

以下为待开发内容，本次没有实现：

1. **修复主参数的切分时序。** 在模型完成物理 FSDP 切分后创建/绑定 FP32 主参数，或同步处理主参数及相关优化器状态的切分，确保 shape、device、layout 一致；不能只调整编译器对象的构造顺序。
2. **补齐静态 FSDP 混合精度。** 若目标是原生 FSDP2 的 FP32 常驻 shard + BF16 计算，需要把策略传入 pass，实际插入 all-gather 前的参数 cast 和 reduce-scatter 前的梯度 cast，并处理梯度回填 dtype、输入输出 dtype、反向重建及 storage 生命周期。预取预算也应按实际通信副本的 dtype 重新计算。
3. **重新验证完整 BF16 更新。** 在上述机制成立后，再做关闭/开启预取的完整训练对照；目前的 BF16 backward-only 记录只能证明已验证图上的前反向调度没有改变所采集梯度。

## 附：源码和证据定位

| 内容 | 路径 |
| --- | --- |
| 本次实现补丁，不含原有用户改动 | [fsdp_prefetch.patch](work/fsdp_prefetch_20261016/fsdp_prefetch.patch) |
| 已验证的 9 个变更源码文件 SHA-256 | [source_manifest.json](work/fsdp_prefetch_20261016/source_manifest.json) |
| 任务开始前已有 tracked 改动备份 | [preexisting.diff](work/fsdp_prefetch_20261016/preexisting.diff) |
| 完整采集逻辑 | [validate_train.py](work/fsdp_prefetch_20261016/validate_train.py) |
| 确定性训练脚本 | [run_deterministic_validation.sh](work/fsdp_prefetch_20261016/run_deterministic_validation.sh) |
| 结果比较脚本 | [compare_results.py](work/fsdp_prefetch_20261016/compare_results.py) |
| BF16 前反向脚本 | [run_bf16_backward_validation.sh](work/fsdp_prefetch_20261016/run_bf16_backward_validation.sh) |

原补丁与源码清单记录的是已验证的实现文件，不包含本文。本次验证未修改优化器实现；项目原有用户改动独立保留。
