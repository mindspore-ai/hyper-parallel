# HyperParallel GraphCompiler 重计算 Pass 开发与使用说明

本文说明本次开发的重计算 pass 做了什么、怎样开启、如何阅读核心代码，以及它与 HyperParallel 动态图和本地 TorchTitan GraphTrainer 实现的关系。

文档核对日期：2026-10-13。对应仓库 `/home/whh/sac_pass`，基线 commit 为 `1c04dcde3a93cf541628795dab8406e35d8b42d2`，包含 AutoModel + GraphCompiler 适配 commit `84deba0321e912b1438e49a3e7293339fe45a833`。重计算实现目前是该基线之上的工作区修改。本文描述当前版本，不代表后续上游版本已经具备全部这些功能。

## 1. 开发内容与设计目标

重计算的目标是缩短部分前向激活的存活时间：反向需要某个中间结果时，重新执行产生它的计算，而不是一直保留该结果。

本实现作用于 GraphCompiler 已经捕获的联合前向、loss、反向 FX 图。它先决定哪些前向节点需要保存、哪些允许重算，再把反向实际需要的计算复制到反向消费者之前。它不会为模型添加动态图 checkpoint wrapper，也不会重新生成整个 autograd 图。

主要代码如下，链接均相对于仓库根目录：

| 文件 | 本次开发或修改的内容 |
| --- | --- |
| [recompute_config.py](hyper_parallel/compile/recompute_config.py) | 独立于 Torch 的 `RecomputeConfig`，定义模式、区域选择、算子保存和自定义策略 |
| [recompute_pass.py](hyper_parallel/compile/passes/recompute/recompute_pass.py) | `RecomputePass`：策略标注、安全约束、子图复制、反向输入重接和统计 |
| [pipeline.py](hyper_parallel/compile/passes/pipeline.py) | 按配置把重计算插入图清理之后、FSDP 之前 |
| [pass_config.py](hyper_parallel/compile/pass_config.py) | `PassConfig.recompute`，以及 Trainer 配置到 pass 配置的转换 |
| [compiler.py](hyper_parallel/compile/compiler.py) | 暴露 `GraphCompiler.recompute_report`，返回统计副本 |
| [build_options.py](hyper_parallel/models/build_options.py) | `CompileConfig.recompute`，支持 YAML 配置入口 |
| [trainer.py](hyper_parallel/trainer/config/trainer.py) | 拒绝与 eager checkpoint/swap、PP 的不支持组合 |
| [compile/__init__.py](hyper_parallel/compile/__init__.py) | 导出 `RecomputeConfig` |
| [train.py](hyper_parallel/compile/examples/recompute/train.py) | 小模型 eager 与 graph 对照示例，验证 loss、梯度、参数更新和 RNG |
| [prepare_automodel.py](hyper_parallel/compile/examples/recompute/prepare_automodel.py) | 从已有 AutoModel 示例派生确定性验证配置和 indexed token 数据 |
| [validate_automodel.py](hyper_parallel/compile/examples/recompute/validate_automodel.py) | 观察现有 TextTrainer 训练步骤，记录逐 rank、逐参数的完整校验值 |
| [test_recompute_pass.py](tests/ut/compile/test_recompute_pass.py) | 重计算正确性、安全边界及 FSDP 组合测试 |

此外，适配了新版 `forward_backward()` 返回 `(loss, loss_dict)` 的接口，增加辅助 loss 输出与配置投影测试，修正了一个上游 tracer 测试的旧式返回值解包；也修复了 [AutoModel 启动脚本](hyper_parallel/compile/examples/automodel_text_graph/run.sh) 引用不存在的 YAML 文件名的问题，并支持转发 CLI 参数。

## 2. 与 GraphCompiler 流程的连接

现有 AutoModel 训练入口保持不变：

```text
scripts/train_lm.py
  → parse_training_args() → TrainerConfig
  → TextTrainer → BaseTrainer._build_graph_compiler()
  → GraphCompiler(trainer_config=...)
  → build_pass_config_from_trainer_config()
  → PassConfig(recompute=RecomputeConfig(...))
```

编译期 pass 顺序为：

```text
联合前向 + loss + 反向图
  → DeadCodeEliminationPass / CanonicalizeGraphPass
  → RecomputePass（mode != none 时）
  → FSDPPass（启用时）
  → PpPass（启用时；当前禁止与重计算同时开启）
  → AutoOverlapPass（启用时）
  → 执行变换后的联合图
```

**重计算必须先于 FSDP 的参数生命周期规划。** 新增的反向计算也会读取参数；先插入这些节点，FSDP 才能把它们纳入反向 all-gather、参数释放和 view 重建的规划。否则可能出现参数已经释放，而重计算又尝试读取旧存储的问题。

当前不支持在同一个已编译图上反复叠加该 pass。修改策略、区域或输入形状后，应重新追踪/建立编译器；重复运行 pass 会明确报错。

## 3. 与动态图重计算的功能对照

### 3.1 先区分三条路径

1. **Hyper 动态图 checkpoint**：在运行时建立 checkpoint 区域，通过 saved-tensor hooks、重执行及 SAC 缓存恢复反向需要的张量。
2. **动态图 checkpoint 的 `torch.compile` 适配**：通过 native non-reentrant checkpoint 和 selective context 接入 Torch 编译。这不是本文的 graph pass。
3. **本文 GraphCompiler pass**：面对联合 FX 图，根据节点元数据进行静态改写，执行时直接运行改写后的图。

对应源码入口是 [distributed/activation_checkpoint.py](hyper_parallel/distributed/activation_checkpoint.py)、[activation_memory/api.py](hyper_parallel/core/activation_memory/api.py)、[checkpoint.py](hyper_parallel/core/activation_memory/checkpoint.py)、[sac.py](hyper_parallel/core/activation_memory/sac.py) 和 [compile_adapter.py](hyper_parallel/core/activation_memory/compile_adapter.py)。

动态图策略函数的形态是 `policy_fn(ctx, op, *args, **kwargs)`，`ctx.is_recompute` 区分前向与重执行；图 pass 的策略函数则是 `policy_fn(fx_node)`，在编译时读取算子及元数据。二者不能直接互换。

### 3.2 功能表

| 功能 | 当前 Hyper 动态图实现 | 当前 GraphCompiler pass |
| --- | --- | --- |
| full 重计算 | 重执行选中的 checkpoint 区域，具体包装方式受模型结构影响 | 重算区域内反向需要的安全节点；随机、通信、副作用等仍保存，因此不是严格等价的 full |
| selective 重计算 | 通过 dispatch/cache 决定保存或重执行算子输出 | 静态标注后复制需要重算的节点 |
| matmul 交替策略 | 每次 checkpoint context 创建独立计数器，分别记录前向和重执行 | `selective_policy=eager` 按匹配到的区域 FQN 统计图中 matmul 节点 |
| 保存昂贵算子 | 动态解析算子对象，结合算子集合及可用扩展 | 静态名称集合加 attention 名称识别；`save_compute` 保存所有所识别的昂贵算子 |
| 指定层/子模块 | 自动发现重复 block 容器；支持 model adapter safe-region 选择 | `module_patterns` 显式匹配模块完整路径；不自动消费 adapter 声明 |
| 排除子模块 | `checkpoint_exclude_wrapper` 有运行时输出缓存、输入句柄等机制 | `exclude_module_patterns` 阻止该子树节点被复制；不是同一套 exclude 缓存语义 |
| RNG / dropout | 默认保存、恢复 RNG 状态，在重执行中复现随机过程 | 不重放 RNG；保存随机结果，确定性的消费者仍可能重算 |
| 原地修改与 alias | SAC 包含缓存版本校验等机制 | 基于算子 schema 保守合并 alias 组，保护被写入存储、别名和直接读取者 |
| early stop | `early_stop` 在需要的张量恢复后停止运行时重执行 | 只复制反向依赖闭包，无需逐步重跑整个 Python 函数；不提供同名开关 |
| 自定义策略 | 可读取实际调用参数和运行阶段 | 可读取 FX Node 元数据；不能依赖未捕获的运行时数值 |
| MUST/PREFER 策略 | 由对应 checkpoint/cache 或编译适配路径消费 | 两种 SAVE 都保存，两种 RECOMPUTE 都进入复制候选；没有成本求解器 |
| CPU swap / 输入 swap / group swap / prefetch | 有对应 offload、reload、pool、wrapper 与管理机制，受具体模式限制 | 未实现；`MUST_SWAP` 明确拒绝 |
| 自定义算子 / 高阶算子 / 通信 | 有扩展算子策略、忽略项及模型相关适配 | 保守保存，未实现递归子图重计算或通信重放 |
| KV sharing / cache 副作用 | 高层接入会检测部分场景并选择子模块包装或调整 cache 配置 | 没有自动等价机制；schema 不可见的 Python 副作用不在支持范围内 |
| 重入 checkpoint | 当前 Hyper 底层实现明确只支持 `use_reentrant=False` | 不提供重入运行时；不能把它列为已覆盖或“比动态图缺失”的能力 |
| `context_fn` / recompute session | 动态图管理前向与重执行上下文、会话状态 | 不重建这些 Python 执行机制 |
| PP | 属于动态运行时及模型包装的组合能力，需按具体场景验证 | 当前显式拒绝 PP + recompute |

### 3.3 三个容易误解的差异

**交替 matmul 只借鉴策略意图，不保证逐算子相同。** 图捕获可能把 linear 分解成 mm、view 等节点；同一模块被多次调用时，图 pass 按 FQN 累加，而不是按每次 checkpoint 调用重新计数。`addmm` 和 `bmm` 在图的两种 selective 策略中均保存。

**昂贵算子集合不是从动态图完整复用的。** 动态图会查询可用的计算密集算子及扩展；图 pass 使用 `_MATMUL`、`_EXPENSIVE` 等本地集合。动态图对 `topk` 还有因原地使用风险而强制重算的规则；图 pass 未照搬该特例，最终行为由普通策略与 alias 安全屏障共同决定。因此不能声称 `save_compute` 与动态图 compile adapter 的所有算子选择完全一致。

**“排除”也不等价。** 图 pass 只是让被排除模块的节点不进入复制集合，同时保存跨区域的前向值；它没有动态图 excluded-region 的输出队列、重计算输入绑定和嵌套调用管理。

## 4. Pass 开启方法

### 4.1 通过 AutoModel / TextTrainer YAML 开启

在现有训练配置中加入：

```yaml
compile:
  enabled: true
  use_joint_graph: true
  recompute:
    mode: selective
    module_patterns: ["model.layers.*"]
    selective_policy: eager

activation_checkpoint:
  mode: off
activation_swap: none
```

Qwen3 的 decoder 路径是 `model.layers.*`；小示例是 `layers.*`。应根据实际模型的模块路径配置，不要把默认值当作所有模型通用的路径。

模式对应关系：

```yaml
# 关闭重计算，仍可保留 GraphCompiler
recompute: {mode: none}

# 尽可能重算安全算子
recompute: {mode: full, module_patterns: ["model.layers.*"]}

# 交替保存/重算 matmul
recompute: {mode: selective, module_patterns: ["model.layers.*"], selective_policy: eager}

# 保存所有识别到的昂贵算子
recompute: {mode: selective, module_patterns: ["model.layers.*"], selective_policy: save_compute}
```

这些 `recompute` 字段均应放在 `compile` 下。**`save_compute` 是 `selective_policy` 的值，不是 `mode` 的值。** 未配置重计算时默认 `none`。

CLI 同样可以覆盖：

```bash
bash hyper_parallel/compile/examples/automodel_text_graph/run.sh \
  --model.pretrained_model_name_or_path=/path/to/model \
  --dataset.data_path=/path/to/indexed_text_document \
  '--compile.recompute={mode: full, module_patterns: [model.layers.*]}'
```

该命令用于展示配置入口；模型权重加载、tokenizer、优化器精度仍必须满足原训练流程要求。当前服务器上已验证的 Qwen3 随机初始化、FP32 配置见第 7 节。

### 4.2 直接使用 GraphCompiler API

下面的 `model`、`train_fn`、`optimizer`、`batch` 由训练程序建立。`train_fn(model, **batch)` 返回标量 loss 或 `(loss, loss_dict)`：

```python
from hyper_parallel.compile import GraphCompiler, PassConfig, RecomputeConfig

compiler = GraphCompiler(
    model=model,
    train_fn=train_fn,
    device=device,
    pass_config=PassConfig(
        fsdp_enabled=False,        # 此片段以单卡为例
        enable_overlap=False,
        recompute=RecomputeConfig(
            mode="selective",
            module_patterns=("layers.*",),
            exclude_module_patterns=("layers.0",),
            selective_policy="eager",
        ),
    ),
)

optimizer.zero_grad(set_to_none=True)
loss, loss_dict = compiler.forward_backward(**batch)
optimizer.step()
print(compiler.recompute_report)
```

`forward_backward` 已执行反向，不要再对返回的 loss 调用一次 `backward()`。首个调用可触发惰性编译；模型应使用没有额外 eager checkpoint/swap 包装的形态。

### 4.3 配置项、匹配规则与自定义策略

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `mode` | `none` | `none` / `full` / `selective` |
| `module_patterns` | `("layers.*",)` | 模块 FQN 的 glob；最外层匹配祖先定义区域 |
| `exclude_module_patterns` | `()` | 排除匹配模块及其子树 |
| `selective_policy` | `eager` | `eager` / `save_compute`，仅影响 selective 的默认策略 |
| `save_ops` | `()` | 要额外保存的精确算子字符串，例如 `aten.mm.default` |
| `policy_fn` | `None` | Python 回调 `(fx.Node) -> CheckpointPolicy`，用于替代内置策略 |

每个 include pattern 都必须匹配追踪到的模块；每个 `save_ops` 条目都必须出现在所选区域，拼错会报错。exclude pattern 当前没有同等的“必须匹配”校验。`model.layers.*` 使用普通 glob 语义，不是正则表达式或专门的层号语法。

例如只选择前两层并排除某个 attention 子树：

```yaml
recompute:
  mode: full
  module_patterns: ["model.layers.0", "model.layers.1"]
  exclude_module_patterns: ["model.layers.1.self_attn"]
```

自定义回调示例：

```python
from hyper_parallel.core.activation_memory.policy import CheckpointPolicy

def graph_policy(node):
    if str(node.target) == "aten.mm.default":
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE

recompute = RecomputeConfig(
    mode="selective",
    module_patterns=("model.layers.*",),
    policy_fn=graph_policy,
)
```

回调通过 Python API 提供，不能直接把 Python 函数正文写进 YAML。它替代内置策略选择，不会自动叠加 `save_ops` 的保存决策；需要的额外保存逻辑应写入回调。配置中的非空 `save_ops` 仍会被检查是否匹配算子。

安全约束在策略之后生效：`PREFER_RECOMPUTE` 可被改为保存；用户明确对不安全节点返回 `MUST_RECOMPUTE` 则报错。回调可返回 Hyper 或 Torch 对应的 SAVE/RECOMPUTE enum，`MUST_SWAP` 不支持。

## 5. 核心代码走读

本节沿 [recompute_pass.py](hyper_parallel/compile/passes/recompute/recompute_pass.py) 的执行顺序说明。

### 5.1 `RecomputePass.run`：入口与约束

入口先调用 `pass_config.validate()`；`none` 直接返回。它拒绝重复变换，并要求图中存在 `autograd_backward` 元数据。之后调用 `_tag_policy`，把策略写入节点 `meta["recompute"]`，计算改写前的保留激活估计，调用 `_rematerialize`，最后执行 `graph.lint()`、`recompile()` 和统计。

只有策略标注而没有图复制，不会自动减少激活：真正改变图中值的使用位置的是 `_rematerialize`。

### 5.2 `_regions`：找到前向重计算区域

该函数只处理没有 `autograd_backward` 标记的 `call_function` 节点：

1. 从 `nn_module_stack` 提取祖先模块 FQN。
2. 统计每个 include pattern 是否匹配。
3. 任何祖先命中 exclude pattern，就排除该节点。
4. 取模块栈中最外层的匹配祖先作为区域 ID。

这样，一个 Transformer block 内的 linear、激活、归一化节点可归入同一区域；模块边界由捕获的元数据确定，而不是猜测 FX 节点名称。

### 5.3 `_default_policy` / `_normalize_policy`：选择保存还是重算

内置策略先检查 `save_ops`。full 对其余节点返回 `PREFER_RECOMPUTE`；selective 对普通节点也优先重算，但会保存昂贵算子。eager 策略对 `_MATMUL` 集合中的节点计数：奇数次保存，偶数次允许重算；save_compute 则全部保存。

自定义策略先经 `_normalize_policy` 转换为 Hyper enum。实现直接把两种 RECOMPUTE 放入同一个集合，不运行 min-cut、动态规划或按显存预算寻找方案。

### 5.4 `_mutation_barriers` / `_safety_reason`：安全优先

`_mutation_barriers` 遍历前向和反向节点，根据 schema 的 alias / write 信息建立保守的存储别名组。`getitem` 也会与其来源合并，因此 tuple 元素可能被保守地归为同一组。包含写入的组及其直接读取者都成为屏障。

检查反向写入也很重要：某个前向结果即使产生时没有原地修改，其存储在反向阶段被写过，再延迟重算相关读取也可能改变结果。

`_safety_reason` 进一步保护：随机算子、标记为非确定性的算子、通信和非 aten/prims 自定义算子、未知/高阶目标、有副作用节点，以及部分标量或符号值。普通 `operator.getitem` 在未触发 alias 屏障时可以继续参与重算。

这是一套保守静态规则，不是对任意隐藏副作用的证明。schema 未声明的副作用、Python cache 修改和任意外部状态不在保障范围内。

### 5.5 `_tag_policy`：保住区域之间的前向边界

策略和安全规则之后还会检查前向用户：若某个输出被另一区域或未选中区域的前向节点使用，则标为 `MUST_SAVE`，原因记为 `region_boundary`。这也覆盖最后一层到模型 epilogue 的边界。

这样不会为了重建一层的输入，而无限向前追溯并重新计算之前的整段网络。相比某些依赖层号或仅检查可重算消费者的规则，这里对任意区域外前向消费者都保守保存。

### 5.6 `_rematerialize`：最早消费者、依赖闭包、拓扑复制

算法可以分为四步：

```text
A. 按图顺序访问原始反向节点
   从其输入向前追溯允许重算的节点
   targets[前向节点] = 第一个需要它的反向消费者

B. 按原图拓扑顺序访问前向节点
   对 targets 中的节点，在对应消费者之前创建副本
   副本输入优先引用已经建立的依赖副本

C. 只重接原始反向节点的 args / kwargs
   原始前向消费者继续读取原始前向节点

D. 逆序删除已复制、没有剩余用户且无副作用的原节点
```

例如某条依赖链在区域内允许重算：

```text
原图：
  a = f(x)
  b = g(a)
  y = h(b)
  ... backward(b, a) ...

改写后（示意）：
  a = f(x)
  b = g(a)
  y = h(b)
  ...
  a_recomputed = f(x)
  b_recomputed = g(a_recomputed)
  backward(b_recomputed, a_recomputed)
```

前向仍可使用 a、b，但它们不再因为这条反向使用边而必须一直保留。若某个原节点仍有其他保存需求，就不会因此消失。

关键细节包括：

- `targets` 只登记一次，确保共享依赖最多建立一个副本，并放在最早的反向消费者之前。
- 沿原始拓扑顺序复制，保证副本的依赖先存在。
- 副本命名追加 `_recomputed`，记录 `recompute_source`。
- 保留模块元数据，将副本标为 `autograd_backward=True`，使后续 FSDP 能识别其执行阶段。
- 为嵌套的 `meta["custom"]` 单独复制字典，避免原节点和副本共用同一可变元数据容器。
- 删除副本上的 `recompute` 标记，避免再把它当成待处理的前向节点。
- 仅重接原始反向节点，避免重写副本输入时引入自引用或循环。
- 使用定向删除，不在这里进行全图重排或无差别 DCE。

该实现不承诺支持 TorchTitan 的 chunked loss、多段反向、复杂高阶子图等所有组合；已有辅助 loss 测试只说明 `(loss, loss_dict)` 输出协议得到保留。

### 5.7 `_saved_activation_bytes` 与报告的含义

统计依据是节点 `meta["val"]` 中的张量存储：对跨前向→反向使用边的非输入存储计数，按 storage identity 去重，并排除 placeholder / get_attr 所代表的输入或参数存储。

`recompute_report` 包括 `recomputed_nodes`、`saved_nodes`、`save_reasons`、`retained_activation_bytes_before/after`。其中：

- `saved_nodes` 是策略判为保存的节点数量，不是张量数量，也不是字节数。
- `save_reasons` 记录安全约束和区域边界等原因，不列出所有昂贵算子策略保存的原因。
- 字节数是重计算 pass 前后的**静态保留激活估计**，统计发生在后续 FSDP 等变换之前。
- 它不等于设备峰值显存；实际峰值还受参数、优化器、临时工作区、存储释放和 allocator 等因素影响。

## 6. 如何借鉴 TorchTitan GraphTrainer

参考的是当前工作目录中的 `torchtitan-main/` 快照，而非声称跟随其在线最新代码。该目录是本地参考资料，不是 HyperParallel 的运行依赖；若只取得 HyperParallel 仓库，下列参考链接可能不存在。

### 6.1 参考入口与对应关系

| TorchTitan 本地源码 | 本实现借鉴的思想 |
| --- | --- |
| [memory_policy.py](torchtitan-main/torchtitan/experiments/graph_trainer/memory_policy.py)：`tag_sac_policy`、full/eager 策略 | 策略标注与图改写分离；用 `meta["recompute"]` 表达保存/重算意图；保护层间输出 |
| [selective_activation_remat.py](torchtitan-main/torchtitan/experiments/graph_trainer/selective_activation_remat.py)：`selective_activation_remat_pass` | 原图内复制需要重算的前向依赖，放在反向消费者之前并改写使用关系 |
| 同文件 `_privatize_custom_meta` | 复制节点时单独处理嵌套可变元数据 |
| 同文件复制与定向删除循环 | 最早反向消费者决定插入点；共享依赖去重；只清理失去用户的原始节点 |
| [passes.py](torchtitan-main/torchtitan/experiments/graph_trainer/passes.py) | 将 memory policy、offload、rematerialization 作为可组合的编译流程环节 |

这里是设计与算法层面的借鉴，并针对 Hyper 的 GraphPass 接口、元数据和 FSDP 流程重新实现；没有调用 TorchTitan 包，也没有把其整套内存管理 pass 搬入 Hyper。

### 6.2 为 Hyper 做的改造

1. **区域标识**：TorchTitan 使用其模块 FQN / layer 元数据与辅助函数；Hyper 使用已有 `nn_module_stack` 和用户 glob。
2. **安全屏障**：针对 Hyper 当前捕获的算子，加入 schema alias/write 分析，以及通信、NPU 自定义算子等保守保存规则。
3. **配置与策略 enum**：复用 Hyper `CheckpointPolicy`，兼容 Torch 的四种 SAVE/RECOMPUTE 值；通过 `PassConfig` 与 Trainer YAML 暴露。
4. **FSDP 顺序**：按照 Hyper 的参数释放、反向重新聚合流程，明确在 `FSDPPass` 前插入重计算，不照抄 TorchTitan 的整个 pass 顺序。
5. **可观察性**：提供重计算副本数量、保存原因和静态保留激活估计，便于区分“配置生效”“确实重算”和“实际省峰值显存”。

### 6.3 没有迁移的 TorchTitan 能力

本地参考版本比本次移植的子集更丰富：

- `memory_policy.py` 有 min-cut 路径，利用 partitioner 选择保存值；Hyper 当前没有成本/显存预算求解。
- TorchTitan 的 remat 会识别模型层反向区域，排除 loss/lm_head 区域，并对多个需要重算的非连续反向区间做限制。Hyper 没有复制整套区域识别约定，不能据此推导 chunked loss 等组合已支持。
- TorchTitan 的 full 策略保存 RNG 输出，remat 拒绝仍被标为需重算的 RNG；Hyper 也不实现 RNG replay，但通过自己的安全规则处理并允许确定性消费者重算。
- TorchTitan 有 CPU offload/reload/wait 链，并能在新增重计算消费者更早出现时前移 reload 依赖。Hyper 没有这些图算子和调度，因此明确拒绝 swap 策略。
- TorchTitan 对 flex-attention 等高阶算子，会为子图 `get_attr` 建立独立引用以适配 regional compilation。Hyper 当前保存 opaque / higher-order 算子，没有搬入这种支持。
- TorchTitan 借助私有 partitioner 辅助函数判定 recompute 并做相关整理；Hyper 没有依赖这些 partitioner helper，但仍使用 FX、算子 schema 和张量元数据，不能理解为完全不依赖 Torch 内部接口。

## 7. 示例、验证结果与边界

### 7.1 简单示例

从仓库根目录执行：

```bash
python -m hyper_parallel.compile.examples.recompute.train --device cpu
python -m hyper_parallel.compile.examples.recompute.train --device cpu --dropout 0.2
python -m hyper_parallel.compile.examples.recompute.train --device npu --dropout 0.2
```

示例对比 eager 与 none/full/selective，包括 loss、所有参数梯度、SGD 更新和 RNG 状态。完整参数及图导出用法见 [示例 README](hyper_parallel/compile/examples/recompute/README.md)。

### 7.2 已有 AutoModel 示例上的四卡复现

以下是当前环境实际使用过的模型资产路径。生成的是确定性测试 token，不是自然语言训练数据；模型使用 Qwen3-0.6B 架构的固定随机初始化。

```bash
cd /home/whh/sac_pass
python -m hyper_parallel.compile.examples.recompute.prepare_automodel \
  --model-assets /home/whh/used/hyper_dev/.assets/models/Qwen3-0.6B \
  --output-dir /tmp/recompute_automodel

for policy in none full selective save_compute; do
  export VALIDATION_OUTPUT="/tmp/recompute_automodel/${policy}"
  if [ "$policy" = none ]; then
    unset VALIDATION_REFERENCE
  else
    export VALIDATION_REFERENCE=/tmp/recompute_automodel/none
  fi
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
    torchrun --standalone --nproc_per_node=4 \
    hyper_parallel/compile/examples/recompute/validate_automodel.py \
    "/tmp/recompute_automodel/${policy}.yaml" || exit 1
done
```

生成配置沿用 [train_lm_graph_tp2_fsdp2.yaml](hyper_parallel/compile/examples/automodel_text_graph/train_lm_graph_tp2_fsdp2.yaml)，保留 TP2 × FSDP2、sequence 128、global batch 4、micro batch 1、10 步 AdamW，并设置 `model.load_base_model: false`、`model_init_dtype: float32`。验证器仍调用现有 `TextTrainer.train()`，只是观察裁剪前梯度和更新后参数，不替换训练算法。

2026-10-12 在上述最终基线上的结果：

| 模式 | 每 rank 复制节点数 | 静态保留激活字节 | 实测峰值 allocated GiB，四 rank / step 2–10 最大值 | 相对 none 的最大 loss 绝对差 |
| --- | ---: | ---: | ---: | ---: |
| none / 改写前 | 0 | 375,593,425（pass 输入图估计） | 4.582930 | 0 |
| full | 2436 | 125,372,881 | 4.598938 | 2.3842e-6 |
| selective / eager | 2352 | 169,413,073 | 4.596241 | 2.8610e-6 |
| selective / save_compute | 2296 | 198,773,201 | 4.598553 | 2.3842e-6 |

四种模式均完成 4 ranks × 10 steps。每一步、每个 rank 的全部 311 个本地参数及梯度的完整 SHA-256 在策略间一致，梯度范数也一致；loss 使用 `atol=1e-5, rtol=0`。关闭重计算的重复实验也出现过约 `2e-6` 的 loss 波动，而参数和梯度校验值一致。

full 的静态保留激活估计减少约 66.6%，但该短序列测试**没有观察到设备峰值显存下降**。因此当前证据支持训练正确性及静态图激活减少，不能宣称峰值显存减少 66.6%。验证器包含 CPU 拷贝和哈希，也不应拿其耗时作为吞吐基准。

在这个模型中，full 仍保留 140 个通信/自定义节点（包括 NPU 融合 attention、TP collectives）和 28 个层边界节点。这是安全策略生效，不是配置没有打开。

### 7.3 不能从本轮验证推导的结论

- **预训练权重加载尚未验证成功**：关闭重计算时，Qwen3 TP 加载就会报 `lm_head.weight` 的 checkpoint shard `(151936, 1024)` 与 target `(75968, 1024)` 不匹配，发生在编译前。
- **BF16 + FP32 主参数优化器尚未跑通**：关闭重计算时，首次更新报 gradient `(37984, 1024)` 与 main parameter `(75968, 1024)` 不匹配，涉及惰性图 FSDP 切分之后主参数形状未同步。本轮采用 FP32 参数对照，没有绕过校验。
- PP + pass 显式不支持；EP、复杂 chunked loss、高阶子图重算、动态形状和任意隐藏副作用均未由本轮四卡实验覆盖。
- 原生动态图的 CPU swap、输入 swap、模型 adapter safe/no-replay 声明等尚未自动映射为图执行语义。

相关测试范围曾在最终基线上得到 **223 passed**：

```bash
python -m pytest -q tests/ut/compile \
  tests/ut/auto_models/test_build_options.py \
  tests/ut/trainer/test_config_overrides.py \
  tests/ut/auto_models/_transformers/test_config_resolver.py
```

这是已有验证记录；本次编写文档没有重新运行训练或整套测试。远程逐 rank JSON、配置和日志保留在 `/tmp/sac_pass_automodel/latest/`；临时目录可能被系统清理，应以保存的证据归档为长期记录。

## 8. 后续扩展方向

若继续完善，优先级可以是：先解决预训练加载和 BF16 主参数的基线集成问题，再对更长序列做真实峰值 profiling；随后按明确的算子契约扩展安全白名单和模型 adapter 选择映射。RNG replay、图级 offload/reload/wait、高阶子图和 min-cut 应作为独立能力设计与验证，不宜仅通过放宽当前安全判断来宣称支持。
