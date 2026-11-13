# PR #1533：Ascend 训推统一执行实现说明

本文面向代码评审，说明 [PR #1533](https://github.com/mindspore-ai/hyper-parallel/pull/1533)
新增和修改的生产代码、示例及交付文件，不展开测试文件。
关联路线图为 [Issue #1213](https://github.com/mindspore-ai/hyper-parallel/issues/1213)。
说明基于提交 `a25d4486`，差异基线为该提交与 `upstream/master` 的共同祖先 `cfedc890`。
本文记录这一版实现，后续修改应以相应提交的代码为准。
对外使用概览见新增的 [Unified Execution README](../rl/unified_execution/README.md)，
本文进一步解释各文件的修改原因、调用关系、计算语义和影响范围。

## 1. 修改目标与整体边界

本 PR 增加可选的 `unified_execution` 路径，让 HyperParallel 训练与 vLLM 推理复用模型定义、
组件替换、并行切分和关键数值实现。原有 native vLLM 模式继续保留。

实现分为三个层次，再补充 MoE 特殊组件：

| 层次 | 解决的问题 | 主要实现 |
| --- | --- | --- |
| 模型与运行时统一 | 两端分别维护模型和算子路径 | 向 vLLM 注册 HP 模型，完整复用训练侧 `plan_overrides`、builder 和并行机制 |
| Batch 不变性 | token 数、批量与序列长度变化引起浮点计算差异 | 两端启用 Torch NPU Level 3，统一 V4 Attention，并应用 rowsum 补丁 |
| TP/SP 不变性 | 不同 K 分片及通信规约改变加法结合顺序 | 普通 row-parallel 接入 PIK，固定全局 leaf 划分和二叉规约树 |
| MoE 处理 | 动态路由不利于图捕获，跨 EP 合并可能改变规约顺序 | 共享 router/experts，固定容量通信，按路由槽位顺序合并 |

这里的一致性以相同权重、相同 token 前缀、相同有效位置及相同算术规则为前提。
当前已验证场景覆盖不同 batch，以及训练 FSDP/TP/SP/EP 与推理 DP/TP/SP/EP 的前向一致性。
训练梯度按与自身前向匹配的一阶数学语义计算，不要求跨拓扑梯度或完整 RL 训练轨迹逐位相同。

职责分配如下：

```text
RL YAML：共享替换计划、数值策略、两端各自的拓扑
    │
    ├─ 训练：原 RL builder → HP 模型构建/切分 → Actor → loss/backward/optimizer
    │
    └─ 推理：vLLM worker → HP 模型构建/切分 → 同一模型 forward
                   │
                   └─ vLLM 调度、分页 KV cache、图执行、采样

两端共享：V4 数值接口、Level 3、PIK、固定容量 EP 组件
两端衔接：逻辑权重转换与版本化权重发布
```

vLLM 提供 worker 进程和已有通信域；HP 在这些 rank 上组织自己的模型 mesh、应用切分及重分布。
因此，本 PR 不只是注册模型名称，也没有把训练的并行计划转换成另一套 vLLM Linear/MoE 实现。

## 2. 新增代码的组织方式

核心代码集中在 [unified_execution](../rl/unified_execution/)：

```text
unified_execution/
├── contract.py                  配置契约、JSON 序列化、指纹和边界检查
├── execution_plan.py            PIK / 固定容量 EP 配方检查
├── validation.py                实际模型结构及组件执行路径检查
├── batch_invariant.py           进程级 Level 3 初始化与状态核验
├── models/
│   ├── registry.py              模型族描述与延迟解析
│   └── qwen3.py                 Qwen3 / Qwen3-MoE 的结构描述
├── components/
│   ├── row_parallel.py          将 PIK 绑定到 HP 普通 row-parallel
│   └── moe.py                   将固定容量 EP 绑定到 HP MoE
├── ops/
│   ├── attention/flash_attention_npu_v4.py
│   ├── moe/fixed_capacity_ep.py
│   └── pik/                     leaf GEMM、固定树、通信及一阶梯度
└── runtimes/
    ├── hyper_parallel/          共享 builder、训练 Attention 入口适配
    └── vllm/                    注册入口、模型协议、mesh、cache、权重加载
```

`models` 描述现有模型，不复制 decoder forward；`ops` 实现数值计算；`components` 将算子接入
HP 替换协议；`runtimes` 处理训练和推理不同的输入、缓存及生命周期。各级 `__init__.py`
提供包入口或必要导出，不承担第二套模型实现。

## 3. 两处公共代码修改

### 3.1 通信兼容：core/utils/communication.py

文件：[communication.py](../../core/utils/communication.py)。

这个公共模块原来已经区分 eager 与编译路径。eager 使用
`torch.distributed.nn.functional`，编译路径使用 `_functional_collectives`。
本次让 HP 模型进入 vLLM 分段编译后，当前 Torch 2.10 环境开始触发原先未覆盖的编译接口问题。

#### 修改前的问题

1. 原编译代码直接调用 `all_gather_single`、`reduce_scatter_single`，当前环境没有这些名称。
2. 直接回退到带 `_autograd` 后缀的旧接口，又会使无梯度推理进入当前环境不适合 Meta/FakeTensor
   追踪的 autograd 算子路径。
3. 旧版 AllGather 非零维拼接内部的 view 转换在 Dynamo 追踪下存在兼容问题。
4. 字符串规约操作经 `ReduceOp` 枚举转换后，再用枚举构造字典或比较，会遇到旧 Dynamo 的追踪限制。

#### 修改方式

新增 `_resolve_functional_collective()`，在模块导入时解析有梯度和仅前向的函数引用：

| 用途 | 优先接口 | 旧版本回退 |
| --- | --- | --- |
| 有梯度 | `*_single_autograd`，其次 `*_single` | `*_tensor_autograd` |
| 无梯度前向 | `*_single` | `*_tensor` |

通过接口是否存在选择，不硬编码 Torch 版本分界。AllGather/ReduceScatter 中还保留了原来的
`*_single` 直接调用分支；新接口存在时继续使用它。

旧接口分支按 `torch.is_grad_enabled()` 选择训练或前向版本。选择依据是梯度上下文，
而不是输入的 `requires_grad`：推理即使持有需要梯度的参数，在 `no_grad` 下也不需要训练接口。

AllGather 的旧版兼容分支先沿第 0 维通信，再用 `unflatten → movedim → reshape` 恢复目标拼接维。
例如两张卡各持有 `[M, N]`，要求沿维度 1 拼接：

```text
AllGather(dim=0): [2M, N]
    → 拆出 rank 维：[2, M, N]
    → 移动 rank 维：[M, 2, N]
    → 合并目标维：[M, 2N]
```

该转换保留 rank 顺序，不进行浮点加法；非零维可能产生连续化复制。
异步结果的 `wait()` 用于确保结果可消费，不是新增全局 barrier。

ReduceScatter 的旧版兼容路径仅接受 SUM、字符串 `avg`，以及 `ReduceOp.SUM`。
`avg` 仍按 SUM 后除以 `dev_num` 实现，其他规约明确报错；该限制也适用于旧版无梯度兼容分支。
eager 和保留的新版 `*_single` 分支不受这一新增限制影响。

AllReduce 对字符串参数直接转换成 functional API 需要的字符串，绕过枚举哈希。
保留历史语义：底层封装的 `avg` 映射 SUM，布局转换层负责需要的除法；`mean` 使用可用的原生 AVG。
显式枚举参数仍走原转换逻辑，此修改不等于解决所有枚举参数的编译问题。

#### 对其他训练代码的影响

- eager 分支及其 autograd 实现保持不变；原训练调用仍使用原接口。
- 调用这些封装的其他 DTensor、EP 等模块，在编译时同样获得兼容修正，影响范围不局限于 RL。
- AllGather/ReduceScatter 的编译反向由 Torch autograd 接口及普通张量变换提供，没有新写通信 backward。
- 模块导入会解析接口，因此基础 Torch 必须提供候选接口；当前方案按已配套的运行环境验证。
- 旧版 functional AllReduce 的编译训练反向问题本次未扩展修复。当前 RL 使用训练 eager、
  推理无梯度编译，不依赖这条反向；其他使用编译训练的调用方仍需单独处理。

这些修改不改变 HP 的 mesh、参数分片和算子布局规划，也不是 PIK 固定规约树的实现位置。

### 3.2 SAPP dry-run：auto_parallel/sapp_ppb/utils/config.py

文件：[config.py](../../auto_parallel/sapp_ppb/utils/config.py)。

这处来自完整门禁中暴露的上游问题。SAPP 为显存建模生成 stage 层数偏移和重计算配置，
再建立满秩系数矩阵。每行前两项为 `[1, layer_per_stage + offset]`。
若参与求解的所有 offset 相同，这两列必然线性相关。

原 `_generate_offset_config()` 检查前 `unknowns + 1` 行是否有变化，
但 `_get_coef_matrix()` 实际只使用前 `unknowns` 行。例如需要 6 行时：

```text
预检查的 7 行 offset：[0, 0, 0, 0, 0, 0, 1]  → 被接受
实际使用的 6 行：    [0, 0, 0, 0, 0, 0]     → 必然秩不足
```

`generate_solvable_config()` 在循环外生成 offset，循环内只重新生成重计算配置。
坏 offset 一旦被接受，后续随机重试无法解除前两列的依赖，因而可能一直循环。

修改仅统一行数：

```python
# 原实现
flat = flat.flatten()[0 : unknowns + 1]

# 当前实现
flat = flat.flatten()[:unknowns]
```

现在坏布局会在 offset 生成阶段被拒绝。后续完整矩阵求秩仍保留，这个预检查不代替最终满秩判断。

影响仅在使用该工具重新生成 dry-run 配置时：候选布局和随机数消耗顺序可能变化。
已有训练配置的 forward、backward、优化器以及运行时 PP 调度均不经过这段配置生成逻辑。

## 4. 配置契约：确保两端使用同一份声明

入口文件：

- [config/schema.py](../rl/config/schema.py)：接受顶层 `consistency`，训练 TP 范围扩展至 1/2/4。
- [config/resolve.py](../rl/config/resolve.py)：新增 `resolve_unified_execution_config()`。
- [config/validate.py](../rl/config/validate.py)：接入统一执行、拓扑和模型边界检查。
- [contract.py](../rl/unified_execution/contract.py)：建立及复核 worker 契约。
- [execution_plan.py](../rl/unified_execution/execution_plan.py)：检查具体替换配方。

配置解析会深拷贝用户配置，根据 `train.plan_overrides` 构建 payload，并将推理选择设置为 `unified`。
以下仅是数值模式的配置片段；完整运行还需要示例中的模型、并行、替换和训练配置：

```yaml
consistency:
  schema_version: 1
  mode: batch_invariant
  attention_backend: flash_attention_npu_v4
  batch_invariant:
    provider: torch_npu_level3
    fill_uninitialized_memory: false
```

`architecture` 模式用于共享架构；`batch_invariant` 模式进一步要求 V4 与显式 Level 3 策略。
架构模式本身不代表已消除所有数值差异。

payload 的关键内容如下：

| 字段 | 作用 |
| --- | --- |
| `plan_overrides` | 完整训练声明，包含模块、方法、参数和激活布局规则 |
| `compute_plan_overrides` | 模块替换子集，供实际替换覆盖检查使用 |
| `excluded_plan_overrides` | 当前实现返回空列表，不通过此字段丢弃并行规则 |
| `training_parallelism` / `parallelism` | 分别记录训练与推理的 TP/CP/PP/SP |
| `expert_parallelism` | MoE 两端各自的 EP 大小 |
| `attention_backend` / `batch_invariant` | 两端共享数值实现和策略 |
| `plan_fingerprint` | 规范化 JSON 的 SHA256，用于检查配置完整性 |
| `training_structure` | Actor 构建后记录的实际结构及其独立指纹 |

`split_plan_overrides()` 虽然保留这个名字，当前不会只取模块替换而丢掉方法或布局规则。
真正送给推理 builder 的是完整 `plan_overrides`；`compute_plan_overrides` 只是审计视图。

worker 的 `parse_unified_payload()` 重新推导字段和指纹，拒绝不一致的 payload。
指纹证明两端使用同一份声明，实际组件是否安装则由 `validation.py` 进一步检查。

`execution_plan.py` 还约束 PIK 的叶数、布局、填充策略和父子模块布局，以及固定容量 EP 的唯一配方。
同一套规则中的 `when: sequence_parallel` 等条件，分别按两端拓扑生效：共享配方不意味着两端 TP/SP 必须相同。

## 5. 模型构建与训练端接入

### 5.1 模型族描述复用现有实现

[models/qwen3.py](../rl/unified_execution/models/qwen3.py) 定义 Qwen3 Dense 与 Qwen3-MoE 的描述：
原 causal LM、decoder、layer、attention、rotary 类型，HP attention 工厂，普通 row 投影位置，
以及 MoE router adapter 名称。

[models/registry.py](../rl/unified_execution/models/registry.py) 用 `ModelFamily` 统一访问这些描述，
在实际构建时才导入模型类。配置解析和插件枚举不需要因此加载完整模型。
Qwen3-MoE 的 vLLM 类直接继承共享模型适配器，不复制另一份 forward。

### 5.2 保留 HP builder、参数身份和布局信息

[workers/adapters/qwen3.py](../rl/workers/adapters/qwen3.py) 将 RL 内的 Qwen3 构建适配扩展到
Qwen3-MoE，同时保留 `HyperAutoModelForCausalLM` 的完整加载和切分链。

`Qwen3ShardingPlanner.plan()` 在调用原 planner 前记录全局参数形状和权重转换描述；
规划后记录模块输入、输出布局，供 vLLM 调度边界适配使用。
记录必须在 TP 修改局部形状、head 数之前完成，否则权重转换可能错误地把局部尺寸当成全局尺寸。

`retain_parameter_layouts()` 复用 HP 的 `build_source_shard_info()`，在没有 FSDP manager
保存布局的情况下补留参数布局。推理权重加载仍需知道目标参数在哪个维度、哪个 rank 上分片。
旧的额外 `apply_qwen3_sharding_plan()` 包装被删除，避免形成重复的手工切分入口。

`PikLinear` 等替换组件保留原 Parameter 对象，HP 继续负责 materialization、加载、FSDP 生命周期。
本 PR 没有抽取或重写公共 `TextTrainer._build_model()`。

### 5.3 训练接入的调用顺序

[trainer/build.py](../rl/trainer/build.py) 中的主要变化：

```text
_TrainerBuilder.__init__
    → resolve_unified_execution_config
    → 原配置解析与校验
    → 初始化训练设备及进程级 Level 3
    → build_role_model：复用 HP builder 构建 Actor
    → capture_training_structure：记录实际计算结构
    → install_training_attention：接入共享 V4
    → guard_training_numerics：检查进程数值状态
    → 原 RL 训练与权重发布生命周期
```

若存在 Reference 模型，也安装共享 Attention 与数值状态检查；Critic 在 batch-invariant 模式下
安装数值状态检查，但这里没有为它无条件安装 Actor 的 Attention 接口。
角色选择和优化器管理仍属于原 builder。

[validation.py](../rl/unified_execution/validation.py) 检查模块类型、逻辑参数形状、参数别名、
模块替换覆盖，以及 attention、MoE、PIK 的实际执行绑定。
比较使用逻辑结构，避免将 TP/FSDP 的局部驻留形状误判为模型定义不同。
推理在安装分页 cache 桥接前进行同样检查，并与 Actor 报告比较。

### 5.4 RoPE 与训练 SP 输入适配

[workers/engine/materialization.py](../rl/workers/engine/materialization.py) 扩展为按模型族识别 rotary，
统一在 CPU/FP32 下重建 `inv_freq`、`original_inv_freq`，再搬到目标设备。
额外的 `with torch.device("cpu")` 保证 HF 内部未显式传 device 的工厂也不落到 meta/NPU 默认设备。
非 meta 构建不会经过后续 rebuild，因此直接对已有 buffer 执行同一配方的 `copy_`。
这不更换 Parameter 或 buffer 身份。

[workers/engine/model.py](../rl/workers/engine/model.py) 避免对已注册共享重建配方的 MoE 再执行旧 rotary 重建，
并新增 `unified` 架构名映射；原 native 和 Qwen3.5 注册路径继续保留。

[workers/actor.py](../rl/workers/actor.py) 对声明了 `unified_sequence_multiple` 的模型执行：

1. 在 SP 分片前，将序列、attention mask、action mask 补到所需长度倍数。
2. 显式传入完整序列的 `position_ids`，避免按 Embedding 分片后的局部长度生成位置。
3. 完成 log-prob 计算后裁回原长度。

普通模型的 multiple 默认为 1，不触发新增 padding。当前位置构造为完整序列的 `arange`，
不能据此推导出任意位置重置或任意 packing 语义均已支持。

## 6. vLLM 注册、并行和调度边界

### 6.1 注册与参数传递

[rollout/vllm.py](../rl/rollout/vllm.py) 在统一模式下给服务进程设置
`HYPER_RL_UNIFIED_EXECUTION=1`，并通过 `--hf-overrides` 传入：

```text
architectures = [HyperParallelQwen3ForCausalLM 或 HyperParallelQwen3MoeForCausalLM]
hyper_rl_unified_execution = 完整 payload
```

[rollout/vllm_plugin.py](../rl/rollout/vllm_plugin.py) 在 worker 中调用
`ModelRegistry.register_model()`，目标为
[model_entrypoints.py](../rl/unified_execution/runtimes/vllm/model_entrypoints.py) 中的类。
`support_torch_compile` 声明 `input_ids`、`positions`、`inputs_embeds` 的 token 维为动态维。

worker 从 `model_config.hf_config.hyper_rl_unified_execution` 读取契约。
`additional_config` 只承载如 `weight_nz_mode` 的平台设置，不再承担模型执行契约。

### 6.2 推理模型真正应用训练侧并行机制

[parallel_context.py](../rl/unified_execution/runtimes/vllm/parallel_context.py) 复用 vLLM 已初始化的 TP group，
构建 HP `DeviceMesh` 与 `MeshContext`。启用 EP 时复用 DP×TP 域和已有 DP 同步调度，
再调用 HP `build_expert_mesh()` 得到专家 mesh。

EP 是 DP×TP 域内的专家分片，不是额外乘一遍卡数。
`rollout.vllm.expert_parallel_size` 表达 HP EP；vLLM 自身的 `enable_expert_parallel` 和 EPLB 被拒绝。
各 DP scheduler 的请求独立，输入切分使用 `src_data_rank=None`，避免广播一个 DP rank 的请求覆盖其他请求。

[hyper_parallel/builder.py](../rl/unified_execution/runtimes/hyper_parallel/builder.py) 将完整计划转换为
HP `PlanOverride`，使用原 `normalize_distributed_setup_overrides()`，再调用模型族 builder 的
`from_config()`。推理从配置构建和切分模型，权重随后由 vLLM `load_weights()` 协议提供。
推理端不创建训练优化器或训练 FSDP manager。

[rollout/topology.py](../rl/rollout/topology.py) 统一计算 worker 数量并明确拒绝未接入的 CP/PP。
代码中存在这些轴的配置字段，不表示实现了对应运行路径。

### 6.3 模型 wrapper 保留原 decoder forward

[model_adapter.py](../rl/unified_execution/runtimes/vllm/model_adapter.py) 的构造顺序是：

```text
解析契约 / 初始化数值策略
    → 构建 HP mesh
    → 共享 builder 创建模型并应用完整计划
    → 校验实际模型结构
    → 保留 model、lm_head，建立逻辑权重 loader
    → 准备调度边界布局变换
    → 安装 vLLM 分页 Attention
```

`forward()` 将 vLLM 扁平 token 调度转换成 HP 模型需要的输入，显式传入绝对位置，
调用原 decoder；`compute_logits()` 使用同一个 HP LM head。
HF `use_cache=False` 只是禁用 HF 自己的 cache，推理 KV cache 仍由每层 vLLM Attention 管理。

SP 时，`_pad_rows()` 采用固定补入 `TP-1` 行后再按符号长度裁切的方式，
避免 Python 的“是否需要补齐”分支被编译特化到某一种 token 数。
输入输出布局按照 planner 保存的声明处理；输出返回 scheduler 前恢复为完整 token 行并裁掉 padding。
已有 HP TP collective lowerer 在追踪前准备，避免在图内重新规划布局。

## 7. Attention：缓存桥接与底层计算分开实现

### 7.1 共用 V4 数值接口

[flash_attention_npu_v4.py](../rl/unified_execution/ops/attention/flash_attention_npu_v4.py) 负责：

- 延迟加载原生 `flash_attn_npu_4` 扩展，核对 0.5.0 基础版本、NPU 和 BF16 输入。
- `dense_attention()` 调用 `flash_attn_func()`，接收 BSND。
- `varlen_attention()` 调用 `flash_attn_varlen_func()`，支持训练 packed TND 或推理分页 KV。
- 两端固定 `causal=True`、`deterministic=True`、`num_splits=1`。
- 明确校验长度张量、page table 和布局契约；长度内容不在算子适配热路径中转回 CPU。

训练反向沿用 V4 原生 autograd。当前可微路径使用 dense 或 packed 输入；带 page table 或
`seqused_k` 的分页调用属于推理路径，不提供训练 KV cache 反向。
版本检查接受 0.5.0 的本地版本后缀，不能单独证明 rowsum 补丁已存在；补丁产物由启动脚本校验哈希。

### 7.2 训练侧去 padding 与恢复

[training.py](../rl/unified_execution/runtimes/hyper_parallel/training.py) 在模型入口解析 mask，
生成有效 token 索引和累计长度，作为本次 forward 的元数据传给 decoder 层及重计算路径。
每层只负责 BNSD→BSND 转换、按索引打包、调用 V4，再恢复原坐标。
全 padding 输入也保持 Q/K/V 梯度连接，输出无效行归零。

mask 检查在训练模型入口有一次 host 处理，不属于推理捕获图。
接受二值 padding mask 和标准 causal-plus-padding mask；不将任意 additive bias 当成 padding。
安装方式是注册并选择 attention interface，保留原 attention 模块、QKV、RoPE 和输出投影。

### 7.3 推理侧保留 paged KV cache

[paged_attention.py](../rl/unified_execution/runtimes/vllm/paged_attention.py) 为每层创建 vLLM `Attention`。
它承接 HP 产生的 Q/K/V，将 BNSD 转成调度 token 行；进入 cache 前裁掉 SP 补齐行，
计算后再补回模型需要的形状。

选择 V4 时，将 `UnifiedAscendAttentionBackend` 直接作为每层 `Attention(attn_backend=...)`
的参数传入，不全局替换所有 vLLM 模型的 backend。

[attention_backend.py](../rl/unified_execution/runtimes/vllm/attention_backend.py) 继承 Ascend 原实现：
原 `forward` 按 `slot_mapping` 写入当前 K/V；重写的 `forward_impl()` 从 cache 读取历史 K/V，
使用 `block_tables`、query 起点和完整逻辑 KV 长度调用 V4。
这保留了 prefix cache 命中和 chunked prefill 的历史对应关系，不把历史 cache 展开成另一份模型输入。

### 7.4 图捕获与 V4 rowsum 补丁

backend 声明 `UNIFORM_SINGLE_TOKEN_DECODE` 图支持，允许 `FULL_DECODE_ONLY` 或
`FULL_AND_PIECEWISE`。示例采用后者：decode 全图，prefill 分段执行，不等于所有 prefill 都捕获为完整图。

V4 0.5.0 的 scheduler metadata 准备会创建普通事件，因此放在捕获外，
每轮复制到地址稳定的设备缓冲区。图回放读取更新后的设备长度及 metadata，而非捕获时固定的 CPU 长度列表。

[canonical rowsum 补丁](../docker/patches/flash-attn-npu-v4-canonical-rowsum.patch)
修改 V4 的 `CalcLocalRowSum()`：256/512 列完整 tile 不再使用另外两棵专用加法树，
统一调用 `RowsumTAILTILE()` 的 64 列规约路径。
原来相同因果前缀在追加未来被 mask 的 token 后，可能因完整块/尾块选择不同而改变浮点结合顺序。
Level 3 无法代替这处 V4 内部修正。
补丁改变的是实际数值实现，可能影响性能；它不是仅调整 Python 标志。

## 8. Batch 不变性：进程级 Level 3

[batch_invariant.py](../rl/unified_execution/batch_invariant.py) 实现策略规范化、初始化和只读核验。
当前固定 Torch 2.10.0、torch-npu 2.10.0.post7，初始化要求已经选择 NPU 设备。

```python
torch_npu.npu.set_deterministic_level(3)
torch.utils.deterministic.fill_uninitialized_memory = False
```

初始化后执行小 GEMM 并同步，用于触发并检查底层选项应用；这不是数值不变性的证明。
策略记录 PID、设备、版本、level、deterministic 标志和填充设置，重复初始化要求策略一致。

训练端在模型前向前核验；推理端在构建、普通 forward、logits、权重发布和 wake-up 等边界核验。
`rollout/vllm.py` 通过 `get_unified_numerical_policy` RPC 收集全部 DP×TP worker 的状态，
与训练进程对照，PID 和设备编号允许不同。

关闭 `fill_uninitialized_memory` 仅取消未初始化内存的自动填充；不会关闭 Level 3，
也不会删除显式序列 padding、MoE 容量补齐或有效性 mask。
该设置为进程级，因此同一进程的其他模型和张量操作也会看到它。

统一路径要求 `rollout.vllm.batch_invariant=false`，拒绝继承的 `VLLM_BATCH_INVARIANT=1`
及已安装的 vLLM-Ascend batch-invariant aten patch，避免两套全局策略叠加。

## 9. PIK：固定算术规则并接入普通 row-parallel

### 9.1 固定全局 K 划分

[plan.py](../rl/unified_execution/ops/pik/plan.py) 的 `PikPlan` 固定全局 leaf 数 G 和物理输入布局。
对 `Y = X @ W.T`，全局 K 被划为 G 个连续片段，要求 G 为 2 的幂、K 能被 G 整除、TP 能整除 G。

```text
G=8，TP=1：一个 rank 持有 leaf 0..7
G=8，TP=2：两个 rank 分别持有 0..3、4..7
G=8，TP=4：四个 rank 分别持有 0..1、2..3、4..5、6..7
```

改变 TP 只改变完整子树归属，不改变 leaf 的 K 范围和全局加法树。
TP=1 也必须使用相同 G 的 PIK，不能用普通整矩阵 GEMM 代替基准。

算术契约为：BF16 输入及 leaf 输出，内部树节点 FP32，最终 root 转 BF16，bias 在转换后按 BF16 加入。
修改 G 会改变数值契约；它不只是性能参数。

### 9.2 一次 BMM 计算本 rank 的全部 leaf

[gemm.py](../rl/unified_execution/ops/pik/gemm.py) 的 `batched_leaves()` 将本地输入组织为：

```text
X_local: [M, K_local] → [L, M, K_leaf]
W_local: [N, K_local] → [L, K_leaf, N]
torch.bmm            → [L, M, N]
```

其中 L=G/TP。leaf GEMM 直接复用 Level 3 的 `torch.bmm`，不自写 GEMM kernel。
`strided` 使用 reshape/permute 视图；`packed` 显式连续化。
两端固定同一选项，代码不按输入形状自动切换不同算术后端。

### 9.3 Triton 固定树与跨 rank 搬运

[_triton_reduction.py](../rl/unified_execution/ops/pik/_triton_reduction.py) 按 leaf 数生成固定顺序的
Triton 加法表达式，一次 kernel launch 完成一棵树的所有层，关闭浮点融合。
Python 递归用于生成源码，不是在每次 forward 中逐节点发射张量加法。
生成文件放在 Triton cache 下，不放入源码目录。

[reduction.py](../rl/unified_execution/ops/pik/reduction.py) 将规约封装为 custom op 并提供 fake 实现，
让编译器识别输出形状，同时保留内部固定算术边界。

[collectives.py](../rl/unified_execution/ops/pik/collectives.py) 提供两条前向通信路径：

| 路径 | 具体行为 |
| --- | --- |
| `all_gather` | 收集各 rank 的 FP32 子树根，按 rank 顺序用固定树求根；SP 时再取本 rank 输出行 |
| `all_to_all` | 按输出行切片发送子树根，由负责 rank 固定树求和；非 SP 时再 AllGather 最终输出 |

两条路径都不让 HCCL 在 PIK 前向中执行浮点 SUM。
本 PR 的 `PikLinear.bind_pik()` 未指定 transport，因此模型接入当前采用 `all_gather` 默认路径。
`all_to_all` 是独立算子的可选能力，要求扁平 token 行数能被 TP 整除。

[linear.py](../rl/unified_execution/ops/pik/linear.py) 的 `prepare_pik()` 在编译外初始化策略并交换
静态契约摘要；`PikContext` 保留拓扑和算术信息，不缓存权重或激活。
forward 复用本地树根，再完成跨 rank 树、dtype 转换和输出布局恢复。

### 9.4 HP 组件绑定与避免重复规约

[components/row_parallel.py](../rl/unified_execution/components/row_parallel.py) 分两步接入：

1. `replace_pik_linear()` 无条件安装 PIK 算术模块，包括 TP1；保留原 weight/bias Parameter。
2. `pik_row_compute_fn()` 由 HP 方法替换绑定实际 TP group 和 SP 开关。

attention 的父模块替换和 `o_proj` 子模块替换不能独立冲突，因此 `replace_attention_with_pik()`
先调用已有 HP attention 工厂，再组合 PIK `o_proj`。

PIK 已返回规约后的输出，所以 child 与 parent 的 `out_src/out_dst` 必须是相同布局：
无 SP 为 Replicate，SP 为 `Shard(1)`。配方还要求相关父模块 `region_dispatch=false`。
这防止 HP 再执行一遍原 row-parallel 的 AllReduce/ReduceScatter。
输入所需布局转换仍交给 HP。

Dense Qwen3 覆盖 attention `o_proj` 和 MLP `down_proj`；Qwen3-MoE 当前只覆盖 attention `o_proj`。
专家内部 GEMM 不在本次 PIK 替换范围内。

### 9.5 一阶梯度与 HP 非 SP 语义

独立算子的 `_PikLinear.backward()` 使用标准线性层梯度：

```text
SP：先 AllGather 各 rank 不重叠的 dY 行
dX = dY @ W
dW = dY.T @ X
dB = sum(dY)
```

该 backward 标记 `once_differentiable`，当前提供一阶梯度。
它不对 BF16 舍入求精确导数，也不承诺跨 TP 梯度逐位相同。

HP 模型中，TP>1 且无 SP 时，下游不同 rank 的梯度贡献需要 SUM。
组件层额外通过 `_NativeRowOutputGradient` 保留这项原生语义：forward 为恒等，backward 对 dY AllReduce。
它只在开启梯度的对应训练路径生效；不改变独立算子契约或推理前向。

## 10. MoE：共享专家计算、固定容量路由和有序合并

[components/moe.py](../rl/unified_execution/components/moe.py) 通过 HP `local_compute` 方法替换绑定
EP group、rank、本地专家数及容量。router 使用 HP `MOE_ROUTER_ADAPTERS`，
专家执行复用 `bind_local_expert_forward(..., use_grouped_gemm=True)` 和原 `module.experts`。

[fixed_capacity_ep.py](../rl/unified_execution/ops/moe/fixed_capacity_ep.py) 的流程为：

```text
本 rank token → 原 router 得到专家 ID 与权重
    → 补到固定 token 容量 C（无效专家 ID=-1）
    → AllGather token 和专家 ID
    → 各 rank 用本地专家计算固定形状 route 行
    → 等长 AllToAll 将结果送回源 token rank
    → 根据专家 ID 选取唯一 owner 的结果
    → 按 top-k 槽位顺序 FP32 加权累加，最后转换 dtype
```

每条 route 只有一个专家 owner。`owner_routes()` 使用索引选择，不沿 rank 维做浮点求和；
`ordered_combine()` 固定槽位顺序，避免 EP 归属变化改变合并顺序。

图捕获需要静态通信尺寸，所以不从设备读取动态路由计数来生成 Python split 列表。
AllToAll 的 splits 固定为 `[C * top_k] * EP`。无效行在专家计算前后被 mask；超过容量直接报错，
不会静默丢 token。这里的 C 是每个 EP rank 进入该 MoE 区域的 token 上限，不是每个专家的容量因子。

固定容量会增加无效计算、通信量和缓冲区占用，尤其在 EP、C、top-k 较大时。
这是本版为图兼容和明确合并顺序采用的实现，尚未引入单边通信或稀疏通信性能优化。

专家通过原 Module 调用入口执行，因此 HP 参数 unshard/reshard hooks 继续生效。
通信、索引选择、专家输出和路由权重的梯度由已有可微通信及 Torch 操作组合完成，
这里没有新增一套手写专家 backward；专家 ID/top-k 选择本身仍是离散路由。

融合专家参数当前按 EP 分片或在 TP 下复制，未接入专家内部 TP/ETP。
更改数值路径可能使临界 router 分数的 top-k 选择变化，因此与另一套 BF16 实现的整网梯度比较
需要区分路由变化和反向实现错误。

## 11. 权重加载与在线发布

### 11.1 使用逻辑权重作为两种物理布局之间的接口

训练和推理可以有不同 TP/EP，不能直接把训练 rank 的本地参数写到同编号推理 rank。
当前示例使用已有 `full_gather` 发布策略，转换链为：

```text
HP 训练物理参数
    → 原发布器收集，并携带 canonical_rows / canonical_experts 元数据
    → 接收端恢复逻辑 HF 权重
    → LogicalWeightLoader 应用 HP 转换描述
    → 按推理目标布局写入本 rank 参数
```

[weight_sync/model_adapter.py](../rl/weight_sync/model_adapter.py) 增加融合 MLP 的行转换描述，
扩展融合 QKV bias 支持，并复用专家权重切片描述。
描述来自实际模块的转换规则和切分前记录，避免另写一套 gate/up 或 QKV 排列公式。
[packed_weight.py](../rl/weight_sync/packed_weight.py) 相应支持一维 bias 的 canonical row 恢复。

### 11.2 LogicalWeightLoader 的行为

[weight_loader.py](../rl/unified_execution/runtimes/vllm/weight_loader.py) 使用已有 HP 转换的逆操作，
对行标记求逆来建立逻辑 Q/K/V、gate/up 到物理参数的索引。
它根据目标布局只写本 rank 所属的行、列或专家区域，保持参数对象身份。

首次 checkpoint 加载是隐式完整事务；在线发布显式执行 `begin → 多个 load bucket → finish`。
loader 检查重复权重、未知名称、shape、缺失逻辑投影以及 tied 参数别名。
整参数别名可互相满足完整性要求，融合参数的不同逻辑片段不能互相替代。

这里的“事务”是发布完整性和恢复服务的控制边界，不表示复制失败后会自动回滚已写入的参数。
失败发布不能作为成功版本恢复服务。

### 11.3 vLLM worker 生命周期兼容

[weight_sync/vllm_worker.py](../rl/weight_sync/vllm_worker.py) 扩展支持的架构，
调用统一模型的 `begin_weight_update()`、`finish_weight_update()`，完整性确认后才提交 policy version。
增加数值策略 RPC，并将其聚合到全部 DP engine。

vLLM-Ascend 0.30 的 `start_weight_update` 去掉了旧参数，reload 归属也发生变化。
代码检查原函数签名，按新旧接口调用；新生命周期在适当位置重置 LoRA 状态。
这些版本兼容同样可能作用于 native 模型，不应归类成完全由 `consistency` 开关隔离的修改。

## 12. 其他兼容修改及已有文件的影响

| 文件 | 修改内容与影响 |
| --- | --- |
| [rollout/vllm_plugin.py](../rl/rollout/vllm_plugin.py) | 接受 vLLM/vLLM-Ascend 0.30.0 组合；只有统一模式的环境开关启用时才注册 HP 架构 |
| [tool_protocol/protocol.py](../rl/tool_protocol/protocol.py) | 兼容 0.30 Qwen3 reasoning/tool parser 的新模块名；仅在指定旧模块缺失时回退，其他导入错误继续抛出 |
| [workers/engine/model.py](../rl/workers/engine/model.py) | `native/unified` 架构映射；按标记避免 MoE rotary 重复重建 |
| [config/validate.py](../rl/config/validate.py) | 允许 Qwen3-MoE colocated unified；保留现有模型/算法限制，检查 PIK 与拓扑等约束 |
| [docs/vllm_rollout.md](vllm_rollout.md) | 补充 unified 模式和使用入口 |
| [docs/rl-architecture.md](../../../docs/rl-architecture.md) | 说明统一执行的架构边界 |
| [docs/rl-navigation.md](../../../docs/rl-navigation.md) | 补充核心文件导航 |
| [.agent/rules/rl/module-map.md](../../../.agent/rules/rl/module-map.md) | 同步开发导航中的模块归属和路径 |

未配置 `consistency` 时，不安装统一 Attention、Level 3、PIK 或固定容量 EP。
但公共通信兼容、训练 TP4 配置范围、RoPE 构建适配、权重转换支持和 0.30 生命周期/parser 兼容，
是共享入口上的实际修改，评审时应分别检查其对既有路径的作用。

## 13. 示例、环境与 V4 交付

本 PR 提供一个完整的四层 Qwen3-MoE 两步同步 RL 示例：

| 文件 | 用途 |
| --- | --- |
| [configs/qwen3_moe_4l_rl.yaml](../examples/consistency/configs/qwen3_moe_4l_rl.yaml) | 显式共享计划、训练/推理拓扑、数值策略和两步训练配置 |
| [scripts/prepare_checkpoint.py](../examples/consistency/scripts/prepare_checkpoint.py) | 从完整模型保留前四层及原始专家权重，生成独立 checkpoint 和校验清单 |
| [scripts/run_consistency.py](../examples/consistency/scripts/run_consistency.py) | 调用正常 `build_trainer()` / RL 循环，在 optimizer 更新前检查采样与重算 log-prob |
| [scripts/run_qwen3_moe_docker.sh](../examples/consistency/scripts/run_qwen3_moe_docker.sh) | 启动本地容器、加载 V4 补丁、准备裁剪权重和运行流程 |
| [examples/consistency/README.md](../examples/consistency/README.md) | 完整的本地复现步骤和输入路径说明 |

默认训练 TP4/SP/EP4，推理 TP2/DP2/无 SP/EP4，共用四张卡。
EP4 复用四个 worker 的域，不额外增加四倍设备。
示例保留 128 专家、top-k8，仅裁剪层数，普通 attention `o_proj` 使用 8-leaf PIK。
开启 prefix cache、chunked prefill 和 `FULL_AND_PIECEWISE`。

YAML 指向裁剪后的 checkpoint；完整权重由准备脚本读取，原模型目录不改动。
复用裁剪产物时核对来源元数据及输出哈希。`run_consistency.py` 不替换样本、奖励或数值路径，
只在原 `targets.prepare` 边界加入观测，要求有效 token 非空、结果有限、最大与平均 log-prob 差均为零。
两步正常完成后写 `consistency.json`。该检查覆盖采样 token 的概率，不等价于比较全部词表或证明非零学习收益。

环境采用已准备的 ARM64/Ascend 镜像，主要版本为 Python 3.12.16、CANN 9.2.1、
Torch 2.10.0、torch-npu 2.10.0.post7、vLLM/vLLM-Ascend 0.30.0、Transformers 5.14.1。
镜像中的 batch-invariant 扩展不是当前配方的注入方式，当前选择 Level 3。

V4 补丁尚未固化到基础镜像。新增 [build_v4_wheel.sh](../docker/build_v4_wheel.sh)
固定源码和 CATLASS 提交，在现有镜像中应用源码 patch，构建 `0.5.0+hpbi1` wheel，
输出源码版本、环境、patch/wheel/原生库哈希和构建记录。支持从本地 Git 源码离线导出构建。
无需 NPU，也不修改基础镜像；输出目录要求为空。

运行脚本核对 wheel 哈希，再安装到独立 `/opt/hp-v4` 目录，使训练和推理使用同一产物。
重建 wheel 的哈希不要求与旧产物相同，运行时应同时配置对应 `V4_WHEEL` 和 `V4_WHEEL_SHA256`。
正式启动器保留 `pip check`，依赖不满足时应先处理环境；它不会默认跳过依赖检查。
更详细的构建参数见 [docker/README.md](../docker/README.md)。

## 14. 当前支持边界与建议评审顺序

当前统一路径支持 Qwen3 Dense 和 Qwen3-MoE，使用同步 RL、BF16 和已配套的 Ascend 环境。
两端沿用同一计划及数值规则，已验证场景实现了不同 batch 和 DP/FSDP/TP/SP/EP 配置间的前向一致。

本版代码明确保留以下后续工作边界：

- CP/PP、vLLM PCP/DCP 尚未接入共享模型路径。
- 专家内部 TP/PIK/ETP 尚未接入；融合专家使用 EP 分片或 TP 复制。
- 当前 `TP1/DP2/EP2 + PIK` 推理配方仍存在重复绑定限制。
- 量化、统一路径的 router replay、异步 RL 不在本版契约中。
- 当前 RL 的 PIK/SP 配方要求 GRPO 和 `train.token_log_probs_mode=dense`。
- 共享 V4 当前使用因果 full attention、零 dropout；sliding window、ALiBi、speculative decoding 等变体被拒绝。
- 公共 functional AllReduce 的编译训练反向单列后续；PIK 提供一阶反向。

建议按以下顺序评审，先核对契约和调用边界，再深入数值算子：

1. 两处公共修改：确认 eager 语义、编译兼容和 SAPP 行数修正。
2. `contract.py → execution_plan.py → config/resolve.py`：确认完整计划没有丢失，约束与配置一致。
3. `trainer/build.py → builder.py → model_adapter.py → parallel_context.py`：确认复用 HP 构建及并行机制。
4. `training.py → paged_attention.py → attention_backend.py → V4 数值接口/补丁`：确认 token、位置、cache 和图元数据。
5. `row_parallel.py → PIK plan/gemm/reduction/collectives/linear`：确认全局算术树、输出布局和梯度边界。
6. `components/moe.py → fixed_capacity_ep.py`：确认原 router/专家、唯一 owner、有序合并和容量语义。
7. 权重发布适配与 `LogicalWeightLoader`：确认不同切分下的逻辑映射、完整性及参数身份。
8. 示例和环境交付：确认源码 patch、wheel、镜像和 YAML 能对应到同一实现。

后续扩展 Qwen3.5 或 DeepSeek 时，先在 `models/` 描述已有 HP 模型，再评估是否需要新增特殊算子
和运行时适配。GDN、MLA 等不能仅靠增加架构名称获得支持；通用的契约、数值策略、权重发布和
已有组件可以复用，特殊状态及 cache 语义应在对应组件和 runtime 中显式实现。
