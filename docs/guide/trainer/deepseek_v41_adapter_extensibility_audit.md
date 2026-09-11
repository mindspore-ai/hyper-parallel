# DeepSeek-V4.1 adapter 与框架扩展性审计

## 1. 审计范围

基线是本地提交 `47ddfde3`：

```text
feat: integrate DeepSeek V4.1 sparse training modules
```

本文回答两个问题：

1. DeepSeek-V4.1 接入代码应属于模型 adapter、可复用组件还是训练框架；
2. 接入下一个新模型时，是否仍存在必须修改框架才能完成模型适配的路径。

判断原则是：模型名称、层号映射、forward 语义和参数布局声明应留在
`models/<family>/adapter`；框架只提供协议、生命周期、并行布局和依赖注入，不能出现
按模型名分支。

## 2. 代码职责归属

### 2.1 应放在模型侧

| 内容 | 当前位置 | 原因 |
| --- | --- | --- |
| 40 层结构到 4 层裁剪的 Full/Reindex/Reuse、Engram 层映射 | `models/deepseek_v41`、training demo | 是 V4.1 的结构语义和验证策略 |
| V4.1 配置字段、placeholder、decoder forward 和共享状态串联 | `models/deepseek_v41/modeling_deepseek_v41.py` | 依赖具体 HF 类、forward 合约和层间共享关系 |
| attention/Engram/mHC 模块替换声明 | `models/deepseek_v41/adapter/replacements.py` 与 recipe | FQN、源模块类型和替换顺序属于 family |
| Indexer Q/merge、MLA、mHC、Engram 的 TP 参数角色 | `models/deepseek_v41/adapter/registration.py` | 框架只消费 `ParamRole`，不能认识 V4.1 参数名 |
| CSA2 的 CP gather 顺序和 offset 传递 | `models/deepseek_v41/adapter/context_parallel.py` | CP 原语通用，但哪个张量 gather 由模型 attention 决定 |
| DeepSeek clamp SwiGLU、router 输出合约和共享专家相加 | `models/deepseek_v41/adapter/expert_parallel.py` | EP dispatcher 通用，路由和激活数学属于模型 |
| compressed packed metadata | `models/deepseek_v41/adapter/packed_sequence.py` | `SharedCompressedPackedSequence` 是该 attention 的输入合约 |
| Online/16 die/4K/EP16/TP 配方、裁剪资产和 handoff | `examples/training_demo` | 是接入与验证配置，不是框架默认策略 |

这些内容不应移动进 Trainer、planner 或通用 CP/EP wrapper。

### 2.2 应放在可复用高性能组件侧

| 内容 | 当前位置 | 原因 |
| --- | --- | --- |
| mHC 系数计算和高性能 post | `components/modules/mhc.py`、`components/functional/mhc_post.py` | 算法与具体模型 FQN 无关，可被多个 family 的替换复用 |
| Engram hash、稀疏表查询和融合 | `components/modules/engram.py` | 是可复用算法模块；表规模和安装层仍由模型决定 |
| CSA2 compressor、Indexer、candidate/Reindex、KL 和 sparse attention | `components/modules/shared_compressed_dsa_attention.py` | 是算子级实现；Full/Reindex 层图仍由模型侧组装 |

组件可以实现特定论文算法，但不能读取 Trainer recipe、匹配模型 FQN 或决定某一层是否启用。

### 2.3 应放在框架侧

| 能力 | 当前位置 | 框架职责 |
| --- | --- | --- |
| 每个 packed sample 的物理对齐 | `data/batching/build_collate_fn.py` | 通用数据语义，防止任意分组/压缩算子跨 sample |
| 按 placement 而非 `experts.*` 名称识别虚拟 EP 参数 | `distributed/_builder` | source mesh 和 FSDP/EP 组合必须对任意参数树成立 |
| 自定义 fused expert gate hook | `distributed/expert_parallel/experts.py` | dispatcher 提供注入点，具体 gate 函数由模型提供 |
| family 自注册 custom model | `models/registry.py` | 新 family 不应编辑中央 architecture 表 |
| 嵌套 target 依赖注入 | `trainer/config` | 通用框架对象应能组合模型侧 adapter，而不需要包装工厂 |
| AutoModel 触发惰性 family discovery | `models/_transformers/config_resolver.py` | 模型构造不能要求调用方预热全局 registry |

首个提交中的框架修改没有引入 DeepSeek 执行分支；它们是把原来依赖参数名或固定
SwiGLU 数学的机制改为语义声明和 callable 注入。

## 3. 发现并修复的三个扩展性缺口

### 3.1 配置不能嵌套构造模型 adapter

原机制只能解析最外层 `_target_`。`ParallelBatch` 已有
`attention_runtime_adapter` 参数，但 YAML 不能向它传入另一个 `_target_` 实例，导致
DeepSeek 必须提供只负责 `new ParallelBatch(...)` 的
`build_deepseek_v41_parallel_batch()`。

修复后：

- resolver 递归识别 target 参数树里的保留 `_target_` 节点；
- `Target.build()` 先构造 dict/list/tuple 中的子 target，再构造父 target；
- `Target.to_dict()` 仍能无损序列化嵌套结构；
- Trainer 注入的 runtime 参数优先级高于 YAML 中的嵌套 target；
- 错误信息保留完整路径，如 `$.target.dependency._target_`。

DeepSeek recipe 现在直接组合通用对象：

```yaml
get_batch:
  _target_: hyper_parallel.data.batching.ParallelBatch
  source_type: online
  attention_mode: compressed
  cp_algorithm: colossal
  attention_runtime_adapter:
    _target_: hyper_parallel.models.deepseek_v41.adapter.packed_sequence.DeepseekV41AttentionRuntimeAdapter
```

因此删除了模型侧包装工厂。后续模型只需要实现 adapter 类并在自己的 recipe 中注入，
无需修改 `ParallelBatch` 或 config resolver。

### 3.2 custom model 注册仍要求调用方预热

`registration.py` 可以调用 `register_custom_model()`，但原来的
`get_is_hf_model()` 直接查询 `MODEL_ARCH_MAPPING`，不会先导入 family registration。
新模型第一次通过 `HyperAutoModel.from_config/from_pretrained` 构造时可能因此回落到 HF
实现。DeepSeek demo 之前用手工 `get_model_adapter("deepseek_v41")` 掩盖了该问题。

修复后，AutoModel 路径判断会按以下顺序执行：

```text
config.model_type / config.architectures[0]
                 │
                 ▼
       lazy discover registration.py
                 │
                 ▼
       resolve custom model class
                 │
          ┌──────┴──────┐
          ▼             ▼
       custom model    HF native
```

DeepSeek demo 已删除 registry 预热。新 family 只需在自己的 `registration.py` 注册，不再
要求模型 builder、Trainer 或中央 registry 增加初始化调用。

### 3.3 虚拟 EP 诊断仍假设所有参数都是 MoE expert

参数分片实现已经按 `EP: Shard(...)` 识别 Engram 表，但 source-mesh 日志和 preflight
错误仍统一称为 routed expert，并只给出 MoE archetype 建议。这不会改变执行结果，却会
让下一个稀疏表或非 MoE 的 EP 模块被错误引导回框架修改。

本次已统一为 virtual-EP parameter 语义：routed MoE 使用通用 archetype，稀疏表等特殊
模块从 `models/<family>/adapter` 提供 `local_compute_fn`，自身已包含 all-to-all 的模块则
显式声明 `region_dispatch: false`。

## 4. 新模型接入后的推荐边界

普通的新 HF family 应只新增：

1. `models/<family>/adapter/registration.py`：能力和参数角色声明；
2. 必要的 replacement、CP、EP、loss、packed runtime adapter；
3. family recipe 和 focused tests；
4. 只有出现新的可复用算法时，才新增 `components/modules` 实现。

以下情况才合理修改框架：

- 出现新的并行语义，现有 placement/collective 无法表达；
- 多个 family 都需要相同生命周期或依赖注入能力；
- 新硬件算子需要平台层或 custom-op 的通用封装；
- checkpoint/FSDP 无法描述一种新的参数所有权关系。

“参数名不同”“forward 参数不同”“某模型使用特殊 router/gate”“某 attention 需要额外
packed metadata”都不再是修改框架的理由，应该由 adapter 和 recipe 处理。

## 5. TP=1 下边界声明不属于扩展性泄漏

`mhc`、Engram 和 shared attention 即使在 TP=1 也仍需要 recipe 边界，原因是边界还承担：

- 模块替换及 forward 安装；
- EP/CP local compute 注入；
- 嵌套 FSDP unshard/reshard 生命周期；
- 参数 source layout、checkpoint 和未来 TP2 的稳定合约；
- required replacement 的 fail-fast 校验。

TP=1 只会把 TP collective 退化为 identity，并不会使模块结构或上述生命周期消失。
因此边界声明保留在模型 recipe 是正确的，不应把 mHC/Engram 名称写进 framework 进行
自动猜测。

## 6. 尚未由本次扩展性修改覆盖的能力

- V4.1 的 production FP4/QAT 需要真实量化算子与 checkpoint 合约，属于组件/平台能力，
  不能靠 adapter 元数据模拟；
- PP>1 的 shadow indexer、跨 stage payload 和共享状态生命周期需要通用 PP 扩展；
- 训练外 KV-cache/replay 属于推理 runtime；
- 完整长序列性能仍需要多步 profile 和 overlap timeline。

这些是尚缺少的真实运行能力，而不是仍残留的 DeepSeek 模型名分支。

## 7. 当前验证

```text
3 passed   tests/ut/trainer/test_target.py
5 passed   tests/ut/auto_models/_transformers/test_config_resolver.py
19 passed  tests/ut/auto_models/models/deepseek_v41/test_deepseek_v41_crop.py
62 passed  nested-target、plan override、virtual EP 和 source-layout 影响面回归
20 passed  Trainer、models registry 与 public API 回归（另含 2 个 subtests）
```

另外已验证完整 training YAML：

- Transformers 5.13.0 被正确选中；
- `dataloader.get_batch` 解析为通用 `ParallelBatch`；
- `attention_runtime_adapter` 保持为延迟构造的模型侧 target；
- `TrainerConfig.to_dict()` 能无损保留该嵌套 target。

16 die、4K、EP16、FSDP16、TP1 的 Online 单步训练也通过新构造路径干净退出：

```text
loss=12.8242  grad_norm=67.3168  step_time=18.7093s
device_max_allocated=50.0054GB  device_max_reserved=60.0371GB
```

首次运行在前反向完成后的梯度裁剪处遇到一次 HCCL `EJ0003` 端口绑定冲突；清理进程后
原命令重试通过，数值与首个提交的 TP1 基线一致，因此该现象归为环境侧延迟建组冲突，
而不是嵌套 target 或模型惰性发现的回归。

本节修改按要求保留在工作区，未创建第二个 commit。
