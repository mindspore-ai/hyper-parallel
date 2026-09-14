# Qwen3-4B 接入当前 master：代码文件与适配方案

本文记录当前隔离适配实现及文件边界，更新日期为 2026-09-14。
它取代此前通过修改共享 planner、参数绑定和物化流程完成接入的方案。
运行时职责见 [架构文档](architecture.md)，策略发布协议见 [vLLM Rollout](vllm_rollout.md)。

最新结论：PPO 两卡 FSDP/full-gather、四卡 FSDP2×TP2/direct-reshard 均通过两步真实训练，
双角色 checkpoint 在新容器恢复后继续两步通过，GRPO 四卡一致性回归也通过。
标准加载和并行适配已迁入 `models/qwen3/runtime.py`；
旧 `rl/roles/model_runtime.py` 已删除，价值头及价值前向由 `rl/roles/policy/critic.py` 持有。
运行配置及当前验证记录以 [PPO 文档](ppo.md) 为准。
第 8–10 节保留此前 GRPO 接入的历史验收和未覆盖范围，不能视为 PPO 全部组合的通过结论。

## 1. 目标和修改范围

目标是让迁入的 RL 项目使用当前 HyperParallel 的 Qwen3 模型组件、加载器及并行能力，
保留既有 GRPO、Actor/Reference、vLLM 注册、agentic、权重同步和 checkpoint 流程。

本工作区的适配目标是 `hyper-parallel_8581` 当前检出的框架提交 `b7d72334`。
RL 和 Qwen3 均使用此工作区中的对应目录；验证直接挂载该工作区，不用其他 master
快照替代它，也不通过 rebase 或修改共享框架文件完成适配。下文 `620130b2` 的记录是此前的
隔离验证历史，不能当作本工作区本轮复验结果。

代码改动限定在：

- `hyper_parallel/models/qwen3/`：模型族注册和模块替换工厂。
- `hyper_parallel/rl/`：RL 配置转换、实例级兼容、rollout 接口、权重发布和测试。

共享模型构建器、planner、classifier、参数切分实现和目录外测试均已恢复 Git 原版。
不修改根目录 `setup.py`，迁移适配也不向共享类或模块新增全局 monkey patch。
用户提供的 `.agent/rules/hyper-rl.md` 保留原样。

### Torch 运行接口

RL 直接使用 `torch` 和 `torch.distributed` 的 Tensor、通信、设备与随机状态接口。
NPU kernel、HCCL 和 IPC 路径保留；Qwen3 adapter 继续使用原生 Torch/torch-npu。
启动配置中的 `HYPER_PARALLEL_PLATFORM=torch` 仅为仍在迁移的共享 HyperParallel 底层选择
Torch 后端，避免同时安装 MindSpore 时选错后端；RL 自身不再查询框架抽象。

### Native-core master 的生命周期适配

上游将 DTensor、DCP 和 Pipeline 通信迁向 native Torch/core 后，RL 的退出清理改由
`rl/distributed.py::destroy_process_group` 承担。它调用 Torch 原生进程组接口，并通过 DeviceMesh 缓存清理入口清除共享
runtime 使用的进程内缓存；P2P 缓存从 `core/pipeline_parallel/_p2p.py` 导入。
共享 DeviceMesh 提供缓存清理入口，RL 不再直接依赖旧通信组注册表。
未初始化或重复清理不会重复销毁进程组；后端销毁报错时，缓存仍被清除，错误继续传播。

DCP 保存/恢复仍使用正式 `save/load` API。CPU round-trip 测试 mock RL 与 DCP 共用的 Torch rank/world/barrier，以及 CPU/设备 RNG
接口；真实模型文件、Adam 状态、
RNG 与数据进度的断言保留。回归入口为 `tests/ut/rl/trainer/test_distributed.py` 和
`tests/ut/rl/trainer/test_checkpoint.py`。

此前隔离验证基线为上游 `620130b2`。验证时只带入上述两个允许目录的快照，
基线的其他受 Git 跟踪文件不变；代码修复均位于 `hyper_parallel/rl/`。

固定镜像、真实 Qwen3-4B/GSM8K 的验收结果：

| 场景 | 卡数 | 完整通过证据 |
| --- | ---: | --- |
| `checkpoint-resume` | 2 | 保存 V1，新容器恢复后发布 V2，再次保存及 HF 导出 |
| `ppo-checkpoint-resume` | 2 | 双角色 V1 checkpoint，新容器恢复后完成 step2/step3，保存 V3 |
| `dense-tp1-full` | 2 | 两步更新、8 条 evaluation（accuracy=0.375）、checkpoint 和 HF 导出 |
| `dense-tp2-consistency-full` | 4 | 两步更新前逐位零误差、更新后负对照有效、bucket 全部 ACK/release |
| `dense-tp2-consistency-direct` | 4 | 两步逐位一致性、负对照及 direct-reshard 发布 |
| `dense-disjoint-consistency-full` | 8 | 训推分离下两步逐位一致性、HCCL 发布及清理 |
| `dense-disjoint-consistency-direct` | 8 | 训推分离下两步逐位一致性、direct HCCL 发布及清理 |
| `ppo-tp2-direct` | 4 | Actor/Critic 两步非零更新，Actor 发布 V1/V2 |

首轮为 7 passed / 1 failed：四卡 full-gather 场景被 SIGKILL 中断，未定位其原因。
保留原配置独立复测后为 1 passed，8 个目标场景均取得完整通过记录；原始失败报告保留。
CPU 的 RL UT 与 ST 自检查联合回归为 449 passed，另有 46 个 subtest 通过，行覆盖率 86.47%。
结果位于 `hyper_parallel/rl/output/latest-master-adapt-20260914/`；
较大的 checkpoint 保留在 `/tmp/rl-master-adapt.8jNDHx/npu/`，临时目录重启后不保证保留。
该轮未重跑 Codex/DeepSeek 真机 Agent 场景，不能据此新增这些场景的通过结论。

初始迁移按 `consistency.enabled=false` 接入和验证普通训练，并保留原有一致性 profile。
按后续要求追加的开启配置验证见 [训推一致性文档](qwen3_training_inference_consistency.md)。
后续 PPO 工作在同一同步训练器接入独立 Critic，具体接口和支持范围见 [PPO](ppo.md)。

## 2. 当前调用关系

```text
RL YAML
  → rl/config.py::build_runtime_config
  → rl/roles/model.py::build_role_model
  → 按 consistency.enabled 选择模型 Target
      ├─ false：models/qwen3::build_causal_lm(fused=True)
      │    → 取得 models/qwen3 的 RMSNorm / GQAAttention 工厂
      │    → 复制 DistributedSetup，添加本次构建的替换和并行声明
      └─ true：models/qwen3::build_causal_lm(fused=False)
           → 沿用原有一致性 profile 的 HF / packed attention 路径
           → 不注入普通模式的融合 QKV replacement
  → Qwen3AutoModel.from_pretrained（两条路径共用，继承共享加载器）
          → 共享 HF 配置解析、模型创建及 checkpoint 加载
          → Qwen3ShardingPlanner → 原版 ShardingPlanner
          → 原版参数切分
          → _Qwen3FSDP2Manager → 原版 FSDP2Manager
          → 本实例 to_empty 后恢复 tied 参数
  → Actor：logprob、算法 loss、反向及 optimizer step
  → PolicySnapshot → full_gather / direct_reshard → vLLM
```

训练模型仍由共享 `HyperAutoModelForCausalLM` 的构建流程创建。
模型目录中的子类只在现有构建边界选择适配对象，没有复制完整加载器、TP 算法或 FSDP 算法。
Critic 调用标准构建接口并传入 `model_transform`，由 RL 添加价值头及其前向，再进入共享并行构建。
开启一致性时，原有 `rl/consistency/qwen3_dense.py` 安装数值 profile，训练侧使用
可反向传播的 FA2 varlen，Hyper-vLLM 使用 FA3 KV-cache attention；它不是普通模式的
融合 GQAAttention 路径。两种模式均使用模型目录中限定于本次实例的 tied 参数、planner 和物化适配。

## 3. Qwen3 模型目录

Qwen3 保留原有五个 adapter 文件，并增加标准模型构建入口：

| 文件 | 当前职责与改动 |
| --- | --- |
| [`models/qwen3/__init__.py`](../../models/qwen3/__init__.py) | 提供 `get_adapter_spec()` 和惰性 `build_causal_lm()` 入口 |
| [`models/qwen3/runtime.py`](../../models/qwen3/runtime.py) | 从 RL 移入标准 Qwen3 加载、replacement 选择及实例级并行适配；接收构建前转换回调 |
| [`adapter/__init__.py`](../../models/qwen3/adapter/__init__.py) | 导出 spec 和三个 replacement 工厂，保持初版结构 |
| [`adapter/registration.py`](../../models/qwen3/adapter/registration.py) | 注册 `Qwen3ForCausalLM` / `qwen3`，保持初版 provider 声明 |
| [`adapter/attention.py`](../../models/qwen3/adapter/attention.py) | 处理 Transformers mask 到 Ascend Attention 的约定，保持初版实现 |
| [`adapter/replacements.py`](../../models/qwen3/adapter/replacements.py) | 复用共享 RMSNorm、GQAAttention、SwiGLUMLP；在初版基础上补充非 meta QKV 权重复制 |

此前新增的 `adapter/runtime.py`、`adapter/plan.py`、`adapter/mlp.py` 和 `adapter/weight_layout.py` 已删除。
实例级兼容目前集中到 `models/qwen3/runtime.py`；此前的分组 MLP 实现不再使用。
模型目录不包含 Critic Value Model、价值头或 PPO 算法实现。

非 meta 权重复制是必要的模型工厂修正：单 rank HF 加载路径先加载真实权重，再替换 Attention。
共享 GQAAttention 构造函数创建新的融合 QKV 参数，工厂需要调用它声明的 converter 填入原 Q/K/V 权重，
否则新参数仍是初始化值。meta 路径继续由共享 checkpoint loader 执行转换和加载。

RL 普通模式（`consistency.enabled=false`）当前选择的 replacement：

| HF 模块 | RL 使用的实现 |
| --- | --- |
| input/post-attention layernorm、最终 norm | Qwen3 工厂返回的共享 RMSNorm |
| self_attn | Qwen3 工厂返回的共享 GQAAttention |
| self_attn 内的 q_norm/k_norm | 由 GQAAttention 保留原模块 |
| MLP | 保留 HF 原生 gate_proj/up_proj/down_proj |
| embedding/lm_head | 保留原模块，通过 RL 适配处理 tied 关系 |

初版 SwiGLU replacement 工厂仍然存在，但 RL 不启用该工厂。
其 gate/up 拼接布局需要额外的 TP 分段切分支持；当前选择原生 MLP，沿用已有的 column/row parallel 规则。
一致性模式不调用上述 replacement 选择入口，继续保留 HF Q/K/V 和 MLP 参数结构，
数值算子由原有 consistency profile 配置。

## 4. 标准模型的实例级兼容

实现文件：[`models/qwen3/runtime.py`](../../models/qwen3/runtime.py)。
旧 `rl/roles/model_runtime.py` 已删除；生产代码和测试直接使用模型目录的公开构建入口。

| 入口 | 作用范围与职责 |
| --- | --- |
| `_require_qwen3` | 要求 `config.model_type='qwen3'`，非 Qwen3 调用直接拒绝 |
| `_PlannerModelView` | 仅向这次 planner 调用暴露不去重的参数枚举，保留 embedding 和 lm_head 两个 FQN；不改原模型的枚举方法 |
| `Qwen3ShardingPlanner.plan` | 将该视图交给原版 planner，复用分类、形状检查和计划推导 |
| `restore_tied_parameter` | 仅处理当前 Qwen3 的 embedding/lm_head；检查形状及 requires_grad 一致后绑定同一个 Parameter |
| `_Qwen3FSDP2Manager.parallelize` | 在共享 FSDP 发现参数所有者之前执行 retie，随后调用原版 manager |
| `adapt_materialization` | 仅包装当前模型实例的 `to_empty`，调用原方法后恢复 tied 参数；不修改 `nn.Module.to_empty` 或共享 builder |
| `apply_qwen3_sharding_plan` | Hyper-vLLM 使用的接入入口：调用共享 apply，再恢复本实例的 tied 参数身份 |
| `Qwen3AutoModel` | 继承共享加载器，替换本次构建的 planner/manager 对象，并在 HF 模型创建后安装本实例的物化适配 |
| `get_module_replacements` | 从 Qwen3 spec 获取现有工厂，返回 RMSNorm、Attention 的标准替换声明 |
| `get_parallel_overrides` | 为融合 Attention 声明 `region_dispatch=False` |
| `from_pretrained` | 使用 `dataclasses.replace` 复制 setup，调用上述加载入口并补齐单 rank HF 导出所需的转换记录 |

标准模型 runtime 的实例级适配边界：

- 不给共享 `ShardingPlanner`、`ParameterClassifier`、FSDP manager 或 builder 重新赋值。
- 不修改 HF Qwen3 类的全局方法；物化包装只写入被构建实例。
- 不修改调用方传入的 `DistributedSetup`。
- shape 或 requires_grad 不一致时拒绝 retie，不静默覆盖矛盾的参数策略。
- 共享能力仍按原接口调用。适配继承了共享加载器的私有构建扩展点，master 后续改变这些内部签名时需重新检查兼容性。

上述实例隔离不应误读为“一致性 profile 没有进程级作用”：迁入时已有的
`rl/consistency/qwen3_dense.py` 在显式开启后注册 attention、启用 batch-invariant，
并替换该进程的 HF Qwen3 RMSNorm forward。该行为不是此次 tied-weight 适配新增的共享补丁，
但开启后不能在同一进程切回 off；普通/一致性测试分别使用新容器和新进程。

## 5. RL 运行代码改动清单

以下文件均位于 `hyper_parallel/rl/`。

Rollout 公共服务、请求调度、拓扑与生命周期保留在 `rl/roles/rollout/`，模型专属推理实现按
family 放在 `rl/roles/rollout/models/`。Qwen3 的 `model.py` 负责 vLLM 模型接口，
`attention.py` 负责 paged attention 边界；公共训练构建仍归 `hyper_parallel/models/qwen3/`。
插件惰性加载 `rl.roles.rollout.models.qwen3.model:HyperQwen3ForCausalLM`，模型架构名及 YAML
配置不变。直接导入旧模型适配模块的下游代码需要更新导入路径；Native-vLLM 仍使用原生模型实现。

```text
rl/roles/rollout/
├── base.py / registry.py       # 生成合同及引擎注册
├── topology.py / worker.py     # 部署拓扑及角色级编排
├── vllm.py                    # 公共 HTTP、服务进程和生成生命周期
├── vllm_plugin.py             # 插件入口，按架构名惰性注册模型
└── models/
    └── qwen3/
        ├── __init__.py        # 轻量包标识，不提前导入 vLLM 模型
        ├── model.py           # 模型包装、输入适配、TP 装配及加载接口
        └── attention.py       # HF 投影与 vLLM paged attention 对接
```

导入迁移：`rl.roles.rollout.vllm_qwen3` 改为 `rl.roles.rollout.models.qwen3.model`；
原 `vllm_qwen3_common` 的 `Qwen3PagedAttention` 移至 `models.qwen3.attention`，
`join_prefix`、`config_value` 和 `normalize_positions` 移至其调用方 `models.qwen3.model`。
不保留旧路径转发模块，不新增自动发现框架。后续 family 的专属实现放入 `models/<family>/`，
公共服务与通信流程不随模型复制；权重同步和 consistency 仍由原目录维护。

本次目录调整不改变模型架构名、参数名、forward、attention、TP 规则或权重加载计算，
既有 YAML 和 Docker 实验入口继续使用，无需修改配置。下文真机数据按各自标注版本解释。

2026-09-13 目录重组工作区使用固定镜像和真实 Qwen3-4B/GSM8K，四卡 colocated
（Trainer FSDP2×TP2、rollout DP2×TP2）复验：普通 Hyper/direct-reshard 和
Native/full-gather 均通过两步非零更新与 V1/V2 发布，Native 每步 73 个 bucket 全部确认并释放。
一致性初次复验和未修改基线均遇到 batch-invariant 求和接口错误；恢复 Actor/Critic 的
显式维度计数后，Hyper/full-gather 与 direct-reshard 均通过两步非零更新、V1/V2 发布及
`0/0/0` 门禁，详见[一致性复验说明](qwen3_training_inference_consistency.md)。
这些结果不扩展到 TP1、disjoint、PPO 或恢复场景。

| 文件 | 改动 |
| --- | --- |
| [`rl/config.py`](../rl/config.py) | Actor/Reference Target 调用 `models.qwen3.build_causal_lm`；Critic Target 调用 `roles.policy.critic.build_value_model`；解析独立 Critic optimizer 与批次配置 |
| [`rl/roles/model.py`](../rl/roles/model.py) | 保留 Actor/Reference 构建与冻结流程；optimizer 改从 `runtime_config.optimizer.target` 构建 |
| `rl/roles/model_runtime.py` | 已删除；不再维护重复兼容转发层 |
| [`rl/roles/policy/critic.py`](../rl/roles/policy/critic.py) | 在共享并行构建前添加价值头、声明 TP 复制布局；输出逐 token value，执行独立 Critic 更新 |
| [`rl/trainer.py`](../rl/trainer.py) | 使用当前分布式初始化入口；纯 TP 时保留 size-one FSDP，以承接布局元数据和 replicated gradient 归约 |
| [`rl/checkpoint.py`](../rl/checkpoint.py) | PPO 保存/恢复 Actor 和 Critic 及两套 optimizer/scheduler；恢复增加局部 DTensor 分发保护；HF 仍只导出 Actor |
| [`rl/roles/rollout/models/qwen3/model.py`](../rl/roles/rollout/models/qwen3/model.py) | Qwen3 专属推理包装；通过公共模型目录的 scoped planner/apply 入口接入 Hyper 并行能力 |
| [`rl/roles/rollout/models/qwen3/attention.py`](../rl/roles/rollout/models/qwen3/attention.py) | Qwen3 paged attention/KV cache 适配；保留 HF 投影、Q/K norm 和 RoPE |
| [`rl/roles/weight_sync/model_adapter.py`](../rl/roles/weight_sync/model_adapter.py) | 描述训练融合 QKV 到标准 HF 权重的行区间；支持 packed 元数据和 direct source 分片描述 |
| [`rl/roles/weight_sync/packed_weight.py`](../rl/roles/weight_sync/packed_weight.py) | 从 DTensor 的本地 tensor 读取标量元数据；接收端按行区间恢复标准 Q/K/V 权重 |
| [`rl/roles/weight_sync/transfer.py`](../rl/roles/weight_sync/transfer.py) | 绑定源模型配置，并将 QKV 转换元数据附加到 full-gather bucket；事务顺序保持原样 |
| [`rl/roles/weight_sync/ipc.py`](../rl/roles/weight_sync/ipc.py) | CPU-offload 的 packed buffer 在广播前搬到通信设备，避免发送方走 CPU/Gloo、接收方走 NPU/HCCL；保留未确认 buffer 生命周期 |
| [`rl/roles/weight_sync/hccl.py`](../rl/roles/weight_sync/hccl.py) | 发送 buffer 在 worker receive RPC 启动前搬到 HCCL group 的设备 |

权重发布仍支持 `full_gather` 和 `direct_reshard`，不自动回退或替换用户选择的策略。
MLP 已恢复原生参数命名，因此普通模式的训练源只需要反转换融合 QKV；
一致性模式保留原始 Q/K/V 命名，不执行该融合反转换。
Native-vLLM 自身的 QKV/gate-up 融合布局继续由原有目标端描述处理。

GRPO 数学、Actor loss/update、agentic runner、工具与环境、Codex/DeepSeek harness 业务代码未改写。
功能保留需要通过回归和真实运行验证，不能仅凭源码未改动就宣称所有运行组合均通过。

## 6. 插件和启动文件

| 文件 | 改动 |
| --- | --- |
| [`pyproject.toml`](../pyproject.toml) | 新增 RL 独立安装声明，以及 `vllm.general_plugins` 的 `hyper_parallel` 入口 |
| [`examples/scripts/install_runtime.sh`](../examples/scripts/install_runtime.sh) | 新增容器安装步骤：将 RL 安装源复制到临时目录构建，源码仍从只读挂载路径导入 |
| [`run_qwen3_tp_docker.sh`](../examples/scripts/run_qwen3_tp_docker.sh) | 调用安装脚本 |
| [`run_qwen3_consistency_docker.sh`](../examples/scripts/run_qwen3_consistency_docker.sh) | 调用安装脚本，提供一致性启动入口；本轮真机结果来自第 9 节记录的 ST 入口，不等同于逐项验证此脚本的所有选项 |
| [`run_qwen3_4b_agentic_docker.sh`](../examples/scripts/run_qwen3_4b_agentic_docker.sh) | 调用安装脚本 |
| [`run_qwen3_4b_deepseek_agentic_docker.sh`](../examples/scripts/run_qwen3_4b_deepseek_agentic_docker.sh) | 调用安装脚本 |

既有 [`hyper_parallel_vllm_plugin.py`](../hyper_parallel_vllm_plugin.py) 转发模块保留；
根目录 `setup.py` 不增加 RL 或 recipe 打包配置。

## 7. 已撤销的目录外修改

以下文件已逐字节核对，与当前 checkout 的 `HEAD` 一致：

| 文件 | 当前处理方式 |
| --- | --- |
| [`distributed/_builder/parameter_sharding.py`](../../distributed/_builder/parameter_sharding.py) | 恢复原版共享 storage 行为；Qwen3 的对象绑定在 RL retie 中完成 |
| [`distributed/_builder/planner.py`](../../distributed/_builder/planner.py) | 恢复原版参数形状枚举；RL planner 视图提供别名 |
| [`distributed/tensor_parallel/param_role.py`](../../distributed/tensor_parallel/param_role.py) | 恢复原版参数分类枚举；不改变其他模型的分类输入 |
| [`models/_transformers/model_builder.py`](../../models/_transformers/model_builder.py) | 恢复原版物化逻辑；RL 只包装目标实例的 `to_empty` |
| [`tests/ut/auto_models/distributed/test_apply.py`](../../../tests/ut/auto_models/distributed/test_apply.py) | 恢复原测试预期，包括两个 Parameter 保留各自身份的断言 |

`git diff --exit-code` 已通过。由于迁入的 RL 和新建的 Qwen3 目录尚未加入 Git，
不能把这个空 diff 理解成这两个目录没有新增代码；它证明原先修改的已跟踪共享文件已恢复。

## 8. 测试文件与此前 GRPO 验证状态

迁入项目缺失的 `tests/ut/rl/` 已从来源项目补回。在此基础上，本次新增或调整的主要测试为：

- [`trainer/test_qwen3_master.py`](../../../tests/ut/rl/trainer/test_qwen3_master.py)：加载、logits/梯度、TP 计划、原生 MLP、权重转换、HF 导出、retie、物化和全局隔离。
- [`trainer/test_config_runtime.py`](../../../tests/ut/rl/trainer/test_config_runtime.py)：当前加载 Target、optimizer 配置及 dtype。
- [`trainer/test_trainer_orchestration.py`](../../../tests/ut/rl/trainer/test_trainer_orchestration.py)：运行时初始化和纯 TP 的 size-one FSDP。
- [`policy/test_actor_roles.py`](../../../tests/ut/rl/policy/test_actor_roles.py)：角色和 optimizer 构建接口。
- [`weight_sync/test_packed_weight.py`](../../../tests/ut/rl/weight_sync/test_packed_weight.py)：DTensor 本地元数据及 packed bucket。
- [`weight_sync/test_weight_sync_transport.py`](../../../tests/ut/rl/weight_sync/test_weight_sync_transport.py)：设备搬运、IPC/HCCL 和失败时 buffer 保留。

真机复验补回了迁入时缺失的 ST 框架文件：
[`st_runtime.py`](../../../tests/torch/rl/st_runtime.py)、[`st_evidence.py`](../../../tests/torch/rl/st_evidence.py)、
[`test_rl_st.py`](../../../tests/torch/rl/test_rl_st.py)、[`test_st_support.py`](../../../tests/torch/rl/test_st_support.py)。
继续使用原有 `_launch.py` / `_worker.py` 和正式训练入口；容器安装当前 RL 插件，
非一致性场景关闭测试配置中的 batch-invariant，Codex 测试提示要求非登录 shell。
这些测试设置不改变生产 YAML、模型实现、算法、reward 或 Agent 会话上限。

PPO 接入前的隔离版本验证记录如下；本次 PPO 的新验证见 [PPO](ppo.md)。

| 检查 | 结果与范围 |
| --- | --- |
| 早期隔离版本 CPU 回归 | 历史批次为 261 passed，另有 12 subtests passed；不与后续重复运行累计 |
| 固定 RL 镜像 | Transformers 5.5.4 下 24 项模型 CPU 测试通过；不等于 NPU 验证 |
| 静态检查 | 修改代码 pylint、语法检查通过；AGENTS catalog 通过；已检查的文档没有新增失效本地链接 |
| 当前隔离版本四卡验证 | FSDP2×TP2 + Hyper-vLLM DP2/TP2、direct-reshard 两步通过；第二步梯度范数 2.03125、发布 V2 |
| 当前隔离版本 TP1/evaluation ST | FSDP2 + Hyper-vLLM DP2/TP1、full-gather 两步通过；梯度范数 1.22656 / 0.000471115；V2 在 GSM8K 前 8 条上评估为 2/8，完成 step2 保存；详见 ST 文档 |
| 本轮 CPU 复验 | RL UT（排除 consistency）与恢复的 ST 框架自测 242 passed；原版共享 apply/planner 另有 35 passed，合计 277 passed |
| 一致性 CPU 追加复验 | 此前排除的 `tests/ut/rl/consistency` 为 10 passed，在宿主 CPU 环境运行，不作为 Docker/NPU 通过证据 |
| 断点恢复 ST | 第一阶段非零更新、保存及 HF 导出通过；第二阶段在 optimizer.load_state_dict 触发 `_fused_adamw_` 缺少并行 layout infer，尚未完成恢复训练 |
| DeepSeek agentic ST | Native-vLLM 两步训练与 full-gather 发布通过；梯度范数 2.73438 / 4.40625，32 个真实 session、64 次 completion，工具往返、token/logprob 与释放全部通过 |
| 八卡训推分离 ST | Trainer FSDP2/TP2 四卡与 Hyper-vLLM DP2/TP2 四卡使用独立设备集合；两步 direct-reshard/HCCL 发布通过，梯度范数 1.82812 / 0.000667572，版本到 V2 |
| 标准四卡 TP2 ST | FSDP2/TP2、完整重计算，两步 direct-reshard 发布通过；梯度范数 1.75 / 0.000667572，版本到 V2 |
| Codex agentic ST | 非登录 shell 复测后两步流程、V1/V2 发布和 32 个 session 的工具往返/释放通过；但每个 prompt 组内奖励相同，优势和梯度均为 0，严格学习验收仍失败 |
| 训推一致性追加验证 | 按后续要求开启配置，四卡 FSDP2/TP2 的 full-gather、direct-reshard 均完成两步；每步 bit-exact 三项误差均为 0，非零更新并发布 V1/V2；详细 token 和梯度见[一致性文档](qwen3_training_inference_consistency.md) |

六个普通模式正式 NPU 场景按最终运行结果计为 **4 通过、2 未通过**；失败项是优化器断点恢复和
Codex 非零学习验收。详细运行目录、初次失败与复测证据见 [ST 验证记录](hyper-rl-st.md#当前-master-隔离适配验证2026-09-12)。
本轮测试只调整 RL 测试框架与文档，未修改生产代码，也未放宽真实奖励和非零梯度验收。

此前修改共享代码的版本曾通过 TP2 的 full-gather/direct-reshard，以及 FSDP2×TP2 的非零更新测试。
那些结果属于已撤销的实现，不能作为当前隔离版本的真机通过证据。

## 9. 最新训推一致性真机验证

真机环境为 Docker 镜像
`swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64`，
挂载当前 master 源码、本地 Qwen3-4B 权重和 GSM8K 数据，并安装当前 RL 插件。
配置为四卡 colocated：Trainer `FSDP-shard2×TP2`、Hyper-vLLM `DP2×TP2`，
BF16 eager、CPU offload、完整 activation checkpoint、学习率 `1e-6`。

使用现有 `st_runtime.run_case(Case(..., tp=2, exact=True))` 派生独立测试配置：

```yaml
consistency:
  enabled: true
rollout:
  vllm:
    batch_invariant: true
    enable_prefix_caching: true
    enable_chunked_prefill: true
```

此处只列关键开关，完整配置保存在各结果目录的 `phase-1.yaml`；没有把普通生产 YAML
的默认开关改为 true。每个 DP rank 每批 2 个 prompt，每个 prompt 4 个 response，
最多生成 512 token，运行两步，使用真实 reward 和完整有效 action mask。

| 同步方式 | 两步有效比较 token | 更新前 mismatch / max abs / mean abs | 完整场景 |
| --- | --- | --- | --- |
| full-gather | 7296 / 7683 | 两步均为 0 / 0 / 0 | 通过，非零梯度更新并发布 V1/V2 |
| direct-reshard | 7296 / 7688 | 两步均为 0 / 0 / 0 | 通过，非零梯度更新并发布 V1/V2 |

结果目录均在 `hyper_parallel/rl/rl_tests/st/results/` 下：

- `dense-tp2-consistency-full-80ee17d7a5/`
- `dense-tp2-consistency-direct-0f9f0582a5/`

每个目录保留实际配置、`phase-1.log` 和 passed 状态的 `result.json`。
full-gather 每步 73 个 bucket 全部 ACK/release；两种模式的更新后旧策略负对照均有效。
精确梯度、负对照计数和比较定义以 [一致性文档](qwen3_training_inference_consistency.md) 为准。
测试完成后容器正常清理，NPU 已释放；这次验证没有修改生产代码或共享实现。

## 10. 历史失败与尚未覆盖的范围

- **此前断点恢复失败，PPO 接入已补充局部修复。** 历史用例第一阶段保存和 HF 导出通过，第二阶段已加载模型，
  但 optimizer 恢复调用触发 `_fused_adamw_` 缺少 DTensor layout infer；
  该错误已在 RL 恢复接口增加 SkipDTensorDispatch 保护；修复后的真机结果见 [PPO](ppo.md)，
  不能把此前“能保存”或一致性通过理解为“恢复已验收”。
- **Codex 非零学习验收仍未通过。** 复测两步流程、同步和 32 个 session 的真实工具往返及释放通过，
  但每个 prompt 内四个 response 的奖励相同，优势与梯度均为零。
  这是本次工作负载未提供有效学习证据，不应修改 reward 或合并不同运行的部分结果来宣称通过。
- **一致性结论仅覆盖四卡 colocated、matched TP2 和上述固定镜像。** 当前隔离版本尚未验收
  TP1、八卡 disjoint、checkpoint resume、agentic、Native-vLLM 等组合的 bit-exact。
  普通模式八卡或 DeepSeek 通过不能替代其一致性验证。
- **没有长期或更大范围结论。** 未覆盖多节点、TP4/TP8、长期训练稳定性和收敛；PPO/Critic 单独记录验收范围；
  logprob bit-exact 不表示 backward、optimizer state 或不同部署间更新后参数逐位一致。

运行方式和更完整的验收约定见 [ST 文档](hyper-rl-st.md)、[UT 文档](hyper_rl_ut.md) 与
[vLLM Rollout](vllm_rollout.md)。本方案不增加测试专用的生产算法分支。
