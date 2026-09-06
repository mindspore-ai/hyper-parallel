# HyperParallel 架构总览(单一事实源)

> **本文件的权威范围 —— 引用前请先读这段。** 本文件只负责一件事:描述
> **顶层模块图**(模块在哪里、如何组合)。在这一件事上,它优先于任何其它文件。
> 它**不负责**以下内容:
>
> | 事实 | 规范出处 |
> | --- | --- |
> | 硬规则与架构不变量 | `.agent/rules/project-overview.md`、`.agent/rules/distributed.md` |
> | RL 内部细节(部署、布局、契约) | `hyper_parallel/rl/docs/`、`.agent/skills/hyper-rl-dev/references/module-map.md` |
> | 功能 → 配置键 → 分支 → 指标 → 测试 | [rl-navigation.md](rl-navigation.md) |
>
> 本文件保持图状结构:读者应能从下面的表格直接回答“X 在哪里”,无需读代码。
> 术语与标识符保留英文(agent 生态惯例);正文用中文。

---

## 0. 这个库是什么

HyperParallel 是一个 **面向 Ascend 超节点(SuperPod)亲和** 的分布式并行加速库,用于模型训练、推理与强化学习,支持 **Ascend NPU 与 Nvidia GPU**,**PyTorch 与 MindSpore** 双后端。它提供 DP、FSDP/HSDP、TP、EP、CP、PP、激活重计算/换出(activation checkpoint/swap)、参数与优化器 offload 的统一抽象;混合策略可自由组合。

三条设计原则(README 有详述,此处仅以表格复述):

| 原则 | 含义 |
|------|------|
| 模型/系统优化解耦 | 系统优化(并行、重计算、offload)隐式注入模型脚本,而非嵌入脚本内部 |
| MPMD 演进 | SPMD → 集群 MPMD → 集群 + 多核 MPMD |
| 有状态 → 无状态 | 计算/状态分离;远端与本地 Tensor 统一编程 |

**映射与架构一一对应**:并行能力位于 `core/`,平台差异位于 `platform/`,RL 位于 `hyper_parallel/rl/`(与其余部分组合)。

---

## 1. 顶层模块图

| 模块 | 路径 | 职责 | 关键子树 |
|------|------|------|----------|
| Platform | `hyper_parallel/platform/` | 抽象层 `get_platform()`;core 内不得 import 后端 | `torch/`、`mindspore/`、`fully_shard/`、`activation_checkpoint/` |
| DTensor | `hyper_parallel/core/dtensor/` | 本地分片 + DeviceMesh + Placements;重分布缓存 | |
| Shard | `hyper_parallel/core/shard/` | `shard_module()` / YAML 算子;`parallel_*.py` | `ops/yaml/`(算子注册表) |
| Tensor Parallel | `hyper_parallel/core/tensor_parallel/` | `parallelize_module()`、`ParallelStyle`、mesh | |
| FSDP / HSDP | `hyper_parallel/core/fully_shard/`、`platform/*/fully_shard/` | 参数 shard/unshard;HSDP(`hsdp_*.py`) | |
| Pipeline | `hyper_parallel/core/pipeline_parallel/`、`platform/*/pipeline_parallel/` | 阶段调度、micro-batch、P2P | |
| Activation | `hyper_parallel/core/activation_checkpoint/`、`platform/torch/activation_checkpoint/` | SAC + 激活 swap | |
| Checkpoint | `hyper_parallel/core/distributed_checkpoint/` | 分布式保存/加载;异步 staging;离线转换 | |
| Collectives | `hyper_parallel/collectives/` | 进程组(`cc.py`) | |
| Models | `hyper_parallel/auto_models/` | HyperAutoModel —— 基于 Transformers 的模型加载与 Trainer | `_transformers/`(内部,注意耦合) |
| RL | `hyper_parallel/rl/` | 同步 LLM RL 运行时,与上述能力组合 | 见 §3 |

> `hyper_parallel/` 下还有 `trainer/`、`models/`、`data/` 等模块;上表列出的是 `AGENTS.md`「Key Modules」所指向的模块。完整清单请直接查看 `hyper_parallel/` 目录。

---

## 2. 平台抽象(不可违反)

- 跨平台的 `core/` 代码**绝不**在模块顶层 import `torch` 或 `mindspore`,而是通过 `hyper_parallel.platform` 的 `get_platform()` 及其返回对象的 API 访问后端。
- 后端相关实现位于 `platform/torch/` 与 `platform/mindspore/`。
- **为什么:** 让同一份算法/特性定义可在两种后端间移植。违反此条属于硬 bug(见 `.agent/rules/distributed.md`)。

`get_platform()` 返回当前平台,其 API 面即“平台契约”。由架构衍生出的规范硬规则(`is_partial()` 是方法、异步读取前必须 `handle.wait()`、`redistribute()` 前须 `reduce_partial()`、`resize_(0)` 释放后不可再访问、跨流需事件同步)集中在 `.agent/rules/project-overview.md` 与 `.agent/rules/distributed.md` —— **不在本文件**。该短清单只保留一处;本文件只描述架构。

### 2.1 这条规则如何被机器强制

pylint 插件 `scripts/pylint_hyperparallel.py` 的 **C9002** 检查上面第一条:

- **只查导入时机为 import 时的导入**(模块顶层,含模块级 `try:` / `if:` 块)。函数或方法内部的惰性 import 是被认可的写法,不报错 —— 模块顶层的 `import torch` 会让整个模块在只装了 MindSpore 的环境里无法加载,函数内的则只在走到该分支时执行。
- **豁免**后端自身的代码:`hyper_parallel/platform/`,以及特性模块自带的 `*/platform/torch/`、`*/platform/mindspore/` 子包(如 `core/multicore/platform/torch/`)。
- 检查器的实际作用域**宽于**本节第一条:它覆盖 `platform/`、`tests/`、`scripts/`、`.agent/` 之外的全部代码,而本节的硬规则只针对 `core/`。`core/` 之外的命中大多在 `auto_models/`、`models/`、`examples/torch` —— 这些层在实践中已与 torch 耦合。**这是已承认的欠账**(2026-09 裁决:按检查器现有作用域执行,不把命中当误报),清偿会随各层重构逐步推进。

存量违规由 `scripts/pre-commit/pylint_baseline.json` 按 (文件, 消息号) 计数冻结,只有新增才会拦截提交;该文件是欠账数量的现时事实源,本文件不复制计数。`core/` 内目前仍存在的模块顶层后端导入集中在 `core/optimizer/` 与 `core/tensor_parallel/mc2*.py` —— 这两处按实现现状是 torch 专用的,清偿方式(改惰性导入,或迁入 `platform/torch/`)属于尚未排期的重构。

---

## 3. 强化学习(Hyper-RL)

> **本节是指针,不是描述。** RL 是仓库中文档密度最高的子系统,其部署图、部署模式、源码布局与接口契约均已有规范出处。在此复述任何一条,都会制造本文件本就要防止的漂移。

| 你要找什么 | 规范出处 |
|------------|----------|
| 部署图、组件、权重发布、一致性边界 | [`hyper_parallel/rl/docs/architecture.md`](../hyper_parallel/rl/docs/architecture.md) |
| 子系统 → 文件映射、接口契约 | [`module-map.md`](../.agent/skills/hyper-rl-dev/references/module-map.md) |
| 设计优先约束、门禁、两个易踩坑的事实 | [`hyper-rl-workflow.md`](../.agent/rules/hyper-rl-workflow.md) |
| Colocated / Disjoint 设备数关系、权重同步事务语义 | [`weight-sync.md`](../.agent/skills/hyper-rl-dev/references/weight-sync.md) |
| 功能 → 配置键 → 分支 → 指标 → 测试 | [rl-navigation.md](rl-navigation.md) |

---

## 4. 相关文档(均链向本单一事实源)

| 受众 | 文件 | 角色 |
|------|------|------|
| Agent(常驻加载) | `AGENTS.md` | 环境法则;链向本文件 |
| Agent(RL 路径生效) | `.agent/rules/hyper-rl-workflow.md` | RL 设计优先 + 约束 |
| Agent(流程 SoT) | `.agent/skills/hyper-rl-dev/` | 实现 + 门禁、module-map |
| 人 | `README.md` | 顶层介绍 + 设计原则 |
| 人(指南) | `docs/index.md` → `docs/guide/*` | 各特性使用指南 |
| 两者 | `docs/rl-navigation.md` | 功能 → 配置键 → 测试 可追溯映射 |
