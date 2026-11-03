# [RFC]: 在 HyperParallel 接入 HyperMulticore HotReplica

| 状态 | 设计评审稿，2026-11-03 |
| --- | --- |
| 接入范围 | HyperParallel 的 native routed EP 和 DeepSeek V4.1 MegaMoE 适配器 |
| 关联实现 | [HP PR #1393](https://github.com/mindspore-ai/hyper-parallel/pull/1393) 的跨仓重构工作树 |
| 评审决策 | 是否采用默认关闭的模型级开关，并通过可选 Python 接口将每次派发委托给 HyperMulticore |

## 动机（Motivation）

HP 持有模型参数、组织 token 交换并执行 FSDP 钩子，因此能确定一次前向中何时可以读取专家权重。HyperMulticore 持有 MegaMoE 执行资源、SHMEM 生命周期和设备内核，因此能确定何时可以借用权重以及何时可以复用缓冲区。若两个仓库分别解释虚拟专家编号，扩展一种放置策略就需要同时修改框架的 dispatch 和设备端实现；若适配器过早缓存权重视图，反向或重算还可能读到已经 reshard 的存储。

本设计将配置与模型生命周期留在 HP，将路由放置、权重借用和梯度归还委托给同一个提供方。使用者继续在既有 native 或 MegaMoE recipe 中选择执行路径。[原始需求 #1128](https://github.com/mindspore-ai/hyper-parallel/issues/1128)

## 目标设计

HP 对外提供 `enable_hot_replica` 开关。开启时调用 HyperMulticore 的 `hyper_multicore.moonep` 运行时。

### 用户入口与依赖

| 既有执行路径 | enable_hot_replica | 实际执行 |
| --- | --- | --- |
| Native EP | false，默认 | HP 原 native EP |
| Native EP | true | HP 调度 token 交换，外部 MoonEP 提供 placement 和借用参数 |
| MegaMoE | false，默认 | HP 已有 MegaMoE 执行器 |
| MegaMoE | true | 外部 MoonEPMegaMoeExperts，复用原模块参数与配置的 push/pull 模式 |

下列片段位于已有 recipe 对应 replacement 的 implementation 层级。Native 配在 **EP compute replacement**；MegaMoE 配在 **专家模块 replacement**。两个片段分别用于各自的 recipe。

~~~yaml
# DeepSeek V4.1，native EP compute replacement
replace:
  implementation:
    _target_: hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel.deepseek_v41_ep_compute_fn
    enable_hot_replica: true
~~~

~~~yaml
# DeepSeek V4.1，MegaMoE 专家模块 replacement
replace:
  implementation:
    _target_: hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe.DeepseekV41MegaMoeExperts
    enable_hot_replica: true
~~~

启用时安装包含配套接口的 hyper-multicore wheel。A3 完整构建在 hyper-multicore 仓库根目录运行，下列命令适用于已激活兼容 CANN、Torch 和 torch_npu 的环境。安装后按该 wheel 的说明激活 MegaMoE 的算子与 SHMEM 环境；训练进程无需重复编译。

~~~bash
bash build.sh --soc-list ascend910_93 --jobs 24
# 将构建日志最后输出的 Wheel 路径赋给 WHEEL，再用训练环境的 Python 安装
python -m pip install "$WHEEL"
~~~

### 跨仓调用顺序

**Native EP。** 以 dispatch 与 combine 分开执行的模式为例，同步入口复用相同的放置契约。

| 时点 | HP 的动作 | HyperMulticore 返回或保证的内容 |
| --- | --- | --- |
| ① 路由与过滤后 | 保留源 token 索引、路由权重和派发顺序，按有效专家编号统计 local_counts。开启功能时调用 select_native_route。 | 返回当前调用的 selection，其中包含收发分片长度，以及可选的借用上下文。 |
| ② 派发前 | 调用 selection.order_dispatch，采用返回的 states、indices 和 order 发起已有 token all-to-all。 | 三者同步重排，以便 combine 能恢复原 token 次序。 |
| ③ 接收后 | 通过 selection.expert_offset 取得本 rank 偏移，经 selection.call_experts 调用原 experts 模块。 | 检查接收行数。需要借用时，把本次 placement 作为关键字参数传入专家 forward。 |
| ④ 专家 forward 内 | FSDP 前置钩子已完成 unshard。HP 进入 run_local_experts，并提供现有排序、计数和权重布局解析函数。 | 读取当前参数视图，按需交换并执行专家计算。反向由外部实现把借用贡献送回原参数所有者。 |
| ⑤ combine | 使用原路由权重和对应的逆排列聚合输出。按已有策略完成异步等待与分块调度。 | selection 仅属于本次前向，后续路由变化不影响本次反向。 |

HP 的 EPRoutedState 已有调用方按位置读取字段。普通派发继续返回既有九字段结构；开启 HotReplica 的平坦派发使用包装对象携带 selection。分块派发把 selection 放入本次 chunks 状态。该状态随 dispatch 结果交还给 experts-and-combine，避免同一层连续两次 forward 相互覆盖。

**MegaMoE。** DeepseekV41MegaMoeExperts.configure 在 EP 组和 top_k 已确定后，按开关选择执行器，并传入 create_parameters=False。每次 forward 都从原模块取得当前 gate_up_proj、down_proj 的本地视图，通过 expert_weights 传入执行器。在这一条路径上，虚拟路由、workspace 扩容、SHMEM 专家缓冲区以及梯度归并都由外部执行器管理。层间共享执行资源通过实际执行器的 share_execution_resources 完成；生命周期结束后通过 close_runtime 释放。

### HP 改动范围

以下文件路径均相对于 hyper_parallel 包。评审可从业务入口沿表中顺序追踪到两个跨仓调用点。

| 文件 | 职责 |
| --- | --- |
| models/deepseek_v41/adapter/distributed/moe_engram_expert_parallel.py | native compute 工厂接收开关，向路由执行传参；DeepSeek 分组专家计算保留其激活与限幅语义。 |
| distributed/expert_parallel/experts.py | 在原 dispatch/combine 边界调用提供方，将 selection 随本次派发传递；维持原专家模块调用边界。 |
| components/modules/grouped_experts.py | 分组专家 forward 接受本次调用的可选权重视图和梯度累加目标，使已有计算可以使用借来的专家。 |
| components/functional/grouped_matmul.py | 通过 original_weight 等可选参数把权重梯度交给指定累加目标。默认仍使用传入的 weight。 |
| models/deepseek_v41/adapter/distributed/megamoe.py | 按开关选择执行器，提供当前权重并管理模型侧生命周期。 |

## 需要设计评审DFX建议

**参数和 checkpoint 兼容性。** 注册参数仍属于原模型模块，替换执行器时保持参数对象、state_dict 键和已有权重转换关系。优化器继续引用原参数。对 FSDP、checkpoint replay 和 delayed backward，接入层必须在每次进入模块时获取当次视图；跨调用缓存应由掌握资源代次和反向生命周期的提供方负责。

**错误处理和回退。** 未启用时按需导入机制允许 HP 在没有 hyper-multicore 的环境配置默认执行器。显式启用但缺少配套 Python 接口或所需动态库时，调用方可看到导入或加载错误。运行中的“本次收益不足”则是正常结果，仍使用所选后端执行普通路由。运维关闭功能时，将对应 replacement 的开关改为 false，在下次构建模型时生效。

**替代方案与性能代价。** 把完整 HotReplica 嵌入 HP 可以少一个包依赖，但需要在 HP 维护设备构建、SHMEM 生命周期及硬件适配。直接用外部新模块替换整个 native EP 则需要重做 HP 的异步 token 交换、分块 combine 和 FSDP 钩子。当前边界允许这两套机制各自由已有维护者演进；代价是启用路径需要匹配的跨仓接口，native 每次需交换专家计数以作决策。收益条件和热点基准详见配套实现篇。

**验证与验收。**

| 验收问题 | 本轮证据 |
| --- | --- |
| 关闭时是否新增强依赖 | 在阻止导入 hyper_multicore 的进程中，HP 成功配置原有 MegaMoE 执行器。 |
| 原参数与 autograd 链路是否可用 | HP native 共 24 项通过，包含 11 项单测、12 项四卡 native 用例及 1 项 CPU/Gloo 传输用例。覆盖异步分块、重算、保留图、冻结参数及推理转训练。 |
| 外部执行器能否使用打包产物 | 已安装 A3 wheel 的双卡 MegaMoE 回归中，push 14 项、pull 7 项均通过。 |
| 两个 DeepSeek 单测如何归因 | 外部权重 CPU FP32 严格相等断言、共享专家缺少 limit 属性两项在迁移前 PR 分支 efab7657 上均复现。两个测试体在 PR 分叉点已存在且相同；未在上游分叉点实跑。 |

**发布前待决。** 两个仓库需约定首次提供上述接口的 hyper-multicore 发布版本，并用该版本的 wheel 跑跨仓回归。当前证据覆盖 A3 模块级功能，整模型训练吞吐和其他硬件组合仍需独立验收。

## 相关的RFCs和API

调用者可在现有构造参数中传入 `enable_hot_replica=True`，即 `MegaMoeExperts(..., enable_hot_replica=True)`。默认值为 `False`。

- **配置契约**：enable_hot_replica 为布尔值，默认 false。现有模型 replacement target 与 dispatch_mode 继续决定模型类型及传输模式。
- **Native 接口**：hyper_multicore.moonep.native.select_native_route 返回本次 selection；hyper_multicore.moonep.execution.run_local_experts 消费本次 placement。HP 使用返回对象提供的方法，虚拟槽编号和借用表布局由提供方解释。
- **MegaMoE 接口**：hyper_multicore.moonep.mega_moe.MoonEPMegaMoeExperts 接受 EP 拓扑和原执行配置，forward 接受 expert_weights 及原路由输入。优化器和 checkpoint 面向 HP 模型参数。
- **关联设计**：配套《在 HyperMulticore 实现 MoonEP 动态专家借用》给出收益公式、放置算法、两种传输后端与精度边界。[社区 MoonEP RFC](https://github.com/mindspore-ai/hyper-parallel/issues/1128) · [HP 接入 PR](https://github.com/mindspore-ai/hyper-parallel/pull/1393) · [社区 RFC 模板](https://github.com/mindspore-ai/hyper-parallel/blob/master/.github/ISSUE_TEMPLATE/003-RFC.yml)

## 完整的反馈期限

建议自公开征求意见之日起至少 7 天，具体截止日期由发布者填写。

## CC List

建议邀请 HP native EP、FSDP、DeepSeek V4.1 模型适配和 HyperMulticore MegaMoE 维护者共同评审。

## 其他补充说明

评审时应配对使用 HP PR #1393 与 hyper-multicore PR #7 的配套版本。公开 PR 的早期描述曾从 HP 构建 MoonEP 动态库；本稿所述构建入口和软件包归属以跨仓重构工作树为准。
