# Hyper-Parallel 需求分解（端到端训练流程版）

> 本文档与 `requirements.xlsx` / `requirements.csv` 同源生成（2026-07-22 更新：状态列按 hyper_models/components/distributed 实现实况标注）。
> 组织方式：**按大模型训练端到端流程（S0→S13）列出需求**，每个需求点按「模块 + 功能」划分，可直接交由程序员开发。
> 交付模型：**DeepSeek V3/V4、Qwen3.5、GLM-Image**（基于 HuggingFace 接入）、**Pangu**（自研，走内置模型路径）。
> 实现状态：**已实现 25 / 部分实现 8 / 未实现 86**（hyper_models/components/distributed 已落地，tests/components/distributed 282 用例全绿）。

## 端到端流程总览

| 流程阶段 | 阶段内容 | 阶段目标 |
|---------|---------|---------|
| S0 前置依赖 | TP 梯度同步集成（D-12 二选一）+ 版本钉板 + 交付模型可获得性确认 | 解锁生产模式 TP 梯度同步；不阻塞正确性（可降级校验模式） |
| S1 启动与配置 | CLI 读入 YAML → ConfigNode → RecipeConfig 类型化配置 | 用户写一个 YAML 即可启动训练 |
| S2 分布式环境 | 并行度声明 → DeviceMesh（主 mesh 无 EP 轴，D-10）→ FSDP2 包裹 + TP 梯度元数据 | 多卡通信拓扑与数据并行就绪 |
| S3 数据供给 | tokenizer → dataset → packing → sampler → collate → DataLoader | 数据按 DP/CP 契约切好喂给模型 |
| S4 模型构建 | 模型注册 → from_pretrained → meta 空壳 → PEFT/freeze | 模型结构就绪（零显存） |
| S5 并行分片 | ShardingPlanner 推导 → apply_sharding_plan 应用（双模式） | 每个参数怎么切自动推导，通信预编译【已实现，282 用例全绿】 |
| S6 权重加载 | 键名映射 → 并行加载 → 按 placement 分发 | 预训练权重落到分片后的模型 |
| S7 训练组件 | 优化器（list[Optimizer]）+ LR 调度器 + 损失函数 | 训练步所需组件就绪 |
| S8 训练循环 | setup 编排 → 训练步（前后向/梯度累积/裁剪/更新）→ 定期验证 | 训练主流程 |
| S9 持久化 | checkpoint 保存/恢复 + HF 导出 + 异步保存 + 断点续训 | 训练状态可保存可恢复 |
| S10 高级并行 | PP 切分与调度 + 激活检查点 + 混合精度 | 更大模型/更长序列 |
| S11 模型交付 | DeepSeek V3/V4 + Qwen3.5 + GLM-Image + Pangu(自研) 适配 | 4 个交付模型全部可训练 |
| S12 CLI与监控 | 命令行入口 + 日志 + 指标 + 配置模板 | 训练可启动可观测 |
| S13 质量保障 | 双模式等价 + 组合覆盖 + 4 模型兼容 + E2E 验收 + 文档 | 交付质量兜底 |

> 流程说明：S0→S1→S2 为启动期；S3 与 S4→S5→S6 可并行准备；S7→S8 进入训练主循环；
> S9 贯穿训练周期；S10 按需开启；S11 与各基础模块并行开发；S12/S13 覆盖全流程。

## 需求总表

### S0 前置依赖（小计 5.0 人·日）

| # | 模块 | 功能/需求点 | 需求描述 | 适用模型 | 工时 | 优先级 | 设计文档 | 状态 |
|--:|------|------------|---------|---------|:----:|:------:|---------|:----:|
| 1 | 外部依赖(FSDP2扩展) | TP 梯度同步集成落地(D-12 二选一) | D-12 裁决的集成落地二选一：(a) 调时序使 fully_shard 先于 _local_params_context 解包执行，走已实现的 DTENSOR_UNIFIED 路径（_orig_dtensor_placements + layout-driven 归约组），无需 fork；(b) 补齐 fully_shard(tp_grad_info=...) 消费端（非 stock API，需 fork 或上游 PR）。现状：tp_grad_info 由 planner/applier 产出但全仓库无消费者（仅 grad_equiv.py 测试模拟旁路） | 全部 | 2 | P0 | 05 §6.7.1/§7/§12.2 D-12 | 未实现 |
| 2 | 外部依赖(FSDP2扩展) | _get_base_spmd_placements DTENSOR_UNIFIED 路径 | stock FSDP2 已实现 DTENSOR_UNIFIED 机制（无需 fork 新分支）：_get_base_spmd_placements 经 DeviceMesh.concatenate 拼统一 mesh，_build_layout_driven_group_info 从最终 layout 的 Replicate 轴推 TP 归约组，_normalize_unsharded_grad_to_local 处理 Partial 梯度回流；剩余为集成时序接线（D-12 选项 a/b） | 全部 | 1 | P0 | 05 §6.7.2/§6.7.3 | 部分实现 |
| 3 | 外部依赖(FSDP2扩展) | DeviceMesh.concatenate helper | 已实现：core/dtensor/device_mesh.py:1035（layout-backed 拼合 + 非重叠校验 + 进程组元数据继承），供 FSDP2 DTENSOR_UNIFIED 路径拼合 [dp,tp] 统一 mesh | 全部 | 0.5 | P0 | 05 §6.7.2 | 已实现 |
| 4 | 外部依赖(版本钉板) | 依赖版本钉板与 API 核验 | 钉死并核验 torch/transformers/torchao/peft/accelerate/datasets/torchdata 版本，消除历史 PyTorch API 误用风险 | 全部 | 1 | P0 | — | 未实现 |
| 5 | 外部依赖(模型可获得性) | 交付模型权重/架构可获得性确认 | 确认 DeepSeek V3/V4、Qwen3.5、GLM-Image 在 HuggingFace 的权重与架构字符串来源；Pangu 权重与词表由内部渠道提供，确认交付形式 | DeepSeek V3/V4 / Qwen3.5 / GLM-Image / Pangu(自研) | 0.5 | P0 | — | 未实现 |

### S1 启动与配置（小计 5.0 人·日）

| # | 模块 | 功能/需求点 | 需求描述 | 适用模型 | 工时 | 优先级 | 设计文档 | 状态 |
|--:|------|------------|---------|---------|:----:|:------:|---------|:----:|
| 6 | 配置解析 | YAML 解析与递归包装 | 读取训练 YAML，将内容递归包装为可点号访问的 ConfigNode | 全部 | 1 | P0 | 01 §2.3 | 未实现 |
| 7 | 配置解析 | 组件引用即时解析 | _target_ 字符串 → 实际 Python 类/函数（点分隔导入 + file.py:attr 两种格式） | 全部 | 0.5 | P0 | 01 §2.4 | 未实现 |
| 8 | 配置解析 | 值类型自动转换 | YAML 字符串 → Python 原生类型（ast.literal_eval + 特殊符号映射） | 全部 | 0.5 | P0 | 01 §2.5 | 未实现 |
| 9 | 配置解析 | 嵌套组件递归实例化 | 按需将嵌套 _target_ 配置递归实例化为 Python 对象 | 全部 | 1 | P0 | 01 §2.9-2.10 | 未实现 |
| 10 | 配置解析 | 辅助访问与序列化 | get / to_dict / to_yaml_dict / instantiate_path，含敏感字段脱敏 | 全部 | 0.5 | P1 | 01 §2.7/2.11 | 未实现 |
| 11 | 配置解析 | 导入安全控制 | 白名单前缀限制 + 用户模块开关，防止 YAML 执行未授权导入 | 全部 | 0.5 | P1 | 01 §2.12 | 未实现 |
| 12 | 配置解析 | 类型化配置桥接 | RecipeConfig（canonical 定义在 01 §3.3）：原始配置按语义分类（优化器/调度器/损失/检查点），每类提供类型安全访问 | 全部 | 1 | P0 | 01 §3.3 | 未实现 |

### S2 分布式环境（小计 5.5 人·日）

| # | 模块 | 功能/需求点 | 需求描述 | 适用模型 | 工时 | 优先级 | 设计文档 | 状态 |
|--:|------|------------|---------|---------|:----:|:------:|---------|:----:|
| 13 | 并行拓扑 | 并行度声明 | ParallelismSizes dataclass：用户声明 tp/cp/pp/dp/ep 各维并行度 | 全部 | 0.5 | P0 | 06 §3.1 | 未实现 |
| 14 | 并行拓扑 | DeviceMesh 构建 | MeshContext.build()：按并行度构建多维 DeviceMesh，自动推导 dp_size；主 mesh 无 EP 轴（D-10：expert mesh 于 apply 期由全 dense 区域派生 (edp,ep)）；DTensor 管理 tp/cp，DP 由 FSDP2 管理 | 全部 | 1 | P0 | 06 §3.2 | 未实现 |
| 15 | 并行拓扑 | 网格查询接口 | MeshContext properties：各维 size/rank 实时查询 | 全部 | 0.5 | P0 | 06 §3.2 | 未实现 |
| 16 | 并行拓扑 | 统一拓扑容器 | DistributedSetup.build()：整合 MeshContext + strategy_config + pipeline_config 单一入口 | 全部 | 0.5 | P0 | 06 §3.3 | 未实现 |
| 17 | 数据并行 | FSDP2 策略配置 | FSDP2Config：sequence_parallel/activation_checkpointing/mp_policy/offload_policy | 全部 | 0.5 | P0 | 06 §4.1 | 未实现 |
| 18 | 数据并行 | FSDP2 数据并行包裹 | FSDP2Manager.parallelize()：DTensor 分片完成后对每个参数施加 fully_shard | 全部 | 1 | P0 | 06 §4.2 | 未实现 |
| 19 | 数据并行 | 参数本地化解包 | 已实现：sharding/apply.py 的 _local_params_context 在 FSDP2 包裹前将 DTensor 参数零拷贝替换为 plain local tensor（与 fully_shard 的先后时序属 D-12 集成项） | 全部 | 0.5 | P0 | 05 §4.4 / 06 §5 | 已实现 |
| 20 | 数据并行 | TP 梯度同步元数据 | tp_grad.py build_tp_grad_info(plan, tp_mesh) 已实现：从 ShardingPlan 读取 TP placement；fully_shard 消费端接线待落地（D-12，全仓库暂无消费者，仅 grad_equiv.py 测试模拟旁路） | 全部 | 1 | P0 | 05 §6.7.1 | 部分实现 |

### S3 数据供给（小计 5.0 人·日）

| # | 模块 | 功能/需求点 | 需求描述 | 适用模型 | 工时 | 优先级 | 设计文档 | 状态 |
|--:|------|------------|---------|---------|:----:|:------:|---------|:----:|
| 21 | 数据加载 | 数据加载器主流程 | 串联 tokenizer → dataset → 分片 → packing → sampler → collate → DataLoader（7 步） | 全部 | 1 | P0 | 02 | 未实现 |
| 22 | 数据加载 | 分词器构建 | 4 路分发（无 key / null / 无 _target_ / 有 _target_）加载 tokenizer | 全部 | 0.5 | P0 | 02 | 未实现 |
| 23 | 数据加载 | HF 数据集加载 | datasets.load_dataset()，仅首个 rank 下载 | 全部 | 0.5 | P0 | 02 | 未实现 |
| 24 | 数据加载 | Megatron 数据集兼容 | 兼容 Megatron 预处理后的二进制数据集格式 | 全部 | 0.5 | P1 | 02 | 未实现 |
| 25 | 序列打包 | 文本序列打包 | 多短序列拼接为长序列（THD 格式 + bin-packing 策略），提升 GPU 利用率 | 全部 | 0.5 | P0 | 02 | 未实现 |
| 26 | 序列打包 | 图像打包(NEAT) | GLM-Image 的 NEAT 图像打包：变分辨率图像 patch 序列化与文本交错打包 | GLM-Image | 1 | P0 | 02 | 未实现 |
| 27 | 采样与拼接 | 分布式采样 | 每个 DP rank 获取不重叠数据分片，支持断点续训状态恢复 | 全部 | 0.5 | P0 | 02 | 未实现 |
| 28 | 采样与拼接 | 批次拼接 | per-key padding + labels 对齐 + CP 序列边界对齐 + PP 链式包装；其中 CP 边界对齐 shard_batch_for_cp 已在 cp_utils.py 实现（含 seq_lens 分片），其余环节未实现 | 全部 | 0.5 | P0 | 02 / 05 §6.3.4 | 部分实现 |

### S4 模型构建（小计 7.5 人·日）

| # | 模块 | 功能/需求点 | 需求描述 | 适用模型 | 工时 | 优先级 | 设计文档 | 状态 |
|--:|------|------------|---------|---------|:----:|:------:|---------|:----:|
| 29 | 模型注册 | 模型架构注册表 | 架构名 → 模型类映射（懒加载 import）；HF 模型与自研模型统一注册 | 全部 | 0.5 | P0 | 01 §5 | 未实现 |
| 30 | 模型注册 | 模型类型判定 | 查表判别 HF 原生模型 vs 内置自研模型（Pangu 走内置路径） | 全部（Pangu(自研) 内置） | 0.5 | P0 | 01 §5 | 未实现 |
| 31 | HF兼容层 | HF 标准接口兼容 | 多重继承 HF AutoModel，提供 from_pretrained / from_config / _build_model | DeepSeek V3/V4 / Qwen3.5 / GLM-Image | 1.5 | P0 | 01 §6.1 | 未实现 |
| 32 | HF兼容层 | 多任务类族 | ForCausalLM（DeepSeek/Qwen3.5/Pangu）+ ForImageTextToText（GLM-Image） | 全部 | 0.5 | P0 | 01 §6.1 | 未实现 |
| 33 | HF兼容层 | from_pretrained 完整实现 | distributed_setup / torch_dtype / attn_implementation / validate_placement 等参数 | 全部 | 1 | P0 | 01 §6.2 | 未实现 |
| 34 | 构建编排 | 模型构建全流程编排 | meta device → PEFT → freeze → ShardingPlanner → ShardingApplier → 权重加载 → FSDP2，严格按依赖顺序（fully_shard 先于 to_empty/load，避免二次显存峰值） | 全部 | 1.5 | P0 | 01 §6.3 / 06 §5.2 | 未实现 |
| 35 | 构建编排 | 模型空壳创建 | meta device 实例化（零显存，仅记录 shape/dtype），自研/HF 路径分发 | 全部 | 1 | P0 | 01 §7 | 未实现 |
| 36 | 参数处理 | PEFT 注入 | 分片前插入 LoRA 等适配器层 | 全部 | 0.5 | P1 | 01 §6.4 | 未实现 |
| 37 | 参数处理 | 参数冻结 | 按配置冻结指定参数（embedding、前 N 层等） | 全部 | 0.5 | P1 | 01 §6.5 | 未实现 |

### S5 并行分片（小计 25.5 人·日）

| # | 模块 | 功能/需求点 | 需求描述 | 适用模型 | 工时 | 优先级 | 设计文档 | 状态 |
|--:|------|------------|---------|---------|:----:|:------:|---------|:----:|
| 38 | 分片规划 | 分片数据结构 | 已实现：sharding_config.py 的 NamedPlacement / ModuleShardingSpec / ShardingPlan：参数分片 + in/out 四元 placement 契约 | 全部 | 1.5 | P0 | 05 §3.1-3.2 | 已实现 |
| 39 | 分片规划 | 分片模板库 | 已实现：7 种模块类型（attention/mlp/norm/embed/lm_head/moe_gate/moe_mlp）× SP/non-SP，TP+CP+EP 三维 placement 声明（sharding_planner.TEMPLATES） | 全部 | 1 | P0 | 05 §3.5 | 已实现 |
| 40 | 分片规划 | 参数角色分类(Phase1) | 已实现：命名规则 + ARCH_OVERRIDES 架构覆盖 → 14 种 ParamRole（D-13 新增 REPLICATED：全维 Replicate，仅经 ARCH_OVERRIDES 显式指派，默认规则不产生） | 全部 | 1.5 | P0 | 05 §3.6.6/§12.2 D-13 | 已实现 |
| 41 | 分片规划 | 边界分组(Phase2) | 已实现：参数按通信边界模块聚合（两趟实现：直属分组 + 深度优先向上合并） | 全部 | 0.5 | P0 | 05 §12.3 | 已实现 |
| 42 | 分片规划 | 语义推断(Phase3) | 已实现：FQN 显式模式 > 叶守卫 > 参数角色组合 → boundary_type | 全部 | 0.5 | P0 | 05 §3.6.2 | 已实现 |
| 43 | 分片规划 | 模板匹配(Phase4) | 已实现：TEMPLATES → _build_spec_from_template：14 角色 → placement 映射（ndim 感知，D-08） | 全部 | 1 | P0 | 05 §3.5 / §12.2 | 已实现 |
| 44 | 分片规划 | 链式传播(Phase5) | 已实现：填充缺省 in_src + 相邻契约校验 + terminal 标记（名字无关单 entry 配对） | 全部 | 1.5 | P0 | 05 §3.6.5 / §12.3 | 已实现 |
| 45 | 分片规划 | 特殊参数处理(Phase6) | 已实现：SPECIAL_HANDLERS 注册表机制（已交付 gated_delta_tp_shard）；融合 QKV 走 FUSED_QKV 角色模板；DeepSeek MLA 经内置 ARCH_OVERRIDES（D-13），无需 SpecialHandler | Qwen3.5 / DeepSeek V3/V4 | 1 | P0 | 05 §6.4.6/§12.2 D-13 | 已实现 |
| 46 | 分片执行 | 参数 DTensor 化 | 已实现：distribute_tensor 按 spec.params 切分（含 EP Shard(0) 统一路径），支持 meta tensor | 全部 | 0.5 | P0 | 05 §4.2 | 已实现 |
| 47 | 分片执行 | 预编译通信计划 | 已实现：precompiled_boundary.py PrecompiledBoundary：in_src→in_dst / out_src→out_dst 编译为静态 RedistOp 序列，运行时零推导；identity 维跳过 | 全部 | 2 | P0 | 05 §4.3 | 已实现 |
| 48 | 分片执行 | 生产模式前向包装 | 已实现：boundary 入口 redistribute → local tensor 计算 → boundary 出口（参数已永久解包） | 全部 | 0.5 | P0 | 05 §4.4.1 | 已实现 |
| 49 | 分片执行 | 校验模式前向包装 | 已实现：DTensor 全程传播 + out_src（核心）/out_dst（terminal）校验 + PlacementMismatchError 报告 | 全部 | 1 | P0 | 05 §5 | 已实现 |
| 50 | 分片执行 | CP attention 包装 | 已实现：双模式同一 all-gather wrapper（D-01''，ring 已否决）：cp_utils.flex_cp_allgather K/V all-gather + offset-aware causal mask（D-04） | 全部 | 1 | P0 | 05 §4.4.2 / §12 | 已实现 |
| 51 | 分片执行 | MoE local region 包装 | 已实现：local_region.py（D-03' 泛化骨架）：boundary 入口 → local all-to-all → 按声明 out_src 重包装，双模式共用 | DeepSeek V3/V4 / Qwen3.5 | 0.5 | P0 | 05 §4.4.3 / §12 | 已实现 |
| 52 | 分片执行 | HF 原生 MoE EP 直通(D-09) | 已实现：planner stacked 元数据（_ep_stack/_moe_router）+ Phase A per-expert 参数堆叠为 [E,...] + wrapper 注入 _hf_native_ep_compute（router adapter + 本地 SwiGLU + 加权聚合），HF 单卡 MoE 脚本 EP>1 零改动 | DeepSeek V3/V4 / Qwen3.5 / GLM-Image | 3 | P0 | 05 §6.4.7 | 已实现 |
| 53 | 分片执行 | EP all_to_all 后端分派 | 已实现：NCCL/HCCL 不等长 all_to_all（零填充）+ gloo pad-to-max all_to_all_single（测试路径），_EPAllToAllUneven/_EPAllToAllPadded autograd 双实现 + _ep_all_to_all 统一入口 | DeepSeek V3/V4 / Qwen3.5 | 1 | P0 | 05 §6.4.7 | 已实现 |
| 54 | 分片执行 | MoE router adapter 注册表 | 已实现：MOE_ROUTER_ADAPTERS 8 键（default/qwen3moe/qwen3_moe/mixtral/deepseekv3/deepseek_v3/glm4moe/glm4_moe）3 实现（softmax-topk / topk-router-module / sigmoid-group），未注册 arch 回落 default | DeepSeek V3/V4 / Qwen3.5 / GLM-Image | 0.5 | P0 | 05 §6.4.7 | 已实现 |
| 55 | 分片执行 | TP-extend-EP 扩展专家并行(D-10) | 已实现：MindSpeed TP 扩展 EP / Megatron etp=1+ep 跨 TP 同构：ep_size 即扩展 EP 组大小（无单独 etp 配置），校验 ep_size ≤ dp_replicate×dp_cp×tp 且整除、num_experts % ep_size == 0。MoE 边界 SP-in identity + 全 dense 区域重分区 (edp,ep) + expert 权重仅 {EP:S0}（完整 expert）+ region 内 a2a（无 AG/RS）。D-11：HF 2025 batched 布局（experts.gate_up_proj [E,2I,H]）免堆叠直通 | DeepSeek V3/V4 / Qwen3.5 | 5 | P1 | 05 §6.4.8 / 06 §4.5.1 | 已实现 |
| 56 | 分片执行 | 派生 expert mesh 构建(D-10) | 已实现：sharding_applier._build_expert_mesh——主 mesh 无 EP 轴，apply 期将全 dense 区域 flatten 重分区为派生 expert mesh (edp,ep)，ep_size 整除校验 | 全部 | 0.5 | P0 | 05 §6.4.8 / 06 §4.5.1 | 已实现 |
| 57 | 分片执行 | vocab 并行 embedding 包装 | 已实现：D-02 Megatron 风格 masked embedding（解包后 vocab mask 显式重建），仅 production 注入 | 全部 | 0.5 | P0 | 05 §12.2 D-02 | 已实现 |
| 58 | 分片执行 | 双模式等价验证 | testing/grad_equiv.py 工具已交付（梯度等价 rtol=1e-3 + simulate_tp_replicate_grad_sync 旁路），TP 组合用例测绿；TP×CP/TP×EP 全组合输出等价覆盖与 FSDP2 集成（D-12 接线后）待补 | 全部 | 1 | P0 | 05 §5.5 / dev_plan S5 | 部分实现 |

### S6 权重加载（小计 3.5 人·日）

| # | 模块 | 功能/需求点 | 需求描述 | 适用模型 | 工时 | 优先级 | 设计文档 | 状态 |
|--:|------|------------|---------|---------|:----:|:------:|---------|:----:|
| 59 | 键名映射 | 权重键名映射 | 预定义映射 + 模型自定义映射（如 _fp32_params.A_log ↔ A_log、MLA 参数重命名） | 全部 | 1 | P0 | 04 §5.3 | 未实现 |
| 60 | 键名映射 | MoE/VLM 权重适配 | DeepSeek/Qwen3.5 MoE expert 合并拆分 + GLM-Image VLM 层级映射 | DeepSeek V3/V4 / Qwen3.5 / GLM-Image | 1 | P0 | 04 §5.4 | 未实现 |
| 61 | 并行加载 | 并行权重加载 | 所有 rank 独立读 safetensors，无需跨 rank 通信 | 全部 | 0.5 | P0 | 04 | 未实现 |
| 62 | 并行加载 | 按 placement 分发 | 完整权重写入已分片模型：自动按 placement 切分，保持原始 dtype | 全部 | 0.5 | P0 | 04 | 未实现 |
| 63 | 验证 | 加载往返测试 | save→load 一致 + 跨 TP 配置加载 | 全部 | 0.5 | P0 | 04 | 未实现 |

### S7 训练组件（小计 3.5 人·日）

| # | 模块 | 功能/需求点 | 需求描述 | 适用模型 | 工时 | 优先级 | 设计文档 | 状态 |
|--:|------|------------|---------|---------|:----:|:------:|---------|:----:|
| 64 | 优化器 | 优化器配置与构建 | AdamW 等配置声明 + factory escape hatch + 归一化入口 | 全部 | 1 | P0 | 03 | 未实现 |
| 65 | 优化器 | 优化器实例化 | decay/no_decay 分组 + model.parts 遍历 + 每 part 独立优化器，返回 list[Optimizer]（canonical，nemo_automodel 惯例；04 OptimizerState 同步接受 list） | 全部 | 0.5 | P0 | 03 | 未实现 |
| 66 | 调度器 | 学习率调度器 | warmup + cosine decay，与 checkpoint 兼容（ratio-based wrapper） | 全部 | 1 | P0 | 03 | 未实现 |
| 67 | 损失函数 | 损失函数构建 | 标准 CE + FusedLinearCrossEntropy + loss_parallel（vocab 并行 CE），自动选择 hidden_states/logits 路径 | 全部 | 1 | P0 | 03 / 05 §3.5 lm_head | 未实现 |

### S8 训练循环（小计 4.5 人·日）

| # | 模块 | 功能/需求点 | 需求描述 | 适用模型 | 工时 | 优先级 | 设计文档 | 状态 |
|--:|------|------------|---------|---------|:----:|:------:|---------|:----:|
| 68 | 流程编排 | 有状态组件注册 | model/optimizer/scheduler/dataloader/rng 自动注册到状态追踪 | 全部 | 0.5 | P0 | 03 | 未实现 |
| 69 | 流程编排 | 组件构建编排 | setup() 按依赖顺序构建全部训练组件（typed .build / untyped .instantiate） | 全部 | 1 | P0 | 03 | 未实现 |
| 70 | 流程编排 | 训练步进管理 | step/epoch 推进、验证/checkpoint/日志触发、梯度累积分组、SIGTERM 优雅退出 | 全部 | 1 | P0 | 03 | 未实现 |
| 71 | 流程编排 | 训练循环执行 | epoch 迭代 → 训练步循环 → 定期验证 → 定期 checkpoint → GC → 安全退出 | 全部 | 0.5 | P0 | 03 | 未实现 |
| 72 | 训练步 | 训练步执行(三阶段) | 统计全局 token 数 → 梯度累积循环（前后向）→ 梯度裁剪与参数更新 + scheduler 步进 | 全部 | 0.5 | P0 | 03 | 未实现 |
| 73 | 训练步 | 前后向执行 | batch 上卡 → CP 上下文（shard_batch_for_cp）→ forward（PrecompiledBoundary）→ loss → backward → FSDP2 梯度同步（DTENSOR_UNIFIED / tp_grad_info 旁路，D-12 二选一） | 全部 | 0.5 | P0 | 03 / 05 §6.7 | 未实现 |
| 74 | 训练步 | 验证评估 | 定期验证集前向 + 指标汇总（各 rank 归约） | 全部 | 0.5 | P1 | 03 | 未实现 |

### S9 持久化（小计 6.5 人·日）

| # | 模块 | 功能/需求点 | 需求描述 | 适用模型 | 工时 | 优先级 | 设计文档 | 状态 |
|--:|------|------------|---------|---------|:----:|:------:|---------|:----:|
| 75 | 检查点 | 检查点配置与管理器 | 保存路径/间隔/HF 导出开关/异步开关 + StorageWriter/Reader + Addons 注册 | 全部 | 1 | P0 | 04 | 未实现 |
| 76 | 检查点 | 模型权重保存 | 5 阶段：ModelState → Adapter.to_hf → index mapping → dcp.save → consolidate | 全部 | 1 | P0 | 04 | 未实现 |
| 77 | 检查点 | 模型权重恢复 | 3 路径：MoE tensor merging / Safetensors fast path / DCP resume + key_mapping | 全部 | 1 | P0 | 04 | 未实现 |
| 78 | 检查点 | MoE stacked 参数 key 映射 | D-09 堆叠后 experts.{proj} ↔ HF per-expert experts.{i}.{proj}.weight：HF 导出时 unstack 拆回（arch 注册映射表）、init 加载零转换（先加载后 apply）、DCP resume 直存 stacked key | DeepSeek V3/V4 / Qwen3.5 / GLM-Image | 1 | P0 | 04 §7.6 / 05 §6.4.7 | 未实现 |
| 79 | 检查点 | HF 格式导出/加载 | 单一 path 参数，内部解析 root_dir+model_name，本地缓存 fallback | 全部 | 0.5 | P1 | 04 | 未实现 |
| 80 | 状态管理 | 模型/优化器状态管理 | DCP 兼容的 ModelState/OptimizerState（接受 list[Optimizer]），tied weights + PEFT 处理 | 全部 | 0.5 | P0 | 04 | 未实现 |
| 81 | 状态管理 | 分布式元数据记录 | 记录 DTensor placement 信息，用于调试审计和 ShardingPlan diff | 全部 | 0.5 | P1 | 04 | 未实现 |
| 82 | 状态管理 | 异步保存与故障恢复 | dcp.async_save 后台写入 + DistributedSignalHandler + LATEST symlink 原子更新 + 断点续训 | 全部 | 1 | P0 | 04 | 未实现 |

### S10 高级并行（小计 6.0 人·日）

| # | 模块 | 功能/需求点 | 需求描述 | 适用模型 | 工时 | 优先级 | 设计文档 | 状态 |
|--:|------|------------|---------|---------|:----:|:------:|---------|:----:|
| 83 | 流水线并行 | PP 层切分 | transformer 层按 PP size 均分到不同 GPU，管理跨 stage 激活传递 | 全部 | 1.5 | P1 | 06 | 未实现 |
| 84 | 流水线并行 | PP 调度策略 | GPipe（同步）/ 1F1B（异步交错）两种调度 + causal mask 预计算 | 全部 | 1.5 | P1 | 06 | 未实现 |
| 85 | 显存优化 | 激活检查点 | checkpoint_wrapper 每层包裹，full/selective 开关，以计算换显存 | 全部 | 1 | P0 | 06 | 未实现 |
| 86 | 混合精度 | 混合精度策略 | param_dtype/reduce_dtype/output_dtype 配置（bf16/fp16） | 全部 | 0.5 | P0 | 06 | 未实现 |
| 87 | 混合精度 | 特定模块精度隔离 | 指定模块（norm/router 等）固定 fp32，不随全局 dtype 转换 | 全部（DeepSeek V3/V4 router 敏感） | 0.5 | P1 | 06 | 未实现 |
| 88 | 验证 | 高级并行 E2E 测试 | PP=2/4 loss 一致 + AC 显存降低 >30% | 全部 | 1 | P1 | — | 未实现 |

### S11 模型交付（小计 23.0 人·日）

| # | 模块 | 功能/需求点 | 需求描述 | 适用模型 | 工时 | 优先级 | 设计文档 | 状态 |
|--:|------|------------|---------|---------|:----:|:------:|---------|:----:|
| 89 | DeepSeek V3/V4 | 架构接入与注册 | 基于 HuggingFace 接入：架构注册 + HFCheckpointingMixin + from_pretrained 全链路 | DeepSeek V3/V4 | 1 | P0 | 01 §12 | 未实现 |
| 90 | DeepSeek V3/V4 | MLA 注意力分片 | planner 级已实现：_DEEPSEEK_MLA_OVERRIDES 内置 ARCH_OVERRIDES（D-13：q_a/kv_a 下投影 → REPLICATED（latent 维不切），q_b/kv_b 上投影 → COLWISE（head 维），o_proj 仍 ROWWISE；architectures 与 model_type 双拼写注册，v2/v3 同构）；无端到端训练验证 | DeepSeek V3/V4 | 2 | P0 | 05 §3.6.1/§12.2 D-13 | 部分实现 |
| 91 | DeepSeek V3/V4 | 细粒度 expert EP 分片 | EP 直通已实现：D-09/D-10 覆盖细粒度 routed experts（deepseekv3 sigmoid-group router adapter + 派生 expert mesh (edp,ep) + region 内 a2a），shared expert 走 SHARED_EXPERT 角色；DeepEP/UCCL-EP 后端 token dispatcher 未实现 | DeepSeek V3/V4 | 2 | P0 | 05 §6.4.7/§6.4.8 | 部分实现 |
| 92 | DeepSeek V3/V4 | MTP 模块适配 | Multi-Token Prediction 模块的分片与损失链路接入 | DeepSeek V3/V4 | 1 | P1 | — | 未实现 |
| 93 | DeepSeek V3/V4 | 权重映射与 E2E 测试 | V3/V4 权重键名映射 + TP/EP 组合 100 步训练 + 输出与 HF 参考容差 1e-5 | DeepSeek V3/V4 | 1.5 | P0 | 04 §5.4 | 未实现 |
| 94 | Qwen3.5 | Dense 架构实现 | GatedDeltaNet 层 + MTP 特殊逻辑 + 架构注册 | Qwen3.5 | 1.5 | P0 | 01 §12 | 未实现 |
| 95 | Qwen3.5 | 架构覆盖规则 | 已实现：GatedDeltaNet SPECIAL 角色 + gated_delta_tp_shard SpecialHandler（按 SSM head 切分，含 a_log/dt_bias 模式映射），hyper_models/components/distributed 交付并测绿；模型级接入属上行条目 | Qwen3.5 | 1 | P0 | 05 §6.4.6 | 已实现 |
| 96 | Qwen3.5 | MoE 架构实现 | MoE expert 合并 + EP 分片已实现（D-09/D-10：qwen3moe router adapter + 堆叠/batched 直通 + 派生 expert mesh）；FSDP2 下 expert 梯度同步待 D-12 集成接线 | Qwen3.5 | 2 | P0 | 05 §6.4 | 部分实现 |
| 97 | Qwen3.5 | E2E 测试 | Dense/MoE 两变体 TP=2/4 100 步 + 输出容差 1e-5 | Qwen3.5 | 1 | P0 | — | 未实现 |
| 98 | GLM-Image | VLM 架构接入 | 基于 HuggingFace：vision encoder + LLM 的 ForImageTextToText 接入与架构注册 | GLM-Image | 1.5 | P0 | 01 §6.1 | 未实现 |
| 99 | GLM-Image | mRoPE 位置编码适配 | 多模态旋转位置编码的 reshape 边界处理（R7：Shard(N) 无法表达逻辑轴——用户显式声明 reshape 后模块的 in_src） | GLM-Image | 1 | P0 | 05 §3.6.5 局限性 | 未实现 |
| 100 | GLM-Image | 权重层级映射 | VLM 视觉塔/投影层/语言模型的权重键名层级映射 | GLM-Image | 1 | P0 | 04 §5.4 | 未实现 |
| 101 | GLM-Image | E2E 测试 | 图文混合 batch TP=2 训练 100 步 + 与 HF 参考一致 | GLM-Image | 1 | P0 | — | 未实现 |
| 102 | Pangu(自研) | 自研模型类实现 | 走内置自研路径（非 HF AutoModel）：模型结构实现 + 内置模型注册表接入 | Pangu(自研) | 2.5 | P0 | 01 §5/§7 | 未实现 |
| 103 | Pangu(自研) | 权重与分词器适配 | Pangu 私有权重格式与词表的加载适配（键名映射 + safetensors 加载路径） | Pangu(自研) | 1 | P0 | 04 | 未实现 |
| 104 | Pangu(自研) | 分片规则注册 | Pangu 命名的 ARCH_OVERRIDES + 特殊参数 SpecialHandler 注册 | Pangu(自研) | 1 | P0 | 05 §8.3 | 未实现 |
| 105 | Pangu(自研) | E2E 测试 | Pangu TP=2/4 训练 100 步 + 与参考实现对拍 | Pangu(自研) | 1 | P0 | — | 未实现 |

### S12 CLI与监控（小计 3.0 人·日）

| # | 模块 | 功能/需求点 | 需求描述 | 适用模型 | 工时 | 优先级 | 设计文档 | 状态 |
|--:|------|------------|---------|---------|:----:|:------:|---------|:----:|
| 106 | 命令行 | 命令行入口 | hyper-parallel config.yaml --train.max_steps 500 风格 CLI | 全部 | 1 | P1 | — | 未实现 |
| 107 | 日志 | 控制台与文件日志 | Rank 0 控制台输出 + JSONL 文件记录 | 全部 | 0.5 | P1 | — | 未实现 |
| 108 | 日志 | 远程日志上报 | WandB + MLflow 对接 | 全部 | 0.5 | P2 | — | 未实现 |
| 109 | 指标 | 训练指标计算 | loss / grad_norm / lr / tokens/sec / MFU | 全部 | 0.5 | P1 | — | 未实现 |
| 110 | 模板 | 配置模板 | 4 套 YAML 参考配置（DeepSeek MoE / Qwen3.5 / GLM-Image VLM / Pangu 多节点） | 全部 | 0.5 | P1 | — | 未实现 |

### S13 质量保障（小计 6.5 人·日）

| # | 模块 | 功能/需求点 | 需求描述 | 适用模型 | 工时 | 优先级 | 设计文档 | 状态 |
|--:|------|------------|---------|---------|:----:|:------:|---------|:----:|
| 111 | 等价性验证 | 零拷贝正确性验证 | 已实现：DTensor._local_tensor 修改与全局视图同步（data_ptr 共享断言），tests/components/distributed 覆盖 | 全部 | 0.5 | P0 | 05 §4.4 | 已实现 |
| 112 | 等价性验证 | 校验/生产模式等价性 | 两模式输出/梯度等价用例已交付（test_dist_s5_mode_equiv / test_dist_s5_grad_equiv）；FSDP2 梯度同步集成测试待 D-12 接线后补（当前 grad_equiv.py 模拟旁路） | 全部 | 1 | P0 | 05 §5.5 | 部分实现 |
| 113 | 等价性验证 | 分片通信组合覆盖 | 已实现：所有 in_src→in_dst 组合的 PrecompiledBoundary 正确性（identity/all_gather/reduce_scatter/all_reduce/redistribute 5 种 collective × 2 模式） | 全部 | 0.5 | P0 | 05 §4.3 | 已实现 |
| 114 | 模型兼容 | 交付模型兼容性测试 | DeepSeek V3/V4、Qwen3.5、GLM-Image、Pangu from_pretrained 推理与参考一致 | DeepSeek V3/V4 / Qwen3.5 / GLM-Image / Pangu(自研) | 1 | P0 | — | 未实现 |
| 115 | 持久化验证 | Checkpoint 往返测试 | save→load 一致 + 跨 TP 配置加载 + 断点续训 loss 连续 | 全部 | 0.5 | P0 | 04 | 未实现 |
| 116 | 端到端 | 端到端训练验收 | 交付模型 8 GPU 1000 步 loss 正常下降 + 断点续训 | 全部 | 1 | P0 | — | 未实现 |
| 117 | 文档 | 迁移指南 | 旧配置 → 新配置映射 + API 变化对照 | 全部 | 0.5 | P2 | — | 未实现 |
| 118 | 文档 | 快速入门与自定义模型指南 | 5 步跑通第一个训练 + 新模型添加 ShardingTemplate/ARCH_OVERRIDES 指南 | 全部 | 1 | P1 | — | 未实现 |
| 119 | 文档 | 配置参考手册与设计文档校对 | YAML 字段完整说明 + 设计文档最终更新（含实施校准回写） | 全部 | 0.5 | P1 | — | 未实现 |

## 模型交付

| 交付模型 | 接入方式 | 架构特点 | 专项适配需求 | 工时小计 |
|---------|---------|---------|-------------|:-------:|
| DeepSeek V3/V4 | HuggingFace | MoE（细粒度 expert + shared expert）+ MLA 注意力 + MTP | MLA 内置 ARCH_OVERRIDES（D-13，planner 级已实现）；EP 直通（D-09/D-10 已实现）；router 精度隔离；V3/V4 权重键名映射与 E2E 验证 | 7.5 |
| Qwen3.5 | HuggingFace | Dense（GatedDeltaNet + MTP）与 MoE 两变体 | GatedDeltaNet SPECIAL 分片（已实现）；MoE EP 直通（已实现）；模型注册/FSDP2 集成与两变体 E2E | 5.5 |
| GLM-Image | HuggingFace | VLM（vision encoder + LLM），ForImageTextToText | mRoPE reshape 边界显式声明；NEAT 图像打包；VLM 权重层级映射 | 4.5 |
| Pangu | 自研（内置路径，非 HF） | 自研架构 | 自研模型类实现与内置注册；私有权重/词表适配；ARCH_OVERRIDES 分片规则注册 | 5.5 |

## 汇总

| 流程阶段 | 需求点数 | 工时小计（人·日） | 已实现 | 部分实现 | 未实现 |
|---------|:-------:|:----------------:|:-----:|:-------:|:-----:|
| S0 前置依赖 | 5 | 5.0 | 1 | 1 | 3 |
| S1 启动与配置 | 7 | 5.0 | 0 | 0 | 7 |
| S2 分布式环境 | 8 | 5.5 | 1 | 1 | 6 |
| S3 数据供给 | 8 | 5.0 | 0 | 1 | 7 |
| S4 模型构建 | 9 | 7.5 | 0 | 0 | 9 |
| S5 并行分片 | 21 | 25.5 | 20 | 1 | 0 |
| S6 权重加载 | 5 | 3.5 | 0 | 0 | 5 |
| S7 训练组件 | 4 | 3.5 | 0 | 0 | 4 |
| S8 训练循环 | 7 | 4.5 | 0 | 0 | 7 |
| S9 持久化 | 8 | 6.5 | 0 | 0 | 8 |
| S10 高级并行 | 6 | 6.0 | 0 | 0 | 6 |
| S11 模型交付 | 17 | 23.0 | 1 | 3 | 13 |
| S12 CLI与监控 | 5 | 3.0 | 0 | 0 | 5 |
| S13 质量保障 | 9 | 6.5 | 2 | 1 | 6 |
| **合计** | **119** | **110.0** | **25** | **8** | **86** |
