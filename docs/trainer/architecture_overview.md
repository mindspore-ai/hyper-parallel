# Hyper-Parallel 代码目录结构与整体架构

> 依据：`docs/detailed_design/` 01–06 + 总计划（2026-07-22 定稿口径）
> 图例：✅ 已实现（tests/components/distributed/ 282 用例全绿）｜🟡 部分实现｜📋 设计定稿、待实现
> 架构图见 [architecture_overview.png](architecture_overview.png)（由 `gen_architecture_diagram.py` 生成）

---

## 1. 目标代码目录结构

```
hyper_parallel/
│
├── recipes/                                # ── 03 训练循环（📋）──
│   ├── base_recipe.py                      #   BaseRecipe：__state_tracked 自动状态追踪 + checkpoint 保存/恢复
│   ├── llm/
│   │   └── train_ft.py                     #   FinetuneRecipe：LLM 训练 recipe（setup 编排 + 训练主循环）
│   └── vlm/
│       └── finetune.py                     #   FinetuneRecipeForVLM：VLM 训练 recipe
│
├── _transformers/                          # ── 01 HF 兼容层（📋）──
│   ├── auto_model.py                       #   HyperAutoModel* 类族 + from_pretrained/from_config 入口
│   ├── registry.py                         #   MODEL_ARCH_MAPPING 注册中心 + 懒加载
│   ├── model_init.py                       #   HF config 获取 + 模型实例化 + 自定义/HF 路径分发
│   └── infrastructure.py                   #   instantiate_infrastructure + apply_model_infrastructure 两阶段
│                                           #   （① PP → ② PEFT → … → ⑤ ShardingPlan → ⑥ apply → ⑨ FSDP2）
│
├── core/                                   # ── 既有核心（✅ 在役）──
│   ├── dtensor/                        ✅  #   自研 DTensor（前向-only）；DeviceMesh.concatenate 已实现（device_mesh.py:1035）
│   └── fully_shard/                    ✅  #   FSDP2 实现：DTENSOR_UNIFIED 模式 / _build_layout_driven_group_info / all_reduce_grad
│
└── platform/
    └── torch/
        └── fully_shard/                ✅  #   torch 适配层：_orig_param_is_dtensor / _orig_dtensor_placements（param.py:149-151）

hyper_models/
└── components/
│   ├── config/                             # ── 01 配置系统（📋）──
│   │   ├── node.py                         #   ConfigNode / _resolve_target（canonical 位置）
│   │   ├── loader.py                     #   load_yaml_config（仅 YAML 加载）
│   │   └── _utils.py                     #   辅助 helpers
│   │
│   ├── datasets/                           # ── 02 数据管道（📋）──
│   │   ├── llm/
│   │   │   ├── dataloader.py               #   build_dataloader() 统一入口（components 层，跨 recipe 复用）
│   │   │   ├── packed_sequence.py          #   THD packing
│   │   │   ├── neat_packing.py             #   NEAT packing（VLM）
│   │   │   ├── megatron_dataset.py         #   Megatron .bin/.idx 数据集封装
│   │   │   └── megatron/
│   │   │       └── sampler.py              #   MegatronPretraining(Random)Sampler
│   │   ├── vlm/
│   │   │   ├── datasets.py                 #   VLM 数据集工厂
│   │   │   └── neat_packing_vlm.py         #   VLM NEAT packing
│   │   └── utils.py                        #   collate 集合（default_collater / packed_sequence_thd_collater / neat_packed_collater）
│   │
│   ├── models/                             # ── 01 模型公共层（📋）──
│   │   └── common/
│   │       ├── state_dict_adapter.py       #   StateDictAdapter 基类（HF key ↔ 内部 FQN）
│   │       ├── hf_checkpointing_mixin.py   #   HFCheckpointingMixin（模型持有 _state_dict_adapter）
│   │       ├── param_utils.py              #   参数工具
│   │       └── packing.py                  #   configure_packing / get_attn_implementation（02 §3.4 契约）
│   │
│   ├── distributed/                        # ── 05 并行策略 + 06 基础设施 ──
│   │   │                                 #   【05 已实现部分 ✅，282 用例全绿】
│   │   ├── sharding_config.py          ✅  #   ShardingPlan / ModuleShardingSpec / ShardingTemplate / TEMPLATES / MeshAxisName（canonical）
│   │   ├── sharding_planner.py         ✅  #   ShardingPlanner：参数角色分类(ParamRole×14) → 边界推断 → 模板匹配 → 链式传播
│   │   │                                 #     ARCH_OVERRIDES（含 _DEEPSEEK_MLA_OVERRIDES，D-13）/ validate_model_compatibility
│   │   ├── sharding_applier.py         ✅  #   apply_sharding_plan：Phase A 分片（D-10 expert/dense 分流）→ Phase B handler → Phase C 包装
│   │   │                                 #     _build_expert_mesh（派生 (edp,ep) expert mesh）/ _wrap_cp_inner_attention / D-02 vocab embed wrapper
│   │   ├── sharding/
│   │   │   └── apply.py                ✅  #   _local_params_context / _set_param_by_path（canonical；生产模式零拷贝）
│   │   ├── precompiled_boundary.py     ✅  #   PrecompiledBoundary / RedistOp：模块边界通信预编译
│   │   ├── cp_utils.py                 ✅  #   shard_batch_for_cp（contiguous chunk + seq_lens 契约）/ flex_cp_allgather（all-gather K/V，ring 已否决 D-01''）
│   │   ├── ep_utils.py                 ✅  #   _ep_all_to_all 后端分派（NCCL/HCCL 不等长 vs gloo pad-to-max）/ MOE_ROUTER_ADAPTERS（8 键 3 实现）
│   │   │                                 #     _hf_native_ep_compute（TP-extend-EP 前向，D-09/D-10）
│   │   ├── local_region.py             ✅  #   local region：DTensor ↔ local tensor 区间（D-03'）
│   │   ├── tp_grad.py                  ✅  #   build_tp_grad_info（已产出；fully_shard 消费端待接线，D-12）
│   │   ├── param_role.py             ✅  #   ParamRole ×14（含 REPLICATED，D-14）/ ParameterClassifier
│   │   ├── testing/
│   │   │   └── grad_equiv.py           ✅  #   run_one_step / assert_grad_equivalence / simulate_tp_replicate_grad_sync
│   │   │                                 #   【06 待实现部分 📋】
│   │   ├── config.py                 📋  #   DistributedSetup — 拓扑 + 策略配置统一容器（frozen dataclass，经 build() 构造）
│   │   ├── mesh.py                   📋  #   MeshContext / ParallelismSizes — init_device_mesh(..., rank_list=...) 唯一构建点（主 mesh 无 EP 轴）
│   │   ├── fsdp2.py                  📋  #   FSDP2Manager — DP 维 FSDP2 包裹（per-block 粒度，nn.ModuleList 定位）
│   │   ├── parallelizer.py           📋  #   fsdp2_strategy_parallelize() — FSDP2 + TP 联合分片（D-12 决策点：DTENSOR_UNIFIED vs tp_grad_info 消费端）
│   │   ├── dtensor_utils.py          📋  #   仅 re-export sharding/apply.py 定义（不另起实现）
│   │   └── utils.py                  📋  #   FirstRankPerNode（每节点首 rank 判定，02 引用）等
│   │
│   ├── checkpoint/                         # ── 04 持久化（📋）──
│   │   ├── checkpointing.py              #   Checkpointer 核心类（moe_mesh 参数保留；MoE consolidated 导出待 expert mesh 导出）
│   │   ├── config.py                     #   CheckpointingConfig（typed，RecipeConfig 直接构造）
│   │   ├── stateful_wrappers.py          #   ModelState / OptimizerState（接受 list[Optimizer]，与 03 canonical 对齐）
│   │   ├── conversion_mapping.py         #   HF key ↔ 模型 FQN 映射 + WeightConverter
│   │   ├── addons.py                     #   ConsolidatedHFAddon / PeftAddon
│   │   └── _backports/
│   │       └── hf_storage.py             #   _HuggingFaceStorageWriter / _HuggingFaceStorageReader
│   │
│   ├── training/                           # ── 03 训练组件（📋）──
│   │   ├── step_scheduler.py             #   StepScheduler（grad_acc 分组 + ckpt/val/log 节奏 + SIGTERM；StepSchedulerConfig 含 global_batch_size）
│   │   ├── rng.py                        #   StatefulRNG / ScopedRNG
│   │   ├── signal_handler.py             #   DistributedSignalHandler（CUDA tensor all_gather）
│   │   └── grad_accum.py                 #   梯度累积 + set_requires_gradient_sync 管理
│   │
│   ├── optim/                              # ── 03 优化器（📋）──
│   │   ├── optimizer.py                  #   AdamWConfig / OptimizerFromFactoryConfig → list[Optimizer]（canonical）
│   │   └── lr_scheduler.py               #   LRSchedulerConfig + OptimizerParamScheduler（自 nemo_automodel port）
│   │
│   ├── loss/                               # ── 03 损失（📋）──
│   │   ├── masked_ce.py                  #   MaskedCrossEntropy
│   │   └── utils.py                      #   calculate_loss dispatcher（FusedLinearCE / 标准 logits 两路径）
│   │
│   └── parallel/                           # ── PP 工具（📋）──
│       └── pp_utils.py                   #   prepare_for_final_backward / PP hooks（03 §7.1）
```

**测试**：`tests/components/distributed/`（282 用例 ✅ 全绿）——S0 fixtures/param_role/spec_fields、S1 plan golden/overrides/MLA(deepseek)、S3 inner-attn 检测、S4 MoE local_map/validate_region/local_compute_fn、S5 HF 原生 MoE/vocab embed/零依赖 lint、S6 ep_extend/hf_native_moe。

---

## 2. 分层架构说明

| 层 | 内容 | 关键契约 |
|----|------|---------|
| **L1 用户接口** | train.yaml + CLI 入口 | 一个 YAML 启动训练（01 §2/§3） |
| **L2 Recipe 编排** | FinetuneRecipe.setup()（④.1–④.14）→ run_train_validation_loop（⑤） | 组件构建顺序以 01 §4.1 时序图为 canonical 编号来源 |
| **L3 训练组件** | ConfigNode/RecipeConfig（01）· build_dataloader（02）· Checkpointer（04）· Optimizer/LR/Loss（03） | typed `.build()` / untyped `.instantiate()` 两路径；optimizer 为 `list[Optimizer]` |
| **L4 模型构建与并行策略** | HyperAutoModel.from_pretrained → ShardingPlanner → apply_sharding_plan → FSDP2Manager | plan(model, mesh, *, tp/cp/ep/sequence_parallel/loss_parallel) → apply → (model, tp_grad_info)；PP 最先执行 |
| **L5 分布式核心** | MeshContext（init_device_mesh+rank_list）· DTensor · FSDP2（DTENSOR_UNIFIED）· PrecompiledBoundary · cp/ep_utils | 主 mesh 无 EP 轴；expert mesh (edp,ep) apply 期派生；CP=all-gather K/V |
| **L6 运行时** | PyTorch / NCCL·HCCL | torch.distributed.fsdp（非 fsdp2 模块） |

**双模式**（05 核心设计）：同一 ShardingPlan 两种执行——**生产模式**（`_local_params_context` 零拷贝解包 + 预编译边界，性能优先）与**校验模式**（纯 DTensor `__torch_dispatch__`，stock PyTorch 可跑），等价性由 `testing/grad_equiv.py` 验证。

**关键待决点（D-12）**：TP-Replicate 参数（norm/bias/router）梯度同步两路径——(a) 调时序走已实现的 DTENSOR_UNIFIED（fully_shard 先于解包）；(b) 补 `fully_shard(tp_grad_info=...)` 消费端。实现 06 `FSDP2Manager.parallelize` 时二选一。
