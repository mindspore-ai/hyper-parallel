# Hyper-Parallel 重构特性 — 文档工作区

> 本目录收纳 hyper-parallel 重构特性的全部工作文档（设计文档除外——
> 设计文档位于 [`../detailed_design/`](../detailed_design/)，为全仓引用最多的路径，原地保留）。
> 仓库通用文档（index/installation/faq/guide/api/contributing/images）在 `docs/` 根，与本特性无关。

## 目录结构

```
docs/
├── detailed_design/                  # 【设计文档 canonical】01–07 + 重构总计划
│   ├── 01_hf_compatibility_layer.md      # HF 兼容层：强类型配置解析（TrainerConfig/Configurable）/registry/from_pretrained
│   ├── 02_data_pipeline.md               # 数据管道：build_dataloader/packing/sampler/collater
│   ├── 03_training_loop.md               # 训练循环：BaseRecipe/FinetuneRecipe/StepScheduler
│   ├── 04_checkpoint.md                  # Checkpoint：Checkpointer/DCP/HF 导出/断点续训
│   ├── 05_dual_mode_dtensor_parallel_strategy.md  # 双模式 DTensor 并行策略（✅ 已实现，282 用例）
│   ├── 06_distributed_infrastructure.md  # 分布式基础设施：MeshContext/FSDP2Manager
│   ├── 07_model_script_generation.md     # 模型脚本生成器
│   └── hyper_parallel_refactor_plan.md   # 重构总计划（需求分解/里程碑/人日）
│
└── refactor/                       # 本特性工作区（本目录）
    ├── README.md                   # 本索引
    │
    ├── requirements.{md,csv,xlsx}  # 需求表（119 条，含实现状态标注）
    ├── gen_requirements.py         # 需求表单一数据源生成器（改 ROWS 后重跑，三件套同步更新）
    │
    ├── architecture_overview.md    # 代码目录结构 + 分层架构说明
    ├── architecture_overview.png   # 整体架构图
    ├── gen_architecture_diagram.py # 架构图生成脚本
    │
    ├── reviews/                    # 设计评审报告
    │   ├── design_review_20260717.md   # 第九轮复核（历史）
    │   └── design_review_20260721.md   # 全量检视 + 修复状态 + 最终评价（最新）
    │
    ├── plans/                      # 计划与分析
    │   ├── parallel_plan.md                # 并行开发方案（stub 驱动 + mock）
    │   ├── dev_plan_05_dual_mode_dtensor.md # 05 实现计划（一步一 UT 版，已执行完毕）
    │   └── hyper_parallel_training_pipeline_gap_analysis.md  # 前期差距分析（Titan/VeOmni/AutoModel 对比）
    │
    ├── guides/                     # 使用与走读
    │   ├── components_distributed_tutorial.md        # hyper_models/components/distributed 使用教程
    │   ├── components_distributed_code_walkthrough.md # 代码详解说明书（按调用时序）
    │   └── hyper_parallel_端到端训练流程详解.md        # 端到端训练流程（现状框架）
    │
    └── archive/                    # 历史版本（仅存档，勿引用）
        ├── dual_mode_dtensor_parallel_strategy.md   # 05 的旧单文件版（2026-07-15，已被 detailed_design/05 取代）
        ├── sharding_config_implementation.md        # ShardingConfig 早期实现稿（2026-07-14）
        └── detailed_design.tar.gz                   # 设计文档旧快照（2026-07-16）
```

## 常用入口

| 我要… | 去看 |
|-------|------|
| 了解整体架构与代码布局 | [architecture_overview.md](architecture_overview.md) + [架构图](architecture_overview.png) |
| 查需求与实现状态 | [requirements.xlsx](requirements.xlsx)（绿=已实现 / 黄=部分 / 蓝=未实现） |
| 按设计写代码 | [`../detailed_design/`](../detailed_design/) 对应模块文档 |
| 了解设计评审结论与遗留缺口 | [reviews/design_review_20260721.md](reviews/design_review_20260721.md) |
| 上手 hyper_models/components/distributed | [guides/components_distributed_tutorial.md](guides/components_distributed_tutorial.md) |
| 排期与里程碑 | [`../detailed_design/hyper_parallel_refactor_plan.md`](../detailed_design/hyper_parallel_refactor_plan.md) |

## 维护约定

1. **代码语义变化 → 同步续记 05 §12.2 的 D-xx 修订编号**；
2. 需求变化 → 改 `gen_requirements.py` 的 ROWS 并重跑（xlsx/md/csv 三件套同步）；
3. 架构变化 → 改 `gen_architecture_diagram.py` 并重跑；
4. 被取代的过程稿移入 `archive/`，不删除；
5. 新设计文档按编号加入 `../detailed_design/` 并更新本索引。
