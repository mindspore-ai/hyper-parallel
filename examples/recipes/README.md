# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
# 场景配方库：按模型 arch / 并行拓扑直接可引用的 YAML 起点。
#
# 用法：
#   1. 找与你模型+拓扑最近的配方，整段拷入你的训练 YAML；
#   2. 按需调整 match 模式（"*.self_attn" / "*.mlp" 覆盖 HF 命名惯例，
#      自研命名改成你的 FQN glob）；
#   3. 用 plan 内省报告核对实际切分是否符合预期：
#        planner = ShardingPlanner(plan_overrides=...)
#        plan = planner.plan(model, mesh, tp_size=..., explain=True)
#      或 plan 后独立调用 print(plan.explain())；
#   4. validate 模式跑通数值对拍后再上生产。
#
# 配方清单：
#   llama_tp.yaml        稠密 LLM（Llama/Qwen-dense 系）纯 TP(+SP)
#   llama_tp_cp.yaml     稠密 LLM TP + CP（长序列，attention 注入 CP wrapper）
#   qwen3moe_tp_ep.yaml  HF 原生 MoE（Qwen3-MoE 系）TP + EP（注入 EP compute）
#   custom_ep_moe.yaml   自研 EP-aware MoE（forward 内含 a2a）——不注入 fn，
#                        只声明 region_dispatch: false
#
# 通用规则（与教程 §10 一致）：
#   - 纯 TP/SP：无需任何 plan_overrides —— 模板自动推导全部切分与通信；
#   - CP/EP：注入是显式的（框架零自动注入），且注入必须伴生
#     region_dispatch 声明（无默认）。判断口诀：注入物含通信原语/自定义
#     kernel/数据依赖分支 → false（黑盒托管）；纯 aten 标准算子 → 可 true
#     （validate 穿透 + out_src 真校验）；
#   - when 字段让一份配置跨拓扑复用：when: cp / when: ep 的条目在对应轴
#     size=1 时自动跳过（INFO 日志可见），单卡调试不用改配置。
