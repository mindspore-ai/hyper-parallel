# Inline Codegen 泛化整改状态

日期：2026-09-16。工作分支：`codegen_master_v3`。

## Phase 1 实现

- `ModelAdapterSpec.inline_codegen` 提供 lazy provider；Qwen3-MoE 注册该能力。
- replacement、strategy 和参数重命名声明迁至 `models/qwen3_moe/adapter/inline.py`。
- `specs.py` 通过模型注册机制发现声明；核心不再维护 Qwen3 target 注册表。
- `pipeline.py` 删除模型策略白名单；同时检查 local compute 和 inner wrapper，未覆盖的规则回退通用生成流程。
- `strategy_pass.py` 使用声明中的目标类、方法及方法体；相同方法的冲突声明报错。
- `meta_plan.py` 按 adapter 声明展开参数名；源码 patch 成功后才修改 metadata。
- emitter 传递模型 architecture；方法提取支持声明中的非 `forward` 方法名。
- inline 模式的产物签名包含 adapter 声明，避免只修改 adapter 时继续使用缓存产物。

旧调用没有传入模型 identity 时，从现有 `hyper_parallel.models.<family>.adapter.*` target 路径发现 adapter。
这保留了已有 Qwen3 调用的行为，同时避免在核心加入 Qwen3 默认值。
显式传入但不支持的模型不会回退到其他模型的声明。

## 本地验证

使用 Python 3.11 专用虚拟环境，已将本工作树 editable 安装并确认 import 路径。

| 检查 | 结果 |
| --- | --- |
| `python -m pytest tests/codegen -q -p no:cacheprovider` | 108 passed，6 个已有 pytest marker 警告 |
| 新增 adapter 回归 | 11 项，包含新模型生成代码执行、兼容回退、QKV 元数据、声明冲突、自定义方法和缓存失效 |
| 真实 Transformers 5.14.1 Qwen3-MoE 源码生成对比 | 与整改前 HEAD 的 pipeline 输出逐字节一致，参数元数据相同 |
| 生成源码编译 | 通过 |
| inline 核心、Qwen3 inline adapter 和新增测试 pylint | 通过 |
| `check_agents_catalog.py` | 通过 |
| `git diff --check` | 通过 |

真实源码生成对比覆盖 RMSNorm、attention、grouped experts 替换及 CP/EP 策略。
生成源码共 36,743 字节，SHA256 为
`ac292311497beef2a5952ba232b207a212ea447c82fad4fe16c5b2f3bcf75004`。
此项是源码及 metadata 兼容性检查，不替代多卡数值验证。

整文件 pylint 仍有旧代码告警，包括 emitter / manager 的局部 import、未使用参数及缺失类型注解；
项目自定义版权检查还会将已有的 `Copyright 2025-2026` 误报为缺少 header，因为检查器写死了 `Copyright 2026`。
保留了原版权年份，没有添加 suppression。

## 待验证和后续范围

- 129 测试服务器 SSH 连接超时，尚未执行本轮 NPU 原生/生成模型 loss 对比。
  按原方案需重新生成产物，验证 `1111`、`1121`、`1211`、`2221` 拓扑。
- Phase 2 的共享 GQA/EP pattern 按原方案在第二个模型正式接入时提取；当前继续复用既有模板，保证产物不变。
- runtime 的 external-state 类白名单按原方案不在本轮范围内。
- DeepSeek adapter 和示例草稿尚未注册为本轮支持的 inline 模型；本轮通过独立 toy adapter 验证通用接入流程。
