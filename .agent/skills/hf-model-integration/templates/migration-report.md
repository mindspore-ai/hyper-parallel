# <模型名> 迁移报告

> 迁移完成后的交付物。每个数字附命令+commit+环境；未实测标 `[待测]`，
> 禁填估值。精度口径见 `rules/precision-acceptance.md`。

| 字段 | 内容 |
|---|---|
| 模型 | [FILL]（来源 HF 仓 / 版本）|
| 目标分支 / PR | [FILL] |
| 负责人 | [FILL] |
| 状态 | 迁移中 / 待验收 / 已验收 |

## 1. 迁移范围

- **模型结构**：[FILL]（层数、注意力/MLP 形态、特殊模块）。
- **复用 vs 新实现**：[FILL]（哪些直接用框架既有高性能模块、哪些新写，各自原因）。
- **不迁移的部分**：[FILL] 及原因。

## 2. 适配项清单

| 适配项 | 做法 | 状态 |
|---|---|---|
| 延迟初始化（lazy init / meta） | [FILL] | [FILL] |
| 高性能模块替换 | [FILL] | [FILL] |
| TP 适配 | [FILL] | [FILL] |
| CP 适配 | [FILL] | [FILL] |
| EP 适配 | [FILL] | [FILL] |
| checkpoint 转换 | [FILL]（权重名映射、分片规则）| [FILL] |
| 数据/输入契约 | [FILL] | [FILL] |

## 3. 产出物

| 产出 | 路径 | 说明 |
|---|---|---|
| 模型适配器 | [FILL] | |
| ckpt 转换脚本 | [FILL] | HF 权重 → 框架格式 |
| 训练 YAML | [FILL] | recipe |
| 启动脚本 | [FILL] | 单机 / 多机 |
| 测试脚本 | [FILL] | UT / ST |

## 4. 验证

### 4.1 功能

| 项目 | 命令 | commit | 环境 | 结果 |
|---|---|---|---|---|
| UT | [FILL] | [FILL] | [FILL] | [FILL] |
| 单卡前反向 | [FILL] | [FILL] | [FILL] | [FILL] |
| 多卡（TP/CP/EP）| [FILL] | [FILL] | [FILL] | [FILL] |

### 4.2 精度

> 口径见 `rules/precision-acceptance.md`（1000 step）；基线必须写明。

- **基线**：[FILL]。
- 平均绝对误差 `[待测]`（≤0.02）/ 平均相对误差 `[待测]`（≤2%）。
- grad_norm [待测] · 首 step [待测] · 确定性 [待测] · 断点续训 [待测]。

### 4.3 性能

- 拆解（计算/通信/Free/优化器）：[待测]（见 `profiling` skill）。
- 显存峰值：[待测]（见 `memory-analysis` skill）。

## 5. 使用限制与已知问题

- **支持范围**：[FILL]。
- **已知不支持 / 待办**：[FILL]。

## 6. 变更记录

| 日期 | 作者 | 变更 |
|---|---|---|
| [FILL] | [FILL] | 初稿 |
