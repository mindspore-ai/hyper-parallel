# <模型 / 特性> 精度分析报告

> 每个数字附命令+commit+环境；基线必须显式；未测标 `[待测]`，需内部对齐
> 逻辑的标 `[待对齐]`。验收口径见 `rules/precision-acceptance.md`。

| 字段 | 内容 |
|---|---|
| 模型 / 特性 | [FILL] |
| 被测轴 | [FILL]（候选 vs 基线差在哪一个轴）|
| 候选 | [FILL]（实现/设备 + commit + 配置 + 环境）|
| 基线 | [FILL]（实现/设备 + commit + 配置 + 环境）|
| 节点 | [FILL]（A/B 同节点背靠背）|

## 1. 结论

- **是否通过验收**：[FILL]（按 precision-acceptance：主判据 abs≤0.02 或
  rel≤2%，四项必查）。
- **首个异常**：[FILL]（module / op / 行 + 执行序）或「无超阈项」。

## 2. 采集

- **工具 / 配置**：msprobe，level=[FILL]，step=[FILL]。
- **命令**：`python .agent/skills/precision/scripts/collect_dump.py ...`
  （候选与基线各一次）。
- **dump 路径**：候选 [FILL] / 基线 [FILL]。
- `[待对齐]`：内部打点策略差异（如有）。

## 3. 比对

- **命令**：`python .agent/skills/precision/scripts/diff_dump.py
  --candidate <d1> --baseline <d2>`。
- **首个超阈项**：[FILL]（名称 / 执行序 / cosine / 最大相对误差）。
- **阈值口径**：cosine ≥ [FILL]、max_rel ≤ [FILL]；bf16 并列差 < 1 ulp
  记为噪声（并列数 [FILL]、worst gap [FILL]）。

## 4. 验收判据（precision-acceptance）

| 项 | 结果 | 阈 |
|---|---|---|
| 平均绝对误差 | [待测] | ≤ 0.02 |
| 平均相对误差 | [待测] | ≤ 2% |
| grad_norm 轨迹 | [待测] | 同误差带 |
| 首 step | [待测] | 紧匹配 |
| 确定性（同种子两跑） | [待测] | 同轨 |
| 断点续训 | [待测] | 连续 |

## 5. 下一步修复意见

- **根因判断**：[FILL]（基于首个异常，非末端症状）。
- **修复方向**：[FILL]。
- **回归验证方式**：[FILL]（同输入复跑，确认首个异常消失且下游恢复）。

## 6. 证据清单

- 命令 / commit / 环境：[FILL]。
- 比对表 / dump / 日志路径：[FILL]。
