# Multicore 网络集成测试

本目录验证 DeepSeek V4.1 使用独立 HyperMulticore 的模型替换、并行接线及数值一致性。
运行环境需要安装 HyperMulticore 和模型依赖，并激活 CANN 环境。
测试入口使用当前 Python 定位 HyperMulticore 的激活脚本，在启动 worker 前设置其运行环境；
脚本缺失或激活失败时停止执行。

| 范围 | 默认参数项数 | 扩展参数项数 | 级别 | 实际卡数 |
| --- | ---: | ---: | --- | --- |
| MegaMoE 单卡集成 | 1 | 1 | level0 | 1 |
| MegaMoE 混合并行 | 6 | 12 | level1 | 8 |
| MegaMoE 动态专家 | 2 | 2 | level1 | 8 |
| MegaGate 模型集成 | 4 | 4 | level1 | 1 或 2 |

单卡集成覆盖 DeepSeek V4.1 文本／视觉路由、MegaMoE 替换、共享专家的前向及梯度，
使用 17、129 两种 token 长度；单项运行预算为 120 秒，包含 worker 启动。
混合并行覆盖 TP/SP/CP/EP、文本及视觉模型、路由和共享专家的前向及梯度。
动态专家覆盖 push/pull、空 token、不等长输入和完整梯度。
MegaGate 覆盖 router、MoE、FSDP BF16 和 EP 一致性。

在仓库根目录执行默认的 13 项测试：

```bash
python -m pytest -v tests/torch/multicore
```

添加 `-m level0` 仅选择单卡集成用例；`-m level1` 选择多卡及 Gate 集成用例。
`HP_MEGA_MOE_LEVEL1_EXTENDED=1` 扩展混合并行组合和动态专家输入，总计 19 个参数项：

```bash
HP_MEGA_MOE_LEVEL1_EXTENDED=1 python -m pytest -v tests/torch/multicore
```

`HP_MEGA_MOE_EVIDENCE_DIR` 指定 MegaMoE 的多 rank JSON 结果根目录，未设置时使用 pytest 临时目录。
结果路径为 `<root>/<worker_case>/<transport>/<variant>/<worker_case>.json`，
内容包含各 rank 的数值结果及实际加载的 HyperMulticore SO 身份。

Launcher 与 worker 的职责见[测试规则](../../../.agent/rules/testing.md)。
