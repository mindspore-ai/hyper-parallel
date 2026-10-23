# MegaMoe 系统测试

Launcher 保持无框架导入，设备断言放在 `_test_*.py` worker 中。

所有 multicore launcher 通过 `_case_launcher.py` 启动 worker：在继承的
`ASCEND_RT_VISIBLE_DEVICES` 范围内选择物理卡，并对重叠设备加进程锁。
未设置掩码时使用从 0 开始的卡号；不同卡集合仍可并行。锁只在 multicore 测试间生效，
等待不计入 worker 超时；默认 420 秒，合并后的动态 token worker 为 840 秒，
等于原两套 worker 的总预算。SHMEM launcher 保留原有不设内部超时的行为。
启动日志分别记录设备等待时间和实际卡号，便于区分排队与执行耗时。
约束见 [测试规则](../../../.agent/rules/testing.md) 和 [环境要求](../../../AGENTS.md)。

| Level1 范围 | 默认实例数 | 关键覆盖 |
| --- | ---: | --- |
| 固定长度核心 | 16 | 容量/reuse/retained backward、poison、Router F+B+SGD、group-list、subgroup、heap growth、布局及代表性性能 |
| 动态 token | 2 | EP8 push/pull、不等长/空 rank/全空/热点、长度边界、完整梯度、扩容及七桶复用、共享资源/stream/checkpoint/SGD |
| 独立 profiler | 1 | AIC/AIV trace、cycle、无 dropped records |
| DSV4.1 adapter | 2 | 外部权重、clamp=10 饱和、变长及完整梯度 |
| DSV4.1 混合并行 | 6 | TP2、CP2、TP2CP2，覆盖拓扑/文本视觉/transport 的所有两两组合 |

合计 **27 个**实例。设置 `HP_MEGA_MOE_LEVEL1_EXTENDED=1` 后再收集 pytest，
可恢复本轮裁剪前的动态 token 和 DSV4.1 输入及交叉组合，合计 **33 个**实例。
默认混合并行采用 TP2 文本 push/视觉 pull、CP2 视觉 push/文本 pull、
TP2CP2 文本 push/视觉 pull；完整三因素组合在扩展矩阵中。

动态 token 与纯自动 token 共用每种 transport 一套 EP8 worker，避免重复初始化进程、
共享层 checkpoint/跨 stream/SGD 及延迟 backward。合并后的主循环默认 15 组输入（扩展 20 组），
保留 `127/128/129`、`255/256/257`、`4095/4096/4097`、空 rank、全空、热点和旧长度重访。
均匀 `385/513/641/4095` 及 `[4096, 1, ...]` 位于扩展矩阵；`385/513` 仍用于
五个 outstanding forward 的正逆序 backward。独立检查首次全空的最小分配、source 扩容、
同桶计划替换和混合 transport 的 heap 重启期间完成旧 backward，以及七桶预热后的复用。
Padding 对照默认 1 组不等长含空 rank 输入（扩展 3 组），复用主循环已有的输出和完整梯度。
DSV4.1 adapter 默认 3 组输入（扩展 5 组），保留全空、tile 边界、部分空及最大长度。
数值阈值不变：动态主断言采用独立 BF16 舍入参考，纯 FP32 误差另作诊断。
上述次数变化不代表实测耗时收益。

固定长度生命周期继续负责三层训练、内存/SHMEM allocation 稳定、ready generation、
无额外 host barrier 和 launch 异常恢复。Group-list 默认专家数为 `13,16,17,32,33,128`；
`HP_MEGA_MOE_GROUP_LIST_EXTENDED=1` 可恢复其完整矩阵。

统一入口设置 `HP_MEGA_MOE_EVIDENCE_DIR` 后，结果路径为：

```text
<root>/<worker_case>/<transport>/<variant>/<worker_case>.json
<root>/<worker_case>/<transport>/<variant>/worker_logs/...
```

variant 区分 heap-growth 或 TP/CP/vision 场景。复用现有 all-gather 证据，要求完整 rank 结果；
启动前删除该场景旧 JSON，避免陈旧结果误通过。完整 rank 日志会归档，失败时输出有界尾部。
JSON 环境身份包含源码与 native 信息；Git 不可用时保留空 SHA 和错误原因，其他错误继续传播。
重复运行请使用不同 root。独立 worker 原有的分 rank 输出环境变量与文件名保持可用。

```bash
python -m pytest -v tests/torch/multicore/test_mega_moe.py \
    tests/torch/multicore/test_mega_moe_profiling.py \
    tests/torch/multicore/test_deepseek_v41_megamoe.py -m level1
```
