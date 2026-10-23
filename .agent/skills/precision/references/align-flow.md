# 精度对齐流程（采集 → 比对 → 首个异常）

基于公开的 MindStudio **msprobe**。标准流程三步：进程内插桩写 dump → 候选与
基线各跑一次 → 比对定位首个超阈 API/module。

> 本文的接口、字段与列名均对着实机核过（msprobe 随 CANN 9.0.0 /
> torch_npu 2.9.0.post2 安装，Ascend 910）。实测记录见本 skill 的 PR 正文。

## 0. 前置：采集必须进程内插桩

**msprobe 没有环境变量入口。** 整个包只读一个 `MSPROBE_*` 变量
（`MSPROBE_LOG_LEVEL`），**没有任何 config-path 变量**。插桩只在训练进程内
构造 `PrecisionDebugger(config_path=...)` 时发生——导出一个配置路径再拉起
命令，采集不会发生、dump 目录为空，而空 dump 在比对侧会被读成「无差异」。

因此采集分两端，与 `profiling` / `memory-analysis` 同形：

| 端 | 文件 | 职责 |
|---|---|---|
| 启动侧 | `scripts/collect_dump.py` | 写配置、导出 `PRECISION_DUMP_*`、拉起 run、**逐 step 校验 dump 落盘** |
| 运行侧 | `scripts/precision_hook.py` | 训练进程内读这些变量、构造 debugger、`capture()` 包住 step |

```python
import sys
sys.path.insert(0, ".agent/skills/precision/scripts")
import precision_hook

precision_hook.start()
for batch in loader:
    with precision_hook.capture():
        train_step(batch)
    precision_hook.step()
```

**一进程一份 dump。** `PrecisionDebugger` 是进程级单例：同进程内构造第二个是
静默空操作，仍沿用第一份配置，于是第二份 dump 目录全空。基线与候选必须是
**两次独立启动**。`precision_hook.start()` 对重复调用直接抛错而不是放过。

基线必须显式：同模型/同配置/同并行/同数据序，仅被测轴不同（见
`rules/precision-acceptance.md`）。

## 1. 配置 dump

msprobe 的真实 schema（对着安装包自带 `config.json` 核过）——顶层是
`task` / `dump_path` / `rank` / `step` / `level`，**按 task 命名的块里**才是
`scope` / `list` / `data_mode` / `summary_mode`。顶层写 `scope` 不在 schema 内、
会被忽略。

```json
{
  "task": "statistics",
  "dump_path": "<dir>", "rank": [], "step": [0, 1], "level": "L1",
  "statistics": {"scope": [], "list": [], "data_mode": ["all"],
                 "summary_mode": "statistics"}
}
```

- **`step` 是要采的步号列表，不是区间**。配置里写 `[0, 2]` 采第 0 与第 2 步、
  **静默跳过第 1 步**；`"0-2"` 这种字符串形式 msprobe 会自己展开成 0/1/2。
  本 skill 的脚本把区间展开成整数列表，好让写出的配置可被逐条校验。
  空列表 = 采全部步。
- **采集粒度**：先 L1（API 级）粗定位首个异常 API；需要定位到 module/行时
  再 L0 + 指定 scope。
- **task 决定比对能看到什么**（见下）：`statistics` 落逐张量统计量，
  `tensor` 落全量张量、**只有它的比对表里有 `Cosine`**。
- **dump 路径有字符校验**：方括号、逗号等特殊字符会被 msprobe 直接判非法路径
  （`FileCheckException`）。按配置值命名目录容易踩到，`collect_dump.py` 提前拦。

脚本：
```bash
python .agent/skills/precision/scripts/collect_dump.py \
    --run-dir <d> --launch "<训练启动命令>" --task statistics --level L1 --step 0-1
```
它校验每个配置步是否产出了含 `dump.json` 的 step 目录；**run 退出码为 0 但
没采到东西一律判失败**，并提示运行侧没有构造 debugger。

## 2. 比对

```bash
python .agent/skills/precision/scripts/diff_dump.py \
    --candidate <cand_dump> --baseline <base_dump> [--step N]
```

底层命令是 `msprobe compare -tp <target> -gp <golden> -o <out>`。
**没有 `-f pytorch`，也没有 `-np` / `-bp`**；默认输出 CSV。
`-tp` / `-gp` 要指到 **step 目录**（含 `dump.json`），指 dump 根目录会被
路径检查直接拒。结果文件名形如 `compare_result_proc<pid>_<时间戳>.csv`，
按修改时间取最新（按名字排序会挑错）。

### 比对表的三个坑

1. **列随 task 不同**。`statistics`：`Max diff` / `Min diff` / `Mean diff` /
   `L2norm diff` + `*RelativeErr`；`tensor`：`Cosine` / `EucDist` /
   `MaxAbsErr` / `MaxRelativeErr` / `One Thousandth Err Ratio`。
   **在 statistics 表里找 `Cosine` 每行都找不到。**
2. **`MaxRelativeErr` 的格式随 task 不同**。statistics 是带百分号的字符串
   （`'1.1049723756906076%'`），tensor 是裸比值（`'0.013333333656'`）。
   对前者直接 `float()` 会抛 `ValueError`；若把异常当成「该行不可比较」而
   跳过，**整表每一行都被跳过，真实发散的运行会被读成「无差异」**。
3. **msprobe 自己的 `Result` 列不是判据**。实测在 `MaxRelativeErr=9368` 的行上
   仍写 `'pass'`。报告里带出来供参考，绝不拿它当门槛。

非张量入参（如 `bias=None`）写 `'unsupported'`，跳过即可——那不是发散。
但 **`nan` 不跳过**：非有限的指标正是本 skill 要找的故障，任何 band 下都报。

### 判据：默认逐位，band 是显式选项

默认任何非零差异即发散，于是报出的是**首因**。给了 band
（`--max-rel` / `--cosine-min`）才按 band 判，而**band 大到能忽略噪声，就大到
能盖住首因**：实测单层权重被缩放 1.01 时，被扰动的权重本身
`MaxRelativeErr=0.0133`、`Cosine=0.999998`，落在 0.02 / 0.99 band **之内**，
而它传播出去的 output 落在 band 之外——于是 band 下报出的是下游症状、首因被丢掉。
这与本 skill「报首因不报末端症状」的硬规则直接冲突，故默认不设 band。

**`rules/precision-acceptance.md` 的 0.02 不是逐张量容差**：那条规则说的是
**1000 步 loss 轨迹**的平均绝对误差 ≤0.02 或平均相对误差 ≤2%，是端到端验收口径。
把它搬来当单算子的 band 没有依据，本 skill 不这么搬。bf16 并列噪声按该规则的
ulp 口径判（并列差 < 1 ulp 记噪声），并在报告里写明并列数与最差差值。

## 3. 产出报告

把首个异常、误差数字（附命令/commit/环境）、下一步修复意见写进
`templates/precision-report.md`。

## 纪律

- 报「首个」异常而非末端症状：下游 NaN 多是上游首个超阈项的传播。
- 每个数字可复现（命令+commit+环境）；A/B 同节点背靠背。
- **负结果先验采集是否真生效**（dump 非空、步数齐）。这条已由
  `collect_dump.py` 的逐步校验机器化——它曾是本 skill 最容易出的错：
  采集什么都没发生，比对照样输出一张「无差异」的表。
