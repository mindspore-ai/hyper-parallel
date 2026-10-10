# 精度对齐流程（采集 → 比对 → 首个异常）

基于公开的 MindStudio **msprobe**（MindStudio Probe，隶属 mindstudio
accuracy tools）。标准流程分三步：配置 dump → 跑出两份 dump（候选 + 基线）
→ 比对定位首个超阈 API/module。专有的打点策略与判据标 `[待对齐]`。

## 0. 前置

- msprobe 随 MindStudio / att 工具链安装；本 skill 的脚本通过子进程调用
  `msprobe` CLI，不在进程内硬 import，故无工具时脚本仍可加载。
- 基线必须显式：同模型/同配置/同并行/同数据序，仅被测轴不同（见
  `rules/precision-acceptance.md`）。

## 1. 配置 dump

msprobe 用一个 dump 配置（JSON）描述采集范围（level=L0 module / L1 API /
mix）、step 范围、采集内容（统计量 / 全量 tensor）。

- **采集粒度**：先 L1（API 级）粗定位首个异常 API；需要定位到 module/行时
  再 L0 + 指定 scope。
- **step 范围**：默认采前几个 step（首 step 对齐最能隔离前反向正确性）。
- `[待对齐]`：内部实现对「自动增加打点」的具体策略（哪些 module 强制打点、
  noise 容忍）与我们的默认配置差异，待对齐。

脚本：`collect_dump.py --run-dir <d> --launch "<训练/推理启动命令>"
--level L1 --step 0-2` —— 它写出 dump 配置、在该配置下拉起 run、把 dump
收集到 run-dir。

## 2. 跑出候选与基线两份 dump

- 候选：被测实现/设备。
- 基线：参考实现/设备（CPU 或可信参考），同输入、同种子。
- 两份 dump 分别落在各自 run-dir；采集期保持确定性（固定种子、关闭非确定
  算子或记录其影响）。

## 3. 比对定位首个异常

`diff_dump.py --candidate <cand_dump> --baseline <base_dump>` 调
`msprobe compare`，产出逐 API/module 的比对表（余弦相似度、最大绝对/相对
误差、是否 NaN/Inf 等），脚本解析出：

- **首个超阈项**：按执行序第一个余弦相似度 < 阈 或 误差 > 阈 的 API/module。
- 该项的名称、所在 module、（L0 下）对应代码位置线索。
- 判据阈值：`[待对齐]` 与内部口径对齐；默认用 msprobe 常用阈（余弦
  ≥ 0.99、最大相对误差在噪声带内），并与 `precision-acceptance` 的 ulp
  并列口径一致（bf16 并列差 < 1 ulp 记为噪声，不算异常）。

## 4. 产出报告

把首个异常、误差数字（附命令/commit/环境）、下一步修复意见写进
`templates/precision-report.md`。

## 纪律

- 报「首个」异常而非末端症状：下游 NaN 多是上游首个超阈项的传播。
- 每个数字可复现（命令+commit+环境）；A/B 同节点背靠背。
- 负结果先验采集是否真生效（dump 非空、打点命中），再下结论。
