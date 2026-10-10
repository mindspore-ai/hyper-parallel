# Autoresearch 操作手册

这是代理的操作手册。一轮 = 一个优化实验：改唯一可改文件 → 跑基准 →
过门禁 → 记录 → keep / discard / crash。循环持续运行直到人工停止。

## [SETUP] 本轮配置

- **优化目标**：<填：优化什么>
- **主指标**：<填：benchmark_cmd 输出的哪个数字，方向>
- **起点**：<填：当前实现与已知性能>
- **脚手架等级**：<填：允许读哪些参考、是否允许联网/读历史>

## 唯一可改文件

见本目录 `run.json` 的 `target_files`。工具会拒绝对该清单之外的
未提交改动开测。

**禁止修改**：<填：参考实现、门禁脚本、trainer 等——改了它们等于改考卷>。
可以在本 run 目录的 `scripts/` 下新建自己的分析工具。

## 读取范围

**可以读**：<填：目标文件、参考实现（只读）、本目录活动文件、自产 profile>
**不要读**：<填：与目标无关的子系统；如需越界先在 experiment_log 写明理由>

## 历史与门禁读取禁令（硬性，不随 [SETUP] 放宽）

- **不要查 git 历史**：禁止 `git log`、`git log -p`、`git show <提交>`、
  `git diff <提交>`、`git blame`——既往提交的标题与 diffstat 会泄漏希望
  你自主重发现的优化技术。工作区口径的 `git status`、不带提交引用的
  `git diff` 可用。要找你本轮自己的提交，只看 `results.tsv` 第二列。
- **不要读门禁脚本的源码**：只运行它；失败时只看输出里的断言消息。
  门禁源码中的符号名同样会泄漏既往优化点。
- 一份材料能不能读拿不准时，一律按**不可读**处理。

## 基准与门禁

由 `run.json` 的 `benchmark_cmd` / `gate_cmd` 定义，`bench_must_match` /
`gate_must_match` 里的正面标记（例如生效置位 `engaged=1`）缺失即按
crash 处理——"没有告警"永远不构成"优化在跑"的证据。首测落在噪声带内
时工具会自动重跑取中位数再判定（`noise_confirm_reruns`），不要自己
用单次读数下结论。

## 记录

由工具维护：`results.tsv` 逐行追加（timestamp/commit/metric/status/
description），实验提交与回滚由工具执行。你负责在每轮后追加
`experiment_log.md` 条目、维护 `ideas.md` 与 `learnings.md`。

## 实验环

永久循环：

1. 重读 `ideas.md`、`learnings.md`、`experiment_log.md`，选下一个想法。
2. 在 `target_files` 里实现改动（保持工作区只有这些文件脏）。上下文
   吃紧时可把实现与跑测委托给子代理：把本手册的禁令与命令**原文**
   透传给它，主环只收结论，不替它复述规则。
3. `python -m hyper_parallel.tools.autoresearch iterate --run-dir <本目录>    --description "<短描述>"` —— 提交、门禁、基准、判定、记账、
   keep/restore 全部由工具完成。
4. 按工具返回的状态追加 `experiment_log.md`，更新 ideas/learnings。

**崩溃**：易修的修了重跑；根上不行就记录后换方向。
**没想法时**：profile、读图、从 `learnings.md` 的反例找新角度，不停。

## 环境纪律（硬性）

- <填：资源占用判据与让路规则；绝不挤占他人任务；跑完清理>
- <填：环境已知坑，如超时、缓存、并发编译>
