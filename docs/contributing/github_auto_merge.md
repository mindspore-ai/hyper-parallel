# GitHub：Reviewers 全员批准后自动合并

此功能按每个 PR 侧栏的 **Reviewers** 确定必审人员，不要求仓库维护团队全员参与。
代码合入默认分支并完成下述仓库设置后才生效；添加工作流文件本身不会修改远端规则。

## 审批规则

必需状态名为 `all-reviewers-approved`，结果写到 PR 当前 head SHA。

GitHub 的 `requested_reviewers` 只包含尚待评审的人；提交评审后，此人从该数组移到 reviews。
因此检查合并两个来源：当前请求评审的用户，以及已经提交正式 review 的用户。
普通 PR Conversation 评论不属于正式 review，不会增加必审人员。
参考 [GitHub Review Requests API](https://docs.github.com/en/rest/pulls/review-requests)。

- 每位评审人的有效结论都必须是针对当前 head SHA 的 `APPROVED`。
- `CHANGES_REQUESTED`、被撤销的批准、仅 `COMMENTED`、仍被请求重新评审，均不通过。
- 批准后的普通 review 评论不会取消批准；未提交的草稿 review 不参与判断。
- 尚未评审的人被移出 Reviewers 后，不再必审。已提交 review 的人仍参与检查。
- 新增评审人后，必须等此人批准。推送新提交后，旧 SHA 上的批准不再满足本检查。
- PR 作者不参与自己的审批；没有任何独立评审人时不通过，避免空列表被视为全员批准。
- 团队仍出现在 `requested_teams` 时等待其评审请求被 GitHub 满足；不展开为团队全员。
- 机器人如果出现在正式 Reviewers 中，同样必须批准；仅发表评论的机器人不会自动视为批准。
- Draft PR 不通过。多个 PR 共用同一个 head SHA 时，所有相关 PR 都满足条件才发布成功。

例如：Reviewers 为 Alice、Bob、Carol 时，只需这三人批准。Alice 批准后，即使
`requested_reviewers` 只剩 Bob、Carol，Alice 仍保留在本次检查的人员集合中。

## 文件和触发方式

| 文件 | 用途 |
| --- | --- |
| [策略配置](../../.github/reviewer-approval-policy.json) | 仓库、目标分支和自动合并开关；不保存人员名单 |
| [检查逻辑](../../.github/scripts/reviewer-approval-gate.mjs) | 读取实时审批、发布必需状态、开启自动合并 |
| [检查工作流](../../.github/workflows/reviewer-approval-gate.yml) | 在可信的默认分支上下文执行 |
| [评审事件转接](../../.github/workflows/reviewer-approval-signal.yml) | 将 Fork 上的只读评审事件交给可信工作流处理 |
| [测试](../../.github/scripts/reviewer-approval-gate.test.mjs) | 审批状态、人员变化、并发更新、分页及写入边界 |

配置覆盖 `master`、`r1.0.0`、`r1.1.0`。新增发布分支时，在策略配置中补充具体名称。
所有目标分支统一使用默认分支上的策略和脚本，PR 不能通过修改自己的配置或脚本放行自己。
默认分支从 `master` 改名时，还需更新工作流中的 `push.branches`。

PR 创建、更新、重开、转 Draft/Ready、调整 Reviewers 和关闭会触发检查。
Review 的提交、编辑和撤销通过独立 signal 工作流触发 `workflow_run`。
signal 不执行 PR 代码、不下载产物、不使用密钥；后续工作流重新向 GitHub 查询真实状态。
主工作流只检出默认分支，不检出 PR head。

每次写入运行重新核对全部目标分支上的开放 PR，并串行执行，避免排队事件被合并后漏掉其他 PR。
另有每 15 分钟一次的兜底核对，以及手动 `workflow_dispatch` 入口。
工作流不会发送评论或请求评审，也不会改变 Reviewers 名单。
长期待审且结论未变化时不会重复发布相同状态。

## 一次性启用

1. 将这些文件合入默认分支。为 `.github/` 的工作流、脚本和策略配置保留适当的代码审查要求。
2. 在 Actions 中手动运行 **Reviewer approval gate**，让 `all-reviewers-approved` 首次出现在开放 PR 上。
3. 在保护目标分支的 **active Ruleset** 中，把 `all-reviewers-approved` 添加到
   **Require status checks to pass before merging**，来源选择 **GitHub Actions**（App ID `15368`）。
   保留已有 `label-check` 及其他质量检查。
4. 保留 **Require a pull request before merging**、至少 1 个批准，以及
   **Dismiss stale pull request approvals when new commits are pushed**。
   现有 Code Owner、最近推送独立批准和讨论解决规则继续生效。
5. 在 **Settings → General → Pull Requests** 勾选 **Allow auto-merge**。
6. 用一个测试 PR 验收：两位 Reviewers 中只有一人批准时仍等待；第二人批准且其他必需检查通过后自动合并。
   再分别验证新增/移除待审人员、撤销批准和推送新提交。

检查通过后，脚本核对目标分支的实际生效规则。只有本检查绑定 GitHub Actions、要求审批并撤销旧批准，
且仓库开启 Auto-merge，才执行带 `--auto --merge --match-head-commit` 的合并命令。
采用仓库当前允许的 Merge commit 方式，不使用 `--admin` 绕过规则。
仓库开关或上述条件未配置时，只发布审批检查，不自动合并。

此实现的自动合并只支持 Rulesets，不自动修改 Rulesets、经典 branch protection 或 bypass 列表。
GitHub Actions 的令牌没有规则管理权限；要让管理员也受门禁约束，应在仓库规则中配置相应 bypass 策略。
若使用 Merge Queue，本实现保留审批检查，但不自动入队，需要另行配置队列授权。

## 本地验证

检查脚本使用 Node.js 22 或更新版本及已登录的 GitHub CLI，不依赖 Python 或框架运行环境。

```bash
node --test .github/scripts/reviewer-approval-gate.test.mjs

GH_REPO=mindspore-ai/hyper-parallel \
  node .github/scripts/reviewer-approval-gate.mjs --dry-run --pr 1335
```

省略 `--pr` 时只读检查全部开放 PR；默认也是 dry-run。
只读模式不会发布状态、开启 Auto-merge 或合并 PR。写入仅在工作流中显式使用 `--write`。
更新本说明后，仓库目录检查使用指定环境：

```bash
conda run -n py312 python .agent/scripts/check_agents_catalog.py
```

## 运行边界

这是异步 GitHub Actions 检查。审批或人员变化到工作流重新发布结果之间存在延迟；
工作流排队、API 不可用和 GitHub 定时任务延迟都可能延长它。
脚本在发布成功前复查提交、目标分支、人员和审批，发生 API 错误时尽量把旧成功改成阻塞状态，
但外部必需检查与 GitHub 合并动作不是同一个原子事务，不能保证零时间窗口。
原生审批规则仍须保留；不能把本检查描述成 GitHub 平台内置的“全员审批原子校验”。

运行日志和 Actions Summary 会列出仍待批准的人及团队；PR 上的状态链接指向发布结果的工作流。
正式启用前还需在 GitHub 上验收事件触发、令牌权限、必需检查绑定和实际自动合并；本地测试不能替代这些验收。

## GitHub 自动化功能范围

2026-10-29 确认的范围如下；未实施的事项不改变现有仓库门禁。

| 功能 | 决定 |
| --- | --- |
| 实际安装包验证 | 优先实现 PR 内的 wheel 构建、隔离安装和 CPU 冒烟检查，见[发布流程](release.md#安装包验证) |
| GPU/NPU 测试资源排队 | 需要，暂缓实施；接入条件见下文 |
| 测试结果留存、文档预览 | 保留已有能力；按最新要求记录为后续完善事项，实施前核对已有覆盖，避免重复建设 |
| 按需选择测试 | 暂不需要 |
| 值班轮转和重新评审提醒 | 不需要 |
| Actions 依赖版本自动升级 | 不需要 |
| Scorecard | 不需要 |
| 自动冻结发布版本 | 不需要 |

## 后续待办：GPU/NPU 测试资源排队

状态：2026-10-29 记录需求，按维护者要求暂缓实施；当前没有新增调度工作流或修改生产 CI。

目标是在有限的 GPU/NPU 资源池上限制并发测试，按入队顺序分配空闲资源，避免多个 PR 同时争用同一组卡。
参考 Megatron-LM 的 [测试队列管理工作流](https://github.com/NVIDIA/Megatron-LM/blob/0a513268eee94d8e903fecbc21fde862eecc31f3/.github/workflows/cicd-approve-test-queue.yml)。

实施前需要确认：

- 实际提交和调度测试的 Jenkins 流水线代码所在仓库或文件路径。
- GPU 与 NPU 的资源池划分、每个任务占用的卡数，以及各池允许的并发数。
- 共享这些资源的其他任务及其调度入口，确保资源限制覆盖所有使用方。

待评估的实现要求：

- 在实际分配硬件的调度层管理资源；单独限制 GitHub Actions 并发不能约束外部 Jenkins 测试。
- GPU、NPU 及不同规格资源分别排队；等待资源期间不占用测试卡。
- 同一 PR 推送新提交后淘汰旧的待运行请求；开始执行和回报结果时核对提交 SHA。
- 任务结束、失败或取消后释放资源；超时和调度故障保持可见，不能把未执行的测试视为通过。
- 在 PR 中提供排队、运行及完成状态，并关联实际测试记录。

该需求独立于 Reviewers 全员批准规则；实现时仍须保留现有审批和 CI 必需检查。
