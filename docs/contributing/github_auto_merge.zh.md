# GitHub：达到评审批准人数后自动合并

[English](github_auto_merge.md) | **简体中文**

此功能按每个 PR 侧栏的 **Reviewers** 确定评审人员，不要求仓库维护团队全员参与。
代码合入默认分支并完成下述仓库设置后才生效；添加工作流文件本身不会修改远端规则。

## 审批规则

必需状态名为 `all-reviewers-approved`，结果写到 PR 当前 head SHA。
为兼容已有 Ruleset，状态名称保持不变；实际检查按批准人数判断。

GitHub 的 `requested_reviewers` 只包含尚待评审的人；提交评审后，此人从该数组移到 reviews。
检查沿用原有逻辑，合并当前评审请求和已提交的正式 review，并排除 PR 作者；
机器人及未经单独邀请而提交的正式 review 也会参与统计。
普通 PR Conversation 评论不属于正式 review，不会增加评审人员。
参考 [GitHub Review Requests API](https://docs.github.com/en/rest/pulls/review-requests)。

- 只有一位评审时，需要一人批准；有两位及以上时，任意两位不同评审批准即可达到人数要求，
  无需等待其他被请求的人批准。
- 只有针对当前 head SHA 的有效 `APPROVED` 才计入批准数；被撤销、过期或被重新请求评审的批准不计入。
- 非作者的评审人（包括机器人）有未解除的 `CHANGES_REQUESTED` 时，即使批准人数达标也不通过；
  仅 `COMMENTED` 不算批准。
- 批准后的普通 review 评论不会取消批准；未提交的草稿 review 不参与判断。
- 尚未评审的人被移出 Reviewers 后，不再计入人数。已提交 review 的人仍参与检查；从一人增至两人时，
  批准门槛会从一人升为两人。
- PR 作者不计入评审人数或批准人数；没有其他评审时不通过。机器人 review 按原有规则统计。
- 团队仍出现在 `requested_teams` 时等待其评审请求被 GitHub 满足；不展开为团队全员。
- Draft PR 不通过。多个 PR 共用同一个 head SHA 时，所有相关 PR 都满足条件才发布成功。

例如：Reviewers 为 Alice、Bob、Carol 时，Alice 和 Bob 批准即可达到人数要求，不必等待 Carol，
但仍需满足其他阻塞评审和团队请求。GitHub 原有的 Code Owner、最近推送独立批准、讨论解决及 CI 检查继续生效。
本自定义检查不验证仓库写权限，因此未经单独邀请的批准也可能计入人数；
GitHub 原生审批和 Code Owner 规则仍会独立检查各自的权限条件。

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
6. 用测试 PR 验收：一位评审需一人批准；两位及以上需两人批准，无需等待第三人；其他必需检查通过后自动合并。
   再分别验证新增/移除评审人、修改请求、待审团队、撤销批准和推送新提交。

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
原生审批规则仍须保留；本外部检查不等同于 GitHub 内置的原子审批校验。

运行日志和 Actions Summary 会列出批准数量与门槛、待审人员和团队以及修改请求；PR 状态链接指向工作流。
正式启用前还需在 GitHub 上验收事件触发、令牌权限、必需检查绑定和实际自动合并；本地测试不能替代这些验收。

## CI 通过标签的 24 小时有效期

`ci-pipeline-passed` 和 `ci-pipeline-passed-but-overtime` 分别从 GitHub 最近一次 `labeled` 事件的
`created_at` 开始计时，满 24 小时即失效。两个标签同时存在时分别计算；只要其中一个仍有效，
`label-check` 就可通过。评论、评审或其他标签变化不会续期。删除后重新添加会重新计时；
对已经存在的标签重复执行添加操作，没有新的 `labeled` 事件时不会续期。

[定时清理工作流](../../.github/workflows/expire-ci-pass-labels.yml) 每 5 分钟检查所有开放 PR，
也支持在 Actions 页面手动运行。它只执行默认分支上的[脚本](../../.github/scripts/ci-label-expiry.mjs)，
使用内置 `GITHUB_TOKEN`，不需要新增密钥或仓库必需检查。关闭或已合并的 PR 不清理。
合入默认分支后即开始生效，包括此前已经添加的标签；标签没有可验证的添加记录时会报错，不猜测时间。

定时任务先撤销失效的 GitHub Actions `label-check` 通过结果，再删除过期标签。
这是因为 `GITHUB_TOKEN` 删除标签不会触发新的 `unlabeled` 工作流，单独删除会留下旧成功结果。
脚本检查 PR head 和当前测试合并提交上的结果；若另一个标签仍有效，则只移除过期标签。
正常标签检查也直接验证添加时间。`Label Gate Check` 完成时会触发一次针对该 head 的核对，
处理标签删除期间仍在运行的检查随后发布旧成功的情况。定时扫描跳过没有 CI 通过标签的 PR，
减少 API 调用；检查完成事件会核对没有标签的情况。

GitHub Actions 定时任务可能排队或延迟，因此实际删除发生在满 24 小时后的下一次执行，
不能保证精确到秒，也不能保证延迟不超过 5 分钟。标签、检查状态和合并不是原子事务；
脚本写入前复查当前提交与标签事件以缩小并发窗口，冲突或 API 错误会在日志中报告；定时扫描下轮重试，事件核对失败时可重跑失败的工作流。
参见 [GitHub 工作流事件说明](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)。

推送新提交后仍由原有的 [Reset CI pass labels](../../.github/workflows/reset-ci-pass-labels.yml)
立即清理这两个标签，不等待 24 小时。需要续期时应重新执行 CI，由 CI 删除旧标签并添加新结果。
此功能不会启动 Jenkins 测试，也不会因为标签过期撤销 Reviewers 的审批。

本地测试与只读预览：

```bash
node --test .github/scripts/ci-label-expiry.test.mjs
GH_REPO=mindspore-ai/hyper-parallel node .github/scripts/ci-label-expiry.mjs --dry-run --pr 1356
```

写入模式仅由可信工作流使用 `--write` 执行。合入后应手动运行一次清理工作流，
验收标签删除和旧 `label-check` 失效；本地模拟测试不能替代 GitHub 写入权限的实际验证。
