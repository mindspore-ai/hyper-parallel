# CLA 检查与 PR 标签

中文 | [English](cla.md)

HyperParallel 使用托管的 [CLA Assistant](https://github.com/cla-assistant/cla-assistant) 集成。
该服务为 PR 贡献者发布 `license/cla` 检查状态。贡献者可通过该状态的 Details 链接签署或重新检查协议。
签署记录由 CLA Assistant 管理，GitHub 标签不是真实签署记录。

## 自动标记创建者的签署状态

[Sync CLA labels](../../.github/workflows/cla-label-sync.yml) 查询 **PR 创建者本人是否签署**：

| 个人查询结果 | 标签 |
| --- | --- |
| 创建者已签署当前协议 | `cla-signed` |
| 创建者未签署当前协议 | `cla-unsigned` |
| 缺少凭据、未配置协议、服务故障或响应无效 | 同步失败，现有标签保持不变 |

[脚本](../../.github/scripts/cla-label-sync.mjs) 将 PR 的 `user.login` 和 `user.id` 传给 CLA Assistant
的 `hasSignature` 接口，不传 PR 编号，从而查询该用户当前的个人签署记录。
查询前先确认仓库已配置协议，避免将“未配置协议”误判成“已签署”。
参考服务的 [API 实现](https://github.com/cla-assistant/cla-assistant/blob/main/src/server/src/api/cla.js)
和[签署查询实现](https://github.com/cla-assistant/cla-assistant/blob/main/src/server/src/services/cla.js)。

现有 `license/cla` 状态可能涉及多位贡献者，其通过或失败不作为创建者本人的签署结果。
例如：创建者已经签署，但另一位贡献者尚未签署，整体检查仍在等待时，创建者标签仍为 `cla-signed`。
本功能不改变现有 CLA 集成的检查逻辑。

同步成功后两个标签互斥。工作流还会从 PR 中移除旧标签 `mindspore-cla/yes`、`mindspore-cla/no`、
`signed-CLA`、`unsigned-CLA`，保留其他标签。上线时由管理员删除旧的仓库标签定义，不能将旧标签直接迁移为
新标签来证明签署状态。新提交会重新查询个人记录；签署不绑定某次代码提交，因此仍已签署的创建者会保留 `cla-signed`。

## 必需配置

个人查询是 CLA Assistant 的管理员认证接口。启用前，必须配置仓库 Actions Secret **`CLA_ASSISTANT_TOKEN`**，
提供 CLA Assistant 接受的、本仓库已授权管理员的专用凭据。该账号必须已在 CLA Assistant 中授权，
任意 GitHub 令牌并不一定可用。内置 `GITHUB_TOKEN` 仅用于 GitHub 标签操作，不能替代此凭据。
不要将凭据写入代码、PR 描述或聊天消息。

配置入口：**Settings → Secrets and variables → Actions → New repository secret**。
缺少凭据或认证失败会使工作流明确报错，不会标记为 `cla-unsigned`。
工作流合入默认分支后，手动运行 **Sync CLA labels**，验证凭据并同步现有开放 PR。
自动化测试使用模拟响应，上线仍需验证托管服务的真实认证查询；账号或服务接口变更后也应重新验证。

## 触发与保护

PR 创建、重开、推送新提交以及 `license/cla` 状态变化都会触发同步。
状态事件仅用于唤醒工作流，标签仍来自新发起的个人查询。
支持手动运行，并每 6 小时兜底扫描开放 PR；关闭或已合并的 PR 跳过。
若补签未触发该 PR 的状态事件，定时扫描或手动运行会刷新标签。同步是异步的，GitHub 定时任务可能延迟。

凭据仅交给默认分支的代码，不执行 PR 代码或下载的产物。
CLA 凭据只发送给固定的 CLA Assistant 签署查询地址，并禁止重定向。
修改标签前重新核对 PR 创建者、当前 head、签署结果、协议及受管标签；并发变化会产生可重试错误。
标签变更与外部签署记录变化不是原子事务。

查询失败会保留此前的标签，因此标签表示最近一次成功查询结果，并非实时保证。
解决故障后可重跑任务或手动同步。标签不替代真实签署记录，也不会自动成为合入条件。
本功能不修改仓库 Rulesets 或分支保护规则。

## 验证

```bash
node --test .github/scripts/cla-label-sync.test.mjs
# 先完成 gh 认证，并通过安全方式在环境中提供 CLA_ASSISTANT_TOKEN。
GH_REPO=mindspore-ai/hyper-parallel node .github/scripts/cla-label-sync.mjs --dry-run --pr 1365
```

默认只读预览，只有可信工作流使用 `--write`。
完成配置并合入后，验证创建者已签署、未签署、补签后的标签转换，以及其他贡献者未签署但创建者已签署的情况。
确认旧标签已清理，其他标签未受影响。本地测试不能替代真实凭据、写入权限与事件触发的验收。
