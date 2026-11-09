# HyperParallel GitHub 开发与协作手册

[English](github_development_manual.md) | **简体中文**

版本：1.0 · 更新日期：2026-09-18 · 适用仓库：[mindspore-ai/hyper-parallel](https://github.com/mindspore-ai/hyper-parallel)

本文面向从 GitCode 迁移到 GitHub 的开发者、评审人、仓库管理员和 CI 维护者，提供实际操作、验收方法及故障处理入口。命令以 Bash / Zsh 为例；Windows 可使用 Git Bash，硬件测试仍须在项目支持的环境执行。

**如何区分规则与建议：**

- **现行配置**：2026-09-18 核实的仓库设置，配置变更后应同步更新本文。
- **操作规范**：按现行规则完成开发和协作的方法。
- **建议落地**：尚未完成的平台改造，不能视作已经启用的能力。
- **历史案例**：某次 PR 或规则执行的快照，不代表该 PR 今天仍处于同一状态。

## 导航与常用入口

| 内容 | 入口 |
| --- | --- |
| 首次提交 | [第 0 章](#0-阅读说明与首次提交快速上手)、[本地配置](#2-本地-github-开发环境配置) |
| 平台差异与团队约定 | [第 1 章](#1-github-与-gitcode-的差异及迁移约定) |
| 管理员配置 | [第 3 章](#3-github-平台与仓库配置) |
| 日常开发、评审与合入 | [第 4 章](#4-日常开发与-pr-提交审查合入) |
| CI/CD 与 Jenkins 集成 | [第 5 章](#5-github-cicd-与现有测试系统集成) |
| 常见报错 | [第 6 章](#6-常见问题与处理步骤) |
| 清单、命令、模板与案例 | [附录 A–L](#附录检查清单模板与迁移案例) |
| GitHub 工作入口 | [PR 列表](https://github.com/mindspore-ai/hyper-parallel/pulls)、[Issue 列表](https://github.com/mindspore-ai/hyper-parallel/issues)、[Actions](https://github.com/mindspore-ai/hyper-parallel/actions) |
| 生效规则 | [仓库规则](https://github.com/mindspore-ai/hyper-parallel/rules)、[master 规则](https://github.com/mindspore-ai/hyper-parallel/rules/23637929)、[发布分支规则](https://github.com/mindspore-ai/hyper-parallel/rules/23641913) |
| 代码归属 | [master CODEOWNERS](https://github.com/mindspore-ai/hyper-parallel/blob/master/CODEOWNERS)、[r1.0.0 CODEOWNERS](https://github.com/mindspore-ai/hyper-parallel/blob/r1.0.0/CODEOWNERS) |

## 0. 阅读说明与首次提交快速上手

### 0.1 适用范围、维护负责人和更新日期

GitHub 提交按本文操作；框架、硬件、代码风格和测试实现以目标分支的 AGENTS.md、规则文件及构建文档为准。不同分支可能处于不同架构阶段，不从另一分支复制已删除模块的运行方法。

本文的维护职责归仓库维护团队；涉及规则、身份和 CI 的章节，由对应管理员复核。维护团队入口为 [HyperParallel Maintainers](https://github.com/orgs/mindspore-ai/teams/hyperparallel-maintainers)，仅有相应权限的组织成员能查看。人员分工见附录 K。

发布新版本时，应重新核对规则、模板和 CI 入口，更新日期与变更记录。本版本没有更改远端权限或门禁设置。

### 0.2 四类角色的职责

| 角色 | 主要职责 | 应交付的结果 |
| --- | --- | --- |
| 开发者 / PR 作者 | 正确身份提交，编写和验证代码，填写描述，处理意见 | 可审查、可复现、提交签名有效的 PR |
| 评审人 / Code Owner | 检查自己负责的代码及接口影响，提交正式评审 | Approve 或有依据的 Request changes |
| 仓库管理员 | 维护团队、代码归属、分支规则、发布及例外权限 | 可验证、可追溯的协作规则 |
| CI 维护者 | 维护触发、执行、结果回传、执行机与服务凭证 | 与正确提交对应的可信检查结论 |

被分配为 Assignee 不等于已请求 Review；成为组织成员不等于已拥有仓库 Write。开源贡献者可以通过 Fork 提交 PR，并不需要先取得上游写权限；承担有效 Code Owner 职责则需要满足权限条件。

### 0.3 首次提交路线

1. 登录正确的 GitHub 账号；需要加入团队的成员接受组织邀请。
2. 完成第 2 章的 SSH、提交邮箱、签名和 CLA 配置。
3. 在自己的 Fork 中创建功能分支，从目标分支的最新状态开始。
4. 完成小范围修改，执行相关本地检查和测试。
5. 推送带签名的提交，向正确的上游目标分支创建 Draft PR。
6. 补全描述和 Issue，自检通过后标为 Ready，并确认 CI 针对当前提交运行。
7. 请求相应 Code Owner；处理意见后由对方正式批准。
8. 合并人核对审批、签名、CI 和冲突状态，按规则合入。
9. 同步上游、清理已完成的个人分支。

首次练习宜使用有价值的小型文档或测试修复；不要以空改动触发昂贵的硬件任务。

### 0.4 每一步的成功标志

| 环节 | 成功标志 | 不满足时联系 |
| --- | --- | --- |
| 登录与推送 | SSH 显示预期用户名；能推送自己的 Fork | 开发者自行排查，必要时联系组织管理员 |
| 提交身份 | Commits 中作者能关联本人；待合入提交显示 Verified | 开发者 |
| CLA | 当前提交的约定 CLA 检查成功 | CLA / CI 维护者 |
| 描述检查 | 已填写真实内容，校验器通过 | 作者；模板本身不兼容时找 CI 维护者 |
| 评审 | 对应模块有非作者的有效批准，讨论已处理 | Code Owner |
| CI | 当前 SHA 对应的必需检查有终态和可访问日志 | CI 维护者 |
| 合入 | PR 标记 Merged，目标分支包含预期结果 | 合并人 |

**迁移现状：**截至本版本，必需检查列表仍为空、默认模板和部分 CODEOWNERS 条目仍待修复。PR 页面没有红色 Checks 不代表测试通过；过渡期合并人仍需人工核对完整证据，并推动第 3、5 章的改造。

## 1. GitHub 与 GitCode 的差异及迁移约定

### 1.1 组织、团队、仓库、Fork 的关系

本项目组织为 mindspore-ai，主仓库为 hyper-parallel。团队在组织内管理成员，再针对具体仓库获得权限。个人 Fork 是独立仓库，拥有自己的分支与设置；它不会自动获得上游的成员授权。

本手册使用 `origin` 表示个人 GitHub Fork，`upstream` 表示 mindspore-ai/hyper-parallel。创建 PR 时：base 是准备合入的上游分支，head 是个人 Fork 中的开发分支。参考 [GitHub Fork 指南](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/fork-a-repo)。

### 1.2 账号、角色与权限对应关系

迁移时分别核实“人是谁”“GitHub 用户名是什么”“对这个仓库拥有什么权限”。GitCode 用户名、显示昵称、Git author.name 和 GitHub login 不能互换使用。

| GitHub 仓库角色 | 本手册中的适用场景 |
| --- | --- |
| Read | 浏览、克隆、通过 Fork 贡献 |
| Triage | 主要处理 Issue / PR 分流 |
| Write | 日常维护代码、具备满足 Code Owner 条件的基础权限 |
| Maintain | 按职责维护仓库协作设置 |
| Admin | 管理访问权限和规则等管理事项 |

具体能力以 [GitHub 仓库角色](https://docs.github.com/en/organizations/managing-user-access-to-your-organizations-repositories/managing-repository-roles/repository-roles-for-an-organization)为准。身份迁移清单只登记工作所需信息，不收集密码、私钥和访问令牌。

### 1.3 GitHub、GitCode、Jenkins、同步机器人的职责

**已观察到的迁移链路：**GitHub PR 与 GitCode 合并请求有 Paired 对应关系，外部 Jenkins 执行测试，机器人将结果回写 GitHub。当前没有注册的 GitHub Actions 工作流；这不妨碍继续使用 Jenkins。

~~~mermaid
flowchart LR
    A["个人 GitHub Fork"] --> B["GitHub PR"]
    B --> C["同步服务 / GitCode 对应请求"]
    C --> D["Jenkins 与硬件测试"]
    D --> E["机器人评论及标签"]
    E --> B
    D -. "待完善：绑定提交的检查状态" .-> F["GitHub Checks / Statuses"]
    F --> B
~~~

每项工作应有一个开发入口和一个对应 PR。看到 sync-managed 或 Paired 标记时，先确认来源及同步方向，再决定在哪端更新。避免两端同时修改同一分支或重复建立竞争 PR。主库、同步方向、允许写入的机器人和停止同步条件由平台维护者登记，不能仅凭仓库更新时间推断。

### 1.4 用户名、提交邮箱与 CLA 身份映射

一个人至少需要确认：

- GitHub login：用于请求评审、CODEOWNERS 和授权。
- 提交邮箱：用于提交与账号关联；应出现在本人的 GitHub 邮箱设置中。
- 签名公钥：用于验证该提交的密码学签名。
- CLA 签署身份及邮箱：用于贡献协议检查。

这四项分别验收。GitCode 上已签 CLA，不代表新平台采用的另一 CLA 服务自动识别；出现双重结果时由维护者确认协议和身份映射。

### 1.5 Issue、PR 编号和链接的迁移规则

GitHub 的 `#123` 指向当前 GitHub 仓库中的编号，不能直接当成 GitCode 的 `!123` 或 Issue #123。迁移记录至少包含源 URL、目标 URL、对应关系、当前状态。

在说明中为跨平台对象填写完整 URL；GitHub 原生关联使用真实 GitHub Issue。当前中文模板的历史注释与跨平台链接支持存在待核实差异，必须以 `/check-pr` 的实际反馈验收，不能捏造同号 Issue 来通过检查。参考 [GitHub Issue 与 PR 关联](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue)。

### 1.6 评论命令、标签、正式评审、Checks 的区别

| 对象 | 表达什么 | 操作入口 |
| --- | --- | --- |
| `/check-pr`、`/retest` | 本项目机器人约定的命令 | PR Conversation 评论 |
| `ci-running`、`ci-pipeline-passed` | 机器人维护的标签 | PR 侧栏 |
| Review request | 请某人进行评审 | Reviewers |
| Approve / Request changes | GitHub 正式评审结论 | Files changed → Review changes |
| Checks / Statuses | 关联提交的机器检查结果 | PR Checks / 合并区域 |
| Rulesets | 哪些条件必须满足才能更新分支 | 仓库 Rules / Settings |

机器人命令不是 GitHub 内置命令。评论“同意”或 `/lgtm` 不能默认替代 Approve；CI 成功也不替代人工审查。具体命令是否生效以当前机器人支持和响应为准。

### 1.7 团队统一约定

个人开发默认采用 Fork 工作流；普通功能面向 master，发布修复根据维护计划面向 r1.0.0 等实际维护分支。每个 PR 先确认目标分支，避免把整批主线差异带入发布分支。

现行受保护分支仅允许 Merge commit，要求签名和独立审批；PR 前整理个人 WIP 提交与“合并时选择 Squash”是两项不同操作。共享主分支、发布分支不执行强推；个人分支重写历史前要确认没有他人协作依赖。代理创建的分支使用仓库要求的 `codex/` 前缀，普通开发按项目分支命名规范执行。

## 2. 本地 GitHub 开发环境配置

### 2.1 Git、GitHub CLI 和开发工具

安装 [Git](https://git-scm.com/downloads) 和 [GitHub CLI](https://cli.github.com/)。SSH 提交签名需要支持该能力的 Git，建议使用 Git 2.34 或更新版本。终端先检查：

~~~bash
git --version
gh --version
conda env list
~~~

已有 py312 环境时直接使用。尚未创建时，根据团队环境规范创建 Python 3.12 环境：

~~~bash
conda create -n py312 python=3.12
~~~

后续 Python 命令使用 `conda run -n py312`，避免误用系统 Python。硬件驱动、CANN、CUDA 和框架组合按目标分支的[安装指南](../installation.zh.md)安装。

### 2.2 SSH / HTTPS 认证与多账号配置

在[个人 SSH 和 GPG 设置](https://github.com/settings/keys)登记自己的公钥。已有可用密钥可复用；需要新密钥时先确认目标文件不存在，避免覆盖：

~~~bash
ssh-keygen -t ed25519 -C "YOUR_VERIFIED_EMAIL" -f ~/.ssh/id_ed25519_github_hp
ssh-add ~/.ssh/id_ed25519_github_hp
~~~

将生成的 `.pub` 公钥添加为 Authentication key，私钥留在本机。然后验证：

~~~bash
ssh -T git@github.com
gh auth login --hostname github.com --git-protocol ssh --web
gh auth status --hostname github.com
~~~

以 SSH 提示中的用户名为准。GitHub 不提供 SSH shell；看到成功认证提示后命令退出码不为零不一定是失败。CLI 登录与 Git SSH 所选身份分别检查。HTTPS 用户用凭证管理器或 CLI 登录，不把令牌写进仓库 URL。

多账号可在 `~/.ssh/config` 添加独立 Host 别名，并把对应仓库 remote 指向该别名；例如：

~~~sshconfig
Host github-hp
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github_hp
    IdentitiesOnly yes
~~~

随后用 `ssh -T git@github-hp` 验证；remote 主机名也应使用 `github-hp`。参考 [SSH 配置与认证](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)。

### 2.3 提交姓名、邮箱及账号关联

进入仓库后设置本仓身份；占位符应替换为本人信息：

~~~bash
git config user.name "YOUR_NAME"
git config user.email "YOUR_VERIFIED_EMAIL"
git config --show-origin --get user.name
git config --show-origin --get user.email
~~~

到[邮箱设置](https://github.com/settings/emails)确认邮箱已验证。使用 GitHub 提供的 noreply 地址时，从页面复制实际地址，并确认团队 CLA 服务是否支持。不要仅修改 name 来修复错误的邮箱关联。

验收：新提交能链接到本人 GitHub 账号；`git show --no-patch --format=fuller HEAD` 显示预期作者和提交者。参考 [设置提交邮箱](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address)。

### 2.4 提交签名与 Verified 验证

以下为 SSH 签名方案。在[密钥设置](https://github.com/settings/keys)中将签名公钥添加为 **Signing key**；即使同一公钥已作为 Authentication key，也要按官方步骤登记签名用途。再配置本仓：

~~~bash
git config gpg.format ssh
git config user.signingkey ~/.ssh/id_ed25519_github_hp.pub
git config commit.gpgsign true
git config --get gpg.format
git config --get user.signingkey
git config --get commit.gpgsign
~~~

提交时使用 `git commit -S`。采用已有 GPG 方案的成员按[官方签名配置](https://docs.github.com/en/authentication/managing-commit-signature-verification/telling-git-about-your-signing-key)登记本人公钥，无须切换到 SSH。

验收以推送后 Commits 页面每个待合入提交的 **Verified** 为准。管理员可只读查询：

~~~bash
gh api --paginate repos/mindspore-ai/hyper-parallel/pulls/PR_NUMBER/commits \
  --jq '.[] | {sha, verified: .commit.verification.verified, reason: .commit.verification.reason}'
~~~

PR_NUMBER 需替换为真实编号。GitHub 上的验证与本地 SSH allowedSignersFile 信任配置不同；本地验证报错时同时检查信任配置。CLA、`Signed-off-by` 文本与密码学签名是不同事项。参考 [添加签名密钥](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)。

### 2.5 CLA 签署与校验

首次 PR 中打开可信 CLA 检查提供的链接，以本人账号和规定的身份完成签署。确认提交邮箱与签署记录一致，再按机器人提示重新检查；本项目已有 `/check-cla` 回写提示，但它是项目服务命令。

迁移期可能同时看到 GitHub CLAassistant 和 GitCode 回写。分别记录检查来源、提交 SHA 和失败邮箱，交 CLA 维护者确认；不要把一个服务的成功当成另一个服务已成功，也不要借用他人身份提交。验收是当前提交在团队确认的判定来源中通过。

### 2.6 Fork、克隆及 origin / upstream 配置

打开[上游仓库](https://github.com/mindspore-ai/hyper-parallel)，点击 Fork，在自己的账号下创建副本。以下仅用于尚未克隆的新目录：

~~~bash
git clone git@github.com:YOUR_GITHUB_LOGIN/hyper-parallel.git
cd hyper-parallel
git remote add upstream git@github.com:mindspore-ai/hyper-parallel.git
git fetch upstream --prune
git remote -v
~~~

使用 SSH Host 别名时替换主机名。验收：origin 指向自己的 GitHub Fork，upstream 指向组织仓库。GitHub CLI 的 PR 命令显式使用 `--repo mindspore-ai/hyper-parallel`，避免默认仓库解析到 Fork。

### 2.7 项目环境、可编辑安装和本地测试

按[项目安装说明](../installation.zh.md)为 py312 安装匹配的框架及硬件依赖，然后在本工作树根目录执行：

~~~bash
conda run -n py312 python -m pip install -e .
conda run -n py312 python -m pip show hyper_parallel
conda run -n py312 python -c "import hyper_parallel; print(hyper_parallel.__file__)"
~~~

导入路径必须指向当前工作树。切换 worktree 或发现旧路径后，重新进行可编辑安装；只设置 PYTHONPATH 不能替代这项验收。

单元测试示例：

~~~bash
conda run -n py312 python -m pytest -vs tests/ut
~~~

先运行受影响模块，再按提交门禁执行所需测试集合。没有 NPU/GPU 时，应如实记录已完成的逻辑测试和待 CI 验证内容。对于通过 torchrun_case 等启动子进程的 ST，按用例入口运行 pytest，由项目启动器管理进程；不要给整套启动器再套一层 torchrun。遵循[测试规则](../../.agent/rules/testing.md)，不把 GitHub 手册写成另一套测试实现规范。

### 2.8 新成员环境自检清单

在首次 PR 描述中记录：Git/CLI 版本、目标分支、Python 环境、导入路径、相关测试命令与结果。公开材料不要附令牌、私钥、内部凭证或完整环境变量。

附录 A 给出可勾选清单。至少完成一次真实提交的 Verified 检查；“签名配置已经写入 Git”还不能证明 GitHub 接受该密钥。

### 2.9 从现有 GitCode 工作目录迁移

先执行 `git status` 和 `git remote -v`，确认本地未提交工作已妥善保留。已有 origin 时，不再重复执行 remote add：

~~~bash
git remote rename origin gitcode-origin
git remote add origin git@github.com:YOUR_GITHUB_LOGIN/hyper-parallel.git
~~~

如果原有 upstream 指向 GitCode，先改名为 gitcode-upstream，再添加 GitHub upstream；如果已指向 GitHub，则保留。名称已被占用时自行选择可辨识名称，不覆盖已有 remote。

~~~bash
git remote rename upstream gitcode-upstream
git remote add upstream git@github.com:mindspore-ai/hyper-parallel.git
git fetch origin --prune
git fetch upstream --prune
git remote -v
~~~

以上第二组命令只适用于“确有旧 GitCode upstream”的情况。随后比较待迁移分支与新目标分支的差异、提交数和签名；分叉较大时，从新的目标分支创建分支，再选择性迁移本人的业务提交。GitCode 与 GitHub 的 SHA 关系应依据实际同步方式核对，不推测相同内容一定拥有相同 SHA。

## 3. GitHub 平台与仓库配置

适用角色：仓库管理员、组织管理员、CI 维护者。修改远端设置前记录现值、目标值、影响范围和验证办法；本章中的建议不自动授予执行变更的权限。

### 3.1 组织基础权限、团队授权与外部贡献者

现行组织基础权限为 Read。针对 hyper-parallel 授权应在本仓完成，无须把整个组织的基础权限提升到 Write。

操作入口：仓库 Settings → Collaborators & teams → Add teams / Add people。给日常维护团队选择 Write；人员接受邀请后检查实际生效角色。外部贡献者可通过 Fork 工作；需要作为长期 Code Owner 时，由管理员明确其授权方式。

可用以下只读查询复核某个真实账号：

~~~bash
gh api repos/mindspore-ai/hyper-parallel/collaborators/LOGIN/permission \
  --jq '{permission, role_name, login: .user.login}'
~~~

LOGIN 替换为 GitHub 用户名。没有管理权限时部分接口会拒绝，不能把鉴权失败解释为对方没有权限。参考[访问权限管理](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-teams-and-people-with-access-to-your-repository)。

### 3.2 模块维护团队与备用评审人

现有 hyperparallel-maintainers 团队已获得仓库 Write。**建议落地：**为需独立审批的模块明确至少两名合格维护者，并设置离岗、休假及跨模块变更的替补流程。

创建团队时，在组织 Teams → New team 选择适当的 Visible 可见性，加入已接受邀请的组织成员，再为团队添加具体仓库权限。团队成员的个人权限不能代替“作为 Code Owner 的团队自身”的权限要求。参考[创建团队](https://docs.github.com/en/organizations/organizing-members-into-teams/creating-a-team)。

每次增删维护者后，检查其负责路径能否由非作者完成审批。不要把一个只能由作者自己审批的路径作为配置完成的标志。

### 3.3 CODEOWNERS 配置与有效性检查

GitHub 从 PR 的目标分支读取 CODEOWNERS。文件位置按 .github/、仓库根目录、docs/ 顺序选择第一个；同一路径使用最后一条匹配规则。匹配更具体的规则后，不会把前面的默认负责人自动追加进来。

示意配置中的团队必须先创建并获得该仓库权限；下面的模块团队名仅为示例：

~~~text
* @mindspore-ai/hyperparallel-maintainers
/hyper_parallel/compile/ @mindspore-ai/EXAMPLE-COMPILE-TEAM
/tests/ut/compile/ @mindspore-ai/EXAMPLE-COMPILE-TEAM
/CODEOWNERS @mindspore-ai/hyperparallel-maintainers
/.github/ @mindspore-ai/hyperparallel-maintainers
~~~

同一规则上的多名 Owner 通常选一人批准即可；PR 涉及不同规则时，审批要覆盖相应文件。普通 Reviewer 可以提出意见，但只有符合要求的审批才能满足合并规则。参考 [CODEOWNERS 官方规则](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)。

检查两个目标分支：

~~~bash
gh api 'repos/mindspore-ai/hyper-parallel/codeowners/errors?ref=master'
gh api 'repos/mindspore-ai/hyper-parallel/codeowners/errors?ref=r1.0.0'
~~~

**当前待修复：**两个分支仍分别有 34、37 处错误。Unknown owner 可能是账号不存在或权限不足；管理员应先核实身份，不能根据近似拼写自动换人。验收要求错误列表为空，并用真实文件路径创建小型 PR，确认能请求预期维护者。给现有 PR 修好权限后还应检查是否需要手动补发评审请求。

### 3.4 分支保护策略

| 分支用途 | 本版本实际情况 | 操作要求 |
| --- | --- | --- |
| master | 受 protect-master 规则保护 | 功能变更通过 PR 合入 |
| r1.0.0 等匹配的发布分支 | 受发布规则保护 | 选择相应基线，按版本计划合入修复 |
| rl、trainer_dev 等共享分支 | 当前未被上述两个规则覆盖 | 管理员按用途决定保护；不能默认为已受保护 |
| 个人 Fork 功能分支 | 由个人管理 | 不作为上游发布或质量验收的替代入口 |

**建议落地：**共享集成分支至少明确谁可写入、是否要求 PR、何时同步主线。发布规则包含限制创建分支，发布负责人需事先检查创建权限，不能等到发版当天才临时放开整个仓库。

### 3.5 Rulesets：审批、签名、讨论与必需检查

现行 master / 发布分支规则包括：至少一个批准、Code Owner 审批、最近一次可评审推送由其他人批准、解决讨论、失效审批处理、提交签名、防删除和防强推；受保护分支只允许 Merge。

**当前缺口：**两个规则都存在“Require status checks”配置，但检查名称列表为空。因此该项尚未强制任何实际 CI 测试。

管理员在 Settings → Rules → Rulesets 中检查目标分支范围、执行状态、具体检查名称和旁路用户。规则集可能叠加；旧 branch protection API 返回 404 不代表没有 Rulesets。参考[创建规则集](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)。

**建议落地：**按第 5 章先建立真实检查，再选入必需检查列表，并绑定预期来源。不要填入没有任何服务会回传的检查名，否则所有 PR 都可能一直等待。

### 3.6 PR / Issue 模板与校验器

维护入口：[默认模板](https://github.com/mindspore-ai/hyper-parallel/blob/master/.github/PULL_REQUEST_TEMPLATE.md)、[中文模板](https://github.com/mindspore-ai/hyper-parallel/blob/master/.github/PULL_REQUEST_TEMPLATE.zh-CN.md)、[Issue 模板](https://github.com/mindspore-ai/hyper-parallel/tree/master/.github/ISSUE_TEMPLATE)。

当前默认模板使用旧英文检查项，机器人却期待另一套字段。管理员应同时更新默认模板、中英文版本和解析器，保留一个稳定的字段契约。修改模板文件不自动更新已经打开的 PR 描述。

验收：创建一个使用默认模板的新 PR，填完整的真实内容，运行描述检查；再验证缺失 Issue、无测试说明等情况能得到准确提示。附录 H 提供起草示例，是否通过现网检查仍以实际反馈为准。

### 3.7 GitHub Apps、Webhook、令牌和机器人权限

优先为自动化建立独立的服务身份或 GitHub App，按实际需要授予仓库访问、检查回传及评论权限。评审、写代码、写检查、管理设置是不同能力，避免全部交给个人长期令牌。

Webhook 配置记录事件类型、处理服务负责人和最近投递结果；服务验证签名并处理重复投递。新增机器人后，应验收“创建 PR、推送个人来源分支、回传检查、按规则合入”等分别需要的权限。

Actions 当前默认令牌为 Write，且允许工作流创建/批准 PR。**建议落地：**启用工作流前改为默认 Read，在需要的 job 显式提升；未经业务确认不保留自动审批能力。参考[Actions 安全指南](https://docs.github.com/en/actions/reference/security/secure-use)。

### 3.8 合并方式、自动合并和分支清理

仓库 General 中启用了 Merge / Squash / Rebase，但目标规则只允许 Merge，所以正常操作应选 Create a merge commit。不要因为全局开关允许就假定目标分支也允许。

自动合并和合并后删除分支当前未启用。待检查稳定后可以评估自动合并；自动合并只等待条件满足，不替代审批或扩大权限。个人 Fork 分支由作者在确认合并及无后续依赖后清理。

### 3.9 管理员旁路、紧急操作与审计

当前两个规则集只有 liuchongming74 被配置为 always bypass。它允许在规则未满足时更新受保护分支，不代表其他检查已经通过。

当前撤销评审的权限也受到规则限制，允许的用户为 liuchongming74。评审存在争议或原评审者无法继续处理时，由有权限的维护者核查并记录原因；撤销评审不能代替修复尚未解决的问题。

**建议落地：**紧急操作登记原因、影响范围、复核人、原始 SHA、目标 SHA、验证证据和恢复方案；事后补充失败条件的处理结论。日常合并不把“请管理员绕过”作为标准步骤。主分支问题优先通过修复/回退 PR 处理；本手册不提供强推共享分支的操作命令。

### 3.10 配置修改后的验收方法

管理员建立一组可识别的测试 PR，使用无旁路权限的路径验证：

1. 未签名提交被阻止；正确签名提交被接受。
2. Owner 配置有效，作者之外的人能完成审批。
3. 普通评论不满足批准，未解决讨论仍提示处理。
4. 必需 CI 失败时不能合并，成功后状态正确。
5. 新 head 有自己的检查，旧 head 的结果不覆盖它。
6. 同步机器人能完成被允许的操作，越权更新被拒。
7. master 与发布分支分别通过上述验收。

测试只在约定的测试分支和测试 PR 上执行，避免为验收向主线合入无意义内容。结果记录到附录 E。

## 4. 日常开发与 PR 提交、审查、合入

### 4.1 从正确的目标分支开始

先确定改动应进入 `master` 还是指定发布分支，再从对应的最新上游提交创建个人工作分支。以下例子针对 `master`，执行前确认当前工作区没有未提交内容：

~~~bash
git status --short
git fetch upstream
git switch -c fix/YOUR_TOPIC upstream/master
~~~

`YOUR_TOPIC` 替换为简短主题。同一个功能分支只解决一件可以独立审查的事情；不要直接在本地 `master` 累积开发提交。修复发布版本时，将示例中的 `master` 替换为已经确认的目标分支，避免将主线新功能带入发布分支。

### 4.2 同步上游与解决冲突

尚未发布或只有本人使用的功能分支，可以通过 rebase 整理到最新上游。它会重写提交，签名也需要重新生成：

~~~bash
git fetch upstream
git rebase --gpg-sign upstream/master
~~~

出现冲突后，按文件理解并修改冲突内容，添加实际解决的文件，再执行 `git rebase --continue`；无法确认解决方案时执行 `git rebase --abort`，保留原有状态。完成后重跑受影响测试并验证 GitHub 上的提交签名。

已经推送的个人来源分支，重写后的推送方法见 6.4。多人共同使用的分支不要自行 rebase 或强推，先与维护者确定同步方式。合并上游时还需确认引入的历史提交符合目标签名规则；仅给最新 merge commit 签名不能修复其他未签名提交。

参考：[同步 Fork](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/syncing-a-fork)、[Git rebase](https://git-scm.com/docs/git-rebase)。

### 4.3 提交说明、整理提交和签名

采用 Conventional Commits，主题约 80 个字符以内，说明改动行为，例如：

~~~text
fix: preserve shard metadata during redistribution
docs: clarify GitHub contributor setup
~~~

提交前只暂存本次需要的文件，检查暂存区后创建签名提交：

~~~bash
git add PATH_TO_CHANGED_FILE
git diff --cached
git commit -S -m "fix: describe the actual change"
~~~

`PATH_TO_CHANGED_FILE` 是实际文件路径。不要无检查地将本地配置、访问令牌、日志或大型生成文件加入提交。

打开 PR 前整理无意义的 WIP 提交；已进入评审阶段后，尽量让评审者能够追踪增量修改。如果确实需要合并提交或重写历史，应说明原因并重新获取必要的评审、签名和 CI 结果。本地整理为一个提交与 GitHub 的 Squash merge 是两件事，当前受保护分支仍按 Merge 方式合入。

### 4.4 提交 PR 前的自检

提交前至少检查：

- 改动范围与目标分支正确，没有带入无关提交。
- 遵循目标分支的 `AGENTS.md` 和对应目录规则；Python 文件、注释和接口满足项目约定。
- 新增或调整了必要测试，记录实际执行的命令、环境和结果。
- 所有将被引入目标分支的提交满足签名要求，提交邮箱可以映射到正确账号。
- Issue、设计说明和兼容性影响有真实依据；未执行测试要写明原因。
- PR 描述不含模板占位符、无效 `Fixes #` 或未经核实的“全部通过”。

建议先检查补丁，再运行当前改动需要的测试：

~~~bash
git diff --check
git diff upstream/master...HEAD
git log --oneline upstream/master..HEAD
conda run -n py312 python -m pytest -vs tests/ut
~~~

完整 UT 可能需要当前平台的依赖；无法在本机执行时，写清缺失条件和交由哪个 CI 环境验证。不能把“命令未运行”记录为“测试通过”。

### 4.5 创建 PR：标题、关联 Issue、描述和验证证据

先推送到个人 Fork：

~~~bash
git push -u origin HEAD
~~~

随后在[新建 PR 页面](https://github.com/mindspore-ai/hyper-parallel/compare)核对 base repository、base branch、head repository 和 compare branch，也可以用 CLI：

~~~bash
gh pr create --repo mindspore-ai/hyper-parallel \
  --base master --head YOUR_GITHUB_LOGIN:YOUR_BRANCH \
  --draft --title "fix: describe the actual change" \
  --body-file /ABSOLUTE/PATH/pr-body.md
~~~

替换账号、分支和描述文件路径。正文至少回答：为什么改、改了什么、影响哪些接口或平台、如何验证、有哪些尚未完成的内容。附录 H 提供结构示例；现网模板与检查器仍待统一，不保证直接粘贴即可通过。

关联 GitHub Issue 时使用真实编号，例如 `Fixes #实际编号`；跨仓库使用完整仓库引用。GitCode Issue 编号应标明平台并附实际链接，不能把同号的 GitHub Issue 当成同一事项。自动关闭关键字仅对目标为默认分支的 PR 生效；面向发布分支时不能依赖这些关键字自动关联或关闭 Issue，应另外记录关联并核对状态。

参考：[gh pr create](https://cli.github.com/manual/gh_pr_create)、[关联 PR 与 Issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue)。

### 4.6 Draft 与 Ready for review

尚未完成实现、验证或自检时使用 Draft。标题中的 WIP 只是文字，不会把 PR 转成 Draft。

准备好后在 PR 页面选择 Ready for review，或执行：

~~~bash
gh pr ready PR_NUMBER --repo mindspore-ai/hyper-parallel
~~~

Draft 阶段不会自动请求 Code Owner 评审；转为 Ready 后才触发相应请求。CI 是否在 Draft 上运行由具体触发配置决定，不要把“Draft 未跑 CI”直接判断为故障。参考 [Draft PR 的行为](https://docs.github.com/en/pull-requests/reference/pull-requests#draft-pull-requests)。

### 4.7 请求评审与多个 Owner 的处理

Code Owner 来自目标分支的有效 CODEOWNERS。作者先检查 Files changed 对应的模块归属，再确认 Reviewers 中是否有适合的、具有权限的人员。

可以手动请求某位维护者：

~~~bash
gh pr edit PR_NUMBER --repo mindspore-ai/hyper-parallel \
  --add-reviewer GITHUB_LOGIN
~~~

手动请求不会修复无效 CODEOWNERS，也不会把普通评审者变成 Code Owner。跨模块 PR 可能需要分别覆盖不同文件集合；同一规则列出多位有效 Owner 时，通常其中一位批准即可满足该规则，不应机械地要求列表中所有人都批准。实际还要同时满足审批数量、最近推送审批等规则。

作者不能通过批准自己的 PR 满足要求。若作者是某个文件集合唯一有效 Owner，由管理员补充真实的备用 Owner，并验证目标分支配置。

参考：[Code Owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)。

### 4.8 Comment、Approve 与 Request changes

| 评审操作 | 用途 | 对合并要求的影响 |
|---|---|---|
| Comment | 提问、建议或记录观察 | 不等于批准 |
| Approve | 已完成检查，认可当前改动 | 满足条件时计入审批 |
| Request changes | 存在需要修改后再检查的问题 | 可能阻止合并，需要后续处理 |

评审者检查实现、测试、平台影响和公开接口后，在 Finish your review 中明确选择结论。仅发送“LGTM”评论不等于提交 Approve。

CLI 示例分别为：

~~~bash
gh pr review PR_NUMBER --repo mindspore-ai/hyper-parallel \
  --comment --body "请补充此场景的测试结果。"
gh pr review PR_NUMBER --repo mindspore-ai/hyper-parallel \
  --approve --body "已检查实现和测试结果，认可当前改动。"
gh pr review PR_NUMBER --repo mindspore-ai/hyper-parallel \
  --request-changes --body "请先修复说明中的阻塞问题，再请求复审。"
~~~

这些命令会发布真实评审，应在完成相应判断后选择执行其中一种。不要为了消除红色提示提交没有依据的批准。

参考：[评审 PR](https://docs.github.com/en/pull-requests/how-tos/review-pull-requests/reviewing-proposed-changes-in-a-pull-request)、[gh pr review](https://cli.github.com/manual/gh_pr_review)。

### 4.9 回复意见、解决讨论与重新评审

每条意见给出处理结果：已修复并说明位置、补充解释，或与评审者达成保留现状的共识。Resolve conversation 表示该讨论已处理，不等于获得 Approve。

当前规则会撤销符合条件的过期审批，并要求最近一次可评审推送由其他人批准。因此，推送修改后要重新检查：

1. 原有审批是否仍有效。
2. 当前 Owner 审批是否覆盖受影响文件。
3. 最近推送者之外是否有人完成所需批准。
4. 所有必须解决的讨论是否已处理。
5. 最新 SHA 的 CI 和签名是否通过。

评审者自己向作者分支推送修复后，也要注意“最近推送者不能批准自己最新推送”的限制，安排其他合格人员复核。

### 4.10 触发 CI、重测和读取结果

优先以 PR 的 Checks 区域和 CI 的实际构建详情为入口。当前仓库还存在 Jenkins / GitCode 通过评论回报结果的情况，应核对评论中的提交 SHA、构建时间、构建编号和链接。

~~~bash
gh pr checks PR_NUMBER --repo mindspore-ai/hyper-parallel
gh pr view PR_NUMBER --repo mindspore-ai/hyper-parallel \
  --json headRefOid,statusCheckRollup,url
~~~

没有 Checks 结果不等于检查通过。若服务依赖评论触发重测，只使用 CI 维护者公布的、已在 GitHub 路径验证过的指令；GitCode 的 `/retest` 等旧指令不能直接假定在 GitHub 同样有效。

重测应说明原因：代码已修复、基础设施恢复或明确的偶发失败。避免反复点击重跑而不阅读失败日志。

### 4.11 合入前检查、合入与清理

合入者按附录 D 检查当前 PR，特别核对当前 head、有效审批、签名、未解决讨论、冲突和真实 CI 结果。当前 required checks 为空，因此还需要人工核验已约定的外部测试证据。

普通合入使用页面上的 Create a merge commit。CLI 可绑定已经检查的 head，防止确认后又有新提交：

~~~bash
gh pr view PR_NUMBER --repo mindspore-ai/hyper-parallel --json headRefOid
gh pr merge PR_NUMBER --repo mindspore-ai/hyper-parallel \
  --merge --match-head-commit FULL_REVIEWED_HEAD_SHA
~~~

`FULL_REVIEWED_HEAD_SHA` 必须是刚刚核对过的完整 SHA。若期间分支更新，重新检查后再合入。禁止将管理员旁路作为解决日常评审或 CI 阻塞的替代步骤。

合入后确认目标分支包含预期结果、相关 Issue 状态正确，随后清理没有后续依赖的个人分支：

~~~bash
git fetch upstream
git switch master
git merge --ff-only upstream/master
git branch -d YOUR_BRANCH
git push origin --delete YOUR_BRANCH
~~~

本地 `master` 不存在时可以先从 `upstream/master` 创建；如果 `--ff-only` 失败或本地分支删除被拒，先检查原因，不要直接使用强制删除。

参考：[gh pr merge](https://cli.github.com/manual/gh_pr_merge)。

### 4.12 大型重构、依赖 PR 与多人并行开发

大型改动按可独立验证的步骤拆分：行为保持的准备性重构、核心功能、调用方迁移、清理。每个 PR 说明前置依赖和独立验证范围。

依赖另一个未合入 PR 时，在正文明确记录依赖链接和临时 base。前置 PR 合入后，更新目标分支并重新核对 diff、提交集合、签名和 CI；不能把此前对临时 base 的测试当成最终目标上的验证。

多人开发同一功能时，指定集成人和模块负责人，约定分支写入权限。共享开发分支也需要明确保护措施；当前主线和发布规则不自动覆盖全部开发分支。涉及公共接口、并行策略或多平台行为的方案应在实现前完成相应设计讨论。

### 4.13 发布分支修复、Backport 与回退

一般先修复主线，再将经过验证的最小改动带到发布分支；线上紧急修复如需先进入发布分支，应补建主线跟进事项。Backport PR 应链接原 PR，并说明是否有适配差异。

从发布分支创建个人分支后，可对已确认的普通提交执行：

~~~bash
git cherry-pick -x -S ORIGINAL_COMMIT_SHA
~~~

`-x` 记录来源，`-S` 为新提交签名。对 merge commit 不能直接套用普通提交命令，需要先确定正确的主线父提交和应提取的改动，由维护者检查后操作。Cherry-pick 产生的新 SHA 要重新运行对应发布分支测试。

回退也通过独立 PR 进行，说明回退原因、受影响版本和后续修复计划。合并提交的回退需要明确主线父提交；不要通过强推改写共享主线历史来回退。

## 5. GitHub CI/CD 与现有测试系统集成

### 5.1 当前链路与目标链路

**已核实的现状：**仓库尚无 GitHub Actions 工作流；部分检查经 GitCode/Jenkins 执行，机器人回写 GitHub 评论和标签，原生状态以 CLA 为主。部分 PR 没有原生检查记录，目标分支 required checks 列表为空。

这意味着“外部 CI 在运行”和“GitHub 在合并前强制要求 CI 成功”还没有完整衔接。建议建立以下闭环：

~~~mermaid
flowchart LR
    A[GitHub PR / 新提交] --> B[校验事件与确定 SHA]
    B --> C[描述 / CLA / 静态检查]
    B --> D[UT]
    B --> E[按改动选择 GPU / NPU ST]
    C --> F[回写 GitHub Checks 或 Status]
    D --> F
    E --> F
    F --> G[目标分支规则校验]
    G --> H[人工审批后合入]
~~~

Jenkins 可以继续承担已有硬件测试；是否新增 Actions 由维护成本与运行需求决定。重点是每项检查有稳定的身份、准确的 SHA 和可阻止不合格合入的规则。

### 5.2 Actions、Jenkins 和同步机器人的职责

| 组件 | 建议职责 | 验收重点 |
|---|---|---|
| GitHub Actions | 适合托管运行的静态检查、元数据检查、部分 UT、流程编排 | 权限最小，Fork 行为明确，结果可追溯 |
| Jenkins / 硬件测试平台 | NPU/GPU、多卡 ST、性能与兼容性测试 | 资源隔离，记录硬件与软件版本，产出真实结果 |
| 同步机器人 | GitHub 与迁移期平台之间的事件/链接映射 | 保留来源信息、避免循环同步、遵守目标分支保护 |
| 检查回传服务 | 将 CI 结果关联到准确的仓库与 SHA | 可信身份、稳定名称、可靠状态转换 |
| 维护者 | 选择必需检查、处理测试例外、审批合入 | 不以机器人评论替代规则和人工判断 |

不要让同步机器人的“成功推送”成为测试成功的信号；同步完成、测试完成和允许合并应分别记录。

### 5.3 分层测试：快速反馈与硬件验证

| 层级 | 典型内容 | 建议触发范围 |
|---|---|---|
| 元数据与静态检查 | PR 描述、格式、许可证头、基础 lint | 所有 PR；文档和源码可采用不同规则 |
| 单元测试 | 无多卡环境的 UT | 与改动相关的模块；主线定期扩大覆盖 |
| 分布式功能测试 | DP/FSDP/TP/PP 等相关场景 | 涉及相应模块或公共基础设施 |
| 平台兼容性 | Ascend NPU、Nvidia GPU，适用框架版本 | 按模块当前支持范围选择 |
| 性能与稳定性 | 吞吐、显存、长时间运行、故障恢复 | 对性能敏感的改动和发布候选版本 |

“按路径跳过”要由明确的选择逻辑决定，保留选择原因和验证结果。公共调度、通信、内存管理等改动不能仅按文件位置缩小到过少测试。

本项目部分模块明确为 Torch-only，测试矩阵应依据目标分支的模块支持约定，不能为追求形式上的全覆盖要求不存在的后端实现。

### 5.4 描述检查、CLA 与签名分别验证

三者解决不同问题：

- PR 描述检查：是否说明目的、关联事项、测试和接口影响。
- CLA：贡献者是否完成项目要求的贡献协议，账号映射是否正确。
- 提交签名：提交是否具有可验证签名及身份关联。

一个通过不代表另外两个通过。建议分别呈现结果和修复说明，避免全部归到含义不清的“门禁失败”标签。模板规则变更应与检查服务同步发布，先在测试 PR 上确认默认模板能够通过。

### 5.5 外部 CI 如何回传 Checks / Commit Status

外部系统可以选择：

- Commit Status API：向准确的提交 SHA 写入 `pending / success / failure / error`，提供稳定的 context 和详情链接。
- Checks API：创建 check run，更新执行状态、结论、摘要和日志链接。写入 check run 通常通过具有相应权限的 GitHub App 完成，具体凭据支持以接口文档为准。

CI 回传至少携带仓库、PR、源 head SHA、实际测试 SHA、目标分支、构建 ID、开始/结束时间和详情 URL。创建任务后及时给出待执行状态，执行结束后必须写入终态；超时和回传失败要有补偿机制。

读取某个已确认 SHA 的结果：

~~~bash
gh api repos/mindspore-ai/hyper-parallel/commits/FULL_HEAD_SHA/status
gh api repos/mindspore-ai/hyper-parallel/commits/FULL_HEAD_SHA/check-runs
~~~

PR 评论可以提供中文摘要，但不能替代规则所需的机器可读检查。参考：[Commit Status API](https://docs.github.com/en/rest/commits/statuses)、[Checks API](https://docs.github.com/en/rest/checks/runs)。

### 5.6 检查名称、来源与 Required checks

以下是**拟议名称示例，当前未部署**：

| 示例名称 | 负责内容 | 合并规则的建议 |
|---|---|---|
| `pr/description` | 描述完整性与字段验证 | 模板一致后纳入 |
| `license/cla` | 唯一明确的 CLA 结果 | 确定权威来源后纳入 |
| `ci/lint` | 静态检查 | 稳定运行后纳入 |
| `ci/ut` | 适用的单元测试 | 纳入 |
| `ci/distributed` | 按矩阵选择的硬件测试汇总结论 | 按分支质量要求纳入 |

每个名称应有唯一含义，避免不同系统使用同名 context 回写不同范围的结果。规则中应选择预期的检查来源，在支持的配置中绑定对应 App，防止其他身份写入同名成功状态。

必需检查必须在预期的每个 PR 上产生结果。路径过滤造成整个工作流不启动时，必需状态可能长期等待；建议由稳定的汇总任务确认“哪些测试需要执行、执行是否完成、跳过是否符合政策”。不能无条件返回成功来解除等待。

参考：[Rulesets 支持的规则](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)。

### 5.7 绑定 head SHA、测试合并 SHA 与构建编号

PR head SHA 是作者提交的版本；测试合并 SHA 是平台或 CI 将该版本与某个目标分支版本合成的测试提交，两者可能不同。

每次构建记录：

| 字段 | 要求 |
|---|---|
| 来源 | 仓库完整名称、PR 编号、来源分支 |
| 源代码版本 | head SHA；必要时附 base SHA |
| 实际被测版本 | checkout 后实际 SHA，明确是否为测试合并提交 |
| 构建标识 | Jenkins job + build number 或等效唯一 ID |
| 结果 | 实际测试范围、结论、日志和报告 URL |

评论中的构建编号、报告目录和链接必须来自同一构建记录。不得推测或拼接不存在的 Jenkins 编号。

当 head 更新时，旧结果应清楚标为历史，不能继续展示为“当前通过”。目标分支变化后，是否重新验证由所选检查策略决定；对合并兼容性敏感的场景应重新测试新的组合。

### 5.8 触发时机与取消过期任务

基础触发应覆盖新 PR、来源分支更新、重新打开、转为 Ready；是否检查描述修改、目标分支变更，由对应检查类型决定。描述变化通常只需重跑元数据检查，不应无差别重跑全部多卡测试。

对同一个 PR 的旧版本可以取消仍在排队或执行的任务，保留最新版本。取消必须写出清晰的终态和原因，避免永久 pending。发布、长时间稳定性任务是否可取消应单独规定。

参考：[Actions 事件触发](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)。

### 5.9 Webhook 去重、重试与评论更新

Webhook 服务验证签名，快速接收事件后将耗时任务交给后台处理。用投递 ID 去重，并结合仓库、PR、head SHA、检查类型识别同一业务任务；合法的人工重跑需要独立的运行编号，不能被错误去重。

同一检查维护一条可更新的摘要评论，或使用 Check summary 展示状态。若必须新增评论，应只在重要状态变化时发送，避免每次重试重复发布相同内容。

失败重试使用退避和次数上限；持续失败形成可操作的告警。重复投递、乱序结果、新旧 head 交错和服务重启都应纳入验收。

参考：[Webhook 最佳实践](https://docs.github.com/en/webhooks/using-webhooks/best-practices-for-using-webhooks)。

### 5.10 Fork PR、运行器和凭据

Fork PR 中的代码属于待审查输入，不应自动获得发布凭据、生产凭据或可访问内网关键资源的权限。Actions 的 Fork 事件对令牌和 secrets 有特殊限制，首次贡献者也可能需要维护者批准运行。

如果使用 `pull_request_target`，必须理解它使用目标仓库上下文及权限；不要在带有敏感凭据的执行阶段检出并运行未经审查的 Fork 代码。Jenkins 和自建运行器同样需要隔离工作区、限制网络和凭据暴露，并清理残留进程及文件。

发布凭据只在受控发布流程中使用；Actions 默认令牌建议 Read，具体 job 按需授权。第三方 Action 的允许范围和固定版本策略由管理员明确。

参考：[Actions 安全指南](https://docs.github.com/en/actions/reference/security/secure-use)、[批准 Fork 工作流运行](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/approve-runs-from-forks)。

### 5.11 日志、覆盖率、测试报告与故障定位

CI 报告至少包含失败阶段、首个有用错误、执行命令、测试范围、环境版本、SHA 和构建链接。多卡故障应保留各 rank 的必要日志，避免只看到主进程最终退出信息。

测试覆盖率可以帮助定位遗漏，但不能替代通信、同步、显存生命周期和结果正确性验证。性能结果记录硬件、卡数、输入规模、基线版本和测量方式。

报告应设置可访问的保留期，确保评审时仍可读取；日志中过滤令牌和敏感信息。排查先区分依赖安装、资源调度、测试执行和结果回传四个阶段，再决定交给作者还是平台维护者。

### 5.12 区分代码失败、基础设施故障与辅助 AI 检查

| 现象 | 初步归属 | 处理方式 |
|---|---|---|
| 可复现断言失败、行为不正确 | 代码或测试 | 作者定位修复，评审者确认 |
| 无可用设备、镜像拉取失败、网络中断 | 基础设施 | CI 维护者恢复，按政策重跑 |
| 写评论或写状态返回 403 | 权限或服务配置，需继续核查 | 保存接口与请求标识，检查身份及权限 |
| 辅助 AI 服务不可用 | 辅助服务 | 单独标明，不宣称代码质量失败 |

HTTP 403 不能单凭状态码判断究竟是哪项配置有误。辅助 AI 检查是否阻塞必须事先定义；若只是建议性工具，不应因服务故障阻止所有开发。

### 5.13 发布流水线与制品验证

发布负责人从已满足规则的版本提交发起发布，记录源提交、版本号、构建环境和依赖。先完成构建、测试和制品核验，再批准对外发布。

发布流程应涵盖：版本与变更记录、适用平台测试、制品校验值、安装/导入验证、发布说明、发布权限和失败恢复。签名提交、签名标签和制品完整性校验是不同对象，不能互相替代。

建议先创建 Draft Release，确认附件、版本和说明无误后由指定发布负责人发布。发布后保留制品与源代码之间的对应关系，避免覆盖既有版本而失去可追溯性。

项目流程见[发布流程规范](./release.zh.md)；GitHub 操作见[管理 Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)。

## 6. 常见问题与处理步骤

### 6.1 CODEOWNERS 配了人，却没有自动请求评审

- **现象：**Reviewers 为空，或没有出现预期模块负责人。
- **常见原因：**PR 仍是 Draft；读取的是另一个目标分支；路径未匹配或被后续规则覆盖；用户名不存在；个人没有 Write；团队不可见或没有显式仓库 Write。
- **检查与修复：**先确认 base 和 Draft 状态，打开该 base 的 CODEOWNERS 查看错误提示；按文件路径核对最后一条匹配规则，再核对真实 GitHub login 和仓库权限。修复应进入目标分支，不能只修改 PR 来源分支。
- **责任人：**作者确认目标与改动范围；管理员修复团队、权限和归属配置。
- **成功标志：**相关文件有有效的非作者 Owner，可以被请求评审并满足对应审批要求。配置修复后用测试 PR 验收，必要时对存量 PR 手动重新请求评审。

只读查询目标分支的文件错误：

~~~bash
gh api 'repos/mindspore-ai/hyper-parallel/codeowners/errors?ref=master'
~~~

目标是发布分支时替换 `ref`。参考 [CODEOWNERS 语法与错误](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)。

### 6.2 已经有人同意，仍显示 Waiting on code owner review

- **现象：**有人留言或批准，但合并区仍列出一组 Owner。
- **常见原因：**实际提交的是 Comment；批准者不是对应文件的有效 Owner；审批已被新提交撤销；尚有其他文件集合未获批准；最近推送审批规则尚未满足。
- **检查与修复：**在评审时间线确认状态为 Approved，核对当前 head、目标分支 Owner 及规则提示，请有资格且满足独立性要求的人完成正式评审。
- **责任人：**作者协调，Code Owner 评审，配置错误由管理员处理。
- **成功标志：**当前提交对应的 Code Owner 要求与其他审批规则都显示满足。

提示中的 “and/or” 不表示必须把列出的所有人都请来批准，也不表示随便一个非 Owner 的批准就够。需要按实际文件归属和规则判断。

### 6.3 作者本人是唯一有效 Owner

- **现象：**规则要求 Owner 审批，但唯一能匹配到的有效 Owner 就是作者。
- **原因：**模块归属缺少备用维护者，或其他 Owner 因权限/用户名错误失效。
- **检查与修复：**管理员核对模块真实维护人，建立至少能够覆盖休假和自有 PR 的团队或备用 Owner，授予所需仓库权限，并把修复合入目标分支。
- **责任人：**模块负责人和管理员；不能让作者批准自己的 PR 解决。
- **成功标志：**另一个有能力审查该模块的有效 Owner 可以完成审批。

对只能通过受阻配置本身修改的特殊情况，管理员按既有治理方式处理并留下审计记录；不要长期依赖这种例外。

### 6.4 Commits must have verified signatures / Unverified / UNKNOWN_KEY

- **现象：**合并被签名规则阻止，或某些提交没有 Verified。
- **常见原因：**提交未签名；签名公钥未登记；账号、邮箱或密钥不匹配；机器人生成未签名提交；rebase / cherry-pick 生成的新提交没有签名。
- **检查：**按 2.4 查询 PR 的每个提交，区分 `unsigned`、`unknown_key` 等原因，不能只看最后一条提交。
- **修复：**先修正公钥和身份配置。已经有签名但公钥未登记时先登记正确公钥并复查；确实未签名的提交，需要在作者自己的分支重新生成签名。
- **责任人：**提交作者；机器人提交由机器人维护者修复签名配置。
- **成功标志：**所有需要引入目标分支的提交满足 GitHub 的签名校验，合并区不再出现签名阻塞。

仅修复本人尚未共享的最后一条提交，可在确认暂存区没有额外改动后使用：

~~~bash
git commit --amend --no-edit -S
~~~

对于多个普通线性提交，先确认当前分支正确、工作区干净、提交范围只包含本次工作。以下操作仅适用于**本人独占的个人 Fork 分支**，不适用于共享分支或上游主线：

~~~bash
git status --short
git branch --show-current
git fetch origin
git fetch upstream
git rev-parse origin/YOUR_BRANCH
git branch backup/YOUR_TOPIC-before-signing
git log --oneline upstream/master..HEAD
git rebase --force-rebase --gpg-sign upstream/master
~~~

将 `git rev-parse` 得到的远端完整 SHA 保存为 `EXPECTED_REMOTE_SHA`，不要在推送前随意改成新值；备份分支名称应唯一。若存在 merge commits、其他人的提交或复杂依赖，停止套用此示例，由维护者确定保留历史的处理方式。处理冲突后核对最终 diff 并重新测试，再执行：

~~~bash
git push --force-with-lease=refs/heads/YOUR_BRANCH:EXPECTED_REMOTE_SHA \
  origin HEAD:refs/heads/YOUR_BRANCH
~~~

命令中的分支和 SHA 必须替换为已核对的真实值。若 lease 检查拒绝，说明远端已变化，先检查新提交，不能用普通 `--force` 覆盖。重写后重新检查签名、CLA、审批和 CI；旧 SHA 的结果不能继续作为新版本的证据。

给 merge commit 签名不能替代对 PR 中其他受规则约束提交的验证。参考：[提交签名验证](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)、[Git push 的 lease 语义](https://git-scm.com/docs/git-push)。

### 6.5 GitHub CLA 成功，机器人却提示 GitCode CLA 失败

- **现象：**同一个 PR 出现互相矛盾的协议检查。
- **常见原因：**两个平台使用不同服务、不同身份映射或不同提交版本；提交邮箱未关联正确账号。
- **检查与修复：**保存两侧检查链接及 SHA，核对协议签署身份与提交邮箱。CLA 维护者明确权威检查及过渡期要求，修复映射后重测。
- **责任人：**作者提供身份线索；CLA / CI 维护者处理服务和政策。
- **成功标志：**当前 SHA 的约定 CLA 结果一致，没有未解释的冲突。

不要只删除失败标签，也不要为了通过另一检查器任意更换作者身份。

### 6.6 使用默认模板仍被 PR 描述检查拒绝

- **现象：**按 GitHub 默认模板填写后，机器人提示缺少测试计划或字段不合法。
- **原因：**当前默认模板、中文模板和解析器并未完全对齐，也可能残留 `<label>`、空 `Fixes #` 等占位符。
- **检查与修复：**按检查器的具体报错补全真实字段，参考附录 H；确认机器人支持后重新触发描述检查。管理员同时修复模板和解析器，不能只要求每个作者手动绕开。
- **责任人：**作者修正文案；模板与 CI 维护者修复共同入口。
- **成功标志：**使用默认模板的新 PR 能通过校验，缺少真实内容的 PR 能得到准确错误提示。

没有对应 Issue 或某项检查不适用时，说明事实并与维护者确定规范，不伪造编号、不勾选没有完成的事项。

### 6.7 CI 没启动、一直 Pending，或等待批准

- **现象：**没有构建、长期排队，或页面显示需要维护者批准工作流。
- **常见原因：**事件未订阅；Draft / 路径过滤；Fork 运行审批；Webhook 投递失败；资源不可用；必需检查名称不匹配；回传丢失。
- **检查与修复：**先确定预期使用 Actions 还是 Jenkins，确认对应事件和当前 SHA 是否产生任务；再看队列、审批、运行器和回传日志。Actions 的运行审批由有权限的维护者在审查来源后完成。
- **责任人：**CI 维护者为主；作者补充 PR、SHA 和等待时间。
- **成功标志：**能找到对应版本的任务，执行后获得明确终态，必需检查名称与规则匹配。

当前没有 Actions 工作流，因此仅查看 Actions 页面为空不足以判断 Jenkins 没有执行。

### 6.8 评论显示测试成功，但 Checks 为空

- **现象：**机器人说成功，GitHub 没有对应状态检查。
- **原因：**集成只写了评论或标签，没有把结果回传为 Checks / Commit Status，或回传到错误 SHA / 仓库。
- **检查与修复：**核对当前 head 和构建证据，通过 5.5 的接口读取结果；CI 维护者补齐机器可读回传，管理员将已验收的检查加入目标规则。
- **责任人：**CI 维护者与规则管理员。
- **成功标志：**最新提交的 PR Checks 中有准确结论，失败时普通合入会被阻止。

### 6.9 新提交仍显示旧结果，或机器人重复刷屏

- **现象：**新 head 使用旧成功标签；相同内容反复评论；报告路径与点击链接编号不同。
- **常见原因：**结果没有绑定 SHA，重复事件未去重，乱序任务覆盖状态，构建链接由不同变量生成。
- **检查与修复：**对照 head SHA、任务执行时间、job 和 build number；旧结果标为历史，取消或隔离过期任务；修复回传契约和幂等逻辑。
- **责任人：**CI / 机器人维护者。
- **成功标志：**当前结果只能由当前版本的构建更新，同一任务不会反复发布相同评论，报告和链接指向同一构建。

### 6.10 403、机器人权限不足或平台合并检查失败

- **现象：**机器人显示 Forbidden、无法写状态、无法评论或同步平台检查失败。
- **常见原因：**安装仓库范围、权限、令牌生命周期或平台策略不满足；具体原因需要服务日志确认。
- **检查与修复：**记录失败接口、时间、服务身份、请求标识与脱敏错误；管理员核查 App 安装范围和所需权限，CI 维护者修复后重试同一业务任务。
- **责任人：**服务维护者和权限管理员。
- **成功标志：**服务按预期完成必要操作，失败原因有可追溯记录。

不要把 403 直接归咎于 PR 代码，也不要把扩大到所有仓库 Admin 作为默认修复。

### 6.11 更新分支后出现冲突、签名失败或审批失效

- **现象：**之前接近可合入，更新后出现新的阻塞。
- **原因：**目标分支变化产生冲突；rebase 改变 SHA / 签名；新推送触发过期审批撤销；CI 仍在测试旧版本。
- **检查与修复：**先检查新的 diff 和提交集合，再处理冲突及签名；补跑最新版本测试，请评审者检查增量并重新批准。
- **责任人：**作者处理代码和提交，评审者复审，CI 维护者排查状态异常。
- **成功标志：**当前版本无冲突，签名有效，审批和 CI 都对应最新版本。

### 6.12 同步机器人向受保护分支推送被拒绝

- **现象：**GitCode 已有更新，但同步到 GitHub master / 发布分支失败。
- **原因：**同步服务仍使用直接推送，而目标规则要求 PR、审批或签名；同步过程中也可能重写历史。
- **检查与修复：**查看具体 rule suite 的失败规则，核对同步方向及写入方式。优先把机器人调整为创建来源分支和 PR，由正常规则校验；确需镜像模式时由管理员单独设计受控同步权限与审计。
- **责任人：**迁移 / 同步服务维护者和规则管理员。
- **成功标志：**同步操作符合既定模型，规则不会反复拒绝正常业务，机器人也不能无记录地改写共享主线。

不要自动重试永久不满足规则的直接推送，更不要为了同步方便临时移除所有保护。

## 附录：检查清单、模板与迁移案例

### A. 首次开发者检查清单

- [ ] GitHub 账号正确，需要的组织邀请已接受。
- [ ] 已确认个人 Fork、上游仓库和目标分支。
- [ ] Git 与 gh 使用预期账号，SSH / HTTPS 认证可用。
- [ ] 本仓 name、email 正确，邮箱已关联 GitHub。
- [ ] 签名密钥登记为 Signing key，新提交能显示 Verified。
- [ ] CLA 已按当前要求签署，账号和邮箱映射正确。
- [ ] Python 使用 py312，editable install 指向当前工作目录。
- [ ] 能执行相关本地检查，理解需要交由硬件 CI 验证的范围。
- [ ] 已阅读目标分支开发规则，知道模块 Owner 和支持入口。
- [ ] 首次 PR 没有模板占位符，附有真实验证结果。

### B. PR 作者检查清单

- [ ] base 正确；diff 和 commits 仅包含本次改动。
- [ ] 标题说明目的，未完成工作使用真正的 Draft。
- [ ] Issue / 设计链接有效，跨平台编号已区分。
- [ ] PR 正文包括动机、实现、影响和测试证据。
- [ ] 待合入提交均满足签名规则，CLA 结果对应当前版本。
- [ ] 已确认最新 SHA 的 CI 结果；未执行的测试如实说明。
- [ ] 已处理评审意见，必要时重新请求正式评审。
- [ ] 不含无关文件、凭据、未经确认的大型制品。
- [ ] 接口、文档、兼容性和发布影响已说明。

### C. 评审人检查清单

- [ ] 确认当前 head，了解 PR 的目标与依赖。
- [ ] 自己具备相关模块知识；Code Owner 身份和权限有效。
- [ ] 检查实现、错误处理、测试覆盖和接口影响。
- [ ] 涉及分布式逻辑时按目标分支规则检查同步、通信和内存生命周期。
- [ ] 阅读真实测试证据，不只依赖标签或一句“通过”。
- [ ] 明确区分阻塞问题与建议。
- [ ] 完成检查后提交 Approve / Request changes；仅评论不视为批准。
- [ ] 新提交影响已有结论时重新评审。

### D. 合入者检查清单

- [ ] 再次确认当前 head SHA 与已审查、已测试版本一致。
- [ ] 目标分支正确，依赖 PR 已按计划处理。
- [ ] 有足够的有效批准，Code Owner 和最近推送审批要求均满足。
- [ ] 待合入提交签名满足规则，CLA 无未解决冲突。
- [ ] 必须解决的讨论已处理，无合并冲突。
- [ ] 所需 CI 已完成且对应当前版本，日志可访问。
- [ ] 过渡期已人工检查未接入 required checks 的外部 CI。
- [ ] 合并方式符合目标规则，日常流程不使用管理员旁路。
- [ ] 合入后检查目标分支和关联事项，必要时跟进发布 / 回退。

### E. 管理员迁移验收清单

以下是改造任务与验收条件，未打勾不代表已经实施。

| 优先级 | 验收项 | 通过条件 |
|---|---|---|
| P0 | CODEOWNERS 有效性 | master、发布分支错误清零；每个重要模块存在非作者可用的备用评审者 |
| P0 | 团队权限 | 模块团队可见、对本仓显式 Write，授权范围有记录 |
| P0 | CI 与合并规则衔接 | 稳定检查回传到正确 SHA，并加入 required checks；失败无法正常合入 |
| P0 | 机器人签名与更新方式 | 提交签名可验证，受保护分支更新符合已确定的 PR / 同步模型 |
| P0 | 模板与解析器 | 默认模板、中英文模板、检查器一致；真实测试 PR 验收通过 |
| P0 | CLA 权威来源 | 过渡期规则、账号映射和重复检查处理方案明确 |
| P1 | Webhook 可靠性 | 重复、重试、乱序、新 head 场景均得到正确结果 |
| P1 | 构建证据 | SHA、build number、报告目录和链接一致 |
| P1 | 合并方式与共享分支 | 仓库全局开关和规则一致，共享开发分支保护范围明确 |
| P1 | 权限与凭据 | Admin 名单按来源和职责复核，Actions 默认权限与 App 权限按需配置 |
| P1 | 例外治理 | 旁路权限、触发条件、审批记录和事后处理责任明确 |
| P1 | 旧工具迁移 | autogit / gate-doctor 等支持边界清楚，GitHub 路径有验收记录 |
| P2 | 自动合并与清理 | 基础质量规则稳定后，评估启用并验收 |
| P2 | 持续维护 | 定期复查成员、Owner、保护分支、令牌和手册 |

每次验收登记：日期、操作者、测试 PR、目标分支、配置前后差异、实际结果和后续责任人。权限收敛前先查清直接授权、团队继承和组织角色，避免误删必要权限。

### F. 分支、权限、评审与 CI 配置矩阵

**2026-09-18 核实的分支规则：**

| 项目 | master / 默认分支规则 | 发布分支规则 |
|---|---|---|
| 规则集 | protect-master（23637929） | protect-r1.0.0（23641913） |
| 匹配范围 | 默认分支 | `r[0-9]*.[0-9]*.[0-9]*`、`release/*` |
| 最少审批数 | 1 | 1 |
| Code Owner 审批 | 要求 | 要求 |
| 最近可评审推送由其他人批准 | 要求 | 要求 |
| 过期审批撤销 | 启用 | 启用 |
| 讨论解决 | 要求 | 要求 |
| 可验证签名 | 要求 | 要求 |
| Required checks | 已有规则，但列表为空 | 已有规则，但列表为空 |
| 合并前要求分支最新 | 当前 strict=false | 当前 strict=false |
| 合并方式 | Merge | Merge |
| 限制创建、删除、非快进更新 | 启用 | 启用 |
| Always bypass | liuchongming74 | liuchongming74 |

模式是 GitHub 规则匹配表达式，不是严格的语义版本校验器。是否覆盖某个分支应在 Rules 页面验证。匹配分支的创建限制也可能影响新发布分支，应由有相应权限的发布负责人操作。

**权限分配建议：**

| 对象 | 建议权限与范围 | 说明 |
|---|---|---|
| 普通外部贡献者 | 上游 Read + 自己的 Fork | 可以创建 PR |
| 模块 Code Owner 团队 | 本仓显式 Write | 团队须可见，成员具备实际评审能力 |
| 日常维护者 | 根据职责选择 Write / Maintain | 避免仅为评审授予 Admin |
| 仓库管理员 | 少量必要 Admin | 负责规则、授权和治理 |
| CI / 同步服务 | 按操作选择精确权限 | 写检查、评论、分支是不同能力 |
| 发布负责人 | 满足受保护分支及发布流程的必要权限 | 与日常代码贡献分开管理 |

组织基础权限当前为 Read；无需为解决本仓评审把整个组织基础权限提升为 Write。当前可见管理员较多，权限调整应在确认继承来源后进行。

### G. 常用 Git / GitHub CLI 命令

所有占位符都需替换；以下是查询和常用操作速查，变更历史的操作仍按正文前置条件执行。

| 目的 | 命令 |
|---|---|
| 确认账号 | `gh auth status --hostname github.com` |
| 查看远端 | `git remote -v` |
| 获取上游 | `git fetch upstream` |
| 查看工作区 | `git status --short` |
| 查看当前分支 | `git branch --show-current` |
| 查看待合入提交 | `git log --oneline upstream/master..HEAD` |
| 查看改动范围 | `git diff --stat upstream/master...HEAD` |
| 检查补丁空白错误 | `git diff --check` |
| 查看本仓 PR | `gh pr list --repo mindspore-ai/hyper-parallel` |
| 浏览器打开 PR | `gh pr view PR_NUMBER --repo mindspore-ai/hyper-parallel --web` |
| 查看 PR 差异 | `gh pr diff PR_NUMBER --repo mindspore-ai/hyper-parallel` |
| 查看检查 | `gh pr checks PR_NUMBER --repo mindspore-ai/hyper-parallel` |
| 查看规则集 | `gh api repos/mindspore-ai/hyper-parallel/rulesets` |
| 查看 Code Owner 错误 | `gh api 'repos/mindspore-ai/hyper-parallel/codeowners/errors?ref=master'` |

聚合读取 PR 的关键状态：

~~~bash
gh pr view PR_NUMBER --repo mindspore-ai/hyper-parallel \
  --json url,state,isDraft,baseRefName,headRefOid,mergeable,mergeStateStatus,reviewDecision,reviewRequests,statusCheckRollup
~~~

`UNKNOWN` 等中间状态可能是平台尚在计算，应稍后重查；查询结果为空或 API 返回 403 / 404 时先区分权限、接口和对象是否存在，不能据此断言没有配置保护。

### H. PR 描述与测试结果起草模板

以下保留当前中文模板的主要字段，便于作者起草。**占位符必须替换为真实内容，检查项按事实勾选；它不是已经通过现网解析器验收的模板。**默认模板与机器人字段契约仍需管理员按 3.6 统一。

源码模板入口：[默认模板](https://github.com/mindspore-ai/hyper-parallel/blob/master/.github/PULL_REQUEST_TEMPLATE.md)、[中文模板](https://github.com/mindspore-ai/hyper-parallel/blob/master/.github/PULL_REQUEST_TEMPLATE.zh-CN.md)。

~~~markdown
**What type of PR is this?**
/kind bug

----

**What does this PR do / why do we need it**:

问题：<真实触发条件、现有行为与影响>
修改：<主要实现和行为变化>
范围：<涉及模块、框架、硬件、接口及兼容性影响>
依赖：<真实设计文档或前置 PR；没有则写明>

----

**Which issue(s) this PR fixes**:

Fixes #<真实 GitHub Issue 编号>
迁移来源：<如有，填写真实 GitCode Issue / PR 完整链接>

----

**Test Plan and Test result：What scenarios were tested, and what were the verification results（Function, performance, reliability, etc.）**：

- 源 head SHA：<实际提交>
- 被测 SHA：<实际提交；如为测试合并提交，说明 base SHA>
- 环境：<操作系统、Python、框架版本、设备型号和卡数>
- 本地测试：<实际命令、测试数量、通过/失败结果>
- CI 测试：<构建编号、实际范围、结论和报告链接>
- 性能/兼容性：<实测结果；不涉及则说明原因>
- 未完成验证：<缺失条件、后续执行环境与责任人>

----

**Self-checklist**:（**请自检，在[ ]内打上x，我们将检视你的完成情况，否则会导致pr无法合入**）

- [ ] **设计**：PR对应的方案是否已经经过Maintainer评审，方案检视意见是否均已答复并完成方案修改
- [ ] **测试**：PR中的代码是否已有UT/ST测试用例进行充分的覆盖，新增测试用例是否随本PR一并上库或已经上库
- [ ] **验证**：PR描述信息中是否已包含对该PR对应的Feature、Refactor、Bugfix的预期目标达成情况的详细验证结果描述
- [ ] **接口**：是否涉及对外接口变更，相应变更已得到接口评审组织的通过，API对应的注释信息已经刷新正确
- [ ] **文档**：是否涉及官网文档修改，如果涉及请及时提交资料到Doc仓
~~~

`/kind bug` 仅适用于缺陷修复，其他改动选择当前模板允许的真实类型。没有 Issue 或检查项不适用时，先写清事实并与维护者对齐规范，不保留空编号或为通过校验虚假勾选。

测试结果还可使用下表补充，所有数值来自实际执行：

| 场景 | 环境 / 规模 | 命令或构建 | 实际结果 | 证据 |
|---|---|---|---|---|
| 相关 UT | 实际环境 | 实际命令 | 实际通过 / 失败数 | 日志位置 |
| 分布式 ST | 设备、框架、卡数 | 实际构建编号 | 真实结论 | 报告链接 |
| 性能回归 | 负载和基线 | 实际测量方法 | 实际变化及解释 | 对比报告 |
| 未执行项目 | 缺失条件 | 后续计划 | 未执行 | 责任人和执行入口 |

### I. 旧 GitCode 工具与 GitHub 替代流程

本次检查中，仓库 `autogit` 仍存在 “Only GitCode is supported” 限制；`gate-doctor` 的已描述流程也以 GitCode 为目标。不要将工具名称相同等同于 GitHub 支持已经完成。

| 旧入口 / 行为 | GitHub 迁移期做法 | 后续验收 |
|---|---|---|
| autogit 创建 Fork / PR | 按第 2、4 章使用 Git / gh / GitHub 页面 | 验证账号、origin/upstream、base/head |
| autogit 提交 | 使用本地 Git 签名提交，保留适用的检查 | 提交风格、签名、钩子行为一致 |
| autogit 本地检查 | 只使用已确认不依赖 GitCode 的部分 | 查清脚本是否含远程调用和平台假设 |
| gate-doctor / 评论重测 | 按当前 GitHub CI 入口和支持命令诊断 | 验证触发与结果回传 |
| GitCode 用户名与 Owner | 建立 GitHub login / 团队映射 | 用户名存在、权限有效 |
| 旧文档仓库链接 | 新开发采用本文 GitHub remote 约定 | 保留必要历史链接，标清平台 |
| 代理与 IDE 钩子 | 核实当前工具是否实际运行这些钩子 | 以可复现检查和 CI 作为共同依据 |

旧脚本没有完成 GitHub 适配前，不应把它当作默认的提交与合入入口。代码风格、测试和设计约束仍然有效。

### J. 迁移期真实案例与对应处理章节

下表为 **2026-09-18 审计快照**；PR 后续可能已经修改、关闭或合并。案例说明流程问题，不直接断言代码本身有缺陷。

| 案例 | 当时观察到的事实 | 对应手册 |
|---|---|---|
| [PR #858](https://github.com/mindspore-ai/hyper-parallel/pull/858) | 待合入提交均已签名；对应 Jenkins 成功评论存在，但原生检查仍缺少完整 CI 接入 | 2.4、5.5、6.8 |
| [PR #846](https://github.com/mindspore-ai/hyper-parallel/pull/846) | 曾同时受到 Owner 和签名规则阻塞；后来通过旁路合入，merge commit 的 Verified 不代表来源提交也已修复 | 3.9、6.2、6.4 |
| [PR #852](https://github.com/mindspore-ai/hyper-parallel/pull/852) | 有 COMMENTED 评审而没有有效批准，并存在未解决讨论 | 4.8、4.9 |
| [PR #864](https://github.com/mindspore-ai/hyper-parallel/pull/864) | compile 路径 Owner 无效，没有请求到相应评审人；同时存在大量历史机器人评论 | 3.3、6.1、5.9 |
| [PR #866](https://github.com/mindspore-ai/hyper-parallel/pull/866) | 多条提交中只有部分签名通过，并存在冲突 | 6.4、6.11 |
| [PR #848](https://github.com/mindspore-ai/hyper-parallel/pull/848) | 机器人提交出现 UNKNOWN_KEY | 2.4、6.4 |
| [PR #867](https://github.com/mindspore-ai/hyper-parallel/pull/867)、[PR #868](https://github.com/mindspore-ai/hyper-parallel/pull/868) | 模板占位符、描述检查和重复机器人反馈暴露默认入口与检查器不一致 | 3.6、6.6、5.9 |
| [PR #867 CLA 评论](https://github.com/mindspore-ai/hyper-parallel/pull/867#issuecomment-5728272209) | GitHub 与同步平台 CLA 结果不一致 | 2.5、6.5 |
| [PR #844 报告评论](https://github.com/mindspore-ai/hyper-parallel/pull/844#issuecomment-5729251542) | 报告路径与点击链接对应的构建编号不一致 | 5.7、6.9 |

当时还观察到同步机器人直接更新受保护发布分支被拒绝，应按 6.12 修复同步模型。历史执行曾包含随后调整的规则，分析旧记录时需同时看执行时间和当时的规则快照，不能把历史失败条件写成今天仍启用的配置。

### K. 问题归属、支持入口与升级路径

| 问题 | 首要责任角色 | 提供的信息 | 升级路径 |
|---|---|---|---|
| SSH、提交邮箱、签名 | 提交作者 | 脱敏错误、提交 SHA、使用的签名方式 | 组织 / 身份管理员 |
| CODEOWNERS 与团队权限 | 仓库管理员、模块负责人 | PR、目标分支、文件路径、无效条目 | 维护团队 |
| 正式评审与未解决讨论 | 对应评审者 / Owner | 当前 SHA、待处理意见 | 模块备用维护者 |
| 描述检查、CLA | CI / CLA 维护者 | 两侧链接、SHA、具体错误 | 迁移负责人 |
| 测试失败 | 先由作者与 CI 维护者分流 | 构建 ID、日志、环境、复现条件 | 对应模块或硬件平台负责人 |
| 重复评论、403、同步失败 | 机器人 / 同步服务维护者 | 事件时间、请求 ID、来源与目标 | 仓库 / 组织管理员 |
| 规则旁路、权限变更 | 规则管理员 | 业务理由、影响、证据、恢复方案 | 项目维护团队 |

公开协作优先在对应 PR 的 Conversation 中记录问题；跨 PR 的平台缺陷可在 [Issues](https://github.com/mindspore-ai/hyper-parallel/issues) 建立可跟踪事项并链接受影响 PR。团队入口为 [HyperParallel Maintainers](https://github.com/orgs/mindspore-ai/teams/hyperparallel-maintainers)。

本手册没有为 CI、CLA 等角色臆造具体负责人。迁移负责人应在发布手册时补齐角色到实际账号的映射，以及团队约定的响应时间。凭据和未脱敏服务日志通过团队批准的私密渠道处理。

### L. 变更记录与维护要求

| 版本 | 日期 | 变化 |
|---|---|---|
| 1.0 | 2026-09-18 | 基于迁移期 PR 与仓库配置审计，建立本地配置、权限与规则、PR 协作、CI/CD、排障和检查清单；区分现状与待实施建议 |

后续修改必须同时检查正文、矩阵、示例和 FAQ 是否一致。修改团队权限、CODEOWNERS、规则集、必需检查、模板字段、CLA 或同步方式时，登记对应配置变更和验收 PR，更新“现行配置”的核实日期。

GitHub 入口名称和规则能力可能调整；操作时以目标仓库当前 Rules 页面及本文引用的官方文档为准。
