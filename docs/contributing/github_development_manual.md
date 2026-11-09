# HyperParallel GitHub Development and Collaboration Manual

**English** | [简体中文](github_development_manual.zh.md)

Version: 1.0 · Updated: 2026-09-18 · Repository: [mindspore-ai/hyper-parallel](https://github.com/mindspore-ai/hyper-parallel)

This manual gives developers, reviewers, repository administrators, and CI maintainers migrating from GitCode to GitHub practical procedures, acceptance checks, and troubleshooting entry points. Commands use Bash/Zsh; Windows users can use Git Bash, while hardware tests still require supported environments.

**Distinguishing rules from recommendations:**

- **Current configuration**: Repository settings verified on 2026-09-18; update this manual when settings change.
- **Operating procedure**: How to develop and collaborate under current rules.
- **Recommended implementation**: Platform changes not yet completed; do not treat them as enabled capabilities.
- **Historical example**: A snapshot of a PR or rule execution, not a claim about its present state.

## Navigation and Common Entry Points

| Topic | Entry point |
| --- | --- |
| First contribution | [Chapter 0](#0-reading-guide-and-first-contribution-quick-start), [Local setup](#2-local-github-development-setup) |
| Platform differences and team conventions | [Chapter 1](#1-githubgitcode-differences-and-migration-conventions) |
| Administrator setup | [Chapter 3](#3-github-platform-and-repository-configuration) |
| Daily development, review, and merge | [Chapter 4](#4-daily-development-pr-submission-review-and-merge) |
| CI/CD and Jenkins integration | [Chapter 5](#5-github-cicd-and-existing-test-system-integration) |
| Common errors | [Chapter 6](#6-common-problems-and-remedies) |
| Checklists, commands, templates, and examples | [Appendices A–L](#appendices-checklists-templates-and-migration-examples) |
| GitHub workspace | [PRs](https://github.com/mindspore-ai/hyper-parallel/pulls), [Issues](https://github.com/mindspore-ai/hyper-parallel/issues), [Actions](https://github.com/mindspore-ai/hyper-parallel/actions) |
| Effective rules | [Repository rules](https://github.com/mindspore-ai/hyper-parallel/rules), [master rules](https://github.com/mindspore-ai/hyper-parallel/rules/23637929), [Release rules](https://github.com/mindspore-ai/hyper-parallel/rules/23641913) |
| Code ownership | [master CODEOWNERS](https://github.com/mindspore-ai/hyper-parallel/blob/master/CODEOWNERS), [r1.0.0 CODEOWNERS](https://github.com/mindspore-ai/hyper-parallel/blob/r1.0.0/CODEOWNERS) |

## 0. Reading Guide and First-Contribution Quick Start

### 0.1 Scope, Ownership, and Update Date

Follow this manual for GitHub submissions. Framework, hardware, style, and test implementation follow the target branch's AGENTS.md, rules, and build documentation. Branches may be at different architectural stages; do not copy instructions for removed modules from another branch.

The repository maintenance team owns this manual; relevant administrators review rules, identity, and CI sections. See [HyperParallel Maintainers](https://github.com/orgs/mindspore-ai/teams/hyperparallel-maintainers), visible only to authorized organization members. Appendix K lists responsibilities.

For each new version, recheck rules, templates, and CI entry points, and update the date and changelog. This version changes no remote permissions or gate settings.

### 0.2 Responsibilities of the Four Roles

| Role | Responsibilities | Expected outcome |
| --- | --- | --- |
| Developer / PR author | Use the correct identity, implement/test code, describe changes, address feedback | Reviewable, reproducible PR with valid commit signatures |
| Reviewer / Code Owner | Review owned code and interface impact; submit formal reviews | Approval or substantiated Request changes |
| Repository administrator | Maintain teams, ownership, branch rules, releases, and exceptions | Verifiable, auditable collaboration rules |
| CI maintainer | Maintain triggers, execution, reporting, runners, and service credentials | Trusted results attached to the correct commit |

An Assignee is not automatically a requested Reviewer; organization membership does not imply repository Write access. Open-source contributors can submit fork PRs without upstream write access, while effective Code Owners must satisfy permission requirements.

### 0.3 First-Contribution Path

1. Sign in to the correct GitHub account; accept the organization invitation if joining a team.
2. Configure SSH, commit email, signing, and CLA as described in Chapter 2.
3. Create a feature branch in your fork from the latest target branch.
4. Make a small change and run relevant local checks/tests.
5. Push signed commits and create a Draft PR against the correct upstream branch.
6. Complete the description and issue links, mark Ready after self-checks, and confirm CI runs on the current commit.
7. Request the appropriate Code Owner and obtain formal approval after addressing feedback.
8. The merger checks approvals, signatures, CI, and conflicts, then merges under the rules.
9. Sync upstream and clean up completed personal branches.

Use a useful small documentation or test fix for your first contribution; do not trigger expensive hardware jobs with empty changes.

### 0.4 Success Criteria at Each Step

| Stage | Success criterion | Contact if unmet |
| --- | --- | --- |
| Login/push | SSH shows the expected username; your fork accepts pushes | Self-diagnose, then organization administrator if needed |
| Commit identity | Author links to your account; commits to merge show Verified | Developer |
| CLA | Agreed CLA check passes on the current commit | CLA / CI maintainer |
| Description | Truthful content is complete and validator passes | Author; CI maintainer for incompatible templates |
| Review | Relevant module has valid non-author approval; discussions handled | Code Owner |
| CI | Required checks for the current SHA have terminal results and accessible logs | CI maintainer |
| Merge | PR is Merged and target branch contains expected results | Merger |

**Migration status:** At this version, required-check lists are still empty, and the default template and some CODEOWNERS entries need repair. No red Checks does not mean tests passed. During transition, mergers must manually verify complete evidence and advance the changes in Chapters 3 and 5.

## 1. GitHub/GitCode Differences and Migration Conventions

### 1.1 Organizations, Teams, Repositories, and Forks

The organization is mindspore-ai and the main repository hyper-parallel. Teams manage organization members and receive permissions for individual repositories. Personal forks are separate repositories with their own branches/settings; they do not inherit upstream membership authorization.

Here, `origin` means your GitHub fork and `upstream` means mindspore-ai/hyper-parallel. A PR's base is the upstream target branch; its head is your fork's development branch. See the [GitHub fork guide](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/fork-a-repo).

### 1.2 Accounts, Roles, and Permissions

During migration, separately verify the person's identity, GitHub username, and repository permissions. GitCode username, display name, Git author.name, and GitHub login are not interchangeable.

| GitHub repository role | Usage in this manual |
| --- | --- |
| Read | Browse, clone, and contribute through forks |
| Triage | Primarily triage issues/PRs |
| Write | Maintain code; baseline permission for Code Owner eligibility |
| Maintain | Maintain collaboration settings within assigned duties |
| Admin | Manage access, rules, and other administrative matters |

See [GitHub repository roles](https://docs.github.com/en/organizations/managing-user-access-to-your-organizations-repositories/managing-repository-roles/repository-roles-for-an-organization) for exact capabilities. Identity migration records contain only work-required information, never passwords, private keys, or tokens.

### 1.3 Responsibilities of GitHub, GitCode, Jenkins, and Sync Bots

**Observed migration path:** GitHub PRs have Paired GitCode merge requests; external Jenkins runs tests and bots report results to GitHub. No GitHub Actions workflows are currently registered; Jenkins can still be used.

~~~mermaid
flowchart LR
    A["Personal GitHub fork"] --> B["GitHub PR"]
    B --> C["Sync service / paired GitCode request"]
    C --> D["Jenkins and hardware tests"]
    D --> E["Bot comments and labels"]
    E --> B
    D -. "Planned: commit-bound check status" .-> F["GitHub Checks / Statuses"]
    F --> B
~~~

Each task should have one development entry point and one corresponding PR. For sync-managed or Paired markers, verify origin and sync direction before choosing where to update. Avoid editing the same branch on both platforms or creating competing PRs. Platform maintainers must record the source repository, sync direction, authorized bots, and stop conditions; timestamps alone cannot establish them.

### 1.4 Username, Commit Email, and CLA Mapping

Each person must verify at least:

- GitHub login: Used for review requests, CODEOWNERS, and authorization.
- Commit email: Associates commits with the account; it should appear in your GitHub email settings.
- Signing public key: Verifies the commit's cryptographic signature.
- CLA identity/email: Used for contribution-agreement checks.

Validate all four separately. A GitCode CLA signature does not imply automatic recognition by another platform's CLA service; maintainers resolve duplicate results and identity mappings.

### 1.5 Migrating Issue/PR Numbers and Links

GitHub `#123` refers to the current GitHub repository, not automatically GitCode `!123` or Issue #123. Migration records need source URL, destination URL, relationship, and current state.

Use complete URLs for cross-platform objects and real GitHub issues for native links. Historical Chinese-template comments and cross-platform link support differ and need verification through actual `/check-pr` feedback. Never invent a same-numbered issue to pass checks. See [linking issues and PRs](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue).

### 1.6 Comment Commands, Labels, Reviews, and Checks

| Object | Meaning | Entry point |
| --- | --- | --- |
| `/check-pr`, `/retest` | Project-specific bot commands | PR Conversation comment |
| `ci-running`, `ci-pipeline-passed` | Bot-managed labels | PR sidebar |
| Review request | Ask someone to review | Reviewers |
| Approve / Request changes | Formal GitHub review decision | Files changed → Review changes |
| Checks / Statuses | Automated result associated with a commit | PR Checks / merge area |
| Rulesets | Conditions required to update branches | Repository Rules / Settings |

Bot commands are not GitHub built-ins. A comment saying “agreed” or `/lgtm` does not automatically replace Approve; passing CI does not replace review. Actual bot support and responses determine command behavior.

### 1.7 Team Conventions

Personal development defaults to forks. Ordinary features target master; release fixes target maintained branches such as r1.0.0 according to the maintenance plan. Confirm the target first to avoid bringing all mainline differences into a release branch.

Protected branches currently allow only Merge commits and require signatures and independent approval. Squashing personal WIP before a PR differs from choosing Squash at merge. Never force-push shared main/release branches. Before rewriting personal history, verify that others do not depend on it. Agent-created branches use the required `codex/` prefix; ordinary development follows project naming rules.

## 2. Local GitHub Development Setup

### 2.1 Git, GitHub CLI, and Development Tools

Install [Git](https://git-scm.com/downloads) and [GitHub CLI](https://cli.github.com/). SSH commit signing requires compatible Git; version 2.34 or later is recommended. Check:

~~~bash
git --version
gh --version
conda env list
~~~

Use an existing py312 environment. Otherwise, create Python 3.12 according to team standards:

~~~bash
conda create -n py312 python=3.12
~~~

Use `conda run -n py312` for subsequent Python commands to avoid system Python. Install drivers, CANN, CUDA, and framework combinations according to the target branch's [Installation Guide](../installation.md).

### 2.2 SSH / HTTPS Authentication and Multiple Accounts

Register your public key in [SSH and GPG settings](https://github.com/settings/keys). Reuse a working key or, before creating one, ensure the destination does not already exist:

~~~bash
ssh-keygen -t ed25519 -C "YOUR_VERIFIED_EMAIL" -f ~/.ssh/id_ed25519_github_hp
ssh-add ~/.ssh/id_ed25519_github_hp
~~~

Add the generated `.pub` as an Authentication key; keep the private key local. Verify:

~~~bash
ssh -T git@github.com
gh auth login --hostname github.com --git-protocol ssh --web
gh auth status --hostname github.com
~~~

Trust the username reported by SSH. GitHub provides no SSH shell, so a nonzero exit after successful authentication does not necessarily indicate failure. Check CLI and Git SSH identities independently. HTTPS users should use a credential manager or CLI login, never embed tokens in repository URLs.

For multiple accounts, add a separate Host alias in `~/.ssh/config` and point the repository remote at it, for example:

~~~sshconfig
Host github-hp
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github_hp
    IdentitiesOnly yes
~~~

Verify with `ssh -T git@github-hp`; the remote hostname must also be `github-hp`. See [SSH configuration/authentication](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

### 2.3 Commit Name, Email, and Account Association

Set repository-local identity after entering the checkout; replace placeholders with your details:

~~~bash
git config user.name "YOUR_NAME"
git config user.email "YOUR_VERIFIED_EMAIL"
git config --show-origin --get user.name
git config --show-origin --get user.email
~~~

Verify the email in [email settings](https://github.com/settings/emails). For GitHub noreply addresses, copy the actual address and check CLA-service support. Changing only the name does not fix an incorrect email association.

Acceptance: new commits link to your GitHub account, and `git show --no-patch --format=fuller HEAD` shows the expected author and committer. See [commit email setup](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address).

### 2.4 Commit Signing and Verified Status

For SSH signing, add the public key as a **Signing key** in [key settings](https://github.com/settings/keys), even if it is already an Authentication key. Follow the official signing registration steps, then configure this repository:

~~~bash
git config gpg.format ssh
git config user.signingkey ~/.ssh/id_ed25519_github_hp.pub
git config commit.gpgsign true
git config --get gpg.format
git config --get user.signingkey
git config --get commit.gpgsign
~~~

Use `git commit -S`. Existing GPG users can register their public key following [official signing configuration](https://docs.github.com/en/authentication/managing-commit-signature-verification/telling-git-about-your-signing-key); switching to SSH is unnecessary.

Acceptance requires **Verified** on every commit to merge after pushing. Administrators can query read-only:

~~~bash
gh api --paginate repos/mindspore-ai/hyper-parallel/pulls/PR_NUMBER/commits \
  --jq '.[] | {sha, verified: .commit.verification.verified, reason: .commit.verification.reason}'
~~~

Replace PR_NUMBER with a real number. GitHub verification differs from local SSH allowedSignersFile trust configuration; inspect that configuration when local verification fails. CLA, `Signed-off-by` text, and cryptographic signatures are separate. See [adding a signing key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

### 2.5 CLA Signing and Validation

On your first PR, open the trusted CLA check's link and sign with your account and required identity. Ensure commit email matches the signature record, then recheck as instructed. The project's `/check-cla` feedback is a project-service command.

Migration may show both GitHub CLAassistant and GitCode feedback. Record each source, SHA, and failing email for the CLA maintainer. Success in one service does not imply success in another; never submit under someone else's identity. Acceptance means the current commit passes the team-designated source.

### 2.6 Fork, Clone, and origin / upstream

Open the [upstream repository](https://github.com/mindspore-ai/hyper-parallel), click Fork, and create your copy. The following applies only to a new, uncloned directory:

~~~bash
git clone git@github.com:YOUR_GITHUB_LOGIN/hyper-parallel.git
cd hyper-parallel
git remote add upstream git@github.com:mindspore-ai/hyper-parallel.git
git fetch upstream --prune
git remote -v
~~~

Replace the hostname if using an SSH alias. Verify origin points to your GitHub fork and upstream to the organization. Explicitly use `--repo mindspore-ai/hyper-parallel` with GitHub CLI PR commands to avoid selecting the fork.

### 2.7 Environment, Editable Installation, and Local Tests

Install matching framework/hardware dependencies into py312 using the [Installation Guide](../installation.md), then run from this worktree's root:

~~~bash
conda run -n py312 python -m pip install -e .
conda run -n py312 python -m pip show hyper_parallel
conda run -n py312 python -c "import hyper_parallel; print(hyper_parallel.__file__)"
~~~

The import path must point to this worktree. Reinstall editably after changing worktrees or discovering a stale path; PYTHONPATH alone does not satisfy this check.

Unit-test example:

~~~bash
conda run -n py312 python -m pytest -vs tests/ut
~~~

Run affected modules first, then required gate suites. Without NPU/GPU resources, report completed logic tests and remaining CI validation honestly. For ST that spawns workers through torchrun_case or similar, run pytest at the case entry point and let the launcher manage processes; do not wrap the entire launcher suite in another torchrun. Follow the [testing rules](../../.agent/rules/testing.md), rather than duplicating test implementation standards here.

### 2.8 New-Member Environment Checklist

Record Git/CLI versions, target branch, Python environment, import path, test commands, and results in the first PR. Public material must exclude tokens, private keys, internal credentials, and full environment dumps.

Appendix A provides checkboxes. Verify at least one actual commit as Verified; writing signing settings into Git does not prove GitHub accepts the key.

### 2.9 Migrating an Existing GitCode Working Directory

Run `git status` and `git remote -v` first and preserve uncommitted work. If origin exists, do not add it again:

~~~bash
git remote rename origin gitcode-origin
git remote add origin git@github.com:YOUR_GITHUB_LOGIN/hyper-parallel.git
~~~

If upstream points to GitCode, rename it gitcode-upstream before adding GitHub upstream; retain it if already on GitHub. Choose a recognizable unused name rather than overwriting an existing remote.

~~~bash
git remote rename upstream gitcode-upstream
git remote add upstream git@github.com:mindspore-ai/hyper-parallel.git
git fetch origin --prune
git fetch upstream --prune
git remote -v
~~~

The second command group applies only when an old GitCode upstream exists. Compare the migrating branch's diff, commit count, and signatures against the new target. For substantial divergence, branch from the new target and selectively migrate your business changes. Verify SHA relationships from the actual sync method; identical content does not imply identical SHAs.

## 3. GitHub Platform and Repository Configuration

For repository/organization administrators and CI maintainers. Before remote changes, record current/target values, impact, and validation. Recommendations here do not themselves authorize changes.

### 3.1 Organization Base Permissions, Teams, and External Contributors

The current organization base permission is Read. Grant hyper-parallel access at repository level; there is no need to raise the entire organization's baseline to Write.

Use repository Settings → Collaborators & teams → Add teams / Add people. Grant routine maintainers Write and verify effective roles after invitation acceptance. External contributors can use forks; administrators explicitly define access for long-term Code Owners.

Verify a real account with this read-only query:

~~~bash
gh api repos/mindspore-ai/hyper-parallel/collaborators/LOGIN/permission \
  --jq '{permission, role_name, login: .user.login}'
~~~

Replace LOGIN with a GitHub username. Some APIs reject callers without administrative access; an authentication failure does not prove the other user lacks permission. See [access management](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-teams-and-people-with-access-to-your-repository).

### 3.2 Module Teams and Backup Reviewers

The hyperparallel-maintainers team already has repository Write. **Recommended implementation:** Assign at least two qualified maintainers per independently reviewed module and define substitutes for absence, leave, and cross-module changes.

In organization Teams → New team, choose appropriate Visible visibility, add members who accepted invitations, and grant repository access. Members' individual access does not replace permissions required by the Code Owner team itself. See [creating teams](https://docs.github.com/en/organizations/organizing-members-into-teams/creating-a-team).

After membership changes, verify that someone other than the author can approve owned paths. A path approvable only by its author is not correctly configured.

### 3.3 CODEOWNERS Configuration and Validation

GitHub reads CODEOWNERS from the PR's base branch, choosing the first file found in .github/, repository root, then docs/. The last matching rule applies; a more specific rule does not automatically retain earlier default owners.

Teams in the example must already exist and have repository access; module-team names below are illustrative:

~~~text
* @mindspore-ai/hyperparallel-maintainers
/hyper_parallel/compile/ @mindspore-ai/EXAMPLE-COMPILE-TEAM
/tests/ut/compile/ @mindspore-ai/EXAMPLE-COMPILE-TEAM
/CODEOWNERS @mindspore-ai/hyperparallel-maintainers
/.github/ @mindspore-ai/hyperparallel-maintainers
~~~

For multiple owners on one rule, one approval usually suffices. PRs matching different rules need approvals covering the corresponding files. Ordinary reviewers may comment, but only qualifying approvals satisfy merge rules. See [official CODEOWNERS rules](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners).

Check both target branches:

~~~bash
gh api 'repos/mindspore-ai/hyper-parallel/codeowners/errors?ref=master'
gh api 'repos/mindspore-ai/hyper-parallel/codeowners/errors?ref=r1.0.0'
~~~

**Pending repair:** The branches still have 34 and 37 errors respectively. Unknown owner can mean nonexistent accounts or insufficient access; verify identity rather than replacing similar spellings automatically. Acceptance requires empty error lists and a small PR with real paths that requests expected maintainers. After repairing access for existing PRs, check whether review requests must be issued manually.

### 3.4 Branch Protection Strategy

| Branch purpose | State in this version | Procedure |
| --- | --- | --- |
| master | Protected by protect-master | Merge features through PRs |
| Matching release branches such as r1.0.0 | Protected by release rules | Use the matching baseline and release plan |
| Shared branches such as rl and trainer_dev | Not covered by those two rules | Administrators decide protection; do not assume it exists |
| Personal fork feature branches | Personally managed | Not a substitute for upstream release/quality acceptance |

**Recommended implementation:** Define writers, PR requirements, and mainline sync timing for shared integration branches. Release rules restrict branch creation; release owners must check permissions in advance rather than opening the whole repository on release day.

### 3.5 Rulesets: Approvals, Signatures, Discussions, and Checks

Current master/release rules require at least one approval, Code Owner review, another person's approval of the latest reviewable push, resolved discussions, stale-approval handling, signed commits, and protection from deletion/force-push. Protected branches allow Merge only.

**Current gap:** Both rules configure “Require status checks,” but their check-name lists are empty, so no real CI test is enforced by that setting.

In Settings → Rules → Rulesets, inspect branch scope, enforcement, specific checks, and bypass users. Rulesets can stack; a 404 from the old branch-protection API does not mean no Rulesets exist. See [creating rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository).

**Recommended implementation:** Establish real checks following Chapter 5, then require them and bind expected sources. Never require a check no service reports, which can leave every PR waiting.

### 3.6 PR / Issue Templates and Validators

Maintenance links: [Default template](https://github.com/mindspore-ai/hyper-parallel/blob/master/.github/PULL_REQUEST_TEMPLATE.md), [Chinese template](https://github.com/mindspore-ai/hyper-parallel/blob/master/.github/PULL_REQUEST_TEMPLATE.zh-CN.md), [Issue templates](https://github.com/mindspore-ai/hyper-parallel/tree/master/.github/ISSUE_TEMPLATE).

The default template currently uses old English checklist items while the bot expects different fields. Update the default, both languages, and parser together around one stable field contract. Editing templates does not update existing PR descriptions.

Acceptance: create a new PR from the default template, fill truthful content, and run validation; verify accurate messages for missing issues/test descriptions. Appendix H provides a draft, but live feedback determines acceptance.

### 3.7 GitHub Apps, Webhooks, Tokens, and Bots

Prefer independent service identities or GitHub Apps with only needed repository, check-reporting, and comment permissions. Reviewing, writing code/checks, and managing settings are separate capabilities; avoid granting all through a personal long-lived token.

Record webhook event types, service owners, and latest delivery results. Services verify signatures and handle duplicate deliveries. For new bots, separately validate permissions to create PRs, push personal source branches, report checks, and merge under rules.

Actions currently defaults to Write tokens and allows workflow PR creation/approval. **Recommended implementation:** Switch to default Read before enabling workflows, elevate explicitly per job, and retain automatic approval only after business confirmation. See [Actions security](https://docs.github.com/en/actions/reference/security/secure-use).

### 3.8 Merge Methods, Auto-Merge, and Cleanup

Repository General enables Merge/Squash/Rebase, but target rules allow only Merge, so normally choose Create a merge commit. Global enablement does not imply target-branch permission.

Auto-merge and branch deletion after merge are currently disabled. Evaluate auto-merge after checks stabilize; it waits for conditions, not replaces approvals or expands access. Authors clean personal fork branches after confirming merge and no remaining dependencies.

### 3.9 Administrator Bypass, Emergencies, and Audit

Only liuchongming74 currently has always bypass in both rulesets. This permits protected-branch updates without satisfying rules; it does not mean checks passed.

Review dismissal is also restricted, currently to liuchongming74. Authorized maintainers investigate disputed/unavailable reviews and record reasons. Dismissal does not replace fixing unresolved issues.

**Recommended implementation:** Record emergency reasons, impact, reviewer, original/target SHA, evidence, and recovery plan; later document resolution of failed conditions. Asking administrators to bypass is not routine merging. Prefer fix/revert PRs for mainline issues; this manual provides no shared-branch force-push command.

### 3.10 Acceptance After Configuration Changes

Create recognizable test PRs and validate without bypass privileges:

1. Unsigned commits are blocked; valid signed commits are accepted.
2. Owner configuration works and a non-author can approve.
3. Ordinary comments do not count as approval; unresolved discussions remain visible.
4. Required CI failures block merge and success is reported correctly.
5. New heads have their own checks; old-head results cannot overwrite them.
6. Sync bots can perform authorized actions and unauthorized updates are rejected.
7. Validate master and release branches separately.

Use agreed test branches/PRs only; do not merge meaningless content into mainline for testing. Record results in Appendix E.

## 4. Daily Development, PR Submission, Review, and Merge

### 4.1 Start from the Correct Target Branch

Decide between `master` and a release branch, then create a personal branch from its latest upstream commit. This example targets `master`; confirm the working tree is clean first:

~~~bash
git status --short
git fetch upstream
git switch -c fix/YOUR_TOPIC upstream/master
~~~

Replace `YOUR_TOPIC` with a short topic. Each feature branch handles one independently reviewable change; do not accumulate development commits on local `master`. For release fixes, replace `master` with the confirmed target to avoid importing new mainline features.

### 4.2 Sync Upstream and Resolve Conflicts

Unpublished or exclusively personal branches may rebase onto current upstream. This rewrites commits and requires new signatures:

~~~bash
git fetch upstream
git rebase --gpg-sign upstream/master
~~~

Resolve conflicts by understanding each file, stage only resolved files, and run `git rebase --continue`. If unsure, use `git rebase --abort` to preserve the original state. Rerun affected tests and verify signatures on GitHub afterward.

For rewritten personal branches already pushed, see 6.4. Do not independently rebase or force-push shared branches; agree on synchronization with maintainers. When merging upstream, ensure introduced historical commits satisfy signing rules; signing only the newest merge commit does not repair unsigned ancestors.

References: [Syncing forks](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/syncing-a-fork), [Git rebase](https://git-scm.com/docs/git-rebase).

### 4.3 Commit Messages, Cleanup, and Signing

Use Conventional Commits with a subject around 80 characters or fewer describing the behavior change, for example:

~~~text
fix: preserve shard metadata during redistribution
docs: clarify GitHub contributor setup
~~~

Stage only files needed for this change, inspect the index, and create a signed commit:

~~~bash
git add PATH_TO_CHANGED_FILE
git diff --cached
git commit -S -m "fix: describe the actual change"
~~~

`PATH_TO_CHANGED_FILE` is an actual path. Do not blindly include local configuration, tokens, logs, or large generated files.

Clean up meaningless WIP before opening a PR. During review, keep incremental changes traceable where possible. Explain necessary squashing/history rewrites and obtain new required reviews, signatures, and CI results. Local consolidation into one commit differs from GitHub Squash merge; protected branches still use Merge.

### 4.4 Pre-PR Self-Checks

At minimum, verify:

- Scope and target branch are correct, without unrelated commits.
- Changes follow the target branch's `AGENTS.md` and scoped rules; Python files, comments, and interfaces meet conventions.
- Necessary tests are added/updated, with actual commands, environments, and results recorded.
- Every commit introduced into the target satisfies signing requirements and maps email to the correct account.
- Issues, designs, and compatibility claims have real evidence; explain unrun tests.
- The description contains no template placeholders, invalid `Fixes #`, or unverified “all passed” claims.

Inspect the patch first, then run tests required by the change:

~~~bash
git diff --check
git diff upstream/master...HEAD
git log --oneline upstream/master..HEAD
conda run -n py312 python -m pytest -vs tests/ut
~~~

Full UT may require platform dependencies. If unavailable locally, state missing conditions and the CI environment responsible for validation. “Not run” cannot be recorded as “passed.”

### 4.5 Create a PR: Title, Issues, Description, and Evidence

Push to your fork first:

~~~bash
git push -u origin HEAD
~~~

On the [new PR page](https://github.com/mindspore-ai/hyper-parallel/compare), verify base repository/branch and head repository/compare branch, or use CLI:

~~~bash
gh pr create --repo mindspore-ai/hyper-parallel \
  --base master --head YOUR_GITHUB_LOGIN:YOUR_BRANCH \
  --draft --title "fix: describe the actual change" \
  --body-file /ABSOLUTE/PATH/pr-body.md
~~~

Replace the account, branch, and body-file path. Explain why, what changed, affected interfaces/platforms, validation, and unfinished work. Appendix H gives a structure; the live template/parser still need alignment, so pasting it does not guarantee acceptance.

Use a real GitHub number, such as `Fixes #ACTUAL_NUMBER`; use full repository references across repositories. Identify GitCode issues by platform and actual URL, never by an unrelated same-numbered GitHub issue. Closing keywords work only for PRs targeting the default branch; release PRs must separately record links and verify issue state.

References: [gh pr create](https://cli.github.com/manual/gh_pr_create), [Linking PRs and issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue).

### 4.6 Draft and Ready for Review

Use Draft while implementation, validation, or self-checks remain incomplete. WIP in the title is only text and does not make a PR Draft.

When ready, select Ready for review or run:

~~~bash
gh pr ready PR_NUMBER --repo mindspore-ai/hyper-parallel
~~~

Draft PRs do not automatically request Code Owner review; becoming Ready triggers requests. Whether CI runs on Draft depends on triggers, so absent Draft CI is not automatically a failure. See [Draft PR behavior](https://docs.github.com/en/pull-requests/reference/pull-requests#draft-pull-requests).

### 4.7 Request Reviews and Handle Multiple Owners

Code Owners come from valid CODEOWNERS on the target branch. Check ownership for Files changed, then confirm Reviewers includes suitable authorized people.

You can request a maintainer manually:

~~~bash
gh pr edit PR_NUMBER --repo mindspore-ai/hyper-parallel \
  --add-reviewer GITHUB_LOGIN
~~~

Manual requests do not repair invalid CODEOWNERS or turn ordinary reviewers into owners. Cross-module PRs may need approvals for separate file sets; when one rule lists multiple valid owners, usually one approval satisfies that rule rather than requiring everyone. Approval counts and latest-push rules still apply.

Authors cannot satisfy requirements by approving their own PRs. If an author is the only valid owner for a file set, administrators must add a real backup and validate the target-branch configuration.

Reference: [Code Owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners).

### 4.8 Comment, Approve, and Request Changes

| Review action | Purpose | Merge effect |
|---|---|---|
| Comment | Ask questions, suggest changes, or record observations | Not approval |
| Approve | Review completed; current changes accepted | Counts when eligibility conditions are met |
| Request changes | Problems require fixes and re-review | May block merge until resolved |

After reviewing implementation, tests, platform impact, and public interfaces, explicitly choose a decision in Finish your review. An “LGTM” comment is not Approve.

CLI examples:

~~~bash
gh pr review PR_NUMBER --repo mindspore-ai/hyper-parallel \
  --comment --body "Please add test results for this scenario."
gh pr review PR_NUMBER --repo mindspore-ai/hyper-parallel \
  --approve --body "I reviewed the implementation and test results and approve these changes."
gh pr review PR_NUMBER --repo mindspore-ai/hyper-parallel \
  --request-changes --body "Please fix the blocking issues described before requesting another review."
~~~

These commands publish real reviews. Choose one only after reaching the corresponding conclusion; do not issue unsupported approval merely to remove a red indicator.

References: [Reviewing PRs](https://docs.github.com/en/pull-requests/how-tos/review-pull-requests/reviewing-proposed-changes-in-a-pull-request), [gh pr review](https://cli.github.com/manual/gh_pr_review).

### 4.9 Respond, Resolve Discussions, and Request Re-Review

Give each comment an outcome: fixed with location, explanation, or agreement to retain the current behavior. Resolve conversation marks discussion handled, not approved.

Current rules dismiss qualifying stale approvals and require someone else to approve the latest reviewable push. After pushing, recheck:

1. Whether earlier approvals remain valid.
2. Whether current owner approvals cover affected files.
3. Whether someone other than the latest pusher supplied required approval.
4. Whether required discussions are resolved.
5. Whether the latest SHA passes CI and signature checks.

Reviewers pushing fixes to an author's branch must also respect the latest-pusher independence rule and arrange another qualified review.

### 4.10 Trigger CI, Retest, and Read Results

Start with PR Checks and actual CI build details. Some results still arrive through Jenkins/GitCode comments; verify their SHA, time, build number, and links.

~~~bash
gh pr checks PR_NUMBER --repo mindspore-ai/hyper-parallel
gh pr view PR_NUMBER --repo mindspore-ai/hyper-parallel \
  --json headRefOid,statusCheckRollup,url
~~~

Missing Checks does not mean success. If retesting uses comments, use only commands published by CI maintainers and validated for GitHub; do not assume old GitCode commands such as `/retest` work identically.

Explain retest reasons: fixed code, restored infrastructure, or a known intermittent failure. Read logs rather than repeatedly rerunning blindly.

### 4.11 Pre-Merge Checks, Merge, and Cleanup

Use Appendix D to inspect current head, valid approvals, signatures, discussions, conflicts, and real CI results. Because required checks are currently empty, manually verify agreed external evidence too.

Normally use Create a merge commit. CLI can bind the reviewed head to prevent merging a later unreviewed update:

~~~bash
gh pr view PR_NUMBER --repo mindspore-ai/hyper-parallel --json headRefOid
gh pr merge PR_NUMBER --repo mindspore-ai/hyper-parallel \
  --merge --match-head-commit FULL_REVIEWED_HEAD_SHA
~~~

`FULL_REVIEWED_HEAD_SHA` must be the full SHA just inspected. Recheck if the branch changes. Administrator bypass is not a routine solution to review or CI blocks.

After merge, confirm target contents and issue states, then remove personal branches with no remaining dependencies:

~~~bash
git fetch upstream
git switch master
git merge --ff-only upstream/master
git branch -d YOUR_BRANCH
git push origin --delete YOUR_BRANCH
~~~

If local `master` is absent, create it from `upstream/master`. If `--ff-only` or branch deletion fails, investigate rather than forcing deletion.

Reference: [gh pr merge](https://cli.github.com/manual/gh_pr_merge).

### 4.12 Large Refactors, Dependent PRs, and Parallel Development

Split large work into independently verifiable steps: behavior-preserving preparation, core features, caller migration, and cleanup. State dependencies and independent validation scope in every PR.

For dependencies on unmerged PRs, record the dependency URL and temporary base. After the prerequisite merges, update the target and recheck diff, commits, signatures, and CI; temporary-base tests do not validate the final target.

For shared features, assign an integrator/module owners and agree on branch writers. Shared development branches need explicit protection; main/release rules do not cover all development branches. Discuss public interfaces, parallel strategies, and multi-platform designs before implementation.

### 4.13 Release Fixes, Backports, and Reverts

Usually fix mainline first, then backport the minimal validated change. Emergency release-first fixes need a mainline follow-up. Backport PRs link the original and explain adaptation differences.

After branching from the release branch, cherry-pick a confirmed ordinary commit:

~~~bash
git cherry-pick -x -S ORIGINAL_COMMIT_SHA
~~~

`-x` records origin; `-S` signs the new commit. Merge commits require selecting the correct mainline parent and intended changes with maintainer review, not blindly using the ordinary command. New cherry-pick SHAs require release-branch tests.

Revert through a separate PR explaining reasons, affected versions, and follow-up. Reverting merges requires an explicit mainline parent. Do not rewrite shared mainline history through force-push.

## 5. GitHub CI/CD and Existing Test-System Integration

### 5.1 Current and Target Workflows

**Verified status:** No GitHub Actions workflows exist yet. Some checks run through GitCode/Jenkins with bot comments/labels; native statuses are mainly CLA. Some PRs lack native checks, and target required-check lists are empty.

External CI execution and GitHub-enforced passing CI are therefore not fully connected. The recommended loop is:

~~~mermaid
flowchart LR
    A[GitHub PR / new commit] --> B[Validate event and determine SHA]
    B --> C[Description / CLA / static checks]
    B --> D[UT]
    B --> E[Select GPU / NPU ST by changes]
    C --> F[Report GitHub Checks or Status]
    D --> F
    E --> F
    F --> G[Validate target-branch rules]
    G --> H[Merge after human approval]
~~~

Jenkins can retain hardware tests; adding Actions depends on maintenance cost and runtime needs. Each check must have stable identity, correct SHA, and enforceable rules.

### 5.2 Actions, Jenkins, and Sync-Bot Responsibilities

| Component | Recommended responsibility | Acceptance focus |
|---|---|---|
| GitHub Actions | Hosted static/metadata checks, some UT, and orchestration | Least privilege, explicit fork behavior, traceable results |
| Jenkins / hardware platform | NPU/GPU, multi-device ST, performance, compatibility | Isolated resources, hardware/software versions, real results |
| Sync bot | Map events/links between GitHub and the migration platform | Preserve origin, avoid loops, respect target protection |
| Result reporting service | Attach results to the exact repository/SHA | Trusted identity, stable names, reliable transitions |
| Maintainers | Select required checks, handle exceptions, approve merges | Bot comments do not replace rules or judgment |

A sync bot's successful push is not test success. Record synchronization, testing, and merge eligibility separately.

### 5.3 Layered Testing: Fast Feedback and Hardware Validation

| Layer | Typical content | Suggested triggers |
|---|---|---|
| Metadata/static checks | PR description, formatting, license headers, basic lint | All PRs, with differing documentation/source rules |
| Unit tests | UT without multi-device setup | Affected modules; periodically broader mainline coverage |
| Distributed functional tests | Relevant DP/FSDP/TP/PP scenarios | Corresponding modules or shared infrastructure |
| Platform compatibility | Ascend NPU, NVIDIA GPU, applicable framework versions | Current module support scope |
| Performance/stability | Throughput, memory, long runs, recovery | Sensitive changes and release candidates |

Path-based skips need explicit selection logic, reasons, and validation. Shared scheduling, communication, or memory changes must not receive inadequate coverage based only on file location.

Some modules are explicitly Torch-only. Follow target-branch support contracts rather than requiring nonexistent backend implementations for superficial coverage.

### 5.4 Validate Description, CLA, and Signatures Separately

They address different questions:

- Description: Are purpose, related work, tests, and interface impact explained?
- CLA: Has the contributor completed the required agreement with correct identity mapping?
- Signature: Does the commit have verifiable signing and identity association?

Passing one does not pass the others. Present separate results/remediation instead of an ambiguous “gate failed” label. Release template/parser changes together and test that the default template passes.

### 5.5 Reporting Checks / Commit Status from External CI

External systems can use:

- Commit Status API: Publish `pending / success / failure / error` against the exact SHA, with stable context and details URL.
- Checks API: Create/update check runs with execution state, conclusion, summary, and logs. Writing check runs typically uses an appropriately authorized GitHub App; consult API documentation for credential support.

Reports need repository, PR, source head SHA, tested SHA, target branch, build ID, start/end times, and details URL. Publish queued state promptly and always a terminal state after execution; compensate for timeouts/reporting failures.

Read results for a confirmed SHA:

~~~bash
gh api repos/mindspore-ai/hyper-parallel/commits/FULL_HEAD_SHA/status
gh api repos/mindspore-ai/hyper-parallel/commits/FULL_HEAD_SHA/check-runs
~~~

PR comments may summarize in Chinese but do not replace machine-readable checks required by rules. References: [Commit Status API](https://docs.github.com/en/rest/commits/statuses), [Checks API](https://docs.github.com/en/rest/checks/runs).

### 5.6 Check Names, Sources, and Required Checks

These are **proposed names, not currently deployed**:

| Example name | Scope | Recommendation |
|---|---|---|
| `pr/description` | Description completeness/fields | Require after template alignment |
| `license/cla` | One authoritative CLA result | Require after authority is established |
| `ci/lint` | Static checks | Require after stable operation |
| `ci/ut` | Applicable UT | Require |
| `ci/distributed` | Aggregate selected hardware matrix | Require according to branch quality standards |

Each name has one meaning; different systems must not report different scopes under the same context. Select the expected source and bind the App where supported to prevent another identity publishing same-name success.

Required checks must report on every intended PR. Path filters that suppress an entire workflow can leave checks pending forever. Use a stable aggregator to decide required tests, completion, and valid skips; never return unconditional success merely to clear waiting.

Reference: [Ruleset-supported rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets).

### 5.7 Bind Head SHA, Test-Merge SHA, and Build Number

PR head SHA identifies the author's version. A test-merge SHA combines it with a target-branch revision and may differ.

Record for every build:

| Field | Requirement |
|---|---|
| Source | Full repository name, PR number, source branch |
| Source version | Head SHA, and base SHA when needed |
| Tested version | Actual checkout SHA, explicitly identifying a test merge |
| Build identity | Jenkins job + build number or equivalent unique ID |
| Results | Actual test scope, conclusion, logs, and report URL |

Build numbers, report directories, and links in comments must come from one build record. Never guess or fabricate Jenkins numbers.

After head updates, clearly mark old results historical rather than currently passing. Retesting after base changes follows the chosen policy; merge-sensitive cases must test the new combination.

### 5.8 Triggers and Canceling Stale Jobs

Basic triggers cover new PRs, source updates, reopening, and Ready conversion. Description/base changes depend on check type. Description edits generally need only metadata checks, not all multi-device tests.

For one PR, cancel queued/running old revisions in favor of the latest, publishing a clear terminal state and reason rather than leaving pending. Define separate cancellation rules for releases and long stability jobs.

Reference: [Actions events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).

### 5.9 Webhook Deduplication, Retries, and Comment Updates

Verify signatures, acknowledge promptly, and process long jobs in the background. Deduplicate delivery IDs and identify business jobs by repository/PR/head/check type. Legitimate manual reruns need separate run IDs and must not be incorrectly deduplicated.

Maintain an updateable summary comment per check or use Check summaries. If new comments are necessary, send only meaningful state changes rather than identical retry updates.

Use backoff and bounded retries; persistent failures need actionable alerts. Test duplicates, out-of-order results, interleaved heads, and service restarts.

Reference: [Webhook best practices](https://docs.github.com/en/webhooks/using-webhooks/best-practices-for-using-webhooks).

### 5.10 Fork PRs, Runners, and Credentials

Fork code is unreviewed input and must not automatically receive release/production credentials or access to critical internal resources. Actions fork events restrict tokens/secrets, and first-time contributors may require approval.

`pull_request_target` uses target-repository context and permissions. Do not check out/run unreviewed fork code in sensitive-credential stages. Jenkins/self-hosted runners likewise need workspace isolation, constrained network/credential exposure, and cleanup of residual processes/files.

Use release credentials only in controlled release flows. Prefer default Read Actions tokens with per-job authorization. Administrators define permitted third-party Actions and version-pinning policies.

References: [Actions security](https://docs.github.com/en/actions/reference/security/secure-use), [Approve fork workflows](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/approve-runs-from-forks).

### 5.11 Logs, Coverage, Reports, and Diagnosis

Reports include failing stage, first useful error, command, scope, environment versions, SHA, and build URL. Preserve necessary per-rank logs for distributed failures rather than only the parent process's final exit.

Coverage helps find omissions but cannot replace communication, synchronization, memory-lifecycle, and numerical checks. Performance reports record hardware, device count, input scale, baseline, and measurement method.

Keep reports accessible through review and filter secrets. First distinguish dependency installation, resource scheduling, test execution, and result reporting, then assign the issue to authors or platform maintainers.

### 5.12 Code Failures, Infrastructure Failures, and Auxiliary AI Checks

| Symptom | Initial ownership | Response |
|---|---|---|
| Reproducible assertion failure or incorrect behavior | Code/tests | Author fixes; reviewer confirms |
| No devices, image-pull failure, network interruption | Infrastructure | CI maintainer restores; rerun by policy |
| Comment/status writes return 403 | Permissions/service configuration; investigate further | Preserve API/request IDs; inspect identity/access |
| Auxiliary AI unavailable | Auxiliary service | Report separately, without claiming code-quality failure |

HTTP 403 alone cannot identify the incorrect setting. Define whether auxiliary AI blocks merging in advance; advisory-service outages should not stop all development.

### 5.13 Release Pipelines and Artifact Validation

Release owners start from a compliant version commit and record source, version, build environment, and dependencies. Complete builds, tests, and artifact validation before approving publication.

Cover version/changelog, platform tests, artifact checksums, install/import tests, notes, permissions, and failure recovery. Signed commits, signed tags, and artifact integrity are distinct and cannot substitute for each other.

Prefer a Draft Release, then have the designated owner publish after checking attachments/version/notes. Preserve source-to-artifact mappings; do not overwrite versions and lose traceability.

See the [project release process](./release.md) and [GitHub Releases guide](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).

## 6. Common Problems and Remedies

### 6.1 CODEOWNERS Lists People but Does Not Request Reviews

- **Symptom:** Reviewers is empty or lacks the expected module owner.
- **Common causes:** Draft PR, different base, unmatched/overridden paths, nonexistent usernames, missing individual Write, invisible teams, or missing explicit team Write.
- **Check/fix:** Confirm base/Draft, inspect that base's CODEOWNERS errors, determine the last matching path rule, then verify actual logins/access. Fix the target branch, not only the PR source.
- **Owner:** Author verifies target/scope; administrators repair teams, permissions, and ownership.
- **Success:** Files have valid non-author owners who can be requested and satisfy approvals. Test repaired configuration with a PR; manually re-request existing reviews if necessary.

Read-only query for target-branch file errors:

~~~bash
gh api 'repos/mindspore-ai/hyper-parallel/codeowners/errors?ref=master'
~~~

Replace `ref` for release targets. See [CODEOWNERS syntax/errors](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners).

### 6.2 Someone Agreed but Waiting on Code Owner Review Remains

- **Symptom:** Someone commented/approved, but the merge area still lists owners.
- **Common causes:** Comment rather than Approve; ineligible owner; stale approval dismissed after commits; uncovered file sets; latest-push approval missing.
- **Check/fix:** Confirm Approved in the timeline, current head, target-branch ownership, and rule messages; request a qualified independent formal review.
- **Owner:** Author coordinates, Code Owner reviews, administrator repairs configuration.
- **Success:** Current-commit Code Owner and other approval requirements are satisfied.

“And/or” does not mean every listed person must approve, nor that any non-owner approval suffices. Apply actual file ownership and rules.

### 6.3 The Author Is the Only Valid Owner

- **Symptom:** Owner approval is required, but the only matching valid owner is the author.
- **Cause:** No backup maintainer, or other owners are invalid because of access/username errors.
- **Check/fix:** Administrators verify real maintainers, establish a team/backups covering leave and their own PRs, grant access, and merge the repair into the target branch.
- **Owner:** Module lead and administrators; self-approval cannot solve it.
- **Success:** Another qualified valid owner can approve.

When the blocked configuration itself must be changed, administrators follow existing governance and preserve audit records; do not depend on this exception permanently.

### 6.4 Commits must have verified signatures / Unverified / UNKNOWN_KEY

- **Symptom:** Signing rules block merge or commits lack Verified.
- **Common causes:** Unsigned commits, unregistered public keys, account/email/key mismatch, unsigned bot commits, or unsigned commits created by rebase/cherry-pick.
- **Check:** Query every PR commit as in 2.4, distinguishing `unsigned`, `unknown_key`, and other reasons; do not inspect only the last commit.
- **Fix:** Correct keys/identity first. If already signed with an unregistered key, register the correct public key and recheck. Truly unsigned commits need new signatures on the author's branch.
- **Owner:** Commit author; bot maintainers repair bot signing.
- **Success:** All commits entering the target pass GitHub signing validation and the merge area no longer reports signature blocks.

To repair only your last unshared commit, first ensure the index contains no extra changes, then use:

~~~bash
git commit --amend --no-edit -S
~~~

For multiple ordinary linear commits, confirm branch, clean worktree, and a range containing only this task. The following is exclusively for **your own nonshared fork branch**, never shared branches or upstream mainline:

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

Save the full remote SHA from `git rev-parse` as `EXPECTED_REMOTE_SHA`; do not casually replace it before pushing. Use a unique backup branch. For merge commits, other people's commits, or complex dependencies, stop applying this example and ask maintainers to choose a history-preserving approach. After conflicts, inspect the final diff and retest, then run:

~~~bash
git push --force-with-lease=refs/heads/YOUR_BRANCH:EXPECTED_REMOTE_SHA \
  origin HEAD:refs/heads/YOUR_BRANCH
~~~

Replace branch/SHA placeholders with verified values. A rejected lease means the remote changed: inspect new commits, never overwrite with ordinary `--force`. Recheck signatures, CLA, approvals, and CI after rewriting; old-SHA results are not evidence for the new version.

Signing a merge commit does not validate other commits governed by the rules. References: [Signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification), [Git push lease semantics](https://git-scm.com/docs/git-push).

### 6.5 GitHub CLA Passes but the Bot Reports GitCode CLA Failure

- **Symptom:** Contradictory agreement checks on one PR.
- **Common causes:** Different services, identity mappings, or commit versions; email associated with the wrong account.
- **Check/fix:** Preserve both check URLs/SHAs and verify CLA identity/email. CLA maintainers define the authoritative check and transition requirements, fix mapping, and retest.
- **Owner:** Author provides identity evidence; CLA/CI maintainers handle services/policy.
- **Success:** Agreed CLA results for the current SHA are consistent, without unexplained conflicts.

Do not merely remove failure labels or arbitrarily change author identity to satisfy another checker.

### 6.6 Default-Template PR Rejected by Description Checks

- **Symptom:** The bot reports missing test plans or invalid fields despite using the default template.
- **Cause:** Default/Chinese templates and parser are misaligned, or placeholders such as `<label>` and empty `Fixes #` remain.
- **Check/fix:** Fill truthful fields according to errors and Appendix H; retrigger through supported commands. Administrators repair templates/parser together rather than making every author work around them.
- **Owner:** Author fixes content; template/CI maintainers fix shared entry points.
- **Success:** New default-template PRs pass when complete; genuinely incomplete PRs get accurate errors.

For absent issues or inapplicable checks, state the facts and agree on conventions; never fabricate numbers or mark unfinished work complete.

### 6.7 CI Does Not Start, Remains Pending, or Awaits Approval

- **Symptom:** No build, prolonged queueing, or required maintainer approval.
- **Common causes:** Missing event subscription, Draft/path filtering, fork approval, failed webhooks, unavailable resources, mismatched check names, or lost reporting.
- **Check/fix:** Identify expected Actions/Jenkins execution and whether the event/current SHA created a job. Inspect queues, approvals, runners, and reporting logs. Authorized maintainers approve Actions only after reviewing the source.
- **Owner:** Primarily CI maintainers; author supplies PR, SHA, and wait duration.
- **Success:** The matching job is discoverable, reaches a terminal result, and reports the required name.

There are currently no Actions workflows, so an empty Actions page alone does not mean Jenkins did not run.

### 6.8 Comments Report Success but Checks Is Empty

- **Symptom:** The bot reports success without a corresponding GitHub check.
- **Cause:** Integration writes only comments/labels, omits Checks/Status, or reports to the wrong SHA/repository.
- **Check/fix:** Verify current head/build evidence and query APIs from 5.5. CI maintainers add machine-readable reporting; administrators require accepted checks.
- **Owner:** CI maintainers and rules administrators.
- **Success:** Latest-commit PR Checks show accurate conclusions; failures block ordinary merge.

### 6.9 New Commits Show Old Results or Bots Repeat Comments

- **Symptom:** New heads retain old success labels, repeated comments, or report/link build-number mismatches.
- **Common causes:** Results not bound to SHA, missing deduplication, out-of-order overwrites, or links generated from inconsistent variables.
- **Check/fix:** Compare head SHA, execution time, job, and build number. Mark old results historical, cancel/isolate stale tasks, and repair reporting/idempotency.
- **Owner:** CI/bot maintainers.
- **Success:** Only matching-version builds update current results; identical comments are not repeated; reports and links identify one build.

### 6.10 403, Insufficient Bot Access, or Platform Merge-Check Failure

- **Symptom:** Forbidden responses, inability to write statuses/comments, or failed sync-platform checks.
- **Common causes:** Installation scope, permissions, token lifetime, or platform policies; service logs establish the actual cause.
- **Check/fix:** Record failed API, time, service identity, request ID, and sanitized error. Administrators inspect App scope/access; CI maintainers repair and retry the same business task.
- **Owner:** Service maintainers and access administrators.
- **Success:** Required operations work as intended and failures are traceable.

Do not automatically blame PR code for 403 or grant Admin across all repositories as the default fix.

### 6.11 Branch Updates Cause Conflicts, Signature Failures, or Stale Approvals

- **Symptom:** A nearly mergeable PR develops new blocks after updating.
- **Causes:** Base changes create conflicts; rebase changes SHAs/signatures; pushes dismiss approvals; CI still tests old code.
- **Check/fix:** Inspect new diff/commit set, resolve conflicts/signatures, test the latest version, and request incremental re-review/approval.
- **Owner:** Author handles code/commits, reviewers re-review, CI maintainers diagnose abnormal statuses.
- **Success:** Current version is conflict-free, signed, approved, and tested.

### 6.12 Sync-Bot Pushes to Protected Branches Are Rejected

- **Symptom:** GitCode updates fail to sync to GitHub master/release branches.
- **Cause:** Direct pushes conflict with PR/approval/signature rules; sync may also rewrite history.
- **Check/fix:** Inspect failed rule-suite rules, sync direction, and write method. Prefer bots creating source branches/PRs under normal rules. If mirroring is necessary, administrators design controlled access and audit separately.
- **Owner:** Migration/sync maintainers and rules administrators.
- **Success:** Sync follows its agreed model without repeated legitimate-operation rejection or unaudited shared-history rewrites.

Do not automatically retry permanently noncompliant pushes or remove all protections for convenience.

## Appendices: Checklists, Templates, and Migration Examples

### A. First-Time Developer Checklist

- [ ] Correct GitHub account; required organization invitations accepted.
- [ ] Personal fork, upstream, and target branch confirmed.
- [ ] Git/gh use expected accounts; SSH/HTTPS works.
- [ ] Repository name/email are correct and associated with GitHub.
- [ ] Signing key registered; new commits show Verified.
- [ ] CLA signed under current requirements with correct account/email mapping.
- [ ] Python uses py312; editable install points to this checkout.
- [ ] Relevant local checks run; hardware-CI scope is understood.
- [ ] Target-branch rules read; module owners/support known.
- [ ] First PR has no placeholders and includes real evidence.

### B. PR Author Checklist

- [ ] Correct base; diff/commits contain only this change.
- [ ] Title explains purpose; unfinished work is truly Draft.
- [ ] Valid issue/design links; platform numbering distinguished.
- [ ] Body includes motivation, implementation, impact, and evidence.
- [ ] Commits satisfy signing rules; CLA matches the current version.
- [ ] Latest-SHA CI verified; unrun tests stated honestly.
- [ ] Review feedback handled; formal re-review requested if needed.
- [ ] No unrelated files, credentials, or unconfirmed large artifacts.
- [ ] Interface, documentation, compatibility, and release impacts explained.

### C. Reviewer Checklist

- [ ] Current head, goal, and dependencies understood.
- [ ] Relevant expertise; valid Code Owner identity/access where required.
- [ ] Implementation, error handling, coverage, and interfaces checked.
- [ ] Distributed synchronization, communication, and memory lifecycles checked under target rules.
- [ ] Actual evidence read, not just labels or “passed.”
- [ ] Blocking findings distinguished from suggestions.
- [ ] Formal Approve/Request changes submitted after review; comments alone are not approval.
- [ ] Re-review when new commits affect conclusions.

### D. Merger Checklist

- [ ] Current head matches the reviewed/tested SHA.
- [ ] Correct target; dependencies handled as planned.
- [ ] Enough valid approvals, including Code Owner/latest-push requirements.
- [ ] Signatures comply; CLA has no unresolved conflicts.
- [ ] Required discussions resolved; no merge conflicts.
- [ ] Necessary CI completed for the current version with accessible logs.
- [ ] During migration, external CI not yet required was manually verified.
- [ ] Merge method follows target rules; no routine administrator bypass.
- [ ] Target/related issues checked after merge; release/revert follow-up if needed.

### E. Administrator Migration Acceptance Checklist

These are implementation tasks and acceptance conditions; unchecked items are not claims of completion.

| Priority | Item | Passing condition |
|---|---|---|
| P0 | CODEOWNERS validity | No errors on master/releases; each important module has a usable non-author backup |
| P0 | Team access | Visible module teams with explicit repository Write and recorded scope |
| P0 | CI/merge integration | Stable checks report to correct SHAs and are required; failures block normal merge |
| P0 | Bot signing/update method | Verifiable signatures; protected updates follow the agreed PR/sync model |
| P0 | Templates/parser | Default and both language templates match the checker; real test PR passes |
| P0 | CLA authority | Clear transition policy, identity mapping, and duplicate-check handling |
| P1 | Webhook reliability | Correct duplicate, retry, ordering, and new-head behavior |
| P1 | Build evidence | SHA, build number, report directory, and URL agree |
| P1 | Merge methods/shared branches | Global settings align with rules; shared-branch protection is explicit |
| P1 | Access/credentials | Admin list reviewed by source/responsibility; Actions/App permissions scoped |
| P1 | Exceptions | Clear bypass access, triggers, approvals, and follow-up ownership |
| P1 | Legacy tools | Clear autogit/gate-doctor boundaries and GitHub acceptance records |
| P2 | Auto-merge/cleanup | Evaluate and validate after basic quality rules stabilize |
| P2 | Ongoing maintenance | Regular member, owner, branch, token, and manual reviews |

Record date, operator, test PR, target, configuration diff, actual results, and follow-up owner for each acceptance. Before reducing access, inspect direct grants, team inheritance, and organization roles to avoid removing necessary permissions.

### F. Branch, Permission, Review, and CI Matrix

**Branch rules verified on 2026-09-18:**

| Item | master / default-branch rules | Release rules |
|---|---|---|
| Ruleset | protect-master (23637929) | protect-r1.0.0 (23641913) |
| Scope | Default branch | `r[0-9]*.[0-9]*.[0-9]*`, `release/*` |
| Minimum approvals | 1 | 1 |
| Code Owner approval | Required | Required |
| Another person approves latest reviewable push | Required | Required |
| Dismiss stale approvals | Enabled | Enabled |
| Resolve discussions | Required | Required |
| Verifiable signatures | Required | Required |
| Required checks | Rule exists, list empty | Rule exists, list empty |
| Up-to-date branch before merge | Currently strict=false | Currently strict=false |
| Merge method | Merge | Merge |
| Restrict creation, deletion, non-fast-forward updates | Enabled | Enabled |
| Always bypass | liuchongming74 | liuchongming74 |

The pattern is GitHub rule syntax, not strict semantic-version validation. Verify coverage on Rules. Creation restrictions may affect new release branches, which require an authorized release owner.

**Recommended access allocation:**

| Subject | Recommended access/scope | Notes |
|---|---|---|
| External contributor | Upstream Read + personal fork | Can create PRs |
| Module Code Owner team | Explicit repository Write | Visible team with capable reviewers |
| Routine maintainer | Write/Maintain by responsibility | Avoid Admin merely for reviewing |
| Repository administrator | Few necessary Admins | Rules, access, governance |
| CI/sync service | Exact permissions by operation | Checks, comments, and branches are distinct |
| Release owner | Necessary protected-branch/release permissions | Managed separately from ordinary contributions |

Organization base access is Read; do not raise it to Write to solve this repository's reviews. Many admins are currently visible; inspect inheritance before changing access.

### G. Common Git / GitHub CLI Commands

Replace all placeholders. This is a query/common-operation reference; history changes retain the prerequisites in the main text.

| Purpose | Command |
|---|---|
| Confirm account | `gh auth status --hostname github.com` |
| List remotes | `git remote -v` |
| Fetch upstream | `git fetch upstream` |
| Inspect worktree | `git status --short` |
| Current branch | `git branch --show-current` |
| Commits to merge | `git log --oneline upstream/master..HEAD` |
| Change scope | `git diff --stat upstream/master...HEAD` |
| Whitespace check | `git diff --check` |
| Repository PRs | `gh pr list --repo mindspore-ai/hyper-parallel` |
| Open PR in browser | `gh pr view PR_NUMBER --repo mindspore-ai/hyper-parallel --web` |
| PR diff | `gh pr diff PR_NUMBER --repo mindspore-ai/hyper-parallel` |
| Checks | `gh pr checks PR_NUMBER --repo mindspore-ai/hyper-parallel` |
| Rulesets | `gh api repos/mindspore-ai/hyper-parallel/rulesets` |
| Code Owner errors | `gh api 'repos/mindspore-ai/hyper-parallel/codeowners/errors?ref=master'` |

Read key PR state together:

~~~bash
gh pr view PR_NUMBER --repo mindspore-ai/hyper-parallel \
  --json url,state,isDraft,baseRefName,headRefOid,mergeable,mergeStateStatus,reviewDecision,reviewRequests,statusCheckRollup
~~~

Intermediate values such as `UNKNOWN` may mean GitHub is still calculating; recheck later. For empty results or 403/404, distinguish permissions, API validity, and object existence before concluding protection is absent.

### H. PR Description and Test-Result Draft Template

The following retains the main fields of the Chinese template for drafting. **Replace placeholders with facts and check boxes truthfully; this template has not passed live-parser acceptance.** Administrators still need to align default template/bot fields under 3.6.

Source templates: [Default](https://github.com/mindspore-ai/hyper-parallel/blob/master/.github/PULL_REQUEST_TEMPLATE.md), [Chinese](https://github.com/mindspore-ai/hyper-parallel/blob/master/.github/PULL_REQUEST_TEMPLATE.zh-CN.md).

~~~markdown
**What type of PR is this?**
/kind bug

----

**What does this PR do / why do we need it**:

Problem: <actual trigger, existing behavior, and impact>
Change: <implementation and behavior changes>
Scope: <modules, framework, hardware, interfaces, and compatibility>
Dependencies: <real design or prerequisite PR; explicitly state none if absent>

----

**Which issue(s) this PR fixes**:

Fixes #<ACTUAL_GITHUB_ISSUE_NUMBER>
Migration source: <actual full GitCode issue/PR URL, if applicable>

----

**Test Plan and Test result：What scenarios were tested, and what were the verification results（Function, performance, reliability, etc.）**：

- Source head SHA: <actual commit>
- Tested SHA: <actual commit; include base SHA for test merges>
- Environment: <OS, Python, framework, device model/count>
- Local tests: <actual commands, test counts, pass/fail results>
- CI tests: <build number, actual scope, conclusion, report URL>
- Performance/compatibility: <measurements, or why inapplicable>
- Outstanding validation: <missing conditions, execution environment, owner>

----

**Self-checklist**: (**Review each item and mark [x] truthfully; completion is checked and missing requirements may prevent merge.**)

- [ ] **Design**: Maintainer review completed; all design feedback answered and changes implemented.
- [ ] **Tests**: Sufficient UT/ST coverage; new tests included here or already merged.
- [ ] **Validation**: Detailed results demonstrate the intended Feature/Refactor/Bugfix goals.
- [ ] **Interfaces**: Public API changes approved by interface reviewers; API documentation updated correctly.
- [ ] **Documentation**: If website documentation changes are needed, submit them promptly to the documentation repository.
~~~

`/kind bug` applies only to bug fixes; choose the actual type allowed by the current template otherwise. For absent issues/inapplicable checks, state facts and agree on conventions rather than leaving empty numbers or falsely checking boxes.

Supplement results with this table; all values come from actual execution:

| Scenario | Environment / scale | Command or build | Actual result | Evidence |
|---|---|---|---|---|
| Relevant UT | Actual environment | Actual command | Actual pass/fail counts | Log location |
| Distributed ST | Device, framework, count | Actual build number | Real conclusion | Report URL |
| Performance regression | Workload and baseline | Actual method | Measured change/explanation | Comparison report |
| Not run | Missing conditions | Follow-up plan | Not run | Owner/entry point |

### I. Legacy GitCode Tools and GitHub Alternatives

At this inspection, repository `autogit` still says “Only GitCode is supported,” and the documented `gate-doctor` workflow targets GitCode. The same tool name does not prove GitHub adaptation is complete.

| Legacy entry / behavior | GitHub migration procedure | Later acceptance |
|---|---|---|
| autogit fork/PR creation | Git/gh/pages in Chapters 2 and 4 | Account, remotes, base/head |
| autogit commit | Local signed Git commits with applicable checks | Style, signing, hook consistency |
| autogit local checks | Only confirmed GitCode-independent portions | Inspect remote calls/platform assumptions |
| gate-doctor / comment retests | Current GitHub CI entry points/commands | Trigger and reporting validation |
| GitCode username/Owner | Map GitHub logins/teams | Existence and effective access |
| Old repository links | GitHub remotes for new development | Keep needed history; label platform |
| Agent/IDE hooks | Verify the active tool actually runs hooks | Reproducible checks/CI as common evidence |

Until adapted, legacy scripts should not be the default GitHub submission/merge entry points. Style, testing, and design constraints still apply.

### J. Real Migration Examples and Relevant Sections

This table is a **2026-09-18 audit snapshot**. PRs may since have changed, closed, or merged. Examples concern process issues, not direct claims of code defects.

| Example | Observation at the time | Sections |
|---|---|---|
| [PR #858](https://github.com/mindspore-ai/hyper-parallel/pull/858) | All incoming commits signed; Jenkins success comment existed, but native CI integration remained incomplete | 2.4, 5.5, 6.8 |
| [PR #846](https://github.com/mindspore-ai/hyper-parallel/pull/846) | Previously blocked by ownership/signatures, later bypass-merged; Verified merge commit did not prove source commits repaired | 3.9, 6.2, 6.4 |
| [PR #852](https://github.com/mindspore-ai/hyper-parallel/pull/852) | COMMENTED reviews without valid approval and unresolved discussions | 4.8, 4.9 |
| [PR #864](https://github.com/mindspore-ai/hyper-parallel/pull/864) | Invalid compile-path owners prevented expected review requests; many historical bot comments | 3.3, 6.1, 5.9 |
| [PR #866](https://github.com/mindspore-ai/hyper-parallel/pull/866) | Only some commits passed signing; conflicts existed | 6.4, 6.11 |
| [PR #848](https://github.com/mindspore-ai/hyper-parallel/pull/848) | Bot commit had UNKNOWN_KEY | 2.4, 6.4 |
| [PR #867](https://github.com/mindspore-ai/hyper-parallel/pull/867), [PR #868](https://github.com/mindspore-ai/hyper-parallel/pull/868) | Placeholders, description failures, and repeated bot feedback exposed template/parser mismatch | 3.6, 6.6, 5.9 |
| [PR #867 CLA comment](https://github.com/mindspore-ai/hyper-parallel/pull/867#issuecomment-5728272209) | GitHub and sync-platform CLA results disagreed | 2.5, 6.5 |
| [PR #844 report comment](https://github.com/mindspore-ai/hyper-parallel/pull/844#issuecomment-5729251542) | Report path and clicked link used different build numbers | 5.7, 6.9 |

Direct sync-bot updates to protected release branches were also rejected; repair the model under 6.12. Historical runs used rules later changed. Read execution times with contemporaneous rule snapshots rather than presenting old failure conditions as current configuration.

### K. Ownership, Support, and Escalation

| Problem | Primary role | Information needed | Escalation |
|---|---|---|---|
| SSH, commit email, signing | Author | Sanitized error, SHA, signing method | Organization/identity administrator |
| CODEOWNERS/team access | Repository administrator, module lead | PR, base, paths, invalid entries | Maintenance team |
| Formal reviews/discussions | Reviewer/Owner | Current SHA, pending feedback | Backup module maintainer |
| Description/CLA | CI/CLA maintainer | Both URLs, SHA, exact error | Migration lead |
| Test failure | Author/CI initial triage | Build ID, logs, environment, reproduction | Module/hardware-platform lead |
| Repeated comments, 403, sync failure | Bot/sync maintainer | Event time, request ID, source/target | Repository/organization administrator |
| Rule bypass/access change | Rules administrator | Business reason, impact, evidence, recovery | Project maintainers |

Record public issues in the relevant PR Conversation. Cross-PR platform defects can use trackable [Issues](https://github.com/mindspore-ai/hyper-parallel/issues) linking affected PRs. Team entry: [HyperParallel Maintainers](https://github.com/orgs/mindspore-ai/teams/hyperparallel-maintainers).

This manual invents no named owners for CI/CLA roles. The migration lead should map roles to real accounts and agreed response times when publishing. Handle credentials and unsanitized service logs through approved private channels.

### L. Changelog and Maintenance Requirements

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-09-18 | Based on migration PR/configuration audits; establishes local setup, access/rules, PR collaboration, CI/CD, troubleshooting, and checklists, distinguishing current state from proposals |

Future edits must keep prose, matrices, examples, and FAQ consistent. When teams, CODEOWNERS, rulesets, checks, template fields, CLA, or sync change, record the configuration change/acceptance PR and update current-configuration verification dates.

GitHub labels and rule capabilities may change. Follow the target repository's current Rules page and the official documentation linked here when operating.
