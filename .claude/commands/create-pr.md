# /create-pr — Create Pull Request to Upstream

Create a PR from the current feature branch to upstream/master.

## Prerequisites

- `GITCODE_TOKEN` environment variable must be set
- Remotes: `origin` (fork), `upstream` (main repo)
- All changes must be committed (no dirty working tree)
- Must be on a feature branch (not master/main)

## Steps

1. **Verify prerequisites**:
   - Check `GITCODE_TOKEN` is set
   - Check remotes exist (`git remote -v`)
   - Check no uncommitted changes (`git status`)
   - Check current branch is not master/main

2. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```
   If rebase conflicts, abort and inform user.

3. **Push to origin**:
   ```bash
   git push origin <branch> -u
   ```

4. **Analyze changes for PR description**:
   - Run `git log upstream/master..HEAD --oneline` for commit list
   - Run `git diff upstream/master...HEAD --stat` for file changes
   - Categorize: feature domain, purpose, affected modules

5. **Generate PR content**:
   - **Title**: Short (<70 chars), format: `[type] description`
   - **Body**: Summary of changes, affected modules, test status

6. **Create PR** via AutoGit or API:
   ```bash
   python3 .claude/skills/autogit/scripts/autogit.py pr
   ```
   Or fallback to GitCode API if autogit unavailable.

7. **Report**: Show the PR URL.

## Arguments

- `--to #N` — Append commits to existing PR #N
- `--squash` — Squash all commits into one before creating PR
- `--reviewer <users>` — Assign reviewers (comma-separated)

## Safety

- Never force-push to master/main
- Rebase failures auto-abort
- Always show PR content for user confirmation before creating
