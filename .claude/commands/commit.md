# /commit — Stage, Check, Commit, Push

Automate the commit workflow using AutoGit.

## Steps

1. **Check for changes**: Run `git status` to verify there are uncommitted changes. If none, inform the user and stop.

2. **Stage all changes**: Run `git add -A` to stage all changes.

3. **Run lint checks** (unless user passed `--no-check`):
   ```bash
   python3 .claude/skills/autogit/scripts/autogit.py check
   ```
   If checks fail, show the errors and ask the user to fix or use `--no-check`.

4. **Generate commit message**: If no `-m` message provided:
   - Analyze staged changes via `git diff --cached --stat` and `git diff --cached`
   - Categorize: `feat:` / `fix:` / `refactor:` / `docs:` / `test:` / `chore:`
   - Keep subject under ~80 chars, imperative voice
   - Show the message and ask for confirmation

5. **Commit and push**:
   ```bash
   git commit -m "<message>"
   git push origin <current-branch>
   ```

6. **Report result**: Show commit hash and push status.

## Arguments

- `-m "message"` — Specify commit message directly
- `--no-check` — Skip lint checks (emergency only)

## Safety

- Never force-push
- Never commit to master/main directly — warn and create a feature branch if on master
- If push fails due to conflict, suggest `git pull --rebase origin <branch>`
