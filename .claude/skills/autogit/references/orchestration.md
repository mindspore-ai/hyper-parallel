# AI Orchestration (autogit)

How Claude (or any agent) should drive `autogit` end-to-end. Companion to
`SKILL.md`, which only sketches the high-level pipelines.

The user types **one** intent. The agent calls **one** autogit subcommand,
then handles per-step gate questions in chat. The script enforces interaction
at every choice and content point — AI cannot silently bypass.

---

## "给我创建 commit" (or `/autogit commit`)

1. Generate a Conventional Commits message from `git diff --cached`.
2. **Show the message to the user in chat.** Wait for OK / edit / cancel.
3. Run `autogit commit -m "<approved msg>"`.
4. Script stages, prints a non-blocking warning if the pre-commit hook is
   missing, runs `git commit` (the hook auto-runs lint), then pushes.
5. **No UT/ST gate at commit time** — heavy regression checks belong to PR
   time, commits should stay cheap.

## "给我提交 PR" (or `/autogit pr`)

1. Run `autogit pr` (no flags).
2. Script errors with the two undecided PR-time gates (UT, ST), each
   accepting `skip` / `changed` / `full`. The error includes the exact
   `--ut … --st …` flags.
3. Ask the user **one at a time** in chat (showing the PR-diff file list):
   - "UT? c=changed (本 PR 涉及的 UT 测试) / f=full (全量 tests/ut) / n=skip — default c"
   - "ST? c=changed / f=full / n=skip — default n (需多卡环境)"
4. Generate PR title+body from
   `autogit pr --analyze-only --ut skip --st skip` + raw diff +
   `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`.
   **Show to user**, get OK / edit / cancel.
5. Run `autogit pr --ut <decision> --st <decision> --title "..." --body "..."`.
6. Script runs gates (skip / scoped pytest / full pytest), then creates PR.

If the working tree is dirty when AI runs `autogit pr`, the script errors
"Uncommitted changes detected, please commit first" — AI then runs the
**commit** subflow above, then re-runs `autogit pr`.

## "追加 commit 到现有 PR" (or `/autogit pr --to N`)

Same shape, but only the UT gate is asked (ST is skipped by design for the
lighter append flow). For `--ut changed`, the scope is the local working-tree
diff (staged + unstaged), since the gate runs before any branch operations.

---

## Failure self-loop

Each stage failure returns non-zero with clear text. AI:

1. Read the error.
2. If fixable in code (lint warning, test assertion, type error) — apply the
   fix and re-run the same stage.
3. If the same root cause repeats across attempts — stop, surface the
   failing output to the user, ask for direction.

No hard retry cap, but **never silently move past a failure**. Repeated
same-root-cause failures mean the AI's mental model is wrong — consult.

---

## PR description content (fill the template)

Read JSON analysis + raw diff, fill each section of
`.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`:

- **What type of PR is this?** → `/kind` label inferred from commit types
  (feat→feature, fix→bug, refactor→refactor, chore/docs/test→task,
  style/cleanup→clean_code)
- **What does this PR do / why** → reason, architecture, API changes,
  file stats, affected functionality
- **Which issue(s) this PR fixes** → extracted from commit messages (`#NNN`)
- **Test Plan and Test result** → test file coverage from diff analysis
- **Self-checklist** → kept from template (user fills)

If the template file is missing, the script falls back to a minimal inline
template.

### Writing rules

1. **Chinese** for all content, English only for code identifiers
   (`` `func_name` ``)
2. **Explain the "why"** — what problem existed, how the change addresses it
3. **Be specific** — reference function names, class names, file paths with
   inline code
4. **Group by module/file** when showing changes
5. **Include statistics** from the analysis (file counts, line counts)
6. **Do NOT include** `test_*` method names, `Test*` class names, or
   `_private()` references
7. Keep total body **30-50 lines**

---

## ANTI-PATTERNS (NEVER do these)

The script-level enforcement exists because of these exact past failures:

1. **Generating a commit message and passing `-m "..."` without showing the
   user first.** Every commit message must be user-approved.
2. **Generating PR title/body and passing `--title/--body` without showing
   the user first.** Same rule.
3. **Adding `--ut skip --st skip` because "the script asks too much".** Only
   skip if the user explicitly chose to skip.
4. **Catching a non-tty `AutoGitError` and retrying with skip flags.** The
   error is asking you to ask the user — read it, then ask.
5. **Self-looping past three failures of the same kind without checking in.**
   Repeated failures mean wrong mental model — stop.
6. **Chaining `autogit commit` → `autogit pr` in one turn without surfacing
   both the commit message and the PR description in chat.** Even with
   script enforcement, the chat must show what is about to happen.

### Rationalizations to reject

- "It will be fine, I'll just force-push quickly" — NO. Confirm with user.
- "Nobody else is using this branch" — You don't know that. Ask first.
- "Tests are too slow, I'll just `--ut skip`" — Only skip if user says so.
