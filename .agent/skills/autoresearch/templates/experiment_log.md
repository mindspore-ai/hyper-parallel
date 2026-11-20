# Autoresearch experiment log

Append one entry per experiment. Never rewrite previous entries. Read the log at the
start of each iteration to avoid repeating failed approaches.

## Entry format

```markdown
## <Short title> — <status> (<commit>)

- **Hypothesis:** What was tried and why it was expected to help.
- **Change:** What was modified; summarize rather than pasting the full diff.
- **Engagement and gate:** Executed path, evidence location, and gate result. On failure, record the first failing stage and assertion.
- **Result:** Primary metric, measurement convention, and breakdown, or reason for noncompletion.
- **Analysis:** Why it helped, had no effect, or failed.
- **Lesson:** What to retain or avoid in future iterations.
- **Retry conditions:** Configuration changes that justify retrying, and how to reapply the candidate to the current best version.
```

---

(Empty. Seeded by the experiment owner or populated by the Agent.)
