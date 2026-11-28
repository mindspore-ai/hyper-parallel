# Automatic RSI experience consolidation

Every RSI task delivers two distinct artifacts: the experiment acceptance report
and an **RSI improvement report**. The second explains what the framework learned,
what already existed, which repository files changed, and how they were validated.
It is required even when no new experience is found. Do not wait for another user
instruction to perform the retrospective.

## Policy

| `rsi_mode` | Repository updates | Agent responsibility |
|---|---|---|
| `acceptance` (default) | At final acceptance | Collect evidence during optimization; reconcile all outcomes before delivery |
| `continuous` | Between completed experiments, after cleanup | Reconcile qualified evidence as it appears; reconcile the entire campaign again at acceptance |

Expose both choices when discussing the task. If no preference is given, use
`acceptance` without delaying startup. Mode is frozen per run. Performance
acceptance and completed experience consolidation remain separate states.

The Agent performs extraction, causal interpretation, semantic duplicate review,
and patch design. The deterministic tool contains no additional LLM. It discovers
the current repository structure, retrieves related rules, checks evidence and
review coverage, applies changes, runs regressions, and generates the report.
Lexical retrieval assists semantic review; it does not prove novelty.

## Required Agent workflow

1. The Harness initializes `rsi/config.json` and invokes the shared tool after
   experiment cleanup. It inventories all outcomes, including crashes, numerical
   failures, noise-band results, and negative experiments.
   Default-mode iterations queue the evidence index without rescanning the whole
   corpus or applying patches. Full review runs at acceptance; continuous mode
   reviews the corpus at each completed experiment boundary.
2. Read `rsi/review-task.json`, the retrieved matches, and current owning Skill
   entrypoints. Architecture and retrieval are regenerated from the checkout;
   no fixed Skill count or historical directory map is assumed.
3. Write `rsi/review.json`. Account for every experiment, explaining when no
   reusable lesson exists. Search the current corpus before claiming novelty.
4. Invoke `learn`. Default mode queues changes during optimization; continuous
   mode validates and applies them between experiments.
5. Before final delivery, invoke `finalize`. Missing review, changed evidence,
   stale architecture, incomplete coverage, and failing regressions prevent
   successful finalization. Resolve them and rerun; disclose unresolved blockers.

`autoresearch` creates a separate local commit containing only validated learning
paths, preserving its clean-target protocol. MFU Riser applies validated worktree
changes and records their hashes. Neither path pushes, opens a PR, or publishes.
Follow the task's existing Git authorization separately.
An automatic learning commit rejects pre-existing edits in its destination files;
it never folds another contributor's pending work into the learning commit.

## Classify before writing

| Decision | Treatment |
|---|---|
| `new` | Add qualified, useful experience absent from the corpus; explain how retrieved matches differ |
| `update` | Improve the existing guidance or implementation in place; identify its current hash and limitation |
| `reuse` | Cite existing files without creating another note or registering a duplicate legacy rule |
| `defer` | Keep an unqualified screen or untested hypothesis in the local report, with missing evidence |
| `reject` | Explain why a finding is local, redundant, unsupported, or not reusable |

Registered equivalent claims are automatically reused. Semantically equivalent
wording requires Agent review of the retrieved corpus. Existing rules can gain
applicability conditions, counterexamples, checks, or regression coverage instead
of duplicate paragraphs in other Skills.

Do not publish machine names, pod IDs, IPs, account names, mount paths, dashboard
URLs, credentials, or site-specific launch commands. Keep these in raw run evidence.
Preserve useful technical constraints: supported shape/dtype/layout, API availability,
hardware capability, allocator behavior, and evidenced software version bounds.
Describe topology as communication domains and bandwidth tiers, not host lists.
The tool blocks common deployment identifiers; the Agent must also review semantic
portability, including identifiers that a pattern detector cannot recognize.

## Routing into the current architecture

| Reusable result | Destination |
|---|---|
| Diagnostic or tuning procedure | Existing owning Skill reference; genuinely new rules go under its `references/experience/` and are linked from its entrypoint |
| Correction to an existing rule | Its existing file, without an additional experience note |
| Deterministic acceptance/accounting invariant | Existing Harness scripts plus positive and negative regressions |
| Repeated collection or analysis operation | Owning Skill's `scripts/`, linked from its workflow |
| Runtime defect | Record the diagnostic and verified repair strategy in the owning Skill. Review the training-module patch through the normal framework workflow. |
| Local deployment workaround | Run directory only |

The shared implementation is [rsi_learning.py](../../../scripts/rsi_learning.py).
It discovers current owners, checks patch base hashes, rejects path escapes, and
rolls back repository writes on validation failure. Executable changes require
regression files and passing argv commands. It cannot rewrite this run's measured
source or frozen validators. Use `learning_repository` to consolidate into a
separate checkout when necessary; changing a comparison requires a new series.

`.agent/learning/registry.json` stores portable claims, owners, forms, revisions,
code locations, and evidence digests. Raw run paths and deployment identities
stay in local reports. New source and documentation use the repository's English
terminology and code style.

## Review contract

Write lesson drafts with `id` and `keywords`, then refresh retrieval:

```bash
python .agent/scripts/rsi_learning.py inspect --repo . --run-dir outputs/example-run
```

Fill `inventory_digest`, every experiment's ID and digest, and `related_review`.
The following illustrates a procedure proposal; inspect the repository before
claiming that this example is new:

```json
{
  "schema": "hp-rsi-learning-review/v1",
  "inventory_digest": "<architecture digest>",
  "coverage": [{
    "event_id": "<experiment ID>", "sha256": "<observed digest>",
    "lesson_ids": ["logical-state-identity"], "reason": "Initialization confounder reproduced"
  }],
  "lessons": [{
    "id": "logical-state-identity", "decision": "new", "owner": "precision",
    "form": "procedure", "qualification": "validated",
    "title": "Logical initial state across layouts",
    "keywords": ["logical", "initialization", "layout"],
    "trigger": "Two numerical runs use different sharding layouts.",
    "mechanism": "Rank-local random streams can initialize different logical weights.",
    "claim": "Verify logical initial state before interpreting cross-layout numerical differences.",
    "applicability": "Equivalent models with different shard placement.",
    "limits": "Initial-state identity alone does not prove gradient or convergence equivalence.",
    "reason": "Explain novelty after reviewing current precision procedures.",
    "related_review": [{
      "path": "<retrieved file>", "sha256": "<current digest>",
      "relationship": "different", "reason": "Explain the actual scope difference"
    }],
    "evidence": [{"path": "identity-check.json", "sha256": "<actual evidence digest>"}]
  }]
}
```

If a rule exists, use `reuse` or `update`. A validated failure may use qualification
`negative` for a bounded counterexample or detection rule. Unqualified screens and
untested hypotheses use `defer` and do not enter repository guidance.

For code or an existing document, add `changes` with `path`, `before_sha256`
(`null` for a new file), and complete UTF-8 `content`. Executable changes also
require `regression_tests` and
`validation: [{"argv": ["python", ".agent/tests/tools/test_example.py"], "timeout_seconds": 120}]`.

Repository updates performed by consolidation belong under `.agent/`. Keep
framework implementation changes in the measured experiment checkout and
link to their evidence when recording a reusable lesson.
Use real repository-relative test commands. A command that merely exits zero is
not validation. Preserve the demonstrated failure mechanism and a valid case.

## Commands and outputs

```bash
python .agent/skills/autoresearch/scripts/autoresearch.py learn --run-dir outputs/example-run
python .agent/skills/autoresearch/scripts/autoresearch.py finalize --run-dir outputs/example-run
python .agent/skills/mfu-riser/scripts/mfu_riser.py learn --run-dir outputs/example-run
python .agent/skills/mfu-riser/scripts/mfu_riser.py finalize --run-dir outputs/example-run
```

Use the entrypoint owning the run, not both. External adapters use
`rsi_learning.py sync --repo . --run-dir ... --phase acceptance`. They supply
`rsi/experiments.json` with `experiments: [{id, path, sha256, decision}]`; paths
are relative to the run and must cover the complete campaign.

Each reconciliation writes `RSI_REPORT.html`, `RSI_REPORT.md`, and `rsi/report.json`:
experience, decision, form, code locations, qualification, test results, and
unresolved items. `rsi/applied.json` records review and file digests for idempotent
finalization. Failed validation retains local logs and restores repository files.
