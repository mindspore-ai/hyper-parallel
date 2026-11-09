# GitHub: automatically merge after the reviewer approval quorum

**English** | [简体中文](github_auto_merge.zh.md)

This feature determines the reviewers from the **Reviewers** sidebar of each PR.
It does not require every member of the repository's maintainer team to participate.
It takes effect after the code is merged into the default branch and the repository settings below are configured.
Adding workflow files alone does not change remote rules.

## Approval rules

The required status is named `all-reviewers-approved` and is published on the PR's current head SHA.
The name is retained for compatibility with existing Rulesets; the check now requires an approval quorum.

GitHub's `requested_reviewers` contains only people whose reviews are still pending. Once a person submits a review,
they move from that array to the reviews collection. As in the existing gate, the check combines outstanding requests
with submitted reviews, excluding the PR author. Submitted reviews include bot and unsolicited reviews.
Ordinary comments in the PR's Conversation tab are not formal reviews and do not add reviewers.
See the [GitHub Review Requests API](https://docs.github.com/en/rest/pulls/review-requests).

- With one reviewer, one approval is required. With two or more, any two distinct approvals
  meet the quorum; the remaining requested reviewers need not approve.
- An approval counts only if the reviewer's latest effective decision is `APPROVED` for the current head SHA and the
  reviewer has not been asked to review again. Dismissed and stale approvals do not count.
- An outstanding `CHANGES_REQUESTED` decision from any non-author reviewer, including a bot, blocks the check even after the quorum is met.
  A `COMMENTED` review is not an approval.
- An ordinary review comment after approval does not cancel that approval. Unsubmitted draft reviews are ignored.
- A person removed from Reviewers before submitting a review leaves the reviewer set. People who have already submitted
  reviews remain in the set. Adding a second reviewer raises the quorum from one approval to two.
- The PR author is excluded. A PR with no other reviewers does not pass. Bot reviews follow the existing reviewer rules.
- A team still present in `requested_teams` blocks the check until GitHub considers its review request satisfied.
  The check does not expand a team into all of its members.
- Draft PRs do not pass. When multiple PRs share the same head SHA, all relevant PRs must satisfy the rules before the
  status is marked successful.

For example, if Reviewers contains Alice, Bob, and Carol, approvals from Alice and Bob meet the quorum without waiting
for Carol, provided no other blocking review or required team request remains. The repository's native Code Owner,
latest-push, conversation-resolution, and CI requirements still apply before GitHub merges the PR.
This custom check does not verify repository write access, so an unsolicited approval can contribute to the quorum.
GitHub's native approval and Code Owner rules independently enforce their own eligibility requirements.

## Files and triggers

| File | Purpose |
| --- | --- |
| [Policy](../../.github/reviewer-approval-policy.json) | Repository, target branches, and auto-merge switch; no fixed reviewer roster |
| [Check logic](../../.github/scripts/reviewer-approval-gate.mjs) | Read live approvals, publish the required status, and enable auto-merge |
| [Check workflow](../../.github/workflows/reviewer-approval-gate.yml) | Run in the trusted default-branch context |
| [Review event signal](../../.github/workflows/reviewer-approval-signal.yml) | Forward read-only review events from fork PRs to the trusted workflow |
| [Tests](../../.github/scripts/reviewer-approval-gate.test.mjs) | Approval states, reviewer changes, concurrent updates, pagination, and write boundaries |

The policy covers `master`, `r1.0.0`, and `r1.1.0`. Add explicit branch names to the policy when introducing release branches.
All target branches use the policy and script from the default branch, so a PR cannot approve itself by changing its own
policy or script. If the default branch is renamed from `master`, also update `push.branches` in the workflow.

Creating, updating, reopening, closing, or changing the draft state or Reviewers of a PR triggers the check.
Review submissions, edits, and dismissals trigger `workflow_run` through a separate signal workflow.
The signal does not execute PR code, download artifacts, or use secrets. The subsequent workflow queries GitHub again
for the current state. The main workflow checks out only the default branch, never the PR head.

Every write run reconciles all open PRs on the configured target branches. Runs execute serially so that coalescing queued
events does not leave other PRs unchecked. A scheduled reconciliation runs every 15 minutes as a fallback, and a manual
`workflow_dispatch` entry point is available. The workflow does not post comments, request reviews, or change Reviewers.
It does not repeatedly publish the same status while a PR remains waiting with an unchanged result.

## One-time setup

1. Merge these files into the default branch. Keep appropriate code review requirements for workflows, scripts, and
   policy files under `.github/`.
2. Manually run **Reviewer approval gate** in Actions so that `all-reviewers-approved` first appears on open PRs.
3. In the **active Ruleset** protecting the target branch, add `all-reviewers-approved` to
   **Require status checks to pass before merging**, selecting **GitHub Actions** as its source (App ID `15368`).
   Keep the existing `label-check` and other quality checks.
4. Keep **Require a pull request before merging**, at least one required approval, and
   **Dismiss stale pull request approvals when new commits are pushed** enabled. Existing requirements for Code Owners,
   independent approval of the most recent push, and conversation resolution continue to apply.
5. Enable **Allow auto-merge** under **Settings → General → Pull Requests**.
6. Verify with test PRs: one reviewer should require one approval; two or more reviewers should require two approvals,
   without waiting for a third. Confirm that other required checks pass before automatic merging. Also test adding or
   removing reviewers, change requests, pending teams, dismissed approvals, and new commits.

After the approval check passes, the script examines the rules actually enforced on the target branch.
It runs the merge command with `--auto --merge --match-head-commit` only when this check is bound to GitHub Actions,
reviews and dismissal of stale approvals are required, and the repository allows auto-merge.
It uses the repository's currently permitted merge-commit method and does not bypass rules with `--admin`.
If the repository setting or those conditions are missing, it only publishes the approval status without enabling
automatic merging.

This implementation supports auto-merge through Rulesets only. It does not modify Rulesets, classic branch protection,
or bypass lists. The GitHub Actions token cannot administer repository rules. To apply the gate to administrators too,
configure the corresponding bypass policy in the repository rules. If Merge Queue is used, the approval check remains
active, but this implementation does not enqueue PRs automatically; queue authorization must be configured separately.

## Local validation

The check script requires Node.js 22 or newer and an authenticated GitHub CLI. It does not depend on Python or a framework
runtime.

```bash
node --test .github/scripts/reviewer-approval-gate.test.mjs

GH_REPO=mindspore-ai/hyper-parallel \
  node .github/scripts/reviewer-approval-gate.mjs --dry-run --pr 1335
```

Omit `--pr` to inspect all open PRs in read-only mode. Dry-run is also the default.
Read-only mode does not publish statuses, enable auto-merge, or merge PRs. Writes require the explicit `--write` option
in the workflow. After updating this document, run the repository catalog check in the designated environment:

```bash
conda run -n py312 python .agent/scripts/check_agents_catalog.py
```

## Operational limitations

This is an asynchronous GitHub Actions check. There is a delay between an approval or reviewer change and publication
of the updated result. Workflow queues, unavailable APIs, and delayed GitHub schedules can extend that delay.
Before publishing success, the script rechecks commits, target branches, reviewers, and approvals. On API errors, it
attempts to replace a previous success with a blocking status. However, the external required check and GitHub's merge
operation are not one atomic transaction, so a zero-length race window cannot be guaranteed.
Keep GitHub's native approval requirements enabled. This external check is not an atomic GitHub approval guarantee.

Run logs and the Actions Summary show the approval count and threshold, pending users and teams, and change requests.
The PR status links to the
workflow that published the result. Before enabling the feature in production, verify event delivery, token permissions,
required-check binding, and actual automatic merging on GitHub. Local tests do not replace this acceptance testing.

## The 24-hour validity period for CI pass labels

`ci-pipeline-passed` and `ci-pipeline-passed-but-overtime` each expire 24 hours after the `created_at` timestamp of their
most recent GitHub `labeled` event. When both labels are present, their lifetimes are calculated independently.
`label-check` can pass as long as either label is still valid. Comments, reviews, and changes to other labels do not renew
the lifetime. Removing and re-adding a label restarts its clock. Adding an already-present label does not renew it unless
a new `labeled` event is generated.

The [scheduled cleanup workflow](../../.github/workflows/expire-ci-pass-labels.yml) checks open PRs every five minutes
and can also be run manually from Actions. It executes only the [script](../../.github/scripts/ci-label-expiry.mjs) from
the default branch, using the built-in `GITHUB_TOKEN`. No new secrets or required checks are needed. Closed or merged PRs
are excluded. The feature takes effect after merging into the default branch and also applies to labels added before
rollout. If a label has no verifiable addition record, the workflow reports an error instead of guessing its age.

The scheduled job invalidates outdated passing GitHub Actions `label-check` results before deleting expired labels.
This is necessary because removing a label with `GITHUB_TOKEN` does not trigger another `unlabeled` workflow; deletion
alone would leave an old success in place. The script inspects results on both the PR head and the current test merge
commit. If the other label is still valid, only the expired label is removed and the passing check is preserved.
The regular label check also validates addition timestamps directly. Completion of **Label Gate Check** triggers another
reconciliation for that head to handle jobs that finish with an outdated success during label cleanup.
Scheduled scans skip PRs without CI pass labels to reduce API usage; completion-event reconciliation also checks PRs
without those labels.

GitHub Actions schedules can be queued or delayed, so removal occurs on the next execution after the 24-hour expiry.
Neither removal at the exact second nor a maximum delay of five minutes is guaranteed. Labels, check statuses, and
merging are not an atomic transaction. The script rechecks the current commit and label events before writes to narrow
the concurrency window. Conflicts and API errors are reported in logs. Scheduled scans retry on the next run; a failed
completion-event reconciliation can be rerun from Actions.
See [GitHub workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).

After new commits are pushed, the existing [Reset CI pass labels](../../.github/workflows/reset-ci-pass-labels.yml)
workflow still clears both labels without waiting for 24 hours. To renew a label, rerun CI and let CI remove the old label
and add a fresh result. This feature does not start Jenkins tests or dismiss reviewer approvals when labels expire.

Local tests and a read-only preview:

```bash
node --test .github/scripts/ci-label-expiry.test.mjs
GH_REPO=mindspore-ai/hyper-parallel node .github/scripts/ci-label-expiry.mjs --dry-run --pr 1356
```

Only the trusted workflow uses `--write`. After merging, manually run the cleanup workflow once to verify label deletion
and invalidation of old `label-check` successes. Local mock tests do not replace validation of write permissions on GitHub.
