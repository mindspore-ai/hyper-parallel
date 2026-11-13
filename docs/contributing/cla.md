# CLA checks and PR labels

[中文](cla_zh.md) | English

HyperParallel uses the hosted [CLA Assistant](https://github.com/cla-assistant/cla-assistant) integration.
It publishes the `license/cla` commit status for contributors to a PR. Contributors follow that status's Details
link to sign or recheck the agreement. Signature records are managed by CLA Assistant, not by GitHub labels.

## Automatic creator labels

[Sync CLA labels](../../.github/workflows/cla-label-sync.yml) queries the **PR creator's own signature**:

| Personal query result | Label |
| --- | --- |
| The creator has signed the current agreement | `cla-signed` |
| The creator has not signed the current agreement | `cla-unsigned` |
| Missing credentials, missing agreement, service failure, or an invalid response | Synchronization fails; existing labels remain unchanged |

The [script](../../.github/scripts/cla-label-sync.mjs) uses the PR's `user.login` and `user.id` to query
CLA Assistant's `hasSignature` endpoint. It omits the PR number so the service checks the current personal signature.
It first verifies that an agreement is configured, because an absent agreement is not proof of signing.
See the service's [API](https://github.com/cla-assistant/cla-assistant/blob/main/src/server/src/api/cla.js)
and [signature lookup](https://github.com/cla-assistant/cla-assistant/blob/main/src/server/src/services/cla.js).

The existing `license/cla` status can cover multiple contributors. Its success or failure is never used as the
creator's signature result. For example, a signed creator receives `cla-signed` even if another contributor's
missing signature makes the aggregate check pending. This does not change the existing CLA integration.

After successful synchronization, the two labels are mutually exclusive. The workflow also removes obsolete
`mindspore-cla/yes`, `mindspore-cla/no`, `signed-CLA`, and `unsigned-CLA` labels from the PR, preserving unrelated labels.
Repository administrators should delete the old repository label definitions during rollout; existing assignments
must not be copied to the new labels as signature evidence. A new commit triggers a personal recheck; signing is
not tied to a particular commit, so a creator who remains signed retains `cla-signed`.

## Required setup

The personal query is an authenticated, administrator-only CLA Assistant API. Before enabling the workflow,
configure the repository Actions secret **`CLA_ASSISTANT_TOKEN`** with a dedicated credential accepted by CLA Assistant
for an authorized administrator of this repository. The account must already be authorized in CLA Assistant;
an arbitrary GitHub token is not sufficient. The built-in `GITHUB_TOKEN` only performs GitHub label operations
and cannot replace this credential. Do not put the credential in code, PR descriptions, or chat messages.

Add the secret under **Settings → Secrets and variables → Actions → New repository secret**. Missing or rejected
credentials cause a visible workflow failure, never a `cla-unsigned` result. After merging the workflow into the
default branch, manually run **Sync CLA labels** to verify the credential and reconcile existing open PRs.
Automated tests use simulated responses; a successful authenticated query against the hosted service is still
required during rollout. If the account or hosted API changes, verify this integration again.

## Triggers and safeguards

Synchronization runs when a PR is opened, reopened, or updated, and when `license/cla` changes. Status events are
only wake-up signals for a fresh personal query. Manual dispatch and a six-hour fallback scan reconcile open PRs;
closed and merged PRs are skipped. If signing does not produce a status event for the PR, the fallback scan or a
manual run refreshes its label. Updates are asynchronous and scheduled runs may be delayed by GitHub.

Only default-branch code receives credentials. The workflow never executes PR code or downloaded artifacts.
The CLA credential is sent only to the fixed CLA Assistant signature endpoint, with redirects disabled.
Before label mutations, the script rechecks the PR creator, current head, signature, agreement, and managed labels.
Concurrent changes cause a retryable error. Labels and external signature changes are not one atomic transaction.

Query failures leave previous labels in place, so labels represent the last successful query, not a live guarantee.
Rerun failed workflows or use manual dispatch after resolving errors. Labels do not replace signature records or
create a merge requirement. This feature does not modify Rulesets or branch protection.

## Validation

```bash
node --test .github/scripts/cla-label-sync.test.mjs
# Authenticate gh and supply CLA_ASSISTANT_TOKEN securely in the environment first.
GH_REPO=mindspore-ai/hyper-parallel node .github/scripts/cla-label-sync.mjs --dry-run --pr 1365
```

Read-only mode is the default; only the trusted workflow uses `--write`. After setup and merge, verify a signed
creator, an unsigned creator, the transition after signing, and a PR with a different unsigned contributor.
Confirm old labels are absent and unrelated labels remain intact. Local tests do not replace live credential,
write-permission, and event-delivery validation.
