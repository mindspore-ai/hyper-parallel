// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { appendFileSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const CHECK_CONTEXT = 'all-reviewers-approved';
export const POLICY_PATH = '.github/reviewer-approval-policy.json';
const ACTIONS_APP_ID = 15368;
const DECISION_STATES = new Set(['APPROVED', 'CHANGES_REQUESTED', 'DISMISSED']);

/** Validate the trusted policy before accepting any approvals. */
export function validate_policy(policy, repository) {
  if (policy.repository !== repository) {
    throw new Error('The review policy belongs to a different repository.');
  }
  if (!Array.isArray(policy.base_branches) || !policy.base_branches.length ||
      policy.base_branches.some(branch => typeof branch !== 'string' || !branch.trim())) {
    throw new Error('base_branches must contain explicit target branch names.');
  }
  if ('reviewers' in policy || 'committers' in policy) {
    throw new Error('Reviewers come from each PR, not a fixed roster in the policy.');
  }
  if (typeof policy.auto_merge !== 'boolean') {
    throw new Error('auto_merge must be a boolean.');
  }
  if (!Number.isSafeInteger(policy.minimum_approvals) || policy.minimum_approvals < 1) {
    throw new Error('minimum_approvals must be a positive integer.');
  }
  return policy;
}

/** Evaluate each person's latest effective decision against the current PR head. */
export function evaluate_reviews(pull, reviews, minimum_approvals = 2) {
  const author = pull.user.login.toLowerCase();
  const requested = new Set((pull.requested_reviewers ?? []).map(user => user.login.toLowerCase()));
  const latest = new Map();
  // Comments and unsubmitted reviews do not replace a submitted decision.
  // A dismissed decision does replace it; an older approval must not revive.
  for (const review of [...reviews].sort((left, right) => left.id - right.id)) {
    if (review.user?.login && review.user.type !== 'Bot' && !review.user.login.toLowerCase().endsWith('[bot]') &&
        review.user.login.toLowerCase() !== author && review.submitted_at && DECISION_STATES.has(review.state)) {
      latest.set(review.user.login.toLowerCase(), review);
    }
  }
  const approved = [...latest.values()].filter(review => review.state === 'APPROVED' &&
    review.commit_id === pull.head.sha && !requested.has(review.user.login.toLowerCase()))
    .map(review => review.user.login);
  const changes_requested = [...latest.values()].filter(review => review.state === 'CHANGES_REQUESTED')
    .map(review => review.user.login);
  return {
    number: pull.number,
    sha: pull.head.sha,
    minimum_approvals,
    approved,
    changes_requested,
    ready: pull.state === 'open' && !pull.draft && approved.length >= minimum_approvals &&
      changes_requested.length === 0,
  };
}

/** Keep automatic merges behind enforced GitHub rules, including stale review dismissal. */
export function permits_auto_merge(rules, minimum_approvals = 2) {
  const has_gate = rules.some(rule => rule.type === 'required_status_checks' &&
    rule.parameters.required_status_checks.some(check =>
      check.context === CHECK_CONTEXT && check.integration_id === ACTIONS_APP_ID));
  const has_reviews = rules.some(rule => rule.type === 'pull_request' &&
    rule.parameters.dismiss_stale_reviews_on_push &&
    rule.parameters.required_approving_review_count >= minimum_approvals);
  return has_gate && has_reviews && !rules.some(rule => rule.type === 'merge_queue');
}

/** Small GitHub CLI adapter; arguments and JSON input never pass through a shell. */
export class GitHubClient {
  constructor(repository, run = execFileSync) {
    if (!/^[a-z\d_.-]+\/[a-z\d_.-]+$/i.test(repository)) {
      throw new Error('GH_REPO must be an owner/repository name.');
    }
    this.repository = repository;
    this.root = `repos/${repository}`;
    this.run = run;
  }

  api(path, { method = 'GET', body, paginate = false } = {}) {
    const endpoint = path ? `${this.root}/${path}` : this.root;
    const args = ['api', '--hostname', 'github.com', '--method', method, endpoint];
    if (paginate) args.push('--paginate', '--slurp');
    if (body) args.push('--input', '-');
    const output = this.run('gh', args, {
      encoding: 'utf8',
      input: body ? JSON.stringify(body) : undefined,
      maxBuffer: 32 * 1024 * 1024,
      timeout: 60000,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    const result = JSON.parse(output);
    return paginate ? result.flat() : result;
  }

  list_pulls() {
    return this.api('pulls?state=open&per_page=100', { paginate: true });
  }

  get_reviews(number) {
    return this.api(`pulls/${number}/reviews?per_page=100`, { paginate: true });
  }

  latest_status(sha) {
    return this.api(`commits/${sha}/statuses?per_page=100`, { paginate: true })
      .find(status => status.context === CHECK_CONTEXT);
  }

  publish(sha, state, description, target_url) {
    return this.api(`statuses/${sha}`, {
      method: 'POST',
      body: { state, description: description.slice(0, 140), context: CHECK_CONTEXT, target_url },
    });
  }

  enable_auto_merge(pull) {
    this.run('gh', ['pr', 'merge', String(pull.number), '--repo', this.repository,
      '--auto', '--merge', '--match-head-commit', pull.head.sha], {
      encoding: 'utf8', timeout: 60000, stdio: ['pipe', 'pipe', 'pipe'],
    });
  }
}

/** Return the Git blob identity of the policy read from the trusted checkout. */
export function policy_blob_sha(source) {
  return createHash('sha1').update(`blob ${Buffer.byteLength(source)}\0`).update(source).digest('hex');
}

function policy_is_current(client, expected_sha, default_branch) {
  return client.api(`contents/${POLICY_PATH}?ref=${encodeURIComponent(default_branch)}`).sha === expected_sha;
}

function same_snapshot(before, after) {
  return before.head.sha === after.head.sha && before.base.ref === after.base.ref &&
    before.base.sha === after.base.sha && before.user.login === after.user.login;
}

/** Reconcile live API state. Dry runs never publish checks or enable merging. */
export function reconcile(client, policy, options = {}) {
  const { write = false, policy_sha, target_url, only_pr } = options;
  if (write && only_pr) throw new Error('A writing run must reconcile all open PRs.');
  validate_policy(policy, client.repository);
  const pulls = client.list_pulls().filter(pull => policy.base_branches.includes(pull.base.ref));
  const target_sha = only_pr ? pulls.find(pull => pull.number === only_pr)?.head.sha : null;
  if (only_pr && !target_sha) throw new Error('The requested PR is not open on a configured target branch.');
  const groups = new Map();
  for (const pull of pulls) {
    if (only_pr && pull.head.sha !== target_sha) continue;
    if (!groups.has(pull.head.sha)) groups.set(pull.head.sha, []);
    groups.get(pull.head.sha).push(pull);
  }
  const repository = write ? client.api('') : null;
  const branch_rules = new Map();
  const reports = [];
  let errors = 0;
  for (const [sha, group] of groups) {
    let previous;
    try {
      if (write) {
        previous = client.latest_status(sha);
        if (!previous || previous.state === 'success') {
          previous = client.publish(sha, 'pending', 'Rechecking current reviewer approvals.', target_url);
        }
      }
      let results = group.map(pull => evaluate_reviews(pull, client.get_reviews(pull.number), policy.minimum_approvals));
      if (write && results.every(result => result.ready)) {
        // Check again before publishing success: a push, retarget, draft change,
        // review dismissal or policy change may have happened during API reads.
        const fresh = group.map(pull => client.api(`pulls/${pull.number}`));
        results = fresh.map(pull => evaluate_reviews(pull, client.get_reviews(pull.number), policy.minimum_approvals));
        const sharing = client.list_pulls().filter(pull =>
          policy.base_branches.includes(pull.base.ref) && pull.head.sha === sha);
        if (!fresh.every((pull, index) => same_snapshot(group[index], pull)) ||
            sharing.length !== group.length || sharing.some(pull => !group.some(old => old.number === pull.number)) ||
            !policy_is_current(client, policy_sha, repository.default_branch)) {
          throw new Error('The PR or trusted policy changed during evaluation; retry with fresh state.');
        }
      }
      const ready = results.every(result => result.ready);
      const state = ready ? 'success' : 'pending';
      const description = ready ? `At least ${policy.minimum_approvals} reviewers approved the current commit.` :
        `Waiting for ${policy.minimum_approvals} current approvals, resolved change requests, and a non-draft PR.`;
      if (write && (previous?.state !== state || previous?.description !== description)) {
        client.publish(sha, state, description, target_url);
      }
      reports.push(...results);
      // Status contexts are commit-scoped. Shared heads must satisfy every PR
      // using that SHA, even when their authors (and exemptions) differ.
      if (write && ready && policy.auto_merge && repository.allow_auto_merge) {
        for (const pull of group) {
          if (!branch_rules.has(pull.base.ref)) {
            branch_rules.set(pull.base.ref, client.api(
              `rules/branches/${encodeURIComponent(pull.base.ref)}?per_page=100`, { paginate: true }));
          }
          if (permits_auto_merge(branch_rules.get(pull.base.ref), policy.minimum_approvals) && !pull.auto_merge) {
            client.enable_auto_merge(pull);
          }
        }
      }
    } catch (error) {
      errors += 1;
      if (write) {
        // Fail closed, including when a prior successful check existed.
        try {
          client.publish(sha, 'error', 'Unable to verify approvals; see the gate workflow log.', target_url);
        } catch (publish_error) {
          console.error(`Could not invalidate ${sha}: ${publish_error.message}`);
        }
      }
      console.error(`PR ${group.map(pull => `#${pull.number}`).join(', ')}: ${error.message}`);
    }
  }
  return { reports, errors };
}

function main() {
  const args = process.argv.slice(2);
  if (args.some((arg, index) => !['--write', '--dry-run', '--pr'].includes(arg) && args[index - 1] !== '--pr')) {
    throw new Error('Usage: node reviewer-approval-gate.mjs [--dry-run [--pr NUMBER] | --write]');
  }
  if (args.includes('--write') && args.includes('--dry-run')) throw new Error('Choose write or dry-run.');
  const only_pr = args.includes('--pr') ? Number(args[args.indexOf('--pr') + 1]) : undefined;
  if (only_pr !== undefined && (!Number.isSafeInteger(only_pr) || only_pr <= 0)) {
    throw new Error('--pr requires a positive PR number.');
  }
  const write = args.includes('--write');
  const client = new GitHubClient(process.env.GH_REPO ?? 'mindspore-ai/hyper-parallel');
  let source;
  let policy;
  try {
    source = readFileSync(POLICY_PATH, 'utf8');
    policy = validate_policy(JSON.parse(source), client.repository);
  } catch (error) {
    if (write) {
      for (const sha of new Set(client.list_pulls().map(pull => pull.head.sha))) {
        client.publish(sha, 'error', 'Invalid trusted reviewer policy; see the workflow log.');
      }
    }
    throw error;
  }
  const target_url = process.env.GITHUB_RUN_ID ?
    `https://github.com/${client.repository}/actions/runs/${process.env.GITHUB_RUN_ID}` : undefined;
  const result = reconcile(client, policy, { write, only_pr, target_url, policy_sha: policy_blob_sha(source) });
  const lines = result.reports.map(report =>
    `PR #${report.number}: ${report.ready ? 'approved' : 'waiting'}; ` +
    `approvals: ${report.approved.length}/${report.minimum_approvals} (${report.approved.join(', ') || 'none'}); ` +
    `changes requested: ${report.changes_requested.join(', ') || '(none)'}; head: ${report.sha}`);
  console.log(`${write ? 'Live reconciliation' : 'Read-only evaluation'}\n${lines.join('\n')}`);
  if (process.env.GITHUB_STEP_SUMMARY) {
    appendFileSync(process.env.GITHUB_STEP_SUMMARY,
      `## Reviewer approval gate\n\n${lines.map(line => `- ${line}`).join('\n')}\n\nErrors: ${result.errors}\n`);
  }
  if (result.errors) process.exitCode = 1;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
