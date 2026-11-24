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

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
  CHECK_CONTEXT, GitHubClient, POLICY_PATH, evaluate_reviews,
  permits_auto_merge, policy_blob_sha, reconcile, validate_policy,
} from './reviewer-approval-gate.mjs';
import { PASS_LABELS, STATUS_CONTEXT, TTL_MS } from './ci-label-expiry.mjs';

const NOW = Date.parse('2026-11-12T00:00:00Z');

const POLICY = {
  repository: 'example/repo', base_branches: ['master'], auto_merge: true,
};
const RULES = [
  { type: 'required_status_checks', parameters: {
    required_status_checks: [CHECK_CONTEXT, STATUS_CONTEXT].map(context => ({ context, integration_id: 15368 })),
  } },
  { type: 'pull_request', parameters: { dismiss_stale_reviews_on_push: true, required_approving_review_count: 1 } },
];

function pull(overrides = {}) {
  return {
    number: 1, state: 'open', draft: false, user: { login: 'contributor' },
    head: { sha: 'head-sha' }, base: { ref: 'master', sha: 'base-sha' }, requested_reviewers: [],
    labels: [{ name: PASS_LABELS[0] }],
    ...overrides,
  };
}

function review(login, overrides = {}) {
  return {
    id: 1, user: { login }, state: 'APPROVED', commit_id: 'head-sha',
    submitted_at: '2026-10-28T00:00:00Z', ...overrides,
  };
}

class FakeGitHub {
  constructor(pulls = [pull()]) {
    this.repository = POLICY.repository;
    this.pulls = pulls;
    this.reviews = [review('alice'), review('bob', { id: 2 })];
    this.statuses = [];
    this.merges = [];
    this.rules = RULES;
    this.auto_merge = true;
    this.policy_sha = 'policy-sha';
    this.saved_status = undefined;
    this.fresh_pull = undefined;
    this.label_events = [{ id: 1, event: 'labeled', label: { name: PASS_LABELS[0] },
      created_at: new Date(NOW - 1000).toISOString() }];
  }

  list_pulls() { return this.pulls; }
  get_reviews() { return this.reviews; }
  get_label_events() { return this.label_events; }
  latest_status() { return this.saved_status; }
  publish(sha, state, description) {
    const status = { sha, state, description };
    this.statuses.push(status);
    return status;
  }
  enable_auto_merge(value) { this.merges.push(value.number); }
  api(path) {
    if (!path) return { default_branch: 'master', allow_auto_merge: this.auto_merge };
    if (path.startsWith('contents/')) return { sha: this.policy_sha };
    if (path.startsWith('rules/')) return this.rules;
    if (path.startsWith('pulls/')) {
      return this.fresh_pull ?? this.pulls.find(item => item.number === Number(path.split('/')[1]));
    }
    throw new Error(`Unexpected API call: ${path}`);
  }
}

function run_live(client, policy = POLICY, options = {}) {
  return reconcile(client, policy, { write: true, policy_sha: 'policy-sha', now: () => NOW, ...options });
}

test('two requested reviewers may approve while a third is still requested', () => {
  const result = evaluate_reviews(pull({ requested_reviewers: [{ login: 'bob' }] }),
    [review('alice'), review('carol')]);
  assert.equal(result.ready, true);
  assert.deepEqual(result.missing, ['bob']);
  assert.equal(result.approval_count, 2);
  assert.equal(result.approval_threshold, 2);
});

test('reviewer roster preserves the existing requested and submitted review logic', () => {
  const bot = { login: 'review-bot[bot]', type: 'Bot' };
  const pull_request = pull({ requested_reviewers: [{ login: 'carol' }, bot] });
  const reviews = [review('alice'), review('outsider', { id: 2, author_association: 'NONE' })];
  const result = evaluate_reviews(pull_request, reviews);
  assert.deepEqual(result.reviewers, ['carol', 'review-bot[bot]', 'alice', 'outsider']);
  assert.equal(result.approval_count, 2);
  assert.equal(result.ready, true);
});

test('one reviewer needs one approval, but two reviewers need two approvals', () => {
  const one = evaluate_reviews(pull(), [review('alice')]);
  assert.equal(one.ready, true);
  assert.equal(one.approval_count, 1);
  assert.equal(one.approval_threshold, 1);

  const two = evaluate_reviews(pull({ requested_reviewers: [{ login: 'bob' }] }), [review('alice')]);
  assert.equal(two.ready, false);
  assert.equal(two.approval_count, 1);
  assert.equal(two.approval_threshold, 2);
});

test('the quorum counts unique current-head approvals, not repeated reviews', () => {
  const result = evaluate_reviews(pull({ requested_reviewers: [{ login: 'bob' }] }),
    [review('alice'), review('ALICE', { id: 2 })]);
  assert.equal(result.ready, false);
  assert.equal(result.approval_count, 1);
});

test('a pending changes request still blocks when two other reviewers approve', () => {
  const result = evaluate_reviews(pull(), [review('alice'), review('bob', { id: 2 }),
    review('carol', { id: 3, state: 'CHANGES_REQUESTED' })]);
  assert.equal(result.ready, false);
  assert.deepEqual(result.changes_requested_by, ['carol']);
});

test('a bot change request blocks even when two other reviewers approve', () => {
  const bot = { login: 'review-bot[bot]', type: 'Bot' };
  const result = evaluate_reviews(pull(), [review('alice'), review('bob', { id: 2 }),
    review(bot.login, { id: 3, user: bot, state: 'CHANGES_REQUESTED' })]);
  assert.equal(result.approval_count, 2);
  assert.equal(result.ready, false);
  assert.deepEqual(result.changes_requested_by, [bot.login]);
});

test('dismissed or re-requested approvals do not contribute to the quorum', () => {
  const dismissed = evaluate_reviews(pull(), [review('alice'), review('bob', { id: 2 }),
    review('bob', { id: 3, state: 'DISMISSED' })]);
  assert.equal(dismissed.ready, false);
  assert.equal(dismissed.approval_count, 1);
  const requested = evaluate_reviews(pull({ requested_reviewers: [{ login: 'bob' }] }),
    [review('alice'), review('bob', { id: 2 })]);
  assert.equal(requested.ready, false);
  assert.equal(requested.approval_count, 1);
});

test('author is exempt and logins are case insensitive', () => {
  const result = evaluate_reviews(pull({ user: { login: 'ALICE' } }), [review('alice'), review('BOB')]);
  assert.equal(result.ready, true);
  assert.deepEqual(result.reviewers, ['BOB']);
});

test('an empty independent reviewer set never passes', () => {
  assert.equal(evaluate_reviews(pull({ user: { login: 'alice' } }), [review('ALICE')]).ready, false);
  assert.equal(evaluate_reviews(pull(), []).ready, false);
});

for (const state of ['CHANGES_REQUESTED', 'DISMISSED']) {
  test(`a later ${state} blocks an earlier approval`, () => {
    const result = evaluate_reviews(pull(), [review('alice', { id: 9, state }),
      review('bob'), review('alice', { id: 2 })]);
    assert.deepEqual(result.missing, ['alice']);
  });
}

test('a comment or unsubmitted review does not erase approval', () => {
  assert.equal(evaluate_reviews(pull(), [review('alice'), review('bob'),
    review('alice', { id: 5, state: 'COMMENTED' }),
    review('bob', { id: 6, state: 'PENDING', submitted_at: null })]).ready, true);
});

test('approval of an older SHA is insufficient after a push', () => {
  const result = evaluate_reviews(pull(), [review('alice', { commit_id: 'old-head' }), review('bob')]);
  assert.deepEqual(result.missing, ['alice']);
  assert.equal(result.approval_count, 1);
  assert.equal(result.ready, false);
});

test('a later current-head approval resolves a change request', () => {
  assert.equal(evaluate_reviews(pull(), [review('alice', { state: 'CHANGES_REQUESTED' }),
    review('alice', { id: 3 }), review('bob', { id: 4 })]).ready, true);
});

test('re-requested review, draft and closed PRs remain blocked', () => {
  const reviews = [review('alice'), review('bob')];
  for (const overrides of [{ requested_reviewers: [{ login: 'bob' }] }, { draft: true }, { state: 'closed' }]) {
    assert.equal(evaluate_reviews(pull(overrides), reviews).ready, false);
  }
});

test('invalid policies and fixed reviewer rosters are rejected', () => {
  for (const changes of [
    { reviewers: ['alice', 'bob'] }, { committers: ['alice'] },
    { base_branches: [] }, { auto_merge: 'true' }, { repository: 'wrong/repo' },
  ]) {
    assert.throws(() => validate_policy({ ...POLICY, ...changes }, POLICY.repository));
  }
});

test('the checked-in policy is valid and has no fixed personnel list', () => {
  const policy = JSON.parse(readFileSync(POLICY_PATH, 'utf8'));
  validate_policy(policy, policy.repository);
  assert.equal('reviewers' in policy, false);
  assert.equal('committers' in policy, false);
});

test('auto-merge requires this check from GitHub Actions and stale approval dismissal', () => {
  assert.equal(permits_auto_merge(RULES), true);
  assert.equal(permits_auto_merge([]), false);
  assert.equal(permits_auto_merge([RULES[0]]), false);
  const wrong_source = structuredClone(RULES);
  wrong_source[0].parameters.required_status_checks[0].integration_id = null;
  assert.equal(permits_auto_merge(wrong_source), false);
  const missing_ci = structuredClone(RULES);
  missing_ci[0].parameters.required_status_checks.pop();
  assert.equal(permits_auto_merge(missing_ci), false);
  const wrong_ci_source = structuredClone(RULES);
  wrong_ci_source[0].parameters.required_status_checks[1].integration_id = null;
  assert.equal(permits_auto_merge(wrong_ci_source), false);
  assert.equal(permits_auto_merge([...RULES, { type: 'merge_queue' }]), false);
});

test('dry-run has no writes, even when every reviewer has approved', () => {
  const client = new FakeGitHub();
  assert.equal(reconcile(client, POLICY, { now: () => NOW }).reports[0].ready, true);
  assert.deepEqual(client.statuses, []);
  assert.deepEqual(client.merges, []);
});

test('live reconciliation publishes to the PR head, then enables auto-merge', () => {
  const client = new FakeGitHub();
  assert.equal(run_live(client).errors, 0);
  assert.deepEqual(client.statuses.map(status => status.state), ['pending', 'success']);
  assert.ok(client.statuses.every(status => status.sha === 'head-sha'));
  assert.deepEqual(client.merges, [1]);
});

test('missing approvals block merging and an unchanged pending status is not duplicated', () => {
  const client = new FakeGitHub();
  client.pulls[0].requested_reviewers = [{ login: 'bob' }];
  client.reviews = [review('alice')];
  run_live(client);
  client.saved_status = client.statuses.at(-1);
  client.statuses = [];
  assert.equal(run_live(client).reports[0].ready, false);
  assert.deepEqual(client.statuses, []);
  assert.deepEqual(client.merges, []);
});

test('shared head statuses require approval for every PR and author exemption', () => {
  const client = new FakeGitHub([pull({ user: { login: 'alice' } }),
    pull({ number: 2, requested_reviewers: [{ login: 'alice' }] })]);
  client.reviews = [review('bob')];
  assert.deepEqual(run_live(client).reports.map(result => result.ready), [true, false]);
  assert.ok(client.statuses.every(status => status.state !== 'success'));
  assert.deepEqual(client.merges, []);
});

test('API failures invalidate prior success and do not merge', () => {
  const client = new FakeGitHub();
  client.saved_status = { state: 'success' };
  client.get_reviews = () => { throw new Error('API unavailable'); };
  assert.equal(run_live(client).errors, 1);
  assert.deepEqual(client.statuses.map(status => status.state), ['pending', 'error']);
  assert.deepEqual(client.merges, []);
});

test('reconciliation does not need review request history', () => {
  const client = new FakeGitHub();
  client.get_review_events = () => { throw new Error('Request history unavailable'); };
  assert.equal(run_live(client).errors, 0);
  assert.equal(client.statuses.at(-1).state, 'success');
  assert.deepEqual(client.merges, [1]);
});

test('head, base branch and base commit races cannot publish success', () => {
  for (const overrides of [
    { head: { sha: 'new-head' } }, { base: { ref: 'other', sha: 'base-sha' } },
    { base: { ref: 'master', sha: 'new-base' } },
  ]) {
    const client = new FakeGitHub();
    client.fresh_pull = pull(overrides);
    assert.equal(run_live(client).errors, 1);
    assert.ok(client.statuses.every(status => status.state !== 'success'));
    assert.deepEqual(client.merges, []);
  }
});

test('policy update during evaluation invalidates the result', () => {
  const client = new FakeGitHub();
  client.policy_sha = 'new-policy';
  assert.equal(run_live(client).errors, 1);
  assert.equal(client.statuses.at(-1).state, 'error');
  assert.deepEqual(client.merges, []);
});

test('review dismissal between reads prevents a successful result', () => {
  const client = new FakeGitHub();
  let calls = 0;
  client.get_reviews = () => [review('alice'), review('bob', {
    state: ++calls === 1 ? 'APPROVED' : 'DISMISSED',
  })];
  assert.equal(run_live(client).reports[0].ready, false);
  assert.ok(client.statuses.every(status => status.state !== 'success'));
});

test('another PR appearing with the same head invalidates the snapshot', () => {
  const client = new FakeGitHub();
  let calls = 0;
  client.list_pulls = () => ++calls === 1 ? [pull()] : [pull(), pull({ number: 2 })];
  assert.equal(run_live(client).errors, 1);
  assert.deepEqual(client.merges, []);
});

test('missing enforced rules or a disabled repository setting prevent auto-merge', () => {
  for (const changes of [{ rules: [] }, { auto_merge: false }]) {
    const client = Object.assign(new FakeGitHub(), changes);
    assert.equal(run_live(client).errors, 0);
    assert.deepEqual(client.merges, []);
  }
});

test('write runs cannot update only one PR sharing a commit', () => {
  assert.throws(() => reconcile(new FakeGitHub(), POLICY, { write: true, only_pr: 1 }));
});

test('GitHub API adapter fetches every page and passes JSON without a shell', () => {
  const calls = [];
  const client = new GitHubClient('example/repo', (command, args, options) => {
    calls.push({ command, args, options });
    return args.includes('--paginate') ? JSON.stringify([[review('alice')], [review('bob')]]) : '{}';
  });
  assert.equal(client.get_reviews(1).length, 2);
  assert.ok(calls[0].args.includes('--paginate'));
  assert.ok(calls[0].args.includes('--slurp'));
  client.publish('head-sha', 'pending', 'Waiting');
  assert.equal(JSON.parse(calls[1].options.input).context, CHECK_CONTEXT);
  assert.ok(calls[1].args.includes('--input'));
  assert.equal(calls[1].options.shell, undefined);
  assert.equal(client.get_label_events(1).length, 2);
  assert.ok(calls[2].args.includes('repos/example/repo/issues/1/events?per_page=100'));
  assert.ok(calls[2].args.includes('--paginate'));
});

test('fresh CI is checked before publishing approval success and again before enabling auto-merge', () => {
  const client = new FakeGitHub();
  const actions = [];
  client.get_label_events = () => { actions.push('check-ci'); return client.label_events; };
  client.publish = (sha, state, description) => {
    actions.push(state);
    return FakeGitHub.prototype.publish.call(client, sha, state, description);
  };
  client.enable_auto_merge = () => actions.push('enable-auto-merge');
  assert.equal(run_live(client).errors, 0);
  assert.deepEqual(actions, ['pending', 'check-ci', 'success', 'check-ci', 'enable-auto-merge']);
});

test('expired CI keeps the approval gate pending even when auto-merge was already enabled', () => {
  for (const auto_merge of [null, { enabled_at: '2026-11-11T00:00:00Z' }]) {
    const client = new FakeGitHub([pull({ auto_merge })]);
    client.saved_status = { state: 'success' };
    client.label_events[0].created_at = new Date(NOW - TTL_MS).toISOString();
    const result = run_live(client);
    assert.equal(result.errors, 0);
    assert.equal(result.reports[0].ready, false);
    assert.equal(result.reports[0].ci_valid, false);
    assert.ok(client.statuses.every(status => status.state === 'pending'));
    assert.deepEqual(client.merges, []);
  }
});

test('missing CI labels block auto-merge and a renewed label restores eligibility', () => {
  const client = new FakeGitHub([pull({ labels: [] })]);
  assert.equal(run_live(client).reports[0].ready, false);
  assert.deepEqual(client.merges, []);
  client.pulls[0].labels = [{ name: PASS_LABELS[0] }];
  assert.equal(run_live(client).reports[0].ci_valid, true);
  assert.deepEqual(client.merges, [1]);
});

test('unverifiable CI events revoke earlier approval success and never enable auto-merge', () => {
  for (const unreadable of [false, true]) {
    const client = new FakeGitHub();
    client.saved_status = { state: 'success' };
    client.label_events = [];
    if (unreadable) client.get_label_events = () => { throw new Error('CI events unavailable'); };
    assert.equal(run_live(client).errors, 1);
    assert.equal(client.statuses.at(-1).state, 'error');
    assert.deepEqual(client.merges, []);
  }
});

test('CI expiring during branch-rule reads prevents the subsequent auto-merge request', () => {
  const client = new FakeGitHub();
  client.label_events[0].created_at = new Date(NOW - TTL_MS).toISOString();
  let clock = NOW - 1;
  client.api = path => {
    if (path.startsWith('rules/')) clock = NOW;
    return FakeGitHub.prototype.api.call(client, path);
  };
  assert.equal(run_live(client, POLICY, { now: () => clock }).errors, 1);
  assert.equal(client.statuses.at(-1).state, 'error');
  assert.deepEqual(client.merges, []);
});

test('a head change after approval success cannot enable auto-merge for an unchecked head', () => {
  const client = new FakeGitHub();
  client.publish = (sha, state, description) => {
    if (state === 'success') client.fresh_pull = pull({ head: { sha: 'new-head' } });
    return FakeGitHub.prototype.publish.call(client, sha, state, description);
  };
  assert.equal(run_live(client).errors, 1);
  assert.equal(client.statuses.at(-1).state, 'error');
  assert.deepEqual(client.merges, []);
});

test('a fresh PR cannot enable auto-merge while a sibling sharing its head has expired CI', () => {
  const client = new FakeGitHub([pull(), pull({ number: 2 })]);
  client.get_label_events = number => number === 1 ? client.label_events :
    [{ ...client.label_events[0], created_at: new Date(NOW - TTL_MS).toISOString() }];
  assert.equal(run_live(client).errors, 0);
  assert.ok(client.statuses.every(status => status.state !== 'success'));
  assert.deepEqual(client.merges, []);
});

test('reviewer reconciliation observes label updates and the shared CI checker', () => {
  const workflow = readFileSync(new URL('../workflows/reviewer-approval-gate.yml', import.meta.url), 'utf8');
  assert.match(workflow, /^      - labeled$/m);
  assert.match(workflow, /^      - unlabeled$/m);
  assert.match(workflow, /workflows: \[Reviewer approval signal, Label Gate Check\]/);
  assert.match(workflow, /\.github\/scripts\/ci-label-expiry\.mjs/);
  const tests = readFileSync(new URL('../workflows/reviewer-approval-tests.yml', import.meta.url), 'utf8');
  assert.equal(tests.match(/\.github\/scripts\/ci-label-expiry\*\.mjs/g).length, 2);
});

test('auto-merge command pins the head and does not bypass branch rules', () => {
  let actual;
  const client = new GitHubClient('example/repo', (_command, args) => { actual = args; return ''; });
  client.enable_auto_merge(pull());
  assert.ok(actual.includes('--auto'));
  assert.ok(actual.includes('--merge'));
  assert.equal(actual[actual.indexOf('--match-head-commit') + 1], 'head-sha');
  assert.equal(actual.includes('--admin'), false);
});

test('policy fingerprint uses Git blob encoding', () => {
  assert.equal(policy_blob_sha(''), 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391');
});

test('the roster follows requested and submitted reviewers throughout the PR lifecycle', () => {
  let current = pull({ requested_reviewers: [{ login: 'alice' }, { login: 'bob' }] });
  assert.deepEqual(evaluate_reviews(current, []).missing, ['alice', 'bob']);
  current = pull({ requested_reviewers: [{ login: 'bob' }] });
  assert.deepEqual(evaluate_reviews(current, [review('alice')]).missing, ['bob']);
  const result = evaluate_reviews(pull(), [review('alice'), review('bob')]);
  assert.equal(result.ready, true);
  assert.deepEqual(result.reviewers, ['alice', 'bob']);
});

test('removing an unsubmitted review request removes that person from the roster', () => {
  const before = pull({ requested_reviewers: [{ login: 'bob' }] });
  assert.equal(evaluate_reviews(before, [review('alice')]).ready, false);
  const after = evaluate_reviews(pull(), [review('alice')]);
  assert.equal(after.ready, true);
  assert.deepEqual(after.reviewers, ['alice']);
});

test('a comment-only reviewer increases the quorum but does not approve', () => {
  const result = evaluate_reviews(pull(), [review('alice'), review('bob', { state: 'COMMENTED' })]);
  assert.deepEqual(result.missing, ['bob']);
  assert.equal(result.ready, false);
});

test('bot reviews follow the existing reviewer rules', () => {
  const bot = { login: 'review-bot[bot]', type: 'Bot' };
  const one = evaluate_reviews(pull({ requested_reviewers: [bot] }),
    [review('alice'), review(bot.login, { user: bot, state: 'COMMENTED' })]);
  assert.equal(one.ready, false);
  assert.deepEqual(one.reviewers, [bot.login, 'alice']);
  const two = evaluate_reviews(pull({ requested_reviewers: [{ login: 'bob' }] }),
    [review('alice'), review(bot.login, { user: bot })]);
  assert.equal(two.ready, true);
  assert.equal(two.approval_count, 2);
  assert.equal(evaluate_reviews(pull(), [review(bot.login, { user: bot })]).ready, true);
});

test('pending teams block without expanding the entire team into mandatory reviewers', () => {
  const result = evaluate_reviews(pull({ requested_teams: [{ slug: 'maintainers' }] }),
    [review('alice'), review('bob')]);
  assert.deepEqual(result.reviewers, ['alice', 'bob']);
  assert.equal(result.approval_count, 2);
  assert.deepEqual(result.pending_teams, ['maintainers']);
  assert.equal(result.ready, false);
  assert.equal(evaluate_reviews(pull(), [review('alice')]).ready, true);
});

test('a newly requested reviewer between API reads prevents success', () => {
  const client = new FakeGitHub();
  client.reviews = [review('alice')];
  client.fresh_pull = pull({ requested_reviewers: [{ login: 'carol' }] });
  let calls = 0;
  client.list_pulls = () => ++calls === 1 ? client.pulls : [client.fresh_pull];
  const result = run_live(client);
  assert.equal(result.errors, 0);
  assert.deepEqual(result.reports[0].missing, ['carol']);
  assert.ok(client.statuses.every(status => status.state !== 'success'));
  assert.deepEqual(client.merges, []);
});

test('a new changes request between API reads prevents success after reaching the quorum', () => {
  const client = new FakeGitHub();
  let calls = 0;
  client.get_reviews = () => ++calls === 1 ? [review('alice'), review('bob', { id: 2 })] :
    [review('alice'), review('bob', { id: 2 }), review('carol', { id: 3, state: 'CHANGES_REQUESTED' })];
  assert.equal(run_live(client).reports[0].ready, false);
  assert.ok(client.statuses.every(status => status.state !== 'success'));
  assert.deepEqual(client.merges, []);
});

test('a review request arriving between the fresh PR read and the final list prevents success', () => {
  const client = new FakeGitHub();
  client.reviews = [review('alice')];
  let calls = 0;
  client.list_pulls = () => ++calls === 1 ? [pull()] :
    [pull({ requested_reviewers: [{ login: 'bob' }] })];
  const result = run_live(client);
  assert.ok(result.errors || !result.reports[0]?.ready);
  assert.ok(client.statuses.every(status => status.state !== 'success'));
  assert.deepEqual(client.merges, []);
});

test('live reconciliation enables auto-merge after two approvals without waiting for a third', () => {
  const client = new FakeGitHub([pull({ requested_reviewers: [{ login: 'carol' }] })]);
  const result = run_live(client);
  assert.equal(result.reports[0].ready, true);
  assert.deepEqual(client.statuses.map(status => status.state), ['pending', 'success']);
  assert.deepEqual(client.merges, [1]);
});

test('live reconciliation includes submitted reviews as in the existing gate', () => {
  const client = new FakeGitHub([pull({ requested_reviewers: [{ login: 'bob' }, { login: 'carol' }] })]);
  client.reviews = [review('alice'), review('outsider', { id: 2, author_association: 'NONE' })];
  const result = run_live(client);
  assert.equal(result.reports[0].approval_count, 2);
  assert.equal(result.reports[0].ready, true);
  assert.deepEqual(client.merges, [1]);
});
