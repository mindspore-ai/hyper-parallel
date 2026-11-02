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

const POLICY = {
  repository: 'example/repo', base_branches: ['master'], auto_merge: true, minimum_approvals: 2,
};
const RULES = [
  { type: 'required_status_checks', parameters: {
    required_status_checks: [{ context: CHECK_CONTEXT, integration_id: 15368 }],
  } },
  { type: 'pull_request', parameters: { dismiss_stale_reviews_on_push: true, required_approving_review_count: 2 } },
];

function pull(overrides = {}) {
  return {
    number: 1, state: 'open', draft: false, user: { login: 'contributor' },
    head: { sha: 'head-sha' }, base: { ref: 'master', sha: 'base-sha' }, requested_reviewers: [],
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
  }

  list_pulls() { return this.pulls; }
  get_reviews() { return this.reviews; }
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

function run_live(client, policy = POLICY) {
  return reconcile(client, policy, { write: true, policy_sha: 'policy-sha' });
}

test('two approvals pass even when other reviewers or teams have not reviewed', () => {
  const result = evaluate_reviews(pull({ requested_reviewers: [{ login: 'carol' }],
    requested_teams: [{ slug: 'maintainers' }] }), [review('alice'), review('bob')]);
  assert.equal(result.ready, true);
  assert.deepEqual(result.approved, ['alice', 'bob']);
});

test('zero or one approval is insufficient; two or more distinct people pass', () => {
  for (const count of [0, 1, 2, 3]) {
    const reviews = ['alice', 'bob', 'carol'].slice(0, count).map(login => review(login));
    assert.equal(evaluate_reviews(pull(), reviews).ready, count >= 2);
  }
});

test('author and bots are excluded; duplicate and case-varied logins count once', () => {
  const result = evaluate_reviews(pull({ user: { login: 'ALICE' } }), [review('alice'), review('BOB'),
    review('bob', { id: 2 }), review('assistant[bot]'),
    review('service', { user: { login: 'service', type: 'Bot' } })]);
  assert.equal(result.ready, false);
  assert.deepEqual(result.approved, ['bob']);
  assert.equal(evaluate_reviews(pull(), [review('ALICE'), review('alice', { id: 2 })]).ready, false);
});

for (const state of ['CHANGES_REQUESTED', 'DISMISSED']) {
  test(`a later ${state} invalidates an earlier approval`, () => {
    const result = evaluate_reviews(pull(), [review('alice', { id: 9, state }),
      review('bob'), review('alice', { id: 2 })]);
    assert.deepEqual(result.approved, ['bob']);
    assert.equal(result.ready, false);
  });
}

test('outstanding changes requested still block even with two other approvals', () => {
  const result = evaluate_reviews(pull(), [review('alice'), review('bob'),
    review('carol', { state: 'CHANGES_REQUESTED', commit_id: 'old-head' })]);
  assert.equal(result.ready, false);
  assert.deepEqual(result.changes_requested, ['carol']);
});

test('a dismissed third review does not block two valid approvals', () => {
  assert.equal(evaluate_reviews(pull(), [review('alice'), review('bob'),
    review('carol', { state: 'DISMISSED' })]).ready, true);
});

test('a comment or unsubmitted review does not erase approval', () => {
  assert.equal(evaluate_reviews(pull(), [review('alice'), review('bob'),
    review('alice', { id: 5, state: 'COMMENTED' }),
    review('bob', { id: 6, state: 'PENDING', submitted_at: null })]).ready, true);
});

test('approval of an older SHA is insufficient after a push', () => {
  assert.deepEqual(evaluate_reviews(pull(), [review('alice', { commit_id: 'old-head' }),
    review('bob')]).approved, ['bob']);
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
    ...[undefined, 0, -1, 1.5, '2', null].map(minimum_approvals => ({ minimum_approvals })),
  ]) {
    assert.throws(() => validate_policy({ ...POLICY, ...changes }, POLICY.repository));
  }
});

test('the checked-in policy is valid and has no fixed personnel list', () => {
  const policy = JSON.parse(readFileSync(POLICY_PATH, 'utf8'));
  validate_policy(policy, policy.repository);
  assert.equal('reviewers' in policy, false);
  assert.equal('committers' in policy, false);
  assert.equal(policy.minimum_approvals, 2);
});

test('auto-merge requires this check from GitHub Actions and stale approval dismissal', () => {
  assert.equal(permits_auto_merge(RULES), true);
  const insufficient = structuredClone(RULES);
  insufficient[1].parameters.required_approving_review_count = 1;
  assert.equal(permits_auto_merge(insufficient), false);
  assert.equal(permits_auto_merge(RULES, 3), false);
  const stale = structuredClone(RULES);
  stale[1].parameters.dismiss_stale_reviews_on_push = false;
  assert.equal(permits_auto_merge(stale), false);
  assert.equal(permits_auto_merge([]), false);
  assert.equal(permits_auto_merge([RULES[0]]), false);
  const wrong_source = structuredClone(RULES);
  wrong_source[0].parameters.required_status_checks[0].integration_id = null;
  assert.equal(permits_auto_merge(wrong_source), false);
  assert.equal(permits_auto_merge([...RULES, { type: 'merge_queue' }]), false);
});

test('dry-run has no writes, even when enough reviewers have approved', () => {
  const client = new FakeGitHub();
  assert.equal(reconcile(client, POLICY).reports[0].ready, true);
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
    pull({ number: 2 })]);
  assert.deepEqual(run_live(client).reports.map(result => result.ready), [false, true]);
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

test('comment-only reviewers and pending teams do not raise the approval threshold', () => {
  const result = evaluate_reviews(pull({ requested_teams: [{ slug: 'maintainers' }] }),
    [review('alice'), review('bob'), review('carol', { state: 'COMMENTED' }),
      review('assistant[bot]', { state: 'COMMENTED' })]);
  assert.equal(result.ready, true);
  assert.deepEqual(result.approved, ['alice', 'bob']);
});

test('removing an unsubmitted reviewer does not turn one approval into two', () => {
  assert.equal(evaluate_reviews(pull({ requested_reviewers: [{ login: 'bob' }] }), [review('alice')]).ready, false);
  assert.equal(evaluate_reviews(pull(), [review('alice')]).ready, false);
});

test('re-requesting one approval does not prevent two other approvals from passing', () => {
  const result = evaluate_reviews(pull({ requested_reviewers: [{ login: 'carol' }] }),
    [review('alice'), review('bob'), review('carol')]);
  assert.equal(result.ready, true);
  assert.deepEqual(result.approved, ['alice', 'bob']);
});

test('a new pending reviewer between API reads does not block two approvals', () => {
  const client = new FakeGitHub();
  client.fresh_pull = pull({ requested_reviewers: [{ login: 'carol' }] });
  const result = run_live(client);
  assert.equal(result.errors, 0);
  assert.equal(result.reports[0].ready, true);
  assert.equal(client.statuses.at(-1).state, 'success');
  assert.deepEqual(client.merges, [1]);
});

test('a new change request between reads prevents success despite two approvals', () => {
  const client = new FakeGitHub();
  let calls = 0;
  client.get_reviews = () => ++calls === 1 ? client.reviews :
    [...client.reviews, review('carol', { state: 'CHANGES_REQUESTED' })];
  assert.equal(run_live(client).reports[0].ready, false);
  assert.ok(client.statuses.every(status => status.state !== 'success'));
  assert.deepEqual(client.merges, []);
});

test('the trusted policy threshold controls reconciliation and native rule requirements', () => {
  const client = new FakeGitHub();
  const policy = { ...POLICY, minimum_approvals: 3 };
  assert.equal(run_live(client, policy).reports[0].ready, false);
  client.reviews.push(review('carol'));
  assert.equal(run_live(client, policy).reports[0].ready, true);
  assert.deepEqual(client.merges, []);
  client.rules = structuredClone(RULES);
  client.rules[1].parameters.required_approving_review_count = 3;
  run_live(client, policy);
  assert.deepEqual(client.merges, [1]);
});
