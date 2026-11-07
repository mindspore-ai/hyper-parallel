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
import test from 'node:test';
import { evaluate_labels, LabelClient, PASS_LABELS, reconcile_labels, TTL_MS } from './ci-label-expiry.mjs';

const NOW = Date.parse('2026-10-30T12:00:00Z');
const [NORMAL, OVERTIME] = PASS_LABELS;
function pull(names = [NORMAL], overrides = {}) {
  return { number: 1, state: 'open', head: { sha: 'head' }, merge_commit_sha: 'merge',
    labels: names.map(name => ({ name })), ...overrides };
}
function event(name = NORMAL, age = TTL_MS, id = 1, action = 'labeled') {
  return { id, event: action, label: { name }, created_at: new Date(NOW - age).toISOString() };
}
function check(id = 1, overrides = {}) {
  return { id, name: 'label-check', status: 'completed', conclusion: 'success', app: { id: 15368 }, ...overrides };
}
function status(state = 'success') {
  return { context: 'label-check', state, creator: { login: 'github-actions[bot]' } };
}
class FakeClient {
  constructor(names = [NORMAL]) {
    this.pull = pull(names);
    this.events = names.map((name, index) => event(name, TTL_MS, index + 1));
    this.checks = { head: [check(1)], merge: [check(2)] };
    this.statuses = { head: [status()], merge: [status()] };
    this.writes = [];
  }
  list_pulls() { return [structuredClone(this.pull)]; }
  get_pull() { return structuredClone(this.pull); }
  get_events() { return structuredClone(this.events); }
  get_checks(sha) { return structuredClone(this.checks[sha] ?? []); }
  get_statuses(sha) { return structuredClone(this.statuses[sha] ?? []); }
  set_status(sha, state) {
    this.writes.push(['status', sha, state]);
    this.statuses[sha] = [status(state)];
  }
  remove_label(number, name) {
    this.writes.push(['remove', number, name]);
    this.pull.labels = this.pull.labels.filter(label => label.name !== name);
  }
}
function sweep(client, options = {}) {
  return reconcile_labels(client, { write: true, now: () => NOW, completed_head: 'head', ...options });
}

test('24 hours is expired; one millisecond earlier is valid for either label', () => {
  for (const name of PASS_LABELS) {
    assert.equal(evaluate_labels(pull([name]), [event(name)], NOW)[0].expired, true);
    assert.equal(evaluate_labels(pull([name]), [event(name)], NOW - 1)[0].expired, false);
  }
});
test('latest re-addition resets only that label, even with out-of-order events', () => {
  const events = [event(NORMAL, 1000, 4), event(OVERTIME, TTL_MS, 2),
    event(NORMAL, TTL_MS * 2, 1), event(NORMAL, 2000, 3, 'unlabeled')];
  const result = evaluate_labels(pull(PASS_LABELS), events, NOW);
  assert.deepEqual(result.map(label => label.expired), [false, true]);
  assert.equal(result[0].event_id, 4);
});
test('unrelated activity and PR updated_at do not renew labels', () => {
  const value = pull([NORMAL, 'bug'], { updated_at: new Date(NOW).toISOString() });
  const result = evaluate_labels(value, [event(), event('bug', 0, 2)], NOW);
  assert.equal(result.length, 1);
  assert.equal(result[0].expired, true);
});
test('unattached labels are ignored; missing, removed, invalid or future additions fail closed', () => {
  assert.deepEqual(evaluate_labels(pull([]), [event()], NOW), []);
  for (const events of [[], [event(NORMAL, 0, 2, 'unlabeled')],
    [{ ...event(), created_at: 'invalid' }], [event(NORMAL, -1000)]]) {
    assert.throws(() => evaluate_labels(pull(), events, NOW), /Cannot establish/);
  }
});
test('both expired labels are removed after blocking head and merge statuses', () => {
  const client = new FakeClient(PASS_LABELS);
  assert.equal(sweep(client).errors.length, 0);
  assert.deepEqual(client.writes, [['status', 'head', 'failure'], ['status', 'merge', 'failure'],
    ['remove', 1, NORMAL], ['remove', 1, OVERTIME]]);
  assert.deepEqual(client.pull.labels, []);
});
test('a still-valid alternative label preserves the passing check', () => {
  const client = new FakeClient(PASS_LABELS);
  client.events[1] = event(OVERTIME, 1000, 2);
  assert.equal(sweep(client).errors.length, 0);
  assert.deepEqual(client.writes, [['remove', 1, NORMAL]]);
});
test('dry runs never write, and closed PRs are untouched', () => {
  const client = new FakeClient();
  assert.equal(sweep(client, { write: false }).reports[0].labels[0].expired, true);
  assert.deepEqual(client.writes, []);
  client.pull.state = 'closed';
  assert.deepEqual(sweep(client).reports, []);
  assert.deepEqual(client.writes, []);
});
test('a re-added label between evaluation and removal is preserved', () => {
  const client = new FakeClient(PASS_LABELS);
  client.events[1] = event(OVERTIME, 1000, 2);
  let reads = 0;
  client.get_events = () => ++reads === 1 ? client.events : [event(NORMAL, 1000, 3), client.events[1]];
  assert.equal(sweep(client).errors.length, 1);
  assert.deepEqual(client.writes, [['status', 'head', 'failure'], ['status', 'merge', 'failure']]);
  assert.equal(client.pull.labels.length, 2);
});
test('a changed head or closed PR does not receive an outdated mutation', () => {
  for (const change of [{ head: { sha: 'new-head' } }, { state: 'closed' }]) {
    const client = new FakeClient();
    client.get_pull = () => ({ ...client.pull, ...change });
    assert.equal(sweep(client).errors.length, 1);
    assert.deepEqual(client.writes, []);
  }
});
test('status publication failure prevents removal and is retried next sweep', () => {
  const client = new FakeClient();
  client.set_status = () => { throw new Error('permission denied'); };
  assert.match(sweep(client).errors[0], /permission denied/);
  assert.equal(client.pull.labels.length, 1);
  client.set_status = FakeClient.prototype.set_status;
  assert.equal(sweep(client).errors.length, 0);
  assert.equal(client.pull.labels.length, 0);
});
test('blocks stale successes even when a prior sweep already removed labels', () => {
  const client = new FakeClient([]);
  assert.equal(sweep(client).errors.length, 0);
  assert.deepEqual(client.writes, [['status', 'head', 'failure'], ['status', 'merge', 'failure']]);
  client.writes = [];
  assert.equal(sweep(client).errors.length, 0);
  assert.deepEqual(client.writes, []);
});
test('a fresh label needs a successful current GitHub Actions label check', () => {
  const client = new FakeClient();
  client.events = [event(NORMAL, 1000)];
  client.pull.merge_commit_sha = null;
  client.checks.head = [check(1, { app: { id: 99 } }), check(2, { name: 'unit-test' }),
    check(3, { conclusion: 'success' }), check(4, { status: 'in_progress', conclusion: null })];
  sweep(client);
  assert.deepEqual(client.writes, [['status', 'head', 'failure']]);
  client.checks.head.push(check(5));
  sweep(client);
  assert.deepEqual(client.writes.at(-1), ['status', 'head', 'success']);
});
test('does not create a status on a merge commit without a label check', () => {
  const client = new FakeClient();
  client.checks.merge = [];
  sweep(client);
  assert.deepEqual(client.writes, [['status', 'head', 'failure'], ['remove', 1, NORMAL]]);
});
test('a renewed CI label restores the status after a successful label check', () => {
  const client = new FakeClient();
  assert.equal(sweep(client).errors.length, 0);
  client.pull.labels = [{ name: NORMAL }];
  client.events = [event(NORMAL, 1000, 2)];
  client.writes = [];
  assert.equal(sweep(client).errors.length, 0);
  assert.deepEqual(client.writes, [['status', 'head', 'success'], ['status', 'merge', 'success']]);
  assert.equal(client.pull.labels.length, 1);
});
test('a missing label addition event blocks the head without deleting the label', () => {
  const client = new FakeClient();
  client.events = [];
  assert.match(sweep(client).errors[0], /Cannot establish/);
  assert.deepEqual(client.writes, [['status', 'head', 'failure']]);
  assert.equal(client.pull.labels.length, 1);
});
test('PRs sharing a head cannot override each other with conflicting CI labels', () => {
  const client = new FakeClient();
  const other = pull([], { number: 2, merge_commit_sha: null });
  client.events = [event(NORMAL, 1000)];
  client.list_pulls = () => [structuredClone(client.pull), structuredClone(other)];
  client.get_pull = number => structuredClone(number === 1 ? client.pull : other);
  assert.equal(sweep(client).errors.length, 0);
  assert.deepEqual(client.writes, [['status', 'head', 'failure'], ['status', 'merge', 'failure']]);
});
test('a newly opened PR sharing the head prevents a stale success', () => {
  const client = new FakeClient();
  client.events = [event(NORMAL, 1000)];
  client.statuses.head = [status('failure')];
  let reads = 0;
  client.list_pulls = () => ++reads === 1 ? [structuredClone(client.pull)] :
    [structuredClone(client.pull), pull([], { number: 2 })];
  assert.match(sweep(client).errors[0], /sharing this head changed/);
  assert.equal(client.writes.some(item => item[0] === 'status' && item[2] === 'success'), false);
  assert.equal(client.pull.labels.length, 1);
});
test('label deletion errors are surfaced unless another actor already removed it', () => {
  const client = new FakeClient();
  client.remove_label = () => { throw new Error('network error'); };
  assert.match(sweep(client).errors[0], /network error/);
  client.remove_label = () => { client.pull.labels = []; throw new Error('already removed'); };
  assert.equal(sweep(client).errors.length, 0);
});
test('API read failure does not delete labels and does not stop other PRs', () => {
  const client = new FakeClient();
  client.list_pulls = () => [pull(), pull([], { number: 2 })];
  client.get_events = () => { throw new Error('API unavailable'); };
  client.get_pull = number => pull([], { number });
  assert.equal(sweep(client).errors.length, 1);
  assert.equal(client.writes.some(item => item[0] === 'remove'), false);
});
test('adapter paginates lists and object pages, handles empty responses, and uses JSON stdin', () => {
  const calls = [];
  const client = new LabelClient('owner/repo', (command, args, options) => {
    calls.push({ command, args, options });
    if (args.includes('POST')) return '';
    if (args.some(arg => arg.includes('check-runs?'))) {
      return JSON.stringify([{ check_runs: [check()] }, { check_runs: [check(2)] }]);
    }
    return JSON.stringify([[event()], [event(NORMAL, 1000, 2)]]);
  });
  assert.equal(client.get_events(1).length, 2);
  assert.equal(client.get_checks('head').length, 2);
  client.set_status('head', 'failure');
  assert.ok(calls[0].args.includes('--paginate'));
  assert.ok(calls[0].args.includes('--slurp'));
  assert.equal(JSON.parse(calls[2].options.input).state, 'failure');
  assert.throws(() => new LabelClient('bad repo'), /Invalid GH_REPO/);
});

test('scheduled scans skip unlabeled PRs; completion signals include their head', () => {
  const client = new FakeClient([]);
  client.get_checks = () => { throw new Error('Unexpected check query'); };
  assert.deepEqual(sweep(client, { completed_head: undefined }).reports, []);
  assert.deepEqual(sweep(client, { completed_head: 'another-head' }).reports, []);
  assert.deepEqual(client.writes, []);
});
